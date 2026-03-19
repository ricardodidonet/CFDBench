"""
inference_flowcast.py

Generate forecasts using a trained FluidDynamicsUNet model.
Supports single-step and multi-step autoregressive prediction.
"""

import logging
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Optional

from dataset import get_auto_dataset
from dataset.wrapper import MultiStepFlowCastDataset, multi_step_collate_fn
from models.fluid_unet import FluidDynamicsUNet
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

logger = logging.getLogger(__name__)


class CFGScaledFluidModel(ModelWrapper):

    """
    Model wrapper for classifier-free guidance on case parameters.

    Similar to CFGScaledModel in eval_loop.py but adapted for fluid dynamics:
    - Guidance on case_params instead of class labels
    - Keeps x_prev_1, x_prev_2 (essential temporal conditioning)
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.nfe_counter = 0  # Track number of function evaluations

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x_prev_1: torch.Tensor,
        x_prev_2: torch.Tensor,
        case_params: Optional[torch.Tensor],
        cfg_scale: float
    ):
        """
        Forward pass with classifier-free guidance.

        Args:
            x: Current state (B, 2, H, W)
            t: Time (scalar or (B,))
            x_prev_1: Previous state at t-1 (B, 2, H, W)
            x_prev_2: Previous state at t-2 (B, 2, H, W)
            case_params: Case parameters (B, D) or None
            cfg_scale: Guidance strength (0.0 = no guidance)

        Returns:
            Predicted velocity field (B, 2, H, W)
        """
        # Broadcast time to batch size if needed
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        if cfg_scale != 0.0 and case_params is not None:
            # Classifier-free guidance
            with torch.no_grad():
                # Conditional prediction (with case_params)
                extra_cond = {
                    'x_prev_1': x_prev_1,
                    'x_prev_2': x_prev_2,
                    'case_params': case_params
                }
                conditional = self.model(x, t, extra_cond)

                # Unconditional prediction (without case_params)
                extra_uncond = {
                    'x_prev_1': x_prev_1,
                    'x_prev_2': x_prev_2,
                    'case_params': None
                }
                unconditional = self.model(x, t, extra_uncond)

            # Guided velocity: v = (1 + scale) * v_cond - scale * v_uncond
            # This matches the flow matching library's CFG implementation
            result = (1.0 + cfg_scale) * conditional - cfg_scale * unconditional
        else:
            # No guidance or no case params
            with torch.no_grad():
                extra = {
                    'x_prev_1': x_prev_1,
                    'x_prev_2': x_prev_2,
                    'case_params': case_params
                }
                result = self.model(x, t, extra)

        self.nfe_counter += 1
        return result.to(dtype=torch.float32)



@torch.no_grad()
def generate_forecast(
    model,
    x_prev_1,
    x_prev_2,
    case_params,
    device,
    num_ode_steps=50,
    cfg_scale=1.0
):
    """
    Generate a single-step forecast using flow matching ODE solver.

    Args:
        model: Trained FluidDynamicsUNet
        x_prev_1: Previous state at t-1, shape (B, 2, H, W)
        x_prev_2: Previous state at t-2, shape (B, 2, H, W)
        case_params: Case parameters, shape (B, D)
        device: Device
        num_ode_steps: Number of ODE integration steps
        cfg_scale: Classifier-free guidance scale. 1.0 = no guidance,
                   >1.0 amplifies conditioning, <1.0 weakens conditioning

    Returns:
        Predicted next state, shape (B, 2, H, W)
    """
    model.eval()

    cfg_model = CFGScaledFluidModel(model)
    cfg_model.eval()
    solver = ODESolver(velocity_model=cfg_model)

    # Start from Gaussian noise
    x_0 = torch.randn_like(x_prev_1)

    # Solve ODE from t=0 to t=1
    x_1 = solver.sample(
        x_init=x_0,
        method='midpoint',
        step_size=0.05,
        cfg_scale=cfg_scale,
        x_prev_1=x_prev_1,
        x_prev_2=x_prev_2,
        case_params=case_params,

        )

    return x_1


@torch.no_grad()
def generate_multistep_forecast(
    model,
    initial_states,
    case_params,
    num_steps,
    device,
    num_ode_steps=50,
    cfg_scale=1.0
):
    """
    Generate multi-step autoregressive forecast.

    Args:
        model: Trained model
        initial_states: List of 2 initial states [x_{t-2}, x_{t-1}], each (B, 2, H, W)
        case_params: Case parameters (B, D)
        num_steps: Number of future steps to predict
        device: Device
        num_ode_steps: Number of ODE integration steps per prediction
        cfg_scale: Classifier-free guidance scale. 
                   
    Returns:
        predictions: List of predicted states, length num_steps, each (B, 2, H, W)
    """
    model.eval()

    predictions = []
    x_prev_2, x_prev_1 = initial_states

    for step in range(num_steps):
        # Generate next prediction
        x_next = generate_forecast(
            model, x_prev_1, x_prev_2, case_params, device, num_ode_steps,
            cfg_scale=cfg_scale
        )
        predictions.append(x_next.cpu())

        # Shift states for next iteration
        x_prev_2 = x_prev_1
        x_prev_1 = x_next

    return predictions


def compute_metrics(predictions, ground_truth, mask=None):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: (T, B, 2, H, W) tensor
        ground_truth: (T, B, 2, H, W) tensor
        mask: Optional (B, 1, H, W) mask
        
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        # Apply mask
        mask = mask.expand_as(predictions[0])
        valid_pixels = mask.sum()
        
        mse = ((predictions - ground_truth) ** 2 * mask).sum() / valid_pixels
        mae = (torch.abs(predictions - ground_truth) * mask).sum() / valid_pixels
    else:
        mse = torch.mean((predictions - ground_truth) ** 2)
        mae = torch.mean(torch.abs(predictions - ground_truth))
    
    # Compute per-timestep metrics
    timestep_mse = []
    for t in range(len(predictions)):
        if mask is not None:
            t_mse = ((predictions[t] - ground_truth[t]) ** 2 * mask).sum() / valid_pixels
        else:
            t_mse = torch.mean((predictions[t] - ground_truth[t]) ** 2)
        timestep_mse.append(t_mse.item())
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': torch.sqrt(mse).item(),
        'timestep_mse': timestep_mse
    }


def visualize_forecast(initial_states, predictions, ground_truth, save_path, sample_idx=0):
    """
    Create visualization of forecast vs ground truth.

    Args:
        initial_states: List of 2 initial states [x_{t-2}, x_{t-1}]
        predictions: List of T predicted states
        ground_truth: List of T ground truth states
        save_path: Path to save figure
        sample_idx: Which sample in batch to visualize
    """
    num_steps = len(predictions)

    # Create figure with GridSpec for better colorbar control
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(3 * (num_steps + 2) + 1, 9))
    gs = GridSpec(3, num_steps + 3, figure=fig, width_ratios=[1] * (num_steps + 2) + [0.05])

    # Create axes for plots (excluding colorbar column)
    axes = []
    for row in range(3):
        row_axes = []
        for col in range(num_steps + 2):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)

    # Collect all data to compute global min/max for consistent scales
    all_u = []
    all_v = []
    all_mag = []

    # Collect initial states
    for state in initial_states:
        u = state[sample_idx, 0].cpu().numpy()
        v = state[sample_idx, 1].cpu().numpy()
        all_u.append(u)
        all_v.append(v)
        all_mag.append(np.sqrt(u**2 + v**2))

    # Collect predictions and ground truth
    for t in range(num_steps):
        pred = predictions[t][sample_idx].cpu().numpy()
        gt = ground_truth[t][sample_idx].cpu().numpy()

        all_u.extend([pred[0], gt[0]])
        all_v.extend([pred[1], gt[1]])
        all_mag.extend([
            np.sqrt(pred[0]**2 + pred[1]**2),
            np.sqrt(gt[0]**2 + gt[1]**2)
        ])

    # Compute global min/max
    u_min, u_max = np.min([u.min() for u in all_u]), np.max([u.max() for u in all_u])
    v_min, v_max = np.min([v.min() for v in all_v]), np.max([v.max() for v in all_v])
    mag_min, mag_max = np.min([m.min() for m in all_mag]), np.max([m.max() for m in all_mag])

    # Plot initial states
    for i, state in enumerate(initial_states):
        u = state[sample_idx, 0].cpu().numpy()
        v = state[sample_idx, 1].cpu().numpy()
        magnitude = np.sqrt(u**2 + v**2)

        im_u = axes[0][i].imshow(u, cmap='RdBu_r', vmin=u_min, vmax=u_max)
        axes[0][i].set_title(f't-{2-i}: u')
        axes[0][i].axis('off')

        im_v = axes[1][i].imshow(v, cmap='RdBu_r', vmin=v_min, vmax=v_max)
        axes[1][i].set_title(f't-{2-i}: v')
        axes[1][i].axis('off')

        im_mag = axes[2][i].imshow(magnitude, cmap='viridis', vmin=mag_min, vmax=mag_max)
        axes[2][i].set_title(f't-{2-i}: |V|')
        axes[2][i].axis('off')

    # Plot predictions and ground truth
    for t in range(num_steps):
        pred = predictions[t][sample_idx].cpu().numpy()
        gt = ground_truth[t][sample_idx].cpu().numpy()

        col = t + 2

        # Prediction u
        axes[0][col].imshow(pred[0], cmap='RdBu_r', vmin=u_min, vmax=u_max)
        axes[0][col].set_title(f't+{t+1}: u (pred)')
        axes[0][col].axis('off')

        # Ground Truth u
        axes[1][col].imshow(gt[0], cmap='RdBu_r', vmin=u_min, vmax=u_max)
        axes[1][col].set_title(f't+{t+1}: Ground Truth u')
        axes[1][col].axis('off')

        # Magnitude comparison (pred and gt side by side in same row)
        pred_mag = np.sqrt(pred[0]**2 + pred[1]**2)
        gt_mag = np.sqrt(gt[0]**2 + gt[1]**2)

        axes[2][col].imshow(pred_mag, cmap='viridis', vmin=mag_min, vmax=mag_max)
        axes[2][col].set_title(f't+{t+1}: |V| (pred)')
        axes[2][col].axis('off')

    # Add colorbars in the rightmost column
    cbar_ax_u = fig.add_subplot(gs[0, -1])
    cbar_u = plt.colorbar(im_u, cax=cbar_ax_u)
    cbar_u.set_label('u velocity', rotation=270, labelpad=15)

    cbar_ax_v = fig.add_subplot(gs[1, -1])
    cbar_v = plt.colorbar(im_v, cax=cbar_ax_v)
    cbar_v.set_label('v velocity', rotation=270, labelpad=15)

    cbar_ax_mag = fig.add_subplot(gs[2, -1])
    cbar_mag = plt.colorbar(im_mag, cax=cbar_ax_mag)
    cbar_mag.set_label('|V|', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {save_path}")

def denormalize_velocity(velocity, stats):
    """Function to denormalize values using training statistics"""
    u_mean, u_std = stats['u_mean'], stats['u_std']
    v_mean, v_std = stats['v_mean'], stats['v_std']
    vel_mean = torch.stack([u_mean, v_mean]).reshape(1, -1, 1, 1).to(velocity.device)
    vel_std = torch.stack([u_std, v_std]).reshape(1, -1, 1, 1).to(velocity.device)
    
    return velocity * vel_std + vel_mean
   
    

def main():
    parser = argparse.ArgumentParser(description='Generate forecasts from trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--data_name', type=str, required=True, help='problem name. Ex: cylinder_geo')
    parser.add_argument('--output_dir', type=str, default='./result', help='Output directory')
    parser.add_argument('--num_forecast_steps', type=int, default=1, help='Number of steps to forecast')
    parser.add_argument('--num_ode_steps', type=int, default=50, help='ODE solver steps')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_numpy', action='store_true', help='Save predictions as numpy arrays')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--cfg_scale', type=float, default=0.3,
                        help='Classifier-free guidance scale. 1.0 = no guidance, '
                             '>1.0 amplifies conditioning, <1.0 weakens conditioning')

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    logger.info("Loading test dataset...")
    data_dir = Path(args.data_dir)
    _, _, base_dataset_test = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['test']
    )
    # Get train stats to use same normalization for testing
    stats_path = Path('./train_stats.pt')
    train_stats = torch.load(stats_path, weights_only=False)
    
    dataset_test = MultiStepFlowCastDataset(
        base_dataset=base_dataset_test,
        num_future_steps=args.num_forecast_steps,
        normalize=True,
        norm_stats=train_stats
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_step_collate_fn
    )
    
    # Load model
    logger.info(f"Loading model from {args.output_dir}...")
    # NOTE: Update these parameters to match your trained checkpoint
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=64,  
        channel_mult=(1, 2, 4),  
        num_res_blocks=2,  
        attention_resolutions=(), 
        dropout=0.2,  
        num_case_params=8,
        use_fourier_conditioning=True,
        num_fourier_freqs=32,  # Updated to match new default
    ).to(device)
    
    checkpoint_path = Path(args.output_dir) / Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    logger.info(f"Using CFG scale: {args.cfg_scale}")

    # Generate forecasts
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating forecasts")):
            if batch_idx * args.batch_size >= args.num_samples:
                break
            
            # Move to device
            x_prev_2 = batch['x_prev_2'].to(device)
            x_prev_1 = batch['x_prev_1'].to(device)
            ground_truth = batch['x_future']
            case_params = batch['case_params'].to(device)
            # Print which case parameters are being used
            print(f'case paramaters: {case_params}')
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # Generate multi-step forecast
            predictions = generate_multistep_forecast(
                model,
                [x_prev_2, x_prev_1],
                case_params,
                args.num_forecast_steps,
                device,
                args.num_ode_steps,
                cfg_scale=args.cfg_scale
            )
            
            
            # Save predictions
            if args.save_numpy:
                for b in range(predictions[0].shape[0]):
                    sample_id = batch_idx * args.batch_size + b
                    pred_array = torch.stack([p[b] for p in predictions]).numpy()
                    np.save(output_dir / f'prediction_{sample_id:04d}.npy', pred_array)
            
            # Visualize
            if args.visualize and batch_idx < 5:  # Only visualize first 5 batches
                # denormalize all data
                x_prev_1_denorm = denormalize_velocity(x_prev_1, train_stats)
                x_prev_2_denorm = denormalize_velocity(x_prev_2, train_stats)
                predictions_denorm = [denormalize_velocity(pred, train_stats) for pred in predictions]
                ground_truth_denorm = [denormalize_velocity(gt, train_stats) for gt in ground_truth]

                visualize_forecast(
                    [x_prev_2_denorm.cpu(), x_prev_1_denorm.cpu()],
                    predictions_denorm,
                    ground_truth_denorm,
                    output_dir / f'forecast_batch_{batch_idx:04d}.png',
                    sample_idx=0
                )
    
                logger.info(f"Forecasts saved to {output_dir}")
    
    # Save summary
    summary = {
        'checkpoint': str(args.checkpoint),
        'num_forecast_steps': args.num_forecast_steps,
        'num_ode_steps': args.num_ode_steps,
        'cfg_scale': args.cfg_scale,
        'num_samples_processed': min(args.num_samples, len(dataset_test)),
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
