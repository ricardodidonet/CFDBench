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

from dataset import get_auto_dataset
from dataset.wrapper import FlowCastWrapperDataset
from dataset.multistep_wrapper import MultiStepFlowCastDataset, multi_step_collate_fn
from models.fluid_unet import FluidDynamicsUNet
from flow_matching.solver import ODESolver

logger = logging.getLogger(__name__)


def collate_fn(batch):
    x_prev_2 = torch.stack([item['x_prev_2'] for item in batch])
    x_prev_1 = torch.stack([item['x_prev_1'] for item in batch])
    x_target = torch.stack([item['x_target'] for item in batch])
    
    # Auto-detect: Get keys from first sample, use consistently
    case_params_list = [item['case_params'] for item in batch]
    param_keys = sorted(case_params_list[0].keys())  # Sort for consistency!
    
    # Convert to tensor with fixed key order
    case_params_tensor = torch.tensor([
        [float(params[key]) for key in param_keys]
        for params in case_params_list
    ])
    
    return {
        'x_prev_2': x_prev_2[:, :2],      # Only velocity channels (u, v)
        'x_prev_1': x_prev_1[:, :2],      # Only velocity channels (u, v)
        'x_target': x_target[:, :2],      # Only velocity channels (u, v)
        'mask': x_target[:, 2:3],         # Keep mask channel separate (shape: B, 1, H, W)
        'case_params': case_params_tensor
    }


@torch.no_grad()
def generate_forecast(model, x_prev_1, x_prev_2, case_params, device, num_ode_steps=50):
    """
    Generate a single-step forecast using flow matching ODE solver.
    
    Args:
        model: Trained FluidDynamicsUNet
        x_prev_1: Previous state at t-1, shape (B, 2, H, W)
        x_prev_2: Previous state at t-2, shape (B, 2, H, W)
        case_params: Case parameters, shape (B, D)
        device: Device
        num_ode_steps: Number of ODE integration steps
        
    Returns:
        Predicted next state, shape (B, 2, H, W)
    """
    model.eval()
    
    # Wrapper for ODESolver
    class ModelWrapper(nn.Module):
        def __init__(self, model, x_prev_1, x_prev_2, case_params):
            super().__init__()
            self.model = model
            self.x_prev_1 = x_prev_1
            self.x_prev_2 = x_prev_2
            self.case_params = case_params
        
        def forward(self, t, x):
            batch_size = x.shape[0]
            if t.dim() == 0:
                t = t.repeat(batch_size)
            
            extra = {
                'x_prev_1': self.x_prev_1,
                'x_prev_2': self.x_prev_2,
                'case_params': self.case_params
            }
            return self.model(x, t, extra)
    
    wrapped_model = ModelWrapper(model, x_prev_1, x_prev_2, case_params)
    solver = ODESolver(wrapped_model)
    
    # Start from Gaussian noise
    x_0 = torch.randn_like(x_prev_1)
    
    # Solve ODE from t=0 to t=1
    x_1 = solver.sample(x_0, step_size=1.0 / num_ode_steps)
    
    return x_1


@torch.no_grad()
def generate_multistep_forecast(
    model, 
    initial_states, 
    case_params, 
    num_steps, 
    device, 
    num_ode_steps=50
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
        
    Returns:
        predictions: List of predicted states, length num_steps, each (B, 2, H, W)
    """
    model.eval()
    
    predictions = []
    x_prev_2, x_prev_1 = initial_states
    
    for step in range(num_steps):
        # Generate next prediction
        x_next = generate_forecast(
            model, x_prev_1, x_prev_2, case_params, device, num_ode_steps
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
    fig, axes = plt.subplots(3, num_steps + 2, figsize=(3 * (num_steps + 2), 9))
    
    # Plot initial states
    for i, state in enumerate(initial_states):
        u = state[sample_idx, 0].cpu().numpy()
        v = state[sample_idx, 1].cpu().numpy()
        magnitude = np.sqrt(u**2 + v**2)
        
        axes[0, i].imshow(u, cmap='RdBu_r')
        axes[0, i].set_title(f't-{2-i}: u')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(v, cmap='RdBu_r')
        axes[1, i].set_title(f't-{2-i}: v')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(magnitude, cmap='viridis')
        axes[2, i].set_title(f't-{2-i}: |V|')
        axes[2, i].axis('off')
    
    # Plot predictions and ground truth
    for t in range(num_steps):
        pred = predictions[t][sample_idx].cpu().numpy()
        gt = ground_truth[t][sample_idx].cpu().numpy()
        
        col = t + 2
        
        # Prediction u
        axes[0, col].imshow(pred[0], cmap='RdBu_r')
        axes[0, col].set_title(f't+{t+1}: u (pred)')
        axes[0, col].axis('off')
        
        # Prediction v
        axes[1, col].imshow(pred[1], cmap='RdBu_r')
        axes[1, col].set_title(f't+{t+1}: v (pred)')
        axes[1, col].axis('off')
        
        # Magnitude with error overlay
        pred_mag = np.sqrt(pred[0]**2 + pred[1]**2)
        gt_mag = np.sqrt(gt[0]**2 + gt[1]**2)
        error = np.abs(pred_mag - gt_mag)
        
        axes[2, col].imshow(pred_mag, cmap='viridis', alpha=0.7)
        im = axes[2, col].imshow(error, cmap='Reds', alpha=0.5)
        axes[2, col].set_title(f't+{t+1}: |V| + error')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate forecasts from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./result', help='Output directory')
    parser.add_argument('--num_forecast_steps', type=int, default=10, help='Number of steps to forecast')
    parser.add_argument('--num_ode_steps', type=int, default=50, help='ODE solver steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_numpy', action='store_true', help='Save predictions as numpy arrays')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
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

    # Get training dataset to compute normalization stats
    base_dataset_train, _, base_dataset_test = get_auto_dataset(
        data_dir=data_dir,
        data_name='cylinder_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['train', 'test']
    )

    # Compute normalization stats from training set
    temp_train = FlowCastWrapperDataset(base_dataset_train, normalize=True)
    norm_stats = temp_train.get_norm_stats()
    logger.info("Using training set normalization stats for test set")

    # Use MultiStepFlowCastDataset for proper ground truth
    dataset_test = MultiStepFlowCastDataset(
        base_dataset_test,
        num_future_steps=args.num_forecast_steps,
        normalize=True,
        norm_stats=norm_stats
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_step_collate_fn  # Use multi-step collate function
    )
    
    # Load model
    logger.info(f"Loading model from {args.output_dir}...")
    # NOTE: Update these parameters to match your trained checkpoint
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=64,  # Updated to match new default
        channel_mult=(1, 2, 4),  # Updated to match new default
        num_res_blocks=1,  # Updated to match new default
        attention_resolutions=(),  # No attention by default
        dropout=0.2,  # Updated to match new default
        num_case_params=8,
        use_fourier_conditioning=True,
        num_fourier_freqs=8,  # Updated to match new default
    ).to(device)
    
    checkpoint_path = Path(args.output_dir) / Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # Generate forecasts
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating forecasts")):
            if batch_idx * args.batch_size >= args.num_samples:
                break
            
            # Move to device
            x_prev_2 = batch['x_prev_2'].to(device)
            x_prev_1 = batch['x_prev_1'].to(device)
            case_params = batch['case_params'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # Ground truth from dataset (list of K tensors, each (B, 2, H, W))
            ground_truth = [x.to(device) for x in batch['x_future']]

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

            # Move predictions to device for metrics
            predictions_device = [p.to(device) for p in predictions]

            # Compute metrics
            metrics = compute_metrics(
                torch.stack(predictions_device),  # (T, B, 2, H, W)
                torch.stack(ground_truth),        # (T, B, 2, H, W)
                mask
            )
            all_metrics.append(metrics)

            logger.info(f"Batch {batch_idx}: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}")

            # Save predictions and ground truth
            if args.save_numpy:
                for b in range(predictions[0].shape[0]):
                    sample_id = batch_idx * args.batch_size + b
                    pred_array = torch.stack([p[b] for p in predictions]).cpu().numpy()
                    gt_array = torch.stack([g[b] for g in ground_truth]).cpu().numpy()
                    np.save(output_dir / f'prediction_{sample_id:04d}.npy', pred_array)
                    np.save(output_dir / f'ground_truth_{sample_id:04d}.npy', gt_array)

            # Visualize
            if args.visualize and batch_idx < 5:  # Only visualize first 5 batches
                visualize_forecast(
                    [x_prev_2.cpu(), x_prev_1.cpu()],
                    predictions,  # Already on CPU
                    [g.cpu() for g in ground_truth],  # Move to CPU
                    output_dir / f'forecast_batch_{batch_idx:04d}.png',
                    sample_idx=0
                )
    
    logger.info(f"Forecasts saved to {output_dir}")

    # Aggregate metrics
    if all_metrics:
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics])
        avg_mae = np.mean([m['mae'] for m in all_metrics])

        # Average timestep MSE
        num_steps = len(all_metrics[0]['timestep_mse'])
        avg_timestep_mse = [
            np.mean([m['timestep_mse'][t] for m in all_metrics])
            for t in range(num_steps)
        ]

        logger.info(f"\n{'='*60}")
        logger.info(f"OVERALL METRICS")
        logger.info(f"{'='*60}")
        logger.info(f"Average MSE:  {avg_mse:.6f}")
        logger.info(f"Average RMSE: {avg_rmse:.6f}")
        logger.info(f"Average MAE:  {avg_mae:.6f}")
        logger.info(f"\nPer-timestep MSE:")
        for t, mse in enumerate(avg_timestep_mse):
            logger.info(f"  t+{t+1}: {mse:.6f}")

    # Save summary
    summary = {
        'checkpoint': str(args.checkpoint),
        'num_forecast_steps': args.num_forecast_steps,
        'num_ode_steps': args.num_ode_steps,
        'cfg_scale': args.cfg_scale,
        'num_samples_processed': min(args.num_samples, len(dataset_test)),
    }

    if all_metrics:
        summary['metrics'] = {
            'avg_mse': float(avg_mse),
            'avg_rmse': float(avg_rmse),
            'avg_mae': float(avg_mae),
            'timestep_mse': [float(x) for x in avg_timestep_mse]
        }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
