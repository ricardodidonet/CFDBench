"""
Evaluation loop for fluid dynamics flow matching model.

Adapted from examples/image/training/eval_loop.py for CFD applications.
Computes prediction accuracy metrics and generates visualizations.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 10


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

            # Guided velocity: v = v_uncond + scale * (v_cond - v_uncond)
            result = unconditional + cfg_scale * (conditional - unconditional)
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

    def reset_nfe_counter(self):
        """Reset the function evaluation counter."""
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        """Get number of function evaluations."""
        return self.nfe_counter


def compute_physical_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute physical metrics for fluid dynamics.

    Args:
        predicted: Predicted velocity fields (B, 2, H, W)
        target: Ground truth velocity fields (B, 2, H, W)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Mean Squared Error
    mse = torch.nn.functional.mse_loss(predicted, target)
    metrics['mse'] = mse.item()

    # Mean Absolute Error
    mae = torch.nn.functional.l1_loss(predicted, target)
    metrics['mae'] = mae.item()

    # Relative error (L2 norm)
    rel_error = torch.norm(predicted - target) / torch.norm(target)
    metrics['relative_error'] = rel_error.item()

    # Per-component errors
    u_mse = torch.nn.functional.mse_loss(predicted[:, 0], target[:, 0])
    v_mse = torch.nn.functional.mse_loss(predicted[:, 1], target[:, 1])
    metrics['u_mse'] = u_mse.item()
    metrics['v_mse'] = v_mse.item()

    # Velocity magnitude error
    pred_mag = torch.sqrt(predicted[:, 0]**2 + predicted[:, 1]**2)
    target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
    mag_mse = torch.nn.functional.mse_loss(pred_mag, target_mag)
    metrics['magnitude_mse'] = mag_mse.item()

    return metrics


def single_step_prediction(
    model: CFGScaledFluidModel,
    x_prev_1: torch.Tensor,
    x_prev_2: torch.Tensor,
    case_params: Optional[torch.Tensor],
    device: torch.device,
    cfg_scale: float = 0.0,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Generate single-step prediction using ODE solver.

    Args:
        model: Wrapped model with CFG
        x_prev_1: Previous state at t-1 (B, 2, H, W)
        x_prev_2: Previous state at t-2 (B, 2, H, W)
        case_params: Case parameters (B, D) or None
        device: Device
        cfg_scale: Guidance scale
        num_steps: Number of ODE integration steps

    Returns:
        Predicted next state (B, 2, H, W)
    """
    model.reset_nfe_counter()

    # Create wrapper for ODE solver
    class SolverWrapper(nn.Module):
        def __init__(self, model, x_prev_1, x_prev_2, case_params, cfg_scale):
            super().__init__()
            self.model = model
            self.x_prev_1 = x_prev_1
            self.x_prev_2 = x_prev_2
            self.case_params = case_params
            self.cfg_scale = cfg_scale

        def forward(self, t, x):
            return self.model(
                x, t,
                self.x_prev_1,
                self.x_prev_2,
                self.case_params,
                self.cfg_scale
            )

    wrapped = SolverWrapper(model, x_prev_1, x_prev_2, case_params, cfg_scale)
    solver = ODESolver(velocity_model=wrapped)

    # Start from Gaussian noise
    x_0 = torch.randn_like(x_prev_1)

    # Sample from flow
    x_1 = solver.sample(x_0, step_size=1.0 / num_steps)

    return x_1


def autoregressive_rollout(
    model: CFGScaledFluidModel,
    x_init_1: torch.Tensor,
    x_init_2: torch.Tensor,
    ground_truth_sequence: torch.Tensor,
    case_params: Optional[torch.Tensor],
    device: torch.device,
    cfg_scale: float = 0.0,
    num_steps_per_pred: int = 50,
) -> Dict[str, torch.Tensor]:
    """
    Perform autoregressive rollout and compute errors at each step.

    Args:
        model: Wrapped model
        x_init_1: Initial state at t-1 (B, 2, H, W)
        x_init_2: Initial state at t-2 (B, 2, H, W)
        ground_truth_sequence: Ground truth states (B, T, 2, H, W)
        case_params: Case parameters (B, D) or None
        device: Device
        cfg_scale: Guidance scale
        num_steps_per_pred: ODE steps per prediction

    Returns:
        Dictionary with predictions and errors
    """
    num_rollout_steps = ground_truth_sequence.shape[1]
    predictions = []
    errors = []

    x_prev_2 = x_init_2
    x_prev_1 = x_init_1

    for step in range(num_rollout_steps):
        # Predict next state
        x_next = single_step_prediction(
            model, x_prev_1, x_prev_2, case_params, device, cfg_scale, num_steps_per_pred
        )

        predictions.append(x_next.cpu())

        # Compute error against ground truth
        gt = ground_truth_sequence[:, step]
        step_error = torch.nn.functional.mse_loss(x_next, gt)
        errors.append(step_error.item())

        # Update history
        x_prev_2 = x_prev_1
        x_prev_1 = x_next

    return {
        'predictions': torch.stack(predictions, dim=1),  # (B, T, 2, H, W)
        'errors': torch.tensor(errors),  # (T,)
    }


def visualize_prediction(
    predicted: torch.Tensor,
    target: torch.Tensor,
    save_path: Path,
    sample_idx: int = 0,
):
    """
    Visualize predicted vs ground truth velocity fields.

    Args:
        predicted: Predicted field (B, 2, H, W)
        target: Ground truth field (B, 2, H, W)
        save_path: Path to save figure
        sample_idx: Which sample in batch to visualize
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert to numpy
    pred_u = predicted[sample_idx, 0].cpu().numpy()
    pred_v = predicted[sample_idx, 1].cpu().numpy()
    target_u = target[sample_idx, 0].cpu().numpy()
    target_v = target[sample_idx, 1].cpu().numpy()

    # Compute magnitudes
    pred_mag = np.sqrt(pred_u**2 + pred_v**2)
    target_mag = np.sqrt(target_u**2 + target_v**2)

    # Plot U component
    im0 = axes[0, 0].imshow(target_u, cmap='RdBu_r')
    axes[0, 0].set_title('Ground Truth U')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[1, 0].imshow(pred_u, cmap='RdBu_r')
    axes[1, 0].set_title('Predicted U')
    plt.colorbar(im1, ax=axes[1, 0])

    # Plot V component
    im2 = axes[0, 1].imshow(target_v, cmap='RdBu_r')
    axes[0, 1].set_title('Ground Truth V')
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 1].imshow(pred_v, cmap='RdBu_r')
    axes[1, 1].set_title('Predicted V')
    plt.colorbar(im3, ax=axes[1, 1])

    # Plot magnitude
    im4 = axes[0, 2].imshow(target_mag, cmap='viridis')
    axes[0, 2].set_title('Ground Truth Magnitude')
    plt.colorbar(im4, ax=axes[0, 2])

    im5 = axes[1, 2].imshow(pred_mag, cmap='viridis')
    axes[1, 2].set_title('Predicted Magnitude')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    cfg_scale: float = 0.0,
    num_ode_steps: int = 50,
    num_rollout_steps: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate fluid dynamics model.

    - Single-step prediction accuracy
    - Autoregressive rollout error
    - Physical metrics
    - Visualization of predictions

    Args:
        model: Trained model
        data_loader: Validation data loader
        device: Device
        epoch: Current epoch (for logging)
        cfg_scale: Classifier-free guidance scale
        num_ode_steps: Number of ODE integration steps
        num_rollout_steps: Number of autoregressive steps
        output_dir: Directory to save visualizations

    Returns:
        Dictionary of evaluation metrics
    """
    gc.collect()

    # Wrap model with CFG
    cfg_model = CFGScaledFluidModel(model=model)
    cfg_model.train(False)

    # Metrics accumulators
    all_single_step_metrics = {
        'mse': [],
        'mae': [],
        'relative_error': [],
        'u_mse': [],
        'v_mse': [],
        'magnitude_mse': [],
    }
    all_rollout_errors = []
    total_nfe = 0
    num_samples = 0

    # Create output directories
    if output_dir:
        output_dir = Path(output_dir)
        (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        (output_dir / "rollouts").mkdir(parents=True, exist_ok=True)

    snapshots_saved = False

    logger.info(f"\nEvaluating model at epoch {epoch}")
    logger.info(f"CFG scale: {cfg_scale}")
    logger.info(f"ODE steps: {num_ode_steps}")

    for batch_idx, batch in enumerate(data_loader):
        # Get data
        x_target = batch['x_target'].to(device)
        x_prev_1 = batch['x_prev_1'].to(device)
        x_prev_2 = batch['x_prev_2'].to(device)
        case_params = batch.get('case_params', None)
        if case_params is not None:
            case_params = case_params.to(device)

        # Single-step prediction
        predicted = single_step_prediction(
            cfg_model, x_prev_1, x_prev_2, case_params, device, cfg_scale, num_ode_steps
        )

        # Compute metrics
        metrics = compute_physical_metrics(predicted, x_target)
        for key, value in metrics.items():
            all_single_step_metrics[key].append(value)

        total_nfe += cfg_model.get_nfe()
        num_samples += x_target.shape[0]

        # Save first batch visualization
        if not snapshots_saved and output_dir:
            visualize_prediction(
                predicted, x_target,
                output_dir / "snapshots" / f"epoch_{epoch}_batch_{batch_idx}.png"
            )
            snapshots_saved = True

        # Autoregressive rollout (on first batch only to save time)
        if batch_idx == 0 and 'ground_truth_sequence' in batch:
            logger.info("Performing autoregressive rollout...")
            gt_sequence = batch['ground_truth_sequence'].to(device)  # (B, T, 2, H, W)

            rollout_results = autoregressive_rollout(
                cfg_model, x_prev_1, x_prev_2, gt_sequence, case_params,
                device, cfg_scale, num_ode_steps
            )

            all_rollout_errors.extend(rollout_results['errors'].tolist())

            # Plot rollout errors
            if output_dir:
                plt.figure(figsize=(10, 6))
                plt.plot(rollout_results['errors'].numpy())
                plt.xlabel('Rollout Step')
                plt.ylabel('MSE')
                plt.title(f'Autoregressive Rollout Error (Epoch {epoch})')
                plt.grid(True)
                plt.savefig(output_dir / "rollouts" / f"rollout_error_epoch_{epoch}.png")
                plt.close()

        # Periodic logging
        if batch_idx % PRINT_FREQUENCY == 0:
            gc.collect()
            avg_mse = np.mean(all_single_step_metrics['mse'])
            logger.info(
                f"Eval [{batch_idx}/{len(data_loader)}] "
                f"MSE: {avg_mse:.6f} "
                f"NFE/sample: {total_nfe/num_samples:.1f}"
            )

    # Compute average metrics
    eval_metrics = {
        f'eval_{key}': np.mean(values)
        for key, values in all_single_step_metrics.items()
    }

    if all_rollout_errors:
        eval_metrics['eval_rollout_error_mean'] = np.mean(all_rollout_errors)
        eval_metrics['eval_rollout_error_final'] = all_rollout_errors[-1]

    eval_metrics['eval_nfe_per_sample'] = total_nfe / num_samples

    # Log final results
    logger.info("\n" + "="*60)
    logger.info(f"Evaluation Results (Epoch {epoch}):")
    logger.info("="*60)
    for key, value in eval_metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    logger.info("="*60)

    return eval_metrics