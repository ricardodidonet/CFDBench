"""
Updated CFGScaledFluidModel to better align with flow_matching library patterns
while supporting autoregressive fluid dynamics requirements.
"""

import torch
import torch.nn as nn
from typing import Optional
from flow_matching.utils import ModelWrapper


class CFGScaledFluidModel(ModelWrapper):
    """
    CFG wrapper for fluid dynamics with temporal conditioning.

    Differences from library's CFGScaledModel:
    - Supports temporal conditioning (x_prev_1, x_prev_2) that's always present
    - Only applies CFG to case_params (physical/geometric conditioning)
    - Compatible with library's ODESolver API
    """

    def __init__(self, model: nn.Module, x_prev_1: torch.Tensor, x_prev_2: torch.Tensor):
        """
        Args:
            model: Base fluid dynamics model
            x_prev_1: Previous state at t-1 (B, 2, H, W) - fixed for this sequence
            x_prev_2: Previous state at t-2 (B, 2, H, W) - fixed for this sequence
        """
        super().__init__(model)
        self.x_prev_1 = x_prev_1
        self.x_prev_2 = x_prev_2
        self.nfe_counter = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        """
        Forward pass compatible with flow_matching library's API.

        Args:
            x: Current state (B, 2, H, W)
            t: Time (scalar or (B,))
            **extras: Must contain 'case_params' and optionally 'cfg_scale'

        Returns:
            Predicted velocity field (B, 2, H, W)
        """
        # Extract from extras (library-compatible API)
        case_params = extras.get('case_params', None)
        cfg_scale = extras.get('cfg_scale', 0.0)

        # Broadcast time to batch size if needed
        if t.dim() == 0:
            t = t.repeat(x.shape[0])

        if cfg_scale != 0.0 and case_params is not None:
            # Classifier-free guidance on case_params only
            with torch.no_grad():
                # Conditional prediction (with case_params)
                extra_cond = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
                    'case_params': case_params
                }
                conditional = self.model(x, t, extra_cond)

                # Unconditional prediction (drop case_params only)
                extra_uncond = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
                    'case_params': None
                }
                unconditional = self.model(x, t, extra_uncond)

            # Guided velocity: v = (1 + scale) * v_cond - scale * v_uncond
            result = (1.0 + cfg_scale) * conditional - cfg_scale * unconditional
        else:
            # No guidance
            with torch.no_grad():
                extra = {
                    'x_prev_1': self.x_prev_1,
                    'x_prev_2': self.x_prev_2,
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


def generate_forecast_improved(
    model: nn.Module,
    x_prev_1: torch.Tensor,
    x_prev_2: torch.Tensor,
    case_params: Optional[torch.Tensor],
    device: torch.device,
    cfg_scale: float = 0.0,
    num_ode_steps: int = 50,
    ode_method: str = 'dopri5',  # Can use adaptive methods now!
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """
    Generate forecast using library-compatible API with advanced solver options.

    Args:
        model: Trained FluidDynamicsUNet
        x_prev_1: Previous state at t-1 (B, 2, H, W)
        x_prev_2: Previous state at t-2 (B, 2, H, W)
        case_params: Case parameters (B, D) or None
        device: Device
        cfg_scale: Classifier-free guidance scale
        num_ode_steps: Number of ODE steps (for fixed-step methods)
        ode_method: ODE solver method ('dopri5', 'euler', 'midpoint', 'rk4')
        atol: Absolute tolerance for adaptive methods
        rtol: Relative tolerance for adaptive methods

    Returns:
        Predicted next state (B, 2, H, W)
    """
    from flow_matching.solver import ODESolver
    import numpy as np

    model.eval()

    # Wrap model with CFG (temporal conditioning baked in)
    cfg_model = CFGScaledFluidModel(model, x_prev_1, x_prev_2)
    cfg_model.train(False)  # Explicit eval mode (library convention)

    # Create solver with the wrapped model
    solver = ODESolver(velocity_model=cfg_model)

    # Initial noise
    x_0 = torch.randn_like(x_prev_1)

    # Time grid
    if ode_method == 'euler' or num_ode_steps:
        # Fixed-step methods
        time_grid = torch.linspace(0, 1, num_ode_steps + 1, device=device)
    else:
        # Adaptive methods just need start and end
        time_grid = torch.tensor([0.0, 1.0], device=device)

    # Reset counter
    cfg_model.reset_nfe_counter()

    # Sample using library's API
    x_1 = solver.sample(
        time_grid=time_grid,
        x_init=x_0,
        method=ode_method,
        return_intermediates=False,
        atol=atol if ode_method == 'dopri5' else None,
        rtol=rtol if ode_method == 'dopri5' else None,
        case_params=case_params,  # Passed to model via **extras
        cfg_scale=cfg_scale,       # Passed to model via **extras
    )

    nfe = cfg_model.get_nfe()
    print(f"NFE: {nfe}, Method: {ode_method}")

    return x_1


# Example usage comparison
def example_usage():
    """
    Example showing how to use the improved API.
    """
    import torch
    from models.fluid_unet import FluidDynamicsUNet

    device = torch.device('cuda')
    model = FluidDynamicsUNet(...).to(device)
    model.load_state_dict(...)

    x_prev_2 = ...  # (B, 2, H, W)
    x_prev_1 = ...  # (B, 2, H, W)
    case_params = ...  # (B, 8)

    # OLD WAY (works but limited):
    # x_1 = generate_forecast(model, x_prev_1, x_prev_2, case_params, device, num_ode_steps=50)

    # NEW WAY (library-compatible, more options):
    x_1 = generate_forecast_improved(
        model, x_prev_1, x_prev_2, case_params, device,
        cfg_scale=1.0,
        num_ode_steps=50,
        ode_method='dopri5',  # Adaptive solver - adjusts step size automatically!
        atol=1e-5,
        rtol=1e-5
    )

    # Can also try different methods:
    # ode_method='euler'    - Simple Euler (fast, less accurate)
    # ode_method='midpoint' - 2nd order
    # ode_method='rk4'      - 4th order Runge-Kutta
    # ode_method='dopri5'   - Adaptive 5th order (best quality, slower)
