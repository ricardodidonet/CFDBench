import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Optional, List
from tqdm import tqdm

from .base_model import AutoCfdModel
from .loss import MseLoss
from .punetg import PUNetGCFD


class FlowCast(AutoCfdModel):
    """
    FlowCast: Conditional Flow Matching Model for CFD.

    Uses Rectified Flow / Flow Matching instead of DDPM for faster,
    more deterministic generation of CFD residuals.

    Predicts the *normalized residual* between frames (X_t - X_{t-1}),
    using second-order conditioning (X_{t-1} and X_{t-2}).

    Key differences from DDPM (GenCast):
    1. Trains on velocity field v = data - noise instead of noise prediction
    2. Uses continuous time t ∈ [0, 1] instead of discrete timesteps
    3. Samples via ODE integration (Euler/Heun) instead of DDPM denoising
    4. Typically faster: 10-50 steps vs 50-1000 for DDPM

    Conditioning Signals:
    1. X_{t-1} (inputs): Spatial conditioning via channel concatenation
    2. X_{t-2} (inputs_prev): Spatial conditioning via channel concatenation
    3. Case parameters (case_params): Global conditioning via PUNetG embedding
    4. Flow time (t): Global conditioning via PUNetG embedding
    5. Mask (mask): Handled internally by PUNetG
    """

    def __init__(
        self,
        in_chan: int,        # Channels of the input frame (e.g., 2 for u, v)
        out_chan: int,       # Channels of the output/residual (e.g., 2 for u, v)
        loss_fn: MseLoss,    # Loss function instance
        n_case_params: int,  # Number of case parameters
        residual_mean: torch.Tensor,  # Pre-calculated mean of residuals
        residual_std: torch.Tensor,   # Pre-calculated std dev of residuals
        image_size: int = 64,         # Height/Width of the input frames
        num_flow_steps: int = 1000,   # Number of flow discretization steps
        use_gradient_checkpointing: bool = True,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(loss_fn)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.n_case_params = n_case_params
        self.image_size = image_size
        self.num_flow_steps = num_flow_steps

        # --- U-Net Input Channels Calculation ---
        # Input: flow state x_t (out_chan)
        # Condition 1: X_{t-1} (in_chan)
        # Condition 2: X_{t-2} (in_chan)
        unet_in_channels = out_chan + in_chan + in_chan

        # --- Initialize PUNetG U-Net ---
        self.unet = PUNetGCFD(
            in_channels=unet_in_channels,
            out_channels=out_chan,  # Output predicts velocity field
            base_channels=base_channels,
            n_case_params=n_case_params,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # --- Store Residual Normalization Statistics ---
        # Ensure shape [1, C, 1, 1] for broadcasting
        if residual_mean.ndim == 1:
            residual_mean = residual_mean.view(1, -1, 1, 1)
        if residual_std.ndim == 1:
            residual_std = residual_std.view(1, -1, 1, 1)
        self.register_buffer('residual_mean', residual_mean)
        self.register_buffer('residual_std', residual_std)

    def forward(
        self,
        inputs: Tensor,             # X_{t-1} velocity [B, in_chan, H, W]
        inputs_prev: Tensor,        # X_{t-2} velocity [B, in_chan, H, W]
        label: Optional[Tensor],    # X_{t} velocity   [B, out_chan, H, W]
        case_params: Tensor,        # Case parameters  [B, n_case_params]
        mask: Optional[Tensor],     # Mask             [B, 1, H, W]
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Training forward pass using Rectified Flow.

        The model learns to predict the velocity field that transports
        noise to data along a straight path (rectified flow).
        """
        if label is None:
            raise ValueError("FlowCast requires a 'label' during training.")
        if mask is None:
            print("Warning: Mask not provided. Assuming all regions are valid.")
            mask = torch.ones_like(label[:, 0:1])

        batch_size = inputs.shape[0]
        device = inputs.device

        # --- 1. Calculate and Normalize the Residual ---
        raw_residual = label - inputs
        normalized_residual = (raw_residual - self.residual_mean) / (self.residual_std + 1e-6)

        # --- 2. Sample Continuous Time t ∈ [0, 1] ---
        # Uniform sampling for rectified flow
        t = torch.rand((batch_size,), device=device)
        t_expanded = t.view(-1, 1, 1, 1)  # Shape: [B, 1, 1, 1] for broadcasting

        # --- 3. Sample Noise (Starting Point at t=0) ---
        noise = torch.randn_like(normalized_residual)

        # --- 4. Linear Interpolation (Rectified Flow Path) ---
        # x_t = t * data + (1-t) * noise
        # This creates a straight path from noise (t=0) to data (t=1)
        x_t = t_expanded * normalized_residual + (1 - t_expanded) * noise

        # --- 5. Compute Target Velocity Field ---
        # The velocity field that moves from noise to data
        # v(x, t) = dx/dt = data - noise (for linear interpolation)
        velocity_target = normalized_residual - noise

        # --- 6. Prepare U-Net Input: Concatenate Conditions ---
        # [flow_state x_t, X_{t-1}, X_{t-2}]
        unet_input = torch.cat([x_t, inputs, inputs_prev], dim=1)

        # --- 7. Prepare Continuous Time for U-Net ---
        # Scale t ∈ [0, 1] to a larger range for better embedding resolution
        # Using num_flow_steps as a scale factor (e.g., t=0.5 → timestep=500.0)
        timesteps = t * self.num_flow_steps  # Keep as float for sinusoidal embedding

        # --- 8. Predict Velocity Field using PUNetG ---
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory during training
            velocity_pred = torch.utils.checkpoint.checkpoint(
                self.unet,
                unet_input,
                timesteps,
                case_params,
                mask,
                use_reentrant=False
            )
        else:
            velocity_pred = self.unet(
                x=unet_input,
                timesteps=timesteps,
                case_params=case_params,
                mask=mask
            )

        # --- 9. Compute Loss on Velocity Prediction ---
        # Train the model to predict the velocity field
        loss_dict = self.loss_fn(preds=velocity_pred, labels=velocity_target)

        # Ensure 'mse' key exists (required by training script)
        if 'mse' not in loss_dict:
            loss_dict['mse'] = loss_dict.get('nmse', torch.tensor(0.0, device=device)) * \
                              (torch.square(velocity_target).mean() + 1e-8)

        return {
            "preds": velocity_pred,
            "loss": loss_dict
        }

    @torch.no_grad()
    def generate(
        self,
        inputs: Tensor,         # X_{t-1} velocity [B, in_chan, H, W]
        inputs_prev: Tensor,    # X_{t-2} velocity [B, in_chan, H, W]
        case_params: Tensor,    # Case parameters  [B, n_case_params]
        mask: Optional[Tensor], # Mask             [B, 1, H, W]
        num_inference_steps: int = 50,   # Number of ODE integration steps
        solver: str = "euler",           # ODE solver: "euler" or "heun"
        **kwargs,
    ) -> Tensor:
        """
        Generates the next frame (X_t) using Flow ODE integration.

        Integrates the learned velocity field from t=0 (noise) to t=1 (data).

        Args:
            inputs: Previous frame X_{t-1}
            inputs_prev: Frame before that X_{t-2}
            case_params: Physical parameters
            mask: Spatial mask
            num_inference_steps: Number of integration steps (fewer = faster)
            solver: ODE solver ("euler" for 1st order, "heun" for 2nd order)

        Returns:
            Generated next frame X_t
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        if mask is None:
            mask = torch.ones_like(inputs[:, 0:1])

        # --- 1. Start with Random Noise (t=0) ---
        residual_shape = (batch_size, self.out_chan, self.image_size, self.image_size)
        x_t = torch.randn(residual_shape, device=device)

        # --- 2. Integrate ODE from t=0 to t=1 ---
        dt = 1.0 / num_inference_steps

        if solver == "euler":
            # Euler method (1st order ODE solver)
            # x_{t+dt} = x_t + dt * v(x_t, t)
            for i in tqdm(range(num_inference_steps), desc="Flow ODE (Euler)", leave=False):
                t = i * dt
                # Create continuous timestep values for sinusoidal embedding
                timestep_batch = torch.full(
                    (batch_size,),
                    t * self.num_flow_steps,
                    device=device,
                    dtype=torch.float32
                )

                # Prepare input: concatenate flow state with conditioning frames
                unet_input = torch.cat([x_t, inputs, inputs_prev], dim=1)

                # Predict velocity at current time
                velocity_pred = self.unet(
                    x=unet_input,
                    timesteps=timestep_batch,
                    case_params=case_params,
                    mask=mask
                )

                # Euler step
                x_t = x_t + dt * velocity_pred

        elif solver == "heun":
            # Heun's method (2nd order ODE solver, more accurate)
            # Similar to RK2 / improved Euler method
            for i in tqdm(range(num_inference_steps), desc="Flow ODE (Heun)", leave=False):
                t = i * dt
                # Create continuous timestep values for sinusoidal embedding
                timestep_batch = torch.full(
                    (batch_size,),
                    t * self.num_flow_steps,
                    device=device,
                    dtype=torch.float32
                )

                # First velocity prediction (at t)
                unet_input = torch.cat([x_t, inputs, inputs_prev], dim=1)
                v1 = self.unet(
                    x=unet_input,
                    timesteps=timestep_batch,
                    case_params=case_params,
                    mask=mask
                )

                # Predictor step: estimate x at t+dt
                x_next = x_t + dt * v1

                # Second velocity prediction (at t+dt)
                t_next = min(t + dt, 1.0)
                # Create continuous timestep values for next step
                timestep_next = torch.full(
                    (batch_size,),
                    t_next * self.num_flow_steps,
                    device=device,
                    dtype=torch.float32
                )
                unet_input_next = torch.cat([x_next, inputs, inputs_prev], dim=1)
                v2 = self.unet(
                    x=unet_input_next,
                    timesteps=timestep_next,
                    case_params=case_params,
                    mask=mask
                )

                # Corrector step: average of two velocity estimates
                x_t = x_t + dt * (v1 + v2) / 2

        else:
            raise ValueError(f"Unknown solver: {solver}. Choose 'euler' or 'heun'.")

        # --- 3. Denormalize the Final Residual ---
        # x_t is now the predicted normalized residual at t=1
        pred_residual = (x_t * self.residual_std) + self.residual_mean

        # --- 4. Calculate the Next Frame ---
        # X_t = X_{t-1} + (X_t - X_{t-1})
        next_frame = inputs + pred_residual

        # --- 5. Apply Mask ---
        if mask is not None:
            next_frame = next_frame * mask

        return next_frame

    def generate_many(
        self,
        inputs: Tensor,         # Initial frame X_t     [B, in_chan, H, W]
        inputs_prev: Tensor,    # Initial frame X_{t-1} [B, in_chan, H, W]
        case_params: Tensor,    # Case parameters       [B, n_params]
        mask: Tensor,           # Mask                  [B, 1, H, W]
        steps: int,             # Number of steps to generate
        num_inference_steps: int = 50,  # ODE steps per frame
        solver: str = "euler",          # ODE solver
    ) -> List[Tensor]:
        """
        Autoregressively generates a sequence of frames using Flow matching.

        Args:
            inputs: Starting frame X_t
            inputs_prev: Previous frame X_{t-1}
            case_params: Physical parameters
            mask: Spatial mask
            steps: Number of frames to generate
            num_inference_steps: ODE integration steps per frame
            solver: ODE solver to use

        Returns:
            List of generated frames
        """
        if inputs.dim() != 4 or inputs_prev.dim() != 4:
            raise ValueError("generate_many expects 4D input tensors (B, C, H, W)")

        generated_frames = []
        current_frame = inputs      # X_t for the first prediction
        prev_frame = inputs_prev    # X_{t-1} for the first prediction

        print(f"Generating {steps} frames autoregressively (FlowCast)...")
        for step_idx in tqdm(range(steps), desc="Autoregressive Generation"):
            # Generate X_{t+1} using (X_t, X_{t-1})
            next_frame = self.generate(
                inputs=current_frame,
                inputs_prev=prev_frame,
                case_params=case_params,
                mask=mask,
                num_inference_steps=num_inference_steps,
                solver=solver
            )

            generated_frames.append(next_frame)

            # Update state: shift frames for next iteration
            # X_{t-1} ← X_t, X_t ← X_{t+1}
            prev_frame = current_frame
            current_frame = next_frame

        return generated_frames
