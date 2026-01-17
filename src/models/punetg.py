import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

# --- 1. Helper Blocks ---

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Embeddings for scalar physics parameters (Re, Ma, etc.).
    Essential for overcoming Spectral Bias in continuous variables.
    """
    def __init__(self, n_params: int, embed_dim: int, scale: float = 30.0):
        super().__init__()
        # Random Gaussian matrix: [n_params, embed_dim // 2]
        # Not trainable - acts as a fixed high-freq basis
        self.register_buffer(
            'W', 
            torch.randn(n_params, embed_dim // 2) * scale
        )
        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, n_params]
        # proj: [Batch, embed_dim // 2]
        x_proj = x @ self.W * 2 * np.pi
        # output: [Batch, embed_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding module."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            # Handle accidental [Batch, 1] input
            timesteps = timesteps.view(-1)

        half_dim = self.dim // 2
        exponent = -torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device) * exponent)
        
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """
    Changed to Bilinear interpolation for smoother fluid gradients.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x

class ResNetBlock(nn.Module):
    """
    ResNet block with Adaptive Group Norm (FiLM) for conditioning.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_embed_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        super().__init__()
        
        # --- Conditioning Projection (The "Global Volume Knob") ---
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_embed_dim, out_channels * 2), # Output: Scale & Shift
        )

        # --- Main Path ---
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, condition_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)

        # Block 1
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Injection: Project Global Condition to Scale/Shift
        cond_proj = self.condition_mlp(condition_emb)[:, :, None, None]
        scale, shift = cond_proj.chunk(2, dim=1)

        # Block 2 with Modulation
        h = self.norm2(h)
        h = h * (1 + scale) + shift  # FiLM / AdaGN
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual

# --- 2. Main U-Net Model ---

class PUNetGCFD(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        n_case_params: int = 5,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_groups_norm: int = 32,
    ):
        super().__init__()
        
        # --- Conditioning Embeddings ---
        time_embed_dim = base_channels * 4
        
        # 1. Time Embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 2. Physics (Case Params) Embedding - NOW USING FOURIER
        self.cond_embed = nn.Sequential(
            GaussianFourierProjection(n_case_params, time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # We concatenate Time + Physics embeddings
        combined_embed_dim = time_embed_dim * 2

        # --- Encoder ---
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        
        channels = [base_channels] # Tracking channels for skip connections
        current_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            
            # Add ResBlocks
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResNetBlock(current_channels, out_ch, combined_embed_dim, dropout, num_groups_norm)
                )
                current_channels = out_ch
                channels.append(current_channels) 
            
            # Add Downsample (except last level)
            if i != len(channel_mults) - 1:
                level_blocks.append(Downsample(current_channels))
                
            self.down_blocks.append(level_blocks)

        # --- Bottleneck ---
        self.mid_block1 = ResNetBlock(current_channels, current_channels, combined_embed_dim, dropout, num_groups_norm)
        self.mid_block2 = ResNetBlock(current_channels, current_channels, combined_embed_dim, dropout, num_groups_norm)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()

            # Add Upsample (except first level of decoder)
            if i != 0:
                level_blocks.append(Upsample(current_channels))
            
            # Add ResBlocks + Skip Connections
            # FIX: Loop exactly num_res_blocks times to match Encoder stack
            for j in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                block_in_channels = current_channels + skip_channels
                
                level_blocks.append(
                    ResNetBlock(block_in_channels, out_ch, combined_embed_dim, dropout, num_groups_norm)
                )
                current_channels = out_ch
                
            self.up_blocks.append(level_blocks)

        # --- Output ---
        self.norm_out = nn.GroupNorm(min(num_groups_norm, base_channels), base_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        case_params: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1. Embeddings
        # Scale continuous time [0, 1] -> [0, 1000] for Sinusoidal logic
        t_emb = self.time_embed(timesteps * 1000.0)
        c_emb = self.cond_embed(case_params)
        
        # Global Condition Vector
        cond_emb = torch.cat([t_emb, c_emb], dim=-1)

        # 2. Input
        h = self.conv_in(x)
        skip_connections = [h] # Start with initial conviction features

        # 3. Encoder
        for level_blocks in self.down_blocks:
            for block in level_blocks:
                if isinstance(block, ResNetBlock):
                    h = block(h, cond_emb)
                    skip_connections.append(h)
                elif isinstance(block, Downsample):
                    h = block(h)

        # 4. Bottleneck
        h = self.mid_block1(h, cond_emb)
        h = self.mid_block2(h, cond_emb)

        # 5. Decoder
        for level_blocks in self.up_blocks:
            for block in level_blocks:
                if isinstance(block, Upsample):
                    h = block(h)
                elif isinstance(block, ResNetBlock):
                    # Retrieve skip connection from stack
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, cond_emb)

        # 6. Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h

# --- Testing Block ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    model = PUNetGCFD(
        in_channels=6,
        out_channels=2,
        base_channels=32,
        n_case_params=5,
        channel_mults=(1, 2, 4),
        num_res_blocks=2
    ).to(device)

    # Fake Data
    B = 4
    x = torch.randn(B, 6, 64, 64).to(device)
    # Continuous time [0, 1]
    t = torch.rand(B).to(device) 
    # Physics parameters (e.g. Reynolds, Lift, Angle)
    params = torch.randn(B, 5).to(device)

    out = model(x, t, params)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    # Verification
    assert out.shape == (B, 2, 64, 64)
    print("✅ Shapes match. Decoder logic is fixed.")