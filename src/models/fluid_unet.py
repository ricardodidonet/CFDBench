
"""
UNet model for fluid dynamics with autoregressive conditioning.
"""

import math
import torch
import torch.nn as nn
import numpy as np


def fourier_embedding(x, num_frequencies=16, max_freq_log2=8):
    """
    Create Fourier features for continuous parameters.

    Args:
        x: Input tensor of shape (B, D) where D is the number of parameters
        num_frequencies: Number of frequency bands
        max_freq_log2: Log2 of maximum frequency

    Returns:
        Fourier features of shape (B, D * num_frequencies * 2)
    """
    frequencies = 2.0 ** torch.linspace(
        0, max_freq_log2, num_frequencies, device=x.device, dtype=x.dtype
    )
    # x: (B, D), frequencies: (F,)
    # Create (B, D, F)
    angular_speeds = 2.0 * math.pi * frequencies[None, None, :] * x[..., None]
    # Concatenate sin and cos: (B, D, F, 2) -> (B, D * F * 2)
    features = torch.cat([torch.sin(angular_speeds), torch.cos(angular_speeds)], dim=-1)
    return features.reshape(x.shape[0], -1)


class TimestepEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,) tensor of timesteps in [0, 1]

        Returns:
            (B, dim) tensor of embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_emb_dim)
        """
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))

        # Add time embedding
        time_emb_out = self.time_emb_proj(time_emb)[:, :, None, None]
        h = h + time_emb_out

        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)

        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj_out(h)

        return x + h


class Downsample(nn.Module):
    """Downsampling layer."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class FluidDynamicsUNet(nn.Module):
    """
    UNet for fluid dynamics with:
    - Autoregressive conditioning on previous timesteps (t-1, t-2)
    - Conditioning on case parameters (geometry parameters)
    - Timestep embedding for flow matching

    Input shape: (B, 2, 64, 64) - 2 channels for velocity components (u, v)
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        model_channels=128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attention_resolutions=(2,),
        dropout=0.1,
        num_case_params=0,
        use_fourier_conditioning=True,
        num_fourier_freqs=16,
        num_heads=4,
    ):
        """
        Args:
            in_channels: Number of input channels (2 for u,v velocity)
            out_channels: Number of output channels (2 for u,v velocity)
            model_channels: Base channel count
            channel_mult: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions at which to use attention
            dropout: Dropout probability
            num_case_params: Number of case parameters (e.g., Reynolds number, geometry params)
            use_fourier_conditioning: Whether to use Fourier features for case parameters
            num_fourier_freqs: Number of Fourier frequencies for case parameters
            num_heads: Number of attention heads
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_case_params = num_case_params
        self.use_fourier_conditioning = use_fourier_conditioning
        self.num_fourier_freqs = num_fourier_freqs

        # Timestep embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Case parameter embedding
        if num_case_params > 0:
            if use_fourier_conditioning:
                fourier_dim = num_case_params * num_fourier_freqs * 2
                self.case_embed = nn.Sequential(
                    nn.Linear(fourier_dim, time_emb_dim),
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                )
            else:
                self.case_embed = nn.Sequential(
                    nn.Linear(num_case_params, time_emb_dim),
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                )

        # Input projection: current noisy state + 2 previous states (t-1, t-2)
        # Total input: 2 (current) + 2 (t-1) + 2 (t-2) = 6 channels
        input_channels = in_channels * 3
        self.input_proj = nn.Conv2d(input_channels, model_channels, 3, padding=1)

        # Build encoder levels
        self.encoder_levels = nn.ModuleList()
        ch = model_channels
        self.encoder_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            level_blocks = nn.ModuleList()
            
            # Add ResBlocks for this level
            for _ in range(num_res_blocks):
                block = nn.ModuleList([
                    ResBlock(ch, model_channels * mult, time_emb_dim, dropout)
                ])
                ch = model_channels * mult
                
                # Add attention if needed
                if level in attention_resolutions:
                    block.append(AttentionBlock(ch, num_heads=num_heads))
                
                level_blocks.append(block)
                self.encoder_channels.append(ch)
            
            # Add downsample if not last level
            if level != len(channel_mult) - 1:
                level_blocks.append(nn.ModuleList([Downsample(ch)]))
                self.encoder_channels.append(ch)
            
            self.encoder_levels.append(level_blocks)

        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, ch, time_emb_dim, dropout),
        ])

        # Build decoder levels
        self.decoder_levels = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            level_blocks = nn.ModuleList()
            
            for i in range(num_res_blocks + 1):
                encoder_ch = self.encoder_channels.pop()
                block = nn.ModuleList([
                    ResBlock(ch + encoder_ch, model_channels * mult, time_emb_dim, dropout)
                ])
                ch = model_channels * mult
                
                # Add attention if needed
                if level in attention_resolutions:
                    block.append(AttentionBlock(ch, num_heads=num_heads))
                
                level_blocks.append(block)
            
            # Add upsample if not first level
            if level != 0:
                level_blocks.append(nn.ModuleList([Upsample(ch)]))
            
            self.decoder_levels.append(level_blocks)

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x_t, t, extra):
        """
        Args:
            x_t: Current noisy state (B, 2, H, W)
            t: Timesteps (B,) in [0, 1]
            extra: Dict with:
                - 'x_prev_1': Previous state at t-1 (B, 2, H, W)
                - 'x_prev_2': Previous state at t-2 (B, 2, H, W)
                - 'case_params': Case parameters (B, D) or None
        """
        x_prev_1 = extra['x_prev_1']
        x_prev_2 = extra['x_prev_2']
        case_params = extra.get('case_params', None)

        # Timestep embedding
        time_emb = self.time_embed(t)

        # Case parameter embedding
        if self.num_case_params > 0 and case_params is not None:
            if self.use_fourier_conditioning:
                case_features = fourier_embedding(
                    case_params,
                    num_frequencies=self.num_fourier_freqs
                )
            else:
                case_features = case_params

            case_emb = self.case_embed(case_features)
            time_emb = time_emb + case_emb

        # Concatenate current state with previous states
        x = torch.cat([x_t, x_prev_1, x_prev_2], dim=1)

        # Input projection
        h = self.input_proj(x)

        # Encoder - save features in forward order
        encoder_features = [h]  # Include initial feature!
        
        for level_blocks in self.encoder_levels:
            for block in level_blocks:
                # Check if this is a downsample block
                if len(block) == 1 and isinstance(block[0], Downsample):
                    h = block[0](h)
                    encoder_features.append(h)
                else:
                    # Process ResBlock and optional Attention
                    for layer in block:
                        if isinstance(layer, ResBlock):
                            h = layer(h, time_emb)
                        else:
                            h = layer(h)
                    encoder_features.append(h)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Decoder - pop in reverse order
        for level_blocks in self.decoder_levels:
            for block in level_blocks:
                # Check if this is an upsample block
                if len(block) == 1 and isinstance(block[0], Upsample):
                    h = block[0](h)
                else:
                    # Concatenate with skip connection BEFORE processing
                    encoder_feat = encoder_features.pop()
                    h = torch.cat([h, encoder_feat], dim=1)
                    
                    # Process ResBlock and optional Attention
                    for layer in block:
                        if isinstance(layer, ResBlock):
                            h = layer(h, time_emb)
                        else:
                            h = layer(h)

        # Output
        return self.output(h)
