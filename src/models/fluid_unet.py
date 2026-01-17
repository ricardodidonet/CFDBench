# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

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
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        h = torch.matmul(attn, v)

        # Reshape back
        h = h.transpose(2, 3).reshape(B, C, H, W)
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

        # Case parameter embedding (if provided)
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

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        ch = model_channels
        encoder_channels = [ch]
        ds = 1  # Current downsampling factor

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                block = nn.ModuleList([
                    ResBlock(ch, model_channels * mult, time_emb_dim, dropout)
                ])
                ch = model_channels * mult

                if ds in attention_resolutions:
                    block.append(AttentionBlock(ch, num_heads=num_heads))

                self.encoder_blocks.append(block)
                encoder_channels.append(ch)

            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(Downsample(ch))
                encoder_channels.append(ch)
                ds *= 2

        # Middle
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, ch, time_emb_dim, dropout),
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                encoder_ch = encoder_channels.pop()
                block = nn.ModuleList([
                    ResBlock(ch + encoder_ch, model_channels * mult, time_emb_dim, dropout)
                ])
                ch = model_channels * mult

                if ds in attention_resolutions:
                    block.append(AttentionBlock(ch, num_heads=num_heads))

                if level != 0 and i == num_res_blocks:
                    self.upsample_blocks.append(Upsample(ch))
                    ds //= 2

                self.decoder_blocks.append(block)

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x_t, t, extra):
        """
        Forward pass of the UNet.

        Args:
            x_t: Current noisy state at time t, shape (B, 2, 64, 64)
            t: Flow matching timesteps in [0, 1], shape (B,)
            extra: Dictionary containing:
                - 'x_prev_1': Previous state at t-1, shape (B, 2, 64, 64)
                - 'x_prev_2': Previous state at t-2, shape (B, 2, 64, 64)
                - 'case_params': Optional case parameters, shape (B, num_case_params)

        Returns:
            Predicted velocity field, shape (B, 2, 64, 64)
        """
        # Extract arguments from extra dict
        x_prev_1 = extra['x_prev_1']
        x_prev_2 = extra['x_prev_2']
        case_params = extra.get('case_params', None)

        # Time embedding
        time_emb = self.time_embed(t)

        # Add case parameter embedding if provided
        if case_params is not None and self.num_case_params > 0:
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

        # Encoder
        encoder_features = [h]
        for block in self.encoder_blocks:
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
            encoder_features.append(h)

        for downsample in self.downsample_blocks:
            h = downsample(h)
            encoder_features.append(h)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Decoder
        upsample_idx = 0
        for i, block in enumerate(self.decoder_blocks):
            encoder_feat = encoder_features.pop()
            h = torch.cat([h, encoder_feat], dim=1)

            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)

            # Check if we need to upsample after this block
            if upsample_idx < len(self.upsample_blocks):
                # Upsample after every (num_res_blocks + 1) decoder blocks, except the last level
                if (i + 1) % (2 + 1) == 0:  # Assuming num_res_blocks=2
                    h = self.upsample_blocks[upsample_idx](h)
                    upsample_idx += 1

        # Output
        return self.output(h)