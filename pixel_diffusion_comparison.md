# Pixel Diffusion: HuggingFace UNet vs PUNetG Comparison

## Architecture Comparison

### HuggingFace UNet (Before)
```
Input: noisy_images [B, 2, 64, 64]
Timestep: t [scalar]
Conditioning: cross-attention
├─ inputs [B, 2, 64, 64] + case_params [B, 5]
├─ Reshape to [B, 2+5, 64*64]
└─ Permute to [B, 4096, 7] for cross-attention

UNet2DConditionModel:
├─ Cross-attention conditioning
├─ Multiple attention layers
├─ Fixed architecture (predefined blocks)
└─ Output: .sample attribute

Output: noise_pred [B, 2, 64, 64]
```

### PUNetG (After)
```
Input: noisy_images [B, 2, 64, 64]
Timestep: t [B] (batched)
Case params: [B, 5]
Mask: [B, 1, 64, 64] (optional)

PUNetGCFD:
├─ Timestep embedding (sinusoidal)
├─ Case params embedding (MLP)
├─ Concat embeddings → combined conditioning
├─ Mask concat with input → [B, 3, 64, 64]
├─ ResNet blocks with conditioning injection
└─ Direct output (no .sample)

Output: noise_pred [B, 2, 64, 64]
```

## Code Comparison

### Forward Pass

#### Before (HuggingFace):
```python
# Prepare cross-attention conditioning
case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(
    -1, -1, inputs.shape[2], inputs.shape[3]
)
conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
conditioning_signal = conditioning_signal.view(
    batch_size, self.in_chan + self.n_case_params, -1
)
conditioning_signal = conditioning_signal.permute(0, 2, 1)

# Predict noise
noise_pred = self.unet(
    sample=noisy_images,
    timestep=timesteps,
    encoder_hidden_states=conditioning_signal
).sample  # Note: .sample attribute
```

#### After (PUNetG):
```python
# Predict noise directly
if self.use_gradient_checkpointing and self.training:
    noise_pred = torch.utils.checkpoint.checkpoint(
        self.unet,
        noisy_images,
        timesteps,
        case_params,
        mask,
        use_reentrant=False
    )
else:
    noise_pred = self.unet(
        x=noisy_images,
        timesteps=timesteps,
        case_params=case_params,
        mask=mask
    )  # Direct output
```

### Generation

#### Before (HuggingFace):
```python
# Prepare cross-attention conditioning
case_params_expanded = case_params.unsqueeze(-1).unsqueeze(-1).expand(
    -1, -1, inputs.shape[2], inputs.shape[3]
)
conditioning_signal = torch.cat([inputs, case_params_expanded], dim=1)
conditioning_signal = conditioning_signal.view(
    batch_size, self.in_chan + self.n_case_params, -1
)
conditioning_signal = conditioning_signal.permute(0, 2, 1)

for t in self.noise_scheduler.timesteps:
    noise_pred = self.unet(
        sample=images,
        timestep=t,
        encoder_hidden_states=conditioning_signal
    ).sample
    images = self.noise_scheduler.step(noise_pred, t, images).prev_sample
```

#### After (PUNetG):
```python
for t in self.noise_scheduler.timesteps:
    timestep_batch = torch.full(
        (batch_size,), t, device=inputs.device, dtype=torch.long
    )

    noise_pred = self.unet(
        x=images,
        timesteps=timestep_batch,
        case_params=case_params,
        mask=mask
    )

    images = self.noise_scheduler.step(noise_pred, t, images).prev_sample
```

## Parameter Count Comparison

### HuggingFace UNet
```python
UNet2DConditionModel(
    sample_size=64,
    in_channels=2,
    out_channels=2,
    cross_attention_dim=7,  # 2 (input) + 5 (case_params)
    block_out_channels=(32, 64, 128, 256),
)
```
Estimated parameters: ~10-20M (includes attention layers)

### PUNetG
```python
PUNetGCFD(
    in_channels=2,
    out_channels=2,
    base_channels=64,
    n_case_params=5,
    channel_mults=(1, 2, 4),
    num_res_blocks=2,
)
```
Estimated parameters: ~5-8M (no attention layers)

## Performance Implications

| Aspect | HuggingFace UNet | PUNetG |
|--------|------------------|---------|
| **Parameters** | More (~10-20M) | Fewer (~5-8M) |
| **Memory** | Higher (attention) | Lower (no attention) |
| **Speed** | Slower (attention) | Faster (no attention) |
| **Configurability** | Limited | Fully configurable |
| **Dependencies** | Requires diffusers | Native PyTorch |
| **Complexity** | Cross-attention | Embedding injection |

## Key Advantages of PUNetG

1. **Simpler Conditioning**: Direct embedding injection vs cross-attention preparation
2. **Less Memory**: No attention mechanisms reduce memory footprint
3. **Faster Training**: Fewer operations per forward pass
4. **More Control**: All architecture details exposed and configurable
5. **Better Integration**: Matches design patterns of other models in codebase
6. **Cleaner Code**: No complex conditioning signal reshaping

## Migration Guide

### Old way (HuggingFace):
```python
from diffusers import UNet2DConditionModel

model = PixelDiffusionCfdModel(
    in_chan=2,
    out_chan=2,
    loss_fn=loss_fn,
    n_case_params=5,
)
```

### New way (PUNetG):
```python
from models.punetg import PUNetGCFD

model = PixelDiffusionCfdModel(
    in_chan=2,
    out_chan=2,
    loss_fn=loss_fn,
    n_case_params=5,
    base_channels=64,        # NEW: configurable
    channel_mults=(1, 2, 4), # NEW: configurable
    num_res_blocks=2,        # NEW: configurable
    dropout=0.1,             # NEW: configurable
)
```

### Training command stays the same:
```bash
python train_auto_v2.py --model pixel_diffusion --data cavity_geo
```

### But now you can customize architecture:
```bash
python train_auto_v2.py \
  --model pixel_diffusion \
  --data cavity_geo \
  --pixel_diffusion_base_channels 128 \
  --pixel_diffusion_num_res_blocks 3
```
