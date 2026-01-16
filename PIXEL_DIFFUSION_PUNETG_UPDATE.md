# Pixel Diffusion Model Update: Replacing HuggingFace UNet with PUNetG

## Summary

Successfully replaced the HuggingFace `UNet2DConditionModel` with the custom `PUNetGCFD` architecture in the pixel diffusion model. This provides better control over the model architecture and removes the dependency on the HuggingFace diffusers UNet while maintaining full functionality.

## Changes Made

### 1. **src/models/pixel_diffusion.py**

#### Key Changes:
- **Import**: Replaced `UNet2DConditionModel` import with `PUNetGCFD` from `punetg.py`
- **Model Architecture**: Switched from cross-attention based UNet to PUNetG's embedding-based conditioning
- **New Parameters**: Added configurable architecture parameters:
  - `base_channels`: Base channel count (default: 64)
  - `channel_mults`: Channel multipliers per stage (default: (1, 2, 4))
  - `num_res_blocks`: ResNet blocks per stage (default: 2)
  - `dropout`: Dropout rate (default: 0.1)

#### Architecture Differences:

**Before (HuggingFace UNet):**
- Used cross-attention to condition on inputs + case_params
- Required reshaping conditioning signal to [Batch, Sequence, Features]
- Called via `.sample` attribute on output

**After (PUNetG):**
- Uses direct timestep + case_params embeddings (no cross-attention)
- Mask is concatenated with input internally
- Returns tensor directly (no `.sample` attribute)
- Simpler interface: `unet(x, timesteps, case_params, mask)`

#### Forward Method Updates:
- Removed conditioning signal preparation (cross-attention format)
- Added gradient checkpointing support via `torch.utils.checkpoint.checkpoint`
- Simplified UNet call to use PUNetG's native interface

#### Generate Method Updates:
- Removed cross-attention conditioning signal preparation
- Added timestep batch creation for proper PUNetG interface
- Simplified denoising loop

### 2. **src/args.py**

Added new pixel diffusion specific arguments (lines 296-307):

```python
# --- Pixel Diffusion (PUNetG) ---
pixel_diffusion_base_channels: int = 64
"""Base channel count for Pixel Diffusion PUNetG model"""

pixel_diffusion_channel_mults: tuple = (1, 2, 4)
"""Channel multipliers for PUNetG stages"""

pixel_diffusion_num_res_blocks: int = 2
"""Number of residual blocks per PUNetG stage"""

pixel_diffusion_dropout: float = 0.1
"""Dropout rate for PUNetG ResNet blocks"""
```

### 3. **src/utils/autoregressive.py**

Updated `pixel_diffusion` model initialization to include new PUNetG parameters:

```python
elif args.model == "pixel_diffusion":
    model = PixelDiffusionCfdModel(
        in_chan=args.in_chan,
        out_chan=args.out_chan,
        loss_fn=loss_fn,
        n_case_params=n_case_params,
        image_size=64,
        noise_scheduler_timesteps=args.ldm_noise_scheduler_timesteps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        base_channels=args.pixel_diffusion_base_channels,
        channel_mults=args.pixel_diffusion_channel_mults,
        num_res_blocks=args.pixel_diffusion_num_res_blocks,
        dropout=args.pixel_diffusion_dropout,
    )
```

## Benefits

1. **No HuggingFace Dependency**: Removes dependency on `UNet2DConditionModel` from diffusers
2. **Custom Architecture**: Full control over model architecture for CFD-specific optimizations
3. **Simpler Interface**: More straightforward conditioning mechanism (embeddings vs cross-attention)
4. **Configurable**: All architecture hyperparameters are exposed and configurable
5. **Gradient Checkpointing**: Implemented via PyTorch's native checkpoint mechanism
6. **Better Integration**: Aligns with other custom models in the codebase

## PUNetG Architecture

The PUNetG model features:

- **RMSNorm**: Root mean square normalization for stability
- **ResNet Blocks**: With timestep embedding injection
- **U-Net Structure**: Encoder-decoder with skip connections
- **Dual Conditioning**: Separate embeddings for timestep (σ) and case parameters (y)
- **No Attention**: Simpler than cross-attention, faster and more memory efficient
- **Mask Integration**: Mask concatenated with input automatically

## Usage

### Training with default settings:
```bash
cd src
python train_auto_v2.py --model pixel_diffusion --data cavity_geo
```

### Training with custom PUNetG architecture:
```bash
cd src
python train_auto_v2.py \
  --model pixel_diffusion \
  --data cylinder_geo \
  --pixel_diffusion_base_channels 128 \
  --pixel_diffusion_channel_mults "(1, 2, 4, 8)" \
  --pixel_diffusion_num_res_blocks 3 \
  --pixel_diffusion_dropout 0.15 \
  --use_gradient_checkpointing=True
```

### Memory-optimized training:
```bash
cd src
python train_auto_v2.py \
  --model pixel_diffusion \
  --data dam_prop_geo \
  --pixel_diffusion_base_channels 32 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_gradient_checkpointing=True \
  --use_mixed_precision=True
```

## Bug Fixes

### Loss Dictionary Compatibility (Issue #1)

**Problem**: The evaluation code in `train_auto_v2.py` expects all metrics promised by `loss_fn.get_score_names()` (mse, rmse, mae, nmse), but the original pixel_diffusion only returned mse and nmse.

**Solution**: Changed the forward method to use `self.loss_fn()` to compute all metrics:

```python
# Before:
loss = F.mse_loss(noise_pred, noise)
return {
    "preds": noise_pred,
    "loss": {"mse": loss, "nmse": loss / (torch.square(noise).mean() + 1e-8)}
}

# After:
loss_dict = self.loss_fn(noise_pred, noise)
return {
    "preds": noise_pred,
    "loss": loss_dict
}
```

This ensures the loss dictionary contains all required metrics (mse, rmse, mae, and optionally nmse) for proper evaluation.

### Hardcoded Loss Key in train_auto_v2.py (Issue #2)

**Problem**: The training script `train_auto_v2.py` hardcoded `outputs["loss"]["nmse"]` at lines 300 and 305, and `dev_scores["mean"]["nmse"]` at line 368. This caused a KeyError when using `--loss_name mse` (default) because the MseLoss with `normalize=False` doesn't return 'nmse'.

**Solution**: Changed hardcoded loss keys to use the configured loss name from args:

```python
# Before (lines 300, 305):
loss = outputs["loss"]["nmse"]

# After:
loss = outputs["loss"][args.loss_name]

# Before (line 368):
dev_loss = dev_scores["mean"]["nmse"]

# After:
dev_loss = dev_scores["mean"][args.loss_name]
```

This allows the training script to work with any loss function (mse, nmse, mae, etc.) specified via `--loss_name`.

## Validation

All modified files passed Python syntax validation:
- ✓ `src/models/pixel_diffusion.py`
- ✓ `src/models/punetg.py`
- ✓ `src/utils/autoregressive.py`
- ✓ `src/args.py`
- ✓ `src/train_auto_v2.py`

Successfully fixed:
- ✓ Evaluation KeyError for 'rmse' metric
- ✓ Training KeyError for 'nmse' metric with default loss_name='mse'

## Backward Compatibility

The changes maintain backward compatibility:
- Existing training scripts work without modification
- Default parameters match reasonable settings for CFD tasks
- The `generate()` and `generate_many()` methods maintain the same interface

## Notes

- PUNetG expects mask to be 2D or 3D; it will add batch/channel dims as needed
- Gradient checkpointing is implemented via `torch.utils.checkpoint` when `use_gradient_checkpointing=True`
- The model still uses the same `DDPMScheduler` from diffusers for noise scheduling
- Loss computation remains unchanged (MSE on predicted vs actual noise)
