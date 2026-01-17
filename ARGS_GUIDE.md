# Quick Reference: Finding Parameters in args.py

The `args.py` file has been reorganized into **6 clear sections** for easy navigation.

## Section Overview

| Section | Lines | What's Inside | When to Use |
|---------|-------|---------------|-------------|
| **1. General Settings** | 18-29 | mode, seed, output_dir | Every run |
| **2. Training Configuration** | 32-75 | lr, epochs, batch_size, loss, logging | Tuning training |
| **3. Dataset Configuration** | 78-106 | data_name, data_dir, grid size, time step | Changing datasets |
| **4. Model Selection** | 109-130 | model, in_chan, out_chan | Choosing model |
| **5. Model-Specific Hyperparameters** | 133-294 | All model hyperparameters | Tuning specific model |
| **6. Advanced Training** | 297-317 | Mixed precision, gradient options | Optimizing memory/speed |

## Quick Lookups

### Most Common Parameters

```bash
# Training basics (Section 2)
--lr 1e-4
--num_epochs 10
--batch_size 8
--loss_name mse

# Dataset (Section 3)
--data_name cylinder_geo
--num_rows 64
--num_cols 64

# Model (Section 4)
--model pixel_diffusion
--in_chan 2
--out_chan 2

# Memory optimization (Section 6)
--use_mixed_precision=True
--gradient_accumulation_steps 4
--use_gradient_checkpointing=True
```

### Finding Model Hyperparameters

All model-specific parameters are in **Section 5**, organized by model:

- **FFN**: Lines 137-142 (`ffn_depth`, `ffn_width`)
- **Auto-FFN**: Lines 144-149 (`autoffn_depth`, `autoffn_width`)
- **DeepONet**: Lines 151-168 (`deeponet_width`, `branch_depth`, `trunk_depth`, ...)
- **Auto-EDeepONet**: Lines 170-178 (`autoedeeponet_width`, `autoedeeponet_depth`, ...)
- **FNO**: Lines 180-191 (`fno_depth`, `fno_hidden_dim`, `fno_modes_x`, ...)
- **U-Net**: Lines 193-198 (`unet_dim`, `unet_insert_case_params_at`)
- **ResNet**: Lines 200-211 (`resnet_depth`, `resnet_hidden_chan`, ...)
- **VAE**: Lines 213-265 (`vae_kl_weight`, `vae_kl_annealing_epochs`, ...)
- **Latent Diffusion**: Lines 267-294 (`ldm_vae_weights_path`, `ldm_latent_dim`, ...)

### Search Tips

**By Task:**
- Need to change learning rate? → Section 2, line 37
- Getting OOM errors? → Section 6, lines 297-317
- Want different dataset? → Section 3, lines 82-91
- Tuning FNO model? → Section 5, lines 180-191

**By Keyword:**
```bash
# Use Ctrl+F or grep to search for keywords:
grep -n "batch_size" args.py     # Line 46
grep -n "mixed_precision" args.py # Line 301
grep -n "fno_" args.py            # Lines 181-190
```

## Key Improvements

1. **Clear section headers** with `===` separators
2. **Logical grouping** - related parameters together
3. **Better documentation** - every parameter explained
4. **Progressive disclosure** - common params first, advanced last
5. **Consistent naming** - model-specific params prefixed by model name

## Examples

### Example 1: Quick training run
```bash
python train_auto_v2.py \
  --model fno \
  --data_name cavity_geo \
  --batch_size 16 \
  --num_epochs 50
```

### Example 2: Memory-optimized training
```bash
python train_auto_v2.py \
  --model pixel_diffusion \
  --data_name cylinder_geo \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --use_mixed_precision=True \
  --use_gradient_checkpointing=True
```

### Example 3: Hyperparameter tuning for FNO
```bash
python train_auto_v2.py \
  --model fno \
  --data_name dam_prop_geo \
  --fno_depth 6 \
  --fno_hidden_dim 64 \
  --fno_modes_x 16 \
  --fno_modes_y 16 \
  --lr 5e-5
```
