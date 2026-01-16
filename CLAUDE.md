# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CFDBench is a large-scale benchmark for evaluating machine learning methods in computational fluid dynamics (CFD). The codebase supports training and evaluating both autoregressive and non-autoregressive neural network models on four classic CFD problems: cavity flow, tube flow, dam flow, and cylinder flow.

## System Requirements

Tested on:
- PyTorch 1.13.3+cu117
- Python 3.9.0
- CUDA GPU required

Note: Reduce `--batch_size` if encountering VRAM limitations.

## Project Structure

```
src/
├── models/           # Neural network model implementations
│   ├── base_model.py        # Base classes: CfdModel, AutoCfdModel
│   ├── deeponet.py          # Non-autoregressive DeepONet
│   ├── auto_deeponet.py     # Autoregressive DeepONet
│   ├── auto_edeeponet.py    # Autoregressive Enhanced DeepONet
│   ├── auto_ffn.py          # Autoregressive FFN
│   ├── ffn.py               # Non-autoregressive FFN
│   ├── resnet.py            # Autoregressive ResNet
│   ├── unet.py              # Autoregressive U-Net
│   ├── fno/fno2d.py         # Fourier Neural Operator
│   ├── cfd_vae.py           # Variational Autoencoder
│   ├── latent_diffusion.py  # Latent diffusion model
│   ├── ldm2.py              # Latent diffusion model v2
│   ├── pixel_diffusion.py   # Pixel-space diffusion model (uses PUNetG)
│   ├── gen_cast_cfd.py      # GenCast-inspired residual diffusion model
│   └── punetg.py            # PUNetG U-Net architecture (GroupNorm-based)
├── dataset/          # Dataset loaders for each CFD problem
│   ├── base.py              # Base classes: CfdDataset, CfdAutoDataset
│   ├── cavity.py            # Lid-driven cavity flow
│   ├── tube.py              # Flow through circular tube
│   ├── dam.py               # Flow over a dam
│   └── cylinder.py          # Flow around a cylinder
├── utils/
│   ├── common.py            # Utilities: plotting, checkpointing, output dirs
│   ├── autoregressive.py    # Model initialization for autoregressive models
│   └── vae.py               # VAE-specific utilities
├── args.py           # Centralized argument parser using tap
├── train.py          # Training script for non-autoregressive models
├── train_auto.py     # Training script for autoregressive models
├── train_gencast.py  # Training script for GenCast model (with mixed precision)
├── train_vae*.py     # VAE training scripts
├── train_ldm*.py     # Latent diffusion model training scripts
├── test_multistep.py # Multi-step inference evaluation
└── plot_losses.py    # Utility to plot training/validation loss curves
scripts/
├── analysis/         # Analysis utilities (VAE interpretation, cylinder location, etc.)
├── visualization/    # Plotting tools (multistep inference, mask overlay, results)
└── utils/            # Additional utilities
```

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training Script Selection

Choose the appropriate training script based on your model type:

- **train.py**: Non-autoregressive models (FFN, DeepONet)
- **train_auto.py**: Autoregressive models (standard training for ResNet, U-Net, FNO, etc.)
- **train_gencast.py**: GenCast residual diffusion model with mixed precision, gradient accumulation, and gradient checkpointing
- **train_vae*.py**: VAE training with various configurations (KL annealing, diffusion science)
- **train_ldm*.py**: Latent diffusion model training
- **train_diffusers.py**: Training with HuggingFace diffusers integration

### Training Non-Autoregressive Models
```bash
cd src
python train.py --model <model_name> --data <data_name>
```

Available non-autoregressive models: `ffn`, `deeponet`

### Training Autoregressive Models
```bash
cd src
python train_auto.py --model <model_name> --data <data_name>
```

Available autoregressive models: `auto_ffn`, `auto_deeponet`, `auto_edeeponet`, `auto_deeponet_cnn`, `resnet`, `unet`, `fno`, `pixel_diffusion`, `latent_diffusion`, `ldm2`

### Training GenCast Residual Diffusion Model

GenCast is a conditional diffusion model that predicts normalized residuals between frames using second-order conditioning:

```bash
cd src
python train_gencast.py --model gencast --data <data_name>
```

Key GenCast features:
- Predicts **normalized residuals** (X_t - X_{t-1}) instead of absolute values
- Uses **second-order conditioning** (X_{t-1} and X_{t-2}) for better temporal coherence
- Requires pre-calculated residual statistics (mean and std) - see scripts for calculation
- Built on PUNetG architecture with GroupNorm and FiLM-style conditioning
- Supports mixed precision, gradient checkpointing, and gradient accumulation

### Dataset Format

Dataset names follow the pattern: `<problem>_<subsets>`

Problems: `cavity`, `tube`, `dam`, `cylinder`

Subsets indicate varying parameters (can combine multiple):
- `bc` - boundary conditions
- `geo` - domain geometries
- `prop` - physical properties

Example: `cylinder_geo`, `cavity_prop_bc_geo`, `dam_prop_geo`

### Testing/Inference
```bash
cd src
python train.py --mode test --model <model_name> --data <data_name>
# or for autoregressive:
python train_auto.py --mode test --model <model_name> --data <data_name>
```

### Multi-Step Inference
```bash
cd src
python test_multistep.py --model <model_name> --data <data_name>
```

## Visualization and Analysis

### Loss Plotting

After training completes, visualize training progress:

```bash
cd src
python plot_losses.py --result_dir result/<model>_<data>

# Custom output location
python plot_losses.py --result_dir result/fno_cavity_geo --output my_loss_curve.png
```

This generates:
- `loss_curve.png` - Training and validation loss curves with best dev loss highlighted
- `loss_summary.txt` - Text summary of training progress

The script reads from `loss_history.json` which is automatically created by [train_gencast.py](src/train_gencast.py).

### Multi-Step Inference Visualization

```bash
cd scripts/visualization
python plot_multistep_inference.py  # Visualize multi-step predictions
python plot_mask_overlay.py         # Visualize masks overlaid on flow fields
python get_result.py                # Extract and summarize results
```

### Analysis Scripts

Located in `scripts/analysis/` for additional analysis tasks:
- VAE latent space interpretation
- Cylinder position analysis
- Flow field statistics

### Key Arguments

Common hyperparameters (see [args.py](src/args.py) for full list):
- `--lr` - Learning rate (default: 1e-4)
- `--num_epochs` - Training epochs (default: 10)
- `--batch_size` - Training batch size (default: 8)
- `--eval_batch_size` - Evaluation batch size (default: 2)
- `--delta_time` - Time step size for autoregressive models (default: 0.1)
- `--loss_name` - Loss function: `mse`, `nmse`, `mae`, `nmae` (default: `mse`)
- `--output_dir` - Results directory (default: `result`)
- `--use_mixed_precision` - Enable automatic mixed precision (default: True)
- `--gradient_accumulation_steps` - Gradient accumulation for larger effective batch sizes (default: 1)
- `--use_gradient_checkpointing` - Enable gradient checkpointing for diffusion models (default: True)

Model-specific hyperparameters are prefixed by model name (e.g., `--fno_depth`, `--unet_dim`, `--resnet_hidden_chan`).

## Architecture Overview

### Model Base Classes

All models inherit from one of two base classes in [models/base_model.py](src/models/base_model.py):

1. **CfdModel** (non-autoregressive): Maps conditions (physics properties, boundary conditions, geometry) directly to solution at a later time
   - Must implement: `forward()`, `generate_one()`

2. **AutoCfdModel** (autoregressive): Generates solutions one frame at a time
   - Must implement: `forward()`, `generate()`, `generate_many()`

### Dataset Base Classes

Datasets inherit from [dataset/base.py](src/dataset/base.py):

1. **CfdDataset**: For non-autoregressive models
2. **CfdAutoDataset**: For autoregressive models
   - Must populate `all_features` attribute for multi-step inference
   - Must implement `__getitem__()` returning `(input, label, mask)`

### GenCast Residual Diffusion Architecture

The GenCast model ([models/gen_cast_cfd.py](src/models/gen_cast_cfd.py)) introduces a novel approach:

1. **Residual Prediction**: Instead of predicting X_t directly, predicts normalized residuals (X_t - X_{t-1})
2. **Second-Order Conditioning**: Uses both X_{t-1} and X_{t-2} as spatial conditioning signals
3. **Normalization**: Residuals are normalized using pre-calculated dataset statistics (mean, std)
4. **PUNetG Backbone**: Uses the custom PUNetG U-Net with:
   - GroupNorm for stability
   - FiLM-style conditioning (scale and shift modulation)
   - Separate embeddings for timestep (σ) and case parameters (y)
   - No attention mechanisms (simpler and more memory efficient)
5. **Diffusion Process**: Standard DDPM with noise prediction objective

Input channels to PUNetG: `noisy_residual(2) + X_{t-1}(2) + X_{t-2}(2) = 6 channels`

### PUNetG Architecture Details

The PUNetG U-Net ([models/punetg.py](src/models/punetg.py)) features:

- **ResNetBlock**: GroupNorm + Conv + FiLM conditioning + Conv + Dropout + Residual
- **FiLM Conditioning**: Scale and shift modulation from combined timestep + case parameter embeddings
- **U-Net Structure**: Encoder with downsampling, bottleneck, decoder with upsampling and skip connections
- **No Attention**: Simpler than cross-attention, faster and more memory efficient
- **Configurable**: Channel multipliers, number of ResBlocks per level, dropout rate

Key difference from original PUNetG paper: Uses GroupNorm instead of LayerNorm for better stability with varying batch sizes.

### Adding New Models

1. Create a class inheriting from `CfdModel` or `AutoCfdModel` in `src/models/`
2. Implement required methods: `forward()`, `generate_one()`, and for autoregressive also `generate_many()`
3. Add model hyperparameters to [args.py](src/args.py)
4. Add model initialization in:
   - `init_model()` in [utils/autoregressive.py](src/utils/autoregressive.py) (for autoregressive)
   - `init_model()` in [train.py](src/train.py) (for non-autoregressive)
5. Add output directory path logic in `get_output_dir()` in [utils/common.py](src/utils/common.py)

### Adding New Datasets

1. Create a class inheriting from `CfdDataset` or `CfdAutoDataset` in `src/dataset/`
2. Implement `__getitem__()` and `__len__()`
3. For multi-step inference, populate `all_features` list with Tensors or NumPy arrays
4. Add dataset instantiation logic in `get_dataset()` or `get_auto_dataset()` in [dataset/__init__.py](src/dataset/__init__.py)

### Data Format

Data is stored as NumPy arrays in the following structure:
```
data/
├── cavity/
│   ├── bc/
│   │   ├── case0000/
│   │   │   ├── u.npy
│   │   │   ├── v.npy
│   │   │   └── mask.npy
│   │   └── case0001/
│   ├── geo/
│   └── prop/
├── tube/
├── dam/
└── cylinder/
```

Each case contains:
- `u.npy`: x-velocity component
- `v.npy`: y-velocity component
- `mask.npy`: Binary mask (1=interior, 0=boundary)

### Training Workflow

1. **Data loading**: Dataset factory functions in [dataset/__init__.py](src/dataset/__init__.py) create train/dev/test splits
2. **Model initialization**: Models created via `init_model()` with loss function and hyperparameters
3. **Training loop**: Standard PyTorch training with AdamW optimizer and ReduceLROnPlateau scheduler
4. **Checkpointing**: Models saved every `eval_interval` epochs to `output_dir/ckpt-<epoch>/`
5. **Evaluation**: Scores computed on dev set; best checkpoint selected by lowest dev loss
6. **Output structure**:
   - `output_dir/ckpt-<epoch>/model.pt` - Model weights
   - `output_dir/ckpt-<epoch>/scores.json` - Evaluation metrics
   - `output_dir/images/` - Visualization of predictions

### Mixed Precision and Memory Optimization

The GenCast training script ([train_gencast.py](src/train_gencast.py)) supports:
- Automatic mixed precision (AMP) via `--use_mixed_precision` (default: True)
- Gradient accumulation via `--gradient_accumulation_steps` (default: 1)
- Gradient checkpointing for diffusion models via `--use_gradient_checkpointing` (default: True)
- Proper handling of both flattened (DeepONet variants) and 2D (CNN-based) model outputs

For memory-constrained training:
```bash
cd src
python train_gencast.py \
  --model gencast \
  --data cylinder_geo \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --use_gradient_checkpointing=True \
  --use_mixed_precision=True
```

### Case Parameters

Models are conditioned on case parameters that vary by problem:
- **cavity/tube/dam**: 5 parameters (vel_in, density, viscosity, height, width)
- **cylinder**: 8 parameters (vel_in, density, viscosity, height, width, radius, center_x, center_y)

These are computed in `get_input_shapes()` in [utils/autoregressive.py](src/utils/autoregressive.py).

### Calculating Residual Statistics for GenCast

Before training GenCast, you need to calculate residual statistics from your dataset:

```bash
cd src/utils
python calculate_residuals_stat.py --data <data_name> --output residual_stats.pt
```

This computes the mean and standard deviation of residuals (X_t - X_{t-1}) across the training dataset, which are used to normalize residuals during training and denormalize predictions during inference.

The statistics are saved as a PyTorch tensor file containing:
- `residual_mean`: [2] tensor for u, v channels
- `residual_std`: [2] tensor for u, v channels

Pass these to the GenCast model via `--residual_stats_path` argument.

## Additional Documentation

For more detailed information, see:
- [ARGS_GUIDE.md](ARGS_GUIDE.md) - Quick reference for finding parameters in args.py organized by section
- [LOSS_LOGGING.md](LOSS_LOGGING.md) - Guide to loss tracking and plotting utilities
- [PIXEL_DIFFUSION_PUNETG_UPDATE.md](PIXEL_DIFFUSION_PUNETG_UPDATE.md) - Details on PUNetG architecture update
- [BUGFIX_SUMMARY.md](BUGFIX_SUMMARY.md) - Recent bug fixes and their solutions
- [README.md](README.md) - Project overview, paper links, data download, and citation information
