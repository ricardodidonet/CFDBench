# Loss Logging and Plotting Guide

## Changes Made to train_auto_v2.py

### 1. Removed Noisy Per-Batch Logging
**Before:**
```
[Epoch 1, Step 51] loss=0.198765, lr=1.00e-04
[Epoch 1, Step 101] loss=0.167890, lr=1.00e-04
[Epoch 1, Step 151] loss=0.145678, lr=1.00e-04
```

**After:**
- Clean output with only epoch summaries
- Progress bar still shows real-time loss

### 2. Added Epoch-Level Loss Tracking

The script now tracks:
- **Training loss per epoch** (average of all batches)
- **Dev loss per epoch** (when evaluated)

### 3. New Output File: `loss_history.json`

After training completes, you'll find:
```
result/
└── pixel_diffusion_cylinder_geo/
    ├── loss_history.json          ← NEW! For plotting
    ├── all_train_losses.json      ← All batch-level losses
    ├── best_model.pt
    ├── ckpt-4/
    │   ├── model.pt
    │   ├── scores.json
    │   ├── dev_scores.json
    │   └── train_losses.json
    └── ...
```

### 4. loss_history.json Format

```json
{
  "train_losses": [0.234, 0.198, 0.167, 0.145, ...],
  "dev_losses": [
    {"epoch": 4, "dev_loss": 0.156},
    {"epoch": 9, "dev_loss": 0.123},
    ...
  ],
  "epochs": [0, 1, 2, 3, 4, ...]
}
```

## How to Plot Losses

### Quick Usage

After training completes:

```bash
# Basic usage
python plot_losses.py --result_dir result/pixel_diffusion_cylinder_geo

# Custom output location
python plot_losses.py --result_dir result/fno_cavity_geo --output my_loss_curve.png
```

### What Gets Generated

1. **loss_curve.png** - Beautiful plot with:
   - Training loss (every epoch)
   - Validation loss (evaluated epochs)
   - Best dev loss highlighted
   - Grid and legend

2. **loss_summary.txt** - Text summary:
   ```
   Total epochs: 50
   Final train loss: 0.045678
   Final dev loss: 0.056789
   Best dev loss: 0.051234 (Epoch 42)
   Train loss improvement: 0.234567 → 0.045678
   Dev loss improvement: 0.156789 → 0.056789
   ```

### Example Plot

The generated plot will show:
- X-axis: Epoch number
- Y-axis: Loss (NMSE)
- Blue line with circles: Training loss (every epoch)
- Orange line with squares: Dev loss (every eval_interval epochs)
- Red dashed line: Best dev loss achieved
- Text box: Best dev loss value and epoch

## Console Output During Training

### Before (Noisy)
```
Epoch 1/10: 25%|██▌       | 50/200 [00:38<01:52, loss=0.198765, lr=1.00e-04]

[Epoch 1, Step 51] loss=0.198765, lr=1.00e-04     ← Removed

Epoch 1/10: 50%|█████     | 100/200 [01:16<01:16, loss=0.167890, lr=1.00e-04]

[Epoch 1, Step 101] loss=0.167890, lr=1.00e-04    ← Removed

Epoch 1/10: 75%|███████▌  | 150/200 [01:54<00:38, loss=0.156789, lr=1.00e-04]

[Epoch 1, Step 151] loss=0.156789, lr=1.00e-04    ← Removed
```

### After (Clean)
```
Epoch 1/10: 100%|██████████| 200/200 [02:34<00:00, loss=0.145678, lr=1.00e-04]

=== Epoch 1 Summary ===
  Train loss: 0.145678
  Epoch time: 154.3s

Epoch 2/10: 100%|██████████| 200/200 [02:31<00:00, loss=0.123456, lr=1.00e-04]

=== Epoch 2 Summary ===
  Train loss: 0.123456
  Epoch time: 151.2s

... (Epochs 3-4) ...

=== Epoch 5 Summary ===
  Train loss: 0.098765
  Epoch time: 148.9s

=== Evaluating ===
Evaluation: 100%|██████████| 50/50 [00:25<00:00]

=== Evaluation Results ===
  nmse    : pred=0.112345, input=0.234567
  
  Checkpoint saved to result/.../ckpt-4/model.pt
  *** New best model saved! Dev loss: 0.112345 ***
```

## Custom Plotting

If you want to customize the plot, you can use the loss_history.json directly:

```python
import json
import matplotlib.pyplot as plt

# Load data
with open("result/your_model/loss_history.json", "r") as f:
    data = json.load(f)

# Extract data
train_losses = data["train_losses"]
dev_epochs = [d["epoch"] for d in data["dev_losses"]]
dev_losses = [d["dev_loss"] for d in data["dev_losses"]]
epochs = data["epochs"]

# Your custom plotting code here
plt.plot(epochs, train_losses, label='Train')
plt.plot(dev_epochs, dev_losses, label='Dev')
plt.legend()
plt.savefig("custom_plot.png")
```

## Benefits

1. **Cleaner console output** - No spam every 50 batches
2. **Easy plotting** - One command to visualize training
3. **Progress tracking** - See convergence at a glance
4. **Debugging** - Spot overfitting, plateaus, or divergence
5. **Reporting** - Professional plots for papers/presentations

## Notes

- The progress bar still shows **real-time loss** during training
- Batch-level losses are still saved to `all_train_losses.json` if needed
- Dev loss is only recorded when evaluation runs (every `eval_interval` epochs)
- All existing checkpoint files are preserved
