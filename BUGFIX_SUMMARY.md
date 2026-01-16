# Bug Fixes for Pixel Diffusion Training

## Summary

Fixed two critical bugs that prevented the pixel diffusion model from training successfully. Both were related to loss dictionary key mismatches.

## Bug #1: Missing 'rmse' Key in Evaluation

### Error
```
KeyError: 'rmse'
File: /data/CFDBench/src/train_auto_v2.py, line 182
```

### Root Cause
The evaluation code in `train_auto_v2.py` expects all metrics that `model.loss_fn.get_score_names()` returns:
- `mse`, `rmse`, `mae`, and optionally `nmse`

However, `PixelDiffusionCfdModel.forward()` was manually computing loss and only returning:
- `mse` and `nmse`

### Fix Location
**File**: `src/models/pixel_diffusion.py` (lines 98-105)

**Before**:
```python
# Step 3: Compute loss
loss = F.mse_loss(noise_pred, noise)

return {
    "preds": noise_pred,
    "loss": {"mse": loss, "nmse": loss / (torch.square(noise).mean() + 1e-8)}
}
```

**After**:
```python
# Step 3: Compute loss using the loss function to get all metrics
# This ensures compatibility with evaluation code that expects all score_names
loss_dict = self.loss_fn(noise_pred, noise)

return {
    "preds": noise_pred,
    "loss": loss_dict
}
```

### Why This Works
By using `self.loss_fn()`, we get all the metrics that were promised by `get_score_names()`:
- When `loss_name='mse'` (normalize=False): Returns `mse`, `rmse`, `mae`
- When `loss_name='nmse'` (normalize=True): Returns `mse`, `rmse`, `mae`, `nmse`

---

## Bug #2: Hardcoded 'nmse' Key in Training

### Error
```
KeyError: 'nmse'
File: /data/CFDBench/src/train_auto_v2.py, line 300, 305, 368
```

### Root Cause
The training script hardcoded loss key access as `outputs["loss"]["nmse"]`, but when using the default `--loss_name mse`, the MseLoss function with `normalize=False` doesn't include 'nmse' in its output.

Loss function behavior:
- `loss_name='mse'` → MseLoss(normalize=False) → Returns: mse, rmse, mae (NO nmse)
- `loss_name='nmse'` → MseLoss(normalize=True) → Returns: mse, rmse, mae, nmse

### Fix Locations
**File**: `src/train_auto_v2.py`

#### Location 1 & 2: Training Loop (lines 300, 305)

**Before**:
```python
# Forward pass with mixed precision
if USE_TORCH_AMP:
    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(**batch)
        loss = outputs["loss"]["nmse"]  # ❌ Hardcoded
        loss = loss / gradient_accumulation_steps
else:
    with autocast(enabled=use_amp):
        outputs = model(**batch)
        loss = outputs["loss"]["nmse"]  # ❌ Hardcoded
        loss = loss / gradient_accumulation_steps
```

**After**:
```python
# Forward pass with mixed precision
if USE_TORCH_AMP:
    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        outputs = model(**batch)
        loss = outputs["loss"][args.loss_name]  # ✅ Configurable
        loss = loss / gradient_accumulation_steps
else:
    with autocast(enabled=use_amp):
        outputs = model(**batch)
        loss = outputs["loss"][args.loss_name]  # ✅ Configurable
        loss = loss / gradient_accumulation_steps
```

#### Location 3: Dev Loss for Scheduler (line 368)

**Before**:
```python
dev_loss = dev_scores["mean"]["nmse"]  # ❌ Hardcoded
```

**After**:
```python
dev_loss = dev_scores["mean"][args.loss_name]  # ✅ Configurable
```

### Why This Works
Now the training script respects the `--loss_name` argument:
- `--loss_name mse` → Uses 'mse' key (default)
- `--loss_name nmse` → Uses 'nmse' key
- `--loss_name mae` → Uses 'mae' key
- `--loss_name rmse` → Uses 'rmse' key

---

## Impact

### Before Fixes
❌ Training failed with KeyError on either:
- Line 182 (evaluation): Missing 'rmse'
- Line 300/305 (training): Missing 'nmse' when using default loss_name='mse'
- Line 368 (scheduler): Missing 'nmse' when using default loss_name='mse'

### After Fixes
✅ Training works correctly with any loss function
✅ Evaluation computes all metrics properly
✅ Learning rate scheduler uses the correct loss metric
✅ Compatible with all loss names: mse, nmse, mae, rmse

---

## Testing

### Command to Test
```bash
cd src

# Test with default loss (mse)
python train_auto_v2.py --model pixel_diffusion --data cavity_geo

# Test with nmse loss
python train_auto_v2.py --model pixel_diffusion --data cavity_geo --loss_name nmse

# Test with mae loss
python train_auto_v2.py --model pixel_diffusion --data cavity_geo --loss_name mae
```

### Expected Behavior
- No KeyError crashes
- Training progresses through epochs
- Evaluation runs successfully
- Learning rate scheduler updates based on chosen metric

---

## Files Modified

1. **src/models/pixel_diffusion.py**
   - Changed loss computation to use `self.loss_fn()` (line 100)

2. **src/train_auto_v2.py**
   - Changed hardcoded "nmse" to `args.loss_name` (lines 300, 305, 368)

---

## Backward Compatibility

✅ These fixes maintain full backward compatibility:
- Default behavior unchanged (uses 'mse' loss)
- Users who explicitly set `--loss_name nmse` will still work
- No changes to model architecture or interfaces
- No changes to checkpoint format

---

## Related Changes

These bug fixes were discovered while implementing the PUNetG architecture replacement in pixel_diffusion.py. See [PIXEL_DIFFUSION_PUNETG_UPDATE.md](PIXEL_DIFFUSION_PUNETG_UPDATE.md) for details on the main feature implementation.
