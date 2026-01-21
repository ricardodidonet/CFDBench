#!/usr/bin/env python3
"""
Diagnostic script to understand dataset size and training dynamics.

Usage:
    python diagnose_dataset.py
"""

import torch
from pathlib import Path
from dataset import get_auto_dataset
from dataset.wrapper import FlowCastWrapperDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load datasets
    data_dir = Path("../data")
    base_dataset_train, base_dataset_val, base_dataset_test = get_auto_dataset(
        data_dir=data_dir,
        data_name='cylinder_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['train', 'dev', 'test']
    )

    # Wrap datasets
    dataset_train = FlowCastWrapperDataset(base_dataset_train, normalize=True)
    norm_stats = dataset_train.get_norm_stats()
    dataset_val = FlowCastWrapperDataset(base_dataset_val, normalize=True, norm_stats=norm_stats)
    dataset_test = FlowCastWrapperDataset(base_dataset_test, normalize=True, norm_stats=norm_stats)

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nBase dataset sizes:")
    print(f"  Training:   {len(base_dataset_train)} samples")
    print(f"  Validation: {len(base_dataset_val)} samples")
    print(f"  Test:       {len(base_dataset_test)} samples")

    print(f"\nWrapper dataset sizes (after filtering for t-2 availability):")
    print(f"  Training:   {len(dataset_train)} samples")
    print(f"  Validation: {len(dataset_val)} samples")
    print(f"  Test:       {len(dataset_test)} samples")

    print(f"\nNormalization statistics:")
    print(f"  U-velocity: mean={norm_stats['u_mean']:.4f}, std={norm_stats['u_std']:.4f}")
    print(f"  V-velocity: mean={norm_stats['v_mean']:.4f}, std={norm_stats['v_std']:.4f}")

    # Calculate training iterations
    batch_size = 32
    accum_iter = 1
    epochs = 100

    batches_per_epoch = len(dataset_train) // batch_size
    gradient_updates_per_epoch = batches_per_epoch // accum_iter
    total_gradient_updates = gradient_updates_per_epoch * epochs

    print(f"\nTraining dynamics (batch_size={batch_size}, accum_iter={accum_iter}):")
    print(f"  Batches per epoch:    {batches_per_epoch}")
    print(f"  Gradient updates/epoch: {gradient_updates_per_epoch}")
    print(f"  Total updates (100 epochs): {total_gradient_updates}")

    # Check if dataset is suspiciously small
    print(f"\nDataset quality assessment:")
    if len(dataset_train) < 1000:
        print(f"  ⚠️  WARNING: Training set is very small ({len(dataset_train)} samples)")
        print(f"      Flow matching models typically need 10k-100k+ samples")
    elif len(dataset_train) < 5000:
        print(f"  ⚠️  CAUTION: Training set is small ({len(dataset_train)} samples)")
        print(f"      May struggle to generalize")
    else:
        print(f"  ✓ Training set size looks reasonable ({len(dataset_train)} samples)")

    # Check train/val ratio
    ratio = len(dataset_train) / len(dataset_val)
    print(f"\n  Train/Val ratio: {ratio:.2f}x")
    if ratio < 4:
        print(f"  ⚠️  WARNING: Train/Val ratio is low (typical: 4-10x)")

    # Sample data statistics
    print(f"\nSampling data to check distributions...")
    sample = dataset_train[0]
    x_prev_2 = sample['x_prev_2']
    x_prev_1 = sample['x_prev_1']
    x_target = sample['x_target']

    print(f"  Shape: {x_target.shape}")
    print(f"  X_prev_2 range: [{x_prev_2.min():.4f}, {x_prev_2.max():.4f}]")
    print(f"  X_prev_1 range: [{x_prev_1.min():.4f}, {x_prev_1.max():.4f}]")
    print(f"  X_target range: [{x_target.min():.4f}, {x_target.max():.4f}]")

    # Check temporal difference
    temporal_diff = (x_target[:2] - x_prev_1[:2]).abs().mean()
    print(f"  Mean temporal change: {temporal_diff:.6f}")

    if temporal_diff < 0.01:
        print(f"  ⚠️  WARNING: Temporal changes are very small!")
        print(f"      Model may struggle to learn meaningful dynamics")

    print("="*60 + "\n")

if __name__ == '__main__':
    main()
