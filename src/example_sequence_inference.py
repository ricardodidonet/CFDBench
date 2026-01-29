"""
Example: Multi-step inference with ground truth sequences using FlowCastWrapperDataset.

This shows how to handle batches that may contain samples from different trajectories.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import get_auto_dataset
from dataset.wrapper import FlowCastWrapperDataset
from utils.sequence_extraction import batch_to_sequences


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_future_steps = 5

    # Load dataset
    data_dir = Path("../data")
    base_train, base_val, _ = get_auto_dataset(
        data_dir=data_dir,
        data_name='cylinder_geo',
        delta_time=0.1,
        norm_props=True,
        norm_bc=True,
        load_splits=['train', 'dev']
    )

    # Create wrapper
    dataset_train = FlowCastWrapperDataset(base_train, normalize=True)
    norm_stats = dataset_train.get_norm_stats()
    dataset_val = FlowCastWrapperDataset(base_val, normalize=True, norm_stats=norm_stats)

    # Create DataLoader
    dataloader = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,  # IMPORTANT: Must be False to track indices
        num_workers=0,  # Set to 0 for simplicity
    )

    print(f"Total samples in validation set: {len(dataset_val)}")

    # Process batches
    total_valid_samples = 0
    total_skipped_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        batch_start_idx = batch_idx * dataloader.batch_size

        # Extract sequences with ground truth
        result = batch_to_sequences(
            batch,
            batch_start_idx,
            dataset_val,
            num_future_steps,
            device
        )

        if result[0] is None:
            # Entire batch crosses case boundaries
            skipped = batch['x_prev_2'].shape[0]
            total_skipped_samples += skipped
            print(f"Batch {batch_idx}: Skipped all {skipped} samples (case boundary)")
            continue

        x_prev_2, x_prev_1, case_params, ground_truth, mask = result

        valid_count = x_prev_2.shape[0]
        original_count = batch['x_prev_2'].shape[0]
        skipped = original_count - valid_count

        total_valid_samples += valid_count
        total_skipped_samples += skipped

        print(f"Batch {batch_idx}: {valid_count}/{original_count} valid samples")

        # Verify ground truth structure
        assert len(ground_truth) == num_future_steps
        for k, gt_k in enumerate(ground_truth):
            assert gt_k.shape == (valid_count, 2, 64, 64), f"GT step {k} has wrong shape"

        # Here you would do inference:
        # predictions = model.predict_multistep(x_prev_2, x_prev_1, case_params, num_steps=5)
        # metrics = compute_metrics(predictions, ground_truth, mask)

        # Example: Check temporal consistency
        if valid_count > 0:
            # First ground truth frame should be close to what we'd predict from x_prev_1
            temporal_diff = (ground_truth[0] - x_prev_1).abs().mean()
            print(f"  Mean temporal change: {temporal_diff:.6f}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total valid samples:   {total_valid_samples}")
    print(f"  Total skipped samples: {total_skipped_samples}")
    print(f"  Utilization:           {100*total_valid_samples/(total_valid_samples+total_skipped_samples):.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
