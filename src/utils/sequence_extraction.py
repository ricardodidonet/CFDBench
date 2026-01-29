"""
Helper functions to extract ground truth sequences from batched data.
"""

import torch
from typing import List, Optional, Tuple, Dict


def extract_ground_truth_sequences(
    batch_start_idx: int,
    batch_size: int,
    num_future_steps: int,
    dataset,
    device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """
    Extract ground truth sequences for multi-step forecasting.

    For each sample in a batch, checks if the next K timesteps are from the
    same trajectory. If yes, extracts them. If no, skips that sample.

    Args:
        batch_start_idx: Starting index in the dataset for this batch
        batch_size: Number of samples in batch
        num_future_steps: Number of future timesteps needed (K)
        dataset: The FlowCastWrapperDataset instance
        device: Device to put tensors on

    Returns:
        List of valid samples, each containing:
        - 'sample_idx': Index within batch
        - 'x_prev_2': Initial condition at t-2
        - 'x_prev_1': Initial condition at t-1
        - 'ground_truth': List of K future frames [x_t, x_{t+1}, ..., x_{t+K-1}]
        - 'case_params': Case parameters
    """
    valid_samples = []

    for i in range(batch_size):
        dataset_idx = batch_start_idx + i

        if dataset_idx >= len(dataset):
            break

        # Check if we can get K future frames from same trajectory
        can_extract, future_frames = check_and_extract_sequence(
            dataset, dataset_idx, num_future_steps
        )

        if can_extract:
            # Get initial conditions
            sample = dataset[dataset_idx]

            valid_samples.append({
                'sample_idx': i,
                'x_prev_2': sample['x_prev_2'].to(device),
                'x_prev_1': sample['x_prev_1'].to(device),
                'ground_truth': [f.to(device) for f in future_frames],
                'case_params': sample['case_params']
            })

    return valid_samples


def check_and_extract_sequence(
    dataset,
    start_idx: int,
    num_steps: int
) -> Tuple[bool, Optional[List[torch.Tensor]]]:
    """
    Check if we can extract a sequence of num_steps from the same trajectory.

    Args:
        dataset: FlowCastWrapperDataset instance
        start_idx: Starting index in dataset
        num_steps: Number of consecutive frames needed

    Returns:
        (can_extract, frames) where:
        - can_extract: True if sequence is valid (same case, consecutive)
        - frames: List of K tensors if valid, None otherwise
    """
    # Check bounds
    if start_idx + num_steps > len(dataset):
        return False, None

    # Get first sample to check case
    first_sample = dataset[start_idx]

    # Get case_id from the wrapper dataset's base dataset
    base_idx_0 = dataset.valid_indices[start_idx]
    case_id_0 = dataset.base_dataset.case_ids[base_idx_0]

    # Collect frames and verify they're all from same case
    frames = [first_sample['x_target']]  # First frame is x_target from start_idx

    for k in range(1, num_steps):
        next_idx = start_idx + k

        # Check this sample is from same case
        base_idx_k = dataset.valid_indices[next_idx]
        case_id_k = dataset.base_dataset.case_ids[base_idx_k]

        if case_id_k != case_id_0:
            # Crossed case boundary!
            return False, None

        # Also verify temporal continuity (base indices should be consecutive)
        expected_base_idx = base_idx_0 + k
        if base_idx_k != expected_base_idx:
            # Gap in the sequence (e.g., due to convergence filtering)
            return False, None

        # Extract frame
        sample_k = dataset[next_idx]
        frames.append(sample_k['x_target'])

    return True, frames


def batch_to_sequences(
    batch: Dict[str, torch.Tensor],
    batch_start_idx: int,
    dataset,
    num_future_steps: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Convert a batch to sequences with ground truth.

    Only returns samples where we can extract full ground truth sequences.

    Args:
        batch: Batch from DataLoader
        batch_start_idx: Global dataset index where this batch starts
        dataset: FlowCastWrapperDataset instance
        num_future_steps: Number of future steps to extract
        device: Device

    Returns:
        (x_prev_2, x_prev_1, case_params, ground_truth_list, mask)
        where ground_truth_list is List of K tensors, each (B_valid, 2, H, W)
        B_valid <= original batch_size (only samples with valid sequences)
    """
    valid_samples = extract_ground_truth_sequences(
        batch_start_idx,
        batch['x_prev_2'].shape[0],  # batch_size
        num_future_steps,
        dataset,
        device
    )

    if not valid_samples:
        # No valid sequences in this batch
        return None, None, None, None, None

    # Stack valid samples
    x_prev_2 = torch.stack([s['x_prev_2'][:2] for s in valid_samples])  # (B', 2, H, W)
    x_prev_1 = torch.stack([s['x_prev_1'][:2] for s in valid_samples])
    mask = torch.stack([s['x_prev_2'][2:3] for s in valid_samples])  # (B', 1, H, W)

    # Convert case_params dicts to tensor
    case_params_list = [s['case_params'] for s in valid_samples]
    param_keys = sorted(case_params_list[0].keys())
    case_params = torch.tensor([
        [float(params[key]) for key in param_keys]
        for params in case_params_list
    ], device=device)

    # Stack ground truth as list of tensors
    ground_truth_list = []
    for k in range(num_future_steps):
        gt_k = torch.stack([s['ground_truth'][k][:2] for s in valid_samples])  # (B', 2, H, W)
        ground_truth_list.append(gt_k)

    return x_prev_2, x_prev_1, case_params, ground_truth_list, mask


# Example usage
def example_usage():
    """
    Example of how to use these functions in inference.
    """
    from torch.utils.data import DataLoader
    from dataset.wrapper import FlowCastWrapperDataset

    dataset = FlowCastWrapperDataset(base_dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    num_future_steps = 5
    device = torch.device('cuda')

    for batch_idx, batch in enumerate(dataloader):
        batch_start_idx = batch_idx * dataloader.batch_size

        # Extract valid sequences with ground truth
        result = batch_to_sequences(
            batch,
            batch_start_idx,
            dataset,
            num_future_steps,
            device
        )

        if result[0] is None:
            print(f"Batch {batch_idx}: No valid sequences (all cross case boundaries)")
            continue

        x_prev_2, x_prev_1, case_params, ground_truth, mask = result

        print(f"Batch {batch_idx}: {x_prev_2.shape[0]} valid samples (out of {batch['x_prev_2'].shape[0]})")

        # Now you can do multi-step forecasting with proper ground truth
        # predictions = model.predict_multistep(x_prev_2, x_prev_1, case_params, num_steps=5)
        # metrics = compute_metrics(predictions, ground_truth, mask)
