"""
Extended wrapper for multi-step forecasting with ground truth.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MultiStepFlowCastDataset(Dataset):
    """
    Extends FlowCastWrapperDataset to provide multiple future timesteps
    for ground truth comparison in multi-step forecasting.

    Returns:
        - x_prev_2, x_prev_1: Initial conditions (t-2, t-1)
        - x_future: List of K future frames [x_t, x_{t+1}, ..., x_{t+K-1}]
        - case_params: Case parameters
    """

    def __init__(self, base_dataset, num_future_steps=10, normalize=True, norm_stats=None):
        """
        Args:
            base_dataset: CfdAutoDataset instance
            num_future_steps: Number of future timesteps to include
            normalize: Whether to normalize
            norm_stats: Normalization statistics
        """
        self.base_dataset = base_dataset
        self.num_future_steps = num_future_steps
        self.normalize = normalize

        if self.normalize:
            if norm_stats is not None:
                self.u_mean = norm_stats['u_mean']
                self.u_std = norm_stats['u_std']
                self.v_mean = norm_stats['v_mean']
                self.v_std = norm_stats['v_std']
            else:
                self.compute_normalization_stats()

        # Pre-calculate valid indices
        # Need indices where we can get: t-2, t-1, t, t+1, ..., t+K-1
        # without crossing case boundaries
        self.valid_indices = []
        logger.info(f"Pre-calculating valid indices for {num_future_steps}-step forecasting...")

        for i in range(len(self.base_dataset)):
            # Check if we can get t-2 (need i > 0)
            if i == 0:
                continue

            # Check if we can get all future steps (need i + num_future_steps - 1 < len)
            if i + num_future_steps > len(self.base_dataset):
                break

            # Check all frames are from the same case
            current_case_id = self.base_dataset.case_ids[i]
            prev_case_id = self.base_dataset.case_ids[i - 1]

            if prev_case_id != current_case_id:
                continue  # Can't get t-2

            # Check all future steps are from same case
            all_same_case = True
            for j in range(1, num_future_steps):
                if i + j >= len(self.base_dataset):
                    all_same_case = False
                    break
                future_case_id = self.base_dataset.case_ids[i + j]
                if future_case_id != current_case_id:
                    all_same_case = False
                    break

            if all_same_case:
                self.valid_indices.append(i)

        logger.info(f"Found {len(self.valid_indices)} valid sequences for {num_future_steps}-step forecasting")
        logger.info(f"(Reduced from {len(base_dataset)} base samples)")

    def compute_normalization_stats(self):
        """Compute mean/std for u and v channels."""
        all_u, all_v = [], []
        for i in range(len(self.base_dataset)):
            x, _, _ = self.base_dataset[i]
            all_u.append(x[0])
            all_v.append(x[1])

        all_u = torch.stack(all_u)
        all_v = torch.stack(all_v)

        self.u_mean, self.u_std = all_u.mean(), all_u.std()
        self.v_mean, self.v_std = all_v.mean(), all_v.std()

        logger.info(f"Normalization stats - u: mean={self.u_mean:.4f}, std={self.u_std:.4f}")
        logger.info(f"Normalization stats - v: mean={self.v_mean:.4f}, std={self.v_std:.4f}")

    def get_norm_stats(self):
        """Return normalization statistics."""
        if not self.normalize:
            return None
        return {
            'u_mean': self.u_mean,
            'u_std': self.u_std,
            'v_mean': self.v_mean,
            'v_std': self.v_std
        }

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict:
        """
        Returns:
            - x_prev_2: Frame at t-2 (C, H, W)
            - x_prev_1: Frame at t-1 (C, H, W)
            - x_future: List of K future frames [t, t+1, ..., t+K-1], each (C, H, W)
            - case_params: Case parameters dict
        """
        base_idx = self.valid_indices[index]

        # Get t-2
        x_prev_2, _, _ = self.base_dataset[base_idx - 1]

        # Get t-1, t, and case params
        x_prev_1, x_t, case_params = self.base_dataset[base_idx]

        # Collect future frames: [t, t+1, t+2, ..., t+K-1]
        x_future = [x_t]
        for k in range(1, self.num_future_steps):
            x_future_k, _, _ = self.base_dataset[base_idx + k]
            x_future.append(x_future_k)

        # Normalize if needed
        if self.normalize:
            x_prev_2[0] = (x_prev_2[0] - self.u_mean) / self.u_std
            x_prev_2[1] = (x_prev_2[1] - self.v_mean) / self.v_std

            x_prev_1[0] = (x_prev_1[0] - self.u_mean) / self.u_std
            x_prev_1[1] = (x_prev_1[1] - self.v_mean) / self.v_std

            for x in x_future:
                x[0] = (x[0] - self.u_mean) / self.u_std
                x[1] = (x[1] - self.v_mean) / self.v_std

        return {
            'x_prev_2': x_prev_2,
            'x_prev_1': x_prev_1,
            'x_future': x_future,  # List of K tensors
            'case_params': case_params
        }


def multi_step_collate_fn(batch):
    """
    Collate function for MultiStepFlowCastDataset.

    Returns:
        - x_prev_2: (B, 2, H, W)
        - x_prev_1: (B, 2, H, W)
        - x_future: List of K tensors, each (B, 2, H, W)
        - mask: (B, 1, H, W)
        - case_params: (B, D)
    """
    x_prev_2 = torch.stack([item['x_prev_2'] for item in batch])
    x_prev_1 = torch.stack([item['x_prev_1'] for item in batch])

    # Stack future frames: convert from list of B lists of K tensors
    # to list of K tensors of shape (B, C, H, W)
    num_future_steps = len(batch[0]['x_future'])
    x_future = []
    for k in range(num_future_steps):
        x_future_k = torch.stack([item['x_future'][k] for item in batch])
        x_future.append(x_future_k)

    # Get case params
    case_params_list = [item['case_params'] for item in batch]
    param_keys = sorted(case_params_list[0].keys())
    case_params_tensor = torch.tensor([
        [float(params[key]) for key in param_keys]
        for params in case_params_list
    ])

    return {
        'x_prev_2': x_prev_2[:, :2],      # Velocity only
        'x_prev_1': x_prev_1[:, :2],
        'x_future': [x[:, :2] for x in x_future],  # List of K velocity tensors
        'mask': x_future[0][:, 2:3],      # Mask from first future frame
        'case_params': case_params_tensor
    }
