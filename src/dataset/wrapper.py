import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Assuming your CFDBench dataset classes are importable
# from dataset.base import CfdAutoDataset # Or the specific class like CavityFlowAutoDataset

class FlowCastWrapperDataset(Dataset):
    """
    Wraps a CFDBench CfdAutoDataset to return three consecutive frames
    (t-2, t-1, t) needed for the FlowCast model.

    It assumes the base dataset provides (X_{t-1}, X_t, case_params).
    """
    def __init__(self, base_dataset, normalize=True):
        """
        Args:
            base_dataset: An instance of a CfdAutoDataset subclass
                          (e.g., CavityFlowAutoDataset).
            normalize: whether to normalize the data.
        """
        self.base_dataset = base_dataset
        self.normalize = normalize

        if self.normalize:
            # compute per-channel mean and std:
            self.compute_normalization_stats()
        
        # We need access to the original sequential data if possible,
        # but the base dataset pre-pairs t-1 and t.
        # We can reconstruct t-2 by looking at the *previous* index
        # in the base_dataset, BUT we must be careful about case boundaries.

        # Pre-calculate valid indices to avoid crossing case boundaries.
        self.valid_indices = []
        logger.info("Pre-calculating valid indices for FlowCastWrapperDataset...")

        for i in range(len(self.base_dataset)):
            # Get case ID for current index (i corresponds to t-1)
            # and previous index (i-1 corresponds to t-2)
            current_case_id = self.base_dataset.case_ids[i]
            if i > 0:
                previous_case_id = self.base_dataset.case_ids[i-1]
                # Only valid if the previous sample is from the same case
                if current_case_id == previous_case_id:
                    self.valid_indices.append(i)
            # The very first sample (i=0) is never valid because it has no t-2
        logger.info(f"Wrapper dataset contains {len(self.valid_indices)} valid (t-2, t-1, t) samples.")

    def compute_normalization_stats(self):
        """Compute mean/std for u and v channels."""
        all_u, all_v = [], []
        for i in range(len(self.base_dataset)):
            x, _, _ = self.base_dataset[i]
            all_u.append(x[0])  # u channel
            all_v.append(x[1])  # v channel
        
        all_u = torch.stack(all_u)
        all_v = torch.stack(all_v)
        
        self.u_mean, self.u_std = all_u.mean(), all_u.std()
        self.v_mean, self.v_std = all_v.mean(), all_v.std()
        
        logger.info(f"Normalization stats - u: mean={self.u_mean:.4f}, std={self.u_std:.4f}")
        logger.info(f"Normalization stats - v: mean={self.v_mean:.4f}, std={self.v_std:.4f}")


    def __len__(self) -> int:
        # The length is the number of valid indices we found
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
        - 'x_prev_2':   X_{t-2} tensor [C, H, W] (including mask channel)
        - 'x_prev_1':   X_{t-1} tensor [C, H, W] (including mask channel)
        - 'x_target':   X_{t}   tensor [C, H, W] (including mask channel)
        - 'case_params': Dictionary of case parameters for this sample.
                         (Will be converted to tensor in collate_fn)
        """
        # 'index' here refers to the index within self.valid_indices
        # Get the corresponding index in the base_dataset
        base_dataset_index = self.valid_indices[index]

        # Get the (X_{t-1}, X_t, case_params) tuple for the *current* step
        x_prev_1, x_target, case_params = self.base_dataset[base_dataset_index]

        # Get the (X_{t-2}, X_{t-1}, case_params) tuple for the *previous* step
        # We know base_dataset_index > 0 and it's within the same case
        # because of how we constructed valid_indices.
        x_prev_2,  _,  _ = self.base_dataset[base_dataset_index - 1]
        
        # Note: x_prev_1 and the label from the previous step should be identical.
        # We assume case_params are consistent for consecutive steps within a case.

        if self.normalize:
            # normalize velocity channels:  
            x_prev_2[0] = (x_prev_2[0] - self.u_mean) / self.u_std
            x_prev_2[1] = (x_prev_2[1] - self.v_mean) / self.v_std

            x_prev_1[0] = (x_prev_1[0] - self.u_mean) / self.u_std
            x_prev_1[1] = (x_prev_1[1] - self.v_mean) / self.v_std

            x_target[0] = (x_target[0] - self.u_mean) / self.u_std
            x_target[1] = (x_target[1] - self.v_mean) / self.v_std

        return {
            'x_prev_2':     x_prev_2,         # This is X_{t-2}
            'x_prev_1':     x_prev_1,         # This is X_{t-1}
            'x_target':     x_target,         # This is X_{t}
            'case_params':  case_params       # Pass the dict, collate handles tensor conversion
        }

