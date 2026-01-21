"""
Data augmentation for fluid dynamics to artificially increase dataset size.
"""

import torch
import torch.nn.functional as F


class FluidAugmentation:
    """
    Augmentation transforms for fluid dynamics data that preserve physical validity.

    Applies geometric transformations that are valid for CFD:
    - Horizontal flip (valid if no asymmetric forces)
    - Vertical flip (valid if gravity not modeled)
    - Small rotations (±10 degrees)
    - Small spatial shifts
    """

    def __init__(
        self,
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.0,  # Usually invalid for flows with gravity
        rotation_prob=0.3,
        max_rotation_degrees=10,
        shift_prob=0.3,
        max_shift_pixels=4,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.max_rotation_degrees = max_rotation_degrees
        self.shift_prob = shift_prob
        self.max_shift_pixels = max_shift_pixels

    def __call__(self, x_prev_2, x_prev_1, x_target, mask):
        """
        Apply augmentation to all three frames consistently.

        Args:
            x_prev_2: (2, H, W) - velocity at t-2
            x_prev_1: (2, H, W) - velocity at t-1
            x_target: (2, H, W) - velocity at t
            mask: (1, H, W) - spatial mask

        Returns:
            Augmented versions of all inputs
        """
        # Stack for consistent transformation
        all_frames = torch.stack([x_prev_2, x_prev_1, x_target], dim=0)  # (3, 2, H, W)

        # Horizontal flip (flips x-velocity sign!)
        if torch.rand(1).item() < self.horizontal_flip_prob:
            all_frames = torch.flip(all_frames, dims=[3])  # Flip width
            all_frames[:, 0] *= -1  # Flip u-velocity sign
            mask = torch.flip(mask, dims=[2])

        # Vertical flip (flips y-velocity sign!)
        if torch.rand(1).item() < self.vertical_flip_prob:
            all_frames = torch.flip(all_frames, dims=[2])  # Flip height
            all_frames[:, 1] *= -1  # Flip v-velocity sign
            mask = torch.flip(mask, dims=[1])

        # Small rotation (requires interpolation, more expensive)
        if torch.rand(1).item() < self.rotation_prob:
            angle = torch.rand(1).item() * 2 * self.max_rotation_degrees - self.max_rotation_degrees
            all_frames, mask = self._rotate(all_frames, mask, angle)

        # Random spatial shift
        if torch.rand(1).item() < self.shift_prob:
            shift_x = torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, (1,)).item()
            shift_y = torch.randint(-self.max_shift_pixels, self.max_shift_pixels + 1, (1,)).item()
            all_frames = torch.roll(all_frames, shifts=(shift_y, shift_x), dims=(2, 3))
            mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(1, 2))

        x_prev_2, x_prev_1, x_target = all_frames[0], all_frames[1], all_frames[2]
        return x_prev_2, x_prev_1, x_target, mask

    def _rotate(self, frames, mask, angle_degrees):
        """Rotate frames and mask by given angle."""
        import math
        angle_rad = math.radians(angle_degrees)

        # Create rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Affine grid expects (N, C, H, W)
        N, C, H, W = frames.shape
        theta = torch.tensor([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0]
        ], dtype=frames.dtype, device=frames.device).unsqueeze(0).repeat(N, 1, 1)

        grid = F.affine_grid(theta, frames.size(), align_corners=False)

        # Rotate velocity frames
        rotated_frames = F.grid_sample(frames, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # Rotate velocity vectors themselves
        u = rotated_frames[:, 0]  # x-velocity
        v = rotated_frames[:, 1]  # y-velocity
        u_rot = cos_a * u - sin_a * v
        v_rot = sin_a * u + cos_a * v
        rotated_frames[:, 0] = u_rot
        rotated_frames[:, 1] = v_rot

        # Rotate mask
        mask_expanded = mask.unsqueeze(0).expand(N, -1, -1, -1)
        theta_mask = theta[:1]  # Just one mask
        grid_mask = F.affine_grid(theta_mask, mask_expanded[:1].size(), align_corners=False)
        rotated_mask = F.grid_sample(mask_expanded[:1], grid_mask, mode='nearest', padding_mode='zeros', align_corners=False)

        return rotated_frames, rotated_mask.squeeze(0)


class AugmentedFlowCastDataset:
    """
    Wrapper that applies augmentation to FlowCastWrapperDataset.
    """

    def __init__(self, base_dataset, augmentation=None):
        self.base_dataset = base_dataset
        self.augmentation = augmentation if augmentation is not None else FluidAugmentation()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]

        x_prev_2 = sample['x_prev_2']  # (3, H, W) - includes mask
        x_prev_1 = sample['x_prev_1']
        x_target = sample['x_target']

        # Separate velocity and mask
        x_prev_2_vel = x_prev_2[:2]  # (2, H, W)
        x_prev_1_vel = x_prev_1[:2]
        x_target_vel = x_target[:2]
        mask = x_target[2:3]  # (1, H, W)

        # Apply augmentation
        x_prev_2_aug, x_prev_1_aug, x_target_aug, mask_aug = self.augmentation(
            x_prev_2_vel, x_prev_1_vel, x_target_vel, mask
        )

        # Recombine
        sample['x_prev_2'] = torch.cat([x_prev_2_aug, mask_aug], dim=0)
        sample['x_prev_1'] = torch.cat([x_prev_1_aug, mask_aug], dim=0)
        sample['x_target'] = torch.cat([x_target_aug, mask_aug], dim=0)

        return sample
