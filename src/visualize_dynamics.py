"""
visualize_temporal_dynamics.py

Visualize consecutive frames from the CFD dataset to understand temporal dynamics.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_case_data(case_dir):
    """Load u and v from a case directory."""
    u = np.load(case_dir / 'u.npy')  # Shape: (T, H, W)
    v = np.load(case_dir / 'v.npy')  # Shape: (T, H, W)
    return u, v


def compute_magnitude(u, v):
    """Compute velocity magnitude."""
    return np.sqrt(u**2 + v**2)

def compute_vorticity(u, v, dx=0.00344, dy=0.00187):
    """compute fluid vorticity"""
    dv_dx = np.gradient(v, dx, axis=-1)
    du_dy = np.gradient(u, dy, axis=-2)
    return dv_dx - du_dy


def visualize_frames(u, v, start_frame=0, num_frames=10, delta_time=0.1, save_path=None):
    """
    Create a grid visualization of consecutive frames.
    
    Args:
        u: x-velocity, shape (T, H, W)
        v: y-velocity, shape (T, H, W)
        start_frame: starting frame index
        num_frames: number of frames to show
        delta_time: time step between frames
        save_path: path to save figure (optional)
    """
    end_frame = min(start_frame + num_frames, u.shape[0])
    actual_frames = end_frame - start_frame
    
    fig, axes = plt.subplots(4, actual_frames, figsize=(3 * actual_frames, 12))
    
    # Compute global min/max for consistent colorscales
    u_subset = u[start_frame:end_frame]
    v_subset = v[start_frame:end_frame]
    mag_subset = compute_magnitude(u_subset, v_subset)
    
    u_vmin, u_vmax = u_subset.min(), u_subset.max()
    v_vmin, v_vmax = v_subset.min(), v_subset.max()
    mag_vmax = mag_subset.max()
    
    # Compute frame differences
    if actual_frames > 1:
        u_diff = np.diff(u_subset, axis=0)
        v_diff = np.diff(v_subset, axis=0)
        diff_max = max(np.abs(u_diff).max(), np.abs(v_diff).max())
    
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        t = frame_idx * delta_time
        
        mag = compute_magnitude(u[frame_idx], v[frame_idx])
        
        # Row 0: u velocity
        im0 = axes[0, i].imshow(u[frame_idx], cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
        axes[0, i].set_title(f't={t:.2f}s', fontsize=10)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('u velocity', fontsize=10)
        
        # Row 1: v velocity
        im1 = axes[1, i].imshow(v[frame_idx], cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('v velocity', fontsize=10)
        
        # Row 2: velocity magnitude
        im2 = axes[2, i].imshow(mag, cmap='viridis', vmin=0, vmax=mag_vmax)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('|V| magnitude', fontsize=10)
        
        # Row 3: frame difference (du/dt approximation)
        if i < actual_frames - 1:
            du = u[frame_idx + 1] - u[frame_idx]
            im3 = axes[3, i].imshow(du, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
            axes[3, i].axis('off')
        else:
            axes[3, i].axis('off')
            axes[3, i].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[3, i].transAxes)
        if i == 0:
            axes[3, i].set_ylabel('Δu (change)', fontsize=10)
    
    # Add colorbars
    fig.colorbar(im0, ax=axes[0, :], shrink=0.8, label='u [m/s]')
    fig.colorbar(im1, ax=axes[1, :], shrink=0.8, label='v [m/s]')
    fig.colorbar(im2, ax=axes[2, :], shrink=0.8, label='|V| [m/s]')
    if actual_frames > 1:
        fig.colorbar(im3, ax=axes[3, :], shrink=0.8, label='Δu [m/s]')
    
    plt.suptitle(f'Temporal Dynamics: Frames {start_frame}-{end_frame-1} (Δt={delta_time}s)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_temporal_statistics(u, v, delta_time=0.001):
    """
    Compute and print temporal statistics to understand dynamics timescale.
    """
    # Velocity statistics
    mag = compute_magnitude(u, v)
    
    print("=" * 60)
    print("TEMPORAL DYNAMICS STATISTICS")
    print("=" * 60)
    print(f"\nDataset shape: {u.shape} (T, H, W)")
    print(f"Total frames: {u.shape[0]}")
    print(f"Delta time: {delta_time} s")
    print(f"Total simulation time: {u.shape[0] * delta_time:.2f} s")
    
    print(f"\n--- Velocity Statistics (over all frames) ---")
    print(f"u velocity: min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    print(f"v velocity: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")
    print(f"|V| magnitude: min={mag.min():.4f}, max={mag.max():.4f}, mean={mag.mean():.4f}")
    
    # Frame-to-frame changes
    u_diff = np.diff(u, axis=0)
    v_diff = np.diff(v, axis=0)
    
    print(f"\n--- Frame-to-Frame Changes (Δt={delta_time}s) ---")
    print(f"Δu: min={u_diff.min():.6f}, max={u_diff.max():.6f}, mean_abs={np.abs(u_diff).mean():.6f}")
    print(f"Δv: min={v_diff.min():.6f}, max={v_diff.max():.6f}, mean_abs={np.abs(v_diff).mean():.6f}")
    
    # Relative changes
    u_rel_change = np.abs(u_diff) / (np.abs(u[:-1]) + 1e-8)
    v_rel_change = np.abs(v_diff) / (np.abs(v[:-1]) + 1e-8)
    
    print(f"\n--- Relative Changes per Frame ---")
    print(f"Mean |Δu/u|: {u_rel_change.mean():.4f} ({u_rel_change.mean()*100:.2f}%)")
    print(f"Mean |Δv/v|: {v_rel_change.mean():.4f} ({v_rel_change.mean()*100:.2f}%)")
    
    # Characteristic timescales
    char_velocity = mag.mean()
    char_length = max(u.shape[1], u.shape[2])  # Use domain size
    if char_velocity > 0:
        advection_time = char_length / char_velocity
        print(f"\n--- Estimated Timescales ---")
        print(f"Characteristic velocity: {char_velocity:.4f} m/s")
        print(f"Domain size: {char_length} cells")
        print(f"Advection timescale (L/V): {advection_time:.2f} s")
        print(f"Frames per advection time: {advection_time / delta_time:.1f}")
    
    print("=" * 60)


def create_animation(u, v, delta_time=0.1, save_path=None, fps=10):
    """
    Create an animation of the temporal evolution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    vorticity = compute_vorticity(u, v)
    u_vmin, u_vmax = u.min(), u.max()
    v_vmin, v_vmax = v.min(), v.max()
    vort_limit = np.percentile(np.abs(vorticity), 90)
    
    # Initial plots
    im0 = axes[0].imshow(u[0], cmap='RdBu_r', vmin=u_vmin, vmax=u_vmax)
    im1 = axes[1].imshow(v[0], cmap='RdBu_r', vmin=v_vmin, vmax=v_vmax)
    im2 = axes[2].imshow(vorticity[0], cmap='RdBu_r', vmin=-vort_limit, vmax=vort_limit)

    
    axes[0].set_title('u velocity')
    axes[1].set_title('v velocity')
    axes[2].set_title('vorticity')
    
    for ax in axes:
        ax.axis('off')
    
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    
    title = fig.suptitle(f't = 0.00 s (frame 0/{u.shape[0]-1})')
    
    def update(frame):
        im0.set_array(u[frame])
        im1.set_array(v[frame])
        im2.set_array(vorticity[frame])
        
        title.set_text(f't = {frame * delta_time:.2f} s (frame {frame}/{u.shape[0]-1})')
        return [im0, im1, im2, title]
    
    anim = FuncAnimation(fig, update, frames=u.shape[0], interval=1000/fps, blit=False)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize temporal dynamics of CFD dataset')
    parser.add_argument('--data_dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--problem', type=str, default='cylinder', 
                        choices=['cavity', 'tube', 'dam', 'cylinder'], help='CFD problem')
    parser.add_argument('--subset', type=str, default='bc', help='Dataset subset (bc, geo, prop)')
    parser.add_argument('--case', type=int, default=49, help='Case number to visualize')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to show')
    parser.add_argument('--delta_time', type=float, default=0.001, help='Time step between frames')
    parser.add_argument('--output', type=str, default=None, help='Output path for figure')
    parser.add_argument('--animate', action='store_true', help='Create animation instead of grid')
    parser.add_argument('--fps', type=int, default=20, help='FPS for animation')
    
    args = parser.parse_args()
    
    # Construct case path
    data_dir = Path(args.data_dir)
    case_dir = data_dir / args.problem / args.subset / f'case{args.case:04d}'
    
    if not case_dir.exists():
        print(f"Case directory not found: {case_dir}")
        print("Available cases:")
        subset_dir = data_dir / args.problem / args.subset
        if subset_dir.exists():
            cases = sorted(subset_dir.glob('case*'))[:10]
            for c in cases:
                print(f"  {c.name}")
        return
    
    print(f"Loading data from: {case_dir}")
    u, v = load_case_data(case_dir)
    
    # Print statistics
    compute_temporal_statistics(u, v, args.delta_time)
    
    if args.animate:
        output = args.output or f'{args.problem}_{args.subset}_case{args.case:04d}_animation.gif'
        create_animation(u, v, args.delta_time, save_path=output, fps=args.fps)
    else:
        output = args.output or f'{args.problem}_{args.subset}_case{args.case:04d}_frames.png'
        visualize_frames(u, v, args.start_frame, args.num_frames, args.delta_time, save_path=output)


if __name__ == '__main__':
    main()
