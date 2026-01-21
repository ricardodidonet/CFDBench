#!/usr/bin/env python3
"""
Plot training and validation losses from JSON log file.

Usage:
    python plot_training_logs.py --log_file training.log
    python plot_training_logs.py --log_file training.log --output losses.png

    # Or read from stdin:
    cat training.log | python plot_training_logs.py
"""

import json
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_file=None):
    """
    Parse JSON log file with one JSON object per line.

    Args:
        log_file: Path to log file, or None to read from stdin

    Returns:
        train_epochs: List of epochs with training loss
        train_losses: List of training losses
        eval_epochs: List of epochs with evaluation metrics
        eval_losses: Dict with 'mse', 'u_mse', 'v_mse', 'magnitude_mse'
    """
    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = {
        'mse': [],
        'u_mse': [],
        'v_mse': [],
        'magnitude_mse': []
    }

    # Read from file or stdin
    if log_file is not None:
        with open(log_file, 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract training loss
        if 'train_loss' in data and 'epoch' in data:
            train_epochs.append(data['epoch'])
            train_losses.append(data['train_loss'])

        # Extract evaluation metrics
        if 'eval_mse' in data and 'epoch' in data:
            eval_epochs.append(data['epoch'])
            eval_losses['mse'].append(data['eval_mse'])

            if 'eval_u_mse' in data:
                eval_losses['u_mse'].append(data['eval_u_mse'])
            if 'eval_v_mse' in data:
                eval_losses['v_mse'].append(data['eval_v_mse'])
            if 'eval_magnitude_mse' in data:
                eval_losses['magnitude_mse'].append(data['eval_magnitude_mse'])

    return train_epochs, train_losses, eval_epochs, eval_losses


def plot_losses(train_epochs, train_losses, eval_epochs, eval_losses, output_path=None):
    """
    Create a comprehensive loss plot with multiple subplots.

    Args:
        train_epochs: List of training epochs
        train_losses: List of training losses
        eval_epochs: List of evaluation epochs
        eval_losses: Dict of evaluation losses
        output_path: Path to save figure (if None, shows interactively)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

    # Plot 1: Training vs Validation MSE
    ax1 = axes[0, 0]
    ax1.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if eval_epochs and eval_losses['mse']:
        ax1.plot(eval_epochs, eval_losses['mse'], 'r-', label='Validation MSE',
                linewidth=2, marker='o', markersize=6)

        # Compute train/val ratio at eval points
        eval_train_losses = [train_losses[e] for e in eval_epochs if e < len(train_losses)]
        if eval_train_losses and eval_losses['mse']:
            ratios = [v/t for v, t in zip(eval_losses['mse'], eval_train_losses) if t > 0]
            if ratios:
                mean_ratio = np.mean(ratios)
                ax1.text(0.95, 0.95, f'Mean Val/Train: {mean_ratio:.1f}x',
                        transform=ax1.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=10)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: U and V component MSE
    ax2 = axes[0, 1]
    if eval_epochs and eval_losses['u_mse']:
        ax2.plot(eval_epochs, eval_losses['u_mse'], 'g-', label='U-velocity MSE',
                linewidth=2, marker='s', markersize=5)
    if eval_epochs and eval_losses['v_mse']:
        ax2.plot(eval_epochs, eval_losses['v_mse'], 'm-', label='V-velocity MSE',
                linewidth=2, marker='^', markersize=5)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Component MSE', fontsize=12)
    ax2.set_title('Velocity Component Errors', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Magnitude MSE
    ax3 = axes[1, 0]
    if eval_epochs and eval_losses['magnitude_mse']:
        ax3.plot(eval_epochs, eval_losses['magnitude_mse'], 'c-',
                label='Magnitude MSE', linewidth=2, marker='d', markersize=5)

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Magnitude MSE', fontsize=12)
    ax3.set_title('Velocity Magnitude Error', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Train/Val Ratio Over Time
    ax4 = axes[1, 1]
    if eval_epochs and eval_losses['mse'] and train_losses:
        eval_train_losses = [train_losses[e] for e in eval_epochs if e < len(train_losses)]
        if eval_train_losses and len(eval_train_losses) == len(eval_losses['mse']):
            ratios = [v/t for v, t in zip(eval_losses['mse'], eval_train_losses) if t > 0]
            if ratios:
                ax4.plot(eval_epochs[:len(ratios)], ratios, 'orange',
                        linewidth=2, marker='o', markersize=6)
                ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='No overfitting')
                ax4.axhline(y=2, color='y', linestyle='--', alpha=0.5, label='Healthy (2x)')
                ax4.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Overfitting (5x)')

                # Highlight the trend
                if len(ratios) > 1:
                    trend = ratios[-1] - ratios[0]
                    trend_text = "Improving" if trend < 0 else "Worsening"
                    ax4.text(0.05, 0.95, f'Trend: {trend_text}',
                            transform=ax4.transAxes, ha='left', va='top',
                            bbox=dict(boxstyle='round',
                                     facecolor='lightgreen' if trend < 0 else 'lightcoral',
                                     alpha=0.5),
                            fontsize=10)

    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation / Training Loss', fontsize=12)
    ax4.set_title('Overfitting Monitor', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary(train_epochs, train_losses, eval_epochs, eval_losses):
    """Print a text summary of training progress."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    if train_losses:
        print(f"Training epochs: {len(train_losses)}")
        print(f"Initial train loss: {train_losses[0]:.6f}")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        improvement = train_losses[0] / train_losses[-1]
        print(f"Improvement: {improvement:.2f}x")

    if eval_losses['mse']:
        print(f"\nEvaluation epochs: {len(eval_losses['mse'])}")
        print(f"Initial val MSE: {eval_losses['mse'][0]:.6f}")
        print(f"Final val MSE: {eval_losses['mse'][-1]:.6f}")
        print(f"Best val MSE: {min(eval_losses['mse']):.6f} (epoch {eval_epochs[np.argmin(eval_losses['mse'])]})")

        # Compute train/val ratio
        if train_losses:
            final_train = train_losses[eval_epochs[-1]] if eval_epochs[-1] < len(train_losses) else train_losses[-1]
            ratio = eval_losses['mse'][-1] / final_train
            print(f"\nFinal train/val ratio: {ratio:.2f}x")

            if ratio < 2:
                print("Status: ✓ Good generalization")
            elif ratio < 5:
                print("Status: ⚠ Moderate overfitting")
            else:
                print("Status: ✗ Severe overfitting")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training and validation losses from JSON log file'
    )
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (if not provided, reads from stdin)')
    parser.add_argument('--output', type=str, default='training_losses.png',
                       help='Output path for plot (default: training_losses.png)')
    parser.add_argument('--no_save', action='store_true',
                       help='Show plot interactively instead of saving')
    parser.add_argument('--summary', action='store_true',
                       help='Print text summary of training progress')

    args = parser.parse_args()

    # Parse logs
    print(f"Reading logs from {'stdin' if args.log_file is None else args.log_file}...")
    train_epochs, train_losses, eval_epochs, eval_losses = parse_log_file(args.log_file)

    if not train_losses:
        print("Error: No training data found in log file!")
        sys.exit(1)

    print(f"Found {len(train_losses)} training epochs and {len(eval_epochs)} evaluation epochs")

    # Print summary if requested
    if args.summary:
        print_summary(train_epochs, train_losses, eval_epochs, eval_losses)

    # Create plot
    output_path = None if args.no_save else args.output
    plot_losses(train_epochs, train_losses, eval_epochs, eval_losses, output_path)


if __name__ == '__main__':
    main()
