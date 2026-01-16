import json
import argparse


def get_train_parser():
    parser = argparse.ArgumentParser(description="Train fluid dynamics flow matching model")

    # Data parameters
    parser.add_argument("--data_path", type=str, default="../data",
                        help="Path to fluid dynamics data")
    parser.add_argument("--num_cases", type=int, default=100,
                        help="Number of simulation cases (for dummy data)")
    parser.add_argument("--num_timesteps", type=int, default=50,
                        help="Number of timesteps per case (for dummy data)")
    parser.add_argument("--num_case_params", type=int, default=3,
                        help="Number of case parameters")

    # Model parameters
    parser.add_argument("--model_channels", type=int, default=128,
                        help="Base channel count for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of residual blocks per resolution")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--use_fourier_conditioning", action="store_true", default=True,
                        help="Use Fourier features for case parameters")
    parser.add_argument("--num_fourier_freqs", type=int, default=16,
                        help="Number of Fourier frequencies")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training")

    # Flow matching parameters
    parser.add_argument("--use_skewed_timesteps", action="store_true",
                        help="Use skewed timestep sampling (focus on difficult regions)")

    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval in batches")
    
    return parser


def load_config_from_file(config_path):
    """Load configuration from a JSON or YAML file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return config_dict


def save_config_to_file(args, config_path):
    """Save configuration to a JSON file."""
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)