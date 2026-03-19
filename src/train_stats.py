# script to get training data statistics: mean, std

import torch
import argparse
from dataset import get_auto_dataset
from dataset.wrapper import MultiStepFlowCastDataset
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Calculate Training Dataset Statistics')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--data_name', type=str, help='Problem name. Ex: cylinder_geo', required=True)
    parser.add_argument('--norm_bc', action='store_true', default=True,  help='normalize boundary conditions')
    parser.add_argument('--norm_props', action='store_true', default=True, help='normalize fluid properties')
    parser.add_argument('--delta_time', type=float, default=0.1, help='delta time used during training.')
    parser.add_argument('--save_path', type=Path, default='./train_stats.pt')
     
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    dataset_train, _ , _ = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_bc=args.norm_bc,
        norm_props=args.norm_props,
        load_splits=['train']
    )

    dataset_train = MultiStepFlowCastDataset(base_dataset=dataset_train, normalize=True, norm_stats=None)
    train_stats = dataset_train.get_norm_stats()
    
    torch.save(train_stats, args.save_path)
    print(f"training statistics saved to {args.save_path}")




if __name__ == '__main__':
    main()