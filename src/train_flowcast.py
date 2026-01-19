

import logging
import sys, os
from pathlib import Path
import time
import json
import datetime

import torch
from torch.utils.data import DataLoader

from dataset import get_auto_dataset
from training.train_loop import train_epoch
from training.eval_loop import eval_model
from training.load_and_save import load_model, save_model
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from dataset.wrapper import FlowCastWrapperDataset
from models.fluid_unet import FluidDynamicsUNet
from args import Args

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
    args = Args.parse_args()

    # Set up device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load or create data
    if args.data_path is not None:
        logger.info(f"Loading data from {args.data_dir}")
    
    # Create base datasets
    base_dataset_train, base_dataset_val = get_auto_dataset(
    data_dir=args.data_dir,
    data_name='cylinder_geo',
    delta_time=0.1,
    norm_props=True,
    norm_bc=True,
    load_splits=['train', 'dev']
)
    # wrapped dataset for flowcast
    dataset_train = FlowCastWrapperDataset(base_dataset_train)
    logger.info(dataset_train)

    dataset_val = FlowCastWrapperDataset(base_dataset_val)
    logger.info(dataset_val)


    # Create data loaders
    logger.info("Intializing DataLoader")
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,                                                                 
    )

    val_loader = DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = FluidDynamicsUNet(
        in_channels=2,
        out_channels=2,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        num_case_params=args.num_case_params,
        use_fourier_conditioning=args.use_fourier_conditioning,
        num_fourier_freqs=args.num_fourier_freqs,
    ).to(device)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    eff_batch_size = (
        args.batch_size * args.accum_iter
    )

    logger.info(f"Learning rate: {args.lr:.2e}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_scaler = NativeScaler()

    load_model(
        args=args,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if not args.eval_only:
            train_stats = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
        else:
            log_stats = {
                "epoch": epoch,
            }

        if args.output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
           
            eval_stats = eval_model(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})

        if args.output_dir:
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.test_run or args.eval_only:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


if __name__ == "__main__":
    main()