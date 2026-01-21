from pathlib import Path
import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_model(args, model, optimizer, lr_schedule, loss_scaler, epoch):
    """Save model checkpoint (single-process version)."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_paths = [
        output_dir / f"checkpoint-{epoch}.pth",
        output_dir / "checkpoint.pth",  
    ]
    
    to_save = {
        "model": model.state_dict(),  
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "args": args.as_dict(),
    }
    
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    
    for checkpoint_path in checkpoint_paths:
        torch.save(to_save, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


def load_model(args, model, optimizer, lr_schedule, loss_scaler):
    """Load model checkpoint (single-process version)."""
    if not args.resume:
        return
    
    if not os.path.exists(args.resume):
        logger.warning(f"{args.resume} not found. Starting fresh.")
        return
    
    logger.info(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location="cpu")
    
    model.load_state_dict(checkpoint["model"])
    
    if "optimizer" in checkpoint and not (hasattr(args, "eval") and args.eval):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_schedule.load_state_dict(checkpoint["lr_schedule"])
        args.start_epoch = checkpoint["epoch"] + 1
        
        if "scaler" in checkpoint and loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        
        logger.info(f"Resuming from epoch {args.start_epoch}")
