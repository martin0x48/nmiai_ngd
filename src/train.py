"""Train YOLOv8x on the converted grocery shelf dataset.

Usage:
    python src/train.py [--config config.yaml] [--epochs 100] [--batch 4] [--imgsz 1280]

Prerequisites:
    1. pip install ultralytics==8.1.0 wandb
    2. Run src/convert_coco_to_yolo.py first to generate data/yolo/ and config.yaml
    3. wandb login  (one-time auth setup)
"""

import argparse
import functools
import shutil
from pathlib import Path

import torch

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

import wandb
from ultralytics import YOLO


def _make_checkpoint_callback(checkpoint_interval: int):
    """Return a callback that saves and uploads a checkpoint every N epochs."""

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        metrics = {k: float(v) for k, v in trainer.metrics.items()}
        metrics["lr/pg0"] = trainer.optimizer.param_groups[0]["lr"]
        wandb.log(metrics, step=epoch)

        if checkpoint_interval <= 0:
            return

        if epoch % checkpoint_interval == 0:
            src = Path(trainer.save_dir) / "weights" / "last.pt"
            if src.exists():
                dst = Path(trainer.save_dir) / "weights" / f"epoch_{epoch}.pt"
                shutil.copy2(src, dst)
                art = wandb.Artifact(
                    f"checkpoint-epoch-{epoch}",
                    type="model",
                    metadata={"epoch": epoch, **metrics},
                )
                art.add_file(str(dst))
                wandb.log_artifact(art)
                print(f"  ↳ Checkpoint saved & uploaded: {dst.name}")

    return on_fit_epoch_end


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8x on grocery shelf data")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to YOLO dataset config"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8x.pt", help="Pretrained model to start from"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="norgesgruppen-cv", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if omitted)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N epochs (0 to disable)",
    )
    args = parser.parse_args()

    train_cfg = dict(
        data=args.config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        degrees=0.0,
        shear=0.0,
    )

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "model": args.model,
            "checkpoint_interval": args.checkpoint_interval,
            **train_cfg,
        },
    )

    model = YOLO(args.model)

    # Remove ultralytics' built-in wandb callbacks to avoid duplicate logging
    for event in list(model.callbacks):
        model.callbacks[event] = [
            fn
            for fn in model.callbacks[event]
            if "wb" not in getattr(fn, "__module__", "")
        ]

    model.add_callback(
        "on_fit_epoch_end",
        _make_checkpoint_callback(args.checkpoint_interval),
    )

    results = model.train(**train_cfg)

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        art = wandb.Artifact(
            "best-model",
            type="model",
            metadata={"source_run": run.id},
        )
        art.add_file(str(best_path))
        wandb.log_artifact(art)
        print(f"\nBest model uploaded to W&B as artifact 'best-model'")

    wandb.finish()

    print(f"\nTraining complete!")
    print(f"Best weights: {best_path}")
    print(f"Results dir:  {results.save_dir}")
    print(f"W&B run:      {run.url}")


if __name__ == "__main__":
    main()
