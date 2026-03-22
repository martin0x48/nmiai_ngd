"""Train EfficientNet-V2-M classifier on extracted product crops.

Expects an ImageFolder layout produced by src/extract_crops.py:
    data/classifier/{class_id}/*.jpg

Handles class imbalance via WeightedRandomSampler.

Usage:
    python src/train_classifier.py \
        --data_dir data/classifier \
        --epochs 50 \
        --batch 32 \
        --imgsz 224

Prerequisites:
    pip install lightning timm wandb torchvision
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import lightning as L
import timm
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class ProductClassifier(L.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, epochs: int = 50):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            "tf_efficientnetv2_m", pretrained=True, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(1) == labels).float().mean()
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=1e-6
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def build_transforms(imgsz: int, is_train: bool):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
                ),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.2),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(imgsz * 1.14)),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def stratified_split(dataset, val_ratio=0.2, seed=42):
    """Split dataset indices with stratification by class."""
    rng = random.Random(seed)
    class_to_indices: dict[int, list] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    for _, indices in class_to_indices.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return train_indices, val_indices


def build_dataloaders(
    data_dir: Path, imgsz: int, batch: int
) -> tuple[DataLoader, DataLoader, int, dict]:
    """Build train/val DataLoaders with stratified split and weighted sampling.

    Returns (train_loader, val_loader, num_classes, idx_to_class).
    """
    full_dataset = datasets.ImageFolder(str(data_dir))
    num_classes = len(full_dataset.classes)
    print(f"  Classes: {num_classes}, Total images: {len(full_dataset)}")

    idx_to_class = {v: int(k) for k, v in full_dataset.class_to_idx.items()}
    train_indices, val_indices = stratified_split(full_dataset, val_ratio=0.2)
    print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")

    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(str(data_dir), transform=build_transforms(imgsz, True)),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(str(data_dir), transform=build_transforms(imgsz, False)),
        val_indices,
    )

    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes, idx_to_class


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-V2-M product classifier"
    )
    parser.add_argument("--data_dir", type=str, default="data/classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="runs/classifier")
    parser.add_argument("--wandb-project", type=str, default="norgesgruppen-cv")
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes, idx_to_class = build_dataloaders(
        data_dir, args.imgsz, args.batch
    )

    mapping_path = output_dir / "idx_to_class.json"
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f)

    model = ProductClassifier(
        num_classes=num_classes, lr=args.lr, epochs=args.epochs
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name or "classifier-effnetv2m",
        config={
            "model": "tf_efficientnetv2_m",
            "stage": "classifier",
            "num_classes": num_classes,
            "imgsz": args.imgsz,
        },
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best_classifier",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = ProductClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
    best_path = output_dir / "best_classifier.pt"
    torch.save(best_model.model.state_dict(), best_path)

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {checkpoint_cb.best_model_score:.4f}")
    print(f"Best weights: {best_path}")
    print(f"Index mapping: {mapping_path}")


if __name__ == "__main__":
    main()
