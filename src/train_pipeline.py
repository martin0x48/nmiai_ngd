"""End-to-end two-stage training pipeline.

Concatenates the COCO and Meny datasets (kept in separate folders) and runs
all stages sequentially:
  0. Convert Meny COCO → YOLO   (flatten subdir images, same category IDs)
  1. Prepare detector data       (remap YOLO labels to nc=1 from both sources)
  2. Train detector              (YOLOv8x, nc=1)
  3. Extract classifier crops    (shelf annotations from both sources + product refs)
  4. Train classifier            (EfficientNet-V2-M)

Usage:
    python src/train_pipeline.py                          # all defaults
    python src/train_pipeline.py --det_epochs 150 --clf_epochs 50
    python src/train_pipeline.py --skip_data_prep         # skip steps 0, 1, 3

Prerequisites:
    pip install ultralytics==8.1.0 lightning timm wandb opencv-python-headless pyyaml
"""

import argparse
import functools
import json
import random
import shutil
from pathlib import Path

import cv2
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

import wandb
import yaml
from ultralytics import YOLO

from train_classifier import ProductClassifier, build_dataloaders

# ──────────────────────────────────────────────────────────────────────
# Stage 0: Convert Meny COCO annotations to YOLO format
# ──────────────────────────────────────────────────────────────────────


def _coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def convert_meny_to_yolo(
    meny_dir: Path, output_dir: Path, val_split: float = 0.2, seed: int = 42
) -> bool:
    """Convert Meny COCO annotations to YOLO format with flattened filenames.

    Meny images live in subdirs (1/, 2/, 3/, Frokost/). Filenames are flattened
    by replacing '/' with '_' to avoid collisions (e.g. 1/IMG_7760.jpeg →
    1_IMG_7760.jpeg).

    Returns True if conversion ran, False if annotations file was not found.
    """
    print("\n" + "=" * 60)
    print("STAGE 0: Converting Meny COCO → YOLO")
    print("=" * 60)

    ann_path = meny_dir / "annotations_coco.json"
    if not ann_path.exists():
        print(f"  No Meny annotations at {ann_path}, skipping")
        return False

    with open(ann_path) as f:
        coco = json.load(f)

    img_id_to_info = {img["id"]: img for img in coco["images"]}

    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_annotations.setdefault(ann["image_id"], []).append(ann)

    image_ids = sorted(img_id_to_info.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * val_split))
    val_ids = set(image_ids[:val_count])

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0}
    ann_stats = {"train": 0, "val": 0}

    for img_id in image_ids:
        img_info = img_id_to_info[img_id]
        split = "val" if img_id in val_ids else "train"
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

        src_img = meny_dir / file_name
        if not src_img.exists():
            print(f"  WARNING: {src_img} not found, skipping")
            continue

        flat_name = file_name.replace("/", "_")
        dst_img = output_dir / "images" / split / flat_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        label_name = Path(flat_name).stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        anns = img_annotations.get(img_id, [])
        with open(label_path, "w") as lf:
            for ann in anns:
                cx, cy, nw, nh = _coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                lf.write(
                    f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                )
                ann_stats[split] += 1

        stats[split] += 1

    for split in ("train", "val"):
        print(
            f"  {split}: {stats[split]} images, {ann_stats[split]} annotations"
        )
    return True


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Prepare detector data
# ──────────────────────────────────────────────────────────────────────


def _remap_single_source(src_dir: Path, output_dir: Path):
    """Remap a single YOLO source to nc=1 (all class IDs → 0)."""
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        src_images = src_dir / "images" / split
        for img in src_images.iterdir():
            dst = output_dir / "images" / split / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

        src_labels = src_dir / "labels" / split
        label_count, ann_count = 0, 0
        for label_file in src_labels.glob("*.txt"):
            lines = label_file.read_text().strip().splitlines()
            remapped = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    remapped.append(" ".join(parts))
                    ann_count += 1
            dst_label = output_dir / "labels" / split / label_file.name
            dst_label.write_text("\n".join(remapped) + "\n" if remapped else "")
            label_count += 1

        print(
            f"    {split}: {label_count} label files, {ann_count} annotations → class 0"
        )


def prepare_detector_data(sources: list[tuple[Path, Path]]):
    """Remap YOLO class IDs to 0 from multiple sources, write combined config.

    Each (src_dir, output_dir) pair is processed independently. The resulting
    config_detector.yaml lists all output dirs so YOLO trains on the
    concatenation of every source.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Preparing detector data (nc=1)")
    print("=" * 60)

    processed = []
    for src_dir, output_dir in sources:
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found, skipping")
            continue
        print(f"\n  {src_dir} → {output_dir}")
        _remap_single_source(src_dir, output_dir)
        processed.append(output_dir)

    if not processed:
        raise FileNotFoundError(
            "No valid YOLO source directories found.\n"
            "Run src/convert_coco_to_yolo.py first."
        )

    train_paths = [str((d / "images" / "train").resolve()) for d in processed]
    val_paths = [str((d / "images" / "val").resolve()) for d in processed]

    config = {
        "path": ".",
        "train": train_paths if len(train_paths) > 1 else train_paths[0],
        "val": val_paths if len(val_paths) > 1 else val_paths[0],
        "nc": 1,
        "names": {0: "product"},
    }
    config_path = Path("config_detector.yaml")
    with open(config_path, "w") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    total_train = sum(
        len(list((d / "images" / "train").iterdir())) for d in processed
    )
    total_val = sum(
        len(list((d / "images" / "val").iterdir())) for d in processed
    )
    print(f"\n  Combined: {total_train} train + {total_val} val images")
    print(f"  Config written to {config_path}")
    return config_path


# ──────────────────────────────────────────────────────────────────────
# Stage 2: Train detector
# ──────────────────────────────────────────────────────────────────────


def _make_detector_callback(checkpoint_interval: int, saved_checkpoints: list):
    """Callback that logs metrics and saves periodic checkpoints to disk.

    Avoids wandb.log_artifact inside training because ultralytics may
    call wandb.finish() in its on_train_end hook, killing the run.
    Artifacts are uploaded after training completes instead.
    """

    def on_fit_epoch_end(trainer):
        if wandb.run is None:
            return
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
                saved_checkpoints.append((epoch, str(dst), metrics.copy()))
                print(f"  ↳ Checkpoint saved: {dst.name}")

    return on_fit_epoch_end


def train_detector(
    config_path: str,
    model_name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    patience: int,
    wandb_project: str,
    checkpoint_interval: int,
) -> Path:
    """Train YOLOv8x as class-agnostic detector. Returns path to best.pt."""
    print("\n" + "=" * 60)
    print("STAGE 2: Training detector (YOLOv8x, nc=1)")
    print("=" * 60)

    train_cfg = dict(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        close_mosaic=15,
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
        project=wandb_project,
        name="detector-nc1",
        config={"model": model_name, "stage": "detector", **train_cfg},
    )

    saved_checkpoints: list[tuple] = []

    model = YOLO(model_name)
    for event in list(model.callbacks):
        model.callbacks[event] = [
            fn
            for fn in model.callbacks[event]
            if "wb" not in getattr(fn, "__module__", "")
        ]
    model.add_callback(
        "on_fit_epoch_end",
        _make_detector_callback(checkpoint_interval, saved_checkpoints),
    )

    results = model.train(**train_cfg)
    best_path = Path(results.save_dir) / "weights" / "best.pt"

    if wandb.run is None:
        run = wandb.init(
            project=wandb_project,
            name="detector-nc1-artifacts",
            config={"model": model_name, "stage": "detector", **train_cfg},
        )

    for epoch, ckpt_path, metrics in saved_checkpoints:
        art = wandb.Artifact(
            f"detector-ckpt-{epoch}",
            type="model",
            metadata={"epoch": epoch, **metrics},
        )
        art.add_file(ckpt_path)
        wandb.log_artifact(art)

    if best_path.exists():
        art = wandb.Artifact(
            "detector-best", type="model", metadata={"source_run": run.id}
        )
        art.add_file(str(best_path))
        wandb.log_artifact(art)

    wandb.finish()

    print(f"  Detector best weights: {best_path}")
    return best_path


# ──────────────────────────────────────────────────────────────────────
# Stage 3: Extract classifier crops
# ──────────────────────────────────────────────────────────────────────


def _crop_bbox_padded(image, bbox, pad_pct=0.05):
    x, y, w, h = bbox
    ih, iw = image.shape[:2]
    pad_x, pad_y = w * pad_pct, h * pad_pct
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(iw, int(x + w + pad_x))
    y2 = min(ih, int(y + h + pad_y))
    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def _extract_crops_from_source(
    annotations_path: Path,
    images_base: Path,
    output_dir: Path,
    pad_pct: float,
    source_tag: str,
) -> tuple[dict[int, str], dict[int, int]]:
    """Extract crops from a single COCO-format annotation file.

    Returns (categories dict, per-class crop counts).
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    image_cache: dict[str, any] = {}
    crop_counts: dict[int, int] = {}

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        img_info = img_id_to_info[ann["image_id"]]
        file_name = img_info["file_name"]

        if file_name not in image_cache:
            img_path = images_base / file_name
            if not img_path.exists():
                continue
            image_cache[file_name] = cv2.imread(str(img_path))

        image = image_cache[file_name]
        if image is None:
            continue

        crop = _crop_bbox_padded(image, ann["bbox"], pad_pct=pad_pct)
        if crop is None:
            continue

        class_dir = output_dir / str(cat_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        ann_id = ann.get("id", id(ann))
        cv2.imwrite(
            str(class_dir / f"{source_tag}_{ann['image_id']}_{ann_id}.jpg"), crop
        )
        crop_counts[cat_id] = crop_counts.get(cat_id, 0) + 1

    total = sum(crop_counts.values())
    print(f"    [{source_tag}] {total} crops across {len(crop_counts)} classes")
    return categories, crop_counts


def extract_crops(
    coco_sources: list[tuple[Path, Path, str]],
    product_dir: Path,
    output_dir: Path,
    pad_pct: float,
):
    """Crop shelf annotations from multiple COCO sources + product reference images.

    coco_sources: list of (annotations_path, images_base_dir, source_tag) tuples.
    Crops from all sources are written into the same ImageFolder layout under
    output_dir, keyed by category_id.
    """
    print("\n" + "=" * 60)
    print("STAGE 3: Extracting classifier crops")
    print("=" * 60)

    all_categories: dict[int, str] = {}

    for ann_path, img_base, tag in coco_sources:
        if not ann_path.exists():
            print(f"    [{tag}] annotations not found at {ann_path}, skipping")
            continue
        cats, _ = _extract_crops_from_source(ann_path, img_base, output_dir, pad_pct, tag)
        all_categories.update(cats)

    name_to_code: dict[str, str] = {}
    meta_path = product_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for p in meta.get("products", []):
            name_to_code[p["product_name"]] = p["product_code"]

    ref_count, matched = 0, 0
    for cat_id, cat_name in all_categories.items():
        product_code = name_to_code.get(cat_name)
        if not product_code:
            continue
        ref_dir = product_dir / product_code
        if not ref_dir.exists():
            continue
        class_dir = output_dir / str(cat_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        matched += 1
        for img_file in ref_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            dst = class_dir / f"ref_{product_code}_{img_file.name}"
            if not dst.exists():
                shutil.copy2(img_file, dst)
                ref_count += 1

    print(f"  {ref_count} reference images for {matched} classes")

    mapping = {str(cat_id): cat_name for cat_id, cat_name in all_categories.items()}
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
    counts = [
        (d.name, len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))) for d in dirs
    ]
    counts.sort(key=lambda x: x[1])
    print(f"  Total classes: {len(counts)}, images: {sum(c for _, c in counts)}")
    print(
        f"  Min/class: {counts[0][1]}, Max/class: {counts[-1][1]}, "
        f"Median: {counts[len(counts) // 2][1]}"
    )


# ──────────────────────────────────────────────────────────────────────
# Stage 4: Train classifier
# ──────────────────────────────────────────────────────────────────────


def train_classifier(
    data_dir: Path,
    output_dir: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    lr: float,
    wandb_project: str,
) -> tuple[Path, Path]:
    """Train EfficientNet-V2-M. Returns (best_weights_path, idx_to_class_path)."""
    print("\n" + "=" * 60)
    print("STAGE 4: Training classifier (EfficientNet-V2-M)")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes, idx_to_class = build_dataloaders(
        data_dir, imgsz, batch
    )

    mapping_path = output_dir / "idx_to_class.json"
    with open(mapping_path, "w") as f:
        json.dump(idx_to_class, f)

    model = ProductClassifier(num_classes=num_classes, lr=lr, epochs=epochs)

    wandb_logger = WandbLogger(
        project=wandb_project,
        name="classifier-effnetv2m",
        config={
            "model": "tf_efficientnetv2_m",
            "stage": "classifier",
            "num_classes": num_classes,
            "imgsz": imgsz,
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
        max_epochs=epochs,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = ProductClassifier.load_from_checkpoint(checkpoint_cb.best_model_path)
    best_path = output_dir / "best_classifier.pt"
    torch.save(best_model.model.state_dict(), best_path)

    print(f"  Best val accuracy: {checkpoint_cb.best_model_score:.4f}")
    print(f"  Weights: {best_path}")
    return best_path, mapping_path


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end two-stage training: detector + classifier"
    )

    parser.add_argument("--yolo_dir", type=str, default="data/yolo")
    parser.add_argument("--detector_dir", type=str, default="data/yolo_detector")
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--product_dir", type=str, default="data/product_images")
    parser.add_argument("--classifier_data_dir", type=str, default="data/classifier")
    parser.add_argument("--classifier_output_dir", type=str, default="runs/classifier")

    parser.add_argument("--meny_dir", type=str, default="data/meny")
    parser.add_argument("--yolo_meny_dir", type=str, default="data/yolo_meny")
    parser.add_argument(
        "--detector_meny_dir", type=str, default="data/yolo_meny_detector"
    )

    parser.add_argument("--det_model", type=str, default="yolov8x.pt")
    parser.add_argument("--det_epochs", type=int, default=150)
    parser.add_argument("--det_batch", type=int, default=4)
    parser.add_argument("--det_imgsz", type=int, default=1280)
    parser.add_argument("--det_patience", type=int, default=30)
    parser.add_argument("--det_checkpoint_interval", type=int, default=25)

    parser.add_argument("--clf_epochs", type=int, default=50)
    parser.add_argument("--clf_batch", type=int, default=32)
    parser.add_argument("--clf_imgsz", type=int, default=224)
    parser.add_argument("--clf_lr", type=float, default=1e-3)
    parser.add_argument("--crop_pad_pct", type=float, default=0.05)

    parser.add_argument("--wandb-project", type=str, default="norgesgruppen-cv")
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        help="Skip stages 0, 1, and 3 (data already prepared)",
    )
    parser.add_argument(
        "--skip_detector",
        action="store_true",
        help="Skip stages 0, 1, and 2 (detector already trained)",
    )
    parser.add_argument(
        "--skip_classifier",
        action="store_true",
        help="Skip stages 3 and 4 (classifier already trained)",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default=None,
        help="Path to existing detector best.pt (when skipping detector training)",
    )

    args = parser.parse_args()

    yolo_dir = Path(args.yolo_dir)
    detector_dir = Path(args.detector_dir)
    coco_dir = Path(args.coco_dir)
    product_dir = Path(args.product_dir)
    clf_data_dir = Path(args.classifier_data_dir)
    clf_output_dir = Path(args.classifier_output_dir)
    meny_dir = Path(args.meny_dir)
    yolo_meny_dir = Path(args.yolo_meny_dir)
    detector_meny_dir = Path(args.detector_meny_dir)
    wandb_project = getattr(args, "wandb_project", "norgesgruppen-cv")

    detector_best = Path(args.detector_weights) if args.detector_weights else None

    # Stage 0: Convert Meny COCO → YOLO (needed for detector)
    if not args.skip_data_prep and not args.skip_detector:
        convert_meny_to_yolo(meny_dir, yolo_meny_dir)
    else:
        print("\n⏭  Skipping stage 0 (Meny → YOLO conversion)")

    # Stage 1: Prepare detector data from both sources
    if not args.skip_data_prep and not args.skip_detector:
        detector_sources = [(yolo_dir, detector_dir)]
        if yolo_meny_dir.exists() and (yolo_meny_dir / "images").exists():
            detector_sources.append((yolo_meny_dir, detector_meny_dir))
        prepare_detector_data(detector_sources)
    else:
        print("\n⏭  Skipping stage 1 (detector data prep)")

    # Stage 2: Train detector
    if not args.skip_detector:
        detector_best = train_detector(
            config_path="config_detector.yaml",
            model_name=args.det_model,
            epochs=args.det_epochs,
            batch=args.det_batch,
            imgsz=args.det_imgsz,
            patience=args.det_patience,
            wandb_project=wandb_project,
            checkpoint_interval=args.det_checkpoint_interval,
        )
    else:
        print("\n⏭  Skipping stage 2 (detector training)")
        if detector_best is None:
            detector_best = Path("detector.pt")

    # Stage 3: Extract classifier crops from both COCO sources
    if not args.skip_data_prep and not args.skip_classifier:
        coco_sources = [
            (coco_dir / "annotations.json", coco_dir / "images", "shelf"),
        ]
        meny_ann = meny_dir / "annotations_coco.json"
        if meny_ann.exists():
            coco_sources.append((meny_ann, meny_dir, "meny"))
        extract_crops(coco_sources, product_dir, clf_data_dir, args.crop_pad_pct)
    else:
        print("\n⏭  Skipping stage 3 (crop extraction)")

    # Stage 4: Train classifier
    if not args.skip_classifier:
        clf_best, clf_mapping = train_classifier(
            data_dir=clf_data_dir,
            output_dir=clf_output_dir,
            epochs=args.clf_epochs,
            batch=args.clf_batch,
            imgsz=args.clf_imgsz,
            lr=args.clf_lr,
            wandb_project=wandb_project,
        )
    else:
        print("\n⏭  Skipping stage 4 (classifier training)")
        clf_best = clf_output_dir / "best_classifier.pt"
        clf_mapping = clf_output_dir / "idx_to_class.json"

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Detector weights:   {detector_best}")
    print(f"  Classifier weights: {clf_best}")
    print(f"  Class mapping:      {clf_mapping}")
    print("\nNext steps:")
    print("  1. Evaluate:")
    print("     python src/evaluate.py \\")
    print(f"         --detector {detector_best} \\")
    print(f"         --classifier {clf_best} \\")
    print(f"         --classifier_mapping {clf_mapping} \\")
    print("         --save_images")
    print("\n  2. Package submission:")
    print("     python src/package_submission.py \\")
    print(f"         --detector {detector_best} \\")
    print(f"         --classifier {clf_best} \\")
    print(f"         --classifier_mapping {clf_mapping}")


if __name__ == "__main__":
    main()
