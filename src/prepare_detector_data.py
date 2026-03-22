"""Prepare class-agnostic (nc=1) YOLO dataset for the detection-only stage.

Reads the existing YOLO labels in data/yolo/ and rewrites them with all class
IDs remapped to 0.  Images are symlinked (not copied) to save disk space.
Generates config_detector.yaml at the project root.

Usage:
    python src/prepare_detector_data.py \
        --src_dir data/yolo \
        --output_dir data/yolo_detector
"""

import argparse
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Remap YOLO labels to nc=1 for detection-only training"
    )
    parser.add_argument(
        "--src_dir", type=str, default="data/yolo",
        help="Source YOLO dataset (with multi-class labels)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/yolo_detector",
        help="Output directory for single-class dataset",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}")
        print("Run src/convert_coco_to_yolo.py first.")
        return

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Link images (avoid duplicating ~GBs of data)
        src_images = src_dir / "images" / split
        for img in src_images.iterdir():
            dst = output_dir / "images" / split / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

        # Remap labels: replace class ID with 0, keep bbox unchanged
        src_labels = src_dir / "labels" / split
        label_count = 0
        ann_count = 0
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

        print(f"{split}: {label_count} label files, {ann_count} annotations → class 0")

    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "product"},
    }
    config_path = Path("config_detector.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nConfig written to {config_path}")
    print("Done! Now train with: python src/train_detector.py")


if __name__ == "__main__":
    main()
