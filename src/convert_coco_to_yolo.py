"""Convert COCO-format annotations to YOLO format and generate config.yaml.

Usage:
    python src/convert_coco_to_yolo.py \
        --coco_dir data/coco \
        --output_dir data/yolo \
        --val_split 0.2
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml


def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> list:
    """Convert COCO [x, y, w, h] (pixels) to YOLO [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return [cx, cy, nw, nh]


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format")
    parser.add_argument("--coco_dir", type=str, default="data/coco",
                        help="Path to COCO dataset (contains images/ and annotations.json)")
    parser.add_argument("--output_dir", type=str, default="data/yolo",
                        help="Output directory for YOLO-formatted dataset")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of images to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    annotations_path = coco_dir / "annotations.json"
    images_dir = coco_dir / "images"

    with open(annotations_path) as f:
        coco = json.load(f)

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    nc = max(categories.keys()) + 1

    # Group annotations by image_id
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_annotations.setdefault(ann["image_id"], []).append(ann)

    # Split images into train/val
    image_ids = sorted(img_id_to_info.keys())
    random.seed(args.seed)
    random.shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * args.val_split))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    print(f"Total images: {len(image_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Categories: {nc}")
    print(f"Total annotations: {len(coco['annotations'])}")

    # Create output directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert and write
    stats = {"train": 0, "val": 0}
    for img_id, img_info in img_id_to_info.items():
        split = "val" if img_id in val_ids else "train"
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Copy image
        src_img = images_dir / file_name
        if not src_img.exists():
            print(f"WARNING: {src_img} not found, skipping")
            continue
        dst_img = output_dir / "images" / split / file_name
        shutil.copy2(src_img, dst_img)

        # Write label file (use .txt with same stem, always .txt regardless of .jpg/.jpeg)
        label_name = Path(file_name).stem + ".txt"
        label_path = output_dir / "labels" / split / label_name

        anns = img_annotations.get(img_id, [])
        with open(label_path, "w") as lf:
            for ann in anns:
                yolo_bbox = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
                cat_id = ann["category_id"]
                lf.write(f"{cat_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                         f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

        stats[split] += 1

    print(f"Written: {stats['train']} train, {stats['val']} val images+labels")

    # Generate config.yaml
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": {i: categories.get(i, f"class_{i}") for i in range(nc)},
    }
    config_path = Path("config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Config written to {config_path}")
    print("Done!")


if __name__ == "__main__":
    main()
