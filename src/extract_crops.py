"""Build an ImageFolder classification dataset from shelf crops + product images.

For each annotation in the COCO training data, crops the bounding box (with
padding) and saves it under data/classifier/{class_id}/.  Product reference
images from data/product_images/ are also copied into the matching class
folder, giving the classifier extra training signal for rare categories.

Usage:
    python src/extract_crops.py \
        --coco_dir data/coco \
        --product_dir data/product_images \
        --output_dir data/classifier \
        --pad_pct 0.05
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


def crop_bbox_padded(
    image, bbox: list, pad_pct: float = 0.05
):
    """Crop COCO [x, y, w, h] bbox with percentage-based padding."""
    x, y, w, h = bbox
    ih, iw = image.shape[:2]

    pad_x = w * pad_pct
    pad_y = h * pad_pct

    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(iw, int(x + w + pad_x))
    y2 = min(ih, int(y + h + pad_y))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def main():
    parser = argparse.ArgumentParser(
        description="Extract classification crops from shelf annotations + product images"
    )
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--product_dir", type=str, default="data/product_images")
    parser.add_argument("--output_dir", type=str, default="data/classifier")
    parser.add_argument(
        "--pad_pct", type=float, default=0.05,
        help="Padding around bbox as fraction of bbox size",
    )
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    product_dir = Path(args.product_dir)
    output_dir = Path(args.output_dir)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # --- Load product metadata for name → code mapping ---
    name_to_code: dict[str, str] = {}
    meta_path = product_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for p in meta.get("products", []):
            name_to_code[p["product_name"]] = p["product_code"]

    # --- 1. Extract shelf crops ---
    print("Extracting shelf crops...")
    image_cache: dict[str, any] = {}
    crop_counts: dict[int, int] = {}

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        img_info = img_id_to_info[img_id]
        file_name = img_info["file_name"]

        if file_name not in image_cache:
            img_path = coco_dir / "images" / file_name
            if not img_path.exists():
                continue
            image_cache[file_name] = cv2.imread(str(img_path))

        image = image_cache[file_name]
        if image is None:
            continue

        crop = crop_bbox_padded(image, ann["bbox"], pad_pct=args.pad_pct)
        if crop is None:
            continue

        class_dir = output_dir / str(cat_id)
        class_dir.mkdir(parents=True, exist_ok=True)

        ann_id = ann.get("id", id(ann))
        out_path = class_dir / f"shelf_{img_id}_{ann_id}.jpg"
        cv2.imwrite(str(out_path), crop)
        crop_counts[cat_id] = crop_counts.get(cat_id, 0) + 1

    total_crops = sum(crop_counts.values())
    print(f"  Saved {total_crops} shelf crops across {len(crop_counts)} classes")

    # --- 2. Copy product reference images ---
    print("Copying product reference images...")
    ref_count = 0
    matched_classes = 0

    for cat_id, cat_name in categories.items():
        product_code = name_to_code.get(cat_name)
        if not product_code:
            continue

        ref_dir = product_dir / product_code
        if not ref_dir.exists():
            continue

        class_dir = output_dir / str(cat_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        matched_classes += 1

        for img_file in ref_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            dst = class_dir / f"ref_{product_code}_{img_file.name}"
            if not dst.exists():
                shutil.copy2(img_file, dst)
                ref_count += 1

    print(f"  Copied {ref_count} reference images for {matched_classes} classes")

    # --- 3. Summary ---
    all_classes = sorted(output_dir.iterdir())
    all_classes = [d for d in all_classes if d.is_dir()]

    counts = []
    for d in all_classes:
        n = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
        counts.append((d.name, n))

    counts.sort(key=lambda x: x[1])

    print(f"\nDataset summary:")
    print(f"  Total classes with images: {len(counts)}")
    print(f"  Total images: {sum(c for _, c in counts)}")
    print(f"  Min images per class: {counts[0][1]} (class {counts[0][0]})")
    print(f"  Max images per class: {counts[-1][1]} (class {counts[-1][0]})")
    median_idx = len(counts) // 2
    print(f"  Median images per class: {counts[median_idx][1]}")

    rare = [(name, n) for name, n in counts if n <= 5]
    print(f"  Classes with ≤5 images: {len(rare)}")

    # Save class-to-name mapping for the classifier
    mapping = {str(cat_id): cat_name for cat_id, cat_name in categories.items()}
    mapping_path = output_dir / "class_names.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\nClass name mapping saved to {mapping_path}")
    print("Done!")


if __name__ == "__main__":
    main()
