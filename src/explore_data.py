"""Explore training data: visualize the rarest product categories.

For each rare category, shows a cropped example from the training shelf images
alongside the product reference photo (if available).

Usage:
    python src/explore_data.py \
        --coco_dir data/coco \
        --product_dir data/product_images \
        --top_n 20
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_coco(coco_dir: Path) -> dict:
    with open(coco_dir / "annotations.json") as f:
        return json.load(f)


def load_product_metadata(product_dir: Path) -> dict[str, str]:
    """Return mapping of product_name -> product_code from metadata.json."""
    meta_path = product_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return {p["product_name"]: p["product_code"] for p in meta.get("products", [])}


def crop_bbox(image: np.ndarray, bbox: list) -> np.ndarray:
    """Crop a COCO [x, y, w, h] bounding box from an image."""
    x, y, w, h = [int(v) for v in bbox]
    ih, iw = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser(description="Visualize rarest product categories")
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--product_dir", type=str, default="data/product_images")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of rarest categories to display")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    product_dir = Path(args.product_dir)
    coco = load_coco(coco_dir)
    name_to_code = load_product_metadata(product_dir)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Count annotations per category
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Print full frequency table
    print(f"{'ID':>4}  {'Count':>5}  Name")
    print("-" * 70)
    for cat_id, count in cat_counts.most_common():
        print(f"{cat_id:>4}  {count:>5}  {categories[cat_id]}")

    # Categories with zero annotations
    all_cat_ids = set(categories.keys())
    annotated_ids = set(cat_counts.keys())
    zero_cats = all_cat_ids - annotated_ids
    if zero_cats:
        print(f"\nCategories with 0 annotations: {sorted(zero_cats)}")

    print(f"\nTotal categories: {len(categories)}")
    print(f"Categories with <= 3 annotations: "
          f"{sum(1 for c in cat_counts.values() if c <= 3)}")
    print(f"Categories with <= 10 annotations: "
          f"{sum(1 for c in cat_counts.values() if c <= 10)}")

    # Group annotations by image for quick lookup
    img_annotations: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_annotations.setdefault(ann["image_id"], []).append(ann)

    # Get the N rarest categories
    rarest = cat_counts.most_common()[-args.top_n:]
    rarest.reverse()  # least frequent first

    # Include zero-annotation categories at the top
    for cat_id in sorted(zero_cats):
        rarest.insert(0, (cat_id, 0))
    rarest = rarest[:args.top_n]

    # Build visualization grid
    fig, axes = plt.subplots(len(rarest), 2, figsize=(10, 3 * len(rarest)))
    if len(rarest) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Rarest {len(rarest)} Categories", fontsize=14, fontweight="bold")

    for row_idx, (cat_id, count) in enumerate(rarest):
        cat_name = categories[cat_id]
        ax_crop = axes[row_idx, 0]
        ax_ref = axes[row_idx, 1]

        # Find a shelf crop for this category
        shelf_crop = None
        if count > 0:
            for ann in coco["annotations"]:
                if ann["category_id"] == cat_id:
                    img_info = img_id_to_info[ann["image_id"]]
                    img_path = coco_dir / "images" / img_info["file_name"]
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            shelf_crop = crop_bbox(img, ann["bbox"])
                    break

        if shelf_crop is not None and shelf_crop.size > 0:
            ax_crop.imshow(shelf_crop)
            ax_crop.set_title(f"Shelf crop (n={count})", fontsize=9)
        else:
            ax_crop.text(0.5, 0.5, f"No crop\n(n={count})",
                         ha="center", va="center", fontsize=10)
        ax_crop.set_ylabel(f"[{cat_id}] {cat_name[:40]}", fontsize=7, rotation=0,
                           labelpad=120, ha="right", va="center")
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])

        # Find reference product image
        ref_img = None
        product_code = name_to_code.get(cat_name)
        if product_code:
            ref_path = product_dir / product_code / "main.jpg"
            if ref_path.exists():
                ref = cv2.imread(str(ref_path))
                if ref is not None:
                    ref_img = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

        if ref_img is not None:
            ax_ref.imshow(ref_img)
            ax_ref.set_title("Reference photo", fontsize=9)
        else:
            ax_ref.text(0.5, 0.5, "No reference\nimage available",
                        ha="center", va="center", fontsize=10)
        ax_ref.set_xticks([])
        ax_ref.set_yticks([])

    plt.tight_layout()
    plt.savefig("rare_categories.png", dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to rare_categories.png")
    plt.show()


if __name__ == "__main__":
    main()
