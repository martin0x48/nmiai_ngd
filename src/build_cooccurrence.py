"""Build co-occurrence matrices from COCO training annotations.

Produces a JSON artifact used by the CooccurrenceRescorer at inference time.
Two matrices are computed:
  - Image-level: which categories appear in the same image
  - Spatial: which categories appear near each other (bbox center distance)

Usage:
    python src/build_cooccurrence.py
    python src/build_cooccurrence.py --spatial_threshold 0.15 --output data/cooccurrence.json
"""

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np


def _bbox_center(bbox):
    """Return (cx, cy) from COCO [x, y, w, h]."""
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def build_image_cooccurrence(annotations, num_categories):
    """Build image-level co-occurrence count matrix.

    C[i][j] = number of images containing both category i and category j.
    Diagonal C[i][i] = number of images containing category i.
    """
    by_image = defaultdict(set)
    for ann in annotations:
        by_image[ann["image_id"]].add(ann["category_id"])

    C = np.zeros((num_categories, num_categories), dtype=np.float64)
    for cats in by_image.values():
        cats = sorted(cats)
        for i, a in enumerate(cats):
            C[a][a] += 1
            for b in cats[i + 1 :]:
                C[a][b] += 1
                C[b][a] += 1

    return C


def build_spatial_cooccurrence(annotations, images, num_categories, threshold_pct=0.15):
    """Build spatial co-occurrence: only count pairs within distance threshold.

    Distance threshold is threshold_pct * diagonal of the image.
    """
    img_dims = {img["id"]: (img["width"], img["height"]) for img in images}

    by_image = defaultdict(list)
    for ann in annotations:
        by_image[ann["image_id"]].append(ann)

    C = np.zeros((num_categories, num_categories), dtype=np.float64)
    diag_counts = np.zeros(num_categories, dtype=np.float64)

    for img_id, anns in by_image.items():
        w, h = img_dims.get(img_id, (1, 1))
        diag = sqrt(w * w + h * h)
        threshold = threshold_pct * diag

        centers = [(_bbox_center(a["bbox"]), a["category_id"]) for a in anns]

        # Count category presence for diagonal
        cats_in_img = set(a["category_id"] for a in anns)
        for c in cats_in_img:
            diag_counts[c] += 1

        for i in range(len(centers)):
            ci, cat_i = centers[i]
            for j in range(i + 1, len(centers)):
                cj, cat_j = centers[j]
                dist = sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                if dist <= threshold:
                    C[cat_i][cat_j] += 1
                    C[cat_j][cat_i] += 1

    # Set diagonal to image-level counts (for normalization)
    for c in range(num_categories):
        C[c][c] = diag_counts[c]

    return C


def normalize_matrix(raw_counts, epsilon=0.1):
    """Convert counts to conditional probabilities with Laplace smoothing.

    P(j|i) = (C[i][j] + epsilon) / (C[i][i] + epsilon * num_categories)
    """
    n = raw_counts.shape[0]
    probs = np.zeros_like(raw_counts)
    for i in range(n):
        denom = raw_counts[i][i] + epsilon * n
        if denom > 0:
            for j in range(n):
                probs[i][j] = (raw_counts[i][j] + epsilon) / denom
    return probs


def main():
    parser = argparse.ArgumentParser(description="Build co-occurrence matrices")
    parser.add_argument(
        "--annotations", default="data/coco/annotations.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--output", default="data/cooccurrence.json",
        help="Output path for co-occurrence artifact",
    )
    parser.add_argument(
        "--spatial_threshold", type=float, default=0.15,
        help="Spatial distance threshold as fraction of image diagonal",
    )
    args = parser.parse_args()

    with open(args.annotations) as f:
        coco = json.load(f)

    annotations = coco["annotations"]
    images = coco["images"]
    categories = coco["categories"]
    cat_ids = sorted(c["id"] for c in categories)
    num_categories = max(cat_ids) + 1  # 0-indexed, so 356

    print(f"Building co-occurrence from {len(annotations)} annotations, "
          f"{len(images)} images, {len(categories)} categories")

    # Image-level co-occurrence
    image_cooc = build_image_cooccurrence(annotations, num_categories)
    image_cooc_norm = normalize_matrix(image_cooc)
    print(f"  Image-level: {int(np.sum(image_cooc > 0))} non-zero entries")

    # Spatial co-occurrence
    spatial_cooc = build_spatial_cooccurrence(
        annotations, images, num_categories, args.spatial_threshold
    )
    spatial_cooc_norm = normalize_matrix(spatial_cooc)
    print(f"  Spatial (threshold={args.spatial_threshold}): "
          f"{int(np.sum(spatial_cooc > 0))} non-zero entries")

    # Category image counts (diagonal of image cooc)
    cat_image_counts = [int(image_cooc[i][i]) for i in range(num_categories)]

    output = {
        "category_ids": cat_ids,
        "image_cooccurrence": image_cooc_norm.tolist(),
        "spatial_cooccurrence": spatial_cooc_norm.tolist(),
        "category_image_counts": cat_image_counts,
        "num_images": len(images),
        "metadata": {
            "num_categories": num_categories,
            "spatial_threshold_pct": args.spatial_threshold,
            "annotations_path": args.annotations,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    file_size = Path(args.output).stat().st_size / 1024
    print(f"\nSaved to {args.output} ({file_size:.0f} KB)")


if __name__ == "__main__":
    main()
