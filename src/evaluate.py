"""Evaluate the two-stage pipeline on the validation set.

Runs the detector + classifier on val images and computes the competition
metric: 0.7 * detection_mAP + 0.3 * classification_mAP.

Usage (two-stage):
    python src/evaluate.py \
        --detector runs/detect/train/weights/best.pt \
        --classifier runs/classifier/best_classifier.pt \
        --classifier_mapping runs/classifier/idx_to_class.json

Usage (single-stage, backward compatible):
    python src/evaluate.py --model best.pt
"""

import argparse
import functools
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

from ultralytics import YOLO

CLASSIFIER_IMGSZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_iou(box_a: list, box_b: list) -> float:
    """Compute IoU between two COCO-format [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list, recalls: list) -> float:
    """Compute Average Precision using 11-point interpolation."""
    if not precisions:
        return 0.0

    recall_levels = np.linspace(0, 1, 11)
    ap = 0.0
    for r_level in recall_levels:
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= r_level]
        ap += max(precisions_at_recall) if precisions_at_recall else 0.0
    return ap / 11


def evaluate_predictions(predictions: list, ground_truths: list, iou_threshold: float = 0.5):
    """
    Evaluate predictions against ground truth.
    Returns detection_mAP (category-agnostic) and classification_mAP.
    """
    gt_by_image: dict[int, list] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt)

    pred_by_image: dict[int, list] = {}
    for pred in predictions:
        pred_by_image.setdefault(pred["image_id"], []).append(pred)

    # --- Detection mAP (category-agnostic) ---
    all_det_scores = []
    total_gt = len(ground_truths)

    for img_id in gt_by_image:
        gts = gt_by_image[img_id]
        preds = sorted(pred_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt = set()

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                all_det_scores.append((pred["score"], True))
                matched_gt.add(best_gt_idx)
            else:
                all_det_scores.append((pred["score"], False))

    all_det_scores.sort(key=lambda x: -x[0])
    det_precisions, det_recalls = [], []
    tp_cum, fp_cum = 0, 0
    for _, is_tp in all_det_scores:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        det_precisions.append(tp_cum / (tp_cum + fp_cum))
        det_recalls.append(tp_cum / total_gt)

    detection_ap = compute_ap(det_precisions, det_recalls)

    # --- Classification mAP (per-category) ---
    all_cat_ids = set(gt["category_id"] for gt in ground_truths)
    per_cat_aps = []

    for cat_id in all_cat_ids:
        cat_gts = [gt for gt in ground_truths if gt["category_id"] == cat_id]
        cat_gt_by_image: dict[int, list] = {}
        for gt in cat_gts:
            cat_gt_by_image.setdefault(gt["image_id"], []).append(gt)

        cat_preds = [p for p in predictions if p["category_id"] == cat_id]
        cat_preds.sort(key=lambda p: -p["score"])

        total_cat_gt = len(cat_gts)
        if total_cat_gt == 0:
            continue

        matched_by_image: dict[int, set] = {}
        scores = []

        for pred in cat_preds:
            img_id = pred["image_id"]
            gts_for_img = cat_gt_by_image.get(img_id, [])
            matched = matched_by_image.setdefault(img_id, set())

            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts_for_img):
                if idx in matched:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx >= 0:
                scores.append((pred["score"], True))
                matched.add(best_idx)
            else:
                scores.append((pred["score"], False))

        precisions, recalls = [], []
        tp_c, fp_c = 0, 0
        for _, is_tp in scores:
            if is_tp:
                tp_c += 1
            else:
                fp_c += 1
            precisions.append(tp_c / (tp_c + fp_c))
            recalls.append(tp_c / total_cat_gt)

        per_cat_aps.append(compute_ap(precisions, recalls))

    classification_map = np.mean(per_cat_aps) if per_cat_aps else 0.0
    return detection_ap, classification_map


def draw_predictions(image: np.ndarray, predictions: list, categories: dict) -> np.ndarray:
    """Draw bounding boxes and labels on an image."""
    vis = image.copy()
    for pred in predictions:
        x, y, w, h = [int(v) for v in pred["bbox"]]
        score = pred["score"]
        cat_id = pred["category_id"]
        label = f"{categories.get(cat_id, cat_id)}: {score:.2f}"

        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(label[:30], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw, y), (0, 255, 0), -1)
        cv2.putText(vis, label[:30], (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return vis


def load_classifier(weights_path, mapping_path, device):
    with open(mapping_path) as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): int(v) for k, v in idx_to_class.items()}

    model = timm.create_model(
        "tf_efficientnetv2_m", pretrained=False, num_classes=len(idx_to_class)
    )
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model, idx_to_class


def classify_crops(pil_image, boxes_xyxy, classifier, idx_to_class, device, pad_pct=0.05):
    """Classify a batch of detection crops, return (category_ids, class_scores)."""
    clf_transform = transforms.Compose([
        transforms.Resize(int(CLASSIFIER_IMGSZ * 1.14)),
        transforms.CenterCrop(CLASSIFIER_IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    iw, ih = pil_image.size
    tensors = []
    for (x1, y1, x2, y2) in boxes_xyxy:
        bw, bh = x2 - x1, y2 - y1
        cx1 = max(0, int(x1 - bw * pad_pct))
        cy1 = max(0, int(y1 - bh * pad_pct))
        cx2 = min(iw, int(x2 + bw * pad_pct))
        cy2 = min(ih, int(y2 + bh * pad_pct))
        crop = pil_image.crop((cx1, cy1, cx2, cy2))
        tensors.append(clf_transform(crop))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.softmax(logits, dim=1)
        scores, indices = probs.max(dim=1)

    cat_ids = [idx_to_class.get(idx.item(), 0) for idx in indices]
    cls_scores = scores.tolist()
    return cat_ids, cls_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate two-stage pipeline")
    # Two-stage args
    parser.add_argument("--detector", type=str, default=None,
                        help="Path to detector .pt (nc=1)")
    parser.add_argument("--classifier", type=str, default=None,
                        help="Path to classifier state_dict .pt")
    parser.add_argument("--classifier_mapping", type=str, default=None,
                        help="Path to idx_to_class.json")
    # Single-stage fallback
    parser.add_argument("--model", type=str, default=None,
                        help="Single-stage YOLO .pt (backward compat)")
    # Common args
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--yolo_val_dir", type=str, default="data/yolo/images/val")
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()

    two_stage = args.detector is not None
    if not two_stage and args.model is None:
        parser.error("Provide either --detector + --classifier or --model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_dir = Path(args.coco_dir)
    val_dir = Path(args.yolo_val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    val_filenames = {p.stem for p in val_dir.iterdir() if p.suffix in (".jpg", ".jpeg")}
    val_img_ids = {
        img["id"] for img in coco["images"]
        if Path(img["file_name"]).stem in val_filenames
    }

    ground_truths = []
    for ann in coco["annotations"]:
        if ann["image_id"] in val_img_ids:
            ground_truths.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
            })

    img_paths = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.jpeg"))
    stem_to_img_id = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}

    # --- Load models ---
    if two_stage:
        detector = YOLO(args.detector)
        clf_path = Path(args.classifier or "runs/classifier/best_classifier.pt")
        map_path = Path(args.classifier_mapping or "runs/classifier/idx_to_class.json")
        classifier_model, idx_to_class = load_classifier(clf_path, map_path, device)
        print(f"Two-stage: detector={args.detector}, classifier={clf_path}")
    else:
        model = YOLO(args.model)
        print(f"Single-stage: model={args.model}")

    print(f"Running inference on {len(img_paths)} val images...")

    predictions = []
    for img_path in img_paths:
        img_id = stem_to_img_id.get(img_path.stem)
        if img_id is None:
            continue

        if two_stage:
            results = detector.predict(
                source=str(img_path), conf=args.conf, augment=True, verbose=False
            )
            boxes_xyxy, det_scores = [], []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    boxes_xyxy.append(box.xyxy[0].tolist())
                    det_scores.append(float(box.conf.item()))

            img_preds = []
            if boxes_xyxy:
                pil_img = Image.open(img_path).convert("RGB")
                cat_ids, cls_scores = classify_crops(
                    pil_img, boxes_xyxy, classifier_model, idx_to_class, device
                )
                for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                    pred = {
                        "image_id": img_id,
                        "category_id": cat_ids[i],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(det_scores[i] * cls_scores[i], 4),
                    }
                    predictions.append(pred)
                    img_preds.append(pred)
        else:
            results = model.predict(
                source=str(img_path), conf=args.conf, verbose=False
            )
            img_preds = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pred = {
                        "image_id": img_id,
                        "category_id": int(box.cls.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(box.conf.item()),
                    }
                    predictions.append(pred)
                    img_preds.append(pred)

        if args.save_images:
            img = cv2.imread(str(img_path))
            vis = draw_predictions(img, img_preds, categories)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_pred.jpg"), vis)

    det_map, cls_map = evaluate_predictions(predictions, ground_truths)
    combined = 0.7 * det_map + 0.3 * cls_map

    print(f"\n{'='*50}")
    print(f"  Detection mAP@0.5:       {det_map:.4f}")
    print(f"  Classification mAP@0.5:  {cls_map:.4f}")
    print(f"  Combined score:          {combined:.4f}")
    print(f"    (0.7 × {det_map:.4f} + 0.3 × {cls_map:.4f})")
    print(f"{'='*50}")
    print(f"\n  Predictions: {len(predictions)}")
    print(f"  Ground truths: {len(ground_truths)}")
    print(f"  Val images: {len(img_paths)}")

    if args.save_images:
        print(f"  Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
