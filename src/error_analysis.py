"""Per-category error analysis for the two-stage detection + classification pipeline.

Runs inference on the validation set and reports:
  - Per-category AP, precision, recall, and F1 at IoU >= 0.5
  - Categories the model completely misses (zero recall)
  - Categories with the most false positives (low precision)
  - Top confusion pairs: what a GT category gets misclassified as
  - Overall detection miss rate (GT boxes with no matching prediction)

Usage (two-stage):
    python src/error_analysis.py \
        --detector detector.pt \
        --classifier classifier.pt \
        --classifier_mapping idx_to_class.json

Usage (single-stage):
    python src/error_analysis.py --model best.pt

Optional:
    --top_n 30        Show top N worst categories (default: 30)
    --save_crops      Save misclassified crops to error_analysis/crops/
    --conf 0.15       Detection confidence threshold (default: 0.15)
"""

import argparse
import functools
import io
import json
from collections import Counter, defaultdict
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
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list, recalls: list) -> float:
    if not precisions:
        return 0.0
    recall_levels = np.linspace(0, 1, 11)
    ap = 0.0
    for r_level in recall_levels:
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= r_level]
        ap += max(precisions_at_recall) if precisions_at_recall else 0.0
    return ap / 11


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
    clf_transform = transforms.Compose([
        transforms.Resize(int(CLASSIFIER_IMGSZ * 1.14)),
        transforms.CenterCrop(CLASSIFIER_IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    iw, ih = pil_image.size
    tensors = []
    for x1, y1, x2, y2 in boxes_xyxy:
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


def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5):
    """Match each GT to its best-scoring prediction. Returns detailed match info."""
    gt_by_image: dict[int, list] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt["image_id"], []).append(gt)

    pred_by_image: dict[int, list] = {}
    for pred in predictions:
        pred_by_image.setdefault(pred["image_id"], []).append(pred)

    gt_matches = []
    pred_matched = set()

    for img_id, gts in gt_by_image.items():
        preds = sorted(pred_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt_indices = set()

        for pred_global_idx, pred in enumerate(preds):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                gt = gts[best_gt_idx]
                gt_matches.append({
                    "image_id": img_id,
                    "gt_category_id": gt["category_id"],
                    "pred_category_id": pred["category_id"],
                    "gt_bbox": gt["bbox"],
                    "pred_bbox": pred["bbox"],
                    "iou": best_iou,
                    "score": pred["score"],
                    "correct": gt["category_id"] == pred["category_id"],
                })
                pred_matched.add((img_id, pred_global_idx))

        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt_indices:
                gt_matches.append({
                    "image_id": img_id,
                    "gt_category_id": gt["category_id"],
                    "pred_category_id": None,
                    "gt_bbox": gt["bbox"],
                    "pred_bbox": None,
                    "iou": 0.0,
                    "score": 0.0,
                    "correct": False,
                })

    unmatched_preds = []
    for img_id, preds in pred_by_image.items():
        for pred_idx, pred in enumerate(
            sorted(preds, key=lambda p: -p["score"])
        ):
            if (img_id, pred_idx) not in pred_matched:
                unmatched_preds.append(pred)

    return gt_matches, unmatched_preds


def per_category_ap(predictions, ground_truths, iou_threshold=0.5):
    """Compute AP per category, same logic as evaluate.py."""
    all_cat_ids = set(gt["category_id"] for gt in ground_truths)
    cat_aps = {}

    for cat_id in all_cat_ids:
        cat_gts = [gt for gt in ground_truths if gt["category_id"] == cat_id]
        cat_gt_by_image: dict[int, list] = {}
        for gt in cat_gts:
            cat_gt_by_image.setdefault(gt["image_id"], []).append(gt)

        cat_preds = sorted(
            [p for p in predictions if p["category_id"] == cat_id],
            key=lambda p: -p["score"],
        )
        total_cat_gt = len(cat_gts)
        if total_cat_gt == 0:
            continue

        matched_by_image: dict[int, set] = {}
        tp, fp = 0, 0
        precisions, recalls = [], []

        for pred in cat_preds:
            img_id = pred["image_id"]
            gts_for_img = cat_gt_by_image.get(img_id, [])
            matched = matched_by_image.setdefault(img_id, set())
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(gts_for_img):
                if idx in matched:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx >= 0:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / total_cat_gt)

        ap = compute_ap(precisions, recalls)
        final_precision = precisions[-1] if precisions else 0.0
        final_recall = recalls[-1] if recalls else 0.0
        f1 = (
            2 * final_precision * final_recall / (final_precision + final_recall)
            if (final_precision + final_recall) > 0
            else 0.0
        )

        cat_aps[cat_id] = {
            "ap": ap,
            "precision": final_precision,
            "recall": final_recall,
            "f1": f1,
            "n_gt": total_cat_gt,
            "n_pred": len(cat_preds),
            "tp": tp,
            "fp": fp,
            "fn": total_cat_gt - tp,
        }

    return cat_aps


def save_misclassified_crop(img_path, bbox_xywh, gt_cat, pred_cat, categories, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    x, y, w, h = [int(v) for v in bbox_xywh]
    ih, iw = img.shape[:2]
    pad = int(max(w, h) * 0.1)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(iw, x + w + pad), min(ih, y + h + pad)
    crop = img[y1:y2, x1:x2]

    gt_name = categories.get(gt_cat, str(gt_cat))[:40]
    pred_name = categories.get(pred_cat, str(pred_cat))[:40]
    safe_gt = "".join(c if c.isalnum() or c in " _-" else "_" for c in gt_name).strip()
    safe_pred = "".join(c if c.isalnum() or c in " _-" else "_" for c in pred_name).strip()

    cat_dir = output_dir / f"gt_{gt_cat}_{safe_gt}"
    cat_dir.mkdir(parents=True, exist_ok=True)
    filename = f"pred_{pred_cat}_{safe_pred}.jpg"
    cv2.imwrite(str(cat_dir / filename), crop)


class TeeWriter:
    """Write to both a StringIO buffer and stdout simultaneously."""

    def __init__(self):
        self._buf = io.StringIO()

    def write(self, text):
        self._buf.write(text)
        import sys as _sys
        _sys.__stdout__.write(text)

    def flush(self):
        import sys as _sys
        _sys.__stdout__.flush()

    def getvalue(self):
        return self._buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Per-category error analysis")
    parser.add_argument("--detector", type=str, default=None)
    parser.add_argument("--classifier", type=str, default=None)
    parser.add_argument("--classifier_mapping", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--yolo_val_dir", type=str, default="data/yolo/images/val")
    parser.add_argument("--output_dir", type=str, default="error_analysis")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--save_crops", action="store_true")
    args = parser.parse_args()

    two_stage = args.detector is not None
    if not two_stage and args.model is None:
        parser.error("Provide either --detector + --classifier or --model")

    import sys
    tee = TeeWriter()
    sys.stdout = tee

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_dir = Path(args.coco_dir)
    val_dir = Path(args.yolo_val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_dir = (coco_dir / "images").resolve()
    val_filenames = {p.stem for p in val_dir.iterdir() if p.suffix in (".jpg", ".jpeg")}
    val_img_ids = {
        img["id"] for img in coco["images"]
        if Path(img["file_name"]).stem in val_filenames
    }
    stem_to_img_id = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}
    ground_truths = [
        {"image_id": ann["image_id"], "category_id": ann["category_id"], "bbox": ann["bbox"]}
        for ann in coco["annotations"]
        if ann["image_id"] in val_img_ids
    ]

    # Read images from the original COCO images dir (val_dir only used for the split list)
    img_paths = [
        images_dir / img["file_name"]
        for img in coco["images"]
        if Path(img["file_name"]).stem in val_filenames
    ]
    img_paths.sort()

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
    for i, img_path in enumerate(img_paths):
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

            if boxes_xyxy:
                pil_img = Image.open(img_path).convert("RGB")
                cat_ids, cls_scores = classify_crops(
                    pil_img, boxes_xyxy, classifier_model, idx_to_class, device
                )
                for j, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_ids[j],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": round(det_scores[j] * cls_scores[j], 4),
                    })
        else:
            results = model.predict(
                source=str(img_path), conf=args.conf, verbose=False
            )
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(box.cls.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(box.conf.item()),
                    })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(img_paths)}] images processed")

    print(f"\nTotal predictions: {len(predictions)}, Total GT: {len(ground_truths)}")

    # =========================================================================
    # 1. Per-category AP breakdown
    # =========================================================================
    cat_aps = per_category_ap(predictions, ground_truths)

    sorted_by_ap = sorted(cat_aps.items(), key=lambda x: x[1]["ap"])
    print(f"\n{'='*90}")
    print(f"  WORST {args.top_n} CATEGORIES BY AP")
    print(f"{'='*90}")
    print(f"{'Cat':>4}  {'AP':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'GT':>4}  Name")
    print(f"{'-'*90}")

    for cat_id, stats in sorted_by_ap[:args.top_n]:
        name = categories.get(cat_id, "?")[:40]
        print(
            f"{cat_id:>4}  {stats['ap']:>6.3f}  {stats['precision']:>6.3f}  "
            f"{stats['recall']:>6.3f}  {stats['f1']:>6.3f}  {stats['tp']:>4}  "
            f"{stats['fp']:>4}  {stats['fn']:>4}  {stats['n_gt']:>4}  {name}"
        )

    # =========================================================================
    # 2. Completely missed categories (zero recall)
    # =========================================================================
    all_gt_cats = set(gt["category_id"] for gt in ground_truths)
    zero_recall = [
        (cat_id, cat_aps[cat_id]["n_gt"])
        for cat_id in all_gt_cats
        if cat_id in cat_aps and cat_aps[cat_id]["recall"] == 0
    ]
    zero_recall.sort(key=lambda x: -x[1])

    print(f"\n{'='*90}")
    print(f"  COMPLETELY MISSED CATEGORIES (recall = 0)  [{len(zero_recall)} categories]")
    print(f"{'='*90}")
    for cat_id, n_gt in zero_recall:
        name = categories.get(cat_id, "?")[:50]
        print(f"  cat {cat_id:>3} ({n_gt:>3} GT instances): {name}")

    # Categories in GT but with zero predictions
    never_predicted = all_gt_cats - set(p["category_id"] for p in predictions)
    never_predicted_in_gt = [
        (cat_id, sum(1 for gt in ground_truths if gt["category_id"] == cat_id))
        for cat_id in never_predicted
    ]
    never_predicted_in_gt.sort(key=lambda x: -x[1])

    if never_predicted_in_gt:
        print("\n  Categories NEVER predicted (not in any prediction):")
        for cat_id, n_gt in never_predicted_in_gt:
            name = categories.get(cat_id, "?")[:50]
            print(f"    cat {cat_id:>3} ({n_gt:>3} GT instances): {name}")

    # =========================================================================
    # 3. Confusion analysis: what gets misclassified as what
    # =========================================================================
    gt_matches, unmatched_preds = match_predictions_to_gt(predictions, ground_truths)

    confusion_pairs = Counter()
    misclass_by_gt = defaultdict(list)

    for m in gt_matches:
        if m["pred_category_id"] is not None and not m["correct"]:
            pair = (m["gt_category_id"], m["pred_category_id"])
            confusion_pairs[pair] += 1
            misclass_by_gt[m["gt_category_id"]].append(m)

    print(f"\n{'='*90}")
    print(f"  TOP {args.top_n} CONFUSION PAIRS (GT -> predicted as)")
    print(f"{'='*90}")
    print(f"{'Count':>5}  {'GT Cat':>6}  {'Pred Cat':>8}  GT Name -> Pred Name")
    print(f"{'-'*90}")

    for (gt_cat, pred_cat), count in confusion_pairs.most_common(args.top_n):
        gt_name = categories.get(gt_cat, "?")[:30]
        pred_name = categories.get(pred_cat, "?")[:30]
        print(f"{count:>5}  {gt_cat:>6}  {pred_cat:>8}  {gt_name} -> {pred_name}")

    # =========================================================================
    # 4. Per-category confusion breakdown (most-confused categories)
    # =========================================================================
    cats_by_misclass_count = sorted(
        misclass_by_gt.items(), key=lambda x: -len(x[1])
    )

    print(f"\n{'='*90}")
    print(f"  CATEGORIES WITH MOST MISCLASSIFICATIONS (top {min(args.top_n, len(cats_by_misclass_count))})")
    print(f"{'='*90}")

    for gt_cat, mismatches in cats_by_misclass_count[:args.top_n]:
        gt_name = categories.get(gt_cat, "?")[:40]
        n_gt = sum(1 for gt in ground_truths if gt["category_id"] == gt_cat)
        pred_counter = Counter(m["pred_category_id"] for m in mismatches)
        top_confused = pred_counter.most_common(3)

        confused_str = ", ".join(
            f"{categories.get(pc, '?')[:25]}({cnt})" for pc, cnt in top_confused
        )
        print(f"  cat {gt_cat:>3} [{len(mismatches):>3} errors / {n_gt:>3} GT]: {gt_name}")
        print(f"         confused with: {confused_str}")

    # =========================================================================
    # 5. Detection miss analysis
    # =========================================================================
    missed_detections = [m for m in gt_matches if m["pred_category_id"] is None]
    missed_by_cat = Counter(m["gt_category_id"] for m in missed_detections)

    print(f"\n{'='*90}")
    print(f"  DETECTION MISSES (GT boxes with no matching prediction)  [{len(missed_detections)} total]")
    print(f"{'='*90}")

    for cat_id, count in missed_by_cat.most_common(args.top_n):
        n_gt = sum(1 for gt in ground_truths if gt["category_id"] == cat_id)
        name = categories.get(cat_id, "?")[:40]
        print(f"  cat {cat_id:>3}: {count:>3} missed / {n_gt:>3} GT ({100*count/n_gt:.0f}% miss rate)  {name}")

    # =========================================================================
    # 6. False positive hotspots
    # =========================================================================
    fp_by_cat = Counter(p["category_id"] for p in unmatched_preds)

    print(f"\n{'='*90}")
    print(f"  FALSE POSITIVE HOTSPOTS (unmatched predictions)  [{len(unmatched_preds)} total]")
    print(f"{'='*90}")

    for cat_id, count in fp_by_cat.most_common(args.top_n):
        name = categories.get(cat_id, "?")[:40]
        n_pred = sum(1 for p in predictions if p["category_id"] == cat_id)
        print(f"  cat {cat_id:>3}: {count:>3} FP / {n_pred:>3} preds ({100*count/n_pred:.0f}% FP rate)  {name}")

    # =========================================================================
    # 7. Save misclassified crops if requested
    # =========================================================================
    if args.save_crops:
        crops_dir = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving misclassified crops to {crops_dir}/ ...")

        img_id_to_path = {}
        for p in img_paths:
            iid = stem_to_img_id.get(p.stem)
            if iid is not None:
                img_id_to_path[iid] = p

        saved = 0
        for m in gt_matches:
            if m["pred_category_id"] is not None and not m["correct"]:
                img_path = img_id_to_path.get(m["image_id"])
                if img_path:
                    save_misclassified_crop(
                        img_path, m["gt_bbox"],
                        m["gt_category_id"], m["pred_category_id"],
                        categories, crops_dir,
                    )
                    saved += 1
        print(f"  Saved {saved} misclassified crops.")

    # =========================================================================
    # 8. Save JSON report
    # =========================================================================
    report = {
        "summary": {
            "total_predictions": len(predictions),
            "total_gt": len(ground_truths),
            "total_misclassified": sum(
                1 for m in gt_matches
                if m["pred_category_id"] is not None and not m["correct"]
            ),
            "total_missed": len(missed_detections),
            "total_false_positives": len(unmatched_preds),
            "categories_with_zero_recall": len(zero_recall),
            "categories_never_predicted": len(never_predicted),
        },
        "per_category": {
            str(cat_id): {
                "name": categories.get(cat_id, "?"),
                **stats,
            }
            for cat_id, stats in cat_aps.items()
        },
        "top_confusion_pairs": [
            {
                "gt_category_id": gt_cat,
                "gt_name": categories.get(gt_cat, "?"),
                "pred_category_id": pred_cat,
                "pred_name": categories.get(pred_cat, "?"),
                "count": count,
            }
            for (gt_cat, pred_cat), count in confusion_pairs.most_common(50)
        ],
    }

    report_path = output_dir / "error_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    # =========================================================================
    # 9. Write the full text log to a file
    # =========================================================================
    log_path = output_dir / "error_analysis.log"
    with open(log_path, "w") as f:
        f.write(tee.getvalue())
    print(f"Full text log saved to {log_path}")

    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
