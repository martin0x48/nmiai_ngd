"""Evaluate baseline vs improved inference pipeline.

Improvements:
  1. WBF (Weighted Boxes Fusion) to merge overlapping detections and reduce FPs
  2. Optimized confidence threshold
  3. Batch classification for speed

Usage:
    python src/evaluate_improved.py \
        --detector runs/detect/train5/weights/best.pt \
        --classifier runs/classifier/best_classifier.pt \
        --classifier_mapping runs/classifier/idx_to_class.json
"""

import argparse
import functools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ensemble_boxes import weighted_boxes_fusion

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

from ultralytics import YOLO

sys.path.insert(0, "src")
from evaluate import evaluate_predictions

CLASSIFIER_IMGSZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def detect_baseline(detector, img_path, conf):
    """Standard inference with TTA."""
    results = detector.predict(source=str(img_path), conf=conf, augment=True, verbose=False)
    boxes_xyxy, det_scores = [], []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            boxes_xyxy.append(box.xyxy[0].tolist())
            det_scores.append(float(box.conf.item()))
    return boxes_xyxy, det_scores


def detect_with_wbf(detector, img_path, conf, iou_thr=0.55, skip_box_thr=0.01):
    """Run TTA + non-TTA, merge with Weighted Boxes Fusion.

    WBF averages overlapping boxes instead of discarding them (like NMS),
    producing better-localized boxes and filtering duplicate detections.
    """
    # Get image dimensions for normalization
    pil_img = Image.open(img_path)
    iw, ih = pil_img.size

    # Run with TTA
    results_tta = detector.predict(source=str(img_path), conf=conf, augment=True, verbose=False)
    # Run without TTA
    results_no_tta = detector.predict(source=str(img_path), conf=conf, augment=False, verbose=False)

    all_boxes_list = []
    all_scores_list = []
    all_labels_list = []

    for results in [results_tta, results_no_tta]:
        boxes_norm = []
        scores = []
        labels = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # WBF expects normalized [0, 1] coordinates
                boxes_norm.append([x1 / iw, y1 / ih, x2 / iw, y2 / ih])
                scores.append(float(box.conf.item()))
                labels.append(0)
        all_boxes_list.append(boxes_norm)
        all_scores_list.append(scores)
        all_labels_list.append(labels)

    if all(len(b) == 0 for b in all_boxes_list):
        return [], []

    # Apply WBF
    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        all_boxes_list,
        all_scores_list,
        all_labels_list,
        weights=[1.0, 1.0],  # equal weight for both runs
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    # De-normalize
    boxes_xyxy = []
    det_scores = []
    for box, score in zip(fused_boxes, fused_scores):
        x1, y1, x2, y2 = box
        boxes_xyxy.append([x1 * iw, y1 * ih, x2 * iw, y2 * ih])
        det_scores.append(float(score))

    return boxes_xyxy, det_scores


def run_pipeline(detector, classifier, idx_to_class, img_paths, stem_to_img_id,
                 device, conf, use_wbf=False, wbf_iou=0.55):
    """Run full pipeline and return predictions."""
    predictions = []
    for img_path in img_paths:
        img_id = stem_to_img_id.get(img_path.stem)
        if img_id is None:
            continue

        if use_wbf:
            boxes_xyxy, det_scores = detect_with_wbf(
                detector, img_path, conf, iou_thr=wbf_iou
            )
        else:
            boxes_xyxy, det_scores = detect_baseline(detector, img_path, conf)

        if not boxes_xyxy:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        cat_ids, cls_scores = classify_crops(
            pil_img, boxes_xyxy, classifier, idx_to_class, device
        )

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            predictions.append({
                "image_id": img_id,
                "category_id": cat_ids[i],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": round(det_scores[i] * cls_scores[i], 4),
            })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--classifier_mapping", required=True)
    parser.add_argument("--coco_dir", default="data/coco")
    parser.add_argument("--yolo_val_dir", default="data/yolo/images/val")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_dir = Path(args.coco_dir)
    val_dir = Path(args.yolo_val_dir)

    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    val_filenames = {p.stem for p in val_dir.iterdir() if p.suffix in (".jpg", ".jpeg")}
    val_img_ids = {
        img["id"] for img in coco["images"]
        if Path(img["file_name"]).stem in val_filenames
    }
    ground_truths = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
        for a in coco["annotations"] if a["image_id"] in val_img_ids
    ]

    img_paths = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.jpeg"))
    stem_to_img_id = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}

    print(f"Loading models...")
    detector = YOLO(args.detector)
    classifier, idx_to_class = load_classifier(
        args.classifier, args.classifier_mapping, device
    )
    print(f"Val images: {len(img_paths)}, GT: {len(ground_truths)}")

    # 1. Baseline (standard TTA, conf=0.15)
    print(f"\n--- Baseline (TTA, conf={args.conf}) ---")
    baseline_preds = run_pipeline(
        detector, classifier, idx_to_class, img_paths, stem_to_img_id,
        device, args.conf, use_wbf=False,
    )
    det_b, cls_b = evaluate_predictions(baseline_preds, ground_truths)
    comb_b = 0.7 * det_b + 0.3 * cls_b
    print(f"  Preds: {len(baseline_preds)}, det={det_b:.4f}, cls={cls_b:.4f}, combined={comb_b:.4f}")

    # 2. WBF sweep
    print(f"\n--- WBF sweep (TTA+noTTA fused) ---")
    print(f"  {'wbf_iou':>8s} {'conf':>6s} {'Preds':>6s} {'det':>9s} {'cls':>9s} {'combined':>9s} {'delta':>8s}")
    print(f"  {'-' * 60}")

    best = comb_b
    best_cfg = None

    for wbf_iou in [0.4, 0.5, 0.55, 0.6, 0.7]:
        for conf in [0.05, 0.10, 0.15]:
            preds = run_pipeline(
                detector, classifier, idx_to_class, img_paths, stem_to_img_id,
                device, conf, use_wbf=True, wbf_iou=wbf_iou,
            )
            det, cls = evaluate_predictions(preds, ground_truths)
            comb = 0.7 * det + 0.3 * cls
            delta = comb - comb_b
            marker = " *" if comb > best else ""
            if comb > best:
                best = comb
                best_cfg = (wbf_iou, conf)
            print(f"  {wbf_iou:>8.2f} {conf:>6.2f} {len(preds):>6d} {det:>9.4f} {cls:>9.4f} {comb:>9.4f} {delta:>+8.4f}{marker}")

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Baseline:  combined={comb_b:.4f} (preds={len(baseline_preds)})")
    if best_cfg:
        print(f"  Best WBF:  combined={best:.4f} (wbf_iou={best_cfg[0]}, conf={best_cfg[1]})")
        print(f"  Delta:     {best - comb_b:+.4f}")
    else:
        print(f"  No WBF configuration improved over baseline.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
