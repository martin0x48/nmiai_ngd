"""Evaluate with sliced inference using supervision.InferenceSlicer.

Tiles images into overlapping crops, runs detection on each, merges with NMS.
Then classifies each detection and evaluates.

Usage:
    python src/evaluate_sliced.py \
        --detector runs/detect/train4/weights/best.pt \
        --classifier runs/classifier/best_classifier.pt \
        --classifier_mapping runs/classifier/idx_to_class.json
"""

import argparse
import functools
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

import supervision as sv
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


def run_standard_inference(detector, img_path, conf):
    """Standard full-image inference."""
    results = detector.predict(source=str(img_path), conf=conf, augment=True, verbose=False)
    boxes_xyxy, det_scores = [], []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            boxes_xyxy.append(box.xyxy[0].tolist())
            det_scores.append(float(box.conf.item()))
    return boxes_xyxy, det_scores


def run_sliced_inference(detector, img_path, conf, slice_wh, overlap_ratio):
    """Sliced inference using supervision.InferenceSlicer."""
    def callback(image_slice: np.ndarray) -> sv.Detections:
        results = detector.predict(
            source=image_slice, conf=conf, augment=False, verbose=False
        )
        detections = sv.Detections.empty()
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            detections = sv.Detections(
                xyxy=r.boxes.xyxy.cpu().numpy(),
                confidence=r.boxes.conf.cpu().numpy(),
                class_id=np.zeros(len(r.boxes), dtype=int),
            )
        return detections

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(slice_wh, slice_wh),
        overlap_ratio_wh=(overlap_ratio, overlap_ratio),
        iou_threshold=0.5,
        thread_workers=1,
    )

    image = cv2.imread(str(img_path))
    detections = slicer(image)

    boxes_xyxy = detections.xyxy.tolist() if len(detections) > 0 else []
    det_scores = detections.confidence.tolist() if len(detections) > 0 else []
    return boxes_xyxy, det_scores


def run_combined_inference(detector, img_path, conf, slice_wh, overlap_ratio):
    """Run both standard and sliced inference, merge with NMS."""
    # Standard (with TTA)
    std_boxes, std_scores = run_standard_inference(detector, img_path, conf)

    # Sliced
    sl_boxes, sl_scores = run_sliced_inference(
        detector, img_path, conf, slice_wh, overlap_ratio
    )

    # Merge
    all_boxes = std_boxes + sl_boxes
    all_scores = std_scores + sl_scores

    if not all_boxes:
        return [], []

    # NMS to remove duplicates
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    keep = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

    final_boxes = [all_boxes[i] for i in keep.tolist()]
    final_scores = [all_scores[i] for i in keep.tolist()]
    return final_boxes, final_scores


def evaluate_mode(
    mode, detector, classifier, idx_to_class, img_paths, stem_to_img_id,
    ground_truths, device, conf, slice_wh=640, overlap_ratio=0.2
):
    """Run inference in a given mode and evaluate."""
    predictions = []
    for img_path in img_paths:
        img_id = stem_to_img_id.get(img_path.stem)
        if img_id is None:
            continue

        if mode == "standard":
            boxes_xyxy, det_scores = run_standard_inference(detector, img_path, conf)
        elif mode == "sliced":
            boxes_xyxy, det_scores = run_sliced_inference(
                detector, img_path, conf, slice_wh, overlap_ratio
            )
        elif mode == "combined":
            boxes_xyxy, det_scores = run_combined_inference(
                detector, img_path, conf, slice_wh, overlap_ratio
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

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

    det_map, cls_map = evaluate_predictions(predictions, ground_truths)
    return det_map, cls_map, len(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--classifier_mapping", required=True)
    parser.add_argument("--coco_dir", default="data/coco")
    parser.add_argument("--yolo_val_dir", default="data/yolo/images/val")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--slice_wh", type=int, default=640)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
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
    print(f"Val images: {len(img_paths)}, GT: {len(ground_truths)}\n")

    # Standard baseline
    print("Running standard inference...")
    det, cls, n_preds = evaluate_mode(
        "standard", detector, classifier, idx_to_class, img_paths, stem_to_img_id,
        ground_truths, device, args.conf,
    )
    base_combined = 0.7 * det + 0.3 * cls
    print(f"  standard: det={det:.4f}, cls={cls:.4f}, combined={base_combined:.4f}, preds={n_preds}\n")

    # Sweep slice sizes and overlaps
    print(f"{'=' * 75}")
    print(f"  SLICED INFERENCE SWEEP (combined = standard + sliced + NMS)")
    print(f"{'=' * 75}")
    print(f"  {'slice':>6s} {'overlap':>8s} {'Preds':>6s} {'det_mAP':>9s} {'cls_mAP':>9s} {'combined':>9s} {'delta':>8s}")
    print(f"  {'-' * 70}")

    for slice_wh in [800, 1024, 1280]:
        for overlap in [0.2, 0.3]:
            det, cls, n_preds = evaluate_mode(
                "combined", detector, classifier, idx_to_class, img_paths,
                stem_to_img_id, ground_truths, device, args.conf, slice_wh, overlap,
            )
            combined = 0.7 * det + 0.3 * cls
            delta = combined - base_combined
            d_str = f"{delta:+.4f}"
            marker = " *" if delta > 0 else ""
            print(f"  {slice_wh:>6d} {overlap:>8.1f} {n_preds:>6d} {det:>9.4f} {cls:>9.4f} {combined:>9.4f} {d_str:>8s}{marker}")

    print(f"  Baseline combined: {base_combined:.4f}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
