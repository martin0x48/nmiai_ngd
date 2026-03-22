"""Evaluate the two-stage pipeline with and without co-occurrence rescoring.

Runs inference once, then compares metrics with and without rescoring.
Supports parameter sweeps to find optimal rescoring hyperparameters.

Usage:
    # First build the co-occurrence matrix:
    python src/build_cooccurrence.py

    # Then evaluate with rescoring comparison:
    python src/evaluate_with_rescoring.py \
        --detector detector.pt \
        --classifier classifier.pt \
        --classifier_mapping idx_to_class.json

    # Sweep over hyperparameters:
    python src/evaluate_with_rescoring.py \
        --detector detector.pt \
        --classifier classifier.pt \
        --classifier_mapping idx_to_class.json \
        --sweep
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

_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

from ultralytics import YOLO

from cooccurrence_rescorer import CooccurrenceRescorer
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


def classify_crops_topk(
    pil_image, boxes_xyxy, classifier, idx_to_class, device, top_k=5, pad_pct=0.05
):
    """Classify crops and return top-k probabilities per detection.

    Returns list of dicts with:
        category_id: int (best class)
        cls_score: float (best class probability)
        cls_probs: dict[int, float] (top-k {category_id: probability})
    """
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

        # Top-1
        best_scores, best_indices = probs.max(dim=1)

        # Top-k
        topk_scores, topk_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(len(boxes_xyxy)):
        best_cat = idx_to_class.get(best_indices[i].item(), 0)
        best_score = best_scores[i].item()

        # Build top-k probs dict mapped to category_ids
        cls_probs = {}
        for j in range(top_k):
            cat_id = idx_to_class.get(topk_indices[i][j].item(), 0)
            cls_probs[cat_id] = topk_scores[i][j].item()

        results.append({
            "category_id": best_cat,
            "cls_score": best_score,
            "cls_probs": cls_probs,
        })

    return results


def run_inference(detector, classifier, idx_to_class, img_paths, stem_to_img_id, device, conf=0.15, top_k=5):
    """Run detector + classifier on val images. Returns rich predictions by image."""
    predictions_by_image = defaultdict(list)

    for img_path in img_paths:
        img_id = stem_to_img_id.get(img_path.stem)
        if img_id is None:
            continue

        results = detector.predict(
            source=str(img_path), conf=conf, augment=True, verbose=False
        )

        boxes_xyxy, det_scores = [], []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                boxes_xyxy.append(box.xyxy[0].tolist())
                det_scores.append(float(box.conf.item()))

        if not boxes_xyxy:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        clf_results = classify_crops_topk(
            pil_img, boxes_xyxy, classifier, idx_to_class, device, top_k
        )

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            det = {
                "image_id": img_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format
                "det_score": det_scores[i],
                "category_id": clf_results[i]["category_id"],
                "cls_score": clf_results[i]["cls_score"],
                "cls_probs": clf_results[i]["cls_probs"],
                "score": round(det_scores[i] * clf_results[i]["cls_score"], 4),
            }
            predictions_by_image[img_id].append(det)

    return predictions_by_image


def to_standard_predictions(predictions_by_image):
    """Convert rich predictions to standard format for evaluation."""
    preds = []
    for img_id, dets in predictions_by_image.items():
        for d in dets:
            preds.append({
                "image_id": d["image_id"],
                "category_id": d["category_id"],
                "bbox": d["bbox"],
                "score": d.get("score", round(d["det_score"] * d["cls_score"], 4)),
            })
    return preds


def print_comparison(baseline_det, baseline_cls, rescored_det, rescored_cls, stats):
    baseline_combined = 0.7 * baseline_det + 0.3 * baseline_cls
    rescored_combined = 0.7 * rescored_det + 0.3 * rescored_cls

    det_delta = rescored_det - baseline_det
    cls_delta = rescored_cls - baseline_cls
    combined_delta = rescored_combined - baseline_combined

    def fmt(v):
        return f"{v:+.4f}" if v != 0 else " 0.0000"

    print(f"\n{'=' * 60}")
    print(f"  RESCORING COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'':25s} {'Baseline':>10s} {'Rescored':>10s} {'Delta':>10s}")
    print(f"  {'-' * 55}")
    print(f"  {'Detection mAP:':25s} {baseline_det:>10.4f} {rescored_det:>10.4f} {fmt(det_delta):>10s}")
    print(f"  {'Classification mAP:':25s} {baseline_cls:>10.4f} {rescored_cls:>10.4f} {fmt(cls_delta):>10s}")
    print(f"  {'Combined (0.7d + 0.3c):':25s} {baseline_combined:>10.4f} {rescored_combined:>10.4f} {fmt(combined_delta):>10s}")
    print(f"{'=' * 60}")

    if stats:
        total = stats["total"]
        rescored = stats["rescored"]
        changed = stats["changed"]
        print(f"  Rescoring stats:")
        print(f"    Total detections:    {total}")
        print(f"    Below conf threshold: {rescored} ({100 * rescored / max(total, 1):.1f}%)")
        print(f"    Category changed:    {changed} ({100 * changed / max(rescored, 1):.1f}% of rescored)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage pipeline with co-occurrence rescoring"
    )
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--classifier_mapping", type=str, required=True)
    parser.add_argument("--coco_dir", type=str, default="data/coco")
    parser.add_argument("--yolo_val_dir", type=str, default="data/yolo/images/val")
    parser.add_argument("--output_dir", type=str, default="eval_output")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--cooccurrence_path", type=str, default="data/cooccurrence.json")

    # Rescoring hyperparameters
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--spatial_weight", type=float, default=0.3)
    parser.add_argument("--top_k_candidates", type=int, default=5)

    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep over alpha and conf_threshold to find best params",
    )
    parser.add_argument(
        "--save_predictions", action="store_true",
        help="Save baseline and rescored prediction JSONs",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_dir = Path(args.coco_dir)
    val_dir = Path(args.yolo_val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

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

    # Load models
    print(f"Loading detector: {args.detector}")
    detector = YOLO(args.detector)
    print(f"Loading classifier: {args.classifier}")
    classifier, idx_to_class = load_classifier(
        args.classifier, args.classifier_mapping, device
    )

    # Run inference ONCE
    print(f"\nRunning inference on {len(img_paths)} val images...")
    predictions_by_image = run_inference(
        detector, classifier, idx_to_class, img_paths,
        stem_to_img_id, device, args.conf, args.top_k_candidates,
    )
    total_preds = sum(len(v) for v in predictions_by_image.values())
    print(f"  Total predictions: {total_preds}")
    print(f"  Ground truths: {len(ground_truths)}")

    # Baseline evaluation
    baseline_preds = to_standard_predictions(predictions_by_image)
    baseline_det, baseline_cls = evaluate_predictions(baseline_preds, ground_truths)
    baseline_combined = 0.7 * baseline_det + 0.3 * baseline_cls

    print(f"\n  Baseline Detection mAP:       {baseline_det:.4f}")
    print(f"  Baseline Classification mAP:  {baseline_cls:.4f}")
    print(f"  Baseline Combined:            {baseline_combined:.4f}")

    # Check co-occurrence file exists
    if not Path(args.cooccurrence_path).exists():
        print(f"\nWARNING: {args.cooccurrence_path} not found.")
        print("Run: python src/build_cooccurrence.py")
        sys.exit(1)

    if args.sweep:
        # Parameter sweep
        alphas = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        conf_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        k_neighbors_list = [4, 8, 12]

        print(f"\n{'=' * 80}")
        print(f"  PARAMETER SWEEP")
        print(f"{'=' * 80}")
        print(f"  {'alpha':>6s} {'conf_th':>8s} {'k_nn':>5s} {'det_mAP':>9s} {'cls_mAP':>9s} {'combined':>9s} {'delta':>8s} {'changed':>8s}")
        print(f"  {'-' * 76}")

        best_combined = baseline_combined
        best_params = None

        for alpha in alphas:
            for conf_th in conf_thresholds:
                for k_nn in k_neighbors_list:
                    rescorer = CooccurrenceRescorer(
                        args.cooccurrence_path,
                        alpha=alpha,
                        conf_threshold=conf_th,
                        k_neighbors=k_nn,
                        spatial_weight=args.spatial_weight,
                        top_k_candidates=args.top_k_candidates,
                    )
                    rescored_preds, stats = rescorer.rescore_all(predictions_by_image)
                    r_det, r_cls = evaluate_predictions(rescored_preds, ground_truths)
                    r_combined = 0.7 * r_det + 0.3 * r_cls
                    delta = r_combined - baseline_combined

                    marker = " *" if r_combined > best_combined else ""
                    print(
                        f"  {alpha:>6.1f} {conf_th:>8.1f} {k_nn:>5d} "
                        f"{r_det:>9.4f} {r_cls:>9.4f} {r_combined:>9.4f} "
                        f"{delta:>+8.4f} {stats['changed']:>8d}{marker}"
                    )

                    if r_combined > best_combined:
                        best_combined = r_combined
                        best_params = (alpha, conf_th, k_nn)

        print(f"\n  Baseline combined: {baseline_combined:.4f}")
        if best_params:
            print(f"  Best combined:     {best_combined:.4f} (delta: {best_combined - baseline_combined:+.4f})")
            print(f"  Best params: alpha={best_params[0]}, conf_threshold={best_params[1]}, k_neighbors={best_params[2]}")
        else:
            print(f"  No improvement found over baseline.")

    else:
        # Single evaluation with given params
        rescorer = CooccurrenceRescorer(
            args.cooccurrence_path,
            alpha=args.alpha,
            conf_threshold=args.conf_threshold,
            k_neighbors=args.k_neighbors,
            spatial_weight=args.spatial_weight,
            top_k_candidates=args.top_k_candidates,
        )
        rescored_preds, stats = rescorer.rescore_all(predictions_by_image)
        rescored_det, rescored_cls = evaluate_predictions(rescored_preds, ground_truths)

        print_comparison(
            baseline_det, baseline_cls, rescored_det, rescored_cls, stats
        )

        # Verify detection mAP unchanged
        if abs(rescored_det - baseline_det) > 1e-6:
            print("  WARNING: Detection mAP changed! This indicates a bug.")

    if args.save_predictions:
        with open(output_dir / "predictions_baseline.json", "w") as f:
            json.dump(baseline_preds, f)
        rescorer = CooccurrenceRescorer(
            args.cooccurrence_path,
            alpha=args.alpha,
            conf_threshold=args.conf_threshold,
            k_neighbors=args.k_neighbors,
            spatial_weight=args.spatial_weight,
            top_k_candidates=args.top_k_candidates,
        )
        rescored_preds, _ = rescorer.rescore_all(predictions_by_image)
        with open(output_dir / "predictions_rescored.json", "w") as f:
            json.dump(rescored_preds, f)
        print(f"  Predictions saved to {output_dir}/")


if __name__ == "__main__":
    main()
