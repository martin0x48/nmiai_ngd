"""Sweep confidence thresholds on existing predictions."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evaluate import evaluate_predictions

with open("eval_output/predictions_baseline.json") as f:
    preds = json.load(f)

with open("data/coco/annotations.json") as f:
    coco = json.load(f)

val_dir = Path("data/yolo/images/val")
val_filenames = {p.stem for p in val_dir.iterdir() if p.suffix in (".jpg", ".jpeg")}
val_img_ids = {
    img["id"] for img in coco["images"] if Path(img["file_name"]).stem in val_filenames
}

ground_truths = []
for ann in coco["annotations"]:
    if ann["image_id"] in val_img_ids:
        ground_truths.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
            }
        )

print(f"GT: {len(ground_truths)}, Preds: {len(preds)}")
header = f"  {'Conf':>6s} {'N_pred':>7s} {'det_mAP':>9s} {'cls_mAP':>9s} {'combined':>9s}"
print(header)
print("  " + "-" * len(header))

best = 0
best_conf = 0
for conf in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    filtered = [p for p in preds if p["score"] >= conf]
    det, cls = evaluate_predictions(filtered, ground_truths)
    combined = 0.7 * det + 0.3 * cls
    marker = " *" if combined > best else ""
    if combined > best:
        best = combined
        best_conf = conf
    print(f"  {conf:>6.2f} {len(filtered):>7d} {det:>9.4f} {cls:>9.4f} {combined:>9.4f}{marker}")

print(f"\nBest: conf={best_conf}, combined={best:.4f}")
