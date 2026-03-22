"""Run YOLO detector + EfficientNet classifier on meny images.

Usage (from norgesgruppen-cv/):
    python labeling/detect.py

Produces: data/meny/detections.json
Resume-safe: skips already-processed images.
"""

import json
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MENY_DIR = ROOT / "data" / "meny"
COCO_PATH = ROOT / "data" / "coco" / "annotations.json"
DETECTIONS_PATH = MENY_DIR / "detections.json"

CLASSIFIER_IMGSZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_classifier(device):
    weights = ROOT / "classifier.pt"
    mapping = ROOT / "idx_to_class.json"
    with open(mapping, encoding="utf-8") as f:
        idx_to_class = {int(k): int(v) for k, v in json.load(f).items()}
    model = timm.create_model(
        "tf_efficientnetv2_m", pretrained=False, num_classes=len(idx_to_class)
    )
    sd = torch.load(str(weights), map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return model, idx_to_class


def build_transform():
    return transforms.Compose([
        transforms.Resize(int(CLASSIFIER_IMGSZ * 1.14)),
        transforms.CenterCrop(CLASSIFIER_IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def crop_padded(pil_image, bbox_xyxy, pad_pct=0.05):
    x1, y1, x2, y2 = bbox_xyxy
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2) + 1, max(y1, y2) + 1
    w, h = x2 - x1, y2 - y1
    iw, ih = pil_image.size
    cx1 = max(0, int(x1 - w * pad_pct))
    cy1 = max(0, int(y1 - h * pad_pct))
    cx2 = min(iw, int(x2 + w * pad_pct))
    cy2 = min(ih, int(y2 + h * pad_pct))
    if cx2 <= cx1:
        cx1, cx2 = 0, min(iw, 10)
    if cy2 <= cy1:
        cy1, cy2 = 0, min(ih, 10)
    return pil_image.crop((cx1, cy1, cx2, cy2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(COCO_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    print("Loading detector...")
    detector = YOLO(str(ROOT / "detector.pt"))

    has_classifier = True
    try:
        print("Loading classifier...")
        classifier, idx_to_class = load_classifier(device)
        clf_transform = build_transform()
    except Exception as e:
        print(f"Warning: classifier not available ({e}), skipping suggestions")
        has_classifier = False

    image_paths = []
    for folder in sorted(MENY_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_paths.append(img)
    print(f"Found {len(image_paths)} images")

    detections = {}
    if DETECTIONS_PATH.exists():
        with open(DETECTIONS_PATH, encoding="utf-8") as f:
            detections = json.load(f)

    for i, img_path in enumerate(image_paths):
        rel = img_path.relative_to(MENY_DIR).as_posix()
        if rel in detections:
            print(f"[{i+1}/{len(image_paths)}] {rel}: cached, skip")
            continue

        print(f"[{i+1}/{len(image_paths)}] {rel}...", end=" ", flush=True)
        pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil.size

        results = detector.predict(
            source=str(img_path), conf=0.15, augment=True, verbose=False
        )
        boxes, confs = [], []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                boxes.append(box.xyxy[0].tolist())
                confs.append(float(box.conf.item()))

        if not boxes:
            detections[rel] = {"width": orig_w, "height": orig_h, "detections": []}
            print("0 detections")
        else:
            dets_list = []
            if has_classifier:
                tensors = [clf_transform(crop_padded(pil, b)) for b in boxes]
                batch = torch.stack(tensors).to(device)
                if device.type == "cuda":
                    batch = batch.half()
                with torch.no_grad():
                    probs = torch.softmax(classifier(batch), dim=1)
                    scores, indices = probs.max(dim=1)

            for j, bbox in enumerate(boxes):
                entry = {
                    "bbox_xyxy": [round(c, 1) for c in bbox],
                    "det_conf": round(confs[j], 4),
                }
                if has_classifier:
                    cat_id = idx_to_class.get(indices[j].item(), 0)
                    entry["suggested_category_id"] = cat_id
                    entry["suggested_category_name"] = cat_id_to_name.get(cat_id, "unknown_product")
                    entry["suggestion_conf"] = round(scores[j].item(), 4)
                dets_list.append(entry)

            detections[rel] = {
                "width": orig_w, "height": orig_h, "detections": dets_list
            }
            print(f"{len(dets_list)} detections")

        with open(DETECTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)

    total = sum(
        len(v["detections"]) if isinstance(v, dict) else len(v)
        for v in detections.values()
    )
    print(f"\nDone! {len(detections)} images, {total} detections")
    print(f"Saved to {DETECTIONS_PATH}")


def process_single(rel_path):
    """Re-detect a single image, print JSON to stdout."""
    import sys as _sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", file=_sys.stderr)

    with open(COCO_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    print("Loading detector...", file=_sys.stderr)
    detector = YOLO(str(ROOT / "detector.pt"))

    has_classifier = True
    try:
        print("Loading classifier...", file=_sys.stderr)
        classifier, idx_to_class = load_classifier(device)
        clf_transform = build_transform()
    except Exception as e:
        print(f"Warning: classifier not available ({e})", file=_sys.stderr)
        has_classifier = False

    img_path = MENY_DIR / rel_path
    if not img_path.exists():
        print(json.dumps({"error": f"Image not found: {rel_path}"}))
        return

    pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil.size

    results = detector.predict(
        source=str(img_path), conf=0.15, augment=True, verbose=False
    )
    boxes, confs = [], []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            boxes.append(box.xyxy[0].tolist())
            confs.append(float(box.conf.item()))

    dets_list = []
    if boxes:
        if has_classifier:
            tensors = [clf_transform(crop_padded(pil, b)) for b in boxes]
            batch = torch.stack(tensors).to(device)
            if device.type == "cuda":
                batch = batch.half()
            with torch.no_grad():
                probs = torch.softmax(classifier(batch), dim=1)
                scores, indices = probs.max(dim=1)

        for j, bbox in enumerate(boxes):
            entry = {
                "bbox_xyxy": [round(c, 1) for c in bbox],
                "det_conf": round(confs[j], 4),
            }
            if has_classifier:
                cat_id = idx_to_class.get(indices[j].item(), 0)
                entry["suggested_category_id"] = cat_id
                entry["suggested_category_name"] = cat_id_to_name.get(
                    cat_id, "unknown_product"
                )
                entry["suggestion_conf"] = round(scores[j].item(), 4)
            dets_list.append(entry)

    output = {"width": orig_w, "height": orig_h, "detections": dets_list}
    print(json.dumps(output))
    print(f"Done: {len(dets_list)} detections", file=_sys.stderr)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--single":
        process_single(sys.argv[2])
    else:
        main()
