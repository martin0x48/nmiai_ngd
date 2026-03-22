"""Two-stage submission: YOLOv8x detector (nc=1) + EfficientNet-V2-M classifier.
Includes unknown_product predictions (category_id 355).

The sandbox executes:
    python run.py --input /data/images --output /output/predictions.json
"""

import torch

_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load

import argparse
import json
from pathlib import Path

import numpy as np
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

CLASSIFIER_IMGSZ = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_classifier(weights_path: Path, mapping_path: Path, device: torch.device):
    with open(mapping_path) as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): int(v) for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)
    model = timm.create_model(
        "tf_efficientnetv2_m", pretrained=False, num_classes=num_classes
    )
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval().half()
    return model, idx_to_class


def crop_and_preprocess(pil_image: Image.Image, bbox_xyxy, pad_pct: float = 0.05):
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    iw, ih = pil_image.size
    pad_x = w * pad_pct
    pad_y = h * pad_pct
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(iw, int(x2 + pad_x))
    cy2 = min(ih, int(y2 + pad_y))
    return pil_image.crop((cx1, cy1, cx2, cy2))


def build_classifier_transform():
    return transforms.Compose([
        transforms.Resize(int(CLASSIFIER_IMGSZ * 1.14)),
        transforms.CenterCrop(CLASSIFIER_IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def detect_with_wbf(detector, img_path, pil_image, conf=0.05, iou_thr=0.7):
    iw, ih = pil_image.size
    all_boxes_list = []
    all_scores_list = []
    all_labels_list = []

    for augment in [True, False]:
        results = detector.predict(source=str(img_path), conf=conf, augment=augment, verbose=False)
        boxes_norm, scores, labels = [], [], []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_norm.append([x1 / iw, y1 / ih, x2 / iw, y2 / ih])
                scores.append(float(box.conf.item()))
                labels.append(0)
        all_boxes_list.append(boxes_norm)
        all_scores_list.append(scores)
        all_labels_list.append(labels)

    if all(len(b) == 0 for b in all_boxes_list):
        return [], []

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        all_boxes_list, all_scores_list, all_labels_list,
        weights=[1.0, 1.0], iou_thr=iou_thr, skip_box_thr=0.01,
    )

    boxes_xyxy = []
    det_scores = []
    for box, score in zip(fused_boxes, fused_scores):
        x1, y1, x2, y2 = box
        boxes_xyxy.append([x1 * iw, y1 * ih, x2 * iw, y2 * ih])
        det_scores.append(float(score))

    return boxes_xyxy, det_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = YOLO(str(root / "detector.pt"))
    classifier, idx_to_class = load_classifier(
        root / "classifier.pt", root / "idx_to_class.json", device,
    )
    clf_transform = build_classifier_transform()

    input_dir = Path(args.input)
    predictions = []

    img_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in img_paths:
        image_id = int(img_path.stem.split("_")[-1])
        pil_image = Image.open(img_path).convert("RGB")

        boxes_xyxy, det_scores = detect_with_wbf(detector, img_path, pil_image)

        if not boxes_xyxy:
            continue

        crop_tensors = []
        for bbox in boxes_xyxy:
            crop = crop_and_preprocess(pil_image, bbox)
            crop_tensors.append(clf_transform(crop))

        batch = torch.stack(crop_tensors).to(device).half()

        with torch.no_grad():
            logits = classifier(batch)
            probs = torch.softmax(logits, dim=1)
            class_scores, class_indices = probs.max(dim=1)

        for i, bbox in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = bbox
            model_idx = class_indices[i].item()
            category_id = idx_to_class.get(model_idx, 0)
            cls_score = class_scores[i].item()
            det_score = det_scores[i]

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": round(det_score * cls_score, 4),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
