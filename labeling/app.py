"""Flask labeling UI for meny shelf images.

Usage (from norgesgruppen-cv/):
    python labeling/app.py

Prerequisites:
    pip install flask pillow
    python labeling/detect.py   (run detection first)

Opens http://localhost:5000
"""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MENY_DIR = DATA_DIR / "meny"
COCO_PATH = DATA_DIR / "coco" / "annotations.json"
PRODUCT_IMG_DIR = DATA_DIR / "product_images"
METADATA_PATH = PRODUCT_IMG_DIR / "metadata.json"
DETECTIONS_PATH = MENY_DIR / "detections.json"
LABELS_PATH = MENY_DIR / "labels.json"

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))


def _load_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── bootstrap ──────────────────────────────────────────────
_coco = _load_json(COCO_PATH) or {"categories": []}
CATEGORIES = {c["id"]: c["name"] for c in _coco["categories"]}
CAT_LIST = sorted(_coco["categories"], key=lambda c: c["name"])

_meta = _load_json(METADATA_PATH) or {}
PRODUCT_BY_NAME = {p["product_name"]: p for p in _meta.get("products", [])}

_det_raw = _load_json(DETECTIONS_PATH) or {}
DETECTIONS = {}
IMAGE_DIMS = {}
for _p, _v in _det_raw.items():
    if isinstance(_v, dict):
        DETECTIONS[_p] = _v.get("detections", [])
        IMAGE_DIMS[_p] = (_v.get("width", 0), _v.get("height", 0))
    else:
        DETECTIONS[_p] = _v
        IMAGE_DIMS[_p] = (0, 0)

LABELS = _load_json(LABELS_PATH) or {}


# ── routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    if not DETECTIONS:
        return (
            "<h2>No detections found.</h2>"
            "<p>Run detection first:</p>"
            "<pre>cd norgesgruppen-cv\npython labeling/detect.py</pre>"
        ), 200
    return render_template("index.html")


@app.route("/api/categories")
def api_categories():
    result = []
    for c in CAT_LIST:
        entry = {"id": c["id"], "name": c["name"]}
        prod = PRODUCT_BY_NAME.get(c["name"])
        if prod and prod.get("has_images"):
            entry["product_code"] = prod["product_code"]
            entry["image_types"] = prod.get("image_types", [])
        result.append(entry)
    return jsonify(result)


@app.route("/api/images")
def api_images():
    folders = {}
    for rel_path in sorted(DETECTIONS.keys()):
        dets = DETECTIONS[rel_path]
        parts = rel_path.replace("\\", "/").split("/")
        folder = parts[0] if len(parts) > 1 else "_root"
        img_labels = LABELS.get(rel_path, [])
        labeled = (
            sum(1 for lb in img_labels if lb is not None)
            if isinstance(img_labels, list) else 0
        )
        folders.setdefault(folder, []).append({
            "path": rel_path,
            "name": parts[-1],
            "detections": len(dets),
            "labeled": labeled,
        })
    return jsonify(folders)


@app.route("/api/image/<path:img_path>")
def api_image(img_path):
    full = MENY_DIR / img_path
    if not full.exists():
        return "Not found", 404
    max_w = request.args.get("w", type=int)
    if max_w:
        img = Image.open(full)
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    return send_file(full, mimetype="image/jpeg")


@app.route("/api/detections/<path:img_path>")
def api_detections(img_path):
    dets = DETECTIONS.get(img_path, [])
    dims = IMAGE_DIMS.get(img_path, (0, 0))
    if dims == (0, 0):
        full = MENY_DIR / img_path
        if full.exists():
            with Image.open(full) as im:
                dims = im.size

    img_labels = LABELS.get(img_path, [])
    while len(img_labels) < len(dets):
        img_labels.append(None)

    result = []
    for i, det in enumerate(dets):
        entry = dict(det, idx=i)
        if i < len(img_labels) and img_labels[i] is not None:
            entry["label"] = img_labels[i]
        result.append(entry)

    return jsonify({"width": dims[0], "height": dims[1], "detections": result})


@app.route("/api/crop/<path:img_path>/<int:idx>")
def api_crop(img_path, idx):
    full = MENY_DIR / img_path
    dets = DETECTIONS.get(img_path, [])
    if idx >= len(dets) or not full.exists():
        return "Not found", 404

    bbox = dets[idx]["bbox_xyxy"]
    img = Image.open(full).convert("RGB")
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad = 0.08
    crop = img.crop((
        max(0, int(x1 - w * pad)),
        max(0, int(y1 - h * pad)),
        min(img.width, int(x2 + w * pad)),
        min(img.height, int(y2 + h * pad)),
    ))

    max_dim = request.args.get("size", 300, type=int)
    if max(crop.size) > max_dim:
        r = max_dim / max(crop.size)
        crop = crop.resize((int(crop.width * r), int(crop.height * r)), Image.LANCZOS)

    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/product-image/<code>/<img_type>")
def api_product_image(code, img_type):
    prod_dir = PRODUCT_IMG_DIR / code
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = prod_dir / f"{img_type}{ext}"
        if p.exists():
            return send_file(str(p))
    return "Not found", 404


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.json
    img_path = data["image_path"]
    idx = data["idx"]
    dets = DETECTIONS.get(img_path, [])
    if idx >= len(dets):
        return jsonify(error="bad index"), 400

    if img_path not in LABELS:
        LABELS[img_path] = [None] * len(dets)
    while len(LABELS[img_path]) < len(dets):
        LABELS[img_path].append(None)

    LABELS[img_path][idx] = {
        "category_id": data.get("category_id"),
        "category_name": data.get("category_name"),
    }
    _save_json(LABELS_PATH, LABELS)
    return jsonify(ok=True)


@app.route("/api/clear-label", methods=["POST"])
def api_clear_label():
    data = request.json
    img_path = data["image_path"]
    idx = data["idx"]
    if img_path in LABELS and idx < len(LABELS[img_path]):
        LABELS[img_path][idx] = None
        _save_json(LABELS_PATH, LABELS)
    return jsonify(ok=True)


@app.route("/api/clear-all-labels", methods=["POST"])
def api_clear_all_labels():
    data = request.json
    img_path = data["image_path"]
    if img_path in LABELS:
        del LABELS[img_path]
        _save_json(LABELS_PATH, LABELS)
    return jsonify(ok=True)


@app.route("/api/delete-detection", methods=["POST"])
def api_delete_detection():
    data = request.json
    img_path = data["image_path"]
    idx = data["idx"]
    dets = DETECTIONS.get(img_path, [])
    if idx >= len(dets):
        return jsonify(error="bad index"), 400

    if img_path not in LABELS:
        LABELS[img_path] = [None] * len(dets)
    while len(LABELS[img_path]) < len(dets):
        LABELS[img_path].append(None)

    LABELS[img_path][idx] = {
        "category_id": -1,
        "category_name": "__false_positive__",
    }
    _save_json(LABELS_PATH, LABELS)
    return jsonify(ok=True)


def _match_detections(cur_dets, ref_dets, ref_labels, cur_dims, ref_dims):
    """Greedy matching: pair current detections to labeled reference detections
    using classifier prediction agreement + normalized spatial distance."""

    def nc(det, dims):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        return ((x1 + x2) / 2 / max(dims[0], 1),
                (y1 + y2) / 2 / max(dims[1], 1))

    pairs = []
    for ci, cd in enumerate(cur_dets):
        cc = nc(cd, cur_dims)
        cur_sug = cd.get("suggested_category_id")

        for ri, rd in enumerate(ref_dets):
            if ri >= len(ref_labels) or ref_labels[ri] is None:
                continue
            rl = ref_labels[ri]
            if rl.get("category_id") in (None, -1):
                continue

            rc = nc(rd, ref_dims)
            dist = ((cc[0] - rc[0]) ** 2 + (cc[1] - rc[1]) ** 2) ** 0.5

            if cur_sug == rl.get("category_id"):
                score = dist
            elif cur_sug == rd.get("suggested_category_id"):
                score = dist + 0.1
            else:
                score = dist + 0.3

            if dist < 0.30:
                pairs.append((score, ci, ri))

    pairs.sort()
    matches, used_c, used_r = [], set(), set()
    for score, ci, ri in pairs:
        if ci in used_c or ri in used_r:
            continue
        matches.append((ci, ri))
        used_c.add(ci)
        used_r.add(ri)
    return matches


@app.route("/api/match-shelf", methods=["POST"])
def api_match_shelf():
    data = request.json
    cur_path = data["current"]
    ref_path = data["reference"]

    cur_dets = DETECTIONS.get(cur_path, [])
    ref_dets = DETECTIONS.get(ref_path, [])
    ref_labels = LABELS.get(ref_path, [])

    if not ref_labels or not cur_dets or not ref_dets:
        return jsonify(applied=0, total=len(cur_dets))

    cur_dims = IMAGE_DIMS.get(cur_path, (1, 1))
    ref_dims = IMAGE_DIMS.get(ref_path, (1, 1))

    if cur_path not in LABELS:
        LABELS[cur_path] = [None] * len(cur_dets)
    while len(LABELS[cur_path]) < len(cur_dets):
        LABELS[cur_path].append(None)

    matches = _match_detections(cur_dets, ref_dets, ref_labels, cur_dims, ref_dims)

    applied = 0
    for ci, ri in matches:
        if LABELS[cur_path][ci] is not None:
            continue
        if ri < len(ref_labels) and ref_labels[ri] is not None:
            lb = ref_labels[ri]
            if lb.get("category_id") not in (None, -1):
                LABELS[cur_path][ci] = dict(lb)
                applied += 1

    _save_json(LABELS_PATH, LABELS)
    return jsonify(applied=applied, total=len(cur_dets))


@app.route("/api/redetect/<path:img_path>", methods=["POST"])
def api_redetect(img_path):
    full = MENY_DIR / img_path
    if not full.exists():
        return jsonify(error="Image not found"), 404

    data = request.get_json(silent=True) or {}
    rot = int(data.get("rotation", 0))

    if rot:
        img = Image.open(full)
        transpose_map = {
            90: Image.Transpose.ROTATE_270,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,
        }
        img = img.transpose(transpose_map[rot])
        img.save(str(full), quality=95)

    detect_python = "python"
    result = subprocess.run(
        [detect_python, "labeling/detect.py", "--single", img_path],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180,
    )
    if result.returncode != 0:
        return jsonify(error=result.stderr[-500:] if result.stderr else "unknown"), 500

    try:
        new_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return jsonify(error="Invalid JSON from detector"), 500

    if "error" in new_data:
        return jsonify(error=new_data["error"]), 500

    DETECTIONS[img_path] = new_data.get("detections", [])
    IMAGE_DIMS[img_path] = (new_data.get("width", 0), new_data.get("height", 0))

    if img_path in LABELS:
        del LABELS[img_path]
        _save_json(LABELS_PATH, LABELS)

    det_raw = _load_json(DETECTIONS_PATH) or {}
    det_raw[img_path] = new_data
    _save_json(DETECTIONS_PATH, det_raw)

    return jsonify(detections=len(new_data.get("detections", [])))


@app.route("/api/export", methods=["POST"])
def api_export():
    images_out, anns_out = [], []
    ann_id = img_id = 1

    for img_path, img_labels in sorted(LABELS.items()):
        if not isinstance(img_labels, list):
            continue
        dets = DETECTIONS.get(img_path, [])
        has_any = any(
            lb and lb.get("category_id") not in (None, -1)
            for lb in img_labels
        )
        if not has_any:
            continue

        dims = IMAGE_DIMS.get(img_path, (0, 0))
        if dims == (0, 0):
            full = MENY_DIR / img_path
            if full.exists():
                with Image.open(full) as im:
                    dims = im.size

        images_out.append({
            "id": img_id,
            "file_name": img_path,
            "width": dims[0],
            "height": dims[1],
        })

        for i, lb in enumerate(img_labels):
            if not lb or lb.get("category_id") in (None, -1) or i >= len(dets):
                continue
            x1, y1, x2, y2 = dets[i]["bbox_xyxy"]
            bw, bh = x2 - x1, y2 - y1
            anns_out.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": lb["category_id"],
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
            })
            ann_id += 1
        img_id += 1

    coco_out = {
        "images": images_out,
        "annotations": anns_out,
        "categories": [
            {"id": c["id"], "name": c["name"], "supercategory": "product"}
            for c in _coco["categories"]
        ],
    }
    out_path = MENY_DIR / "annotations_coco.json"
    _save_json(out_path, coco_out)
    return jsonify(
        path=str(out_path),
        images=len(images_out),
        annotations=len(anns_out),
    )


if __name__ == "__main__":
    n_det = sum(len(d) for d in DETECTIONS.values())
    n_lbl = sum(
        sum(1 for lb in lbs if lb is not None)
        for lbs in LABELS.values()
        if isinstance(lbs, list)
    )
    print(f"  Categories : {len(CATEGORIES)}")
    print(f"  Detections : {n_det} across {len(DETECTIONS)} images")
    print(f"  Labels     : {n_lbl} saved")
    print(f"  Products   : {len(PRODUCT_BY_NAME)} with reference images")
    print()
    print("  Open http://localhost:5000")
    print()
    app.run(debug=False, port=5000)
