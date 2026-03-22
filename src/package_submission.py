"""Build a submission.zip for the two-stage pipeline.

Strips and converts model weights to FP16, copies run.py and the class mapping,
then validates against sandbox constraints.

Usage:
    python src/package_submission.py \
        --detector runs/detect/train/weights/best.pt \
        --classifier runs/classifier/best_classifier.pt \
        --classifier_mapping runs/classifier/idx_to_class.json \
        --output submission.zip
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import torch

MAX_ZIP_SIZE_MB = 420
MAX_FILES = 1000
MAX_PY_FILES = 10
MAX_WEIGHT_FILES = 3
WEIGHT_EXTENSIONS = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".cfg"} | WEIGHT_EXTENSIONS


def strip_yolo_checkpoint(src_path: Path, dst_path: Path):
    """Strip everything except model weights from a YOLO checkpoint, convert to FP16."""
    ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)

    model = ckpt.get("ema", ckpt.get("model"))
    if model is None:
        print(f"  WARNING: Could not find 'model' or 'ema' key in {src_path}")
        print(f"  Keys found: {list(ckpt.keys())}")
        shutil.copy2(src_path, dst_path)
        return

    dropped = [k for k in ckpt if k not in ("model", "ema")]
    if dropped:
        print(f"  Dropping keys: {dropped}")

    torch.save({"model": model.half()}, str(dst_path))

    src_mb = src_path.stat().st_size / (1024 * 1024)
    dst_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  Detector: {src_mb:.1f} MB → {dst_mb:.1f} MB (FP16, weights only)")


def strip_classifier_checkpoint(src_path: Path, dst_path: Path):
    """Keep only model parameter tensors, convert to FP16."""
    data = torch.load(str(src_path), map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        state_dict = {
            k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v
            for k, v in data.items()
            if torch.is_tensor(v)
        }
        non_tensor = [k for k, v in data.items() if not torch.is_tensor(v)]
        if non_tensor:
            print(f"  Dropping non-tensor keys: {non_tensor}")
    else:
        state_dict = data

    torch.save(state_dict, str(dst_path))

    src_mb = src_path.stat().st_size / (1024 * 1024)
    dst_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"  Classifier: {src_mb:.1f} MB → {dst_mb:.1f} MB (FP16, weights only)")


def main():
    parser = argparse.ArgumentParser(description="Package two-stage submission zip")
    parser.add_argument(
        "--detector", type=str,
        default="runs/detect/train/weights/best.pt",
        help="Path to detector YOLO checkpoint",
    )
    parser.add_argument(
        "--classifier", type=str,
        default="runs/classifier/best_classifier.pt",
        help="Path to classifier state_dict",
    )
    parser.add_argument(
        "--classifier_mapping", type=str,
        default="runs/classifier/idx_to_class.json",
        help="Path to idx_to_class.json",
    )
    parser.add_argument(
        "--output", type=str, default="submission.zip",
        help="Output zip file path",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    submission_dir = project_root / "submission"

    detector_path = Path(args.detector)
    classifier_path = Path(args.classifier)
    mapping_path = Path(args.classifier_mapping)

    for p, name in [(detector_path, "Detector"), (classifier_path, "Classifier"),
                     (mapping_path, "Mapping")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            return

    run_py = project_root / "run.py"
    if not run_py.exists():
        print(f"ERROR: run.py not found at project root: {run_py}")
        return

    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True)

    # Copy run.py
    shutil.copy2(run_py, submission_dir / "run.py")

    # Copy mapping
    shutil.copy2(mapping_path, submission_dir / "idx_to_class.json")

    # Strip and copy weights
    print("Processing weights...")
    strip_yolo_checkpoint(detector_path, submission_dir / "detector.pt")
    strip_classifier_checkpoint(classifier_path, submission_dir / "classifier.pt")

    # --- Validation ---
    print("\nSubmission contents:")
    files = [f for f in submission_dir.rglob("*") if f.is_file()]

    total_size = 0
    py_count = 0
    weight_count = 0

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        suffix = f.suffix.lower()
        if suffix == ".py":
            py_count += 1
        if suffix in WEIGHT_EXTENSIONS:
            weight_count += 1
        print(f"  {f.relative_to(submission_dir)}  ({size_mb:.1f} MB)")

    print(f"\nTotal size: {total_size:.1f} MB / {MAX_ZIP_SIZE_MB} MB")
    print(f"Files: {len(files)} / {MAX_FILES}")
    print(f"Python files: {py_count} / {MAX_PY_FILES}")
    print(f"Weight files: {weight_count} / {MAX_WEIGHT_FILES}")

    errors = []
    if total_size > MAX_ZIP_SIZE_MB:
        errors.append(f"Total size {total_size:.1f} MB exceeds {MAX_ZIP_SIZE_MB} MB limit")
    if len(files) > MAX_FILES:
        errors.append(f"File count {len(files)} exceeds {MAX_FILES} limit")
    if py_count > MAX_PY_FILES:
        errors.append(f"Python file count {py_count} exceeds {MAX_PY_FILES} limit")
    if weight_count > MAX_WEIGHT_FILES:
        errors.append(f"Weight file count {weight_count} exceeds {MAX_WEIGHT_FILES} limit")

    for f in files:
        if f.suffix.lower() not in ALLOWED_EXTENSIONS:
            errors.append(f"Disallowed file type: {f.name}")

    if not (submission_dir / "run.py").exists():
        errors.append("run.py missing from submission root")

    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print("\nSubmission NOT created.")
        return

    # Create zip
    zip_path = Path(args.output)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arcname = f.relative_to(submission_dir)
            zf.write(f, arcname)

    print(f"\nCreated: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB compressed)")
    print("\nZip contents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename}  ({info.file_size / (1024*1024):.1f} MB uncompressed)")

    print("\nSubmission ready for upload!")


if __name__ == "__main__":
    main()
