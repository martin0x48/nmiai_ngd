# NorgesGruppen Grocery Shelf Detection

Object detection model for identifying grocery products on store shelves. Fine-tunes YOLOv8x on COCO-format shelf annotations across 356 product categories.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (pinned to match sandbox versions)
pip install -r requirements.txt
```

For GPU training with CUDA:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.1.0
```

## Data Setup

Place the downloaded datasets in `data/`:

```
data/
├── coco/
│   ├── images/           ← unzip NM_NGD_coco_dataset.zip here
│   │   ├── img_00001.jpg
│   │   └── ...
│   └── annotations.json
└── product_images/       ← unzip NM_NGD_product_images.zip here
    ├── 7040913336691/
    │   ├── main.jpg
    │   ├── front.jpg
    │   └── ...
    └── metadata.json
```

## Workflow

### 1. Explore the data

```bash
python src/explore_data.py --coco_dir data/coco --product_dir data/product_images --top_n 20
```

Shows the rarest product categories with shelf crops and reference photos.

### 2. Convert COCO to YOLO format

```bash
python src/convert_coco_to_yolo.py --coco_dir data/coco --output_dir data/yolo
```

Creates `data/yolo/` with train/val split and generates `config.yaml`.

### 3. Train

```bash
python src/train.py --epochs 100 --batch 8 --imgsz 640
```

Trains YOLOv8x with COCO-pretrained weights. Best model saved to `runs/detect/train/weights/best.pt`.

### 4. Evaluate locally

```bash
python src/evaluate.py --model runs/detect/train/weights/best.pt --save_images
```

Reports the competition-style combined score: `0.7 * detection_mAP + 0.3 * classification_mAP`.

### 5. Package submission

```bash
python src/package_submission.py --weights runs/detect/train/weights/best.pt
```

Creates `submission.zip` with `run.py` and `best.pt` at the root. Validates size and file constraints.

### 6. Submit

Upload `submission.zip` on the competition submit page. The sandbox runs:

```
python run.py --input /data/images --output /output/predictions.json
```

## Sandbox Environment

| Resource    | Value                        |
|-------------|------------------------------|
| GPU         | NVIDIA L4 (24 GB VRAM)       |
| Python      | 3.11                         |
| ultralytics | 8.1.0                        |
| torch       | 2.6.0+cu124                  |
| Timeout     | 300 seconds                  |
| Network     | None (fully offline)         |
| Max zip     | 420 MB uncompressed          |

## Scoring

```
Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
```

- **Detection** (70%): did you find the products? (IoU >= 0.5, category ignored)
- **Classification** (30%): did you identify the right product? (IoU >= 0.5 AND correct category_id)
