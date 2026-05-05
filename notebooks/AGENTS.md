# AGENTS.md — D:\cctv\stage\notebooks

## Project overview
Jupyter notebooks for benchmarking YOLO object detection models (v8, v10, v11, v26) on CCTV person-detection data. Single-class detection (person only).

## Environment
- **Python venv**: `D:\cctv\myenv` (Python 3.12.5)
- **Key packages**: `ultralytics` 8.4.x, `torch` 2.5.1+cu121, `torchvision`, `pandas`, `matplotlib`, `roboflow`
- **GPU**: NVIDIA RTX 2000 Ada (16 GB VRAM), CUDA 13.0

## Key paths
| Path | Purpose |
|---|---|
| `D:\cctv\stage\datasets\People.v1i.yolov8\` | Primary dataset (v1, COCO-style) |
| `D:\cctv\stage\datasets\People.v3i.yolov8\` | Dataset v3 (5005 train / 395 val images) |
| `D:\ECG generation\runs\detect\` | YOLO training outputs (non-obvious default) |
| `D:\cctv\stage\notebooks\` | Notebooks + `.pt` model weights |

## Notebook inventory
| File | Status | Notes |
|---|---|---|
| `compare-yolo-versions.ipynb` | Partially executed | Trains yolov8n/v10n/v11n/yolo26n, collects mAP+latency, plots results. Interrupted on first model (KeyboardInterrupt). |
| `compare-models.ipynb` | Broken | Custom FasterRCNNDataset + Faster R-CNN + SSD training code. **Bugs**: (1) `__getitem__` does not join `img_dir` to filename — causes `FileNotFoundError`. (2) `load_annotations` method is referenced but never defined. |

## Known issues to fix before re-running
1. **FasterRCNNDataset path bug** (`compare-models.ipynb`): `img_path = self.imgs[idx]` yields a bare filename. Must be `img_path = os.path.join(self.img_dir, self.imgs[idx])`.
2. **Missing `load_annotations`** (`compare-models.ipynb`): Method called in `__getitem__` but never implemented. Need to parse COCO-format labels from `label_dir`.
3. **Escape-sequence warnings**: Paths like `'D:\cctv\stage\datasets\...'` trigger `invalid escape sequence '\c'`. Use raw strings (`r'...'`) or forward slashes.
4. **YOLO `imgsz` stride warning**: `imgsz=512` is not a multiple of 32. YOLO auto-rounds to 544. Use `imgsz=512` (ultralytics handles it) or set `imgsz=544` explicitly to suppress the warning.

## Conventions
- Datasets use YOLOv8 format: `data.yaml` in dataset root, `{train,val}/images/` and `{train,val}/labels/`.
- Model weights (`.pt`) live alongside notebooks — gitignored by convention, not committed.
- `roboflow.zip` contains dataset export; already unpacked to `datasets/`.
