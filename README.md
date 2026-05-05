# CCTV Person Detection

Training and evaluation pipeline for person detection from CCTV footage using Faster R-CNN, SSD, and YOLO models.

## Setup

```bash
# Create virtual environment
python -m venv myenv
myenv\Scripts\activate

# Install dependencies
pip install torch torchvision ultralytics opencv-python matplotlib
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/compare-models.ipynb` | Train and evaluate Faster R-CNN, SSD, and YOLO on custom dataset |
| `notebooks/compare-yolo-versions.ipynb` | Benchmark YOLOv8n/s/m/l/nano and export production ONNX models |

## Training

All notebooks support:
- Training from scratch
- Resuming from latest `.pth` checkpoint
- Evaluation with mAP, precision, recall metrics
- ONNX export for CPU deployment

## Environment

- GPU: NVIDIA RTX 2000 Ada (16 GB)
- CUDA: 13.0
- Python: 3.12.5
- Batch size: 16
- Image size: 544x544

## Production

Exported ONNX models are saved to `production/` for use with FastAPI + ONNX Runtime CPU (`CPUExecutionProvider`).

## Dataset

- Format: YOLOv8 / COCO
- Classes: 1 (person only)
- Versions: v1, v3 available in `datasets/`
