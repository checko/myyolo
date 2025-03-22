# YOLO Object Detection Project

This project implements YOLOv4 (You Only Look Once version 4) for object detection using PyTorch. The implementation includes training on the Pascal VOC dataset and features modern deep learning techniques such as CSPDarknet53 backbone, PANet feature aggregation, and advanced data augmentation strategies.

## Project Structure
```
.
├── configs/
│   └── model_config.py     # Model and training configuration
├── data/
│   ├── dataset.py          # VOC dataset implementation
│   └── transforms.py       # Data augmentation and transforms
├── models/
│   ├── backbone.py         # CSPDarknet53 implementation
│   ├── neck.py            # Feature pyramid network (PANet)
│   ├── head.py            # YOLO detection heads
│   └── yolo.py            # Main YOLO model
├── outputs/
│   ├── weights/           # Saved model checkpoints
│   ├── tensorboard/       # Training logs
│   └── visualizations/    # Detection visualizations
├── train.py               # Training script
├── eval.py                # Evaluation script
└── requirements.txt       # Python dependencies
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Pascal VOC dataset:
```bash
# Create data directory
mkdir -p data/pascal_voc

# Download and extract Pascal VOC 2012
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar -C data/pascal_voc
```

## Training

To train the model:

```bash
python train.py
```

Training configurations can be modified in `configs/model_config.py`. The script will:
- Save model checkpoints in `outputs/weights/`
- Log training metrics to TensorBoard
- Use data augmentation including Mosaic, CutMix, and other techniques
- Implement multi-scale training

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir outputs/tensorboard
```

## Evaluation

To evaluate a trained model:

```bash
python eval.py --checkpoint outputs/weights/best.pth
```

This will:
- Run inference on the test set
- Save visualizations to `outputs/visualizations/`
- Print per-class detection statistics

## Model Architecture

### Backbone
- CSPDarknet53 with Cross Stage Partial Networks
- Mish activation function
- Skip connections and residual blocks

### Neck
- PANet (Path Aggregation Network)
- SPP (Spatial Pyramid Pooling)
- Feature pyramid with both top-down and bottom-up paths

### Head
- Three detection heads for different scales
- Anchor-based detection
- CIoU loss for bounding box regression

## Features

- Advanced data augmentation:
  - Mosaic augmentation
  - CutMix
  - Random geometric transforms
  - Color jittering
- Modern training techniques:
  - AdamW optimizer
  - OneCycle learning rate scheduler
  - Warmup epochs
- TensorBoard integration:
  - Loss tracking
  - Learning rate monitoring
  - Performance metrics

## Requirements

Main dependencies:
- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- OpenCV
- Albumentations
- TensorBoard
- NumPy
- tqdm

See `requirements.txt` for complete list.

## License

This project is released under the MIT License.
