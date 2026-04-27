# Small CNN for Image Classification

6-Block Deep CNN for 64x64 image classification, designed for deployment on Spinnaker neuromorphic chips.

## Architecture

**DeepCNN6Block** - A 6-block convolutional neural network with no skip connections (Spinnaker compatible).

| Block | Channels | Output Size |
|-------|----------|-------------|
| 1 | 3 → 32 | 32×32 |
| 2 | 32 → 64 | 16×16 |
| 3 | 64 → 128 | 8×8 |
| 4 | 128 → 256 | 4×4 |
| 5 | 256 → 256 | 2×2 |
| 6 | 256 → 256 | 1×1 |

**Classifier:** Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→9)

- **Input size:** 64×64 RGB images
- **Output:** 9 classes
- **Parameters:** ~1.5M

## Files

| File | Description |
|------|-------------|
| `train_deepcnn_6block.py` | Training script with early stopping |
| `quantize_deepcnn_int8.py` | INT8 post-training quantization |
| `inference_deepcnn_compare.py` | Compare FP32 vs INT8 accuracy |
| `inference_smallcnn_new.py` | Original 3-block SmallCNN inference (32x32) |

## Quick Start

### 1. Train the Model

```bash
python train_deepcnn_6block.py
```

**Configuration** (edit in script):
- `DATA_PATH`: Path to dataset with train/test folders
- `EPOCHS`: Max epochs (default: 50)
- `EARLY_STOP_PATIENCE`: Stop after N epochs without improvement (default: 2)
- `BATCH_SIZE`: Batch size (default: 64)
- `DEVICE`: "cuda" or "cpu"

### 2. Quantize to INT8

```bash
python quantize_deepcnn_int8.py
```

This performs post-training static quantization:
1. Loads trained FP32 model
2. Fuses Conv+BN+ReLU layers
3. Calibrates with training data (1000 samples)
4. Converts to INT8

### 3. Compare Models

```bash
python inference_deepcnn_compare.py
```

Outputs accuracy comparison and per-class metrics for both FP32 and INT8 models.

## Results

| Metric | FP32 | INT8 |
|--------|------|------|
| Model Size | 6.27 MB | 1.60 MB |
| Compression | 1x | 3.92x |
| Accuracy | 65.86% | 65.78% |
| Accuracy Drop | - | 0.09% |

## Dataset Structure

Expected ImageFolder format:
```
DOES/
├── train/
│   ├── BACKGROUND/
│   ├── E1/
│   ├── E2/
│   ├── E3/
│   ├── E5H/
│   ├── E6/
│   ├── E8/
│   ├── E40/
│   └── EHRB/
└── test/
    ├── BACKGROUND/
    ├── E1/
    └── ...
```

## Checkpoints

Models are saved to `deepcnn_6block_checkpoints/`:

| File | Description |
|------|-------------|
| `best_model.pth` | Best FP32 model (state dict) |
| `best_model_int8.pth` | INT8 quantized model (state dict) |
| `checkpoint_epoch_*.pth` | Training checkpoints |
| `quantization_summary.txt` | Quantization metrics |
| `inference_comparison_results.txt` | Full comparison report |

## Loading Models

### FP32 Model
```python
from train_deepcnn_6block import DeepCNN6Block

model = DeepCNN6Block(num_classes=9)
model.load_state_dict(torch.load("deepcnn_6block_checkpoints/best_model.pth"))
model.eval()
```

### INT8 Model
```python
from quantize_deepcnn_int8 import DeepCNN6BlockQuantizable
import torch.quantization as quant

model = DeepCNN6BlockQuantizable(num_classes=9)
model.eval()
model.fuse_model()
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)
quant.convert(model, inplace=True)
model.load_state_dict(torch.load("deepcnn_6block_checkpoints/best_model_int8.pth"))
```

## Data Transforms

```python
from torchvision import transforms

# Training (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Alternative Models (Backup Options)

All options below are smaller than VGG16 (138M params / 528 MB):

| Model | Parameters | Size | Notes |
|-------|------------|------|-------|
| **ResNet-50** | 25.6M | ~98 MB | Current choice, best accuracy |
| ResNet-34 | 21.8M | ~85 MB | Lighter, basic blocks |
| ResNet-18 | 11.7M | ~45 MB | Smallest ResNet |
| MobileNetV2 | 3.4M | ~14 MB | Edge device optimized |
| EfficientNet-B0 | 5.3M | ~21 MB | Best accuracy/size ratio |

## Notes

- ResNet-50 uses skip connections (now compatible with deployment)
- INT8 quantization uses PyTorch's fbgemm backend (x86)
- For ARM deployment, change to 'qnnpack' backend
- Early stopping monitors validation accuracy
