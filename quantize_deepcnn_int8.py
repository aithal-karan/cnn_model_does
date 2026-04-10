import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

# ============================================
# 6-Block Deep CNN (Same architecture as training)
# ============================================

class DeepCNN6Block(nn.Module):
    """
    6-block CNN for 64x64 input images.
    Each block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """
    def __init__(self, num_classes=9):
        super().__init__()

        # Block 1: 3 -> 32 channels, 64x64 -> 32x32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 2: 32 -> 64 channels, 32x32 -> 16x16
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 3: 64 -> 128 channels, 16x16 -> 8x8
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 4: 128 -> 256 channels, 8x8 -> 4x4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 5: 256 -> 256 channels, 4x4 -> 2x2
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 6: 256 -> 256 channels, 2x2 -> 1x1
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classifier: 256 * 1 * 1 -> num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        return x


# ============================================
# Quantization-Ready Model (with QuantStub/DeQuantStub)
# ============================================

class DeepCNN6BlockQuantizable(nn.Module):
    """
    Quantization-ready version with QuantStub and DeQuantStub.
    """
    def __init__(self, num_classes=9):
        super().__init__()

        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 5
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 6
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classifier (no dropout for quantization)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU for better quantization performance."""
        for block_name in ['block1', 'block2', 'block3', 'block4', 'block5', 'block6']:
            block = getattr(self, block_name)
            torch.quantization.fuse_modules(block, ['0', '1', '2'], inplace=True)


# ============================================
# Configuration
# ============================================
MODEL_PATH = "/netscratch/krajshekar/ESCADE/small_cnn/deepcnn_6block_checkpoints/best_model.pth"
CALIBRATION_PATH = "/netscratch/krajshekar/ESCADE/DOES/train"
TEST_PATH = "/netscratch/krajshekar/ESCADE/DOES/test"
SAVE_DIR = "/netscratch/krajshekar/ESCADE/small_cnn/deepcnn_6block_checkpoints"
NUM_CLASSES = 9
BATCH_SIZE = 32
CALIBRATION_SAMPLES = 1000  # Number of samples for calibration
DEVICE = "cpu"  # Quantization must be done on CPU

# ============================================
# Data Transform (same as validation)
# ============================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_calibration_loader(data_path, num_samples):
    """Get a subset of data for calibration."""
    dataset = ImageFolder(data_path, transform=transform)

    # Sample evenly from each class
    targets = np.array(dataset.targets)
    samples_per_class = num_samples // NUM_CLASSES

    indices = []
    for class_id in range(NUM_CLASSES):
        class_indices = np.where(targets == class_id)[0]
        sampled = np.random.choice(class_indices, size=min(samples_per_class, len(class_indices)), replace=False)
        indices.extend(sampled)

    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)


def evaluate(model, loader, device="cpu"):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    return accuracy, all_preds, all_labels


def get_model_size(model):
    """Get model size in MB."""
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size_mb


# ============================================
# Main Quantization Pipeline
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("INT8 Post-Training Static Quantization")
    print("=" * 60)

    # Load test dataset
    test_dataset = ImageFolder(TEST_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")

    # ============================================
    # Step 1: Load FP32 Model
    # ============================================
    print("\n[1/5] Loading FP32 model...")

    # Load into quantizable model
    model_fp32 = DeepCNN6BlockQuantizable(num_classes=NUM_CLASSES)

    # Load weights from original model (need to handle dropout difference)
    original_state = torch.load(MODEL_PATH, map_location="cpu")

    # Map weights correctly - the trained model has Dropout which shifts classifier indices
    # Trained model: classifier.0=Flatten, classifier.1=Linear, classifier.2=ReLU, classifier.3=Dropout, classifier.4=Linear
    # Quantizable:   classifier.0=Flatten, classifier.1=Linear, classifier.2=ReLU, classifier.3=Linear
    model_state = model_fp32.state_dict()

    for name, param in original_state.items():
        # Handle classifier.4 -> classifier.3 mapping (final Linear layer)
        if name.startswith('classifier.4'):
            new_name = name.replace('classifier.4', 'classifier.3')
            if new_name in model_state and model_state[new_name].shape == param.shape:
                model_state[new_name] = param
                print(f"  Mapped {name} -> {new_name}")
        elif name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param

    model_fp32.load_state_dict(model_state)
    model_fp32.eval()

    fp32_size = get_model_size(model_fp32)
    print(f"FP32 model size: {fp32_size:.2f} MB")

    # ============================================
    # Step 2: Evaluate FP32 Model
    # ============================================
    print("\n[2/5] Evaluating FP32 model...")
    fp32_acc, _, _ = evaluate(model_fp32, test_loader, device="cpu")
    print(f"FP32 Accuracy: {fp32_acc*100:.2f}%")

    # ============================================
    # Step 3: Fuse Modules
    # ============================================
    print("\n[3/5] Fusing Conv+BN+ReLU layers...")
    model_fp32.fuse_model()

    # ============================================
    # Step 4: Calibration
    # ============================================
    print("\n[4/5] Calibrating with training data...")

    # Set quantization config
    model_fp32.qconfig = quant.get_default_qconfig('fbgemm')  # Use 'qnnpack' for ARM
    print(f"Quantization config: {model_fp32.qconfig}")

    # Prepare for quantization
    quant.prepare(model_fp32, inplace=True)

    # Calibration - run representative data through model
    calibration_loader = get_calibration_loader(CALIBRATION_PATH, CALIBRATION_SAMPLES)
    print(f"Calibrating with {CALIBRATION_SAMPLES} samples...")

    with torch.no_grad():
        for images, _ in tqdm(calibration_loader, desc="Calibrating"):
            model_fp32(images)

    # ============================================
    # Step 5: Convert to INT8
    # ============================================
    print("\n[5/5] Converting to INT8...")
    model_int8 = quant.convert(model_fp32, inplace=False)

    int8_size = get_model_size(model_int8)
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Compression ratio: {fp32_size/int8_size:.2f}x")

    # ============================================
    # Evaluate INT8 Model
    # ============================================
    print("\n" + "=" * 60)
    print("Evaluating INT8 Quantized Model")
    print("=" * 60)

    int8_acc, int8_preds, int8_labels = evaluate(model_int8, test_loader, device="cpu")
    print(f"\nINT8 Accuracy: {int8_acc*100:.2f}%")
    print(f"Accuracy drop: {(fp32_acc - int8_acc)*100:.2f}%")

    # Classification report
    print("\nPer-Class Classification Report (INT8):")
    print(classification_report(int8_labels, int8_preds, target_names=class_names, digits=4))

    # ============================================
    # Save Quantized Model
    # ============================================
    # Save state dict
    int8_path = os.path.join(SAVE_DIR, "best_model_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)
    print(f"\nINT8 state dict saved to: {int8_path}")

    # Save full model for easier loading
    int8_full_path = os.path.join(SAVE_DIR, "best_model_int8_full.pth")
    torch.save(model_int8, int8_full_path)
    print(f"INT8 full model saved to: {int8_full_path}")

    # Also save as TorchScript for deployment
    try:
        scripted_path = os.path.join(SAVE_DIR, "best_model_int8_scripted.pt")
        scripted_model = torch.jit.script(model_int8)
        scripted_model.save(scripted_path)
        print(f"TorchScript model saved to: {scripted_path}")
    except Exception as e:
        print(f"Warning: Could not save TorchScript model: {e}")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'FP32':<15} {'INT8':<15}")
    print("-" * 55)
    print(f"{'Model Size':<25} {fp32_size:.2f} MB{'':<7} {int8_size:.2f} MB")
    print(f"{'Accuracy':<25} {fp32_acc*100:.2f}%{'':<9} {int8_acc*100:.2f}%")
    print(f"{'Accuracy Drop':<25} {'-':<15} {(fp32_acc - int8_acc)*100:.2f}%")
    print(f"{'Compression':<25} {'1x':<15} {fp32_size/int8_size:.2f}x")
    print("=" * 60)

    # Save summary
    with open(os.path.join(SAVE_DIR, "quantization_summary.txt"), "w") as f:
        f.write("INT8 Quantization Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"FP32 Model Size: {fp32_size:.2f} MB\n")
        f.write(f"INT8 Model Size: {int8_size:.2f} MB\n")
        f.write(f"Compression Ratio: {fp32_size/int8_size:.2f}x\n")
        f.write(f"FP32 Accuracy: {fp32_acc*100:.2f}%\n")
        f.write(f"INT8 Accuracy: {int8_acc*100:.2f}%\n")
        f.write(f"Accuracy Drop: {(fp32_acc - int8_acc)*100:.2f}%\n")

    print("\nQuantization complete!")
