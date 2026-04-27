import torch
import torch.nn as nn
from torch.ao.quantization import quantize_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization import default_qconfig
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import os
import copy

# ============================================
# ResNet-50 INT8 Quantization using FX Graph Mode
# (Handles skip connections properly)
# ============================================

# ============================================
# Configuration
# ============================================
MODEL_PATH = "/netscratch/krajshekar/ESCADE/small_cnn/resnet50_checkpoints/best_model.pth"
CALIBRATION_PATH = "/netscratch/krajshekar/ESCADE/DOES/train"
TEST_PATH = "/netscratch/krajshekar/ESCADE/DOES/test"
SAVE_DIR = "/netscratch/krajshekar/ESCADE/small_cnn/resnet50_checkpoints"
NUM_CLASSES = 9
BATCH_SIZE = 32
CALIBRATION_SAMPLES = 1000
DEVICE = "cpu"  # Quantization must be done on CPU

# ============================================
# Data Transform
# ============================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_calibration_loader(data_path, num_samples):
    """Get a subset of data for calibration."""
    dataset = ImageFolder(data_path, transform=transform)
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


def calibrate(model, data_loader):
    """Run calibration data through the model."""
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc="Calibrating"):
            model(images)


# ============================================
# Main Quantization Pipeline
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("ResNet-50 INT8 Quantization (FX Graph Mode)")
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

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please run train_resnet50.py first!")
        exit(1)

    # Load ResNet-50 and replace fc layer
    model_fp32 = models.resnet50(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, NUM_CLASSES)
    model_fp32.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
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
    # Step 3: Prepare for FX Quantization
    # ============================================
    print("\n[3/5] Preparing FX Graph Mode quantization...")

    # Set the backend
    torch.backends.quantized.engine = 'fbgemm'  # Use 'qnnpack' for ARM

    # Create QConfig mapping
    qconfig_mapping = QConfigMapping().set_global(default_qconfig)

    # Prepare example input for tracing
    example_input = torch.randn(1, 3, 64, 64)

    # Prepare the model for quantization
    model_prepared = quantize_fx.prepare_fx(
        copy.deepcopy(model_fp32),
        qconfig_mapping,
        example_input
    )

    print("FX preparation complete")

    # ============================================
    # Step 4: Calibration
    # ============================================
    print("\n[4/5] Calibrating with training data...")

    calibration_loader = get_calibration_loader(CALIBRATION_PATH, CALIBRATION_SAMPLES)
    print(f"Calibrating with {CALIBRATION_SAMPLES} samples...")

    calibrate(model_prepared, calibration_loader)

    # ============================================
    # Step 5: Convert to INT8
    # ============================================
    print("\n[5/5] Converting to INT8...")
    model_int8 = quantize_fx.convert_fx(model_prepared)

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
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save state dict
    int8_path = os.path.join(SAVE_DIR, "best_model_int8.pth")
    torch.save(model_int8.state_dict(), int8_path)
    print(f"\nINT8 state dict saved to: {int8_path}")

    # Save full model
    int8_full_path = os.path.join(SAVE_DIR, "best_model_int8_full.pth")
    torch.save(model_int8, int8_full_path)
    print(f"INT8 full model saved to: {int8_full_path}")

    # Save as TorchScript
    try:
        scripted_path = os.path.join(SAVE_DIR, "best_model_int8_scripted.pt")
        scripted_model = torch.jit.trace(model_int8, example_input)
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
        f.write("ResNet-50 INT8 Quantization Summary (FX Graph Mode)\n")
        f.write("=" * 40 + "\n")
        f.write(f"FP32 Model Size: {fp32_size:.2f} MB\n")
        f.write(f"INT8 Model Size: {int8_size:.2f} MB\n")
        f.write(f"Compression Ratio: {fp32_size/int8_size:.2f}x\n")
        f.write(f"FP32 Accuracy: {fp32_acc*100:.2f}%\n")
        f.write(f"INT8 Accuracy: {int8_acc*100:.2f}%\n")
        f.write(f"Accuracy Drop: {(fp32_acc - int8_acc)*100:.2f}%\n")

    print("\nQuantization complete!")
