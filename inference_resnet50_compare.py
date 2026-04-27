import torch
import torch.nn as nn
from torch.ao.quantization import quantize_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization import default_qconfig
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import os
import time
import copy

# ============================================
# ResNet-50 Inference Comparison: FP32 vs INT8
# ============================================

# ============================================
# Configuration
# ============================================
FP32_MODEL_PATH = "/netscratch/krajshekar/ESCADE/small_cnn/resnet50_checkpoints/best_model.pth"
INT8_MODEL_PATH = "/netscratch/krajshekar/ESCADE/small_cnn/resnet50_checkpoints/best_model_int8.pth"
TEST_PATH = "/netscratch/krajshekar/ESCADE/DOES/test"
NUM_CLASSES = 9
BATCH_SIZE = 64
MAX_SAMPLES_PER_CLASS = None  # Set to a number to limit samples, None for all
DEVICE_FP32 = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INT8 = "cpu"  # INT8 quantized models run on CPU

# ============================================
# Data Transform
# ============================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0


def evaluate_model(model, loader, device, desc="Evaluating"):
    """Evaluate model and return metrics."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)

            start_time = time.time()
            outputs = model(images)
            total_time += time.time() - start_time

            _, predicted = outputs.max(1)
            predicted = predicted.cpu()

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    avg_latency = (total_time / total) * 1000  # ms per sample
    throughput = total / total_time  # samples per second

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'preds': all_preds,
        'labels': all_labels,
        'avg_latency_ms': avg_latency,
        'throughput': throughput
    }


def create_quantized_model(fp32_model_path, int8_state_dict_path, num_classes):
    """Recreate the FX quantized model structure and load weights."""
    # Load FP32 model
    model_fp32 = models.resnet50(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, num_classes)
    model_fp32.load_state_dict(torch.load(fp32_model_path, map_location="cpu", weights_only=True))
    model_fp32.eval()

    # Set backend
    torch.backends.quantized.engine = 'fbgemm'

    # Prepare for FX quantization (same as in quantization script)
    qconfig_mapping = QConfigMapping().set_global(default_qconfig)
    example_input = torch.randn(1, 3, 64, 64)

    model_prepared = quantize_fx.prepare_fx(
        copy.deepcopy(model_fp32),
        qconfig_mapping,
        example_input
    )

    # Convert to quantized (without calibration - we'll load the calibrated weights)
    model_int8 = quantize_fx.convert_fx(model_prepared)

    # Load the quantized state dict
    model_int8.load_state_dict(torch.load(int8_state_dict_path, map_location="cpu", weights_only=True))

    return model_int8


def print_comparison_table(fp32_results, int8_results, fp32_size, int8_size):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON: FP32 vs INT8")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'FP32':<20} {'INT8':<20}")
    print("-" * 70)

    print(f"{'Model Size':<30} {fp32_size:.2f} MB{'':<12} {int8_size:.2f} MB")
    print(f"{'Compression Ratio':<30} {'1.00x':<20} {fp32_size/int8_size:.2f}x")
    print("-" * 70)

    print(f"{'Accuracy':<30} {fp32_results['accuracy']*100:.2f}%{'':<14} {int8_results['accuracy']*100:.2f}%")
    print(f"{'Correct / Total':<30} {fp32_results['correct']}/{fp32_results['total']:<13} {int8_results['correct']}/{int8_results['total']}")

    acc_diff = (int8_results['accuracy'] - fp32_results['accuracy']) * 100
    sign = "+" if acc_diff >= 0 else ""
    print(f"{'Accuracy Difference':<30} {'-':<20} {sign}{acc_diff:.2f}%")
    print("-" * 70)

    print(f"{'Device':<30} {DEVICE_FP32:<20} {DEVICE_INT8}")
    print(f"{'Avg Latency (per sample)':<30} {fp32_results['avg_latency_ms']:.3f} ms{'':<10} {int8_results['avg_latency_ms']:.3f} ms")
    print(f"{'Throughput':<30} {fp32_results['throughput']:.1f} samples/s{'':<4} {int8_results['throughput']:.1f} samples/s")

    print("=" * 70)


def print_per_class_comparison(fp32_results, int8_results, class_names):
    """Print per-class accuracy comparison."""
    print("\n" + "=" * 70)
    print(" PER-CLASS ACCURACY COMPARISON")
    print("=" * 70)

    fp32_preds = np.array(fp32_results['preds'])
    fp32_labels = np.array(fp32_results['labels'])
    int8_preds = np.array(int8_results['preds'])
    int8_labels = np.array(int8_results['labels'])

    print(f"\n{'Class':<20} {'FP32 Acc':<15} {'INT8 Acc':<15} {'Difference':<15}")
    print("-" * 65)

    for i, class_name in enumerate(class_names):
        # FP32 per-class accuracy
        fp32_mask = fp32_labels == i
        fp32_class_acc = np.mean(fp32_preds[fp32_mask] == fp32_labels[fp32_mask]) if fp32_mask.sum() > 0 else 0

        # INT8 per-class accuracy
        int8_mask = int8_labels == i
        int8_class_acc = np.mean(int8_preds[int8_mask] == int8_labels[int8_mask]) if int8_mask.sum() > 0 else 0

        diff = (int8_class_acc - fp32_class_acc) * 100
        sign = "+" if diff >= 0 else ""

        print(f"{class_name:<20} {fp32_class_acc*100:.2f}%{'':<9} {int8_class_acc*100:.2f}%{'':<9} {sign}{diff:.2f}%")

    print("=" * 70)


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print(" ResNet-50 Model Inference Comparison")
    print(" FP32 vs INT8 Quantized")
    print("=" * 70)

    # ============================================
    # Load Test Dataset
    # ============================================
    print("\n[1/4] Loading test dataset...")
    full_dataset = ImageFolder(TEST_PATH, transform=transform)
    class_names = full_dataset.classes

    if MAX_SAMPLES_PER_CLASS is not None:
        targets = np.array(full_dataset.targets)
        indices = []
        for class_id in range(len(class_names)):
            class_indices = np.where(targets == class_id)[0]
            sampled = class_indices[:MAX_SAMPLES_PER_CLASS]
            indices.extend(sampled)
        test_dataset = Subset(full_dataset, indices)
    else:
        test_dataset = full_dataset

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {class_names}")
    print(f"Batch size: {BATCH_SIZE}")

    # ============================================
    # Load and Evaluate FP32 Model
    # ============================================
    print("\n[2/4] Loading and evaluating FP32 model...")
    print(f"Model path: {FP32_MODEL_PATH}")
    print(f"Device: {DEVICE_FP32}")

    if not os.path.exists(FP32_MODEL_PATH):
        print(f"ERROR: FP32 model not found at {FP32_MODEL_PATH}")
        print("Please run train_resnet50.py first!")
        exit(1)

    # Load ResNet-50 FP32
    model_fp32 = models.resnet50(weights=None)
    model_fp32.fc = nn.Linear(model_fp32.fc.in_features, NUM_CLASSES)
    model_fp32.load_state_dict(torch.load(FP32_MODEL_PATH, map_location=DEVICE_FP32, weights_only=True))
    model_fp32 = model_fp32.to(DEVICE_FP32)
    model_fp32.eval()

    fp32_size = get_model_size_mb(FP32_MODEL_PATH)
    fp32_results = evaluate_model(model_fp32, test_loader, DEVICE_FP32, desc="FP32 Inference")

    print(f"\nFP32 Results:")
    print(f"  Accuracy: {fp32_results['accuracy']*100:.2f}%")
    print(f"  Model Size: {fp32_size:.2f} MB")

    # ============================================
    # Load and Evaluate INT8 Model
    # ============================================
    print("\n[3/4] Loading and evaluating INT8 model...")
    print(f"Model path: {INT8_MODEL_PATH}")
    print(f"Device: {DEVICE_INT8}")

    if not os.path.exists(INT8_MODEL_PATH):
        print(f"WARNING: INT8 model not found at {INT8_MODEL_PATH}")
        print("Please run quantize_resnet50_int8.py first!")
        print("\nSkipping INT8 evaluation. Showing FP32 results only.\n")

        # Print FP32 classification report
        print("\n" + "=" * 70)
        print(" FP32 MODEL - CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(fp32_results['labels'], fp32_results['preds'],
                                    target_names=class_names, digits=4))
        exit(0)

    # Recreate quantized model structure and load weights
    print("Recreating quantized model structure...")
    model_int8 = create_quantized_model(FP32_MODEL_PATH, INT8_MODEL_PATH, NUM_CLASSES)
    model_int8.eval()

    int8_size = get_model_size_mb(INT8_MODEL_PATH)

    # Need CPU data loader for INT8
    test_loader_cpu = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    int8_results = evaluate_model(model_int8, test_loader_cpu, DEVICE_INT8, desc="INT8 Inference")

    print(f"\nINT8 Results:")
    print(f"  Accuracy: {int8_results['accuracy']*100:.2f}%")
    print(f"  Model Size: {int8_size:.2f} MB")

    # ============================================
    # Print Comparison
    # ============================================
    print("\n[4/4] Generating comparison report...")

    print_comparison_table(fp32_results, int8_results, fp32_size, int8_size)
    print_per_class_comparison(fp32_results, int8_results, class_names)

    # Detailed classification reports
    print("\n" + "=" * 70)
    print(" FP32 MODEL - CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(fp32_results['labels'], fp32_results['preds'],
                                target_names=class_names, digits=4))

    print("\n" + "=" * 70)
    print(" INT8 MODEL - CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(int8_results['labels'], int8_results['preds'],
                                target_names=class_names, digits=4))

    # ============================================
    # Save Results
    # ============================================
    results_path = os.path.dirname(FP32_MODEL_PATH)
    results_file = os.path.join(results_path, "inference_comparison_results.txt")

    with open(results_file, "w") as f:
        f.write("ResNet-50 Model Inference Comparison\n")
        f.write("=" * 50 + "\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(f"FP32 Model Size: {fp32_size:.2f} MB\n")
        f.write(f"INT8 Model Size: {int8_size:.2f} MB\n")
        f.write(f"Compression Ratio: {fp32_size/int8_size:.2f}x\n\n")

        f.write(f"FP32 Accuracy: {fp32_results['accuracy']*100:.2f}%\n")
        f.write(f"INT8 Accuracy: {int8_results['accuracy']*100:.2f}%\n")
        f.write(f"Accuracy Difference: {(int8_results['accuracy'] - fp32_results['accuracy'])*100:.2f}%\n\n")

        f.write(f"FP32 Throughput: {fp32_results['throughput']:.1f} samples/s\n")
        f.write(f"INT8 Throughput: {int8_results['throughput']:.1f} samples/s\n\n")

        f.write("FP32 CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(fp32_results['labels'], fp32_results['preds'],
                                      target_names=class_names, digits=4))

        f.write("\n\nINT8 CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(int8_results['labels'], int8_results['preds'],
                                      target_names=class_names, digits=4))

    print(f"\nResults saved to: {results_file}")
    print("\nInference comparison complete!")
