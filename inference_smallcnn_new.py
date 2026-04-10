import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from collections import Counter
from PIL import Image
from tqdm import tqdm
import os
"""
# === Define the same model ===
class SmallCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
"""


class SmallCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # must match training
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# === Config ===
MODEL_PATH = "smallcnn_final_balanced/best_model_balanced.pth"
TEST_PATH = "DOES/test"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PER_CLASS = 10000
USE_RANDOM = True  #  Set to False for first-100 deterministic

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# === Load full dataset ===
full_dataset = ImageFolder(TEST_PATH, transform=transform)
targets = np.array(full_dataset.targets)
class_names = full_dataset.classes

# === Sample 100 per class ===
subset_indices = []
for class_id in range(len(class_names)):
    class_indices = np.where(targets == class_id)[0]
    if USE_RANDOM:
        sample_size = min(MAX_PER_CLASS, len(class_indices))
        sampled = np.random.choice(class_indices, size=sample_size, replace=False)

    else:
        sampled = class_indices[:MAX_PER_CLASS]
    subset_indices.extend(sampled)

subset = Subset(full_dataset, subset_indices)
subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)

# === Print class distribution ===
subset_targets = [full_dataset.targets[i] for i in subset_indices]
print(" Subset class distribution:", dict(Counter(subset_targets)))

# === Load model ===
model = SmallCNN(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Inference with tqdm and metrics ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(subset_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# === Overall accuracy ===
overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n Overall Accuracy: {overall_acc:.4f} ({np.sum(np.array(all_preds) == np.array(all_labels))}/{len(all_labels)})")

# === Class-wise metrics ===
print("\n Per-Class Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# === Optional: Inference on single image ===
def infer_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(dim=1).item()
        print(f"Predicted Class: {class_names[predicted_class]}")

# === Uncomment to test a single image
infer_single_image("DOES/test/E3/E3_87.png")
