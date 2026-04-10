import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import os
from collections import Counter

# ============================================
# 6-Block Deep CNN for 64x64 Input (FP32)
# No skip connections - Spinnaker compatible
# ============================================

class DeepCNN6Block(nn.Module):
    """
    6-block CNN for 64x64 input images.
    Each block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    Spatial reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
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
# Configuration
# ============================================
DATA_PATH = "/netscratch/krajshekar/ESCADE/DOES"
SAVE_DIR = "/netscratch/krajshekar/ESCADE/small_cnn/deepcnn_6block_checkpoints_2"
NUM_CLASSES = 9
BATCH_SIZE = 64
EPOCHS = 50  # Max epochs (early stopping will likely trigger before this)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15  # 15% for validation
DEVICE = "cpu"  # Force CPU (set to "cuda" if GPU compatibility is fixed)
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 2  # Stop if no improvement for 2 consecutive epochs

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================
# Data Transforms
# ============================================
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# Load Dataset
# ============================================
print(f"Loading dataset from: {DATA_PATH}")

# Check if train/test folders exist
train_path = os.path.join(DATA_PATH, "train")
test_path = os.path.join(DATA_PATH, "test")

if os.path.exists(train_path) and os.path.exists(test_path):
    print("Found train/test split in dataset")
    print("Using: train folder for training, test folder for validation")

    # Use all train data for training
    train_dataset = ImageFolder(train_path, transform=train_transform)

    # Use test data for validation (no augmentation)
    val_dataset = ImageFolder(test_path, transform=val_transform)

    # No splitting needed - use full datasets
    train_subset = train_dataset
    val_subset = val_dataset
    test_dataset = val_dataset  # Same as validation for final evaluation

else:
    print("No train/test split found, using full dataset with random split")
    full_dataset = ImageFolder(DATA_PATH, transform=train_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_subset, val_subset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Get class names
if os.path.exists(train_path):
    class_names = ImageFolder(train_path).classes
else:
    class_names = ImageFolder(DATA_PATH).classes

print(f"Classes: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Create data loaders
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train samples: {len(train_subset)}")
print(f"Val/Test samples: {len(val_subset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================
# Model, Loss, Optimizer
# ============================================
model = DeepCNN6Block(num_classes=NUM_CLASSES).to(DEVICE)
print(f"\nModel architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# ============================================
# Training Functions
# ============================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / len(loader), correct / total, all_preds, all_labels


# ============================================
# Training Loop
# ============================================
print(f"\n{'='*50}")
print(f"Starting training on {DEVICE}")
print(f"{'='*50}\n")

best_val_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 30)

    # Train
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    # Validate
    val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)

    # Update scheduler
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model and check early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        print(f"✓ New best model saved! (Val Acc: {val_acc*100:.2f}%)")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s)")

    # Early stopping check
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n{'='*50}")
        print(f"Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
        print(f"Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
        print(f"{'='*50}")
        break

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth"))

# ============================================
# Final Evaluation on Test Set
# ============================================
print(f"\n{'='*50}")
print("Final Evaluation on Test Set")
print(f"{'='*50}")

# Load best model
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"\nBest model was from epoch {best_epoch} with val acc {best_val_acc*100:.2f}%")

# Classification report
print("\nPer-Class Classification Report:")
print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))

# Save final model info
with open(os.path.join(SAVE_DIR, "training_info.txt"), "w") as f:
    f.write(f"Model: DeepCNN6Block\n")
    f.write(f"Input size: 64x64\n")
    f.write(f"Num classes: {NUM_CLASSES}\n")
    f.write(f"Total parameters: {total_params:,}\n")
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Best val accuracy: {best_val_acc*100:.2f}%\n")
    f.write(f"Test accuracy: {test_acc*100:.2f}%\n")

print(f"\nTraining complete! Models saved to: {SAVE_DIR}")
