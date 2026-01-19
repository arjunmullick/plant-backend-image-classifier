"""
Species Classification Training Module

This module contains the core training logic for species classification.
It's designed to be imported by modal_train.py or run standalone for local testing.

Training Strategy:
1. Phase 1 (epochs 1-5): Freeze backbone, train classifier head only
2. Phase 2 (epochs 6-10): Unfreeze all layers, standard learning rate
3. Phase 3 (epochs 11-15): Full fine-tuning with reduced learning rate

Data Augmentation:
- RandomResizedCrop (scale 0.7-1.0)
- Horizontal/Vertical flip
- Color jitter (brightness, contrast, saturation, hue)
- Random rotation (up to 30 degrees)
- Gaussian blur (occasional)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_name: str = "efficientnet_b0"
    num_classes: int = 38
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    freeze_backbone_epochs: int = 5
    gradient_clip: float = 1.0
    device: str = "cuda"


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a model using timm.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        PyTorch model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Get training data augmentation transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(30),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except classifier."""
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    return total_loss / len(train_loader), 100. * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    Validate the model.

    Returns:
        Tuple of (average loss, accuracy, all predictions, all labels)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(val_loader),
        100. * correct / total,
        all_preds,
        all_labels
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    callback: Optional[Callable[[TrainingMetrics], None]] = None,
) -> tuple[nn.Module, dict]:
    """
    Full training loop with progressive unfreezing.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        callback: Optional callback for logging metrics

    Returns:
        Tuple of (trained model, final metrics)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_model_state = None

    for epoch in range(config.epochs):
        # Progressive unfreezing
        if epoch < config.freeze_backbone_epochs:
            freeze_backbone(model)
            lr = config.learning_rate * 10
            phase = "head_only"
        elif epoch < config.freeze_backbone_epochs + 5:
            unfreeze_all(model)
            lr = config.learning_rate
            phase = "full_model"
        else:
            lr = config.learning_rate * 0.1
            phase = "fine_tuning"

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=config.weight_decay,
        )

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.gradient_clip
        )

        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(
            model, val_loader, criterion, device
        )

        # Create metrics
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=lr,
        )

        # Callback for logging
        if callback:
            callback(metrics)

        print(f"Epoch {epoch+1}/{config.epochs} ({phase}): "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, {
        "best_val_acc": best_val_acc,
        "final_predictions": all_preds,
        "final_labels": all_labels,
    }


if __name__ == "__main__":
    # Example local testing
    print("This module is designed to be imported by modal_train.py")
    print("For local testing, create a small dataset and call train_model()")
