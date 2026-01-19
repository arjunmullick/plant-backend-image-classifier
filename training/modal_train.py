"""
Modal-based Training Script for Plant Image Classifier

This script runs on Modal Labs GPU infrastructure.
Run locally with: modal run training/modal_train.py

Prerequisites:
1. Install Modal: pip install modal
2. Setup Modal: modal setup
3. Create secrets:
   modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxxxx
   modal secret create wandb-secret WANDB_API_KEY=xxxxx

Usage:
    modal run training/modal_train.py --dataset-path your-username/plantvillage
    modal run training/modal_train.py --epochs 15 --model efficientnet_b2
"""

import modal

# Define the Modal app
app = modal.App("plant-classifier-training")

# Define the training image with all dependencies
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "timm>=0.9.0",
    "datasets>=2.14.0",
    "huggingface_hub>=0.19.0",
    "wandb>=0.16.0",
    "albumentations>=1.3.0",
    "scikit-learn>=1.3.0",
    "pillow>=10.0.0",
    "tqdm>=4.66.0",
)


@app.function(
    image=training_image,
    gpu="T4",  # Options: "T4", "A10G", "A100"
    timeout=7200,  # 2 hours max
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_species_classifier(
    dataset_path: str = "your-username/plantvillage-cleaned",
    model_name: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    freeze_backbone_epochs: int = 5,
    run_name: str = "species_v1",
    push_to_hub: bool = True,
    hub_repo: str = "your-username/plant-classifier",
):
    """
    Train species classifier on Modal GPU.

    Args:
        dataset_path: HF Hub dataset path
        model_name: timm model name (efficientnet_b0, efficientnet_b2, convnext_tiny)
        epochs: Total training epochs
        batch_size: Training batch size
        learning_rate: Base learning rate
        freeze_backbone_epochs: Epochs to train with frozen backbone
        run_name: Name for this training run
        push_to_hub: Whether to push model to HF Hub
        hub_repo: HF Hub repository to push to
    """
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from datasets import load_dataset
    import timm
    import wandb
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    from huggingface_hub import HfApi

    print(f"Starting training with config:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize wandb
    wandb.init(
        project="plant-classifier",
        name=run_name,
        config={
            "model": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset": dataset_path,
        }
    )

    # Load dataset from HF Hub
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)

    # Get number of classes
    num_classes = len(dataset["train"].features["label"].names)
    class_names = dataset["train"].features["label"].names
    print(f"Found {num_classes} classes")

    # Create model with pretrained backbone
    print(f"Creating model: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create dataloaders
    def train_collate_fn(batch):
        images = torch.stack([
            train_transform(x["image"].convert("RGB"))
            for x in batch
        ])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels

    def val_collate_fn(batch):
        images = torch.stack([
            val_transform(x["image"].convert("RGB"))
            for x in batch
        ])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Use validation split if available, otherwise use test
    val_split = "validation" if "validation" in dataset else "test"
    val_loader = DataLoader(
        dataset[val_split],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop with progressive unfreezing
    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # Progressive unfreezing strategy
        if epoch < freeze_backbone_epochs:
            # Phase 1: Freeze backbone, train head only
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in model.get_classifier().parameters():
                param.requires_grad = True
            lr = learning_rate * 10  # Higher LR for head only
            phase = "Phase 1: Head only"

        elif epoch < freeze_backbone_epochs + 5:
            # Phase 2: Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
            lr = learning_rate
            phase = "Phase 2: Full model"

        else:
            # Phase 3: Fine-tuning with lower LR
            lr = learning_rate * 0.1
            phase = "Phase 3: Fine-tuning"

        # Create optimizer for this phase
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} ({phase})")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100. * val_correct / val_total

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc,
            "learning_rate": lr,
            "phase": phase,
        }
        wandb.log(metrics)

        print(f"\nEpoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  New best model! Val Acc: {val_acc:.2f}%")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    # Final evaluation
    print("\nFinal Evaluation:")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )

    final_metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")

    wandb.log({"final_metrics": final_metrics})

    # Save model
    model_path = f"/tmp/{run_name}.pt"
    torch.save(best_model_state, model_path)
    print(f"\nModel saved to {model_path}")

    # Push to HF Hub
    if push_to_hub:
        print(f"\nPushing to HF Hub: {hub_repo}")
        api = HfApi()

        try:
            # Upload model file
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"checkpoints/{run_name}.pt",
                repo_id=hub_repo,
                commit_message=f"Add {run_name} - accuracy: {final_metrics['accuracy']:.2%}",
            )

            # Create version tag
            api.create_tag(repo_id=hub_repo, tag=run_name)

            print(f"Model uploaded to: https://huggingface.co/{hub_repo}")
        except Exception as e:
            print(f"Failed to push to HF Hub: {e}")

    wandb.finish()

    return {
        "run_name": run_name,
        "final_metrics": final_metrics,
        "best_val_acc": best_val_acc,
        "model_path": model_path,
    }


@app.function(
    image=training_image,
    gpu="T4",
    timeout=7200,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_disease_detector(
    dataset_path: str,
    crop: str = "tomato",
    model_name: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
):
    """
    Train crop-specific disease detector.

    This trains a specialized model for a specific crop's diseases.
    """
    # Similar to train_species_classifier but filtered by crop
    # Implementation follows same pattern
    pass


@app.local_entrypoint()
def main(
    dataset_path: str = "your-username/plantvillage-cleaned",
    model: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    run_name: str = "species_v1",
):
    """Local entrypoint for running training from CLI."""
    print("Triggering training job on Modal...")

    result = train_species_classifier.remote(
        dataset_path=dataset_path,
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        run_name=run_name,
    )

    print(f"\nTraining complete!")
    print(f"Results: {result}")
