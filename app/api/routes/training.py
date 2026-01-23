"""
Training API routes for local model training.

Provides endpoints to:
- Download datasets (PlantVillage, PlantDoc)
- Start/stop training jobs
- Monitor training progress
- Test trained models
- Deploy trained models
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import shutil

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter(prefix="/training", tags=["training"])

# ============================================
# Request/Response Models
# ============================================


class DatasetDownloadRequest(BaseModel):
    """Request to download a dataset."""

    dataset_name: str = Field(
        default="plantvillage", description="Dataset to download: plantvillage, plantdoc, custom"
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    base_model: str = Field(default="mobilenet_v2", description="Base model architecture")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, description="Batch size")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    device: str = Field(default="auto", description="Device: auto, mps, cpu")
    augmentation: str = Field(default="standard", description="Augmentation level")
    enable_wandb: bool = Field(default=False, description="Enable W&B logging")
    save_checkpoints: bool = Field(default=True, description="Save checkpoints")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    dataset: str = Field(default="plantvillage", description="Dataset to use")


class TestModelRequest(BaseModel):
    """Request to test a trained model."""

    image: str = Field(..., description="Base64 encoded image")
    model_path: Optional[str] = Field(None, description="Path to model checkpoint")


class DeployModelRequest(BaseModel):
    """Request to deploy a trained model."""

    model_path: str = Field(..., description="Path to model checkpoint")


class DownloadModelRequest(BaseModel):
    """Request to download a trained model."""

    model_path: str = Field(..., description="Path to model checkpoint")


class ExportHFRequest(BaseModel):
    """Request to export model to HuggingFace."""

    model_path: str = Field(..., description="Path to model checkpoint")
    repo_name: str = Field(..., description="HuggingFace repository name")


# ============================================
# Training State Management
# ============================================

# In-memory training job state (would be Redis/DB in production)
_training_jobs: Dict[str, Dict[str, Any]] = {}
_current_job_id: Optional[str] = None
_stop_requested: bool = False


def get_data_dir() -> Path:
    """Get the data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_models_dir() -> Path:
    """Get the models directory."""
    models_dir = Path("models") / "trained"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# ============================================
# Dataset Endpoints
# ============================================


@router.get("/dataset/status")
async def get_dataset_status():
    """Check dataset availability and info."""
    data_dir = get_data_dir()

    # Check for PlantVillage dataset
    plantvillage_dir = data_dir / "plantvillage"
    plantdoc_dir = data_dir / "plantdoc"

    if plantvillage_dir.exists():
        # Count classes and images
        classes = [d for d in plantvillage_dir.iterdir() if d.is_dir()]
        num_images = sum(
            len(list(c.glob("*.jpg"))) + len(list(c.glob("*.JPG"))) + len(list(c.glob("*.png")))
            for c in classes
        )
        return {
            "available": True,
            "name": "PlantVillage",
            "path": str(plantvillage_dir),
            "num_classes": len(classes),
            "num_images": num_images,
            "downloading": False,
        }
    elif plantdoc_dir.exists():
        classes = [d for d in plantdoc_dir.iterdir() if d.is_dir()]
        num_images = sum(len(list(c.glob("*.jpg"))) + len(list(c.glob("*.png"))) for c in classes)
        return {
            "available": True,
            "name": "PlantDoc",
            "path": str(plantdoc_dir),
            "num_classes": len(classes),
            "num_images": num_images,
            "downloading": False,
        }
    else:
        return {
            "available": False,
            "name": None,
            "path": None,
            "num_classes": 0,
            "num_images": 0,
            "downloading": False,
            "message": "No dataset found. Click 'Download Dataset' to get started.",
        }


@router.post("/dataset/download")
async def download_dataset(request: DatasetDownloadRequest, background_tasks: BackgroundTasks):
    """Start downloading a dataset."""
    data_dir = get_data_dir()
    dataset_name = request.dataset_name.lower()

    if dataset_name == "plantvillage":
        # PlantVillage dataset download
        background_tasks.add_task(_download_plantvillage, data_dir)
        return {
            "status": "started",
            "message": "PlantVillage dataset download started",
            "dataset": "plantvillage",
        }
    elif dataset_name == "plantdoc":
        background_tasks.add_task(_download_plantdoc, data_dir)
        return {
            "status": "started",
            "message": "PlantDoc dataset download started",
            "dataset": "plantdoc",
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset_name}")


async def _download_plantvillage(data_dir: Path):
    """Download PlantVillage dataset from Kaggle or mirror."""
    import subprocess

    output_dir = data_dir / "plantvillage"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try using kaggle CLI
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "emmarex/plantdisease",
                "-p",
                str(data_dir),
                "--unzip",
            ],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: create a readme with manual instructions
        readme_path = output_dir / "README.txt"
        with open(readme_path, "w") as f:
            f.write(
                """PlantVillage Dataset Download Instructions
==========================================

The automatic download requires Kaggle CLI. To set it up:

1. Install Kaggle CLI: pip install kaggle
2. Create Kaggle API token at: https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/
4. Run: kaggle datasets download -d emmarex/plantdisease -p data/ --unzip

Alternatively, download manually:
1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Click 'Download' button
3. Extract to: data/plantvillage/

The dataset should have folders like:
- data/plantvillage/Tomato___Early_blight/
- data/plantvillage/Tomato___healthy/
- etc.
"""
            )


async def _download_plantdoc(data_dir: Path):
    """Download PlantDoc dataset."""
    output_dir = data_dir / "plantdoc"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_path = output_dir / "README.txt"
    with open(readme_path, "w") as f:
        f.write(
            """PlantDoc Dataset Download Instructions
======================================

PlantDoc is a real-world plant disease dataset.

Download from: https://github.com/pratikkayal/PlantDoc-Dataset

1. Clone the repository or download ZIP
2. Extract images to: data/plantdoc/

The dataset contains real-world images with:
- Apple diseased
- Apple healthy
- Tomato diseased
- etc.
"""
        )


# ============================================
# Training Endpoints
# ============================================


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a training job."""
    global _current_job_id, _stop_requested

    # Check if training is already running
    if _current_job_id and _training_jobs.get(_current_job_id, {}).get("status") == "running":
        raise HTTPException(status_code=400, detail="Training job already running")

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    _current_job_id = job_id
    _stop_requested = False

    # Initialize job state
    _training_jobs[job_id] = {
        "job_id": job_id,
        "status": "initializing",
        "config": config.model_dump(),
        "current_epoch": 0,
        "total_epochs": config.epochs,
        "train_loss": None,
        "train_accuracy": None,
        "val_loss": None,
        "val_accuracy": None,
        "best_val_accuracy": 0.0,
        "learning_rate": config.learning_rate,
        "eta": None,
        "started_at": datetime.now().isoformat(),
        "model_path": None,
        "logs": [],
        "message": "Initializing training...",
    }

    # Start training in background
    background_tasks.add_task(_run_training, job_id, config)

    return {"job_id": job_id, "status": "started", "message": "Training job started"}


async def _run_training(job_id: str, config: TrainingConfig):
    """Run the training loop."""
    global _stop_requested

    job = _training_jobs[job_id]
    job["status"] = "running"
    job["message"] = "Loading dataset..."

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split
        from torchvision import transforms, datasets
        import timm

        # Determine device
        if config.device == "auto":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                job["logs"].append({"level": "info", "message": "Using Apple Silicon GPU (MPS)"})
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                job["logs"].append({"level": "info", "message": "Using CUDA GPU"})
            else:
                device = torch.device("cpu")
                job["logs"].append({"level": "info", "message": "Using CPU"})
        else:
            device = torch.device(config.device)

        job["message"] = f"Using device: {device}"

        # Load dataset
        data_dir = get_data_dir() / config.dataset
        if not data_dir.exists():
            raise ValueError(f"Dataset not found: {data_dir}")

        # Define transforms based on augmentation level
        if config.augmentation == "standard":
            train_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif config.augmentation == "aggressive":
            train_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            train_transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        val_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        job["message"] = "Loading dataset..."
        job["logs"].append({"level": "info", "message": f"Loading dataset from {data_dir}"})

        # Load dataset
        full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transforms)
        num_classes = len(full_dataset.classes)
        job["logs"].append(
            {"level": "info", "message": f"Found {len(full_dataset)} images in {num_classes} classes"}
        )

        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply val transforms to validation set
        val_dataset.dataset = datasets.ImageFolder(str(data_dir), transform=val_transforms)

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        job["logs"].append(
            {"level": "info", "message": f"Train: {len(train_dataset)}, Val: {len(val_dataset)}"}
        )

        # Create model
        job["message"] = "Creating model..."
        model = timm.create_model(config.base_model, pretrained=True, num_classes=num_classes)
        model = model.to(device)
        job["logs"].append({"level": "info", "message": f"Created {config.base_model} model"})

        # Optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

        # Training loop
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        models_dir = get_models_dir()

        for epoch in range(config.epochs):
            if _stop_requested:
                job["status"] = "stopped"
                job["message"] = "Training stopped by user"
                return

            job["current_epoch"] = epoch + 1
            job["message"] = f"Training epoch {epoch + 1}/{config.epochs}..."

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if _stop_requested:
                    break

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total

            # Update job state
            job["train_loss"] = train_loss
            job["train_accuracy"] = train_acc
            job["val_loss"] = val_loss
            job["val_accuracy"] = val_acc
            job["learning_rate"] = scheduler.get_last_lr()[0]

            # Estimate ETA
            epochs_remaining = config.epochs - (epoch + 1)
            job["eta"] = f"~{epochs_remaining * 2} min" if epochs_remaining > 0 else "Almost done"

            job["logs"].append(
                {
                    "level": "info",
                    "message": f"Epoch {epoch + 1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}",
                }
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                job["best_val_accuracy"] = best_val_acc
                patience_counter = 0

                # Save checkpoint
                model_path = models_dir / f"best_model_{job_id}.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_accuracy": val_acc,
                        "num_classes": num_classes,
                        "class_names": full_dataset.classes,
                        "config": config.model_dump(),
                    },
                    model_path,
                )
                job["model_path"] = str(model_path)
                job["logs"].append(
                    {"level": "success", "message": f"New best model saved: {val_acc:.4f}"}
                )
            else:
                patience_counter += 1

            # Early stopping
            if config.early_stopping and patience_counter >= patience:
                job["logs"].append(
                    {"level": "warning", "message": f"Early stopping at epoch {epoch + 1}"}
                )
                break

            scheduler.step()

        # Training complete
        job["status"] = "completed"
        job["message"] = f"Training complete! Best accuracy: {best_val_acc:.4f}"
        job["best_val_accuracy"] = best_val_acc
        job["total_epochs"] = epoch + 1
        job["training_time"] = "Completed"
        job["model_size"] = (
            f"{(models_dir / f'best_model_{job_id}.pt').stat().st_size / 1024 / 1024:.1f} MB"
        )

    except ImportError as e:
        job["status"] = "failed"
        job["error"] = f"Missing dependency: {str(e)}. Install with: pip install torch timm"
        job["message"] = "Training failed - missing dependencies"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Training failed: {str(e)}"
        job["logs"].append({"level": "error", "message": str(e)})


@router.get("/status/{job_id}")
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = _training_jobs[job_id]
    return job


@router.post("/stop")
async def stop_training():
    """Stop the current training job."""
    global _stop_requested

    if not _current_job_id:
        raise HTTPException(status_code=400, detail="No training job running")

    _stop_requested = True
    return {"status": "stopping", "message": "Stop signal sent"}


# ============================================
# Model Testing and Deployment
# ============================================


@router.post("/test")
async def test_model(request: TestModelRequest):
    """Test a trained model with an image."""
    import base64
    from io import BytesIO

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import timm

        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Find model path
        model_path = request.model_path
        if not model_path:
            # Use most recent model
            models_dir = get_models_dir()
            model_files = list(models_dir.glob("best_model_*.pt"))
            if not model_files:
                raise HTTPException(status_code=400, detail="No trained model found")
            model_path = str(max(model_files, key=lambda p: p.stat().st_mtime))

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        num_classes = checkpoint["num_classes"]
        class_names = checkpoint["class_names"]
        config = checkpoint.get("config", {})

        # Create model
        model = timm.create_model(
            config.get("base_model", "mobilenet_v2"), pretrained=False, num_classes=num_classes
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Transform image
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = transform(image).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, min(5, num_classes))

        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            predictions.append({"class": class_names[idx], "confidence": prob.item()})

        return {"predictions": predictions, "model_path": model_path}

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy")
async def deploy_model(request: DeployModelRequest):
    """Deploy a trained model as the default model."""
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Model file not found")

    # Copy to standard location
    deploy_path = Path("models") / "deployed" / "current_model.pt"
    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, deploy_path)

    return {"status": "deployed", "path": str(deploy_path), "message": "Model deployed successfully"}


@router.post("/download")
async def download_model(request: DownloadModelRequest):
    """Download a trained model file."""
    from fastapi.responses import FileResponse

    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Model file not found")

    return FileResponse(
        model_path, media_type="application/octet-stream", filename=f"trained_model.pt"
    )


@router.post("/export-hf")
async def export_to_huggingface(request: ExportHFRequest):
    """Export model to HuggingFace Hub."""
    model_path = Path(request.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Model file not found")

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        # Create repository
        repo_url = create_repo(request.repo_name, exist_ok=True)

        # Upload model
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="model.pt",
            repo_id=request.repo_name,
        )

        return {"status": "exported", "repo_url": repo_url, "message": "Model exported to HuggingFace"}

    except ImportError:
        raise HTTPException(status_code=500, detail="huggingface_hub not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
