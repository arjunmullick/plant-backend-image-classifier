# GPU-Free Training Guide for Plant Image Classifier

A comprehensive guide for training and improving the plant image classifier **without local GPU access**, using managed cloud services and cost-efficient approaches.

---

## Table of Contents

1. [Overview of GPU-Free Training Options](#1-overview-of-gpu-free-training-options)
2. [Recommended Training Architecture](#2-recommended-training-architecture)
3. [Dataset Preparation Strategy](#3-dataset-preparation-strategy)
4. [Role of OpenAI / Claude APIs](#4-role-of-openai--claude-apis)
5. [Step-by-Step Training Pipeline](#5-step-by-step-training-pipeline)
6. [Cost & Iteration Strategy](#6-cost--iteration-strategy)
7. [Iteration Roadmap](#7-iteration-roadmap)
8. [Risks, Tradeoffs, and Mitigations](#8-risks-tradeoffs-and-mitigations)

---

## 1. Overview of GPU-Free Training Options

### Option A: Hugging Face AutoTrain + Spaces

**What it is**: Managed fine-tuning service that handles GPU provisioning automatically.

| Aspect | Details |
|--------|---------|
| **When to use** | MVP, quick iteration, small-medium datasets (<100K images) |
| **Cost** | ~$0.60-1.20/hour for training; ~$5-50 per fine-tuning run |
| **Pros** | Zero infrastructure, version control built-in, easy deployment |
| **Cons** | Less customization, dataset size limits, queuing during peak |
| **MVP fit** | Excellent |
| **Production fit** | Good for small-medium scale |

```
Training Flow:
Dataset (HF Hub) → AutoTrain Job → Model Card → Inference Endpoint
```

---

### Option B: Google Vertex AI Training

**What it is**: Managed ML training with custom containers or AutoML Vision.

| Aspect | Details |
|--------|---------|
| **When to use** | Production scale, custom training loops, >100K images |
| **Cost** | ~$1-3/hour (T4), ~$2-6/hour (A100); $20-200 per training run |
| **Pros** | Full control, scales to large datasets, integrates with GCS |
| **Cons** | More setup, requires some MLOps knowledge |
| **MVP fit** | Overkill |
| **Production fit** | Excellent |

```
Training Flow:
GCS Bucket → Custom Training Job → Model Registry → Vertex Endpoint
```

---

### Option C: AWS SageMaker Training Jobs

**What it is**: Managed training with built-in algorithms or custom scripts.

| Aspect | Details |
|--------|---------|
| **When to use** | AWS-centric stack, enterprise requirements |
| **Cost** | ~$1-4/hour (ml.g4dn), higher egress costs |
| **Pros** | Enterprise features, spot instances (70% savings), model monitoring |
| **Cons** | Complex pricing, AWS lock-in, verbose API |
| **MVP fit** | Moderate |
| **Production fit** | Excellent |

---

### Option D: Modal Labs / RunPod Serverless (Recommended for MVP)

**What it is**: Pay-per-second GPU compute with Python-native APIs.

| Aspect | Details |
|--------|---------|
| **When to use** | Cost-sensitive training, burst compute needs |
| **Cost** | ~$0.30-0.80/hour (T4/A10), pay only for active time |
| **Pros** | Cheapest option, no idle costs, simple Python API |
| **Cons** | Less managed, requires more code |
| **MVP fit** | Good |
| **Production fit** | Good with proper orchestration |

---

### Option E: Replicate Training

**What it is**: Train and deploy models via API with versioned outputs.

| Aspect | Details |
|--------|---------|
| **When to use** | Rapid prototyping, sharing models publicly |
| **Cost** | ~$0.50-1.50/hour, simple per-second billing |
| **Pros** | Instant deployment, version control, public sharing |
| **Cons** | Limited customization, cost adds up at scale |
| **MVP fit** | Excellent |
| **Production fit** | Moderate |

---

### Comparison Matrix

| Criterion | HF AutoTrain | Vertex AI | SageMaker | Modal | Replicate |
|-----------|--------------|-----------|-----------|-------|-----------|
| Setup time | 1 hour | 1 day | 1 day | 2 hours | 1 hour |
| Cost per run | $5-50 | $20-200 | $30-250 | $5-30 | $10-60 |
| Customization | Low | High | High | High | Low |
| MVP suitability | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★☆ |
| Production scale | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |

---

### Recommendation: Start with Modal Labs, Graduate to Vertex AI

**Phase 1 (MVP)**: Modal Labs for training + Hugging Face Hub for model storage
- Lowest cost, fastest iteration
- Full control over training code
- Pay only for actual GPU seconds

**Phase 2 (Production)**: Migrate to Vertex AI
- Better monitoring and MLOps
- Scalable to millions of images
- Enterprise-grade reliability

---

## 2. Recommended Training Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Training Infrastructure                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │   Local Machine   │      │    Modal Labs    │      │  Hugging Face    │  │
│  │   (No GPU)        │      │   (GPU Compute)  │      │     Hub          │  │
│  │                   │      │                   │      │                   │  │
│  │  • Dataset prep   │ ───► │  • Training jobs │ ───► │  • Model storage │  │
│  │  • Config files   │      │  • T4/A10/A100   │      │  • Versioning    │  │
│  │  • Job triggers   │      │  • Pay per second│      │  • Model cards   │  │
│  │  • Validation     │      │  • Auto-scaling  │      │                   │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
│           │                          │                          │           │
│           │                          │                          │           │
│           ▼                          ▼                          ▼           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Cloud Storage (S3/GCS)                        │  │
│  │                                                                        │  │
│  │    datasets/              training_logs/           checkpoints/       │  │
│  │    ├── plantvillage/      ├── run_001/            ├── species_v1/    │  │
│  │    ├── custom_labeled/    ├── run_002/            ├── disease_v1/    │  │
│  │    └── augmented/         └── ...                 └── ...            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      FastAPI Backend (Your App)                       │  │
│  │                                                                        │  │
│  │    1. Pull model from HF Hub on startup                               │  │
│  │    2. Load weights into SpeciesClassifier / DiseaseDetector           │  │
│  │    3. Serve predictions via /api/v1/classify                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Training Jobs Are Triggered

Create `training/modal_train.py`:

```python
# training/modal_train.py - Run this locally (no GPU needed)

import modal

# Define the Modal app
app = modal.App("plant-classifier-training")

# Define the training image with all dependencies
training_image = modal.Image.debian_slim().pip_install(
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "timm>=0.9.0",
    "datasets>=2.14.0",
    "huggingface_hub>=0.19.0",
    "wandb>=0.16.0",
    "albumentations>=1.3.0",
    "scikit-learn>=1.3.0",
)

# Mount your training code
training_code = modal.Mount.from_local_dir("./training", remote_path="/app/training")

@app.function(
    image=training_image,
    gpu="T4",  # or "A10G" for faster training
    timeout=3600,  # 1 hour max
    mounts=[training_code],
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train_species_classifier(
    dataset_path: str,
    model_name: str = "google/efficientnet-b0",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    run_name: str = "species_v1",
):
    """Train species classifier on Modal GPU."""
    import torch
    from training.train_species import train_model

    # Training happens on Modal's GPU
    model, metrics = train_model(
        dataset_path=dataset_path,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Push to Hugging Face Hub
    from huggingface_hub import HfApi
    api = HfApi()

    # Save model locally first
    torch.save(model.state_dict(), f"/tmp/{run_name}.pt")

    # Upload to HF Hub
    api.upload_file(
        path_or_fileobj=f"/tmp/{run_name}.pt",
        path_in_repo=f"checkpoints/{run_name}.pt",
        repo_id="your-username/plant-classifier",
        repo_type="model",
    )

    return metrics


# Trigger training from your local machine
if __name__ == "__main__":
    with app.run():
        metrics = train_species_classifier.remote(
            dataset_path="your-username/plantvillage-cleaned",
            epochs=15,
            run_name="species_v2_efficientnet",
        )
        print(f"Training complete! Metrics: {metrics}")
```

### Where Datasets Live

**Recommended: Hugging Face Datasets Hub**
- Free hosting for datasets
- Streaming support (no full download needed)
- Built-in train/val/test splits
- Version control

**Alternative: Cloud Object Storage**
- S3/GCS for larger datasets
- More control over access
- Better for private/proprietary data

**Upload dataset to HF Hub:**

```python
# scripts/upload_to_hub.py
from datasets import Dataset, DatasetDict, Image

def prepare_dataset():
    """Prepare and upload PlantVillage to HF Hub."""
    from datasets import load_dataset

    # Load images from local directory
    dataset = load_dataset("imagefolder", data_dir="./plantvillage_raw")

    # Add metadata
    dataset = dataset.map(lambda x: {
        "label_name": LABEL_NAMES[x["label"]],
        "crop": extract_crop(x["label"]),
        "disease": extract_disease(x["label"]),
    })

    # Push to Hub
    dataset.push_to_hub("your-username/plantvillage-cleaned")
```

### Model Versioning Strategy

```python
# scripts/version_model.py

from huggingface_hub import HfApi, ModelCard

def upload_trained_model(
    model_path: str,
    version: str,
    metrics: dict,
    config: dict,
):
    """Upload model with full versioning."""
    api = HfApi()
    repo_id = "your-username/plant-classifier"

    # Create model card with training info
    card_content = f"""
---
license: mit
tags:
  - plant-classification
  - disease-detection
  - agriculture
metrics:
  - accuracy: {metrics['accuracy']:.4f}
  - f1: {metrics['f1']:.4f}
---

# Plant Image Classifier v{version}

## Training Details
- Dataset: PlantVillage + Custom
- Base model: {config['base_model']}
- Epochs: {config['epochs']}
- Final accuracy: {metrics['accuracy']:.2%}

## Usage
```python
from your_package import load_classifier
model = load_classifier("your-username/plant-classifier", revision="{version}")
```
"""

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"checkpoints/model_v{version}.pt",
        repo_id=repo_id,
        commit_message=f"Add model v{version} - accuracy: {metrics['accuracy']:.2%}",
    )

    # Create a git tag for this version
    api.create_tag(repo_id, tag=f"v{version}")

    return f"https://huggingface.co/{repo_id}/tree/v{version}"
```

### Loading Models in Backend

Add to `app/ml/model_loader.py`:

```python
from huggingface_hub import hf_hub_download
import torch
from pathlib import Path
import os

class ModelLoader:
    """Load trained models from Hugging Face Hub."""

    REPO_ID = "your-username/plant-classifier"
    CACHE_DIR = Path("./model_cache")

    @classmethod
    def load_species_model(
        cls,
        version: str = "latest",
        device: str = "cpu",
    ) -> torch.nn.Module:
        """
        Load species classifier from HF Hub.

        Args:
            version: Model version tag (e.g., "v1.2.0") or "latest"
            device: Target device

        Returns:
            Loaded PyTorch model
        """
        # Determine revision
        revision = None if version == "latest" else f"v{version}"

        # Download model file
        model_path = hf_hub_download(
            repo_id=cls.REPO_ID,
            filename="checkpoints/species_classifier.pt",
            revision=revision,
            cache_dir=cls.CACHE_DIR,
        )

        # Load model architecture
        from app.ml.architectures import SpeciesClassifierNet
        model = SpeciesClassifierNet(num_classes=38)  # PlantVillage classes

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model

    @classmethod
    def load_disease_model(
        cls,
        crop: str,
        version: str = "latest",
        device: str = "cpu",
    ) -> torch.nn.Module:
        """Load crop-specific disease model."""
        revision = None if version == "latest" else f"v{version}"

        model_path = hf_hub_download(
            repo_id=cls.REPO_ID,
            filename=f"checkpoints/disease_{crop}.pt",
            revision=revision,
            cache_dir=cls.CACHE_DIR,
        )

        from app.ml.architectures import DiseaseClassifierNet
        num_classes = CROP_DISEASE_COUNTS[crop]
        model = DiseaseClassifierNet(num_classes=num_classes)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model
```

---

## 3. Dataset Preparation Strategy

### Step 1: Download and Organize Raw Data

Create `scripts/prepare_dataset.py`:

```python
import os
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import json

def download_plantvillage():
    """Download PlantVillage dataset."""
    dataset = load_dataset("plantvillage", split="train")
    return dataset

def download_plantdoc():
    """Download PlantDoc for real-world variety."""
    dataset = load_dataset("jxie/plantdoc", split="train")
    return dataset

def create_unified_dataset(output_dir: Path):
    """Create unified dataset with consistent labeling."""

    # Download base datasets
    plantvillage = download_plantvillage()
    plantdoc = download_plantdoc()

    # Create unified label mapping
    label_mapping = create_label_mapping([plantvillage, plantdoc])

    # Save mapping for training
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Process and save
    unified = process_and_merge(plantvillage, plantdoc, label_mapping)
    unified.save_to_disk(output_dir / "unified")

    return unified
```

### Step 2: CPU-Based Augmentation Pipeline

Create `scripts/augment_dataset.py`:

```python
import albumentations as A
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

# Augmentation pipeline (CPU-based)
augmentation_pipeline = A.Compose([
    # Geometric transforms
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=30, p=0.5),

    # Color transforms (crucial for plant images)
    A.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
        p=0.8
    ),

    # Simulate real-world conditions
    A.OneOf([
        A.GaussianBlur(blur_limit=3),
        A.MotionBlur(blur_limit=3),
    ], p=0.2),

    # Simulate partial occlusion (leaves overlapping)
    A.CoarseDropout(
        max_holes=8,
        max_height=20,
        max_width=20,
        fill_value=0,
        p=0.1
    ),

    # Simulate different lighting
    A.RandomBrightnessContrast(p=0.5),
    A.RandomShadow(p=0.2),
])

def augment_image(args):
    """Augment a single image (for parallel processing)."""
    image_path, output_dir, num_augmentations = args

    image = np.array(Image.open(image_path))
    base_name = Path(image_path).stem

    augmented_paths = []
    for i in range(num_augmentations):
        augmented = augmentation_pipeline(image=image)["image"]

        output_path = output_dir / f"{base_name}_aug{i}.jpg"
        Image.fromarray(augmented).save(output_path, quality=95)
        augmented_paths.append(output_path)

    return augmented_paths

def augment_dataset_parallel(
    input_dir: Path,
    output_dir: Path,
    augmentations_per_image: int = 5,
    num_workers: int = 8,
):
    """Augment entire dataset using CPU parallelization."""

    image_paths = list(input_dir.glob("**/*.jpg")) + list(input_dir.glob("**/*.png"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments
    args = [(p, output_dir, augmentations_per_image) for p in image_paths]

    # Process in parallel using CPU
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(augment_image, args),
            total=len(args),
            desc="Augmenting images"
        ))

    print(f"Created {sum(len(r) for r in results)} augmented images")
```

### Step 3: Handle Class Imbalance

Create `scripts/balance_dataset.py`:

```python
from collections import Counter
import random
from pathlib import Path
import shutil

def analyze_class_distribution(dataset_dir: Path) -> dict:
    """Analyze class distribution."""
    class_counts = Counter()

    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = count

    return class_counts

def balance_classes(
    dataset_dir: Path,
    output_dir: Path,
    strategy: str = "oversample",  # or "undersample" or "weighted"
    target_count: int = None,
):
    """Balance class distribution."""

    class_counts = analyze_class_distribution(dataset_dir)

    if strategy == "oversample":
        # Oversample minority classes to match majority
        target = target_count or max(class_counts.values())

        for class_name, count in class_counts.items():
            class_dir = dataset_dir / class_name
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            images = list(class_dir.glob("*.jpg"))

            # Copy all original images
            for img in images:
                shutil.copy(img, output_class_dir / img.name)

            # Oversample if needed
            if count < target:
                needed = target - count
                oversampled = random.choices(images, k=needed)
                for i, img in enumerate(oversampled):
                    shutil.copy(img, output_class_dir / f"oversample_{i}_{img.name}")

    elif strategy == "weighted":
        # Return class weights for weighted loss function
        total = sum(class_counts.values())
        weights = {
            cls: total / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        return weights

    return analyze_class_distribution(output_dir)

def generate_class_weights(class_counts: dict) -> list:
    """Generate weights for CrossEntropyLoss."""
    total = sum(class_counts.values())
    num_classes = len(class_counts)

    # Inverse frequency weighting
    weights = []
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        weight = total / (num_classes * count)
        weights.append(weight)

    return weights
```

### Step 4: Label Validation

Create `scripts/validate_labels.py`:

```python
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict

def validate_dataset(dataset_dir: Path) -> dict:
    """Validate dataset integrity and labels."""

    issues = defaultdict(list)
    stats = {
        "total_images": 0,
        "valid_images": 0,
        "corrupt_images": 0,
        "classes": {},
    }

    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_count = 0

        for img_path in class_dir.glob("*"):
            stats["total_images"] += 1

            # Check if image is readable
            try:
                with Image.open(img_path) as img:
                    img.verify()
                stats["valid_images"] += 1
                class_count += 1
            except Exception as e:
                issues["corrupt_images"].append({
                    "path": str(img_path),
                    "error": str(e)
                })
                stats["corrupt_images"] += 1

        stats["classes"][class_name] = class_count

        # Check for suspiciously low counts
        if class_count < 50:
            issues["low_sample_classes"].append({
                "class": class_name,
                "count": class_count,
                "recommendation": "Consider augmentation or merging with similar class"
            })

    # Check for class imbalance
    counts = list(stats["classes"].values())
    if max(counts) > 10 * min(counts):
        issues["severe_imbalance"].append({
            "max_class_count": max(counts),
            "min_class_count": min(counts),
            "ratio": max(counts) / min(counts),
            "recommendation": "Use class weighting or resampling"
        })

    return {"stats": stats, "issues": dict(issues)}

def validate_taxonomy_consistency(
    dataset_dir: Path,
    taxonomy_file: Path,
) -> dict:
    """Ensure labels match expected taxonomy."""

    with open(taxonomy_file) as f:
        expected_taxonomy = json.load(f)

    expected_classes = set(expected_taxonomy.keys())
    actual_classes = set(d.name for d in dataset_dir.iterdir() if d.is_dir())

    return {
        "missing_in_dataset": list(expected_classes - actual_classes),
        "unexpected_in_dataset": list(actual_classes - expected_classes),
        "matched": len(expected_classes & actual_classes),
    }
```

---

## 4. Role of OpenAI / Claude APIs

### LLM Usage Boundaries (Critical)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM USAGE BOUNDARIES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ✅ GOOD USES (Assistance Layer)          ❌ BAD USES (Core Classification) │
│  ─────────────────────────────────        ──────────────────────────────── │
│                                                                              │
│  • Label suggestions for review            • Primary species classifier      │
│  • Explanation text generation             • Primary disease detector        │
│  • Prediction validation/flagging          • Confidence score source         │
│  • Data quality auditing                   • Treatment recommendations*      │
│  • User-facing natural language            • Production classification       │
│  • Edge case triage                        • Real-time inference             │
│                                                                              │
│  *Treatment text can use LLM, but                                           │
│   treatment LOGIC should be rule-based                                      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHY NOT USE LLMs FOR CORE CLASSIFICATION?                                  │
│                                                                              │
│  1. COST: $0.003-0.02 per image vs $0.0001 for ML inference                │
│     → 30-200x more expensive at scale                                       │
│                                                                              │
│  2. LATENCY: 500-2000ms vs 50-100ms                                        │
│     → 10-20x slower                                                         │
│                                                                              │
│  3. CONSISTENCY: LLMs can give different answers for same image            │
│     → Not reproducible for scientific/agricultural use                      │
│                                                                              │
│  4. ACCURACY: Vision LLMs are general-purpose                              │
│     → Fine-tuned CNNs outperform on domain-specific tasks                  │
│                                                                              │
│  5. OFFLINE: ML models work without internet                               │
│     → Critical for field/rural deployment                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Good Use Case 1: Weak Labeling / Label Suggestions

Create `scripts/llm_labeling_assistant.py`:

```python
import anthropic
import base64
from pathlib import Path
import json
from tqdm import tqdm

client = anthropic.Anthropic()

def get_label_suggestion(image_path: Path, possible_labels: list[str]) -> dict:
    """
    Use Claude to suggest labels for ambiguous images.

    IMPORTANT: This is for ASSISTANCE only, not final labeling.
    Human review is still required.
    """

    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    suffix = image_path.suffix.lower()
    media_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""You are helping label plant disease images for a machine learning dataset.

Analyze this plant image and suggest the most likely label from this list:
{json.dumps(possible_labels, indent=2)}

Respond with JSON only:
{{
    "suggested_label": "exact label from list",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation",
    "visible_symptoms": ["symptom1", "symptom2"],
    "needs_human_review": true/false
}}

If you cannot determine the label with reasonable confidence, set needs_human_review to true."""
                    }
                ],
            }
        ],
    )

    return json.loads(response.content[0].text)

def batch_label_suggestions(
    unlabeled_dir: Path,
    possible_labels: list[str],
    output_file: Path,
):
    """Process batch of unlabeled images."""

    results = []
    images = list(unlabeled_dir.glob("*.jpg"))

    for img_path in tqdm(images, desc="Getting label suggestions"):
        try:
            suggestion = get_label_suggestion(img_path, possible_labels)
            suggestion["image_path"] = str(img_path)
            results.append(suggestion)
        except Exception as e:
            results.append({
                "image_path": str(img_path),
                "error": str(e),
                "needs_human_review": True
            })

    # Save for human review
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    high_conf = sum(1 for r in results if r.get("confidence") == "high")
    needs_review = sum(1 for r in results if r.get("needs_human_review"))

    print(f"Results: {high_conf} high confidence, {needs_review} need review")

    return results
```

### Good Use Case 2: Prediction Validation

Add to `app/services/llm_validation.py`:

```python
import anthropic
import json

client = anthropic.Anthropic()

async def validate_prediction_with_llm(
    image_base64: str,
    ml_prediction: dict,
) -> dict:
    """
    Use Claude to validate ML predictions.

    This adds a confidence check, NOT a replacement for the ML model.
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""Our ML model predicted:
- Species: {ml_prediction['species']} (confidence: {ml_prediction['species_confidence']:.0%})
- Disease: {ml_prediction['disease']} (confidence: {ml_prediction['disease_confidence']:.0%})

Please verify if this prediction appears reasonable based on the image.

Respond with JSON:
{{
    "species_agreement": "agree/disagree/uncertain",
    "disease_agreement": "agree/disagree/uncertain",
    "concerns": ["any concerns about the prediction"],
    "alternative_suggestions": ["if you disagree, what might it be"],
    "recommendation": "accept/flag_for_review/reject"
}}"""
                    }
                ],
            }
        ],
    )

    validation = json.loads(response.content[0].text)

    # If LLM disagrees, flag for human review but still return ML prediction
    if validation["recommendation"] != "accept":
        validation["flag_reason"] = "LLM validation disagreed with ML prediction"

    return validation
```

### Good Use Case 3: Explanation Enhancement

Add to `app/ml/llm_explainer.py`:

```python
import anthropic

client = anthropic.Anthropic()

async def enhance_explanation(
    image_base64: str,
    ml_prediction: dict,
    grad_cam_regions: list[str],
) -> str:
    """
    Use Claude to generate better explanations.

    This is a GOOD use case - LLMs excel at natural language generation.
    """

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""Our plant disease classifier identified:
- Disease: {ml_prediction['disease']}
- Visual symptoms detected: {ml_prediction['visual_symptoms']}
- Model attention focused on: {grad_cam_regions}

Write a clear, farmer-friendly explanation (2-3 sentences) of:
1. What the disease looks like in this image
2. What visual features led to this diagnosis

Be specific to what's visible in the image. Use simple language."""
                    }
                ],
            }
        ],
    )

    return response.content[0].text
```

### Good Use Case 4: Training Data Quality Review

Create `scripts/llm_data_quality.py`:

```python
import anthropic
import base64
from pathlib import Path
import json

client = anthropic.Anthropic()

def review_dataset_quality(
    sample_images: list[Path],
    expected_class: str,
) -> dict:
    """Use Claude to check if images match their labels."""

    mismatches = []

    for img_path in sample_images:
        with open(img_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""This image is labeled as: {expected_class}

Does this label appear correct? Respond with JSON:
{{"correct": true/false, "actual_appearance": "what you see", "confidence": "high/medium/low"}}"""
                        }
                    ],
                }
            ],
        )

        result = json.loads(response.content[0].text)
        if not result["correct"]:
            mismatches.append({
                "path": str(img_path),
                "expected": expected_class,
                "actual_appearance": result["actual_appearance"],
            })

    return {
        "total_reviewed": len(sample_images),
        "mismatches": mismatches,
        "mismatch_rate": len(mismatches) / len(sample_images),
    }
```

---

## 5. Step-by-Step Training Pipeline

### Complete Training Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE OVERVIEW                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  PHASE 1: Data Preparation (Local, No GPU)                                   │
│  ─────────────────────────────────────────                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │Download │ → │ Clean & │ → │Augment  │ → │Balance  │ → │ Upload  │   │
│  │Datasets │    │Validate │    │(CPU)    │    │Classes  │    │ to Hub  │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│       ↓              ↓              ↓              ↓              ↓          │
│   PlantVillage   Fix corrupt    5x augment    Oversample     HF Datasets    │
│   PlantDoc       images         per image     minorities     your-user/     │
│   iNaturalist    Fix labels                                  plantdata      │
│                                                                               │
│  PHASE 2: Training (Modal Labs GPU)                                          │
│  ──────────────────────────────────                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │ Trigger │ → │  Load   │ → │  Train  │ → │Validate │ → │  Save   │   │
│  │   Job   │    │ Dataset │    │ Model   │    │ Metrics │    │ Model   │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│       ↓              ↓              ↓              ↓              ↓          │
│   modal run      Stream from   EfficientNet   Accuracy/F1    HF Hub +       │
│   train.py       HF Hub        fine-tune      per class      W&B logging    │
│                                                                               │
│  PHASE 3: Deployment (Your Backend)                                          │
│  ──────────────────────────────────                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                   │
│  │  Pull   │ → │  Load   │ → │  Test   │ → │ Deploy  │                   │
│  │ Model   │    │ Weights │    │Endpoint │    │   API   │                   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                   │
│       ↓              ↓              ↓              ↓                         │
│   hf_hub_       torch.load    Smoke tests    Update env                     │
│   download()                   + accuracy    MODEL_VERSION                   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Prepare Dataset (Local)

```bash
# Run these commands locally (no GPU needed)

# 1. Create project structure
mkdir -p data/{raw,processed,augmented}
mkdir -p training
mkdir -p scripts

# 2. Download datasets
python scripts/download_datasets.py

# 3. Clean and validate
python scripts/validate_labels.py --input data/raw --output data/validated

# 4. Augment (CPU parallelized)
python scripts/augment_dataset.py \
    --input data/validated \
    --output data/augmented \
    --augmentations-per-image 5 \
    --workers 8

# 5. Balance classes
python scripts/balance_dataset.py \
    --input data/augmented \
    --output data/balanced \
    --strategy oversample

# 6. Upload to Hugging Face Hub
python scripts/upload_to_hub.py \
    --input data/balanced \
    --repo your-username/plant-disease-dataset
```

### Step 2: Training Script (Runs on Modal)

Create `training/train_species.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
import wandb
from tqdm import tqdm

def train_model(
    dataset_path: str,
    model_name: str = "efficientnet_b0",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    freeze_backbone_epochs: int = 5,
):
    """
    Train species/disease classifier using transfer learning.

    Training Strategy:
    1. Freeze backbone, train head only (epochs 1-5)
    2. Unfreeze last 2 blocks, continue training (epochs 6-10)
    3. Unfreeze all, fine-tune with low LR (epochs 11-15)
    """

    # Initialize wandb
    wandb.init(project="plant-classifier", config={
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })

    # Load dataset from HF Hub (streaming to avoid full download)
    dataset = load_dataset(dataset_path)

    # Get number of classes
    num_classes = len(dataset["train"].features["label"].names)
    class_names = dataset["train"].features["label"].names

    # Create model with pretrained backbone
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
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
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
    def collate_fn(batch):
        images = torch.stack([train_transform(x["image"].convert("RGB")) for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: (
            torch.stack([val_transform(x["image"].convert("RGB")) for x in b]),
            torch.tensor([x["label"] for x in b])
        ),
        num_workers=4,
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Training loop with progressive unfreezing
    for epoch in range(epochs):
        # Progressive unfreezing strategy
        if epoch < freeze_backbone_epochs:
            # Phase 1: Freeze backbone
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_classifier().parameters():
                param.requires_grad = True
            lr = learning_rate * 10  # Higher LR for head only

        elif epoch < freeze_backbone_epochs + 5:
            # Phase 2: Unfreeze last blocks
            for param in model.parameters():
                param.requires_grad = True
            lr = learning_rate

        else:
            # Phase 3: Full fine-tuning
            lr = learning_rate * 0.1

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.2f}%"
            })

        # Validation
        model.eval()
        val_loss = 0
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

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_acc": 100. * correct / total,
            "val_loss": val_loss / len(val_loader),
            "val_acc": 100. * val_correct / val_total,
            "learning_rate": lr,
        }
        wandb.log(metrics)

        print(f"Epoch {epoch+1}: Train Acc: {metrics['train_acc']:.2f}%, Val Acc: {metrics['val_acc']:.2f}%")

    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, class_names, device)
    wandb.log({"final_metrics": final_metrics})

    return model, final_metrics


def evaluate_model(model, val_loader, class_names, device):
    """Compute per-class metrics."""
    from sklearn.metrics import classification_report

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
    )

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": report,
    }
```

### Step 3: Trigger Training (Local Command)

```bash
# Install Modal CLI
pip install modal

# Setup Modal (one-time)
modal setup

# Create secrets for HF Hub and W&B
modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxxxx
modal secret create wandb-secret WANDB_API_KEY=xxxxx

# Run training job
modal run training/modal_train.py \
    --dataset-path your-username/plant-disease-dataset \
    --epochs 15 \
    --model efficientnet_b0

# Monitor in real-time
# - Modal dashboard: https://modal.com/apps
# - W&B dashboard: https://wandb.ai/your-username/plant-classifier
```

### Step 4: Deploy Model to Backend

Create `scripts/deploy_model.py`:

```python
import os
from huggingface_hub import HfApi

def deploy_new_model(
    model_path: str,
    version: str,
    metrics: dict,
):
    """Deploy trained model to production."""

    # 1. Upload to HF Hub with version tag
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"checkpoints/species_v{version}.pt",
        repo_id="your-username/plant-classifier",
        commit_message=f"Deploy v{version} - accuracy: {metrics['accuracy']:.2%}",
    )
    api.create_tag(
        repo_id="your-username/plant-classifier",
        tag=f"v{version}",
    )

    # 2. Update backend environment variable
    print(f"Update SPECIES_MODEL_VERSION to: {version}")

    # 3. Trigger backend restart to load new model
    print("Restart backend to load new model")

    return f"Deployed v{version}"
```

---

## 6. Cost & Iteration Strategy

### Cost Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COST ANALYSIS PER TRAINING RUN                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Dataset: 50,000 images (PlantVillage scale)                                │
│  Model: EfficientNet-B0                                                      │
│  Epochs: 15                                                                  │
│  Batch size: 32                                                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ COMPUTE COSTS (Modal Labs)                                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ GPU: T4 (16GB)           │ $0.59/hour                               │    │
│  │ Training time            │ ~2 hours                                 │    │
│  │ Per-run compute cost     │ ~$1.20                                   │    │
│  │                          │                                          │    │
│  │ GPU: A10G (24GB)         │ $1.10/hour                               │    │
│  │ Training time            │ ~1 hour (faster)                         │    │
│  │ Per-run compute cost     │ ~$1.10                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STORAGE COSTS                                                       │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ Hugging Face Hub         │ Free (public repos)                      │    │
│  │ Hugging Face Hub         │ $9/month (private repos)                 │    │
│  │ S3/GCS (50GB dataset)    │ ~$1-2/month                              │    │
│  │ W&B logging              │ Free tier sufficient                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LLM COSTS (Optional - Label Assistance)                             │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ Claude Sonnet per image  │ ~$0.01-0.02                              │    │
│  │ 1,000 images reviewed    │ ~$10-20                                  │    │
│  │ Use sparingly!           │ Only for ambiguous cases                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ MONTHLY BUDGET SCENARIOS                                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ MVP (2 training runs/month)     │ ~$5-10/month                      │    │
│  │ Active iteration (8 runs/month) │ ~$15-25/month                     │    │
│  │ Production (4 runs/month)       │ ~$10-15/month                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### When to Retrain

Create `scripts/retrain_decision.py`:

```python
from datetime import datetime

def should_retrain(metrics: dict, feedback: dict, config: dict) -> tuple[bool, str]:
    """
    Decide if retraining is worth it.

    Returns: (should_retrain, reason)
    """

    # 1. Accuracy degradation detected
    if metrics["recent_accuracy"] < metrics["baseline_accuracy"] - 0.05:
        return True, "Accuracy dropped >5% from baseline"

    # 2. Significant new data available
    new_samples = feedback["new_labeled_samples"]
    if new_samples > config["retrain_threshold_samples"]:  # e.g., 1000
        return True, f"{new_samples} new labeled samples available"

    # 3. New class added
    if feedback["new_classes_added"] > 0:
        return True, f"New classes added: {feedback['new_classes_added']}"

    # 4. High error rate on specific class
    for class_name, error_rate in metrics["per_class_errors"].items():
        if error_rate > 0.3:  # >30% error rate
            return True, f"High error rate ({error_rate:.0%}) on {class_name}"

    # 5. Scheduled periodic retrain (e.g., monthly)
    days_since_last = (datetime.now() - metrics["last_train_date"]).days
    if days_since_last > config["max_days_without_retrain"]:  # e.g., 30
        return True, f"Scheduled retrain ({days_since_last} days since last)"

    return False, "No retrain triggers met"
```

### Incremental Improvement Without Full Retrain

```python
# scripts/incremental_training.py

import torch
import torch.nn as nn

def add_new_class_incrementally(
    new_class_name: str,
    new_class_images: list,
    existing_model_path: str,
):
    """
    Add new class with minimal retraining.

    Strategy: Freeze backbone, only train new classifier head
    """

    # 1. Load existing model
    model = load_model(existing_model_path)

    # 2. Expand classifier head
    old_num_classes = model.classifier.out_features
    new_num_classes = old_num_classes + 1

    # Create new classifier with extra class
    old_classifier = model.classifier
    new_classifier = nn.Linear(
        old_classifier.in_features,
        new_num_classes,
    )

    # Copy old weights
    with torch.no_grad():
        new_classifier.weight[:old_num_classes] = old_classifier.weight
        new_classifier.bias[:old_num_classes] = old_classifier.bias

    model.classifier = new_classifier

    # 3. Freeze backbone, train only classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 4. Fine-tune on new class + sample of old classes
    balanced_dataset = create_balanced_finetuning_set(
        new_class_images,
        sample_from_old_classes=1000,  # Keep old knowledge
    )

    # 5. Short training (2-3 epochs is usually enough)
    fine_tune(model, balanced_dataset, epochs=3)

    return model
```

### Improving Accuracy Without Full Retrain

| Technique | Effort | Impact | When to Use |
|-----------|--------|--------|-------------|
| Better augmentation | Low | Medium | First try |
| Class balancing | Low | Medium | Imbalanced classes |
| Learning rate tuning | Low | Medium | Accuracy plateau |
| More epochs | Low | Low-Medium | Underfitting |
| Larger model (B0→B2) | Medium | Medium | Plenty of data |
| Add training data | Medium | High | Specific class errors |
| Ensemble models | High | Medium-High | Last 1-2% accuracy |
| Crop-specific models | High | High | Different crops need different models |

---

## 7. Iteration Roadmap

### From Placeholder to First Real Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ITERATION ROADMAP: MVP TO PRODUCTION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WEEK 1-2: First Real Model                                                  │
│  ──────────────────────────                                                  │
│  □ Download PlantVillage dataset                                             │
│  □ Run validation script (fix corrupt images)                                │
│  □ Upload to HF Hub                                                          │
│  □ Run first training job on Modal (T4, 10 epochs)                          │
│  □ Deploy to staging backend                                                 │
│  □ Test with 100 sample images                                               │
│  Target: 80%+ accuracy on PlantVillage test set                             │
│                                                                              │
│  WEEK 3-4: Improved Model                                                    │
│  ────────────────────────                                                    │
│  □ Add data augmentation (5x per image)                                      │
│  □ Add PlantDoc dataset (real-world variety)                                 │
│  □ Implement class balancing                                                 │
│  □ Train longer (15 epochs) with progressive unfreezing                      │
│  □ Add per-class metrics tracking                                            │
│  Target: 85%+ accuracy, <20% error on any single class                      │
│                                                                              │
│  WEEK 5-6: Production Hardening                                              │
│  ───────────────────────────                                                 │
│  □ Add model versioning to HF Hub                                            │
│  □ Implement A/B testing in backend                                          │
│  □ Add confidence calibration                                                │
│  □ Set up W&B monitoring                                                     │
│  □ Create retraining trigger logic                                           │
│  Target: Stable production deployment with monitoring                        │
│                                                                              │
│  ONGOING: Continuous Improvement                                             │
│  ───────────────────────────────                                             │
│  □ Collect user feedback via /feedback endpoint                              │
│  □ Monthly LLM-assisted label review (sample 500 predictions)                │
│  □ Quarterly retrain with accumulated feedback                               │
│  □ Add new crops/diseases as needed (incremental training)                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Validating Improvements Without Full Ground Truth

Create `scripts/validate_model_improvement.py`:

```python
import numpy as np
import random

def validate_without_ground_truth(
    model_a,  # Old model
    model_b,  # New model
    unlabeled_images: list,
) -> dict:
    """Compare models without ground truth labels."""

    results = {
        "agreement_rate": 0,
        "confidence_improvement": 0,
        "flipped_predictions": [],
    }

    agreements = 0
    confidence_diffs = []

    for img in unlabeled_images:
        pred_a = model_a.predict(img)
        pred_b = model_b.predict(img)

        # Track agreement
        if pred_a["class"] == pred_b["class"]:
            agreements += 1
        else:
            results["flipped_predictions"].append({
                "image": img.path,
                "old_pred": pred_a["class"],
                "new_pred": pred_b["class"],
                "old_conf": pred_a["confidence"],
                "new_conf": pred_b["confidence"],
            })

        # Track confidence changes
        confidence_diffs.append(pred_b["confidence"] - pred_a["confidence"])

    results["agreement_rate"] = agreements / len(unlabeled_images)
    results["confidence_improvement"] = np.mean(confidence_diffs)

    # Flag: If too many flips, investigate
    if results["agreement_rate"] < 0.9:
        results["warning"] = "High disagreement - manually review flipped predictions"

    # Use LLM to validate sample of flipped predictions
    if results["flipped_predictions"]:
        sample = random.sample(
            results["flipped_predictions"],
            min(50, len(results["flipped_predictions"]))
        )
        llm_validation = validate_flips_with_llm(sample)
        results["llm_validation"] = llm_validation

    return results

def validate_flips_with_llm(flipped_predictions: list) -> dict:
    """Use Claude to judge which model is correct on disagreements."""

    model_a_correct = 0
    model_b_correct = 0
    unclear = 0

    for flip in flipped_predictions:
        response = claude_judge_prediction(
            image_path=flip["image"],
            option_a=flip["old_pred"],
            option_b=flip["new_pred"],
        )

        if response["winner"] == "A":
            model_a_correct += 1
        elif response["winner"] == "B":
            model_b_correct += 1
        else:
            unclear += 1

    return {
        "old_model_wins": model_a_correct,
        "new_model_wins": model_b_correct,
        "unclear": unclear,
        "new_model_better": model_b_correct > model_a_correct,
    }
```

---

## 8. Risks, Tradeoffs, and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model overfits to PlantVillage | High | Medium | Add PlantDoc + augmentation for variety |
| Class imbalance hurts rare diseases | High | High | Use class weighting + oversampling |
| LLM labeling introduces errors | Medium | Medium | Always require human review of LLM suggestions |
| Modal/HF service downtime | Low | Medium | Keep local model cache, version checkpoints |
| Training costs exceed budget | Low | Low | Set Modal spending limits, use T4 over A100 |
| New disease not in training data | Medium | High | Implement "unknown" class with confidence threshold |
| Model confidence poorly calibrated | Medium | Medium | Use temperature scaling post-training |

### Critical Success Factors

1. **Start small**: PlantVillage only, 10 crops, prove the pipeline works
2. **Version everything**: Models, datasets, configs - all in version control
3. **Monitor per-class**: Overall accuracy hides class-specific failures
4. **Collect feedback**: The /feedback endpoint is your path to improvement
5. **Don't over-engineer**: Get to 85% accuracy before worrying about the last 5%

---

## Quick Start Summary

### Immediate Actions (This Week)

```bash
# 1. Install Modal CLI
pip install modal
modal setup

# 2. Download PlantVillage
python scripts/download_datasets.py

# 3. Upload to HF Hub
python scripts/upload_to_hub.py

# 4. Run first training
modal run training/modal_train.py --epochs 10
```

### Next 2 Weeks

1. Add augmentation pipeline
2. Train improved model (15 epochs)
3. Deploy to staging backend
4. Test with real images

### Ongoing

1. Collect user feedback
2. Monthly retrain decisions based on metrics
3. Add new crops incrementally as needed

---

## Total Estimated Costs

| Phase | Cost |
|-------|------|
| MVP (first model) | ~$5-10 |
| Several iterations | ~$20-50 |
| Monthly maintenance | ~$10-25 |

---

## Additional Resources

- [Modal Labs Documentation](https://modal.com/docs)
- [Hugging Face Hub Guide](https://huggingface.co/docs/hub)
- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)
- [timm Model Zoo](https://huggingface.co/docs/timm)
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
