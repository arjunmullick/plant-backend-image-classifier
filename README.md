# Plant Image Classification Backend

A production-grade backend service for plant species identification, disease detection, and treatment recommendations using computer vision and transfer learning.

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - System overview and API documentation |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | **GPU-free training guide** - How to train models without local GPU |

## Quick Links

- **Run the API**: `uvicorn app.main:app --reload`
- **Train a model**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **API Docs**: http://localhost:8000/docs (after starting server)

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                              │
├─────────────────────────────────────────────────────────────────────────┤
│  POST /api/v1/classify                                                   │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Classification Service (Orchestrator)               │    │
│  │                                                                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│    │
│  │  │  Image   │  │ Species  │  │ Disease  │  │  Explainability  ││    │
│  │  │Preprocess│→ │Classifier│→ │ Detector │→ │     Engine       ││    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘│    │
│  │       │              │             │               │            │    │
│  │       ▼              ▼             ▼               ▼            │    │
│  │  Quality        Hierarchical   Crop-specific    Grad-CAM       │    │
│  │  Assessment     Taxonomy       Routing          + Reasoning    │    │
│  │                                                                 │    │
│  │                              ┌──────────────────┐               │    │
│  │                              │    Treatment     │               │    │
│  │                              │  Recommendation  │←── Region     │    │
│  │                              │      Engine      │    Filtering  │    │
│  │                              └──────────────────┘               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities

1. **Hierarchical Plant Identification**
   - Family → Genus → Species classification
   - Confidence scores at each taxonomic level
   - Alternative species suggestions

2. **Disease Detection**
   - Binary healthy/diseased classification
   - Multi-class disease identification
   - Visual symptom descriptions
   - Severity estimation
   - Crop-specific model routing

3. **Treatment Recommendations**
   - Organic treatment options
   - Chemical treatments with safety notes
   - Preventive care guidelines
   - Region-specific filtering

4. **Explainability**
   - Grad-CAM visualizations
   - Human-readable reasoning
   - Confidence context
   - Uncertainty factors

## ML Architecture

### Model Selection Rationale

| Component | Model | Rationale |
|-----------|-------|-----------|
| Species Classification | EfficientNetV2-S | Best accuracy/speed tradeoff, compound scaling |
| Disease Detection | EfficientNet + Crop-specific heads | Enables specialized models per crop |
| Severity Estimation | (Optional) U-Net/DeepLabV3 | Segmentation for affected area |

### Transfer Learning Strategy

```
ImageNet Pretrained Backbone
            │
            ▼
    Global Average Pooling
            │
            ▼
      Dropout (0.3)
            │
    ┌───────┴───────┐
    │       │       │
    ▼       ▼       ▼
 Family   Genus  Species
  Head    Head    Head
```

### Training Configuration

```python
{
    "optimizer": "AdamW",
    "lr": 1e-4,
    "scheduler": "CosineAnnealingWarmRestarts",
    "epochs": 50,
    "fine_tuning_phases": [
        {"freeze_backbone": True, "epochs": 10, "lr": 1e-3},
        {"unfreeze": "last_2_blocks", "epochs": 20, "lr": 1e-4},
        {"unfreeze": "all", "epochs": 20, "lr": 1e-5}
    ]
}
```

## Data Pipeline

### Supported Datasets

- **PlantVillage**: 54,306 images, 38 crop-disease classes
- **PlantNet**: 306,293 images, 1,081 species
- **iNaturalist**: Plant observations with verified labels

### Data Augmentation Strategy

```python
transforms = [
    RandomResizedCrop(224, scale=(0.7, 1.0)),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.2),
    RandomRotation(30),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    GaussianBlur(p=0.2),
    RandomErasing(p=0.1),  # Simulates partial occlusion
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

## API Design

### Main Classification Endpoint

```http
POST /api/v1/classify
Content-Type: application/json

{
  "image": "<base64_encoded_image>",
  "region": "US-CA",
  "include_treatment": true,
  "include_explainability": true
}
```

### Response Format

```json
{
  "plant": {
    "family": { "name": "Solanaceae", "confidence": 0.94 },
    "genus": { "name": "Solanum", "confidence": 0.91 },
    "species": { "name": "Solanum lycopersicum", "confidence": 0.89 },
    "common_name": "Tomato"
  },
  "health": {
    "status": "Diseased",
    "disease": "Early Blight",
    "confidence": 0.92,
    "visual_symptoms": [
      "brown concentric rings on older leaves",
      "yellowing around lesions"
    ],
    "disease_stage": "Moderate"
  },
  "treatment": {
    "organic": ["Neem oil spray", "Remove infected leaves"],
    "chemical": ["Chlorothalonil (follow label instructions)"],
    "prevention": ["Avoid overhead watering", "Rotate crops"],
    "urgency": "soon"
  },
  "explainability": {
    "model_reasoning": "Lesion pattern and leaf discoloration matched Early Blight training examples",
    "confidence_notes": "High confidence due to clear visual markers",
    "key_features": ["concentric ring lesions", "lower leaf distribution"],
    "grad_cam": {
      "heatmap_base64": "...",
      "focus_regions": ["High attention on center region of leaf"]
    }
  },
  "metadata": {
    "processing_time_ms": 245.3,
    "model_versions": {
      "species_classifier": "0.1.0",
      "disease_detector": "0.1.0"
    }
  }
}
```

### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/classify` | POST | Full classification pipeline |
| `/api/v1/classify/species` | POST | Species identification only |
| `/api/v1/classify/disease` | POST | Disease detection only |
| `/api/v1/classify/batch` | POST | Batch classification (up to 10) |
| `/api/v1/classify/feedback` | POST | Submit prediction feedback |
| `/api/v1/classify/supported-crops` | GET | List supported crops |
| `/api/v1/classify/supported-diseases` | GET | List supported diseases |
| `/api/v1/health` | GET | Basic health check |
| `/api/v1/health/ready` | GET | Detailed readiness check |

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd plant-backend-image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
```

### Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Example Client Usage

```python
import requests
import base64

# Load and encode image
with open("plant_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/api/v1/classify",
    json={
        "image": image_base64,
        "region": "US-CA"
    }
)

result = response.json()
print(f"Species: {result['plant']['species']['name']}")
print(f"Health: {result['health']['status']}")
if result['health']['disease']:
    print(f"Disease: {result['health']['disease']}")
    print(f"Treatments: {result['treatment']['organic']}")
```

## Iteration & Scaling Plan

### Adding New Crops

1. **Add to Taxonomy Database** (`app/ml/species_classifier.py`):
```python
TaxonomyLabel(index, "Family", "Genus", "Species name", "Common name")
```

2. **Add Disease Labels** (`app/ml/disease_detector.py`):
```python
DiseaseLabel(index, "Disease Name", "crop", False,
    visual_symptoms=["symptom1", "symptom2"],
    severity_stages=["Early", "Moderate", "Severe"])
```

3. **Add Treatment Plan** (`app/services/treatment_service.py`):
```python
TreatmentDatabase.TREATMENTS["New Disease"] = TreatmentPlan(...)
```

4. **Train/Fine-tune Model** on new crop data

### Adding New Diseases

1. Add to `DiseaseDatabase.DISEASES` with visual symptoms
2. Add treatment plan to `TreatmentDatabase.TREATMENTS`
3. Fine-tune disease detector on new disease images

### Improving Accuracy Over Time

1. **Human-in-the-Loop Feedback**:
   - Use `/api/v1/classify/feedback` endpoint
   - Analyze error patterns with `HumanInTheLoopFeedback.get_error_analysis()`
   - Retrain on corrected examples

2. **Active Learning**:
   - Flag low-confidence predictions for expert review
   - Prioritize uncertain samples for labeling

3. **Model Updates**:
   - Use model versioning in registry
   - A/B test new models before deployment
   - Monitor metrics per model version

### Production Scaling

1. **Horizontal Scaling**:
   - Stateless API design enables load balancing
   - Model loading happens once per worker

2. **GPU Inference**:
   - Set `device="cuda"` in model initialization
   - Use TorchServe for production GPU inference

3. **Caching**:
   - Redis for treatment database
   - CDN for Grad-CAM images

4. **Monitoring**:
   - Prometheus metrics for latency/throughput
   - Model confidence drift detection

## Explainability Details

### Grad-CAM Implementation

Gradient-weighted Class Activation Mapping highlights regions influencing predictions:

```python
# Conceptual implementation
def grad_cam(model, image, target_class):
    # Forward pass
    features = model.backbone(image)
    logits = model.classifier(features)

    # Backward pass
    target_score = logits[0, target_class]
    target_score.backward()

    # Compute weights
    gradients = model.backbone.gradients
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted combination
    cam = (weights * features).sum(dim=1)
    cam = F.relu(cam)  # Only positive contributions
    cam = cam / cam.max()  # Normalize

    return cam
```

### When to Use Grad-CAM

- **Internal validation**: Verify model focuses on relevant features
- **User trust**: Show why predictions were made
- **Debugging**: Identify spurious correlations
- **Quality control**: Flag predictions based on background

## Project Structure

```
plant-backend-image-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entry
│   ├── api/
│   │   ├── routes/
│   │   │   ├── classify.py         # Classification endpoints
│   │   │   └── health.py           # Health check endpoints
│   ├── core/
│   │   └── config.py               # Configuration settings
│   ├── models/
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── enums.py                # Enumerations
│   ├── ml/
│   │   ├── base.py                 # ML component interfaces
│   │   ├── preprocessor.py         # Image preprocessing
│   │   ├── species_classifier.py   # Species classification
│   │   ├── disease_detector.py     # Disease detection
│   │   └── explainability.py       # Grad-CAM and explanations
│   └── services/
│       ├── classification_service.py  # Pipeline orchestration
│       └── treatment_service.py       # Treatment recommendations
├── tests/
│   └── test_api.py                 # API tests
├── training/                        # Training scripts (Modal Labs)
│   ├── modal_train.py              # Modal GPU training trigger
│   └── train_species.py            # Core training logic
├── scripts/                         # Utility scripts
│   ├── download_datasets.py        # Dataset download
│   └── upload_to_hub.py            # Upload to HF Hub
├── requirements.txt
├── README.md
└── TRAINING_GUIDE.md               # GPU-free training documentation
```

---

## Training Models (No Local GPU Required)

This project is designed to train models **without local GPU access** using cloud services.

### Quick Start Training

```bash
# 1. Install Modal CLI
pip install modal
modal setup

# 2. Create secrets
modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxxxx
modal secret create wandb-secret WANDB_API_KEY=xxxxx

# 3. Download dataset
python scripts/download_datasets.py --dataset plantvillage --output data/raw

# 4. Upload to HF Hub
python scripts/upload_to_hub.py --input data/raw/plantvillage --repo your-username/plantvillage

# 5. Train on Modal (GPU in cloud)
modal run training/modal_train.py --dataset-path your-username/plantvillage --epochs 15
```

### Training Cost Estimate

| Phase | Cost |
|-------|------|
| First model (10 epochs, T4 GPU) | ~$1-2 |
| Full training run (15 epochs) | ~$2-5 |
| Monthly iteration budget | ~$10-25 |

For detailed training documentation, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

---

## Configuration

Environment variables (prefix: `PLANT_CLASSIFIER_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `MODEL_CACHE_DIR` | `./models` | Model weights directory |
| `SPECIES_MODEL_PATH` | `None` | Species model checkpoint |
| `DISEASE_MODEL_PATH` | `None` | Disease model checkpoint |
| `ENABLE_GRAD_CAM` | `true` | Enable Grad-CAM visualizations |
| `ENABLE_REGION_FILTERING` | `false` | Filter treatments by region |

## License

MIT License - See LICENSE file for details.
