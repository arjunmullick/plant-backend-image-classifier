# Plant Image Classification Backend

A production-grade backend service for plant species identification, disease detection, and treatment recommendations using computer vision and transfer learning.

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - System overview and API documentation |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | **GPU-free training guide** - How to train models without local GPU |

## Quick Links

- **Install**: `pip install -r requirements.txt` (includes torch, transformers)
- **Run the API**: `uvicorn app.main:app --reload --port 8000`
- **Web UI**: http://localhost:8000 (interactive classification interface)
- **API Docs**: http://localhost:8000/docs (Swagger documentation)
- **Compare Models**: http://localhost:8000 → "Compare Models" tab
- **Train a model**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Run tests**: `pytest tests/ -v`

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

### Real Model Implementation

The application uses **real HuggingFace models** for inference (not placeholders):

| Component | Model | Source | Classes |
|-----------|-------|--------|---------|
| **Internal Species/Disease** | MobileNetV2 | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` | 38 PlantVillage classes |
| **External MobileNetV2** | MobileNetV2 | Same as above | 38 classes |
| **External ViT** | Vision Transformer | `wambugu71/crop_leaf_diseases_vit` | 14 classes (Corn, Potato, Rice, Wheat) |
| **External PlantNet** | API | PlantNet REST API | 50,000+ species |
| **External Kindwise** | API | Plant.id/Kindwise REST API | Superior accuracy |
| **External ResNet50** | CNN | `Diginsa/Plant-Disease-Detection-Project` | PlantVillage trained |
| **External EfficientNet** | CNN | `ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease` | High accuracy |

### Model Version Tracking

- **Real models**: Version `1.0.0-mobilenet_v2`
- **Fallback (if transformers not installed)**: Version `0.1.0-placeholder`

Check model version in API responses:
```json
{
  "metadata": {
    "model_versions": {
      "species_classifier": "1.0.0-mobilenet_v2",
      "disease_detector": "1.0.0-mobilenet_v2"
    }
  }
}
```

### Model Selection Rationale

| Component | Model | Rationale |
|-----------|-------|-----------|
| Species Classification | MobileNetV2 | Fast inference, 38 classes, ~95% accuracy |
| Disease Detection | MobileNetV2 | Combined species+disease labels, crop-specific routing |
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
| `/api/v1/classify/compare` | POST | **Compare across multiple models** |
| `/api/v1/classify/early-warning` | POST | **AI Crop Disease Early Warning System** |
| `/api/v1/classify/compare/models` | GET | List available comparison models |
| `/api/v1/classify/feedback` | POST | Submit prediction feedback |
| `/api/v1/classify/supported-crops` | GET | List supported crops |
| `/api/v1/classify/supported-diseases` | GET | List supported diseases |
| `/api/v1/health` | GET | Basic health check |
| `/api/v1/health/ready` | GET | Detailed readiness check with model versions |

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

# Install dependencies (includes torch and transformers for real ML inference)
pip install -r requirements.txt
```

### Dependencies

The project requires PyTorch and HuggingFace Transformers for real ML inference:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.1.0 | PyTorch deep learning framework |
| `torchvision` | >=0.16.0 | Image processing utilities |
| `transformers` | >=4.36.0 | HuggingFace model hub access |

**Note**: On first run, the models will be automatically downloaded from HuggingFace Hub (~100-200MB per model).

### Running the Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --port 8000

# Production
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
```

### Accessing the Application

After starting the server:

| URL | Description |
|-----|-------------|
| http://localhost:8000 | **Web UI** - Interactive plant classification interface |
| http://localhost:8000/docs | **API Docs** - Swagger/OpenAPI documentation |
| http://localhost:8000/redoc | **ReDoc** - Alternative API documentation |

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run real inference tests (requires torch/transformers)
pytest tests/test_real_inference.py -v

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
│   │   ├── explainability.py       # Grad-CAM and explanations
│   │   └── external_models.py      # External model integrations
│   └── services/
│       ├── classification_service.py  # Pipeline orchestration
│       ├── treatment_service.py       # Treatment recommendations
│       └── early_warning_service.py   # Early Warning System with severity scoring
├── static/                          # Web frontend
│   ├── index.html                  # Main web application
│   ├── css/
│   │   └── style.css               # Responsive styles
│   └── js/
│       └── app.js                  # API integration & UI logic
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

## Web Frontend

The application includes a modern web interface for testing and demonstrating all classification features.

### Running the Frontend

```bash
# Start the server
uvicorn app.main:app --reload --port 8000

# Open browser to:
http://localhost:8000
```

### Frontend Features

| Mode | Description |
|------|-------------|
| **Full Analysis** | Complete pipeline: species identification + disease detection + treatment recommendations |
| **Early Warning** | **NEW**: Comprehensive disease analysis using ALL models in parallel |
| **Species Only** | Fast species identification with taxonomy (Family → Genus → Species) |
| **Disease Only** | Disease detection with optional crop hint for better accuracy |
| **Batch (10 max)** | Process multiple images at once with summary statistics |
| **Compare Models** | Side-by-side comparison across multiple ML models |

### Compare Models Mode

The Compare Models feature allows you to validate predictions across multiple models:

**Features:**
- **Model Selection Checkboxes** - Choose which models to include
- **Agreement Score** - Visual indicator showing model consensus (High/Medium/Low)
- **Side-by-Side Cards** - Each model's prediction with:
  - Predicted disease/health status
  - Confidence score with visual bar
  - Top-3 alternative predictions
  - Processing time
  - Raw label from model
- **Recommendation Banner** - Guidance based on agreement level

**How to Use:**
1. Click the **"Compare Models"** tab
2. Select models using checkboxes (Internal Model selected by default)
3. Upload a plant image
4. Click **"Analyze Plant"**
5. Review side-by-side results

### UI Components

- **Header**: API status indicator (green = online, red = offline)
- **Tabs**: Mode selection with icons
- **Upload Area**: Drag-and-drop with preview
- **Options Panel**: Treatment, explainability, region settings
- **Model Selection Panel** (Compare mode): Checkboxes with model descriptions
- **Result Cards**: Taxonomy tree, health status, treatment tabs
- **Comparison Grid** (Compare mode): Model cards with predictions
- **API Panel**: Collapsible endpoint list and loaded models info
- **Toast Notifications**: Success/error feedback

---

## Model Comparison Feature

Compare predictions across multiple ML models side-by-side. All models now use **real HuggingFace inference**.

### Available Models for Comparison

| Model | Type | Classes | Accuracy | Description |
|-------|------|---------|----------|-------------|
| **Internal Model** | MobileNetV2 | 38 | ~95% | Our integrated classifier (species + disease) |
| **MobileNetV2 (HF)** | CNN | 38 | 95.41% | [HuggingFace](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification) |
| **ViT Crop Diseases** | Transformer | 14 | 98% | [HuggingFace](https://huggingface.co/wambugu71/crop_leaf_diseases_vit) - Corn, Potato, Rice, Wheat |
| **ResNet50 (HF)** | CNN | 38 | ~95% | [HuggingFace](https://huggingface.co/Diginsa/Plant-Disease-Detection-Project) - Deep architecture |
| **EfficientNet (HF)** | CNN | 38 | ~96% | [HuggingFace](https://huggingface.co/ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease) - Efficient, high accuracy |
| **Pl@ntNet API** | External API | 50,000+ | N/A | [PlantNet](https://my.plantnet.org/) - Requires API key |
| **Kindwise/Plant.id** | External API | 30,000+ | Superior | [Kindwise](https://www.kindwise.com/) - Best accuracy, requires API key |

### Using the Comparison UI

1. **Start the server**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

2. **Open the Web UI**: http://localhost:8000

3. **Select "Compare Models" tab**

4. **Choose models to compare**:
   - Check/uncheck the model checkboxes
   - Internal Model (recommended - always include)
   - MobileNetV2 (HuggingFace)
   - ViT Crop Diseases (best for Corn, Potato, Rice, Wheat)
   - PlantNet (requires API key)

5. **Upload an image** (drag & drop or click)

6. **Click "Analyze Plant"**

7. **View results**:
   - Side-by-side model predictions
   - Confidence scores with visual bars
   - Top-3 alternative predictions per model
   - Agreement score (High/Medium/Low)
   - Processing time per model

### Comparison Endpoint

```http
POST /api/v1/classify/compare
Content-Type: application/json

{
  "image": "<base64_encoded_image>",
  "models": ["internal", "mobilenet_v2", "vit_crop", "plantnet"],
  "include_confidence": true
}
```

### Comparison Response

```json
{
  "comparison": {
    "internal": {
      "model_name": "Plant Classifier v0.1.0",
      "disease": "Early Blight",
      "confidence": 0.92,
      "processing_time_ms": 145
    },
    "mobilenet_v2": {
      "model_name": "MobileNetV2 Plant Disease",
      "disease": "Tomato___Early_blight",
      "confidence": 0.89,
      "processing_time_ms": 78
    },
    "vit_crop": {
      "model_name": "ViT Crop Diseases",
      "disease": "Not supported (tomato)",
      "confidence": null,
      "processing_time_ms": 0
    },
    "plantnet": {
      "model_name": "Pl@ntNet",
      "species": "Solanum lycopersicum",
      "confidence": 0.94,
      "processing_time_ms": 523
    }
  },
  "agreement_score": 0.67,
  "recommendation": "High confidence - models agree on disease identification"
}
```

### Enabling External Models

```bash
# Set environment variables for API keys (optional - for PlantNet)
export PLANTNET_API_KEY=your_plantnet_api_key
```

### Verifying Real Models are Working

**Check model versions in API response:**
```bash
curl http://localhost:8000/api/v1/health/ready | jq
```

Expected output with real models:
```json
{
  "status": "healthy",
  "components": {
    "species_classifier": {
      "version": "1.0.0-mobilenet_v2"  // Real model
    },
    "disease_detector": {
      "version": "1.0.0-mobilenet_v2"  // Real model
    }
  }
}
```

If you see `0.1.0-placeholder`, install torch and transformers:
```bash
pip install torch transformers
```

**Run determinism tests:**
```bash
pytest tests/test_real_inference.py -v
```

This verifies:
- Same image produces identical predictions every time
- Model version indicates real inference
- Confidence scores are in valid range

### Model Integration Details

#### 1. MobileNetV2 Plant Disease (HuggingFace)
```python
from transformers import pipeline

classifier = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
)
results = classifier("plant_image.jpg")
```

#### 2. ViT Crop Diseases (HuggingFace)
```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
model = ViTForImageClassification.from_pretrained('wambugu71/crop_leaf_diseases_vit')

image = Image.open('plant_image.jpg')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
predicted_class = model.config.id2label[outputs.logits.argmax(-1).item()]
```

#### 3. Pl@ntNet API
```python
import requests

response = requests.get(
    "https://my-api.plantnet.org/v2/identify/all",
    params={
        "images": ["https://example.com/plant.jpg"],
        "organs": ["leaf"],
        "api-key": PLANTNET_API_KEY
    }
)
results = response.json()
```

---

## AI Crop Disease Early Warning System

A comprehensive disease analysis system that runs **ALL available models in parallel** to provide farmers with the most accurate diagnosis and actionable treatment recommendations.

### Features

| Feature | Description |
|---------|-------------|
| **Parallel Model Execution** | Runs 6+ models simultaneously for maximum accuracy |
| **Consensus Algorithm** | Confidence-weighted voting across all models |
| **Severity Scoring** | 0-100 score based on disease type, confidence, and model agreement |
| **Model Explanations** | Detailed reasoning for each model's prediction |
| **Localized Treatments** | Region-specific recommendations with regulatory notes |

### Severity Levels

| Level | Score | Action Timeline |
|-------|-------|-----------------|
| **CRITICAL** | 80-100 | Immediate action within 24-48 hours |
| **HIGH** | 60-79 | Action needed within 3-5 days |
| **MODERATE** | 40-59 | Address within 1-2 weeks |
| **LOW** | 1-39 | Monitor and treat preventively |
| **HEALTHY** | 0 | Continue regular care |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Upload Crop Image                             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Run ALL Models in Parallel                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │MobileNet │ │   ViT    │ │ ResNet50 │ │EfficientN│           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│  ┌────┴─────┐ ┌────┴─────┐                                      │
│  │ PlantNet │ │ Kindwise │ (if API keys configured)             │
│  └────┬─────┘ └────┬─────┘                                      │
└───────┼────────────┼────────────────────────────────────────────┘
        │            │
        ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│           Consensus Algorithm (Confidence-Weighted)              │
│                                                                  │
│  Disease Votes:                                                  │
│  - Early Blight: 3 models (85% avg confidence) ← WINNER         │
│  - Late Blight: 1 model (62% confidence)                        │
│  - Healthy: 1 model (45% confidence)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Severity Calculation                             │
│                                                                  │
│  Base Severity (from disease DB):     60                        │
│  + High confidence modifier:          +10                       │
│  + Strong model agreement:            +10                       │
│  ─────────────────────────────────────────                      │
│  Final Severity Score:                80 (CRITICAL)             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Localized Treatment Generation                      │
│                                                                  │
│  1. Lookup disease in TREATMENT_DATABASE                        │
│  2. Adjust recommendations based on severity                    │
│  3. Add REGIONAL_NOTES for user's location                      │
│  4. Include monitoring schedule & recovery estimate             │
└─────────────────────────────────────────────────────────────────┘
```

### Treatment Recommendation System

The treatment system uses a 3-layer architecture:

#### Layer 1: Disease Severity Database

```python
# app/services/early_warning_service.py
DISEASE_SEVERITY_DATABASE = {
    "late blight": {
        "base_severity": 90,
        "spread_rate": "rapid",
        "crop_impact": "Can destroy entire crop within 7-10 days",
        "urgency": "Immediate action required"
    },
    "early blight": {
        "base_severity": 60,
        "spread_rate": "moderate",
        "crop_impact": "Can cause 20-50% yield loss",
        "urgency": "Act within 3-5 days"
    },
    # ... 25+ diseases
}
```

#### Layer 2: Localized Treatment Database

```python
LOCALIZED_TREATMENT_DATABASE = {
    "early blight": {
        "immediate_actions": [
            "Remove and destroy infected leaves immediately",
            "Increase plant spacing for better air circulation",
            "Avoid overhead watering - use drip irrigation"
        ],
        "organic_treatments": [
            "Copper-based fungicide (Bordeaux mixture) - apply every 7-10 days",
            "Neem oil spray (2-3 tablespoons per gallon) weekly"
        ],
        "chemical_treatments": [
            "Chlorothalonil (Daconil) - apply at first sign",
            "Mancozeb - effective preventive"
        ],
        "prevention_measures": [
            "Rotate crops - don't plant tomatoes in same spot for 3 years",
            "Use disease-resistant varieties"
        ],
        "monitoring_schedule": "Inspect every 2-3 days during humid weather",
        "estimated_recovery": "2-4 weeks with proper treatment"
    },
    # ... treatments for all diseases
}
```

#### Layer 3: Regional Notes

```python
REGIONAL_NOTES = {
    "US-CA": "California: Check local restrictions on copper applications.",
    "EU": "European Union: Many conventional pesticides restricted. Focus on IPM.",
    "IN-MH": "Maharashtra: Monsoon season increases fungal disease risk.",
    "AU": "Australia: Strict biosecurity - report unusual diseases."
}
```

### Using the Early Warning System

#### Via Web UI

1. **Start the server**: `uvicorn app.main:app --reload --port 8000`
2. **Open**: http://localhost:8000
3. **Click "Early Warning" tab**
4. **Configure API keys** (optional - enables more models)
5. **Upload crop image**
6. **Click "Analyze Plant"**
7. **Review results**:
   - Severity banner with score and urgency
   - Consensus disease identification
   - Each model's prediction with explanation
   - Tabbed treatment recommendations
   - Severity factors breakdown

#### Via API

```http
POST /api/v1/classify/early-warning
Content-Type: application/json

{
  "image": "<base64_encoded_image>",
  "region": "US-CA",
  "plantnet_api_key": "optional_key",
  "kindwise_api_key": "optional_key"
}
```

#### Response Format

```json
{
  "model_predictions": [
    {
      "model_name": "MobileNetV2 Plant Disease (HF)",
      "model_type": "mobilenet_v2",
      "prediction": "Early Blight",
      "confidence": 0.89,
      "explanation": "This model identified 'Early Blight' with high confidence (89.0%). The prediction is based on visual pattern recognition trained on similar disease presentations.",
      "contributing_factors": [
        "Strong visual pattern match",
        "Identified crop: Tomato"
      ]
    },
    // ... more models
  ],
  "consensus": {
    "disease_name": "Early Blight",
    "confidence": 0.87,
    "model_agreement": 0.83,
    "supporting_models": ["MobileNetV2", "ResNet50", "EfficientNet"],
    "dissenting_models": ["ViT Crop"],
    "is_healthy": false,
    "reasoning": "Strong consensus: 3 out of 4 models identified Early Blight. High confidence in diagnosis."
  },
  "severity": {
    "level": "high",
    "score": 75,
    "factors": [
      "Base severity for Early Blight: 60/100",
      "High detection confidence (+10)",
      "Strong model agreement (+10)",
      "Spread rate: moderate",
      "Potential impact: Can cause 20-50% yield loss"
    ],
    "urgency": "HIGH: Take action within 3-5 days",
    "action_timeline": "Act within 3-5 days"
  },
  "treatment": {
    "immediate_actions": [
      "Remove and destroy infected leaves immediately",
      "Increase plant spacing for better air circulation"
    ],
    "organic_treatments": [
      "Copper-based fungicide (Bordeaux mixture) - apply every 7-10 days",
      "Neem oil spray weekly"
    ],
    "chemical_treatments": [
      "Chlorothalonil (Daconil) - apply at first sign"
    ],
    "prevention_measures": [
      "Rotate crops - 3 year cycle",
      "Use disease-resistant varieties"
    ],
    "monitoring_schedule": "Inspect every 2-3 days during humid weather",
    "estimated_recovery": "2-4 weeks with proper treatment",
    "regional_notes": "California: Check local restrictions on copper applications."
  },
  "metadata": {
    "total_processing_time_ms": 2453,
    "models_consulted": 6,
    "region": "US-CA"
  }
}
```

### Supported Diseases

The system includes severity data and treatments for 25+ diseases:

| Crop | Diseases |
|------|----------|
| **Tomato** | Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus |
| **Potato** | Early Blight, Late Blight |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust |
| **Corn** | Common Rust, Gray Leaf Spot, Northern Leaf Blight |
| **Grape** | Black Rot, Esca, Leaf Blight |
| **Rice** | Brown Spot, Leaf Blast |
| **Wheat** | Brown Rust, Yellow Rust |

### Adding New Diseases

1. **Add to Severity Database** (`app/services/early_warning_service.py`):
```python
DISEASE_SEVERITY_DATABASE["new_disease"] = {
    "base_severity": 65,
    "spread_rate": "moderate",
    "crop_impact": "Description of impact",
    "urgency": "Action timeline"
}
```

2. **Add Treatment Plan**:
```python
LOCALIZED_TREATMENT_DATABASE["new_disease"] = {
    "immediate_actions": [...],
    "organic_treatments": [...],
    "chemical_treatments": [...],
    "prevention_measures": [...],
    "monitoring_schedule": "...",
    "estimated_recovery": "..."
}
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
| `PLANTNET_API_KEY` | `None` | PlantNet API key for external model |
| `KINDWISE_API_KEY` | `None` | Kindwise/Plant.id API key for external model |

---

## Troubleshooting

### Models showing "placeholder" version

**Problem**: API returns `model_version: "0.1.0-placeholder"`

**Solution**: Install PyTorch and Transformers:
```bash
pip install torch transformers
```

Then restart the server. First run will download models from HuggingFace (~100-200MB).

### Slow first request

**Cause**: Models are lazily loaded on first inference.

**Solution**: This is expected. Subsequent requests will be fast (~100-500ms).

### "transformers not installed" warning

**Cause**: The `transformers` package is not installed.

**Solution**:
```bash
pip install transformers>=4.36.0
```

### Compare Models not showing results

**Check**:
1. At least one model is selected
2. Image is uploaded
3. Server logs for errors (`uvicorn app.main:app --reload`)

### PlantNet returns error

**Cause**: Missing or invalid API key.

**Solution**:
```bash
export PLANTNET_API_KEY=your_api_key_here
```

Get a free API key at https://my.plantnet.org/

---

## Complete Setup Commands

```bash
# 1. Clone and setup
git clone <repository-url>
cd plant-backend-image-classifier
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install all dependencies (including ML libraries)
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --reload --port 8000

# 4. Open in browser
# Web UI: http://localhost:8000
# API Docs: http://localhost:8000/docs

# 5. Test the API
curl -X POST http://localhost:8000/api/v1/classify/compare \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_image>", "models": ["internal", "mobilenet_v2"]}'

# 6. Run tests
pytest tests/ -v
pytest tests/test_real_inference.py -v  # Real model tests
```

## License

MIT License - See LICENSE file for details.
