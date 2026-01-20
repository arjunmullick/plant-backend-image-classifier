# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plant image classification backend built with FastAPI. Provides plant species identification, disease detection, and treatment recommendations using computer vision and transfer learning.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --port 8000

# Run production server
gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=app --cov-report=html

# Lint and format
black app/ tests/
isort app/ tests/
ruff check app/ tests/
mypy app/
```

### Model Training (GPU-free via Modal Labs)

```bash
# Setup Modal
pip install modal
modal setup

# Create secrets
modal secret create huggingface-secret HUGGING_FACE_HUB_TOKEN=hf_xxxxx
modal secret create wandb-secret WANDB_API_KEY=xxxxx

# Download dataset
python scripts/download_datasets.py --dataset plantvillage --output data/raw

# Upload to Hugging Face Hub
python scripts/upload_to_hub.py --input data/raw/plantvillage --repo your-username/plantvillage

# Train on Modal (GPU in cloud)
modal run training/modal_train.py --dataset-path your-username/plantvillage --epochs 15
```

## Architecture

```
FastAPI Application
    └── POST /api/v1/classify
            │
            ▼
    Classification Service (Orchestrator)
            │
    ┌───────┼───────┬───────────────┐
    │       │       │               │
    ▼       ▼       ▼               ▼
  Image   Species  Disease    Explainability
Preprocess Classifier Detector    Engine
    │       │         │            │
    ▼       ▼         ▼            ▼
 Quality  Hierarchical Crop-specific Grad-CAM
Assessment Taxonomy   Routing     + Reasoning
                                    │
                            Treatment
                           Recommendation
```

### Key Directories

- `app/ml/` - ML components: preprocessor, species classifier, disease detector, explainability (Grad-CAM)
- `app/services/` - Business logic: classification orchestration, treatment recommendations
- `app/api/routes/` - API endpoints for classification and health checks
- `app/core/` - Configuration and dependencies
- `app/models/` - Pydantic schemas and enums
- `training/` - Modal Labs GPU training scripts
- `scripts/` - Dataset download and Hugging Face Hub upload utilities
- `static/` - Web frontend for testing

### Configuration

Environment variables are prefixed with `PLANT_CLASSIFIER_`:
- `DEBUG` - Enable debug mode
- `MODEL_CACHE_DIR` - Model weights directory
- `SPECIES_MODEL_PATH` / `DISEASE_MODEL_PATH` - Model checkpoints
- `ENABLE_GRAD_CAM` - Enable Grad-CAM visualizations
- `ENABLE_REGION_FILTERING` - Filter treatments by region

### ML Pipeline Flow

1. **Preprocessing** (`app/ml/preprocessor.py`) - Image quality assessment, normalization
2. **Species Classification** (`app/ml/species_classifier.py`) - Hierarchical taxonomy: Family → Genus → Species
3. **Disease Detection** (`app/ml/disease_detector.py`) - Crop-specific model routing
4. **Explainability** (`app/ml/explainability.py`) - Grad-CAM heatmaps, human-readable reasoning
5. **Treatment** (`app/services/treatment_service.py`) - Organic/chemical treatments, region filtering

### Adding New Crops/Diseases

1. Add taxonomy to `TaxonomyLabel` in `app/ml/species_classifier.py`
2. Add disease labels to `DiseaseLabel` in `app/ml/disease_detector.py`
3. Add treatment plan to `TreatmentDatabase.TREATMENTS` in `app/services/treatment_service.py`
4. Train/fine-tune model on new data

### API Endpoints

- `POST /api/v1/classify` - Full classification pipeline
- `POST /api/v1/classify/species` - Species identification only
- `POST /api/v1/classify/disease` - Disease detection only
- `POST /api/v1/classify/batch` - Batch classification (up to 10)
- `POST /api/v1/classify/compare` - Compare with external models
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Detailed readiness check

### Web UI

Access at `http://localhost:8000` after starting the server. Supports drag-and-drop image upload, mode selection (Full/Species/Disease/Batch), and displays results with treatment recommendations.
