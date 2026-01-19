"""
Plant Image Classification API

FastAPI application for plant species identification, disease detection,
and treatment recommendations.

This is the main entry point for the application.

Usage:
    uvicorn app.main:app --reload
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Production:
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi

from app.core.config import get_settings
from app.api.routes import classify_router, health_router
from app.api.routes.health import set_startup_time
from app.services.classification_service import get_classification_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup:
    - Load ML models
    - Warm up inference

    Runs on shutdown:
    - Clean up resources
    """
    logger.info("Starting Plant Image Classification API...")

    # Record startup time
    set_startup_time()

    # Initialize and warm up the classification service
    try:
        service = get_classification_service()
        warmup_times = service.warmup()
        logger.info(f"Model warmup complete: {warmup_times}")
    except Exception as e:
        logger.error(f"Failed to initialize classification service: {e}")
        # Continue startup - models will load lazily

    logger.info("Application startup complete")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Plant Image Classification API...")


# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Plant Image Classification API

A production-grade backend service for plant species identification,
disease detection, and treatment recommendations.

### Features

- **Hierarchical Plant Identification**: Family → Genus → Species with confidence scores
- **Disease Detection**: Identify plant diseases with visual symptom descriptions
- **Treatment Recommendations**: Organic, chemical, and preventive treatment options
- **Explainability**: Understand why the model made its predictions
- **Region-Specific Filtering**: Treatment recommendations filtered by regional regulations

### Getting Started

1. Send a POST request to `/api/v1/classify` with a base64-encoded plant image
2. Receive species identification, disease detection, and treatment recommendations
3. Use the explainability information to understand and verify predictions

### API Endpoints

- `POST /api/v1/classify` - Full classification pipeline
- `POST /api/v1/classify/species` - Species identification only
- `POST /api/v1/classify/disease` - Disease detection only
- `POST /api/v1/classify/batch` - Batch classification (up to 10 images)
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/ready` - Detailed readiness check

### Image Requirements

- Format: JPEG or PNG (base64-encoded)
- Recommended resolution: 224x224 or higher
- Clear view of leaves or affected areas
- Good lighting conditions

### Model Information

- Species Classification: EfficientNetV2-S with hierarchical heads
- Disease Detection: Crop-specific models with 38+ disease classes
- Trained on PlantVillage, PlantNet, and iNaturalist data
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
            "details": str(exc) if settings.debug else None
        }
    )


# Include routers
app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(classify_router, prefix=settings.api_prefix)


# Static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"


# Root endpoint - serve the frontend
@app.get("/", tags=["Root"], response_class=FileResponse)
async def root():
    """Serve the frontend web application."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    # Fallback to API info if no frontend
    return JSONResponse({
        "name": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "health_check": f"{settings.api_prefix}/health",
        "classification_endpoint": f"{settings.api_prefix}/classify"
    })


# API info endpoint
@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "health_check": f"{settings.api_prefix}/health",
        "classification_endpoint": f"{settings.api_prefix}/classify"
    }


# Mount static files (must be after specific routes)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom tags
    openapi_schema["tags"] = [
        {
            "name": "Classification",
            "description": "Plant image classification endpoints"
        },
        {
            "name": "Health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "Root",
            "description": "API root and information"
        }
    ]

    # Add example request/response
    openapi_schema["info"]["x-example-request"] = {
        "image": "<base64_encoded_image>",
        "region": "US-CA"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Entry point for running with Python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )
