"""
Health check and system status endpoints.

Provides endpoints for:
- Basic health check
- Detailed system status
- Model readiness checks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check with component status."""
    status: str
    timestamp: float
    version: str
    components: dict[str, dict]
    uptime_seconds: Optional[float] = None


# Track startup time
_startup_time: Optional[float] = None


def set_startup_time() -> None:
    """Set the startup time (called on app startup)."""
    global _startup_time
    _startup_time = time.time()


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.

    Returns:
        Simple health status indicating the service is running.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="0.1.0"
    )


@router.get("/ready", response_model=DetailedHealthResponse)
async def readiness_check() -> DetailedHealthResponse:
    """
    Detailed readiness check including ML models.

    Verifies:
    - Species classifier is loaded
    - Disease detector is loaded
    - All dependencies are available

    Returns:
        Detailed status of all components.
    """
    from app.services.classification_service import get_classification_service

    components = {}
    overall_healthy = True

    try:
        service = get_classification_service()

        # Check species classifier
        try:
            species_info = service.species_classifier.get_model_info()
            components["species_classifier"] = {
                "status": "ready",
                "version": species_info.version,
                "architecture": species_info.architecture,
                "num_classes": species_info.num_classes
            }
        except Exception as e:
            components["species_classifier"] = {
                "status": "error",
                "error": str(e)
            }
            overall_healthy = False

        # Check disease detector
        try:
            disease_info = service.disease_detector.get_model_info()
            components["disease_detector"] = {
                "status": "ready",
                "version": disease_info.version,
                "num_classes": disease_info.num_classes,
                "crop_specific_models": service.disease_detector.get_crop_specific_models()
            }
        except Exception as e:
            components["disease_detector"] = {
                "status": "error",
                "error": str(e)
            }
            overall_healthy = False

        # Check treatment service
        components["treatment_service"] = {
            "status": "ready",
            "supported_diseases": service.get_supported_diseases()
        }

    except Exception as e:
        components["service"] = {
            "status": "error",
            "error": str(e)
        }
        overall_healthy = False

    uptime = None
    if _startup_time:
        uptime = time.time() - _startup_time

    if not overall_healthy:
        raise HTTPException(status_code=503, detail="Service not ready")

    return DetailedHealthResponse(
        status="ready" if overall_healthy else "degraded",
        timestamp=time.time(),
        version="0.1.0",
        components=components,
        uptime_seconds=uptime
    )


@router.get("/live")
async def liveness_check() -> dict:
    """
    Simple liveness probe for Kubernetes.

    Returns 200 if the process is running.
    """
    return {"status": "alive"}
