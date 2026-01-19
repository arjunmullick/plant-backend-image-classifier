"""
Classification API endpoints.

Main endpoints for plant image classification including:
- Full classification with species, disease, and treatment
- Species-only classification
- Disease-only detection
- Batch classification
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from app.models.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ErrorResponse,
)
from app.services.classification_service import (
    ClassificationService,
    get_classification_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["Classification"])


# === Request/Response Models for specific endpoints ===

class SpeciesOnlyResponse(BaseModel):
    """Response for species-only classification."""
    family: dict
    genus: dict
    species: dict
    common_name: Optional[str] = None
    alternatives: Optional[list[dict]] = None
    metadata: Optional[dict] = None


class DiseaseOnlyRequest(BaseModel):
    """Request for disease-only detection."""
    image: str
    crop: Optional[str] = None  # Optional crop hint for better accuracy


class DiseaseOnlyResponse(BaseModel):
    """Response for disease-only detection."""
    status: str
    disease: Optional[str] = None
    confidence: float
    visual_symptoms: list[str]
    metadata: Optional[dict] = None


class BatchClassificationRequest(BaseModel):
    """Request for batch classification."""
    images: list[str]  # List of base64-encoded images
    region: Optional[str] = None
    include_treatment: bool = True


class BatchClassificationResponse(BaseModel):
    """Response for batch classification."""
    results: list[ClassificationResponse]
    summary: dict


class FeedbackRequest(BaseModel):
    """Request to submit prediction feedback."""
    image_id: str
    predicted_species: str
    predicted_disease: Optional[str] = None
    correct_species: Optional[str] = None
    correct_disease: Optional[str] = None
    feedback_type: str = "correction"  # correction, confirmation, uncertain
    notes: Optional[str] = None


# === Main Endpoints ===

@router.post(
    "",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Classify plant image",
    description="""
    Complete plant classification pipeline.

    Takes a plant image and returns:
    - Hierarchical species identification (Family → Genus → Species)
    - Disease detection with visual symptoms
    - Treatment recommendations (organic, chemical, preventive)
    - Model explainability information

    **Image Requirements:**
    - Base64-encoded JPEG or PNG
    - Recommended resolution: 224x224 or higher
    - Clear view of leaves or affected areas

    **Region Codes:**
    - US-CA: California, USA
    - EU: European Union
    - IN-MH: Maharashtra, India
    - (Custom regions can be configured)
    """
)
async def classify_image(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
) -> ClassificationResponse:
    """
    Classify a plant image.

    This is the main endpoint that runs the complete classification pipeline.
    """
    try:
        logger.info(f"Received classification request (region={request.region})")

        response = service.classify(
            image_base64=request.image,
            region=request.region,
            include_treatment=request.include_treatment,
            include_explainability=request.include_explainability
        )

        logger.info(
            f"Classification complete: {response.plant.species.name} "
            f"({response.plant.species.confidence:.2%}), "
            f"Health: {response.health.status}"
        )

        return response

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post(
    "/species",
    response_model=SpeciesOnlyResponse,
    summary="Species identification only",
    description="Identify plant species without disease detection or treatment recommendations."
)
async def classify_species_only(
    request: ClassificationRequest,
    service: ClassificationService = Depends(get_classification_service)
) -> SpeciesOnlyResponse:
    """
    Species classification only.

    Faster endpoint when only species identification is needed.
    """
    try:
        # Run classification without treatment/disease
        preprocessed = service.preprocessor.preprocess_from_base64(request.image)
        species_result = service.species_classifier.predict(preprocessed.tensor)
        pred = species_result.prediction

        return SpeciesOnlyResponse(
            family={"name": pred.family, "confidence": pred.family_confidence},
            genus={"name": pred.genus, "confidence": pred.genus_confidence},
            species={"name": pred.species, "confidence": pred.species_confidence},
            common_name=pred.common_name,
            alternatives=[
                {"name": alt[0], "confidence": alt[1]}
                for alt in pred.alternatives[:3]
            ] if pred.alternatives else None,
            metadata={
                "processing_time_ms": species_result.latency_ms,
                "model_version": species_result.model_version
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Species classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/disease",
    response_model=DiseaseOnlyResponse,
    summary="Disease detection only",
    description="Detect plant disease without full species classification."
)
async def detect_disease_only(
    request: DiseaseOnlyRequest,
    service: ClassificationService = Depends(get_classification_service)
) -> DiseaseOnlyResponse:
    """
    Disease detection only.

    Can optionally specify crop type for better accuracy
    using crop-specific models.
    """
    try:
        preprocessed = service.preprocessor.preprocess_from_base64(request.image)
        disease_result = service.disease_detector.predict(
            preprocessed.tensor,
            crop=request.crop
        )
        pred = disease_result.prediction

        return DiseaseOnlyResponse(
            status="Healthy" if pred.is_healthy else "Diseased",
            disease=pred.disease_name,
            confidence=pred.confidence,
            visual_symptoms=pred.visual_symptoms,
            metadata={
                "processing_time_ms": disease_result.latency_ms,
                "model_version": disease_result.model_version,
                "crop_model_used": request.crop
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Disease detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/batch",
    response_model=BatchClassificationResponse,
    summary="Batch classification",
    description="Classify multiple images in a single request. Limited to 10 images."
)
async def classify_batch(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    service: ClassificationService = Depends(get_classification_service)
) -> BatchClassificationResponse:
    """
    Batch classification for multiple images.

    Processes up to 10 images in a single request.
    """
    if len(request.images) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request"
        )

    if len(request.images) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required"
        )

    results = []
    errors = 0
    diseases_found = []

    for i, image in enumerate(request.images):
        try:
            result = service.classify(
                image_base64=image,
                region=request.region,
                include_treatment=request.include_treatment,
                include_explainability=False  # Skip for batch efficiency
            )
            results.append(result)

            if result.health.disease:
                diseases_found.append(result.health.disease)

        except Exception as e:
            logger.warning(f"Batch item {i} failed: {e}")
            errors += 1
            # Append a minimal error result
            results.append(ClassificationResponse(
                plant={
                    "family": {"name": "Unknown", "confidence": 0.0},
                    "genus": {"name": "Unknown", "confidence": 0.0},
                    "species": {"name": "Unknown", "confidence": 0.0}
                },
                health={
                    "status": "Unknown",
                    "confidence": 0.0,
                    "visual_symptoms": []
                },
                metadata={"error": str(e)}
            ))

    # Build summary
    summary = {
        "total_images": len(request.images),
        "successful": len(request.images) - errors,
        "failed": errors,
        "diseases_detected": list(set(diseases_found)),
        "healthy_count": sum(1 for r in results if r.health.status == "Healthy"),
        "diseased_count": sum(1 for r in results if r.health.status == "Diseased")
    }

    return BatchClassificationResponse(
        results=results,
        summary=summary
    )


@router.post(
    "/feedback",
    summary="Submit prediction feedback",
    description="Submit feedback on prediction accuracy for model improvement."
)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks
) -> dict:
    """
    Submit feedback on a prediction.

    This enables human-in-the-loop model improvement.
    Feedback is stored for later analysis and model retraining.
    """
    from app.ml.explainability import HumanInTheLoopFeedback

    feedback_system = HumanInTheLoopFeedback()

    feedback = feedback_system.record_feedback(
        image_id=request.image_id,
        predicted_species=request.predicted_species,
        predicted_disease=request.predicted_disease,
        correct_species=request.correct_species,
        correct_disease=request.correct_disease,
        expert_notes=request.notes,
        feedback_type=request.feedback_type
    )

    return {
        "status": "recorded",
        "feedback_id": request.image_id,
        "message": "Thank you for your feedback. It will help improve our models."
    }


@router.get(
    "/supported-crops",
    summary="List supported crops",
    description="Get list of crops with specialized disease detection models."
)
async def get_supported_crops(
    service: ClassificationService = Depends(get_classification_service)
) -> dict:
    """Get list of crops with specialized models."""
    return {
        "crops": service.get_supported_crops(),
        "note": "Specifying crop in /disease endpoint improves accuracy"
    }


@router.get(
    "/supported-diseases",
    summary="List supported diseases",
    description="Get list of diseases with treatment recommendations."
)
async def get_supported_diseases(
    service: ClassificationService = Depends(get_classification_service)
) -> dict:
    """Get list of diseases with treatment plans."""
    return {
        "diseases": service.get_supported_diseases(),
        "total": len(service.get_supported_diseases())
    }
