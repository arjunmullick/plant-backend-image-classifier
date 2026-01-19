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
    ComparisonRequest,
    ComparisonResponse,
    ExternalModelPrediction,
    AvailableModelsResponse,
)
from app.services.classification_service import (
    ClassificationService,
    get_classification_service,
)
from app.ml.external_models import (
    get_external_model_registry,
    ExternalModelType,
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


# === Model Comparison Endpoints ===

@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare predictions across models",
    description="""
    Compare predictions from our internal model with open-source models.

    **Available Models:**
    - `internal`: Our trained model with full pipeline
    - `mobilenet_v2`: MobileNetV2 Plant Disease (HuggingFace) - 38 classes, ~95% accuracy
    - `vit_crop`: ViT Crop Diseases (HuggingFace) - 14 classes, ~98% accuracy
    - `plantnet`: Pl@ntNet API - 50,000+ species (requires API key)

    **Use Cases:**
    - Validate predictions across multiple models
    - Understand model agreement/disagreement
    - Research and comparison purposes
    """
)
async def compare_models(
    request: ComparisonRequest,
    service: ClassificationService = Depends(get_classification_service)
) -> ComparisonResponse:
    """
    Compare predictions from multiple models.

    Runs the same image through selected models and returns
    side-by-side comparison of results.
    """
    import base64
    from io import BytesIO
    from PIL import Image
    import time

    try:
        start_time = time.time()

        # Decode image
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Results containers
        internal_result = None
        external_results = {}

        # Run internal model if requested
        if "internal" in request.models:
            try:
                internal_response = service.classify(
                    image_base64=request.image,
                    region=None,
                    include_treatment=False,
                    include_explainability=False
                )
                internal_result = {
                    "species": internal_response.plant.species.name,
                    "species_confidence": internal_response.plant.species.confidence,
                    "common_name": internal_response.plant.common_name,
                    "disease": internal_response.health.disease,
                    "health_status": internal_response.health.status,
                    "disease_confidence": internal_response.health.confidence,
                    "processing_time_ms": internal_response.metadata.get("processing_time_ms", 0)
                        if internal_response.metadata else 0
                }
            except Exception as e:
                logger.error(f"Internal model error: {e}")
                internal_result = {"error": str(e)}

        # Map model names to types
        model_type_map = {
            "mobilenet_v2": ExternalModelType.MOBILENET_V2,
            "vit_crop": ExternalModelType.VIT_CROP,
            "plantnet": ExternalModelType.PLANTNET,
        }

        # Run external models
        registry = get_external_model_registry()
        external_model_types = [
            model_type_map[m]
            for m in request.models
            if m in model_type_map
        ]

        if external_model_types:
            comparison_results = await registry.run_comparison(
                pil_image,
                model_types=external_model_types
            )

            for model_key, result in comparison_results.items():
                external_results[model_key] = ExternalModelPrediction(
                    model_name=result.model_name,
                    model_type=result.model_type.value,
                    prediction=result.prediction,
                    confidence=result.confidence,
                    raw_label=result.raw_label,
                    processing_time_ms=result.processing_time_ms,
                    error=result.error,
                    additional_info=result.additional_info
                )

        # Calculate agreement score
        agreement_score = _calculate_agreement_score(internal_result, external_results)

        # Generate recommendation
        recommendation = _generate_recommendation(agreement_score, internal_result, external_results)

        total_time = (time.time() - start_time) * 1000

        return ComparisonResponse(
            internal=internal_result,
            external_models=external_results,
            agreement_score=agreement_score,
            recommendation=recommendation,
            metadata={
                "total_processing_time_ms": total_time,
                "models_compared": len(request.models),
                "models_requested": request.models
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/compare/models",
    response_model=AvailableModelsResponse,
    summary="List available comparison models",
    description="Get list of available models for comparison with their status."
)
async def get_available_comparison_models(
    service: ClassificationService = Depends(get_classification_service)
) -> AvailableModelsResponse:
    """Get list of available models for comparison."""
    registry = get_external_model_registry()

    return AvailableModelsResponse(
        internal_model={
            "name": "Plant Classifier",
            "version": "0.1.0",
            "capabilities": ["species", "disease", "treatment", "explainability"],
            "is_loaded": True
        },
        external_models=registry.get_available_models()
    )


def _calculate_agreement_score(
    internal_result: Optional[dict],
    external_results: dict
) -> Optional[float]:
    """
    Calculate agreement score between models.

    Returns a score from 0 to 1 based on how many models agree.
    """
    if not internal_result or "error" in internal_result:
        return None

    internal_disease = internal_result.get("disease", "").lower() if internal_result.get("disease") else "healthy"

    agreements = 0
    total_comparisons = 0

    for model_key, result in external_results.items():
        if result.error:
            continue

        total_comparisons += 1
        external_prediction = (result.prediction or "").lower()

        # Check for disease agreement (fuzzy matching)
        if internal_disease == "healthy" and external_prediction == "healthy":
            agreements += 1
        elif internal_disease != "healthy" and external_prediction != "healthy":
            # Both detected disease - check if similar
            if _diseases_similar(internal_disease, external_prediction):
                agreements += 1
            else:
                agreements += 0.5  # Partial credit for both detecting disease

    if total_comparisons == 0:
        return None

    return agreements / total_comparisons


def _diseases_similar(disease1: str, disease2: str) -> bool:
    """Check if two disease names are similar."""
    # Normalize names
    d1 = disease1.lower().replace("_", " ").replace("-", " ")
    d2 = disease2.lower().replace("_", " ").replace("-", " ")

    # Direct match
    if d1 == d2:
        return True

    # Check for key disease terms
    disease_terms = [
        "blight", "rust", "spot", "mold", "rot", "virus", "mites",
        "scab", "mildew", "wilt", "canker"
    ]

    for term in disease_terms:
        if term in d1 and term in d2:
            return True

    return False


def _generate_recommendation(
    agreement_score: Optional[float],
    internal_result: Optional[dict],
    external_results: dict
) -> str:
    """Generate recommendation based on comparison results."""
    if agreement_score is None:
        return "Unable to calculate agreement - check model results for errors"

    if agreement_score >= 0.8:
        return "High confidence - models strongly agree on the diagnosis"
    elif agreement_score >= 0.5:
        return "Moderate confidence - some agreement between models. Consider expert verification for critical decisions."
    else:
        return "Low agreement between models - recommend expert verification before taking action"
