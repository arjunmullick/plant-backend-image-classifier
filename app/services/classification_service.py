"""
Classification Orchestration Service

Coordinates the complete plant classification pipeline:
1. Image preprocessing
2. Species classification (hierarchical)
3. Disease detection
4. Explainability generation
5. Treatment recommendations

This service acts as the main entry point for image classification,
managing the flow between ML components and aggregating results.

Design Principles:
- Components are independently replaceable
- Failures are gracefully handled with partial results
- All timing is tracked for performance monitoring
- Results include confidence and uncertainty information
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from app.ml.preprocessor import ImagePreprocessor, PreprocessedImage
from app.ml.species_classifier import SpeciesClassifier
from app.ml.disease_detector import DiseaseDetector
from app.ml.explainability import ExplainabilityEngine, ExplainabilityResult
from app.ml.base import HierarchicalPrediction, DiseasePrediction
from app.services.treatment_service import TreatmentRecommendationService
from app.models.schemas import (
    ClassificationResponse,
    PlantIdentification,
    TaxonomicLevel,
    HealthAssessment,
    TreatmentRecommendation,
    ExplainabilityInfo,
    GradCAMResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics from the classification pipeline."""
    total_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    species_classification_time_ms: float = 0.0
    disease_detection_time_ms: float = 0.0
    explainability_time_ms: float = 0.0
    treatment_time_ms: float = 0.0


@dataclass
class ClassificationResult:
    """Internal result from classification pipeline."""
    species_prediction: HierarchicalPrediction
    disease_prediction: DiseasePrediction
    explainability: Optional[ExplainabilityResult] = None
    treatment: Optional[dict] = None
    preprocessed_image: Optional[PreprocessedImage] = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ClassificationService:
    """
    Main orchestration service for plant classification.

    Manages the complete ML pipeline from image input to
    actionable results with explanations.

    Usage:
        service = ClassificationService()
        response = service.classify(
            image_base64="...",
            region="US-CA",
            include_treatment=True,
            include_explainability=True
        )

    Pipeline Flow:
    ```
    Image Input
         |
    ┌────▼────┐
    │Preprocess│ → Quality assessment
    └────┬────┘
         |
    ┌────▼────┐
    │ Species │ → Family, Genus, Species with confidence
    │Classifier│
    └────┬────┘
         |
    ┌────▼────┐
    │ Disease │ → Healthy/Diseased, disease name, symptoms
    │Detector │ (uses species to select crop-specific model)
    └────┬────┘
         |
    ┌────▼────┐
    │Explain- │ → Grad-CAM, reasoning, uncertainty
    │ability  │
    └────┬────┘
         |
    ┌────▼────┐
    │Treatment│ → Organic, chemical, preventive options
    │Recommend│ (filtered by region if specified)
    └────┬────┘
         |
    Response
    ```
    """

    def __init__(
        self,
        preprocessor: Optional[ImagePreprocessor] = None,
        species_classifier: Optional[SpeciesClassifier] = None,
        disease_detector: Optional[DiseaseDetector] = None,
        explainability_engine: Optional[ExplainabilityEngine] = None,
        treatment_service: Optional[TreatmentRecommendationService] = None,
    ):
        """
        Initialize classification service with components.

        Components can be injected for testing or replaced with
        alternative implementations.
        """
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.species_classifier = species_classifier or SpeciesClassifier()
        self.disease_detector = disease_detector or DiseaseDetector()
        self.explainability_engine = explainability_engine or ExplainabilityEngine()
        self.treatment_service = treatment_service or TreatmentRecommendationService()

        # Ensure models are loaded
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Load all ML models."""
        logger.info("Initializing classification service models...")
        self.species_classifier.ensure_loaded()
        self.disease_detector.ensure_loaded()
        logger.info("All models loaded successfully")

    def classify(
        self,
        image_base64: str,
        region: Optional[str] = None,
        include_treatment: bool = True,
        include_explainability: bool = True
    ) -> ClassificationResponse:
        """
        Run complete classification pipeline.

        Args:
            image_base64: Base64-encoded image
            region: Optional region code for treatment filtering
            include_treatment: Include treatment recommendations
            include_explainability: Include model explainability

        Returns:
            ClassificationResponse with all results
        """
        pipeline_start = time.perf_counter()
        metrics = PipelineMetrics()
        warnings = []

        # Step 1: Preprocess image
        logger.debug("Preprocessing image...")
        preprocess_start = time.perf_counter()

        try:
            preprocessed = self.preprocessor.preprocess_from_base64(
                image_base64, return_pil=True
            )
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")

        metrics.preprocessing_time_ms = (time.perf_counter() - preprocess_start) * 1000

        # Check image quality
        if preprocessed.quality_report and not preprocessed.quality_report.is_usable:
            warnings.append(
                f"Image quality is poor ({preprocessed.quality_report.overall_score:.2f}). "
                f"Issues: {', '.join(preprocessed.quality_report.issues)}"
            )

        # Step 2: Species classification
        logger.debug("Running species classification...")
        species_start = time.perf_counter()

        species_result = self.species_classifier.predict(preprocessed.tensor)
        species_pred = species_result.prediction

        metrics.species_classification_time_ms = (time.perf_counter() - species_start) * 1000

        # Step 3: Disease detection (using species for routing)
        logger.debug("Running disease detection...")
        disease_start = time.perf_counter()

        # Extract crop name from species for model routing
        crop_name = self._extract_crop_name(species_pred.common_name or species_pred.species)

        disease_result = self.disease_detector.predict(
            preprocessed.tensor,
            crop=crop_name
        )
        disease_pred = disease_result.prediction

        metrics.disease_detection_time_ms = (time.perf_counter() - disease_start) * 1000

        # Step 4: Explainability (optional)
        explainability_result = None
        if include_explainability:
            logger.debug("Generating explanations...")
            explain_start = time.perf_counter()

            original_pil = preprocessed.metadata.get("pil_image")
            explainability_result = self.explainability_engine.explain(
                species_prediction=species_pred,
                disease_prediction=disease_pred,
                image_tensor=preprocessed.tensor,
                original_image=original_pil
            )

            metrics.explainability_time_ms = (time.perf_counter() - explain_start) * 1000

        # Step 5: Treatment recommendations (optional)
        treatment_dict = None
        if include_treatment and not disease_pred.is_healthy:
            logger.debug("Generating treatment recommendations...")
            treatment_start = time.perf_counter()

            treatment_dict = self.treatment_service.get_recommendations(
                disease=disease_pred.disease_name,
                region=region,
                severity=disease_pred.disease_stage
            )

            metrics.treatment_time_ms = (time.perf_counter() - treatment_start) * 1000

        # Calculate total time
        metrics.total_time_ms = (time.perf_counter() - pipeline_start) * 1000

        # Build response
        return self._build_response(
            species_pred=species_pred,
            disease_pred=disease_pred,
            explainability=explainability_result,
            treatment=treatment_dict,
            metrics=metrics,
            warnings=warnings
        )

    def _extract_crop_name(self, name: str) -> Optional[str]:
        """
        Extract crop name for model routing.

        Maps common names to crop identifiers used by
        crop-specific disease models.
        """
        if not name:
            return None

        name_lower = name.lower()

        crop_mapping = {
            "tomato": "tomato",
            "potato": "potato",
            "apple": "apple",
            "corn": "corn",
            "maize": "corn",
            "grape": "grape",
            "bell pepper": "pepper",
            "pepper": "pepper",
            "strawberry": "strawberry",
            "orange": "citrus",
            "lemon": "citrus",
            "cucumber": "cucumber",
            "squash": "squash",
            "wheat": "wheat",
            "rice": "rice",
            "soybean": "soybean",
        }

        for key, crop in crop_mapping.items():
            if key in name_lower:
                return crop

        return None

    def _build_response(
        self,
        species_pred: HierarchicalPrediction,
        disease_pred: DiseasePrediction,
        explainability: Optional[ExplainabilityResult],
        treatment: Optional[dict],
        metrics: PipelineMetrics,
        warnings: list[str]
    ) -> ClassificationResponse:
        """Build the API response from internal results."""

        # Build plant identification
        plant_id = PlantIdentification(
            family=TaxonomicLevel(
                name=species_pred.family,
                confidence=species_pred.family_confidence
            ),
            genus=TaxonomicLevel(
                name=species_pred.genus,
                confidence=species_pred.genus_confidence
            ),
            species=TaxonomicLevel(
                name=species_pred.species,
                confidence=species_pred.species_confidence
            ),
            common_name=species_pred.common_name,
            alternative_species=[
                TaxonomicLevel(name=alt[0], confidence=alt[1])
                for alt in species_pred.alternatives[:3]
            ] if species_pred.alternatives else None
        )

        # Build health assessment
        health = HealthAssessment(
            status="Healthy" if disease_pred.is_healthy else "Diseased",
            disease=disease_pred.disease_name,
            confidence=disease_pred.confidence,
            visual_symptoms=disease_pred.visual_symptoms,
            disease_stage=disease_pred.disease_stage,
            affected_area_percentage=disease_pred.affected_area
        )

        # Build treatment recommendation
        treatment_rec = None
        if treatment:
            treatment_rec = TreatmentRecommendation(
                organic=treatment.get("organic", []),
                chemical=treatment.get("chemical", []),
                prevention=treatment.get("prevention", []),
                biological=treatment.get("biological"),
                cultural=treatment.get("cultural"),
                urgency=treatment.get("urgency"),
                region_specific_notes=treatment.get("region_specific_notes")
            )

        # Build explainability info
        explain_info = None
        if explainability:
            grad_cam = None
            if explainability.grad_cam:
                grad_cam = GradCAMResult(
                    heatmap_base64=explainability.heatmap_base64,
                    focus_regions=explainability.grad_cam.focus_regions
                )

            explain_info = ExplainabilityInfo(
                model_reasoning=explainability.model_reasoning,
                confidence_notes=explainability.confidence_notes,
                key_features=explainability.key_features,
                uncertainty_factors=explainability.uncertainty_factors,
                grad_cam=grad_cam
            )

        # Build metadata
        metadata = {
            "processing_time_ms": round(metrics.total_time_ms, 2),
            "timing_breakdown": {
                "preprocessing": round(metrics.preprocessing_time_ms, 2),
                "species_classification": round(metrics.species_classification_time_ms, 2),
                "disease_detection": round(metrics.disease_detection_time_ms, 2),
                "explainability": round(metrics.explainability_time_ms, 2),
                "treatment": round(metrics.treatment_time_ms, 2),
            },
            "model_versions": {
                "species_classifier": self.species_classifier.get_model_info().version,
                "disease_detector": self.disease_detector.get_model_info().version,
            }
        }

        if warnings:
            metadata["warnings"] = warnings

        return ClassificationResponse(
            plant=plant_id,
            health=health,
            treatment=treatment_rec,
            explainability=explain_info,
            metadata=metadata
        )

    def warmup(self) -> dict:
        """
        Warm up all models with dummy inference.

        Returns timing information for each component.
        """
        import numpy as np

        logger.info("Warming up classification service...")

        # Create dummy image
        dummy_image = np.random.rand(3, 224, 224).astype(np.float32)

        times = {}

        # Warmup species classifier
        start = time.perf_counter()
        self.species_classifier.predict(dummy_image)
        times["species_classifier_ms"] = (time.perf_counter() - start) * 1000

        # Warmup disease detector
        start = time.perf_counter()
        self.disease_detector.predict(dummy_image)
        times["disease_detector_ms"] = (time.perf_counter() - start) * 1000

        logger.info(f"Warmup complete: {times}")
        return times

    def get_supported_crops(self) -> list[str]:
        """Get list of crops with specialized models."""
        return self.disease_detector.get_crop_specific_models()

    def get_supported_diseases(self) -> list[str]:
        """Get list of diseases with treatment plans."""
        from app.services.treatment_service import TreatmentDatabase
        return TreatmentDatabase.list_diseases()


# Singleton instance for dependency injection
_classification_service: Optional[ClassificationService] = None


def get_classification_service() -> ClassificationService:
    """Get or create the classification service singleton."""
    global _classification_service
    if _classification_service is None:
        _classification_service = ClassificationService()
    return _classification_service
