"""
FastAPI dependency injection.

Provides dependency injection for services and components,
enabling easy testing and component swapping.
"""

from functools import lru_cache
from typing import Generator

from app.services.classification_service import ClassificationService, get_classification_service
from app.services.treatment_service import TreatmentRecommendationService
from app.ml.preprocessor import ImagePreprocessor
from app.ml.species_classifier import SpeciesClassifier
from app.ml.disease_detector import DiseaseDetector
from app.ml.explainability import ExplainabilityEngine


@lru_cache()
def get_preprocessor() -> ImagePreprocessor:
    """Get cached image preprocessor."""
    return ImagePreprocessor()


@lru_cache()
def get_species_classifier() -> SpeciesClassifier:
    """Get cached species classifier."""
    classifier = SpeciesClassifier()
    classifier.ensure_loaded()
    return classifier


@lru_cache()
def get_disease_detector() -> DiseaseDetector:
    """Get cached disease detector."""
    detector = DiseaseDetector()
    detector.ensure_loaded()
    return detector


@lru_cache()
def get_explainability_engine() -> ExplainabilityEngine:
    """Get cached explainability engine."""
    return ExplainabilityEngine()


@lru_cache()
def get_treatment_service() -> TreatmentRecommendationService:
    """Get cached treatment service."""
    return TreatmentRecommendationService()


# Re-export the main service getter
__all__ = [
    "get_classification_service",
    "get_preprocessor",
    "get_species_classifier",
    "get_disease_detector",
    "get_explainability_engine",
    "get_treatment_service",
]
