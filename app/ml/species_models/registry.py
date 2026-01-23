"""
Species Model Registry

Manages species identification models and coordinates parallel inference.

Supports:
- HuggingFace models (iNaturalist ViT, EfficientNet, etc.)
- External APIs (PlantNet, Kindwise)
- Internal model (existing SpeciesClassifier)
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from PIL import Image

from app.ml.species_models.base import (
    SpeciesModelInterface,
    SpeciesPrediction,
    TaxonomyDepth,
)

logger = logging.getLogger(__name__)


class SpeciesModelType(str, Enum):
    """Available species identification models."""
    # External APIs
    PLANTNET = "plantnet"
    KINDWISE = "kindwise"

    # HuggingFace models
    INATURALIST_VIT = "inaturalist_vit"
    EFFICIENTNET_FLORA = "efficientnet_flora"
    PLANTCLEF_SWIN = "plantclef_swin"

    # Internal model
    INTERNAL = "internal"


class SpeciesModelRegistry:
    """
    Registry for species identification models.

    Manages model lifecycle, provides parallel inference,
    and handles graceful degradation on model failures.
    """

    _instance: Optional['SpeciesModelRegistry'] = None

    def __init__(self):
        """Initialize registry with available models."""
        self._models: Dict[SpeciesModelType, SpeciesModelInterface] = {}
        self._model_info: Dict[SpeciesModelType, Dict[str, Any]] = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> 'SpeciesModelRegistry':
        """Get singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(
        self,
        plantnet_api_key: Optional[str] = None,
        kindwise_api_key: Optional[str] = None,
        enable_huggingface: bool = True,
        enable_internal: bool = True,
    ):
        """
        Initialize models.

        Args:
            plantnet_api_key: API key for PlantNet
            kindwise_api_key: API key for Kindwise/Plant.id
            enable_huggingface: Whether to load HuggingFace models
            enable_internal: Whether to use internal model
        """
        if self._initialized:
            return

        logger.info("Initializing species model registry...")

        # Initialize external API models
        if plantnet_api_key:
            try:
                from app.ml.species_models.models.plantnet_model import PlantNetSpeciesModel
                self._models[SpeciesModelType.PLANTNET] = PlantNetSpeciesModel(
                    api_key=plantnet_api_key
                )
                logger.info("PlantNet model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PlantNet: {e}")

        if kindwise_api_key:
            try:
                from app.ml.species_models.models.kindwise_model import KindwiseSpeciesModel
                self._models[SpeciesModelType.KINDWISE] = KindwiseSpeciesModel(
                    api_key=kindwise_api_key
                )
                logger.info("Kindwise model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Kindwise: {e}")

        # Initialize HuggingFace models
        if enable_huggingface:
            try:
                from app.ml.species_models.models.inaturalist_vit import iNaturalistViTModel
                self._models[SpeciesModelType.INATURALIST_VIT] = iNaturalistViTModel()
                logger.info("iNaturalist ViT model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize iNaturalist ViT: {e}")

            try:
                from app.ml.species_models.models.efficientnet_flora import EfficientNetFloraModel
                self._models[SpeciesModelType.EFFICIENTNET_FLORA] = EfficientNetFloraModel()
                logger.info("EfficientNet Flora model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EfficientNet Flora: {e}")

        # Initialize internal model
        if enable_internal:
            try:
                from app.ml.species_models.models.internal_model import InternalSpeciesModel
                self._models[SpeciesModelType.INTERNAL] = InternalSpeciesModel()
                logger.info("Internal species model initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize internal model: {e}")

        self._initialized = True
        logger.info(f"Species model registry initialized with {len(self._models)} models")

    def get_model(self, model_type: SpeciesModelType) -> Optional[SpeciesModelInterface]:
        """Get a specific model by type."""
        return self._models.get(model_type)

    def get_available_models(self) -> List[SpeciesModelType]:
        """Get list of available (loaded) models."""
        return list(self._models.keys())

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get info for all available models."""
        return [
            {
                "type": model_type.value,
                **model.get_model_info()
            }
            for model_type, model in self._models.items()
        ]

    async def predict_single(
        self,
        model_type: SpeciesModelType,
        image: Image.Image
    ) -> SpeciesPrediction:
        """
        Run prediction with a single model.

        Args:
            model_type: Model to use
            image: PIL Image to classify

        Returns:
            SpeciesPrediction
        """
        model = self._models.get(model_type)
        if model is None:
            from app.ml.species_models.base import NormalizedTaxonomy
            return SpeciesPrediction(
                model_name=model_type.value,
                model_type=model_type.value,
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error=f"Model {model_type.value} not available"
            )

        try:
            return await model.predict(image)
        except Exception as e:
            logger.error(f"Error in {model_type.value} prediction: {e}")
            from app.ml.species_models.base import NormalizedTaxonomy
            return SpeciesPrediction(
                model_name=model.model_name,
                model_type=model_type.value,
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error=str(e)
            )

    async def predict_all(
        self,
        image: Image.Image,
        model_types: Optional[List[SpeciesModelType]] = None,
        timeout: float = 30.0
    ) -> Dict[SpeciesModelType, SpeciesPrediction]:
        """
        Run prediction with all (or specified) models in parallel.

        Args:
            image: PIL Image to classify
            model_types: Specific models to use (None = all available)
            timeout: Maximum time to wait for all models

        Returns:
            Dict mapping model type to prediction
        """
        if model_types is None:
            model_types = list(self._models.keys())

        # Filter to available models
        available_types = [mt for mt in model_types if mt in self._models]

        if not available_types:
            logger.warning("No models available for prediction")
            return {}

        # Create tasks for parallel execution
        tasks = {
            model_type: self.predict_single(model_type, image)
            for model_type in available_types
        }

        # Run with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks.values(), return_exceptions=True),
                timeout=timeout
            )

            # Map results back to model types
            predictions = {}
            for model_type, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    from app.ml.species_models.base import NormalizedTaxonomy
                    predictions[model_type] = SpeciesPrediction(
                        model_name=model_type.value,
                        model_type=model_type.value,
                        taxonomy=NormalizedTaxonomy(),
                        confidence=0.0,
                        error=str(result)
                    )
                else:
                    predictions[model_type] = result

            return predictions

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for species models after {timeout}s")
            return {
                model_type: SpeciesPrediction(
                    model_name=model_type.value,
                    model_type=model_type.value,
                    taxonomy=NormalizedTaxonomy(),
                    confidence=0.0,
                    error="Timeout"
                )
                for model_type in available_types
            }

    def set_api_key(self, model_type: SpeciesModelType, api_key: str):
        """
        Set or update API key for external model.

        Args:
            model_type: Model to update
            api_key: New API key
        """
        model = self._models.get(model_type)
        if model and hasattr(model, 'api_key'):
            model.api_key = api_key
            if hasattr(model, '_is_loaded'):
                model._is_loaded = True
            logger.info(f"Updated API key for {model_type.value}")

    def register_model(
        self,
        model_type: SpeciesModelType,
        model: SpeciesModelInterface
    ):
        """
        Register a custom model.

        Args:
            model_type: Type identifier for the model
            model: Model instance implementing SpeciesModelInterface
        """
        self._models[model_type] = model
        logger.info(f"Registered custom model: {model_type.value}")


# Singleton accessor
_registry: Optional[SpeciesModelRegistry] = None


def get_species_model_registry() -> SpeciesModelRegistry:
    """Get the species model registry singleton."""
    global _registry
    if _registry is None:
        _registry = SpeciesModelRegistry.get_instance()
    return _registry
