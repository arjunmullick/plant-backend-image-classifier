"""
Base classes and interfaces for ML components.

All ML components inherit from BaseMLComponent to ensure consistent
interfaces and enable easy swapping of implementations.

Design Principles:
1. Each component is independently replaceable
2. Components declare their dependencies explicitly
3. All predictions include confidence scores
4. Components support both CPU and GPU inference
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Generic
from pathlib import Path
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PredictionResult(Generic[T]):
    """
    Generic prediction result wrapper.

    Attributes:
        prediction: The actual prediction result
        confidence: Confidence score (0-1)
        latency_ms: Inference time in milliseconds
        model_version: Version of model used
        metadata: Additional metadata
    """
    prediction: T
    confidence: float
    latency_ms: float = 0.0
    model_version: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    version: str
    architecture: str
    input_size: tuple[int, int]
    num_classes: int
    class_names: list[str]
    device: str
    loaded_at: float = field(default_factory=time.time)


class BaseMLComponent(ABC):
    """
    Abstract base class for all ML components.

    All ML components (species classifier, disease detector, etc.)
    must inherit from this class to ensure consistent interfaces.

    Subclasses must implement:
        - load_model(): Load model weights and prepare for inference
        - predict(): Run inference on preprocessed input
        - get_model_info(): Return information about the loaded model
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the ML component.

        Args:
            model_path: Path to model weights/checkpoint
            device: Device for inference ("cpu", "cuda", "mps")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._is_loaded = False
        self._model_info: Optional[ModelInfo] = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights and prepare for inference.

        This method should:
        1. Load model architecture
        2. Load pretrained weights
        3. Move model to the appropriate device
        4. Set model to evaluation mode
        5. Populate self._model_info
        """
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> PredictionResult:
        """
        Run inference on preprocessed input.

        Args:
            input_data: Preprocessed input array (typically image tensor)

        Returns:
            PredictionResult with prediction and confidence
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return information about the loaded model."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Ensure model is loaded, loading if necessary."""
        if not self._is_loaded:
            logger.info(f"Loading model: {self.__class__.__name__}")
            self.load_model()
            self._is_loaded = True

    def warmup(self, input_shape: tuple = (1, 3, 224, 224)) -> float:
        """
        Warm up the model with a dummy inference.

        Returns:
            Warmup inference time in milliseconds
        """
        self.ensure_loaded()
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        start = time.perf_counter()
        self.predict(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Model warmup completed in {elapsed:.2f}ms")
        return elapsed


class ModelRegistry:
    """
    Registry for managing multiple ML models.

    Supports:
    - Registering models by name and version
    - Loading models on-demand
    - Model versioning for A/B testing
    - Graceful model swapping

    Usage:
        registry = ModelRegistry()
        registry.register("species_classifier", SpeciesClassifier, "/path/to/model.pt")
        classifier = registry.get("species_classifier")
    """

    _instance: Optional["ModelRegistry"] = None
    _models: dict[str, BaseMLComponent] = {}
    _model_configs: dict[str, dict] = {}

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._model_configs = {}
        return cls._instance

    def register(
        self,
        name: str,
        model_class: type[BaseMLComponent],
        model_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any
    ) -> None:
        """
        Register a model class for lazy loading.

        Args:
            name: Unique model identifier
            model_class: Class to instantiate
            model_path: Path to model weights
            device: Inference device
            **kwargs: Additional arguments for model initialization
        """
        self._model_configs[name] = {
            "class": model_class,
            "model_path": model_path,
            "device": device,
            **kwargs
        }
        logger.info(f"Registered model: {name}")

    def get(self, name: str, load: bool = True) -> BaseMLComponent:
        """
        Get a model instance, loading if necessary.

        Args:
            name: Model identifier
            load: Whether to load the model if not already loaded

        Returns:
            Model instance
        """
        if name not in self._models:
            if name not in self._model_configs:
                raise KeyError(f"Model '{name}' not registered")

            config = self._model_configs[name]
            model_class = config.pop("class")
            self._models[name] = model_class(**config)

            if load:
                self._models[name].ensure_loaded()

        return self._models[name]

    def is_registered(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._model_configs or name in self._models

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(set(self._model_configs.keys()) | set(self._models.keys()))

    def unload(self, name: str) -> None:
        """Unload a model to free memory."""
        if name in self._models:
            del self._models[name]
            logger.info(f"Unloaded model: {name}")

    def reload(self, name: str) -> BaseMLComponent:
        """Reload a model (useful for model updates)."""
        self.unload(name)
        return self.get(name)


@dataclass
class HierarchicalPrediction:
    """
    Hierarchical taxonomy prediction result.

    Used for plant species classification where predictions
    are made at multiple taxonomic levels.
    """
    family: str
    family_confidence: float
    genus: str
    genus_confidence: float
    species: str
    species_confidence: float
    common_name: Optional[str] = None
    alternatives: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class DiseasePrediction:
    """
    Disease detection prediction result.
    """
    is_healthy: bool
    disease_name: Optional[str]
    confidence: float
    visual_symptoms: list[str] = field(default_factory=list)
    affected_area: Optional[float] = None  # Percentage
    disease_stage: Optional[str] = None
