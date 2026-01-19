"""
External Model Integrations

Provides integration with open-source pre-trained models for comparison:
- MobileNetV2 Plant Disease (HuggingFace)
- ViT Crop Diseases (HuggingFace)
- Pl@ntNet API (External API)

These can be used for side-by-side comparison with our internal models.
"""

import os
import time
import base64
import logging
import asyncio
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class ExternalModelType(str, Enum):
    """Supported external models for comparison."""
    MOBILENET_V2 = "mobilenet_v2"
    VIT_CROP = "vit_crop"
    PLANTNET = "plantnet"


@dataclass
class ExternalModelResult:
    """Result from an external model."""
    model_name: str
    model_type: ExternalModelType
    prediction: Optional[str]
    confidence: Optional[float]
    raw_label: Optional[str]
    processing_time_ms: float
    error: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class BaseExternalModel(ABC):
    """Base class for external model integrations."""

    def __init__(self):
        self.is_loaded = False
        self.load_error: Optional[str] = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> ExternalModelType:
        """Model type identifier."""
        pass

    @abstractmethod
    async def predict(self, image: Image.Image) -> ExternalModelResult:
        """Run prediction on an image."""
        pass

    def _create_error_result(self, error: str, processing_time_ms: float = 0) -> ExternalModelResult:
        """Create an error result."""
        return ExternalModelResult(
            model_name=self.model_name,
            model_type=self.model_type,
            prediction=None,
            confidence=None,
            raw_label=None,
            processing_time_ms=processing_time_ms,
            error=error
        )


class MobileNetV2PlantDisease(BaseExternalModel):
    """
    MobileNetV2 Plant Disease Model from HuggingFace.

    Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
    Classes: 38 plant disease classes from PlantVillage
    Accuracy: ~95.41%

    Reference: https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
    """

    MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

    # Label mapping for user-friendly display
    LABEL_MAPPING = {
        "Tomato___Early_blight": "Early Blight",
        "Tomato___Late_blight": "Late Blight",
        "Tomato___Bacterial_spot": "Bacterial Spot",
        "Tomato___Leaf_Mold": "Leaf Mold",
        "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites",
        "Tomato___Target_Spot": "Target Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
        "Tomato___Tomato_mosaic_virus": "Mosaic Virus",
        "Tomato___healthy": "Healthy",
        "Potato___Early_blight": "Early Blight",
        "Potato___Late_blight": "Late Blight",
        "Potato___healthy": "Healthy",
        "Apple___Apple_scab": "Apple Scab",
        "Apple___Black_rot": "Black Rot",
        "Apple___Cedar_apple_rust": "Cedar Apple Rust",
        "Apple___healthy": "Healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
        "Corn_(maize)___Common_rust_": "Common Rust",
        "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
        "Corn_(maize)___healthy": "Healthy",
        "Grape___Black_rot": "Black Rot",
        "Grape___Esca_(Black_Measles)": "Esca (Black Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf Blight",
        "Grape___healthy": "Healthy",
    }

    def __init__(self):
        super().__init__()
        self._pipeline = None

    @property
    def model_name(self) -> str:
        return "MobileNetV2 Plant Disease (HF)"

    @property
    def model_type(self) -> ExternalModelType:
        return ExternalModelType.MOBILENET_V2

    def _load_model(self):
        """Lazy load the model pipeline."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading MobileNetV2 model: {self.MODEL_ID}")
                self._pipeline = pipeline(
                    "image-classification",
                    model=self.MODEL_ID,
                    device=-1  # CPU
                )
                self.is_loaded = True
                logger.info("MobileNetV2 model loaded successfully")
            except Exception as e:
                self.load_error = str(e)
                logger.error(f"Failed to load MobileNetV2 model: {e}")
                raise

    def _parse_label(self, raw_label: str) -> str:
        """Parse raw label to user-friendly format."""
        # Check if we have a mapping
        if raw_label in self.LABEL_MAPPING:
            return self.LABEL_MAPPING[raw_label]

        # Otherwise, parse the label format: "Crop___Disease"
        if "___" in raw_label:
            parts = raw_label.split("___")
            if len(parts) == 2:
                disease = parts[1].replace("_", " ")
                return disease

        return raw_label.replace("_", " ")

    def _extract_crop(self, raw_label: str) -> Optional[str]:
        """Extract crop name from label."""
        if "___" in raw_label:
            crop = raw_label.split("___")[0]
            return crop.replace("_", " ").replace("(maize)", "").strip()
        return None

    async def predict(self, image: Image.Image) -> ExternalModelResult:
        """Run prediction using MobileNetV2."""
        start_time = time.time()

        try:
            # Load model if needed
            self._load_model()

            # Run prediction in thread pool to not block
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._pipeline(image, top_k=3)
            )

            processing_time = (time.time() - start_time) * 1000

            if results and len(results) > 0:
                top_result = results[0]
                raw_label = top_result["label"]
                confidence = top_result["score"]

                return ExternalModelResult(
                    model_name=self.model_name,
                    model_type=self.model_type,
                    prediction=self._parse_label(raw_label),
                    confidence=confidence,
                    raw_label=raw_label,
                    processing_time_ms=processing_time,
                    additional_info={
                        "crop": self._extract_crop(raw_label),
                        "top_3": [
                            {
                                "label": r["label"],
                                "disease": self._parse_label(r["label"]),
                                "confidence": r["score"]
                            }
                            for r in results[:3]
                        ]
                    }
                )
            else:
                return self._create_error_result("No predictions returned", processing_time)

        except ImportError as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"transformers library not installed: {e}",
                processing_time
            )
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"MobileNetV2 prediction error: {e}")
            return self._create_error_result(str(e), processing_time)


class ViTCropDiseases(BaseExternalModel):
    """
    Vision Transformer Crop Diseases Model from HuggingFace.

    Model: wambugu71/crop_leaf_diseases_vit
    Classes: 14 (Corn, Potato, Rice, Wheat diseases)
    Accuracy: ~98%

    Reference: https://huggingface.co/wambugu71/crop_leaf_diseases_vit
    """

    MODEL_ID = "wambugu71/crop_leaf_diseases_vit"

    # Supported crops for this model
    SUPPORTED_CROPS = {"corn", "potato", "rice", "wheat"}

    # Label mapping
    LABEL_MAPPING = {
        "Corn_Common_Rust": "Common Rust",
        "Corn_Gray_Leaf_Spot": "Gray Leaf Spot",
        "Corn_Leaf_Blight": "Leaf Blight",
        "Corn_Healthy": "Healthy",
        "Potato_Early_Blight": "Early Blight",
        "Potato_Late_Blight": "Late Blight",
        "Potato_Healthy": "Healthy",
        "Rice_Brown_Spot": "Brown Spot",
        "Rice_Leaf_Blast": "Leaf Blast",
        "Rice_Healthy": "Healthy",
        "Wheat_Brown_Rust": "Brown Rust",
        "Wheat_Yellow_Rust": "Yellow Rust",
        "Wheat_Healthy": "Healthy",
        "Invalid": "Invalid/Unknown"
    }

    def __init__(self):
        super().__init__()
        self._model = None
        self._feature_extractor = None

    @property
    def model_name(self) -> str:
        return "ViT Crop Diseases (HF)"

    @property
    def model_type(self) -> ExternalModelType:
        return ExternalModelType.VIT_CROP

    def _load_model(self):
        """Lazy load the model and feature extractor."""
        if self._model is None:
            try:
                from transformers import ViTImageProcessor, ViTForImageClassification
                import torch

                logger.info(f"Loading ViT model: {self.MODEL_ID}")

                self._feature_extractor = ViTImageProcessor.from_pretrained(self.MODEL_ID)
                self._model = ViTForImageClassification.from_pretrained(
                    self.MODEL_ID,
                    ignore_mismatched_sizes=True
                )
                self._model.eval()

                self.is_loaded = True
                logger.info("ViT model loaded successfully")
            except Exception as e:
                self.load_error = str(e)
                logger.error(f"Failed to load ViT model: {e}")
                raise

    def _parse_label(self, raw_label: str) -> str:
        """Parse raw label to user-friendly format."""
        if raw_label in self.LABEL_MAPPING:
            return self.LABEL_MAPPING[raw_label]
        return raw_label.replace("_", " ")

    def _extract_crop(self, raw_label: str) -> Optional[str]:
        """Extract crop name from label."""
        for crop in self.SUPPORTED_CROPS:
            if raw_label.lower().startswith(crop):
                return crop.capitalize()
        return None

    async def predict(self, image: Image.Image) -> ExternalModelResult:
        """Run prediction using ViT."""
        start_time = time.time()

        try:
            import torch

            # Load model if needed
            self._load_model()

            # Process image
            def run_inference():
                inputs = self._feature_extractor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # Get top 3 predictions
                    top_k = torch.topk(probs, k=min(3, probs.shape[-1]))

                    return top_k.indices[0].tolist(), top_k.values[0].tolist()

            loop = asyncio.get_event_loop()
            indices, scores = await loop.run_in_executor(None, run_inference)

            processing_time = (time.time() - start_time) * 1000

            # Get labels
            id2label = self._model.config.id2label
            top_label = id2label.get(indices[0], "Unknown")
            top_confidence = scores[0]

            return ExternalModelResult(
                model_name=self.model_name,
                model_type=self.model_type,
                prediction=self._parse_label(top_label),
                confidence=top_confidence,
                raw_label=top_label,
                processing_time_ms=processing_time,
                additional_info={
                    "crop": self._extract_crop(top_label),
                    "supported_crops": list(self.SUPPORTED_CROPS),
                    "top_3": [
                        {
                            "label": id2label.get(idx, "Unknown"),
                            "disease": self._parse_label(id2label.get(idx, "Unknown")),
                            "confidence": score
                        }
                        for idx, score in zip(indices, scores)
                    ]
                }
            )

        except ImportError as e:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result(
                f"transformers/torch library not installed: {e}",
                processing_time
            )
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"ViT prediction error: {e}")
            return self._create_error_result(str(e), processing_time)


class PlantNetAPI(BaseExternalModel):
    """
    Pl@ntNet API Integration for species identification.

    API: https://my.plantnet.org/
    Species: 50,000+ plant species

    Requires API key from https://my.plantnet.org/
    """

    API_BASE_URL = "https://my-api.plantnet.org/v2/identify/all"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("PLANTNET_API_KEY")
        if self.api_key:
            self.is_loaded = True

    @property
    def model_name(self) -> str:
        return "Pl@ntNet API"

    @property
    def model_type(self) -> ExternalModelType:
        return ExternalModelType.PLANTNET

    async def predict(self, image: Image.Image) -> ExternalModelResult:
        """Run prediction using PlantNet API."""
        start_time = time.time()

        if not self.api_key:
            return self._create_error_result(
                "PlantNet API key not configured. Set PLANTNET_API_KEY environment variable.",
                0
            )

        try:
            # Convert image to bytes
            buffer = BytesIO()
            image_rgb = image.convert("RGB")
            image_rgb.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()

            # Make API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"images": ("plant.jpg", image_bytes, "image/jpeg")}
                params = {
                    "api-key": self.api_key,
                    "organs": "leaf"  # Default to leaf
                }

                response = await client.post(
                    self.API_BASE_URL,
                    files=files,
                    params=params
                )

                processing_time = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    data = response.json()

                    if "results" in data and len(data["results"]) > 0:
                        top_result = data["results"][0]
                        species = top_result.get("species", {})

                        scientific_name = species.get("scientificNameWithoutAuthor", "Unknown")
                        common_names = species.get("commonNames", [])
                        common_name = common_names[0] if common_names else None
                        confidence = top_result.get("score", 0)

                        # Get taxonomy
                        family = species.get("family", {}).get("scientificNameWithoutAuthor")
                        genus = species.get("genus", {}).get("scientificNameWithoutAuthor")

                        return ExternalModelResult(
                            model_name=self.model_name,
                            model_type=self.model_type,
                            prediction=common_name or scientific_name,
                            confidence=confidence,
                            raw_label=scientific_name,
                            processing_time_ms=processing_time,
                            additional_info={
                                "scientific_name": scientific_name,
                                "common_names": common_names[:3] if common_names else [],
                                "family": family,
                                "genus": genus,
                                "top_3": [
                                    {
                                        "species": r.get("species", {}).get("scientificNameWithoutAuthor"),
                                        "common_name": r.get("species", {}).get("commonNames", [None])[0],
                                        "confidence": r.get("score", 0)
                                    }
                                    for r in data["results"][:3]
                                ]
                            }
                        )
                    else:
                        return self._create_error_result(
                            "No species identified",
                            processing_time
                        )

                elif response.status_code == 401:
                    return self._create_error_result(
                        "Invalid PlantNet API key",
                        processing_time
                    )
                elif response.status_code == 429:
                    return self._create_error_result(
                        "PlantNet API rate limit exceeded",
                        processing_time
                    )
                else:
                    return self._create_error_result(
                        f"PlantNet API error: {response.status_code}",
                        processing_time
                    )

        except httpx.TimeoutException:
            processing_time = (time.time() - start_time) * 1000
            return self._create_error_result("PlantNet API timeout", processing_time)
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"PlantNet API error: {e}")
            return self._create_error_result(str(e), processing_time)


class ExternalModelRegistry:
    """
    Registry for external models.

    Provides lazy loading and unified access to all external models.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._initialized = False
        return cls._instance

    def _init_models(self):
        """Initialize available models."""
        if not self._initialized:
            self._models = {
                ExternalModelType.MOBILENET_V2: MobileNetV2PlantDisease(),
                ExternalModelType.VIT_CROP: ViTCropDiseases(),
                ExternalModelType.PLANTNET: PlantNetAPI(),
            }
            self._initialized = True

    def get_model(self, model_type: ExternalModelType) -> BaseExternalModel:
        """Get a specific model by type."""
        self._init_models()
        return self._models.get(model_type)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their status."""
        self._init_models()

        models = []
        for model_type, model in self._models.items():
            models.append({
                "type": model_type.value,
                "name": model.model_name,
                "is_loaded": model.is_loaded,
                "error": model.load_error
            })
        return models

    async def run_comparison(
        self,
        image: Image.Image,
        model_types: Optional[List[ExternalModelType]] = None
    ) -> Dict[str, ExternalModelResult]:
        """
        Run predictions on multiple models for comparison.

        Args:
            image: PIL Image to classify
            model_types: List of model types to use, or None for all

        Returns:
            Dictionary mapping model type to results
        """
        self._init_models()

        if model_types is None:
            model_types = list(self._models.keys())

        # Run predictions concurrently
        tasks = []
        selected_types = []

        for model_type in model_types:
            if model_type in self._models:
                model = self._models[model_type]
                tasks.append(model.predict(image))
                selected_types.append(model_type)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dictionary
        comparison = {}
        for model_type, result in zip(selected_types, results):
            if isinstance(result, Exception):
                comparison[model_type.value] = ExternalModelResult(
                    model_name=self._models[model_type].model_name,
                    model_type=model_type,
                    prediction=None,
                    confidence=None,
                    raw_label=None,
                    processing_time_ms=0,
                    error=str(result)
                )
            else:
                comparison[model_type.value] = result

        return comparison


# Singleton instance
def get_external_model_registry() -> ExternalModelRegistry:
    """Get the external model registry singleton."""
    return ExternalModelRegistry()
