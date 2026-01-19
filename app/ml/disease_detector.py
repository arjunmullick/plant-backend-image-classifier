"""
Disease Detection Module

Multi-task disease detection with support for:
- Binary healthy/diseased classification
- Multi-class disease identification
- Visual symptom extraction
- Severity estimation (optional segmentation)

Architecture Strategy:
- Crop-agnostic base model for initial disease detection
- Crop-specific fine-tuned models for detailed diagnosis
- Model routing based on species classification result

Model Architecture (Production):
1. Base Disease Detector: ResNet50 or EfficientNet
2. Crop-Specific Models: Fine-tuned variants per major crop
3. Severity Segmentation: U-Net or DeepLabV3 (optional)

Training Data:
- PlantVillage: 38 crop-disease combinations
- PlantDoc: Real-world images with higher variance
- Custom datasets per region/crop as needed
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

import numpy as np

from app.ml.base import BaseMLComponent, PredictionResult, ModelInfo, DiseasePrediction

logger = logging.getLogger(__name__)


@dataclass
class DiseaseLabel:
    """Disease label with associated metadata."""
    index: int
    name: str
    crop: str
    is_healthy: bool
    visual_symptoms: list[str]
    severity_stages: list[str]
    scientific_name: Optional[str] = None


class DiseaseDatabase:
    """
    Database of plant diseases and their visual symptoms.

    Iteration Point: Add new diseases by extending this database.
    In production, load from JSON/database file.
    """

    DISEASES: list[DiseaseLabel] = [
        # Healthy states
        DiseaseLabel(
            0, "Healthy", "tomato", True,
            visual_symptoms=[],
            severity_stages=["N/A"]
        ),
        DiseaseLabel(
            1, "Healthy", "potato", True,
            visual_symptoms=[],
            severity_stages=["N/A"]
        ),
        DiseaseLabel(
            2, "Healthy", "apple", True,
            visual_symptoms=[],
            severity_stages=["N/A"]
        ),

        # Tomato diseases
        DiseaseLabel(
            3, "Early Blight", "tomato", False,
            visual_symptoms=[
                "brown concentric rings on older leaves",
                "yellowing around lesions",
                "dark spots with target-like pattern",
                "lower leaves affected first"
            ],
            severity_stages=["Early", "Moderate", "Severe", "Critical"],
            scientific_name="Alternaria solani"
        ),
        DiseaseLabel(
            4, "Late Blight", "tomato", False,
            visual_symptoms=[
                "water-soaked lesions on leaves",
                "white fuzzy growth on leaf undersides",
                "dark brown to black lesions",
                "rapid wilting and death of foliage"
            ],
            severity_stages=["Early", "Moderate", "Severe", "Critical"],
            scientific_name="Phytophthora infestans"
        ),
        DiseaseLabel(
            5, "Bacterial Spot", "tomato", False,
            visual_symptoms=[
                "small dark spots with yellow halos",
                "raised scab-like lesions on fruit",
                "water-soaked appearance initially",
                "spots may merge into larger areas"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Xanthomonas spp."
        ),
        DiseaseLabel(
            6, "Septoria Leaf Spot", "tomato", False,
            visual_symptoms=[
                "small circular spots with dark borders",
                "gray or tan centers in spots",
                "tiny black dots in lesion centers",
                "starts on lower leaves"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Septoria lycopersici"
        ),
        DiseaseLabel(
            7, "Yellow Leaf Curl Virus", "tomato", False,
            visual_symptoms=[
                "upward curling of leaf margins",
                "yellowing between leaf veins",
                "stunted plant growth",
                "small, deformed leaves"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="TYLCV"
        ),

        # Potato diseases
        DiseaseLabel(
            8, "Early Blight", "potato", False,
            visual_symptoms=[
                "dark brown spots with concentric rings",
                "target-like pattern on leaves",
                "yellowing around lesions",
                "premature leaf drop"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Alternaria solani"
        ),
        DiseaseLabel(
            9, "Late Blight", "potato", False,
            visual_symptoms=[
                "water-soaked spots on leaves",
                "white mold on leaf undersides",
                "brown to black stem lesions",
                "tuber rot with reddish-brown color"
            ],
            severity_stages=["Early", "Moderate", "Severe", "Critical"],
            scientific_name="Phytophthora infestans"
        ),

        # Apple diseases
        DiseaseLabel(
            10, "Apple Scab", "apple", False,
            visual_symptoms=[
                "olive-green to brown spots on leaves",
                "velvety or corky texture on fruit",
                "distorted or cracked fruit surface",
                "premature leaf and fruit drop"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Venturia inaequalis"
        ),
        DiseaseLabel(
            11, "Black Rot", "apple", False,
            visual_symptoms=[
                "frog-eye leaf spots",
                "brown rot expanding on fruit",
                "black pycnidia on fruit surface",
                "mummified fruit on tree"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Botryosphaeria obtusa"
        ),
        DiseaseLabel(
            12, "Cedar Apple Rust", "apple", False,
            visual_symptoms=[
                "bright orange spots on leaves",
                "tube-like projections on leaf undersides",
                "yellow halos around spots",
                "deformed fruit with orange lesions"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Gymnosporangium juniperi-virginianae"
        ),

        # Grape diseases
        DiseaseLabel(
            13, "Black Rot", "grape", False,
            visual_symptoms=[
                "brown circular spots on leaves",
                "dark margins around leaf spots",
                "shriveled black mummified berries",
                "black fruiting bodies visible"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Guignardia bidwellii"
        ),
        DiseaseLabel(
            14, "Esca (Black Measles)", "grape", False,
            visual_symptoms=[
                "interveinal leaf discoloration",
                "tiger-stripe pattern on leaves",
                "dark spots on berries",
                "white fungal growth in wood"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Phaeomoniella chlamydospora"
        ),

        # Corn diseases
        DiseaseLabel(
            15, "Common Rust", "corn", False,
            visual_symptoms=[
                "small circular to elongated pustules",
                "reddish-brown spore masses",
                "pustules on both leaf surfaces",
                "golden to brown color as pustules age"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Puccinia sorghi"
        ),
        DiseaseLabel(
            16, "Northern Leaf Blight", "corn", False,
            visual_symptoms=[
                "long elliptical gray-green lesions",
                "lesions 1-6 inches long",
                "cigar-shaped spots",
                "tan coloring as lesions mature"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Exserohilum turcicum"
        ),
        DiseaseLabel(
            17, "Gray Leaf Spot", "corn", False,
            visual_symptoms=[
                "rectangular gray to tan lesions",
                "lesions run parallel to leaf veins",
                "distinct parallel edges",
                "lesions may coalesce in severe cases"
            ],
            severity_stages=["Early", "Moderate", "Severe"],
            scientific_name="Cercospora zeae-maydis"
        ),
    ]

    def __init__(self):
        self._index_to_disease = {d.index: d for d in self.DISEASES}
        self._name_to_disease = {d.name: d for d in self.DISEASES}
        self._crop_diseases = self._build_crop_index()

    def _build_crop_index(self) -> dict[str, list[DiseaseLabel]]:
        """Build index of diseases by crop."""
        index = {}
        for disease in self.DISEASES:
            if disease.crop not in index:
                index[disease.crop] = []
            index[disease.crop].append(disease)
        return index

    def get_by_index(self, index: int) -> Optional[DiseaseLabel]:
        return self._index_to_disease.get(index)

    def get_by_name(self, name: str) -> Optional[DiseaseLabel]:
        return self._name_to_disease.get(name)

    def get_diseases_for_crop(self, crop: str) -> list[DiseaseLabel]:
        """Get all diseases for a specific crop."""
        return self._crop_diseases.get(crop.lower(), [])

    @property
    def num_classes(self) -> int:
        return len(self.DISEASES)


class DiseaseDetector(BaseMLComponent):
    """
    Disease detection with visual symptom extraction.

    Supports:
    - Crop-agnostic disease detection
    - Crop-specific routing for better accuracy
    - Visual symptom description
    - Severity estimation

    Model Routing Strategy:
    1. If crop is known (from species classifier):
       - Use crop-specific disease model
    2. If crop is unknown:
       - Use general disease detector
       - Lower confidence threshold

    Architecture (Production):
    ```
    Input Image
         |
    Shared Backbone (EfficientNet/ResNet)
         |
    ┌────┴────┐
    │         │
    Binary    Multi-class
    Head      Head
    (H/D)     (Disease ID)
    ```
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        crop_specific: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize disease detector.

        Args:
            model_path: Path to model weights
            device: Inference device
            crop_specific: Whether to use crop-specific models
            confidence_threshold: Minimum confidence for disease detection
        """
        super().__init__(model_path, device)
        self.crop_specific = crop_specific
        self.confidence_threshold = confidence_threshold
        self.disease_db = DiseaseDatabase()

        # Placeholder for crop-specific model registry
        self._crop_models: dict[str, "PlaceholderDiseaseModel"] = {}

    def load_model(self) -> None:
        """
        Load disease detection models.

        Production Implementation:
        ```python
        # Load base model
        self.base_model = models.efficientnet_v2_s(pretrained=True)

        # Replace classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()

        # Binary head (healthy/diseased)
        self.binary_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)
        )

        # Disease identification head
        self.disease_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, self.disease_db.num_classes)
        )

        # Load weights
        if self.model_path:
            state_dict = torch.load(self.model_path)
            self.load_state_dict(state_dict)
        ```
        """
        logger.info("Loading disease detection models")

        # Load general disease detector
        self.model = PlaceholderDiseaseModel(
            num_classes=self.disease_db.num_classes
        )

        # Load crop-specific models (in production)
        if self.crop_specific:
            for crop in ["tomato", "potato", "apple", "corn", "grape"]:
                self._crop_models[crop] = PlaceholderDiseaseModel(
                    num_classes=len(self.disease_db.get_diseases_for_crop(crop)),
                    crop=crop
                )

        self._model_info = ModelInfo(
            name="disease_detector",
            version="0.1.0-placeholder",
            architecture="efficientnet_v2_s",
            input_size=(224, 224),
            num_classes=self.disease_db.num_classes,
            class_names=[d.name for d in self.disease_db.DISEASES],
            device=self.device
        )

        self._is_loaded = True
        logger.info(f"Disease detector loaded: {self.disease_db.num_classes} classes")

    def predict(
        self,
        input_data: np.ndarray,
        crop: Optional[str] = None
    ) -> PredictionResult[DiseasePrediction]:
        """
        Run disease detection.

        Args:
            input_data: Preprocessed image tensor
            crop: Optional crop name for routing to specific model

        Returns:
            PredictionResult containing DiseasePrediction
        """
        self.ensure_loaded()

        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, 0)

        start_time = time.perf_counter()

        # Select model based on crop
        if crop and crop.lower() in self._crop_models:
            model = self._crop_models[crop.lower()]
            logger.debug(f"Using crop-specific model for: {crop}")
        else:
            model = self.model

        # Get predictions
        predictions = model.forward(input_data)

        # Process binary prediction (healthy/diseased)
        binary_probs = predictions["binary_probs"][0]
        is_healthy = binary_probs[0] > binary_probs[1]
        health_confidence = float(max(binary_probs))

        # Process disease identification
        disease_probs = predictions["disease_probs"][0]
        top_disease_idx = int(np.argmax(disease_probs))
        disease_confidence = float(disease_probs[top_disease_idx])

        # Get disease info
        disease_info = self.disease_db.get_by_index(top_disease_idx)

        if disease_info is None:
            disease_info = DiseaseLabel(
                top_disease_idx, "Unknown Disease", "unknown", False,
                visual_symptoms=["Unable to determine specific symptoms"],
                severity_stages=["Unknown"]
            )

        # Build prediction
        if is_healthy or disease_confidence < self.confidence_threshold:
            disease_pred = DiseasePrediction(
                is_healthy=True,
                disease_name=None,
                confidence=health_confidence,
                visual_symptoms=[],
                affected_area=0.0,
                disease_stage=None
            )
        else:
            disease_pred = DiseasePrediction(
                is_healthy=False,
                disease_name=disease_info.name,
                confidence=disease_confidence,
                visual_symptoms=disease_info.visual_symptoms,
                affected_area=predictions.get("affected_area", [25.0])[0],
                disease_stage=self._estimate_severity(disease_confidence, disease_info)
            )

        latency = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            prediction=disease_pred,
            confidence=disease_pred.confidence,
            latency_ms=latency,
            model_version=self._model_info.version if self._model_info else "unknown",
            metadata={
                "crop_specific_model_used": crop is not None and crop.lower() in self._crop_models,
                "binary_healthy_prob": float(binary_probs[0]),
                "binary_diseased_prob": float(binary_probs[1]),
                "top_diseases": [
                    (self.disease_db.get_by_index(int(i)).name if self.disease_db.get_by_index(int(i)) else f"disease_{i}",
                     float(disease_probs[i]))
                    for i in np.argsort(disease_probs)[::-1][:3]
                ]
            }
        )

    def _estimate_severity(self, confidence: float, disease: DiseaseLabel) -> str:
        """
        Estimate disease severity based on confidence and visual analysis.

        Production: This should use segmentation model output
        to estimate actual affected area percentage.
        """
        stages = disease.severity_stages
        if len(stages) <= 1:
            return stages[0] if stages else "Unknown"

        # Simple heuristic: higher confidence often correlates with
        # more visible/severe symptoms
        if confidence >= 0.9:
            return stages[-1] if len(stages) > 1 else stages[0]
        elif confidence >= 0.8:
            return stages[len(stages) // 2]
        else:
            return stages[0]

    def get_model_info(self) -> ModelInfo:
        """Return model information."""
        self.ensure_loaded()
        return self._model_info

    def get_crop_specific_models(self) -> list[str]:
        """List available crop-specific models."""
        return list(self._crop_models.keys())

    def add_disease(self, disease: DiseaseLabel) -> None:
        """
        Add a new disease to the database.

        Iteration Point: This allows extending the system with new diseases
        without retraining the full model (requires fine-tuning).
        """
        self.disease_db.DISEASES.append(disease)
        logger.info(f"Added new disease: {disease.name} for crop: {disease.crop}")


class PlaceholderDiseaseModel:
    """
    Placeholder model for disease detection.

    Generates realistic predictions for testing.
    Replace with PyTorch model in production.
    """

    def __init__(self, num_classes: int, crop: Optional[str] = None):
        self.num_classes = num_classes
        self.crop = crop

    def forward(self, x: np.ndarray) -> dict:
        """
        Generate placeholder predictions.

        Production implementation would be:
        ```python
        with torch.no_grad():
            features = self.backbone(x)

            binary_logits = self.binary_head(features)
            disease_logits = self.disease_head(features)

            return {
                "binary_probs": F.softmax(binary_logits, dim=1),
                "disease_probs": F.softmax(disease_logits, dim=1),
                "features": features
            }
        ```
        """
        batch_size = x.shape[0]

        # Binary prediction (healthy/diseased)
        binary_logits = np.random.randn(batch_size, 2)
        # Slightly bias toward diseased for interesting demo
        binary_logits[:, 1] += 0.5
        binary_probs = self._softmax(binary_logits)

        # Disease classification
        disease_logits = np.random.randn(batch_size, self.num_classes)
        # Make one disease more prominent
        dominant_disease = np.random.randint(3, self.num_classes)  # Skip healthy classes
        disease_logits[:, dominant_disease] += 2.5
        disease_probs = self._softmax(disease_logits)

        return {
            "binary_probs": binary_probs,
            "disease_probs": disease_probs,
            "affected_area": np.random.uniform(10, 50, size=(batch_size,)),
            "features": np.random.randn(batch_size, 1280)
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
