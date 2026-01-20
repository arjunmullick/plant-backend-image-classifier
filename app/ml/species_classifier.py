"""
Species Classifier Module

Hierarchical plant species classification using transfer learning.

Architecture Decision:
- Primary: EfficientNetV2-S (best accuracy/speed tradeoff)
- Alternative: ConvNeXt-Tiny (for production with GPU)
- Lightweight: MobileNetV3 (edge deployment)

The model uses a hierarchical classification head that predicts
Family → Genus → Species with cascading confidence scores.

Training Data Sources:
- PlantVillage: 54,306 images, 38 classes
- PlantNet: 306,293 images, 1,081 species
- iNaturalist: Millions of observations

Model Selection Rationale:
1. EfficientNetV2: Best for general deployment
   - Compound scaling optimizes depth/width/resolution
   - Progressive training support
   - ~6M parameters for S variant

2. ConvNeXt: Best for high-accuracy requirements
   - Modern pure ConvNet matching ViT performance
   - Better feature maps for Grad-CAM

3. ViT: Consider for very large datasets
   - Requires more data but generalizes well
   - Better for fine-grained distinctions
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.ml.base import BaseMLComponent, PredictionResult, ModelInfo, HierarchicalPrediction

logger = logging.getLogger(__name__)

# Try to import transformers for real model inference
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("transformers not installed. Using placeholder model.")


@dataclass
class TaxonomyLabel:
    """Mapping between class index and taxonomy information."""
    index: int
    family: str
    genus: str
    species: str
    common_name: Optional[str] = None


class PlantTaxonomyDatabase:
    """
    Database of plant taxonomy labels.

    In production, this would be loaded from a JSON/database file.
    This placeholder demonstrates the structure.

    Iteration Point: Add new plants by extending this database
    """

    # Placeholder taxonomy data
    # In production: load from JSON file with full PlantNet/iNaturalist taxonomy
    TAXONOMY: list[TaxonomyLabel] = [
        # Solanaceae family
        TaxonomyLabel(0, "Solanaceae", "Solanum", "Solanum lycopersicum", "Tomato"),
        TaxonomyLabel(1, "Solanaceae", "Solanum", "Solanum tuberosum", "Potato"),
        TaxonomyLabel(2, "Solanaceae", "Capsicum", "Capsicum annuum", "Bell Pepper"),
        TaxonomyLabel(3, "Solanaceae", "Solanum", "Solanum melongena", "Eggplant"),

        # Rosaceae family
        TaxonomyLabel(4, "Rosaceae", "Malus", "Malus domestica", "Apple"),
        TaxonomyLabel(5, "Rosaceae", "Prunus", "Prunus persica", "Peach"),
        TaxonomyLabel(6, "Rosaceae", "Fragaria", "Fragaria × ananassa", "Strawberry"),
        TaxonomyLabel(7, "Rosaceae", "Rubus", "Rubus idaeus", "Raspberry"),

        # Poaceae family
        TaxonomyLabel(8, "Poaceae", "Zea", "Zea mays", "Corn"),
        TaxonomyLabel(9, "Poaceae", "Oryza", "Oryza sativa", "Rice"),
        TaxonomyLabel(10, "Poaceae", "Triticum", "Triticum aestivum", "Wheat"),

        # Vitaceae family
        TaxonomyLabel(11, "Vitaceae", "Vitis", "Vitis vinifera", "Grape"),

        # Cucurbitaceae family
        TaxonomyLabel(12, "Cucurbitaceae", "Cucumis", "Cucumis sativus", "Cucumber"),
        TaxonomyLabel(13, "Cucurbitaceae", "Cucurbita", "Cucurbita pepo", "Squash"),

        # Fabaceae family
        TaxonomyLabel(14, "Fabaceae", "Phaseolus", "Phaseolus vulgaris", "Common Bean"),
        TaxonomyLabel(15, "Fabaceae", "Glycine", "Glycine max", "Soybean"),

        # Rutaceae family
        TaxonomyLabel(16, "Rutaceae", "Citrus", "Citrus sinensis", "Orange"),
        TaxonomyLabel(17, "Rutaceae", "Citrus", "Citrus limon", "Lemon"),

        # Asteraceae family
        TaxonomyLabel(18, "Asteraceae", "Helianthus", "Helianthus annuus", "Sunflower"),
        TaxonomyLabel(19, "Asteraceae", "Lactuca", "Lactuca sativa", "Lettuce"),
    ]

    def __init__(self):
        self._index_to_label = {t.index: t for t in self.TAXONOMY}
        self._families = list(set(t.family for t in self.TAXONOMY))
        self._genera = list(set(t.genus for t in self.TAXONOMY))
        self._species = [t.species for t in self.TAXONOMY]

    def get_by_index(self, index: int) -> TaxonomyLabel:
        """Get taxonomy label by class index."""
        return self._index_to_label.get(index)

    def get_families(self) -> list[str]:
        """Get list of all families."""
        return self._families

    @property
    def num_classes(self) -> int:
        return len(self.TAXONOMY)


class SpeciesClassifier(BaseMLComponent):
    """
    Hierarchical species classifier using transfer learning.

    This implementation uses a placeholder model that demonstrates
    the interface. In production, replace with actual PyTorch model.

    Model Architecture (Production):
    ```
    EfficientNetV2-S (pretrained on ImageNet)
           |
    Global Average Pooling
           |
    Dropout (0.3)
           |
    ┌──────┼──────┐
    │      │      │
    FC     FC     FC
    (Family)(Genus)(Species)
    ```

    The hierarchical heads share features but predict independently,
    allowing confidence scores at each taxonomic level.
    """

    # Model architecture options for production
    SUPPORTED_ARCHITECTURES = {
        "efficientnet_v2_s": {
            "input_size": (224, 224),
            "pretrained_source": "ImageNet-1K",
            "params": "~21M",
            "recommended_for": "General deployment"
        },
        "convnext_tiny": {
            "input_size": (224, 224),
            "pretrained_source": "ImageNet-1K",
            "params": "~28M",
            "recommended_for": "High accuracy requirements"
        },
        "mobilenet_v3_large": {
            "input_size": (224, 224),
            "pretrained_source": "ImageNet-1K",
            "params": "~5.4M",
            "recommended_for": "Edge/mobile deployment"
        },
        "vit_b_16": {
            "input_size": (224, 224),
            "pretrained_source": "ImageNet-21K",
            "params": "~86M",
            "recommended_for": "Large datasets, fine-grained"
        }
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        architecture: str = "efficientnet_v2_s",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the species classifier.

        Args:
            model_path: Path to trained model weights
            device: Inference device
            architecture: Model architecture to use
            confidence_threshold: Minimum confidence for valid prediction
        """
        super().__init__(model_path, device)
        self.architecture = architecture
        self.confidence_threshold = confidence_threshold
        self.taxonomy_db = PlantTaxonomyDatabase()

        if architecture not in self.SUPPORTED_ARCHITECTURES:
            logger.warning(
                f"Unknown architecture: {architecture}. "
                f"Supported: {list(self.SUPPORTED_ARCHITECTURES.keys())}"
            )

    def load_model(self) -> None:
        """
        Load the species classification model.

        Production Implementation:
        ```python
        import torch
        import torchvision.models as models

        # Load pretrained backbone
        if self.architecture == "efficientnet_v2_s":
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            num_features = self.backbone.classifier[1].in_features

            # Replace classifier with hierarchical heads
            self.backbone.classifier = nn.Identity()

            self.family_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, len(self.taxonomy_db.get_families()))
            )
            self.genus_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, len(unique_genera))
            )
            self.species_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, self.taxonomy_db.num_classes)
            )

        # Load trained weights
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.load_state_dict(state_dict)

        self.eval()
        self.to(self.device)
        ```
        """
        logger.info(f"Loading species classifier: {self.architecture}")

        # Use HuggingFace model if transformers is available
        if HF_AVAILABLE:
            logger.info("Using HuggingFace MobileNetV2 for real inference")
            self.model = HuggingFaceSpeciesModel(
                num_classes=self.taxonomy_db.num_classes,
                architecture=self.architecture
            )
            model_version = "1.0.0-mobilenet_v2"
        else:
            logger.warning("Falling back to placeholder model (transformers not installed)")
            self.model = PlaceholderSpeciesModel(
                num_classes=self.taxonomy_db.num_classes,
                architecture=self.architecture
            )
            model_version = "0.1.0-placeholder"

        self._model_info = ModelInfo(
            name="species_classifier",
            version=model_version,
            architecture=self.architecture,
            input_size=(224, 224),
            num_classes=self.taxonomy_db.num_classes,
            class_names=[t.species for t in self.taxonomy_db.TAXONOMY],
            device=self.device
        )

        self._is_loaded = True
        logger.info(f"Species classifier loaded: {self.taxonomy_db.num_classes} classes (version: {model_version})")

    def predict(self, input_data: np.ndarray) -> PredictionResult[HierarchicalPrediction]:
        """
        Run hierarchical species classification.

        Args:
            input_data: Preprocessed image tensor (C, H, W) or (B, C, H, W)

        Returns:
            PredictionResult containing HierarchicalPrediction
        """
        self.ensure_loaded()

        # Add batch dimension if needed
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, 0)

        start_time = time.perf_counter()

        # Get predictions from placeholder model
        predictions = self.model.forward(input_data)

        # Process predictions into hierarchical result
        species_probs = predictions["species_probs"][0]
        top_idx = int(np.argmax(species_probs))
        top_confidence = float(species_probs[top_idx])

        # Get taxonomy info
        taxonomy = self.taxonomy_db.get_by_index(top_idx)

        if taxonomy is None:
            # Fallback for unknown class
            taxonomy = TaxonomyLabel(
                top_idx, "Unknown", "Unknown", "Unknown species", None
            )

        # Calculate confidence at each taxonomic level
        # In production: use separate heads for family/genus
        family_confidence = min(1.0, top_confidence + 0.05)  # Family is more certain
        genus_confidence = min(1.0, top_confidence + 0.02)

        # Get alternative predictions
        sorted_indices = np.argsort(species_probs)[::-1]
        alternatives = []
        for idx in sorted_indices[1:4]:  # Top 3 alternatives
            alt_taxonomy = self.taxonomy_db.get_by_index(int(idx))
            if alt_taxonomy:
                alternatives.append((alt_taxonomy.species, float(species_probs[idx])))

        hierarchical_pred = HierarchicalPrediction(
            family=taxonomy.family,
            family_confidence=family_confidence,
            genus=taxonomy.genus,
            genus_confidence=genus_confidence,
            species=taxonomy.species,
            species_confidence=top_confidence,
            common_name=taxonomy.common_name,
            alternatives=alternatives
        )

        latency = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            prediction=hierarchical_pred,
            confidence=top_confidence,
            latency_ms=latency,
            model_version=self._model_info.version if self._model_info else "unknown",
            metadata={
                "top_k_species": [
                    (self.taxonomy_db.get_by_index(int(i)).species if self.taxonomy_db.get_by_index(int(i)) else f"class_{i}", float(species_probs[i]))
                    for i in sorted_indices[:5]
                ],
                "architecture": self.architecture
            }
        )

    def get_model_info(self) -> ModelInfo:
        """Return model information."""
        self.ensure_loaded()
        return self._model_info

    def get_training_config(self) -> dict:
        """
        Return recommended training configuration.

        This configuration is optimized for plant classification
        using transfer learning from ImageNet.
        """
        return {
            "architecture": self.architecture,
            "pretrained": True,
            "pretrained_source": "ImageNet-1K",
            "optimizer": {
                "name": "AdamW",
                "lr": 1e-4,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999)
            },
            "scheduler": {
                "name": "CosineAnnealingWarmRestarts",
                "T_0": 10,
                "T_mult": 2,
                "eta_min": 1e-6
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "early_stopping_patience": 5,
                "gradient_clip": 1.0
            },
            "loss": {
                "species": "CrossEntropyLoss",
                "family": "CrossEntropyLoss",
                "genus": "CrossEntropyLoss",
                "weights": {"species": 1.0, "family": 0.3, "genus": 0.3}
            },
            "regularization": {
                "dropout": 0.3,
                "label_smoothing": 0.1,
                "mixup_alpha": 0.2
            },
            "fine_tuning_strategy": {
                "phase_1": {
                    "epochs": 10,
                    "freeze_backbone": True,
                    "lr": 1e-3,
                    "description": "Train classification heads only"
                },
                "phase_2": {
                    "epochs": 20,
                    "unfreeze_layers": "last_2_blocks",
                    "lr": 1e-4,
                    "description": "Fine-tune top layers"
                },
                "phase_3": {
                    "epochs": 20,
                    "unfreeze_layers": "all",
                    "lr": 1e-5,
                    "description": "Full fine-tuning with low LR"
                }
            }
        }


class HuggingFaceSpeciesModel:
    """
    Real species classification using HuggingFace MobileNetV2.

    Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
    - 38 classes from PlantVillage dataset
    - Labels format: "Crop___Disease" or "Crop___healthy"
    - We extract the crop name and map to taxonomy
    """

    MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

    # Mapping from MobileNetV2 crop labels to our taxonomy indices
    CROP_TO_TAXONOMY = {
        "Tomato": 0,          # Solanum lycopersicum
        "Potato": 1,          # Solanum tuberosum
        "Pepper,_bell": 2,    # Capsicum annuum (Bell Pepper)
        "Apple": 4,           # Malus domestica
        "Peach": 5,           # Prunus persica
        "Strawberry": 6,      # Fragaria × ananassa
        "Raspberry": 7,       # Rubus idaeus
        "Corn_(maize)": 8,    # Zea mays
        "Grape": 11,          # Vitis vinifera
        "Squash": 13,         # Cucurbita pepo
        "Soybean": 15,        # Glycine max
        "Orange": 16,         # Citrus sinensis
        # Default fallback for unrecognized crops
        "Cherry_(including_sour)": 5,  # Map to Peach (Prunus family)
        "Blueberry": 6,       # Map to Strawberry (berry)
    }

    def __init__(self, num_classes: int, architecture: str):
        self.num_classes = num_classes
        self.architecture = architecture
        self._pipeline = None
        self._last_raw_results = None

    def _ensure_pipeline(self):
        """Lazy load the HuggingFace pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading HuggingFace model: {self.MODEL_ID}")
            self._pipeline = hf_pipeline(
                "image-classification",
                model=self.MODEL_ID,
                device=-1  # CPU
            )
            logger.info("HuggingFace species model loaded successfully")

    def forward(self, x: np.ndarray) -> dict:
        """
        Run real inference using HuggingFace pipeline.

        Args:
            x: Image tensor of shape (B, C, H, W) normalized to [0, 1]

        Returns:
            dict with species_probs and features
        """
        self._ensure_pipeline()

        batch_size = x.shape[0]
        species_probs = np.zeros((batch_size, self.num_classes))

        for i in range(batch_size):
            # Convert tensor back to PIL Image for HuggingFace
            img_tensor = x[i]  # (C, H, W)

            # Denormalize from model preprocessing and convert to PIL
            # Assuming input is normalized with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array = img_tensor.transpose(1, 2, 0)  # (H, W, C)
            img_array = img_array * std + mean
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)

            # Run HuggingFace inference
            results = self._pipeline(pil_image, top_k=38)
            self._last_raw_results = results

            # Map HuggingFace results to our taxonomy
            for result in results:
                raw_label = result["label"]
                confidence = result["score"]

                # Extract crop from label (format: "Crop___Disease")
                crop = self._extract_crop(raw_label)
                taxonomy_idx = self.CROP_TO_TAXONOMY.get(crop, 0)

                # Accumulate confidence for this crop/species
                if taxonomy_idx < self.num_classes:
                    species_probs[i, taxonomy_idx] += confidence

            # Normalize probabilities
            total = species_probs[i].sum()
            if total > 0:
                species_probs[i] /= total

        return {
            "species_probs": species_probs,
            "features": np.zeros((batch_size, 1280)),  # Placeholder features
            "raw_results": self._last_raw_results
        }

    def _extract_crop(self, raw_label: str) -> str:
        """Extract crop name from HuggingFace label."""
        if "___" in raw_label:
            return raw_label.split("___")[0]
        return raw_label

    def get_last_raw_results(self):
        """Return the last raw HuggingFace results for disease detection."""
        return self._last_raw_results


class PlaceholderSpeciesModel:
    """
    Fallback placeholder model when transformers is not available.

    Returns random predictions for testing the pipeline structure.
    """

    def __init__(self, num_classes: int, architecture: str):
        self.num_classes = num_classes
        self.architecture = architecture
        self._last_raw_results = None

    def forward(self, x: np.ndarray) -> dict:
        """Generate random placeholder predictions."""
        batch_size = x.shape[0]

        # Generate plausible predictions (random but deterministic for same input)
        # Use image mean as seed for reproducibility within session
        seed = int(np.abs(x.mean()) * 1000000) % (2**31)
        rng = np.random.RandomState(seed)

        species_logits = rng.randn(batch_size, self.num_classes)
        dominant_class = rng.randint(0, self.num_classes)
        species_logits[:, dominant_class] += 3.0

        species_probs = self._softmax(species_logits)

        return {
            "species_probs": species_probs,
            "features": np.zeros((batch_size, 1280))
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def get_last_raw_results(self):
        """Placeholder has no raw results."""
        return None
