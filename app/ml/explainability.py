"""
Explainability Module

Provides visual and textual explanations for model predictions using:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Saliency maps
- Feature attribution

This module does NOT use LLM for core predictions - only for
generating human-readable explanations of deterministic ML outputs.

Grad-CAM Implementation:
1. Forward pass to get feature maps and prediction
2. Backward pass to get gradients of target class
3. Global average pool gradients as weights
4. Weighted combination of feature maps
5. Apply ReLU and normalize

References:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
- https://arxiv.org/abs/1610.02391
"""

import logging
import base64
import io
from typing import Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.ml.base import HierarchicalPrediction, DiseasePrediction
from app.models.enums import ConfidenceLevel

logger = logging.getLogger(__name__)


@dataclass
class GradCAMOutput:
    """Grad-CAM analysis output."""
    heatmap: np.ndarray  # Shape: (H, W), values 0-1
    overlay_image: Optional[np.ndarray] = None  # RGB overlay
    focus_regions: list[str] = None
    activation_percentage: float = 0.0  # % of image with high activation

    def __post_init__(self):
        if self.focus_regions is None:
            self.focus_regions = []


@dataclass
class ExplainabilityResult:
    """Complete explainability analysis result."""
    model_reasoning: str
    confidence_notes: str
    key_features: list[str]
    uncertainty_factors: list[str]
    grad_cam: Optional[GradCAMOutput] = None
    heatmap_base64: Optional[str] = None


class ExplainabilityEngine:
    """
    Engine for generating model explanations.

    Combines multiple explanation techniques:
    1. Grad-CAM for visual attention maps
    2. Confidence analysis for uncertainty communication
    3. Feature-based reasoning for textual explanations

    This module translates model outputs into human-understandable
    explanations without changing the underlying predictions.

    Usage:
        engine = ExplainabilityEngine()
        result = engine.explain(
            species_prediction=species_result,
            disease_prediction=disease_result,
            image_tensor=preprocessed_image,
            model_features=features_from_model
        )
    """

    # Templates for generating explanations
    SPECIES_REASONING_TEMPLATES = {
        ConfidenceLevel.VERY_HIGH: [
            "The leaf shape, venation pattern, and overall morphology strongly match {species}.",
            "Distinctive features of {species} are clearly visible, including {features}.",
        ],
        ConfidenceLevel.HIGH: [
            "Visual characteristics are consistent with {species}, particularly {features}.",
            "The plant exhibits typical {species} features with minor variations.",
        ],
        ConfidenceLevel.MODERATE: [
            "The plant shows several features consistent with {species}, though some characteristics are ambiguous.",
            "Partial match to {species} based on {features}; additional angles may improve accuracy.",
        ],
        ConfidenceLevel.LOW: [
            "Limited visual evidence for {species} identification; consider providing clearer images.",
            "Some features suggest {species}, but confidence is low due to {uncertainty}.",
        ],
        ConfidenceLevel.VERY_LOW: [
            "Unable to confidently identify species. Image quality or plant visibility may be insufficient.",
            "Multiple species show similar characteristics; more detailed images needed.",
        ]
    }

    DISEASE_REASONING_TEMPLATES = {
        "diseased": [
            "Visual symptoms match {disease} patterns: {symptoms}.",
            "The {symptoms} are characteristic indicators of {disease}.",
            "Lesion morphology and distribution are consistent with {disease} infection.",
        ],
        "healthy": [
            "No visible signs of disease detected. Leaf coloration and structure appear normal.",
            "Plant shows healthy tissue without lesions, discoloration, or deformation.",
        ],
        "uncertain": [
            "Some potential symptoms observed, but insufficient for definitive diagnosis.",
            "Early-stage symptoms possible; recommend monitoring for progression.",
        ]
    }

    def __init__(self, enable_grad_cam: bool = True):
        """
        Initialize explainability engine.

        Args:
            enable_grad_cam: Whether to generate Grad-CAM visualizations
        """
        self.enable_grad_cam = enable_grad_cam

    def explain(
        self,
        species_prediction: HierarchicalPrediction,
        disease_prediction: DiseasePrediction,
        image_tensor: Optional[np.ndarray] = None,
        model_features: Optional[np.ndarray] = None,
        original_image: Optional[Image.Image] = None
    ) -> ExplainabilityResult:
        """
        Generate comprehensive explanation for predictions.

        Args:
            species_prediction: Species classification result
            disease_prediction: Disease detection result
            image_tensor: Preprocessed image tensor (for Grad-CAM)
            model_features: Feature vectors from model (for analysis)
            original_image: Original PIL image (for overlay generation)

        Returns:
            ExplainabilityResult with reasoning and visualizations
        """
        # Generate textual explanations
        model_reasoning = self._generate_reasoning(
            species_prediction, disease_prediction
        )

        confidence_notes = self._generate_confidence_notes(
            species_prediction, disease_prediction
        )

        key_features = self._extract_key_features(
            species_prediction, disease_prediction
        )

        uncertainty_factors = self._identify_uncertainty_factors(
            species_prediction, disease_prediction
        )

        # Generate Grad-CAM if enabled
        grad_cam_output = None
        heatmap_base64 = None

        if self.enable_grad_cam and image_tensor is not None:
            grad_cam_output = self._generate_grad_cam(
                image_tensor,
                model_features,
                target_class=None  # Use predicted class
            )

            if original_image is not None and grad_cam_output is not None:
                heatmap_base64 = self._create_overlay_image(
                    original_image, grad_cam_output.heatmap
                )

        return ExplainabilityResult(
            model_reasoning=model_reasoning,
            confidence_notes=confidence_notes,
            key_features=key_features,
            uncertainty_factors=uncertainty_factors,
            grad_cam=grad_cam_output,
            heatmap_base64=heatmap_base64
        )

    def _generate_reasoning(
        self,
        species_pred: HierarchicalPrediction,
        disease_pred: DiseasePrediction
    ) -> str:
        """Generate human-readable reasoning for the prediction."""
        parts = []

        # Species reasoning
        confidence_level = ConfidenceLevel.from_score(species_pred.species_confidence)
        templates = self.SPECIES_REASONING_TEMPLATES.get(
            confidence_level,
            self.SPECIES_REASONING_TEMPLATES[ConfidenceLevel.MODERATE]
        )

        # Select template and fill in values
        import random
        template = random.choice(templates)

        features = self._get_species_features(species_pred)
        uncertainty = self._get_uncertainty_description(species_pred)

        species_text = template.format(
            species=species_pred.common_name or species_pred.species,
            features=", ".join(features[:2]) if features else "general morphology",
            uncertainty=uncertainty
        )
        parts.append(species_text)

        # Disease reasoning
        if disease_pred.is_healthy:
            disease_templates = self.DISEASE_REASONING_TEMPLATES["healthy"]
        elif disease_pred.confidence >= 0.7:
            disease_templates = self.DISEASE_REASONING_TEMPLATES["diseased"]
        else:
            disease_templates = self.DISEASE_REASONING_TEMPLATES["uncertain"]

        template = random.choice(disease_templates)

        if disease_pred.disease_name:
            symptoms = disease_pred.visual_symptoms[:2] if disease_pred.visual_symptoms else ["observed lesions"]
            disease_text = template.format(
                disease=disease_pred.disease_name,
                symptoms=", ".join(symptoms)
            )
        else:
            disease_text = template

        parts.append(disease_text)

        return " ".join(parts)

    def _generate_confidence_notes(
        self,
        species_pred: HierarchicalPrediction,
        disease_pred: DiseasePrediction
    ) -> str:
        """Generate confidence context for users."""
        notes = []

        # Species confidence
        species_level = ConfidenceLevel.from_score(species_pred.species_confidence)
        if species_level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            notes.append(
                f"High confidence ({species_pred.species_confidence:.0%}) in species identification"
            )
        elif species_level == ConfidenceLevel.MODERATE:
            notes.append(
                f"Moderate confidence ({species_pred.species_confidence:.0%}) - "
                f"consider alternatives: {', '.join(a[0] for a in species_pred.alternatives[:2])}"
            )
        else:
            notes.append(
                f"Low confidence ({species_pred.species_confidence:.0%}) - "
                "recommend providing additional images from different angles"
            )

        # Disease confidence
        if not disease_pred.is_healthy:
            if disease_pred.confidence >= 0.85:
                notes.append(
                    f"Clear disease indicators detected ({disease_pred.confidence:.0%} confidence)"
                )
            elif disease_pred.confidence >= 0.6:
                notes.append(
                    f"Disease symptoms detected with moderate confidence ({disease_pred.confidence:.0%})"
                )
            else:
                notes.append(
                    "Possible disease symptoms observed but requires expert verification"
                )
        else:
            notes.append("No significant disease indicators detected")

        return "; ".join(notes)

    def _extract_key_features(
        self,
        species_pred: HierarchicalPrediction,
        disease_pred: DiseasePrediction
    ) -> list[str]:
        """Extract key visual features that influenced the prediction."""
        features = []

        # Species features
        species_features = self._get_species_features(species_pred)
        features.extend(species_features)

        # Disease features
        if disease_pred.visual_symptoms:
            features.extend(disease_pred.visual_symptoms[:3])

        return features

    def _get_species_features(self, pred: HierarchicalPrediction) -> list[str]:
        """Get characteristic features for a species."""
        # In production, this would come from a feature database
        feature_db = {
            "Solanum lycopersicum": [
                "compound pinnate leaves",
                "serrated leaflet margins",
                "distinctive plant odor",
                "hairy stems"
            ],
            "Solanum tuberosum": [
                "pinnately compound leaves",
                "white or purple flowers",
                "tuber formation"
            ],
            "Malus domestica": [
                "simple oval leaves",
                "serrated leaf margins",
                "branching pattern"
            ],
            "Zea mays": [
                "long linear leaves",
                "parallel leaf venation",
                "tall single stalk"
            ],
            "Vitis vinifera": [
                "palmate lobed leaves",
                "tendril presence",
                "climbing vine habit"
            ]
        }

        return feature_db.get(pred.species, ["leaf morphology", "plant structure"])

    def _get_uncertainty_description(self, pred: HierarchicalPrediction) -> str:
        """Describe sources of uncertainty."""
        if pred.species_confidence < 0.5:
            return "limited visible features and potential image quality issues"
        elif pred.species_confidence < 0.7:
            return "some overlapping characteristics with similar species"
        else:
            return "minor varietal differences"

    def _identify_uncertainty_factors(
        self,
        species_pred: HierarchicalPrediction,
        disease_pred: DiseasePrediction
    ) -> list[str]:
        """Identify factors contributing to prediction uncertainty."""
        factors = []

        # Species uncertainty
        if species_pred.species_confidence < 0.8:
            if species_pred.alternatives:
                similar = species_pred.alternatives[0]
                if similar[1] > 0.3:  # Alternative has significant probability
                    factors.append(
                        f"Similar appearance to {similar[0]} ({similar[1]:.0%})"
                    )

        if species_pred.family_confidence > species_pred.species_confidence + 0.1:
            factors.append(
                "Clearer at family level than species level"
            )

        # Disease uncertainty
        if not disease_pred.is_healthy and disease_pred.confidence < 0.8:
            factors.append("Disease symptoms may be in early stages")

        if disease_pred.affected_area and disease_pred.affected_area < 20:
            factors.append("Limited affected area visible for analysis")

        if not factors:
            factors.append("No significant uncertainty factors identified")

        return factors

    def _generate_grad_cam(
        self,
        image_tensor: np.ndarray,
        model_features: Optional[np.ndarray],
        target_class: Optional[int] = None
    ) -> GradCAMOutput:
        """
        Generate Grad-CAM heatmap.

        Production Implementation (PyTorch):
        ```python
        # Register hooks on target layer
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        output = model(image_tensor)
        target_score = output[0, target_class]

        # Backward pass
        model.zero_grad()
        target_score.backward()

        # Compute Grad-CAM
        activation = activations[0]
        gradient = gradients[0]

        # Global average pooling of gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / cam.max()

        # Resize to input size
        cam = F.interpolate(cam, size=image_tensor.shape[2:], mode='bilinear')
        return cam.squeeze().cpu().numpy()
        ```
        """
        # Placeholder: Generate synthetic heatmap for demonstration
        h, w = 224, 224

        # Create a realistic-looking heatmap
        # In production, this comes from actual gradient computation
        heatmap = self._generate_placeholder_heatmap(h, w)

        # Identify focus regions
        focus_regions = self._analyze_heatmap_regions(heatmap)

        # Calculate activation percentage
        threshold = 0.5
        activation_pct = np.mean(heatmap > threshold) * 100

        return GradCAMOutput(
            heatmap=heatmap,
            focus_regions=focus_regions,
            activation_percentage=activation_pct
        )

    def _generate_placeholder_heatmap(self, h: int, w: int) -> np.ndarray:
        """Generate a placeholder heatmap for demonstration."""
        # Create base noise
        heatmap = np.random.rand(h, w) * 0.3

        # Add some focused regions (simulating leaf/lesion focus)
        # Central region - typical focus for leaf images
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2

        # Main focus area
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        sigma = min(h, w) / 3
        central_focus = np.exp(-(dist_from_center ** 2) / (2 * sigma ** 2))
        heatmap += central_focus * 0.5

        # Add some off-center hotspots (simulating lesions)
        for _ in range(2):
            hot_y = np.random.randint(h // 4, 3 * h // 4)
            hot_x = np.random.randint(w // 4, 3 * w // 4)
            dist = np.sqrt((x - hot_x) ** 2 + (y - hot_y) ** 2)
            hotspot = np.exp(-(dist ** 2) / (2 * (sigma / 3) ** 2))
            heatmap += hotspot * 0.3

        # Normalize to 0-1
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def _analyze_heatmap_regions(self, heatmap: np.ndarray) -> list[str]:
        """Analyze heatmap to describe focus regions."""
        h, w = heatmap.shape
        regions = []

        # Divide into 3x3 grid and find high-activation regions
        grid_h, grid_w = h // 3, w // 3

        region_names = [
            ["upper-left", "upper-center", "upper-right"],
            ["middle-left", "center", "middle-right"],
            ["lower-left", "lower-center", "lower-right"]
        ]

        for i in range(3):
            for j in range(3):
                region = heatmap[
                    i * grid_h:(i + 1) * grid_h,
                    j * grid_w:(j + 1) * grid_w
                ]
                mean_activation = np.mean(region)

                if mean_activation > 0.5:
                    regions.append(f"High attention on {region_names[i][j]} region of leaf")

        if not regions:
            regions.append("Attention distributed across leaf surface")

        return regions[:3]  # Limit to top 3

    def _create_overlay_image(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray
    ) -> str:
        """
        Create heatmap overlay on original image.

        Returns base64-encoded PNG.
        """
        # Resize heatmap to match image
        img_w, img_h = original_image.size
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.Resampling.BILINEAR
            )
        )

        # Create colormap (blue -> red)
        colored_heatmap = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        colored_heatmap[:, :, 0] = heatmap_resized  # Red channel
        colored_heatmap[:, :, 2] = 255 - heatmap_resized  # Blue channel

        # Convert original to RGB array
        original_array = np.array(original_image.convert("RGB"))

        # Blend
        alpha = 0.4
        overlay = (
            (1 - alpha) * original_array + alpha * colored_heatmap
        ).astype(np.uint8)

        # Convert to base64
        overlay_image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_image.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")


class HumanInTheLoopFeedback:
    """
    System for collecting and incorporating human feedback.

    This enables continuous model improvement through:
    1. Expert corrections of misclassifications
    2. Confidence calibration from user feedback
    3. Active learning sample selection

    Iteration Point: Use feedback to improve models over time.

    Workflow:
    1. User flags incorrect prediction
    2. Expert reviews and provides correct label
    3. Feedback stored for retraining
    4. Model fine-tuned on corrected examples
    5. Confidence thresholds adjusted based on error patterns
    """

    def __init__(self):
        self.feedback_store: list[dict] = []

    def record_feedback(
        self,
        image_id: str,
        predicted_species: str,
        predicted_disease: str,
        correct_species: Optional[str] = None,
        correct_disease: Optional[str] = None,
        expert_notes: Optional[str] = None,
        feedback_type: str = "correction"
    ) -> dict:
        """
        Record user/expert feedback on a prediction.

        Args:
            image_id: Unique identifier for the image
            predicted_species: Model's species prediction
            predicted_disease: Model's disease prediction
            correct_species: Expert-provided correct species
            correct_disease: Expert-provided correct disease
            expert_notes: Additional notes from expert
            feedback_type: Type of feedback (correction, confirmation, uncertain)

        Returns:
            Feedback record
        """
        import time

        feedback = {
            "image_id": image_id,
            "timestamp": time.time(),
            "feedback_type": feedback_type,
            "predicted": {
                "species": predicted_species,
                "disease": predicted_disease
            },
            "corrected": {
                "species": correct_species,
                "disease": correct_disease
            },
            "expert_notes": expert_notes,
            "processed": False
        }

        self.feedback_store.append(feedback)

        logger.info(
            f"Recorded feedback for {image_id}: "
            f"{predicted_species} -> {correct_species or 'confirmed'}"
        )

        return feedback

    def get_retraining_samples(
        self,
        min_samples: int = 100,
        include_confirmations: bool = False
    ) -> list[dict]:
        """
        Get samples ready for model retraining.

        Returns corrections and optionally confirmations for
        active learning and model fine-tuning.
        """
        samples = []

        for feedback in self.feedback_store:
            if feedback["processed"]:
                continue

            if feedback["feedback_type"] == "correction":
                samples.append(feedback)
            elif include_confirmations and feedback["feedback_type"] == "confirmation":
                samples.append(feedback)

        if len(samples) >= min_samples:
            logger.info(f"Ready for retraining with {len(samples)} samples")

        return samples

    def get_error_analysis(self) -> dict:
        """
        Analyze prediction errors for model improvement.

        Returns patterns in misclassifications to guide
        model architecture or training data improvements.
        """
        corrections = [f for f in self.feedback_store if f["feedback_type"] == "correction"]

        if not corrections:
            return {"message": "No corrections recorded yet"}

        # Analyze common confusions
        confusion_pairs = {}
        for c in corrections:
            pred = c["predicted"]["species"]
            actual = c["corrected"]["species"]
            if actual:
                pair = (pred, actual)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        return {
            "total_corrections": len(corrections),
            "common_confusions": sorted(
                confusion_pairs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "recommendations": self._generate_improvement_recommendations(confusion_pairs)
        }

    def _generate_improvement_recommendations(self, confusions: dict) -> list[str]:
        """Generate recommendations based on error patterns."""
        recommendations = []

        if len(confusions) > 10:
            recommendations.append(
                "High confusion rate suggests need for more training data diversity"
            )

        # Check for specific patterns
        # (In production, this would be more sophisticated)
        recommendations.append(
            "Consider augmenting training data with edge cases from feedback"
        )
        recommendations.append(
            "Review confidence calibration for frequently confused classes"
        )

        return recommendations
