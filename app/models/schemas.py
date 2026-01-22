"""
Pydantic schemas for API request/response validation.

These schemas define the contract between the API and clients,
ensuring type safety and automatic documentation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import base64


# === Request Schemas ===

class ClassificationRequest(BaseModel):
    """
    Request schema for plant image classification.

    Attributes:
        image: Base64-encoded image data (JPEG, PNG supported)
        region: Optional region code for localized recommendations (e.g., "US-CA")
        include_treatment: Whether to include treatment recommendations
        include_explainability: Whether to include model explainability info
    """
    image: str = Field(
        ...,
        description="Base64-encoded image data",
        min_length=100  # Minimum reasonable base64 image size
    )
    region: Optional[str] = Field(
        default=None,
        description="Region code for localized recommendations (e.g., 'US-CA', 'IN-MH')",
        pattern=r"^[A-Z]{2}(-[A-Z0-9]{1,4})?$"
    )
    include_treatment: bool = Field(
        default=True,
        description="Include treatment recommendations in response"
    )
    include_explainability: bool = Field(
        default=True,
        description="Include model explainability information"
    )

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the image is valid base64."""
        # Remove data URL prefix if present
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            decoded = base64.b64decode(v)
            if len(decoded) < 100:
                raise ValueError("Image data too small")
            if len(decoded) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Image exceeds 10MB limit")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")


# === Taxonomy Schemas ===

class TaxonomicLevel(BaseModel):
    """Single level in taxonomic hierarchy with confidence."""
    name: str = Field(..., description="Taxonomic name at this level")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0-1)"
    )
    common_name: Optional[str] = Field(
        default=None,
        description="Common name if available"
    )


class PlantIdentification(BaseModel):
    """
    Hierarchical plant identification result.

    Follows taxonomic hierarchy: Family → Genus → Species
    with confidence scores at each level.
    """
    family: TaxonomicLevel = Field(..., description="Plant family classification")
    genus: TaxonomicLevel = Field(..., description="Plant genus classification")
    species: TaxonomicLevel = Field(..., description="Plant species classification")

    # Optional extended taxonomy
    common_name: Optional[str] = Field(
        default=None,
        description="Most common name for this plant"
    )
    alternative_species: Optional[list[TaxonomicLevel]] = Field(
        default=None,
        description="Alternative species matches ranked by confidence"
    )


# === Health Assessment Schemas ===

class VisualSymptom(BaseModel):
    """Detailed visual symptom description with location."""
    description: str = Field(
        ...,
        description="Human-readable symptom description"
    )
    location: Optional[str] = Field(
        default=None,
        description="Location on plant (e.g., 'older leaves', 'stem base')"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Symptom severity: mild, moderate, severe"
    )


class HealthAssessment(BaseModel):
    """
    Plant health assessment including disease detection.
    """
    status: str = Field(
        ...,
        description="Overall health status: 'Healthy' or 'Diseased'"
    )
    disease: Optional[str] = Field(
        default=None,
        description="Detected disease name if status is 'Diseased'"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the health assessment"
    )
    visual_symptoms: list[str] = Field(
        default_factory=list,
        description="List of observed visual symptoms"
    )
    detailed_symptoms: Optional[list[VisualSymptom]] = Field(
        default=None,
        description="Detailed symptom descriptions with metadata"
    )
    disease_stage: Optional[str] = Field(
        default=None,
        description="Estimated disease progression stage"
    )
    affected_area_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Estimated percentage of affected plant area"
    )


# === Treatment Schemas ===

class TreatmentRecommendation(BaseModel):
    """
    Treatment recommendations organized by type.
    """
    organic: list[str] = Field(
        default_factory=list,
        description="Organic/natural treatment options"
    )
    chemical: list[str] = Field(
        default_factory=list,
        description="Chemical treatment options with application notes"
    )
    prevention: list[str] = Field(
        default_factory=list,
        description="Preventive care recommendations"
    )
    biological: Optional[list[str]] = Field(
        default=None,
        description="Biological control options"
    )
    cultural: Optional[list[str]] = Field(
        default=None,
        description="Cultural practice recommendations"
    )
    urgency: Optional[str] = Field(
        default=None,
        description="Treatment urgency: immediate, soon, routine"
    )
    region_specific_notes: Optional[str] = Field(
        default=None,
        description="Region-specific treatment considerations"
    )


# === Explainability Schemas ===

class GradCAMResult(BaseModel):
    """Grad-CAM visualization result."""
    heatmap_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded heatmap overlay image"
    )
    focus_regions: list[str] = Field(
        default_factory=list,
        description="Description of key focus regions"
    )


class ExplainabilityInfo(BaseModel):
    """
    Model explainability information.

    Provides insights into why the model made its predictions,
    helping users understand and verify the results.
    """
    model_reasoning: str = Field(
        ...,
        description="Human-readable explanation of model decision"
    )
    confidence_notes: str = Field(
        ...,
        description="Context about confidence level"
    )
    key_features: Optional[list[str]] = Field(
        default=None,
        description="Key visual features that influenced the prediction"
    )
    similar_cases: Optional[list[str]] = Field(
        default=None,
        description="Similar cases from training data"
    )
    grad_cam: Optional[GradCAMResult] = Field(
        default=None,
        description="Grad-CAM visualization if enabled"
    )
    uncertainty_factors: Optional[list[str]] = Field(
        default=None,
        description="Factors contributing to model uncertainty"
    )


# === Response Schemas ===

class ClassificationResponse(BaseModel):
    """
    Complete classification response.

    Contains plant identification, health assessment,
    treatment recommendations, and explainability info.
    """
    plant: PlantIdentification = Field(
        ...,
        description="Hierarchical plant identification"
    )
    health: HealthAssessment = Field(
        ...,
        description="Health status and disease detection"
    )
    treatment: Optional[TreatmentRecommendation] = Field(
        default=None,
        description="Treatment recommendations if applicable"
    )
    explainability: Optional[ExplainabilityInfo] = Field(
        default=None,
        description="Model explainability information"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Additional metadata (processing time, model versions, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "plant": {
                    "family": {"name": "Solanaceae", "confidence": 0.94},
                    "genus": {"name": "Solanum", "confidence": 0.91},
                    "species": {"name": "Solanum lycopersicum", "confidence": 0.89},
                    "common_name": "Tomato"
                },
                "health": {
                    "status": "Diseased",
                    "disease": "Early Blight",
                    "confidence": 0.92,
                    "visual_symptoms": [
                        "brown concentric rings on older leaves",
                        "yellowing around lesions"
                    ]
                },
                "treatment": {
                    "organic": ["Neem oil spray", "Remove infected leaves"],
                    "chemical": ["Chlorothalonil (follow label instructions)"],
                    "prevention": ["Avoid overhead watering", "Rotate crops"]
                },
                "explainability": {
                    "model_reasoning": "Lesion pattern and leaf discoloration matched Early Blight training examples",
                    "confidence_notes": "High confidence due to clear visual markers"
                }
            }
        }


# === Error Schemas ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")


class ImageQualityWarning(BaseModel):
    """Warning about image quality issues."""
    quality_score: float = Field(..., ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    recommendation: Optional[str] = None


# === Model Comparison Schemas ===

class ComparisonRequest(BaseModel):
    """
    Request for model comparison.

    Allows comparing our internal model with external open-source models.
    """
    image: str = Field(
        ...,
        description="Base64-encoded image data",
        min_length=100
    )
    models: list[str] = Field(
        default=["internal", "mobilenet_v2", "vit_crop"],
        description="Models to compare: internal, mobilenet_v2, vit_crop, plantnet, kindwise, resnet50_plant, efficientnet_plant"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence scores in comparison"
    )
    plantnet_api_key: Optional[str] = Field(
        default=None,
        description="PlantNet API key (optional, overrides environment variable)"
    )
    kindwise_api_key: Optional[str] = Field(
        default=None,
        description="Kindwise/Plant.id API key (optional, overrides environment variable)"
    )

    @field_validator("image")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that the image is valid base64."""
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            decoded = base64.b64decode(v)
            if len(decoded) < 100:
                raise ValueError("Image data too small")
            if len(decoded) > 10 * 1024 * 1024:
                raise ValueError("Image exceeds 10MB limit")
            return v
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")


class ExternalModelPrediction(BaseModel):
    """Prediction from an external model."""
    model_name: str = Field(..., description="Human-readable model name")
    model_type: str = Field(..., description="Model type identifier")
    prediction: Optional[str] = Field(None, description="Predicted class/disease")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    raw_label: Optional[str] = Field(None, description="Raw label from model")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    additional_info: Optional[dict] = Field(None, description="Additional model-specific info")


class ComparisonResponse(BaseModel):
    """
    Response for model comparison.

    Contains predictions from multiple models for side-by-side comparison.
    """
    internal: Optional[dict] = Field(
        None,
        description="Prediction from our internal model"
    )
    external_models: dict[str, ExternalModelPrediction] = Field(
        default_factory=dict,
        description="Predictions from external models"
    )
    agreement_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Agreement score between models (0-1)"
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommendation based on model agreement"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Comparison metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "internal": {
                    "disease": "Early Blight",
                    "confidence": 0.92,
                    "species": "Solanum lycopersicum"
                },
                "external_models": {
                    "mobilenet_v2": {
                        "model_name": "MobileNetV2 Plant Disease (HF)",
                        "model_type": "mobilenet_v2",
                        "prediction": "Early Blight",
                        "confidence": 0.89,
                        "processing_time_ms": 78
                    },
                    "vit_crop": {
                        "model_name": "ViT Crop Diseases (HF)",
                        "model_type": "vit_crop",
                        "prediction": "Not supported",
                        "confidence": None,
                        "processing_time_ms": 0,
                        "error": "Tomato not in supported crops"
                    }
                },
                "agreement_score": 0.85,
                "recommendation": "High confidence - models agree on disease identification"
            }
        }


class AvailableModelsResponse(BaseModel):
    """Response listing available models for comparison."""
    internal_model: dict = Field(..., description="Internal model info")
    external_models: list[dict] = Field(..., description="Available external models")
