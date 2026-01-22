"""
AI Crop Disease Early Warning Service

Provides comprehensive disease analysis by:
1. Running ALL available models in parallel
2. Aggregating results with confidence-weighted voting
3. Determining disease severity
4. Generating localized treatment recommendations
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image

from app.ml.external_models import (
    get_external_model_registry,
    ExternalModelType,
    ExternalModelResult,
)

logger = logging.getLogger(__name__)


class SeverityLevel(str, Enum):
    """Disease severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    HEALTHY = "healthy"


@dataclass
class ModelPrediction:
    """Individual model prediction with explanation."""
    model_name: str
    model_type: str
    prediction: Optional[str]
    confidence: float
    raw_label: Optional[str]
    processing_time_ms: float
    explanation: str
    contributing_factors: List[str]
    error: Optional[str] = None
    additional_info: Optional[Dict] = None


@dataclass
class DiseaseConsensus:
    """Consensus disease identification from multiple models."""
    disease_name: str
    confidence: float
    model_agreement: float  # 0-1, how many models agree
    supporting_models: List[str]
    dissenting_models: List[str]
    is_healthy: bool
    reasoning: str


@dataclass
class SeverityAssessment:
    """Disease severity assessment."""
    level: SeverityLevel
    score: float  # 0-100
    factors: List[str]
    urgency: str
    action_timeline: str


@dataclass
class LocalizedTreatment:
    """Localized treatment recommendation."""
    immediate_actions: List[str]
    organic_treatments: List[str]
    chemical_treatments: List[str]
    prevention_measures: List[str]
    monitoring_schedule: str
    estimated_recovery: str
    regional_notes: Optional[str] = None
    weather_considerations: Optional[str] = None


@dataclass
class EarlyWarningResult:
    """Complete early warning system result."""
    model_predictions: List[ModelPrediction]
    consensus: DiseaseConsensus
    severity: SeverityAssessment
    treatment: LocalizedTreatment
    metadata: Dict[str, Any] = field(default_factory=dict)


# Disease severity database - maps diseases to base severity and characteristics
DISEASE_SEVERITY_DATABASE = {
    # Tomato diseases
    "early blight": {
        "base_severity": 60,
        "spread_rate": "moderate",
        "crop_impact": "Can cause 20-50% yield loss",
        "urgency": "Act within 3-5 days"
    },
    "late blight": {
        "base_severity": 90,
        "spread_rate": "rapid",
        "crop_impact": "Can destroy entire crop within 7-10 days",
        "urgency": "Immediate action required"
    },
    "bacterial spot": {
        "base_severity": 55,
        "spread_rate": "moderate",
        "crop_impact": "Reduces fruit quality and marketability",
        "urgency": "Act within 5-7 days"
    },
    "leaf mold": {
        "base_severity": 45,
        "spread_rate": "slow",
        "crop_impact": "Reduces photosynthesis, 10-30% yield loss",
        "urgency": "Monitor and treat within 7-10 days"
    },
    "septoria leaf spot": {
        "base_severity": 50,
        "spread_rate": "moderate",
        "crop_impact": "Defoliation leads to sunscald on fruit",
        "urgency": "Act within 5-7 days"
    },
    "spider mites": {
        "base_severity": 40,
        "spread_rate": "rapid in hot weather",
        "crop_impact": "Stunted growth, reduced yield",
        "urgency": "Monitor closely, treat if spreading"
    },
    "target spot": {
        "base_severity": 55,
        "spread_rate": "moderate",
        "crop_impact": "Defoliation and fruit lesions",
        "urgency": "Act within 5-7 days"
    },
    "yellow leaf curl virus": {
        "base_severity": 85,
        "spread_rate": "rapid via whiteflies",
        "crop_impact": "Severe stunting, 50-100% yield loss",
        "urgency": "Immediate vector control needed"
    },
    "mosaic virus": {
        "base_severity": 70,
        "spread_rate": "moderate",
        "crop_impact": "Mottled leaves, reduced fruit quality",
        "urgency": "Remove infected plants immediately"
    },

    # Potato diseases
    "potato early blight": {
        "base_severity": 55,
        "spread_rate": "moderate",
        "crop_impact": "Tuber quality reduction",
        "urgency": "Act within 5-7 days"
    },
    "potato late blight": {
        "base_severity": 95,
        "spread_rate": "extremely rapid",
        "crop_impact": "Complete crop loss possible",
        "urgency": "Emergency action required"
    },

    # Apple diseases
    "apple scab": {
        "base_severity": 65,
        "spread_rate": "moderate in wet conditions",
        "crop_impact": "Fruit unmarketable, defoliation",
        "urgency": "Preventive sprays critical"
    },
    "black rot": {
        "base_severity": 70,
        "spread_rate": "moderate",
        "crop_impact": "Fruit rot, cankers on branches",
        "urgency": "Act within 3-5 days"
    },
    "cedar apple rust": {
        "base_severity": 50,
        "spread_rate": "slow",
        "crop_impact": "Defoliation, reduced fruit size",
        "urgency": "Preventive management needed"
    },

    # Corn diseases
    "common rust": {
        "base_severity": 45,
        "spread_rate": "moderate",
        "crop_impact": "10-20% yield loss if severe",
        "urgency": "Monitor, treat if spreading rapidly"
    },
    "gray leaf spot": {
        "base_severity": 60,
        "spread_rate": "moderate to rapid",
        "crop_impact": "Significant yield reduction",
        "urgency": "Act within 5-7 days"
    },
    "northern leaf blight": {
        "base_severity": 65,
        "spread_rate": "moderate",
        "crop_impact": "30-50% yield loss possible",
        "urgency": "Act within 5-7 days"
    },
    "leaf blight": {
        "base_severity": 60,
        "spread_rate": "moderate",
        "crop_impact": "Reduced photosynthesis and yield",
        "urgency": "Act within 5-7 days"
    },

    # Grape diseases
    "grape black rot": {
        "base_severity": 75,
        "spread_rate": "rapid in humid conditions",
        "crop_impact": "Total fruit loss possible",
        "urgency": "Immediate fungicide application"
    },
    "esca": {
        "base_severity": 80,
        "spread_rate": "slow but chronic",
        "crop_impact": "Vine decline and death",
        "urgency": "Long-term management needed"
    },
    "grape leaf blight": {
        "base_severity": 55,
        "spread_rate": "moderate",
        "crop_impact": "Reduced fruit quality",
        "urgency": "Act within 5-7 days"
    },

    # Rice diseases
    "brown spot": {
        "base_severity": 50,
        "spread_rate": "moderate",
        "crop_impact": "Grain discoloration, 10-30% loss",
        "urgency": "Act within 7 days"
    },
    "leaf blast": {
        "base_severity": 85,
        "spread_rate": "very rapid",
        "crop_impact": "Can destroy entire field",
        "urgency": "Emergency action required"
    },

    # Wheat diseases
    "brown rust": {
        "base_severity": 60,
        "spread_rate": "rapid",
        "crop_impact": "20-40% yield loss",
        "urgency": "Act within 3-5 days"
    },
    "yellow rust": {
        "base_severity": 75,
        "spread_rate": "very rapid",
        "crop_impact": "50-70% yield loss possible",
        "urgency": "Immediate action required"
    },
}

# Enhanced treatment database with severity-based recommendations
LOCALIZED_TREATMENT_DATABASE = {
    "early blight": {
        "immediate_actions": [
            "Remove and destroy infected leaves immediately",
            "Increase plant spacing for better air circulation",
            "Avoid overhead watering - use drip irrigation",
            "Apply mulch to prevent soil splash"
        ],
        "organic_treatments": [
            "Copper-based fungicide (Bordeaux mixture) - apply every 7-10 days",
            "Neem oil spray (2-3 tablespoons per gallon) weekly",
            "Baking soda solution (1 tbsp per gallon + soap) as preventive",
            "Compost tea foliar spray to boost plant immunity"
        ],
        "chemical_treatments": [
            "Chlorothalonil (Daconil) - apply at first sign, repeat every 7-14 days",
            "Mancozeb - effective preventive, apply before symptoms appear",
            "Azoxystrobin (Quadris) - systemic protection for 14-21 days"
        ],
        "prevention_measures": [
            "Rotate crops - don't plant tomatoes/potatoes in same spot for 3 years",
            "Use disease-resistant varieties (Mountain Merit, Defiant)",
            "Remove plant debris at end of season",
            "Stake plants to keep foliage off ground"
        ],
        "monitoring_schedule": "Inspect plants every 2-3 days during humid weather",
        "estimated_recovery": "2-4 weeks with proper treatment"
    },
    "late blight": {
        "immediate_actions": [
            "URGENT: Remove and bag infected plants immediately",
            "Do NOT compost - burn or dispose in sealed bags",
            "Spray remaining plants with fungicide within 24 hours",
            "Alert neighboring farmers - disease spreads rapidly"
        ],
        "organic_treatments": [
            "Copper hydroxide - higher rates than for early blight",
            "Bacillus subtilis (Serenade) - apply every 5-7 days",
            "Potassium bicarbonate sprays for prevention"
        ],
        "chemical_treatments": [
            "Mefenoxam/Metalaxyl (Ridomil) - systemic, very effective",
            "Cymoxanil + Mancozeb combination for resistance management",
            "Fluopicolide (Presidio) - excellent curative action",
            "Mandipropamid (Revus) - rainfast protection"
        ],
        "prevention_measures": [
            "Use only certified disease-free seed potatoes",
            "Plant resistant varieties when available",
            "Destroy volunteer potatoes and tomatoes",
            "Monitor weather - apply preventive sprays before rain events"
        ],
        "monitoring_schedule": "Daily inspection during disease-favorable weather",
        "estimated_recovery": "Disease is often fatal - focus on protecting remaining plants"
    },
    "bacterial spot": {
        "immediate_actions": [
            "Reduce leaf wetness - avoid overhead irrigation",
            "Remove severely infected leaves",
            "Disinfect tools between plants (10% bleach solution)"
        ],
        "organic_treatments": [
            "Copper-based bactericides (must apply before infection)",
            "Acibenzolar-S-methyl (Actigard) to induce plant resistance",
            "Bacteriophage products where available"
        ],
        "chemical_treatments": [
            "Copper + Mancozeb tank mix for better efficacy",
            "Streptomycin (where legal) for severe outbreaks"
        ],
        "prevention_measures": [
            "Use certified disease-free transplants",
            "Hot water seed treatment (122°F for 25 minutes)",
            "Avoid working with wet plants",
            "Resistant varieties: BHN 444, Mountain Magic"
        ],
        "monitoring_schedule": "Check every 3-4 days, especially after rain",
        "estimated_recovery": "3-4 weeks; may persist throughout season"
    },
    "common rust": {
        "immediate_actions": [
            "Scout field edges and low areas first",
            "Calculate disease severity to determine treatment need",
            "Check weather forecast for conditions favoring rust"
        ],
        "organic_treatments": [
            "Sulfur-based fungicides - apply early",
            "Neem oil can slow progression",
            "Remove heavily infected plants in small gardens"
        ],
        "chemical_treatments": [
            "Triazole fungicides (propiconazole, tebuconazole)",
            "Strobilurin fungicides (azoxystrobin, pyraclostrobin)",
            "Apply at tasseling if disease is present and spreading"
        ],
        "prevention_measures": [
            "Plant rust-resistant hybrids",
            "Early planting to avoid peak rust season",
            "Balanced fertilization - avoid excess nitrogen"
        ],
        "monitoring_schedule": "Weekly scouting from V8 through grain fill",
        "estimated_recovery": "Plants can recover if treated early; late infections have less impact"
    },
    "powdery mildew": {
        "immediate_actions": [
            "Improve air circulation around plants",
            "Remove heavily infected leaves",
            "Reduce nitrogen fertilization"
        ],
        "organic_treatments": [
            "Milk spray (40% milk to water ratio) - surprisingly effective",
            "Potassium bicarbonate (1 tbsp per gallon)",
            "Neem oil weekly applications",
            "Sulfur dust or spray (not in hot weather)"
        ],
        "chemical_treatments": [
            "Myclobutanil (Immunox) - systemic protection",
            "Trifloxystrobin (Flint) - excellent control",
            "Chlorothalonil for prevention"
        ],
        "prevention_measures": [
            "Plant in full sun with good air circulation",
            "Use resistant varieties",
            "Water at soil level, not on leaves",
            "Space plants adequately"
        ],
        "monitoring_schedule": "Check undersides of leaves weekly",
        "estimated_recovery": "2-3 weeks with consistent treatment"
    },
    "yellow leaf curl virus": {
        "immediate_actions": [
            "Remove and destroy infected plants immediately",
            "Control whitefly vectors aggressively",
            "Use yellow sticky traps to monitor whiteflies",
            "Cover young plants with insect netting"
        ],
        "organic_treatments": [
            "Insecticidal soap for whitefly control",
            "Neem oil (repels whiteflies)",
            "Release beneficial insects (Encarsia formosa)",
            "Reflective mulch to repel whiteflies"
        ],
        "chemical_treatments": [
            "Imidacloprid soil drench for whitefly control",
            "Pyriproxyfen (insect growth regulator)",
            "Spiromesifen (Oberon) - excellent whitefly control"
        ],
        "prevention_measures": [
            "Use virus-resistant varieties (Ty genes)",
            "Screen greenhouse openings",
            "Remove weed hosts around fields",
            "Avoid planting near infected fields"
        ],
        "monitoring_schedule": "Daily whitefly monitoring; remove infected plants immediately",
        "estimated_recovery": "No cure - focus on vector control and removing infected plants"
    },
}

# Regional treatment notes
REGIONAL_NOTES = {
    "US-CA": "California: Check local restrictions on copper applications. Organic options preferred in many areas.",
    "US-FL": "Florida: High humidity increases disease pressure. More frequent applications may be needed.",
    "US-TX": "Texas: Heat stress compounds disease issues. Water management critical.",
    "EU": "European Union: Many conventional pesticides restricted. Focus on integrated pest management.",
    "IN-MH": "Maharashtra: Monsoon season increases fungal disease risk. Post-rain applications critical.",
    "IN-KA": "Karnataka: Coffee-growing regions have specific disease pressures. Consult local extension.",
    "AU": "Australia: Strict biosecurity - report unusual diseases. Many chemicals require permits.",
}


class EarlyWarningService:
    """Service for comprehensive crop disease early warning."""

    def __init__(self):
        self.registry = get_external_model_registry()

    async def analyze(
        self,
        image: Image.Image,
        region: Optional[str] = None,
        include_internal: bool = True,
        plantnet_api_key: Optional[str] = None,
        kindwise_api_key: Optional[str] = None,
    ) -> EarlyWarningResult:
        """
        Run comprehensive disease analysis.

        Args:
            image: PIL Image to analyze
            region: Optional region code for localized advice
            include_internal: Whether to include internal model
            plantnet_api_key: Optional PlantNet API key
            kindwise_api_key: Optional Kindwise API key

        Returns:
            EarlyWarningResult with full analysis
        """
        import time
        start_time = time.time()

        # Override API keys if provided
        if plantnet_api_key:
            plantnet_model = self.registry.get_model(ExternalModelType.PLANTNET)
            if plantnet_model:
                plantnet_model.api_key = plantnet_api_key
                plantnet_model.is_loaded = True

        if kindwise_api_key:
            kindwise_model = self.registry.get_model(ExternalModelType.KINDWISE)
            if kindwise_model:
                kindwise_model.api_key = kindwise_api_key
                kindwise_model.is_loaded = True

        # Run all models in parallel
        model_results = await self._run_all_models(image, include_internal)

        # Convert to predictions with explanations
        predictions = self._create_predictions(model_results)

        # Determine consensus disease
        consensus = self._determine_consensus(predictions)

        # Calculate severity
        severity = self._calculate_severity(consensus, predictions)

        # Generate treatment recommendations
        treatment = self._generate_treatment(consensus, severity, region)

        total_time = (time.time() - start_time) * 1000

        return EarlyWarningResult(
            model_predictions=predictions,
            consensus=consensus,
            severity=severity,
            treatment=treatment,
            metadata={
                "total_processing_time_ms": total_time,
                "models_consulted": len(predictions),
                "region": region,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }
        )

    async def _run_all_models(
        self,
        image: Image.Image,
        include_internal: bool
    ) -> Dict[str, ExternalModelResult]:
        """Run all available models in parallel."""

        # Get all model types
        all_model_types = list(ExternalModelType)

        # Run comparison
        results = await self.registry.run_comparison(image, model_types=all_model_types)

        return results

    def _create_predictions(
        self,
        results: Dict[str, ExternalModelResult]
    ) -> List[ModelPrediction]:
        """Create detailed predictions with explanations."""
        predictions = []

        for model_key, result in results.items():
            # Generate explanation based on result
            explanation, factors = self._generate_explanation(result)

            predictions.append(ModelPrediction(
                model_name=result.model_name,
                model_type=result.model_type.value if hasattr(result.model_type, 'value') else str(result.model_type),
                prediction=result.prediction,
                confidence=result.confidence or 0,
                raw_label=result.raw_label,
                processing_time_ms=result.processing_time_ms,
                explanation=explanation,
                contributing_factors=factors,
                error=result.error,
                additional_info=result.additional_info
            ))

        return predictions

    def _generate_explanation(self, result: ExternalModelResult) -> tuple:
        """Generate human-readable explanation for a model's prediction."""
        if result.error:
            return f"Model encountered an error: {result.error}", ["Error during analysis"]

        if not result.prediction:
            return "Model could not make a prediction", ["Insufficient data"]

        prediction = result.prediction.lower()
        confidence = result.confidence or 0

        # Build explanation based on confidence and prediction
        factors = []

        if confidence >= 0.9:
            confidence_text = "very high confidence"
            factors.append("Strong visual pattern match")
        elif confidence >= 0.7:
            confidence_text = "high confidence"
            factors.append("Clear disease indicators detected")
        elif confidence >= 0.5:
            confidence_text = "moderate confidence"
            factors.append("Some disease indicators present")
        else:
            confidence_text = "low confidence"
            factors.append("Weak or ambiguous indicators")

        # Add model-specific factors
        additional = result.additional_info or {}

        if additional.get("crop"):
            factors.append(f"Identified crop: {additional['crop']}")

        if additional.get("top_3"):
            top_3 = additional["top_3"]
            if len(top_3) > 1:
                second_conf = top_3[1].get("confidence", 0)
                if second_conf > 0.3:
                    factors.append(f"Alternative diagnosis possible: {top_3[1].get('disease', top_3[1].get('label', 'Unknown'))}")

        if "healthy" in prediction:
            explanation = f"This model detected no disease with {confidence_text} ({confidence*100:.1f}%). The plant appears healthy based on visual analysis."
        else:
            explanation = f"This model identified '{result.prediction}' with {confidence_text} ({confidence*100:.1f}%). The prediction is based on visual pattern recognition trained on similar disease presentations."

        return explanation, factors

    def _determine_consensus(self, predictions: List[ModelPrediction]) -> DiseaseConsensus:
        """Determine consensus disease from all predictions."""

        # Filter out errors and get valid predictions
        valid_predictions = [p for p in predictions if not p.error and p.prediction]

        if not valid_predictions:
            return DiseaseConsensus(
                disease_name="Unknown",
                confidence=0,
                model_agreement=0,
                supporting_models=[],
                dissenting_models=[p.model_name for p in predictions],
                is_healthy=False,
                reasoning="No models could make a valid prediction"
            )

        # Normalize predictions for comparison
        def normalize(pred: str) -> str:
            return pred.lower().strip().replace("_", " ").replace("-", " ")

        # Count votes with confidence weighting
        disease_votes = {}
        healthy_votes = 0
        total_confidence = 0

        for pred in valid_predictions:
            norm_pred = normalize(pred.prediction)

            if "healthy" in norm_pred:
                healthy_votes += pred.confidence
            else:
                if norm_pred not in disease_votes:
                    disease_votes[norm_pred] = {
                        "confidence_sum": 0,
                        "count": 0,
                        "supporters": [],
                        "original_name": pred.prediction
                    }
                disease_votes[norm_pred]["confidence_sum"] += pred.confidence
                disease_votes[norm_pred]["count"] += 1
                disease_votes[norm_pred]["supporters"].append(pred.model_name)

            total_confidence += pred.confidence

        # Determine winner
        is_healthy = healthy_votes > sum(d["confidence_sum"] for d in disease_votes.values())

        if is_healthy:
            agreement = healthy_votes / total_confidence if total_confidence > 0 else 0
            healthy_supporters = [p.model_name for p in valid_predictions if "healthy" in normalize(p.prediction)]
            dissenters = [p.model_name for p in valid_predictions if "healthy" not in normalize(p.prediction)]

            return DiseaseConsensus(
                disease_name="Healthy",
                confidence=healthy_votes / len(valid_predictions) if valid_predictions else 0,
                model_agreement=agreement,
                supporting_models=healthy_supporters,
                dissenting_models=dissenters,
                is_healthy=True,
                reasoning=f"{len(healthy_supporters)} out of {len(valid_predictions)} models detected no disease. The plant appears healthy."
            )

        # Find top disease
        if disease_votes:
            top_disease = max(disease_votes.items(), key=lambda x: x[1]["confidence_sum"])
            disease_name = top_disease[1]["original_name"]
            supporters = top_disease[1]["supporters"]
            agreement = len(supporters) / len(valid_predictions)
            avg_confidence = top_disease[1]["confidence_sum"] / top_disease[1]["count"]

            dissenters = [p.model_name for p in valid_predictions
                         if normalize(p.prediction) != top_disease[0] and p.model_name not in supporters]

            # Build reasoning
            if agreement >= 0.8:
                reasoning = f"Strong consensus: {len(supporters)} out of {len(valid_predictions)} models identified {disease_name}. High confidence in diagnosis."
            elif agreement >= 0.5:
                reasoning = f"Moderate consensus: {len(supporters)} out of {len(valid_predictions)} models identified {disease_name}. Some models suggested alternative diagnoses."
            else:
                reasoning = f"Weak consensus: Only {len(supporters)} out of {len(valid_predictions)} models identified {disease_name}. Consider multiple possibilities."

            return DiseaseConsensus(
                disease_name=disease_name,
                confidence=avg_confidence,
                model_agreement=agreement,
                supporting_models=supporters,
                dissenting_models=dissenters,
                is_healthy=False,
                reasoning=reasoning
            )

        return DiseaseConsensus(
            disease_name="Unknown",
            confidence=0,
            model_agreement=0,
            supporting_models=[],
            dissenting_models=[p.model_name for p in predictions],
            is_healthy=False,
            reasoning="Could not determine disease from model predictions"
        )

    def _calculate_severity(
        self,
        consensus: DiseaseConsensus,
        predictions: List[ModelPrediction]
    ) -> SeverityAssessment:
        """Calculate disease severity based on consensus and disease characteristics."""

        if consensus.is_healthy:
            return SeverityAssessment(
                level=SeverityLevel.HEALTHY,
                score=0,
                factors=["No disease detected", "Plant appears healthy"],
                urgency="No immediate action required",
                action_timeline="Continue regular monitoring and preventive care"
            )

        # Look up disease in database
        disease_key = consensus.disease_name.lower().strip()
        disease_info = None

        for key, info in DISEASE_SEVERITY_DATABASE.items():
            if key in disease_key or disease_key in key:
                disease_info = info
                break

        # Calculate severity score
        factors = []

        # Base severity from disease database
        if disease_info:
            base_severity = disease_info["base_severity"]
            factors.append(f"Base severity for {consensus.disease_name}: {base_severity}/100")
        else:
            base_severity = 50  # Default moderate
            factors.append("Disease not in database - using moderate baseline")

        # Adjust based on confidence
        confidence_modifier = 0
        if consensus.confidence >= 0.9:
            confidence_modifier = 10
            factors.append("High detection confidence (+10)")
        elif consensus.confidence >= 0.7:
            confidence_modifier = 5
            factors.append("Good detection confidence (+5)")
        elif consensus.confidence < 0.5:
            confidence_modifier = -10
            factors.append("Low detection confidence (-10)")

        # Adjust based on model agreement
        agreement_modifier = 0
        if consensus.model_agreement >= 0.8:
            agreement_modifier = 10
            factors.append("Strong model agreement (+10)")
        elif consensus.model_agreement >= 0.5:
            agreement_modifier = 0
            factors.append("Moderate model agreement (0)")
        else:
            agreement_modifier = -15
            factors.append("Weak model agreement - diagnosis uncertain (-15)")

        # Calculate final score
        final_score = max(0, min(100, base_severity + confidence_modifier + agreement_modifier))

        # Determine severity level
        if final_score >= 80:
            level = SeverityLevel.CRITICAL
            urgency = "CRITICAL: Immediate action required within 24-48 hours"
        elif final_score >= 60:
            level = SeverityLevel.HIGH
            urgency = "HIGH: Take action within 3-5 days"
        elif final_score >= 40:
            level = SeverityLevel.MODERATE
            urgency = "MODERATE: Address within 1-2 weeks"
        elif final_score > 0:
            level = SeverityLevel.LOW
            urgency = "LOW: Monitor and treat preventively"
        else:
            level = SeverityLevel.HEALTHY
            urgency = "No action needed"

        # Get action timeline from disease info
        if disease_info:
            action_timeline = disease_info.get("urgency", urgency)
            factors.append(f"Spread rate: {disease_info.get('spread_rate', 'unknown')}")
            factors.append(f"Potential impact: {disease_info.get('crop_impact', 'unknown')}")
        else:
            action_timeline = urgency

        return SeverityAssessment(
            level=level,
            score=final_score,
            factors=factors,
            urgency=urgency,
            action_timeline=action_timeline
        )

    def _generate_treatment(
        self,
        consensus: DiseaseConsensus,
        severity: SeverityAssessment,
        region: Optional[str]
    ) -> LocalizedTreatment:
        """Generate localized treatment recommendations."""

        if consensus.is_healthy:
            return LocalizedTreatment(
                immediate_actions=["Continue regular care and monitoring"],
                organic_treatments=["Preventive neem oil spray every 2 weeks", "Compost tea foliar applications for plant immunity"],
                chemical_treatments=["No chemical treatment needed for healthy plants"],
                prevention_measures=[
                    "Maintain proper watering schedule",
                    "Ensure adequate spacing for air circulation",
                    "Regular soil health management",
                    "Scout for early signs of stress or disease"
                ],
                monitoring_schedule="Weekly visual inspection recommended",
                estimated_recovery="N/A - Plant is healthy",
                regional_notes=REGIONAL_NOTES.get(region)
            )

        # Look up treatment in database
        disease_key = consensus.disease_name.lower().strip()
        treatment_info = None

        for key, info in LOCALIZED_TREATMENT_DATABASE.items():
            if key in disease_key or disease_key in key:
                treatment_info = info
                break

        if treatment_info:
            # Adjust treatments based on severity
            immediate = treatment_info["immediate_actions"].copy()

            if severity.level == SeverityLevel.CRITICAL:
                immediate.insert(0, "⚠️ CRITICAL: This is an emergency situation")
                immediate.insert(1, "Document with photos and contact local agricultural extension")

            return LocalizedTreatment(
                immediate_actions=immediate,
                organic_treatments=treatment_info["organic_treatments"],
                chemical_treatments=treatment_info["chemical_treatments"],
                prevention_measures=treatment_info["prevention_measures"],
                monitoring_schedule=treatment_info["monitoring_schedule"],
                estimated_recovery=treatment_info["estimated_recovery"],
                regional_notes=REGIONAL_NOTES.get(region),
                weather_considerations="Monitor weather forecasts - humid conditions increase disease spread"
            )

        # Generic treatment for unknown diseases
        return LocalizedTreatment(
            immediate_actions=[
                "Isolate affected plants if possible",
                "Remove and destroy severely infected plant parts",
                "Improve air circulation around plants",
                "Photograph symptoms for expert consultation"
            ],
            organic_treatments=[
                "Copper-based fungicide as general treatment",
                "Neem oil spray for fungal/pest issues",
                "Baking soda solution (1 tbsp/gallon) for fungal problems"
            ],
            chemical_treatments=[
                "Consult local agricultural extension for specific recommendations",
                "Broad-spectrum fungicide may help if fungal origin suspected"
            ],
            prevention_measures=[
                "Practice crop rotation",
                "Use disease-resistant varieties when replanting",
                "Maintain proper plant nutrition",
                "Ensure good drainage"
            ],
            monitoring_schedule="Daily monitoring until disease is identified",
            estimated_recovery="Varies - consult local experts for specific disease",
            regional_notes=REGIONAL_NOTES.get(region),
            weather_considerations="Reduce irrigation during treatment period if disease is fungal"
        )


# Singleton instance
_early_warning_service = None

def get_early_warning_service() -> EarlyWarningService:
    """Get the early warning service singleton."""
    global _early_warning_service
    if _early_warning_service is None:
        _early_warning_service = EarlyWarningService()
    return _early_warning_service
