"""
AI Crop Disease Early Warning Service

Provides comprehensive disease analysis by:
1. Running ALL available models in parallel
2. Aggregating results with confidence-weighted voting
3. Determining disease severity
4. Generating localized treatment recommendations

Treatment data is loaded from external JSON files.
See data/treatments/disease_treatments.json for the data format.

Recommended Data Sources:
- PlantVillage (Penn State): https://plantvillage.psu.edu/
- EPPO Global Database: https://gd.eppo.int/
- UC Davis IPM: https://ipm.ucanr.edu/
- CABI Crop Protection Compendium: https://www.cabi.org/cpc/
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
from app.services.treatment_data_loader import (
    get_treatment_loader,
    TreatmentDataLoader,
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
    is_fallback: bool = False  # True if severity data not found in database


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
    data_source: str = "Unknown"
    data_source_url: Optional[str] = None
    is_fallback: bool = False  # True if treatment data not found in database


@dataclass
class SpeciesConsensusResult:
    """Species identification consensus from multiple models."""
    family: str
    genus: str
    species: str
    common_name: Optional[str]
    confidence: float
    agreement: float
    supporting_models: List[str]
    notes: str
    disagreements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EarlyWarningResult:
    """Complete early warning system result."""
    model_predictions: List[ModelPrediction]
    consensus: DiseaseConsensus
    severity: SeverityAssessment
    treatment: LocalizedTreatment
    species_consensus: Optional[SpeciesConsensusResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EarlyWarningService:
    """Service for comprehensive crop disease early warning."""

    def __init__(self):
        self.registry = get_external_model_registry()
        self.treatment_loader = get_treatment_loader()

        # Species consensus components (lazy-loaded)
        self._species_registry = None
        self._consensus_engine = None

        # Log data loading status
        metadata = self.treatment_loader.get_metadata()
        if metadata["loaded"]:
            logger.info(f"Treatment data loaded: {metadata['disease_count']} diseases, "
                       f"sources: {metadata['data_sources']}")
        else:
            logger.warning(f"Treatment data not loaded: {metadata['load_error']}. "
                          f"Using fallback responses.")

    def _get_species_registry(self):
        """Lazy-load species model registry."""
        if self._species_registry is None:
            try:
                from app.ml.species_models import get_species_model_registry
                self._species_registry = get_species_model_registry()
            except ImportError:
                logger.warning("Species models not available")
        return self._species_registry

    def _get_consensus_engine(self):
        """Lazy-load consensus engine."""
        if self._consensus_engine is None:
            try:
                from app.ml.species_models import SpeciesConsensusEngine
                self._consensus_engine = SpeciesConsensusEngine()
            except ImportError:
                logger.warning("Consensus engine not available")
        return self._consensus_engine

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

        # Run all disease models in parallel
        model_results = await self._run_all_models(image, include_internal)

        # Convert to predictions with explanations
        predictions = self._create_predictions(model_results)

        # Determine consensus disease
        consensus = self._determine_consensus(predictions)

        # Run species models and compute consensus
        species_consensus = await self._compute_species_consensus(
            image,
            plantnet_api_key=plantnet_api_key,
            kindwise_api_key=kindwise_api_key,
        )

        # Calculate severity (uses data loader)
        severity = self._calculate_severity(consensus, predictions)

        # Adjust severity based on species confidence
        if species_consensus:
            severity = self._adjust_severity_by_species(severity, species_consensus)

        # Generate treatment recommendations (uses data loader)
        treatment = self._generate_treatment(consensus, severity, region)

        total_time = (time.time() - start_time) * 1000

        # Get data loader metadata
        loader_metadata = self.treatment_loader.get_metadata()

        return EarlyWarningResult(
            model_predictions=predictions,
            consensus=consensus,
            severity=severity,
            treatment=treatment,
            species_consensus=species_consensus,
            metadata={
                "total_processing_time_ms": total_time,
                "models_consulted": len(predictions),
                "region": region,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "treatment_data": {
                    "loaded": loader_metadata["loaded"],
                    "disease_count": loader_metadata["disease_count"],
                    "data_sources": loader_metadata["data_sources"],
                    "last_updated": loader_metadata["last_updated"],
                },
                "severity_is_fallback": severity.is_fallback,
                "treatment_is_fallback": treatment.is_fallback,
                "species_consensus_computed": species_consensus is not None,
            }
        )

    async def _run_all_models(
        self,
        image: Image.Image,
        include_internal: bool
    ) -> Dict[str, ExternalModelResult]:
        """Run all available models in parallel."""
        all_model_types = list(ExternalModelType)
        results = await self.registry.run_comparison(image, model_types=all_model_types)
        return results

    def _create_predictions(
        self,
        results: Dict[str, ExternalModelResult]
    ) -> List[ModelPrediction]:
        """Create detailed predictions with explanations."""
        predictions = []

        for model_key, result in results.items():
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

        def normalize(pred: str) -> str:
            return pred.lower().strip().replace("_", " ").replace("-", " ")

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

        if disease_votes:
            top_disease = max(disease_votes.items(), key=lambda x: x[1]["confidence_sum"])
            disease_name = top_disease[1]["original_name"]
            supporters = top_disease[1]["supporters"]
            agreement = len(supporters) / len(valid_predictions)
            avg_confidence = top_disease[1]["confidence_sum"] / top_disease[1]["count"]

            dissenters = [p.model_name for p in valid_predictions
                         if normalize(p.prediction) != top_disease[0] and p.model_name not in supporters]

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
        """Calculate disease severity using data from treatment loader."""

        if consensus.is_healthy:
            return SeverityAssessment(
                level=SeverityLevel.HEALTHY,
                score=0,
                factors=["No disease detected", "Plant appears healthy"],
                urgency="No immediate action required",
                action_timeline="Continue regular monitoring and preventive care",
                is_fallback=False
            )

        # Get severity data from loader
        severity_data = self.treatment_loader.get_severity(consensus.disease_name)
        is_fallback = severity_data.is_fallback

        factors = []

        # Base severity from database
        base_severity = severity_data.base_severity
        if is_fallback:
            factors.append(f"⚠️ FALLBACK: '{consensus.disease_name}' not in database, using default severity (50)")
        else:
            factors.append(f"Base severity for {consensus.disease_name}: {base_severity}/100")

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

        # Add spread rate and impact info from data
        if not is_fallback:
            factors.append(f"Spread rate: {severity_data.spread_rate}")
            factors.append(f"Potential impact: {severity_data.crop_impact}")

        return SeverityAssessment(
            level=level,
            score=final_score,
            factors=factors,
            urgency=urgency,
            action_timeline=urgency,
            is_fallback=is_fallback
        )

    def _generate_treatment(
        self,
        consensus: DiseaseConsensus,
        severity: SeverityAssessment,
        region: Optional[str]
    ) -> LocalizedTreatment:
        """Generate treatment recommendations using data from treatment loader."""

        if consensus.is_healthy:
            return LocalizedTreatment(
                immediate_actions=["Continue regular care and monitoring"],
                organic_treatments=[
                    "Preventive neem oil spray every 2 weeks",
                    "Compost tea foliar applications for plant immunity"
                ],
                chemical_treatments=["No chemical treatment needed for healthy plants"],
                prevention_measures=[
                    "Maintain proper watering schedule",
                    "Ensure adequate spacing for air circulation",
                    "Regular soil health management",
                    "Scout for early signs of stress or disease"
                ],
                monitoring_schedule="Weekly visual inspection recommended",
                estimated_recovery="N/A - Plant is healthy",
                regional_notes=self._get_regional_notes(region),
                data_source="Standard healthy plant care",
                is_fallback=False
            )

        # Get treatment data from loader
        treatment_data = self.treatment_loader.get_treatment(consensus.disease_name)
        is_fallback = treatment_data.is_fallback

        # Get regional notes
        regional_notes = self._get_regional_notes(region)

        # Adjust treatments based on severity
        immediate_actions = treatment_data.immediate_actions.copy()

        if severity.level == SeverityLevel.CRITICAL and not is_fallback:
            immediate_actions.insert(0, "⚠️ CRITICAL: This is an emergency situation requiring immediate action")
            immediate_actions.insert(1, "Document with photos and contact local agricultural extension")

        # Add fallback warning if applicable
        if is_fallback:
            immediate_actions.insert(0, f"⚠️ FALLBACK RESPONSE: No specific data for '{consensus.disease_name}'")

        return LocalizedTreatment(
            immediate_actions=immediate_actions,
            organic_treatments=treatment_data.organic_treatments,
            chemical_treatments=treatment_data.chemical_treatments,
            prevention_measures=treatment_data.prevention_measures,
            monitoring_schedule=treatment_data.monitoring_schedule,
            estimated_recovery=treatment_data.estimated_recovery,
            regional_notes=regional_notes,
            weather_considerations="Monitor weather forecasts - humid conditions increase disease spread" if not is_fallback else "Consult local agricultural extension for weather-specific advice",
            data_source=treatment_data.data_source,
            data_source_url=treatment_data.data_source_url,
            is_fallback=is_fallback
        )

    def _get_regional_notes(self, region: Optional[str]) -> Optional[str]:
        """Get regional notes for the specified region."""
        if not region:
            return None

        regional_data = self.treatment_loader.get_regional_notes(region)
        if regional_data:
            return regional_data.notes

        return f"No specific notes for region '{region}'. Consult local agricultural extension."

    async def _compute_species_consensus(
        self,
        image: Image.Image,
        plantnet_api_key: Optional[str] = None,
        kindwise_api_key: Optional[str] = None,
    ) -> Optional[SpeciesConsensusResult]:
        """
        Compute species consensus from multiple models.

        Args:
            image: PIL Image to analyze
            plantnet_api_key: Optional PlantNet API key
            kindwise_api_key: Optional Kindwise API key

        Returns:
            SpeciesConsensusResult or None if species models unavailable
        """
        try:
            species_registry = self._get_species_registry()
            consensus_engine = self._get_consensus_engine()

            if species_registry is None or consensus_engine is None:
                logger.debug("Species consensus not available - models not loaded")
                return None

            # Initialize species registry with API keys
            species_registry.initialize(
                plantnet_api_key=plantnet_api_key,
                kindwise_api_key=kindwise_api_key,
                enable_huggingface=True,
                enable_internal=True,
            )

            # Run all species models in parallel
            predictions = await species_registry.predict_all(image, timeout=30.0)

            if not predictions:
                logger.warning("No species predictions returned")
                return None

            # Convert to list of predictions
            prediction_list = list(predictions.values())

            # Compute consensus
            consensus = consensus_engine.compute_consensus(prediction_list)

            # Convert to result format
            return SpeciesConsensusResult(
                family=consensus.family.name,
                genus=consensus.genus.name,
                species=consensus.species.name,
                common_name=consensus.common_name,
                confidence=consensus.overall_confidence,
                agreement=consensus.agreement_score,
                supporting_models=consensus.supporting_models,
                notes=consensus.notes,
                disagreements=[d.to_dict() for d in consensus.disagreements],
            )

        except Exception as e:
            logger.error(f"Failed to compute species consensus: {e}")
            return None

    def _adjust_severity_by_species(
        self,
        severity: SeverityAssessment,
        species_consensus: SpeciesConsensusResult
    ) -> SeverityAssessment:
        """
        Adjust severity based on species identification confidence.

        Logic:
        - High disease confidence + low species confidence → downgrade severity
        - High agreement across both → can increase severity
        """
        adjustment = 0
        additional_factors = []

        # High species confidence increases reliability
        if species_consensus.confidence >= 0.85 and species_consensus.agreement >= 0.8:
            adjustment += 5
            additional_factors.append(
                f"Species identification high ({species_consensus.confidence:.0%}) - reliable crop ID"
            )
        elif species_consensus.confidence >= 0.7:
            additional_factors.append(
                f"Species identification good ({species_consensus.confidence:.0%})"
            )

        # Low species confidence decreases reliability
        if species_consensus.confidence < 0.5:
            adjustment -= 10
            additional_factors.append(
                f"⚠️ Low species confidence ({species_consensus.confidence:.0%}) - treatment uncertain"
            )
        elif species_consensus.confidence < 0.7:
            adjustment -= 5
            additional_factors.append(
                f"Moderate species uncertainty ({species_consensus.confidence:.0%})"
            )

        # Low agreement indicates conflicting identifications
        if species_consensus.agreement < 0.5:
            adjustment -= 5
            additional_factors.append(
                f"Low model agreement ({species_consensus.agreement:.0%}) - species disputed"
            )

        # Family-level disagreements are severe
        family_disagreements = sum(
            1 for d in species_consensus.disagreements if d.get("level") == "family"
        )
        if family_disagreements > 0:
            adjustment -= 10
            additional_factors.append(
                f"⚠️ {family_disagreements} model(s) disagree on plant family"
            )

        # Calculate adjusted score
        adjusted_score = max(0, min(100, severity.score + adjustment))

        # Determine new severity level based on adjusted score
        if adjusted_score >= 80:
            new_level = SeverityLevel.CRITICAL
            new_urgency = "CRITICAL: Immediate action required within 24-48 hours"
        elif adjusted_score >= 60:
            new_level = SeverityLevel.HIGH
            new_urgency = "HIGH: Take action within 3-5 days"
        elif adjusted_score >= 40:
            new_level = SeverityLevel.MODERATE
            new_urgency = "MODERATE: Address within 1-2 weeks"
        elif adjusted_score > 0:
            new_level = SeverityLevel.LOW
            new_urgency = "LOW: Monitor and treat preventively"
        else:
            new_level = SeverityLevel.HEALTHY
            new_urgency = "No action needed"

        # Return updated severity
        return SeverityAssessment(
            level=new_level,
            score=adjusted_score,
            factors=severity.factors + additional_factors,
            urgency=new_urgency,
            action_timeline=new_urgency,
            is_fallback=severity.is_fallback,
        )


# Singleton instance
_early_warning_service = None


def get_early_warning_service() -> EarlyWarningService:
    """Get the early warning service singleton."""
    global _early_warning_service
    if _early_warning_service is None:
        _early_warning_service = EarlyWarningService()
    return _early_warning_service
