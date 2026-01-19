"""
Treatment Recommendation Service

Provides actionable treatment recommendations based on disease detection.

Features:
- Organic treatment options
- Chemical treatment options (with safety notes)
- Preventive care recommendations
- Region-specific filtering (designed for, not hard-coded)
- Severity-adjusted urgency levels

Design Considerations:
- All treatments come from verified agricultural databases
- Chemical recommendations include safety guidelines
- Region-specific regulations can filter available treatments
- Treatments are practical and field-ready

Iteration Point: Add new treatments by extending the treatment database.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TreatmentOption:
    """Individual treatment option with metadata."""
    name: str
    type: str  # organic, chemical, biological, cultural
    description: str
    application_method: str
    frequency: str
    safety_notes: Optional[str] = None
    effectiveness_rating: float = 0.8  # 0-1
    cost_level: str = "moderate"  # low, moderate, high
    regions_allowed: Optional[list[str]] = None  # None = all regions
    regions_banned: Optional[list[str]] = None


@dataclass
class TreatmentPlan:
    """Complete treatment plan for a disease."""
    disease: str
    organic: list[TreatmentOption]
    chemical: list[TreatmentOption]
    biological: list[TreatmentOption]
    cultural: list[TreatmentOption]
    prevention: list[str]
    urgency: str  # immediate, soon, routine
    estimated_recovery_time: str
    notes: str = ""


class TreatmentDatabase:
    """
    Database of treatment options organized by disease.

    In production, this would be loaded from a database or JSON file
    and regularly updated with new treatments and regulatory changes.

    Iteration Point: Extend with new diseases and treatments.
    """

    TREATMENTS: dict[str, TreatmentPlan] = {
        "Early Blight": TreatmentPlan(
            disease="Early Blight",
            organic=[
                TreatmentOption(
                    name="Neem oil spray",
                    type="organic",
                    description="Natural fungicide that disrupts fungal cell membranes",
                    application_method="Foliar spray, cover all leaf surfaces",
                    frequency="Every 7-14 days during growing season",
                    effectiveness_rating=0.7,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Copper-based fungicide (organic)",
                    type="organic",
                    description="Copper hydroxide or copper sulfate formulations",
                    application_method="Foliar spray at first sign of disease",
                    frequency="Every 7-10 days",
                    safety_notes="Avoid application in hot weather to prevent phytotoxicity",
                    effectiveness_rating=0.8,
                    cost_level="moderate"
                ),
                TreatmentOption(
                    name="Bacillus subtilis (Serenade)",
                    type="organic",
                    description="Biological fungicide using beneficial bacteria",
                    application_method="Foliar spray",
                    frequency="Every 7 days preventively",
                    effectiveness_rating=0.7,
                    cost_level="moderate"
                ),
                TreatmentOption(
                    name="Remove infected leaves",
                    type="organic",
                    description="Physical removal of infected plant material",
                    application_method="Hand removal and disposal (do not compost)",
                    frequency="As soon as symptoms appear",
                    effectiveness_rating=0.6,
                    cost_level="low"
                ),
            ],
            chemical=[
                TreatmentOption(
                    name="Chlorothalonil",
                    type="chemical",
                    description="Broad-spectrum protectant fungicide",
                    application_method="Foliar spray per label instructions",
                    frequency="Every 7-14 days",
                    safety_notes="Follow label PHI (pre-harvest interval). Wear protective equipment during application.",
                    effectiveness_rating=0.9,
                    cost_level="moderate",
                    regions_banned=["EU"]  # Example restriction
                ),
                TreatmentOption(
                    name="Mancozeb",
                    type="chemical",
                    description="Contact fungicide effective against early blight",
                    application_method="Foliar spray",
                    frequency="Every 7-10 days",
                    safety_notes="24-hour REI (restricted entry interval). Check local regulations.",
                    effectiveness_rating=0.85,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Azoxystrobin (Quadris)",
                    type="chemical",
                    description="Systemic strobilurin fungicide",
                    application_method="Foliar spray",
                    frequency="Every 14 days, rotate with other modes of action",
                    safety_notes="Risk of resistance development - rotate fungicide classes",
                    effectiveness_rating=0.9,
                    cost_level="high"
                ),
            ],
            biological=[
                TreatmentOption(
                    name="Trichoderma harzianum",
                    type="biological",
                    description="Beneficial fungus that competes with pathogens",
                    application_method="Soil drench or foliar spray",
                    frequency="At planting and every 2-4 weeks",
                    effectiveness_rating=0.65,
                    cost_level="moderate"
                ),
            ],
            cultural=[
                TreatmentOption(
                    name="Crop rotation",
                    type="cultural",
                    description="Rotate with non-solanaceous crops for 2-3 years",
                    application_method="Planning for next season",
                    frequency="Seasonal",
                    effectiveness_rating=0.8,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Proper spacing",
                    type="cultural",
                    description="Increase plant spacing for better air circulation",
                    application_method="24-36 inches between plants",
                    frequency="At planting",
                    effectiveness_rating=0.6,
                    cost_level="low"
                ),
            ],
            prevention=[
                "Avoid overhead irrigation - use drip irrigation instead",
                "Water early in the day to allow foliage to dry",
                "Mulch around plants to prevent soil splash",
                "Remove plant debris at end of season",
                "Use disease-resistant varieties when available",
                "Stake or cage plants to improve air circulation",
                "Avoid working with plants when wet",
            ],
            urgency="soon",
            estimated_recovery_time="2-4 weeks with treatment",
            notes="Early blight spreads rapidly in warm, humid conditions. Begin treatment at first sign of symptoms."
        ),

        "Late Blight": TreatmentPlan(
            disease="Late Blight",
            organic=[
                TreatmentOption(
                    name="Copper fungicide (Bordeaux mixture)",
                    type="organic",
                    description="Traditional copper-based preventive fungicide",
                    application_method="Foliar spray before infection",
                    frequency="Every 5-7 days during high-risk periods",
                    safety_notes="Can cause copper buildup in soil with repeated use",
                    effectiveness_rating=0.75,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Remove and destroy infected plants",
                    type="organic",
                    description="Complete removal of heavily infected plants",
                    application_method="Uproot and bag for disposal - do not compost",
                    frequency="Immediately upon severe infection",
                    effectiveness_rating=0.9,
                    cost_level="low"
                ),
            ],
            chemical=[
                TreatmentOption(
                    name="Mefenoxam (Ridomil)",
                    type="chemical",
                    description="Systemic fungicide highly effective against Phytophthora",
                    application_method="Foliar spray or soil drench",
                    frequency="Per label - typically every 14 days",
                    safety_notes="Restricted use pesticide in some areas. Risk of resistance.",
                    effectiveness_rating=0.95,
                    cost_level="high"
                ),
                TreatmentOption(
                    name="Cymoxanil + Mancozeb",
                    type="chemical",
                    description="Combination systemic and contact fungicide",
                    application_method="Foliar spray",
                    frequency="Every 7-10 days",
                    safety_notes="Follow label directions carefully",
                    effectiveness_rating=0.9,
                    cost_level="moderate"
                ),
            ],
            biological=[],
            cultural=[
                TreatmentOption(
                    name="Plant resistant varieties",
                    type="cultural",
                    description="Use late blight resistant tomato/potato varieties",
                    application_method="Variety selection before planting",
                    frequency="Seasonal planning",
                    effectiveness_rating=0.85,
                    cost_level="low"
                ),
            ],
            prevention=[
                "Scout fields regularly during cool, wet weather",
                "Destroy volunteer potatoes and tomatoes",
                "Ensure good drainage in fields",
                "Avoid irrigation during evening hours",
                "Monitor late blight forecasting systems (e.g., BLITECAST)",
                "Plant certified disease-free seed potatoes",
            ],
            urgency="immediate",
            estimated_recovery_time="Prevention is key - severe infections often fatal to plants",
            notes="CRITICAL: Late blight can destroy entire crops within days. Act immediately upon detection."
        ),

        "Apple Scab": TreatmentPlan(
            disease="Apple Scab",
            organic=[
                TreatmentOption(
                    name="Sulfur spray",
                    type="organic",
                    description="Traditional organic fungicide",
                    application_method="Foliar spray during infection periods",
                    frequency="Every 7-10 days from green tip to petal fall",
                    safety_notes="Do not apply when temperatures exceed 85°F",
                    effectiveness_rating=0.75,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Potassium bicarbonate",
                    type="organic",
                    description="Contact fungicide that changes leaf surface pH",
                    application_method="Foliar spray",
                    frequency="Every 7-14 days",
                    effectiveness_rating=0.65,
                    cost_level="low"
                ),
            ],
            chemical=[
                TreatmentOption(
                    name="Captan",
                    type="chemical",
                    description="Protectant fungicide",
                    application_method="Foliar spray",
                    frequency="Every 7-10 days during primary scab season",
                    safety_notes="Can be tank-mixed with other fungicides",
                    effectiveness_rating=0.85,
                    cost_level="moderate"
                ),
                TreatmentOption(
                    name="Myclobutanil (Rally)",
                    type="chemical",
                    description="Systemic SI fungicide",
                    application_method="Foliar spray",
                    frequency="Every 10-14 days",
                    safety_notes="Rotate with other fungicide classes",
                    effectiveness_rating=0.9,
                    cost_level="high"
                ),
            ],
            biological=[],
            cultural=[
                TreatmentOption(
                    name="Sanitation",
                    type="cultural",
                    description="Remove fallen leaves in autumn",
                    application_method="Rake and destroy or shred leaves",
                    frequency="After leaf fall",
                    effectiveness_rating=0.7,
                    cost_level="low"
                ),
            ],
            prevention=[
                "Plant scab-resistant apple varieties",
                "Prune trees for good air circulation",
                "Remove fallen leaves to reduce inoculum",
                "Apply urea to fallen leaves to speed decomposition",
                "Monitor weather for infection periods",
            ],
            urgency="soon",
            estimated_recovery_time="Season-long management required",
            notes="Apple scab requires season-long management with multiple applications."
        ),

        "Powdery Mildew": TreatmentPlan(
            disease="Powdery Mildew",
            organic=[
                TreatmentOption(
                    name="Milk spray",
                    type="organic",
                    description="Diluted milk (10-40%) acts as natural fungicide",
                    application_method="Foliar spray",
                    frequency="Every 7-10 days",
                    effectiveness_rating=0.6,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Neem oil",
                    type="organic",
                    description="Natural fungicide and insecticide",
                    application_method="Foliar spray in evening",
                    frequency="Every 7-14 days",
                    safety_notes="Avoid application in hot sun",
                    effectiveness_rating=0.7,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Potassium bicarbonate",
                    type="organic",
                    description="Disrupts fungal cell walls",
                    application_method="Foliar spray",
                    frequency="Every 7-14 days",
                    effectiveness_rating=0.75,
                    cost_level="low"
                ),
            ],
            chemical=[
                TreatmentOption(
                    name="Sulfur",
                    type="chemical",
                    description="Oldest and effective fungicide for PM",
                    application_method="Dust or spray",
                    frequency="Every 7-10 days",
                    safety_notes="Do not apply in temperatures above 85°F",
                    effectiveness_rating=0.8,
                    cost_level="low"
                ),
                TreatmentOption(
                    name="Trifloxystrobin",
                    type="chemical",
                    description="Strobilurin fungicide",
                    application_method="Foliar spray",
                    frequency="Every 14 days",
                    safety_notes="Rotate fungicide groups to prevent resistance",
                    effectiveness_rating=0.9,
                    cost_level="moderate"
                ),
            ],
            biological=[
                TreatmentOption(
                    name="Bacillus pumilus",
                    type="biological",
                    description="Beneficial bacteria that outcompetes pathogens",
                    application_method="Foliar spray",
                    frequency="Every 7-10 days",
                    effectiveness_rating=0.65,
                    cost_level="moderate"
                ),
            ],
            cultural=[
                TreatmentOption(
                    name="Improve air circulation",
                    type="cultural",
                    description="Prune dense growth and increase spacing",
                    application_method="Selective pruning",
                    frequency="Early season and as needed",
                    effectiveness_rating=0.6,
                    cost_level="low"
                ),
            ],
            prevention=[
                "Plant resistant varieties when available",
                "Avoid excess nitrogen fertilization",
                "Ensure adequate spacing between plants",
                "Avoid overhead watering",
                "Remove infected plant parts promptly",
            ],
            urgency="routine",
            estimated_recovery_time="1-3 weeks with consistent treatment",
            notes="Powdery mildew is rarely fatal but reduces yield and quality."
        ),
    }

    @classmethod
    def get_treatment_plan(cls, disease: str) -> Optional[TreatmentPlan]:
        """Get treatment plan for a disease."""
        return cls.TREATMENTS.get(disease)

    @classmethod
    def list_diseases(cls) -> list[str]:
        """List all diseases with treatment plans."""
        return list(cls.TREATMENTS.keys())


@dataclass
class RegionRegulation:
    """Regional pesticide regulation information."""
    region_code: str
    region_name: str
    banned_chemicals: list[str]
    restricted_chemicals: list[str]
    organic_only: bool = False
    notes: str = ""


class RegionRegulationDatabase:
    """
    Database of regional pesticide regulations.

    Iteration Point: Add new regions and regulatory updates.
    This is designed for extensibility, not hard-coded.
    """

    REGULATIONS: dict[str, RegionRegulation] = {
        "US-CA": RegionRegulation(
            region_code="US-CA",
            region_name="California, USA",
            banned_chemicals=["Chlorpyrifos"],
            restricted_chemicals=["Chlorothalonil"],
            notes="California has stricter regulations than federal EPA standards"
        ),
        "EU": RegionRegulation(
            region_code="EU",
            region_name="European Union",
            banned_chemicals=["Chlorothalonil", "Chlorpyrifos", "Mancozeb"],
            restricted_chemicals=["Glyphosate"],
            notes="EU follows precautionary principle for pesticide approval"
        ),
        "IN-MH": RegionRegulation(
            region_code="IN-MH",
            region_name="Maharashtra, India",
            banned_chemicals=[],
            restricted_chemicals=[],
            notes="Follow Central Insecticides Board and Registration Committee guidelines"
        ),
        # Default for unknown regions
        "DEFAULT": RegionRegulation(
            region_code="DEFAULT",
            region_name="Default (No specific regulations)",
            banned_chemicals=[],
            restricted_chemicals=[],
            notes="Always check local regulations before applying treatments"
        ),
    }

    @classmethod
    def get_regulation(cls, region_code: str) -> RegionRegulation:
        """Get regulation for a region, defaulting if unknown."""
        return cls.REGULATIONS.get(region_code, cls.REGULATIONS["DEFAULT"])


class TreatmentRecommendationService:
    """
    Service for generating treatment recommendations.

    Coordinates treatment database and regional regulations
    to provide appropriate, actionable recommendations.
    """

    def __init__(self, enable_region_filtering: bool = True):
        """
        Initialize treatment service.

        Args:
            enable_region_filtering: Whether to filter by regional regulations
        """
        self.enable_region_filtering = enable_region_filtering

    def get_recommendations(
        self,
        disease: str,
        region: Optional[str] = None,
        severity: Optional[str] = None,
        organic_only: bool = False
    ) -> dict:
        """
        Get treatment recommendations for a disease.

        Args:
            disease: Disease name
            region: Region code for regulatory filtering
            severity: Disease severity for urgency adjustment
            organic_only: Only return organic treatments

        Returns:
            Dictionary with organized treatment recommendations
        """
        # Get base treatment plan
        treatment_plan = TreatmentDatabase.get_treatment_plan(disease)

        if treatment_plan is None:
            logger.warning(f"No treatment plan found for disease: {disease}")
            return self._get_generic_recommendations(disease)

        # Get regional regulations
        regulation = None
        if self.enable_region_filtering and region:
            regulation = RegionRegulationDatabase.get_regulation(region)
            if regulation.organic_only:
                organic_only = True

        # Build recommendations
        recommendations = {
            "organic": self._format_treatments(
                treatment_plan.organic,
                regulation
            ),
            "chemical": [] if organic_only else self._format_treatments(
                treatment_plan.chemical,
                regulation
            ),
            "prevention": treatment_plan.prevention,
            "biological": self._format_treatments(
                treatment_plan.biological,
                regulation
            ) if treatment_plan.biological else None,
            "cultural": self._format_treatments(
                treatment_plan.cultural,
                regulation
            ) if treatment_plan.cultural else None,
            "urgency": self._adjust_urgency(treatment_plan.urgency, severity),
            "region_specific_notes": self._get_region_notes(regulation, disease),
        }

        # Filter out None values
        recommendations = {k: v for k, v in recommendations.items() if v is not None}

        return recommendations

    def _format_treatments(
        self,
        treatments: list[TreatmentOption],
        regulation: Optional[RegionRegulation]
    ) -> list[str]:
        """Format treatments for response, applying regional filtering."""
        formatted = []

        for treatment in treatments:
            # Check regional bans
            if regulation:
                if treatment.name in regulation.banned_chemicals:
                    continue
                if treatment.regions_banned and regulation.region_code in treatment.regions_banned:
                    continue

            # Format treatment string
            treatment_str = treatment.name
            if treatment.safety_notes:
                treatment_str += f" ({treatment.safety_notes})"

            # Add restriction note if applicable
            if regulation and treatment.name in regulation.restricted_chemicals:
                treatment_str += " [Restricted in your region - check local requirements]"

            formatted.append(treatment_str)

        return formatted

    def _adjust_urgency(self, base_urgency: str, severity: Optional[str]) -> str:
        """Adjust urgency based on disease severity."""
        if severity == "Critical" or severity == "Severe":
            return "immediate"
        elif severity == "Moderate" and base_urgency == "routine":
            return "soon"
        return base_urgency

    def _get_region_notes(
        self,
        regulation: Optional[RegionRegulation],
        disease: str
    ) -> Optional[str]:
        """Generate region-specific notes."""
        if not regulation or regulation.region_code == "DEFAULT":
            return "Always verify treatment options with local agricultural extension services."

        notes = [regulation.notes]

        if regulation.banned_chemicals:
            notes.append(
                f"Note: Some common treatments for {disease} may be restricted in {regulation.region_name}."
            )

        return " ".join(notes)

    def _get_generic_recommendations(self, disease: str) -> dict:
        """
        Provide generic recommendations when specific plan is unavailable.
        """
        return {
            "organic": [
                "Neem oil spray (general fungicide/insecticide)",
                "Remove and destroy infected plant material",
                "Improve air circulation around plants"
            ],
            "chemical": [
                "Consult local agricultural extension for specific fungicide recommendations"
            ],
            "prevention": [
                "Practice crop rotation",
                "Maintain proper plant spacing",
                "Avoid overhead irrigation",
                "Remove plant debris at end of season",
                "Use disease-resistant varieties when available"
            ],
            "urgency": "routine",
            "region_specific_notes": (
                f"No specific treatment plan found for '{disease}'. "
                "Please consult local agricultural extension services for targeted recommendations."
            )
        }

    def add_disease_treatment(self, treatment_plan: TreatmentPlan) -> None:
        """
        Add a new disease treatment plan.

        Iteration Point: Allows extending the system with new diseases.
        """
        TreatmentDatabase.TREATMENTS[treatment_plan.disease] = treatment_plan
        logger.info(f"Added treatment plan for: {treatment_plan.disease}")

    def add_regional_regulation(self, regulation: RegionRegulation) -> None:
        """
        Add or update regional regulation.

        Iteration Point: Allows adding new region support.
        """
        RegionRegulationDatabase.REGULATIONS[regulation.region_code] = regulation
        logger.info(f"Added/updated regulation for: {regulation.region_code}")
