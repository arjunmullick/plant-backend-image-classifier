"""
Enumerations for the plant classification system.

These enums provide type safety and clear documentation of valid values.
"""

from enum import Enum


class HealthStatus(str, Enum):
    """Plant health status classification."""
    HEALTHY = "Healthy"
    DISEASED = "Diseased"
    UNKNOWN = "Unknown"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels for explainability."""
    VERY_HIGH = "very_high"      # >= 0.95
    HIGH = "high"                # >= 0.85
    MODERATE = "moderate"        # >= 0.70
    LOW = "low"                  # >= 0.50
    VERY_LOW = "very_low"        # < 0.50

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert a numeric confidence score to a level."""
        if score >= 0.95:
            return cls.VERY_HIGH
        elif score >= 0.85:
            return cls.HIGH
        elif score >= 0.70:
            return cls.MODERATE
        elif score >= 0.50:
            return cls.LOW
        else:
            return cls.VERY_LOW


class TreatmentType(str, Enum):
    """Types of treatment recommendations."""
    ORGANIC = "organic"
    CHEMICAL = "chemical"
    CULTURAL = "cultural"  # Farming practices
    BIOLOGICAL = "biological"  # Beneficial organisms
    PREVENTIVE = "preventive"


class ImageQuality(str, Enum):
    """Assessed quality of input image."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class TaxonomicRank(str, Enum):
    """Taxonomic hierarchy levels."""
    KINGDOM = "kingdom"
    PHYLUM = "phylum"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"
