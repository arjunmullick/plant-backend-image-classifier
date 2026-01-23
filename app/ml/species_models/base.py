"""
Base interfaces and data structures for species identification models.

Provides:
- SpeciesModelInterface: Abstract base for all species models
- SpeciesPrediction: Individual model prediction
- NormalizedTaxonomy: Standardized taxonomy representation
- SpeciesConsensus: Multi-model consensus result
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from PIL import Image


class TaxonomyDepth(str, Enum):
    """Depth of taxonomy identification."""
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"
    SUBSPECIES = "subspecies"
    UNKNOWN = "unknown"


@dataclass
class NormalizedTaxonomy:
    """
    Standardized taxonomy representation.

    All species models must normalize their output to this format
    for consistent consensus computation.
    """
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None  # Full binomial: "Solanum lycopersicum"
    subspecies: Optional[str] = None
    common_names: List[str] = field(default_factory=list)

    # Metadata
    taxonomy_depth: TaxonomyDepth = TaxonomyDepth.UNKNOWN
    source_label: Optional[str] = None  # Original label from model
    gbif_id: Optional[int] = None  # GBIF backbone taxonomy ID

    @property
    def binomial(self) -> Optional[str]:
        """Get binomial name (Genus species)."""
        if self.genus and self.species:
            # Species might already include genus
            if self.species.startswith(self.genus):
                return self.species
            return f"{self.genus} {self.species}"
        return self.species

    @property
    def primary_common_name(self) -> Optional[str]:
        """Get primary common name."""
        return self.common_names[0] if self.common_names else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "family": self.family,
            "genus": self.genus,
            "species": self.binomial,
            "common_name": self.primary_common_name,
            "common_names": self.common_names,
            "taxonomy_depth": self.taxonomy_depth.value,
        }


@dataclass
class SpeciesPrediction:
    """
    Individual species prediction from a single model.

    Each model produces one of these, which are then aggregated
    by the consensus engine.
    """
    # Model identification
    model_name: str
    model_type: str

    # Prediction
    taxonomy: NormalizedTaxonomy
    confidence: float  # 0.0 - 1.0

    # Per-level confidence (if available)
    family_confidence: Optional[float] = None
    genus_confidence: Optional[float] = None
    species_confidence: Optional[float] = None

    # Alternative predictions
    alternatives: List['SpeciesPrediction'] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0
    raw_label: Optional[str] = None
    error: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def species_name(self) -> Optional[str]:
        """Convenience accessor for species binomial."""
        return self.taxonomy.binomial

    @property
    def taxonomy_depth(self) -> TaxonomyDepth:
        """Get the depth of this prediction."""
        return self.taxonomy.taxonomy_depth


@dataclass
class TaxonomyLevel:
    """
    Single taxonomy level with confidence and model support.

    Used in consensus results to show agreement at each level.
    """
    name: str
    confidence: float
    supporting_models: List[str] = field(default_factory=list)
    alternative_names: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "supporting_models": self.supporting_models,
            "alternative_names": self.alternative_names,
        }


@dataclass
class TaxonomyDisagreement:
    """Records a disagreement between models."""
    model: str
    prediction: str
    confidence: float
    level: str  # "family", "genus", or "species"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 3),
            "level": self.level,
        }


@dataclass
class SpeciesConsensus:
    """
    Multi-model species consensus result.

    Aggregates predictions from multiple models into a single
    consensus with confidence and disagreement tracking.
    """
    # Consensus taxonomy at each level
    family: TaxonomyLevel
    genus: TaxonomyLevel
    species: TaxonomyLevel

    # Common name (most frequent across models)
    common_name: Optional[str] = None

    # Overall metrics
    overall_confidence: float = 0.0
    agreement_score: float = 0.0  # 0.0 - 1.0

    # Model tracking
    supporting_models: List[str] = field(default_factory=list)
    total_models: int = 0

    # Disagreements
    disagreements: List[TaxonomyDisagreement] = field(default_factory=list)

    # Explanation
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "family": self.family.to_dict(),
            "genus": self.genus.to_dict(),
            "species": self.species.to_dict(),
            "common_name": self.common_name,
            "overall_confidence": round(self.overall_confidence, 3),
            "agreement_score": round(self.agreement_score, 3),
            "supporting_models": self.supporting_models,
            "total_models": self.total_models,
            "disagreements": [d.to_dict() for d in self.disagreements],
            "notes": self.notes,
        }


class SpeciesModelInterface(ABC):
    """
    Abstract base class for species identification models.

    All species models (HuggingFace, external APIs, local) must implement
    this interface to participate in the consensus system.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type identifier (e.g., 'plantnet', 'inaturalist_vit')."""
        pass

    @property
    @abstractmethod
    def taxonomy_depth(self) -> TaxonomyDepth:
        """Maximum taxonomy depth this model can provide."""
        pass

    @property
    def priority(self) -> float:
        """
        Model priority for consensus weighting (0.0 - 2.0).
        Higher priority models have more influence on consensus.
        Default is 1.0.
        """
        return 1.0

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return True

    @abstractmethod
    async def predict(self, image: Image.Image) -> SpeciesPrediction:
        """
        Predict species from image.

        Args:
            image: PIL Image to classify

        Returns:
            SpeciesPrediction with taxonomy and confidence
        """
        pass

    async def predict_batch(self, images: List[Image.Image]) -> List[SpeciesPrediction]:
        """
        Predict species for multiple images.

        Default implementation processes sequentially.
        Override for batch optimization.
        """
        import asyncio
        return await asyncio.gather(*[self.predict(img) for img in images])

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for API responses."""
        return {
            "name": self.model_name,
            "type": self.model_type,
            "taxonomy_depth": self.taxonomy_depth.value,
            "priority": self.priority,
            "is_loaded": self.is_loaded,
        }
