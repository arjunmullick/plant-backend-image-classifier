"""
Species Models Package

Provides multi-model species identification with hierarchical taxonomy support
and confidence-weighted consensus.

Components:
- SpeciesModelInterface: Base interface for all species models
- SpeciesModelRegistry: Manages and coordinates species models
- SpeciesConsensusEngine: Computes consensus from multiple predictions
- TaxonomyResolver: Normalizes taxonomy names across models
"""

from app.ml.species_models.base import (
    SpeciesModelInterface,
    SpeciesPrediction,
    NormalizedTaxonomy,
    TaxonomyDepth,
    SpeciesConsensus,
    TaxonomyLevel,
)
from app.ml.species_models.registry import (
    SpeciesModelRegistry,
    SpeciesModelType,
    get_species_model_registry,
)
from app.ml.species_models.consensus_engine import SpeciesConsensusEngine
from app.ml.species_models.taxonomy_resolver import TaxonomyResolver, get_taxonomy_resolver

__all__ = [
    "SpeciesModelInterface",
    "SpeciesPrediction",
    "NormalizedTaxonomy",
    "TaxonomyDepth",
    "SpeciesConsensus",
    "TaxonomyLevel",
    "SpeciesModelRegistry",
    "SpeciesModelType",
    "get_species_model_registry",
    "SpeciesConsensusEngine",
    "TaxonomyResolver",
    "get_taxonomy_resolver",
]
