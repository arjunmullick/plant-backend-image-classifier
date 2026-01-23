# Multi-Model Species Consensus Architecture

## Overview

This document describes the enhanced architecture for deep hierarchical plant identification
using multiple open-source species models with confidence-weighted consensus.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ POST /classify  │  │ POST /compare   │  │ POST /early-warning             │  │
│  │ (Full Analysis) │  │ (Compare Models)│  │ (Early Warning + Species)       │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────────┬────────────────┘  │
└───────────┼─────────────────────┼───────────────────────────┼───────────────────┘
            │                     │                           │
            ▼                     ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Orchestration Layer                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ClassificationService (Enhanced)                      │   │
│  │  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐ │   │
│  │  │ Single Mode │  │ Multi-Model     │  │ Consensus Mode               │ │   │
│  │  │ (Default)   │  │ Comparison      │  │ (Early Warning + Full)       │ │   │
│  │  └─────────────┘  └─────────────────┘  └──────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
            │                     │                           │
            ▼                     ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Species Consensus Engine (NEW)                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      SpeciesConsensusEngine                              │   │
│  │  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │   │
│  │  │ Run All Models │  │ Normalize        │  │ Compute Weighted        │  │   │
│  │  │ in Parallel    │→ │ Taxonomy Outputs │→ │ Consensus               │  │   │
│  │  └────────────────┘  └──────────────────┘  └─────────────────────────┘  │   │
│  │                                                        │                 │   │
│  │  ┌────────────────────────────────────────────────────┴──────────────┐  │   │
│  │  │                    Output: SpeciesConsensus                        │  │   │
│  │  │  • family: {name, confidence, supporting_models}                   │  │   │
│  │  │  • genus: {name, confidence, supporting_models}                    │  │   │
│  │  │  • species: {name, confidence, supporting_models}                  │  │   │
│  │  │  • disagreements: [{model, prediction, confidence}]                │  │   │
│  │  │  • agreement_score: float (0-1)                                    │  │   │
│  │  └───────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
            │                     │                           │
            ▼                     ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Species Model Registry (NEW)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     SpeciesModelRegistry                                 │   │
│  │                                                                          │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │ Model Type          │ Source      │ Taxonomy Depth │ Priority   │    │   │
│  │  ├─────────────────────┼─────────────┼────────────────┼────────────│    │   │
│  │  │ PLANTNET_API        │ External    │ Full (F→G→S)   │ High       │    │   │
│  │  │ KINDWISE_API        │ External    │ Full (F→G→S)   │ High       │    │   │
│  │  │ INATURALIST_VIT     │ HuggingFace │ Full (F→G→S)   │ High       │    │   │
│  │  │ PLANTCLEF_SWIN      │ HuggingFace │ Full (F→G→S)   │ Medium     │    │   │
│  │  │ EFFICIENTNET_FLORA  │ HuggingFace │ Species only   │ Medium     │    │   │
│  │  │ MOBILENET_PLANT     │ HuggingFace │ Species only   │ Low        │    │   │
│  │  │ INTERNAL            │ Local       │ Full (F→G→S)   │ Baseline   │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Individual Species Models                                    │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐                │
│  │ PlantNetModel    │ │ iNaturalistViT   │ │ PlantCLEFSwin    │                │
│  │ (External API)   │ │ (HuggingFace)    │ │ (HuggingFace)    │                │
│  │                  │ │                  │ │                  │                │
│  │ • 50K+ species   │ │ • 10K+ species   │ │ • PlantCLEF      │                │
│  │ • Full taxonomy  │ │ • iNat trained   │ │ • Competition    │                │
│  │ • Common names   │ │ • Full taxonomy  │ │   winner arch    │                │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘                │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐                │
│  │ EfficientNetFlora│ │ KindwiseModel    │ │ InternalModel    │                │
│  │ (HuggingFace)    │ │ (External API)   │ │ (Local)          │                │
│  │                  │ │                  │ │                  │                │
│  │ • Efficient arch │ │ • Plant.id API   │ │ • 20 species     │                │
│  │ • Good balance   │ │ • Health assess  │ │ • Baseline       │                │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Taxonomy Resolution Layer (NEW)                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      TaxonomyResolver                                    │   │
│  │                                                                          │   │
│  │  Input Normalization:                                                    │   │
│  │  • "Tomato" → Solanum lycopersicum (Solanaceae)                         │   │
│  │  • "Solanum lycopersicum L." → Solanum lycopersicum                     │   │
│  │  • "Cherry Tomato" → Solanum lycopersicum var. cerasiforme              │   │
│  │                                                                          │   │
│  │  Data Sources:                                                           │   │
│  │  • GBIF Backbone Taxonomy (800K+ species)                               │   │
│  │  • NCBI Taxonomy (1.9M+ taxa)                                           │   │
│  │  • Local cache for common crops                                          │   │
│  │                                                                          │   │
│  │  Output: NormalizedTaxonomy                                              │   │
│  │  • family: str                                                           │   │
│  │  • genus: str                                                            │   │
│  │  • species: str                                                          │   │
│  │  • common_names: list[str]                                               │   │
│  │  • taxonomy_depth: TaxonomyDepth (FAMILY, GENUS, SPECIES)               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Consensus Algorithm

### Confidence-Weighted Voting

```python
def compute_species_consensus(predictions: list[SpeciesPrediction]) -> SpeciesConsensus:
    """
    Compute consensus using confidence-weighted voting with taxonomy depth rewards.

    Algorithm:
    1. Group predictions by normalized species name
    2. For each group:
       - Sum confidence scores (weighted by model priority)
       - Award bonus for deeper taxonomy (species > genus > family)
    3. Select top species by weighted score
    4. Propagate upward: species → genus → family
    5. Calculate agreement score
    6. Identify disagreements
    """

    # Taxonomy depth weights (reward specificity)
    DEPTH_WEIGHTS = {
        TaxonomyDepth.SPECIES: 1.0,    # Full credit for species
        TaxonomyDepth.GENUS: 0.7,      # Partial credit for genus only
        TaxonomyDepth.FAMILY: 0.4,     # Minimal credit for family only
    }

    # Model priority weights
    MODEL_WEIGHTS = {
        "plantnet": 1.2,      # High priority (large species database)
        "kindwise": 1.2,      # High priority (verified accuracy)
        "inaturalist_vit": 1.1,
        "plantclef_swin": 1.0,
        "efficientnet_flora": 0.9,
        "internal": 0.8,      # Baseline (limited species)
    }

    # Compute weighted scores per species
    species_scores = defaultdict(lambda: {"score": 0, "supporters": [], "confidences": []})

    for pred in predictions:
        if pred.error:
            continue

        normalized = taxonomy_resolver.normalize(pred.species_name)
        depth_weight = DEPTH_WEIGHTS[pred.taxonomy_depth]
        model_weight = MODEL_WEIGHTS.get(pred.model_type, 1.0)

        weighted_confidence = pred.confidence * depth_weight * model_weight

        species_scores[normalized.species]["score"] += weighted_confidence
        species_scores[normalized.species]["supporters"].append(pred.model_name)
        species_scores[normalized.species]["confidences"].append(pred.confidence)

    # Find consensus species
    top_species = max(species_scores.items(), key=lambda x: x[1]["score"])

    # Calculate agreement
    total_models = len([p for p in predictions if not p.error])
    agreement = len(top_species[1]["supporters"]) / total_models

    # Find disagreements
    disagreements = [
        {
            "model": pred.model_name,
            "prediction": pred.species_name,
            "confidence": pred.confidence
        }
        for pred in predictions
        if pred.species_name != top_species[0] and not pred.error
    ]

    return SpeciesConsensus(
        family=normalized.family,
        genus=normalized.genus,
        species=top_species[0],
        confidence=mean(top_species[1]["confidences"]),
        agreement=agreement,
        supporting_models=top_species[1]["supporters"],
        disagreements=disagreements
    )
```

## API Response Enhancements

### 1. Full Analysis (`POST /classify`)

```json
{
  "plant": {
    "family": {"name": "Solanaceae", "confidence": 0.95},
    "genus": {"name": "Solanum", "confidence": 0.92},
    "species": {"name": "Solanum lycopersicum", "confidence": 0.89},
    "common_name": "Tomato",
    "alternative_species": [...]
  },
  "taxonomy_consensus": {
    "family": {
      "name": "Solanaceae",
      "confidence": 0.96,
      "supporting_models": ["plantnet", "kindwise", "inaturalist_vit", "internal"]
    },
    "genus": {
      "name": "Solanum",
      "confidence": 0.93,
      "supporting_models": ["plantnet", "kindwise", "inaturalist_vit"]
    },
    "species": {
      "name": "Solanum lycopersicum",
      "confidence": 0.88,
      "supporting_models": ["plantnet", "kindwise"]
    },
    "agreement_score": 0.85,
    "disagreements": [
      {
        "model": "efficientnet_flora",
        "prediction": "Solanum tuberosum",
        "confidence": 0.41,
        "level": "species"
      }
    ]
  },
  "health": {...},
  "treatment": {...}
}
```

### 2. Compare Models (`POST /classify/compare`)

```json
{
  "models": {
    "plantnet": {
      "species": {
        "family": "Solanaceae",
        "genus": "Solanum",
        "species": "Solanum lycopersicum",
        "common_name": "Tomato"
      },
      "confidence": 0.94,
      "taxonomy_depth": "species",
      "processing_time_ms": 320,
      "raw_label": "Solanum lycopersicum L."
    },
    "inaturalist_vit": {
      "species": {
        "family": "Solanaceae",
        "genus": "Solanum",
        "species": "Solanum lycopersicum",
        "common_name": "Cherry Tomato"
      },
      "confidence": 0.87,
      "taxonomy_depth": "species",
      "processing_time_ms": 145
    }
  },
  "taxonomy_agreement": {
    "family_agreement": 1.0,
    "genus_agreement": 1.0,
    "species_agreement": 0.8,
    "divergence_level": "species",
    "divergence_details": [
      {
        "model_a": "plantnet",
        "model_b": "efficientnet",
        "level": "species",
        "severity": "minor"
      }
    ]
  }
}
```

### 3. Early Warning (`POST /classify/early-warning`)

```json
{
  "species_consensus": {
    "family": "Solanaceae",
    "genus": "Solanum",
    "species": "Solanum lycopersicum",
    "common_name": "Tomato",
    "confidence": 0.91,
    "agreement": 0.88,
    "supporting_models": ["plantnet", "kindwise", "inaturalist_vit"],
    "notes": "High agreement (88%) across 4 models. Species identification reliable."
  },
  "disease_consensus": {
    "disease_name": "Early Blight",
    "confidence": 0.87,
    "model_agreement": 0.75,
    "supporting_models": [...],
    "reasoning": "..."
  },
  "severity": {
    "level": "MODERATE",
    "score": 65,
    "adjusted_by_species_confidence": true,
    "factors": [
      "Disease confidence: 87%",
      "Species confidence: 91% (reliable crop identification)",
      "Model agreement: 75%"
    ]
  }
}
```

## File Structure

```
app/ml/species_models/
├── __init__.py
├── base.py                    # SpeciesModelInterface, NormalizedTaxonomy
├── registry.py                # SpeciesModelRegistry (singleton)
├── consensus_engine.py        # SpeciesConsensusEngine
├── taxonomy_resolver.py       # TaxonomyResolver (name normalization)
├── models/
│   ├── __init__.py
│   ├── plantnet_model.py      # PlantNet API wrapper
│   ├── kindwise_model.py      # Kindwise API wrapper
│   ├── inaturalist_vit.py     # iNaturalist ViT from HuggingFace
│   ├── plantclef_swin.py      # PlantCLEF Swin from HuggingFace
│   ├── efficientnet_flora.py  # EfficientNet flora classifier
│   └── internal_model.py      # Wrapper around existing SpeciesClassifier
└── data/
    └── taxonomy_cache.json    # Local taxonomy lookup cache
```

## Integration Points

1. **ClassificationService**: Add `run_species_consensus()` method
2. **EarlyWarningService**: Add `_determine_species_consensus()` alongside existing disease consensus
3. **API Routes**: Update response schemas to include new consensus fields
4. **External Models**: Extend existing `ExternalModelRegistry` to include species models

## Performance Considerations

- **Parallel Execution**: All models run concurrently via `asyncio.gather()`
- **Caching**: Taxonomy lookups cached in memory and JSON file
- **Graceful Degradation**: If external APIs fail, continue with available models
- **Timeout Handling**: Individual model timeouts don't block others

## Migration Safety

- All existing endpoints remain backward compatible
- New fields are additive (no breaking changes)
- Consensus data is optional in responses
- Feature flags control new functionality
