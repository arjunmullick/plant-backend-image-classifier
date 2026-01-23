"""
Species Consensus Engine

Computes confidence-weighted consensus from multiple species model predictions.

Algorithm:
1. Normalize all predictions to standard taxonomy
2. Group predictions by species (with fuzzy matching)
3. Compute weighted scores based on confidence, model priority, and taxonomy depth
4. Select consensus species and propagate upward to genus and family
5. Calculate agreement score and identify disagreements
"""

import logging
from collections import defaultdict
from statistics import mean, stdev
from typing import List, Dict, Optional, Any, Tuple

from app.ml.species_models.base import (
    SpeciesPrediction,
    SpeciesConsensus,
    TaxonomyLevel,
    TaxonomyDisagreement,
    TaxonomyDepth,
    NormalizedTaxonomy,
)
from app.ml.species_models.taxonomy_resolver import TaxonomyResolver, get_taxonomy_resolver

logger = logging.getLogger(__name__)


class SpeciesConsensusEngine:
    """
    Engine for computing species consensus from multiple model predictions.

    Features:
    - Confidence-weighted voting
    - Taxonomy depth rewards (species > genus > family)
    - Model priority weighting
    - Disagreement tracking at each taxonomy level
    - Graceful degradation when models fail
    """

    # Taxonomy depth weights (reward specificity)
    DEPTH_WEIGHTS = {
        TaxonomyDepth.SPECIES: 1.0,
        TaxonomyDepth.SUBSPECIES: 1.0,
        TaxonomyDepth.GENUS: 0.7,
        TaxonomyDepth.FAMILY: 0.4,
        TaxonomyDepth.UNKNOWN: 0.3,
    }

    # Default model priority weights
    DEFAULT_MODEL_WEIGHTS = {
        "plantnet": 1.2,          # Large species database, high accuracy
        "kindwise": 1.2,          # Verified accuracy, health assessment
        "inaturalist_vit": 1.1,   # Community-verified, good coverage
        "plantclef_swin": 1.0,    # Competition-winning architecture
        "efficientnet_flora": 0.9,
        "mobilenet_plant": 0.8,
        "internal": 0.8,          # Limited species coverage
    }

    def __init__(
        self,
        taxonomy_resolver: Optional[TaxonomyResolver] = None,
        model_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize consensus engine.

        Args:
            taxonomy_resolver: Resolver for normalizing species names
            model_weights: Custom model priority weights
        """
        self.taxonomy_resolver = taxonomy_resolver or get_taxonomy_resolver()
        self.model_weights = model_weights or self.DEFAULT_MODEL_WEIGHTS.copy()

    def compute_consensus(
        self,
        predictions: List[SpeciesPrediction]
    ) -> SpeciesConsensus:
        """
        Compute species consensus from multiple predictions.

        Args:
            predictions: List of SpeciesPrediction from different models

        Returns:
            SpeciesConsensus with aggregated taxonomy and agreement metrics
        """
        # Filter valid predictions
        valid_predictions = [p for p in predictions if not p.has_error and p.taxonomy]
        total_models = len(predictions)
        valid_models = len(valid_predictions)

        if valid_models == 0:
            return self._empty_consensus(total_models)

        # Normalize all predictions
        normalized = self._normalize_predictions(valid_predictions)

        # Compute consensus at each level
        family_consensus = self._compute_level_consensus(
            normalized, "family", valid_models
        )
        genus_consensus = self._compute_level_consensus(
            normalized, "genus", valid_models
        )
        species_consensus = self._compute_level_consensus(
            normalized, "species", valid_models
        )

        # Get most common common name
        common_name = self._get_consensus_common_name(normalized)

        # Calculate overall metrics
        overall_confidence = self._compute_overall_confidence(
            family_consensus, genus_consensus, species_consensus
        )
        agreement_score = self._compute_agreement_score(
            species_consensus, genus_consensus, family_consensus, valid_models
        )

        # Identify disagreements
        disagreements = self._identify_disagreements(
            normalized, species_consensus, genus_consensus, family_consensus
        )

        # Generate notes
        notes = self._generate_notes(
            agreement_score, valid_models, total_models, disagreements
        )

        return SpeciesConsensus(
            family=family_consensus,
            genus=genus_consensus,
            species=species_consensus,
            common_name=common_name,
            overall_confidence=overall_confidence,
            agreement_score=agreement_score,
            supporting_models=[p.model_name for p in valid_predictions],
            total_models=total_models,
            disagreements=disagreements,
            notes=notes,
        )

    def _normalize_predictions(
        self,
        predictions: List[SpeciesPrediction]
    ) -> List[Dict[str, Any]]:
        """Normalize all predictions using taxonomy resolver."""
        normalized = []

        for pred in predictions:
            # Get the species name to normalize
            species_name = pred.species_name or ""

            # If no species name, try to get from taxonomy
            if not species_name and pred.taxonomy:
                species_name = (
                    pred.taxonomy.species or
                    pred.taxonomy.genus or
                    pred.taxonomy.family or
                    ""
                )

            # Normalize through taxonomy resolver
            resolved = self.taxonomy_resolver.normalize(
                species_name,
                source_model=pred.model_type
            )

            # Use provided taxonomy if better
            if pred.taxonomy.family and not resolved.family:
                resolved.family = pred.taxonomy.family
            if pred.taxonomy.genus and not resolved.genus:
                resolved.genus = pred.taxonomy.genus

            normalized.append({
                "model_name": pred.model_name,
                "model_type": pred.model_type,
                "confidence": pred.confidence,
                "taxonomy": resolved,
                "original": pred,
                "family": resolved.family,
                "genus": resolved.genus,
                "species": resolved.binomial or resolved.species,
                "common_names": resolved.common_names,
                "depth": resolved.taxonomy_depth,
            })

        return normalized

    def _compute_level_consensus(
        self,
        normalized: List[Dict[str, Any]],
        level: str,
        total_valid: int
    ) -> TaxonomyLevel:
        """Compute consensus for a single taxonomy level."""
        # Group by name at this level
        groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"score": 0.0, "supporters": [], "confidences": []}
        )

        for pred in normalized:
            name = pred.get(level)
            if not name:
                continue

            name_lower = name.lower()

            # Compute weighted score
            depth_weight = self.DEPTH_WEIGHTS.get(pred["depth"], 0.5)
            model_weight = self.model_weights.get(pred["model_type"], 1.0)
            weighted_score = pred["confidence"] * depth_weight * model_weight

            groups[name_lower]["score"] += weighted_score
            groups[name_lower]["supporters"].append(pred["model_name"])
            groups[name_lower]["confidences"].append(pred["confidence"])
            groups[name_lower]["original_name"] = name  # Preserve capitalization

        if not groups:
            return TaxonomyLevel(name="Unknown", confidence=0.0, supporting_models=[])

        # Find winner
        winner = max(groups.items(), key=lambda x: x[1]["score"])
        winner_key, winner_data = winner

        # Calculate confidence
        if winner_data["confidences"]:
            confidence = mean(winner_data["confidences"])
        else:
            confidence = 0.0

        # Get alternatives
        alternatives = []
        for key, data in sorted(
            groups.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[1:4]:  # Top 3 alternatives
            alternatives.append({
                "name": data.get("original_name", key),
                "confidence": mean(data["confidences"]) if data["confidences"] else 0.0,
                "supporting_models": data["supporters"],
            })

        return TaxonomyLevel(
            name=winner_data.get("original_name", winner_key.capitalize()),
            confidence=confidence,
            supporting_models=winner_data["supporters"],
            alternative_names=alternatives,
        )

    def _get_consensus_common_name(
        self,
        normalized: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Get most common common name across predictions."""
        name_counts: Dict[str, int] = defaultdict(int)

        for pred in normalized:
            for name in pred.get("common_names", []):
                name_counts[name.lower()] += 1

        if not name_counts:
            return None

        # Get most common, preserving original case
        most_common = max(name_counts.items(), key=lambda x: x[1])
        most_common_lower = most_common[0]

        # Find original case version
        for pred in normalized:
            for name in pred.get("common_names", []):
                if name.lower() == most_common_lower:
                    return name

        return most_common_lower.title()

    def _compute_overall_confidence(
        self,
        family: TaxonomyLevel,
        genus: TaxonomyLevel,
        species: TaxonomyLevel
    ) -> float:
        """Compute overall consensus confidence."""
        # Weighted average favoring species confidence
        weights = [0.2, 0.3, 0.5]  # family, genus, species
        confidences = [family.confidence, genus.confidence, species.confidence]

        weighted_sum = sum(w * c for w, c in zip(weights, confidences))
        return weighted_sum

    def _compute_agreement_score(
        self,
        species: TaxonomyLevel,
        genus: TaxonomyLevel,
        family: TaxonomyLevel,
        total_valid: int
    ) -> float:
        """Compute agreement score based on model support."""
        if total_valid == 0:
            return 0.0

        # Count unique models supporting consensus at each level
        species_agreement = len(species.supporting_models) / total_valid
        genus_agreement = len(genus.supporting_models) / total_valid
        family_agreement = len(family.supporting_models) / total_valid

        # Weighted average (species agreement most important)
        agreement = (
            0.5 * species_agreement +
            0.3 * genus_agreement +
            0.2 * family_agreement
        )

        return min(1.0, agreement)

    def _identify_disagreements(
        self,
        normalized: List[Dict[str, Any]],
        species_consensus: TaxonomyLevel,
        genus_consensus: TaxonomyLevel,
        family_consensus: TaxonomyLevel
    ) -> List[TaxonomyDisagreement]:
        """Identify disagreements from consensus."""
        disagreements = []
        consensus_species = species_consensus.name.lower()
        consensus_genus = genus_consensus.name.lower()
        consensus_family = family_consensus.name.lower()

        for pred in normalized:
            pred_species = (pred.get("species") or "").lower()
            pred_genus = (pred.get("genus") or "").lower()
            pred_family = (pred.get("family") or "").lower()

            # Check species disagreement
            if pred_species and pred_species != consensus_species:
                # Determine severity
                if pred_genus == consensus_genus:
                    level = "species"  # Same genus, different species
                elif pred_family == consensus_family:
                    level = "genus"  # Same family, different genus
                else:
                    level = "family"  # Different family (severe)

                disagreements.append(TaxonomyDisagreement(
                    model=pred["model_name"],
                    prediction=pred.get("species") or pred.get("genus") or pred.get("family") or "Unknown",
                    confidence=pred["confidence"],
                    level=level,
                ))

        return disagreements

    def _generate_notes(
        self,
        agreement: float,
        valid_models: int,
        total_models: int,
        disagreements: List[TaxonomyDisagreement]
    ) -> str:
        """Generate human-readable notes about the consensus."""
        notes_parts = []

        # Agreement level
        if agreement >= 0.9:
            notes_parts.append(f"Strong consensus ({agreement:.0%}) across {valid_models} models.")
        elif agreement >= 0.7:
            notes_parts.append(f"Good agreement ({agreement:.0%}) across {valid_models} models.")
        elif agreement >= 0.5:
            notes_parts.append(f"Moderate agreement ({agreement:.0%}) across {valid_models} models.")
        else:
            notes_parts.append(f"Low agreement ({agreement:.0%}) - species identification uncertain.")

        # Model failures
        failed_models = total_models - valid_models
        if failed_models > 0:
            notes_parts.append(f"{failed_models} model(s) failed to produce predictions.")

        # Disagreement severity
        family_disagreements = sum(1 for d in disagreements if d.level == "family")
        genus_disagreements = sum(1 for d in disagreements if d.level == "genus")

        if family_disagreements > 0:
            notes_parts.append(
                f"⚠️ {family_disagreements} model(s) identified a different plant family - "
                "consider manual verification."
            )
        elif genus_disagreements > 0:
            notes_parts.append(
                f"Some models ({genus_disagreements}) predicted a different genus - "
                "species may be ambiguous."
            )

        return " ".join(notes_parts)

    def _empty_consensus(self, total_models: int) -> SpeciesConsensus:
        """Return empty consensus when no valid predictions."""
        return SpeciesConsensus(
            family=TaxonomyLevel(name="Unknown", confidence=0.0),
            genus=TaxonomyLevel(name="Unknown", confidence=0.0),
            species=TaxonomyLevel(name="Unknown", confidence=0.0),
            common_name=None,
            overall_confidence=0.0,
            agreement_score=0.0,
            supporting_models=[],
            total_models=total_models,
            disagreements=[],
            notes=f"No valid predictions from {total_models} model(s). Unable to identify species.",
        )

    def adjust_severity_by_species_confidence(
        self,
        base_severity: float,
        species_consensus: SpeciesConsensus,
        disease_confidence: float
    ) -> Tuple[float, List[str]]:
        """
        Adjust disease severity based on species identification confidence.

        Logic:
        - High disease confidence + low species confidence → downgrade severity
          (can't treat effectively if we don't know the plant)
        - High agreement across both → upgrade severity
          (confident diagnosis)

        Args:
            base_severity: Original severity score (0-100)
            species_consensus: Species consensus result
            disease_confidence: Disease detection confidence (0-1)

        Returns:
            Tuple of (adjusted_severity, list of adjustment factors)
        """
        factors = []
        adjustment = 0.0

        species_confidence = species_consensus.overall_confidence
        species_agreement = species_consensus.agreement_score

        # High species confidence and agreement → increase reliability
        if species_confidence >= 0.85 and species_agreement >= 0.8:
            adjustment += 5
            factors.append(f"Species confidence high ({species_confidence:.0%}) - reliable crop ID (+5)")
        elif species_confidence >= 0.7 and species_agreement >= 0.6:
            adjustment += 2
            factors.append(f"Species confidence good ({species_confidence:.0%}) (+2)")

        # Low species confidence → uncertainty in treatment
        if species_confidence < 0.5:
            adjustment -= 10
            factors.append(f"⚠️ Low species confidence ({species_confidence:.0%}) - treatment uncertain (-10)")
        elif species_confidence < 0.7:
            adjustment -= 5
            factors.append(f"Moderate species uncertainty ({species_confidence:.0%}) (-5)")

        # Low agreement → conflicting identifications
        if species_agreement < 0.5:
            adjustment -= 5
            factors.append(f"Low model agreement ({species_agreement:.0%}) - species disputed (-5)")

        # Family-level disagreements → major uncertainty
        family_disagreements = sum(
            1 for d in species_consensus.disagreements if d.level == "family"
        )
        if family_disagreements > 0:
            adjustment -= 10
            factors.append(f"⚠️ {family_disagreements} model(s) disagree on plant family (-10)")

        # Calculate final severity
        adjusted = max(0, min(100, base_severity + adjustment))

        return adjusted, factors
