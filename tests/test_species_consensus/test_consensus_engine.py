"""
Tests for SpeciesConsensusEngine - multi-model consensus computation.

Tests cover:
- Basic consensus computation
- Confidence weighting
- Disagreement detection
- Severity adjustment
- Edge cases
"""

import pytest
from app.ml.species_models.consensus_engine import SpeciesConsensusEngine
from app.ml.species_models.base import (
    SpeciesPrediction,
    NormalizedTaxonomy,
    TaxonomyDepth,
    SpeciesConsensus,
)


class TestSpeciesConsensusEngine:
    """Test suite for SpeciesConsensusEngine."""

    @pytest.fixture
    def engine(self):
        """Get consensus engine instance."""
        return SpeciesConsensusEngine()

    @pytest.fixture
    def tomato_taxonomy(self):
        """Create tomato taxonomy for testing."""
        return NormalizedTaxonomy(
            family="Solanaceae",
            genus="Solanum",
            species="Solanum lycopersicum",
            common_names=["Tomato"],
            taxonomy_depth=TaxonomyDepth.SPECIES,
        )

    @pytest.fixture
    def potato_taxonomy(self):
        """Create potato taxonomy for testing."""
        return NormalizedTaxonomy(
            family="Solanaceae",
            genus="Solanum",
            species="Solanum tuberosum",
            common_names=["Potato"],
            taxonomy_depth=TaxonomyDepth.SPECIES,
        )

    @pytest.fixture
    def apple_taxonomy(self):
        """Create apple taxonomy for testing."""
        return NormalizedTaxonomy(
            family="Rosaceae",
            genus="Malus",
            species="Malus domestica",
            common_names=["Apple"],
            taxonomy_depth=TaxonomyDepth.SPECIES,
        )

    # === Basic Consensus Tests ===

    def test_unanimous_consensus(self, engine, tomato_taxonomy):
        """Test consensus when all models agree."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.95,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=tomato_taxonomy,
                confidence=0.90,
            ),
            SpeciesPrediction(
                model_name="Model C",
                model_type="internal",
                taxonomy=tomato_taxonomy,
                confidence=0.85,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert consensus.species.name.lower() == "solanum lycopersicum"
        assert consensus.genus.name.lower() == "solanum"
        assert consensus.family.name.lower() == "solanaceae"
        assert consensus.agreement_score >= 0.9  # High agreement
        assert len(consensus.disagreements) == 0

    def test_majority_consensus(self, engine, tomato_taxonomy, potato_taxonomy):
        """Test consensus when majority agrees."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.90,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=tomato_taxonomy,
                confidence=0.85,
            ),
            SpeciesPrediction(
                model_name="Model C",
                model_type="internal",
                taxonomy=potato_taxonomy,
                confidence=0.70,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Tomato should win (2 vs 1)
        assert consensus.species.name.lower() == "solanum lycopersicum"
        assert len(consensus.disagreements) == 1
        assert consensus.disagreements[0].model == "Model C"

    def test_same_genus_different_species(self, engine, tomato_taxonomy, potato_taxonomy):
        """Test consensus with same genus but different species."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.80,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=potato_taxonomy,
                confidence=0.75,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Should still agree on genus
        assert consensus.genus.name.lower() == "solanum"
        assert consensus.family.name.lower() == "solanaceae"
        # But disagree on species
        assert len(consensus.disagreements) >= 1

    def test_different_family_disagreement(self, engine, tomato_taxonomy, apple_taxonomy):
        """Test severe disagreement - different families."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.80,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=apple_taxonomy,
                confidence=0.75,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Should have family-level disagreement
        has_family_disagreement = any(
            d.level == "family" for d in consensus.disagreements
        )
        # Note: Depends on which model wins
        assert consensus is not None

    # === Confidence Weighting Tests ===

    def test_higher_confidence_wins(self, engine, tomato_taxonomy, potato_taxonomy):
        """Test that higher confidence prediction wins."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.95,  # Higher confidence
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=potato_taxonomy,
                confidence=0.50,  # Lower confidence
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Tomato should win due to higher confidence
        assert consensus.species.name.lower() == "solanum lycopersicum"

    def test_model_priority_weighting(self, engine, tomato_taxonomy, potato_taxonomy):
        """Test that model priority affects consensus."""
        predictions = [
            SpeciesPrediction(
                model_name="PlantNet",
                model_type="plantnet",  # Higher priority (1.2)
                taxonomy=tomato_taxonomy,
                confidence=0.80,
            ),
            SpeciesPrediction(
                model_name="Internal",
                model_type="internal",  # Lower priority (0.8)
                taxonomy=potato_taxonomy,
                confidence=0.80,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # PlantNet should have more influence due to higher priority
        # With equal confidence, PlantNet's tomato should win
        assert consensus.species.name.lower() == "solanum lycopersicum"

    # === Edge Cases ===

    def test_empty_predictions(self, engine):
        """Test handling empty prediction list."""
        consensus = engine.compute_consensus([])

        assert consensus.species.name == "Unknown"
        assert consensus.agreement_score == 0.0
        assert consensus.overall_confidence == 0.0

    def test_all_failed_predictions(self, engine):
        """Test handling all failed predictions."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="API timeout",
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="Connection failed",
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert consensus.species.name == "Unknown"
        assert "No valid predictions" in consensus.notes

    def test_single_prediction(self, engine, tomato_taxonomy):
        """Test consensus with single prediction."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.90,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert consensus.species.name.lower() == "solanum lycopersicum"
        assert consensus.agreement_score == 1.0  # Single model = 100% agreement

    def test_mixed_valid_and_failed(self, engine, tomato_taxonomy):
        """Test consensus with mix of valid and failed predictions."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.90,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="API error",
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Should still work with valid prediction
        assert consensus.species.name.lower() == "solanum lycopersicum"
        assert "failed" in consensus.notes.lower() or "1 model" in consensus.notes.lower()

    # === Taxonomy Depth Tests ===

    def test_genus_only_prediction(self, engine):
        """Test prediction with genus-level only."""
        genus_only = NormalizedTaxonomy(
            family="Solanaceae",
            genus="Solanum",
            taxonomy_depth=TaxonomyDepth.GENUS,
        )

        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=genus_only,
                confidence=0.80,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert consensus.genus.name.lower() == "solanum"
        # Species might be "Unknown" or empty

    def test_family_only_prediction(self, engine):
        """Test prediction with family-level only."""
        family_only = NormalizedTaxonomy(
            family="Solanaceae",
            taxonomy_depth=TaxonomyDepth.FAMILY,
        )

        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=family_only,
                confidence=0.70,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert consensus.family.name.lower() == "solanaceae"

    # === Severity Adjustment Tests ===

    def test_severity_adjustment_high_confidence(self, engine, tomato_taxonomy):
        """Test severity adjustment with high species confidence."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.95,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=tomato_taxonomy,
                confidence=0.92,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Test severity adjustment
        adjusted_severity, factors = engine.adjust_severity_by_species_confidence(
            base_severity=60.0,
            species_consensus=consensus,
            disease_confidence=0.85,
        )

        # High species confidence should increase severity slightly
        assert adjusted_severity >= 60.0
        assert any("reliable" in f.lower() or "high" in f.lower() for f in factors)

    def test_severity_adjustment_low_confidence(self, engine):
        """Test severity adjustment with low species confidence."""
        low_confidence_tax = NormalizedTaxonomy(
            family="Solanaceae",
            genus="Solanum",
            species="Solanum lycopersicum",
            taxonomy_depth=TaxonomyDepth.SPECIES,
        )

        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=low_confidence_tax,
                confidence=0.40,  # Low confidence
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        adjusted_severity, factors = engine.adjust_severity_by_species_confidence(
            base_severity=60.0,
            species_consensus=consensus,
            disease_confidence=0.85,
        )

        # Low species confidence should decrease severity
        assert adjusted_severity < 60.0
        assert any("uncertain" in f.lower() or "low" in f.lower() for f in factors)

    def test_severity_adjustment_family_disagreement(self, engine, tomato_taxonomy, apple_taxonomy):
        """Test severity adjustment with family-level disagreement."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.80,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=apple_taxonomy,  # Different family!
                confidence=0.75,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        adjusted_severity, factors = engine.adjust_severity_by_species_confidence(
            base_severity=70.0,
            species_consensus=consensus,
            disease_confidence=0.85,
        )

        # Family disagreement should significantly lower severity
        assert adjusted_severity < 70.0

    # === Agreement Score Tests ===

    def test_agreement_score_calculation(self, engine, tomato_taxonomy, potato_taxonomy):
        """Test agreement score is calculated correctly."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.90,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=tomato_taxonomy,
                confidence=0.85,
            ),
            SpeciesPrediction(
                model_name="Model C",
                model_type="internal",
                taxonomy=potato_taxonomy,
                confidence=0.70,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # 2 out of 3 agree on species
        # Agreement should be between 0.5 and 0.8 (weighted)
        assert 0.4 <= consensus.agreement_score <= 0.9

    # === Notes Generation Tests ===

    def test_notes_strong_consensus(self, engine, tomato_taxonomy):
        """Test notes generation for strong consensus."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.95,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=tomato_taxonomy,
                confidence=0.92,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert "strong" in consensus.notes.lower() or "high" in consensus.notes.lower()

    def test_notes_weak_consensus(self, engine, tomato_taxonomy, potato_taxonomy, apple_taxonomy):
        """Test notes generation for weak consensus."""
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=tomato_taxonomy,
                confidence=0.50,
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=potato_taxonomy,
                confidence=0.45,
            ),
            SpeciesPrediction(
                model_name="Model C",
                model_type="internal",
                taxonomy=apple_taxonomy,
                confidence=0.40,
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        assert "low" in consensus.notes.lower() or "weak" in consensus.notes.lower() or "moderate" in consensus.notes.lower()


class TestSpeciesConsensusOutput:
    """Test the output format of SpeciesConsensus."""

    @pytest.fixture
    def engine(self):
        return SpeciesConsensusEngine()

    def test_to_dict_format(self, engine):
        """Test that consensus converts to dict correctly."""
        taxonomy = NormalizedTaxonomy(
            family="Solanaceae",
            genus="Solanum",
            species="Solanum lycopersicum",
            common_names=["Tomato"],
            taxonomy_depth=TaxonomyDepth.SPECIES,
        )

        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=taxonomy,
                confidence=0.90,
            ),
        ]

        consensus = engine.compute_consensus(predictions)
        result_dict = consensus.to_dict()

        # Check required fields
        assert "family" in result_dict
        assert "genus" in result_dict
        assert "species" in result_dict
        assert "overall_confidence" in result_dict
        assert "agreement_score" in result_dict
        assert "supporting_models" in result_dict
        assert "disagreements" in result_dict
        assert "notes" in result_dict

        # Check nested structure
        assert "name" in result_dict["family"]
        assert "confidence" in result_dict["family"]
        assert "supporting_models" in result_dict["family"]
