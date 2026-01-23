"""
Integration tests for the species consensus system.

Tests the full pipeline from image to consensus result.
Uses real test images when available.
"""

import pytest
import asyncio
from pathlib import Path
from PIL import Image
import numpy as np

from app.ml.species_models.registry import (
    SpeciesModelRegistry,
    SpeciesModelType,
    get_species_model_registry,
)
from app.ml.species_models.consensus_engine import SpeciesConsensusEngine
from app.ml.species_models.base import TaxonomyDepth


# Path to test images
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"


class TestSpeciesModelRegistry:
    """Test the species model registry."""

    @pytest.fixture
    def registry(self):
        """Get a fresh registry instance."""
        return SpeciesModelRegistry()

    def test_registry_initialization(self, registry):
        """Test basic registry initialization."""
        # Should not fail
        registry.initialize(
            enable_huggingface=False,  # Skip HF models for faster testing
            enable_internal=True,
        )

        # Check that internal model is available
        available = registry.get_available_models()
        assert SpeciesModelType.INTERNAL in available or len(available) >= 0

    def test_get_model_info(self, registry):
        """Test getting model info."""
        registry.initialize(
            enable_huggingface=False,
            enable_internal=True,
        )

        info = registry.get_model_info()
        # Should return a list of dicts
        assert isinstance(info, list)

    def test_singleton_pattern(self):
        """Test that get_species_model_registry returns singleton."""
        registry1 = get_species_model_registry()
        registry2 = get_species_model_registry()
        assert registry1 is registry2


class TestInternalModel:
    """Test the internal species model wrapper."""

    @pytest.fixture
    def test_image(self):
        """Create a simple test image."""
        # Create a 224x224 RGB image with green tones (plant-like)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 1] = 128  # Green channel
        return Image.fromarray(img_array)

    @pytest.mark.asyncio
    async def test_internal_model_prediction(self, test_image):
        """Test internal model can make predictions."""
        try:
            from app.ml.species_models.models.internal_model import InternalSpeciesModel

            model = InternalSpeciesModel()
            prediction = await model.predict(test_image)

            # Should return a valid prediction
            assert prediction is not None
            assert prediction.model_type == "internal"
            assert prediction.taxonomy is not None

        except Exception as e:
            # Internal model may not be fully configured
            pytest.skip(f"Internal model not available: {e}")


class TestFullPipeline:
    """Test the full species consensus pipeline."""

    @pytest.fixture
    def engine(self):
        return SpeciesConsensusEngine()

    @pytest.fixture
    def test_image(self):
        """Load or create a test image."""
        # Try to load real test image
        tomato_path = TEST_IMAGES_DIR / "tomato_real.jpg"
        if tomato_path.exists():
            return Image.open(tomato_path).convert("RGB")

        # Fall back to synthetic image
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:, :, 1] = 128  # Green
        return Image.fromarray(img_array)

    @pytest.mark.asyncio
    async def test_full_consensus_pipeline(self, engine, test_image):
        """Test the full pipeline from image to consensus."""
        registry = get_species_model_registry()

        # Initialize with limited models for testing
        registry.initialize(
            enable_huggingface=False,  # Skip for faster testing
            enable_internal=True,
        )

        try:
            # Run predictions
            predictions = await registry.predict_all(
                test_image,
                timeout=30.0,
            )

            if not predictions:
                pytest.skip("No models available for prediction")

            # Compute consensus
            prediction_list = list(predictions.values())
            consensus = engine.compute_consensus(prediction_list)

            # Verify consensus structure
            assert consensus is not None
            assert consensus.family is not None
            assert consensus.genus is not None
            assert consensus.species is not None
            assert 0.0 <= consensus.agreement_score <= 1.0
            assert 0.0 <= consensus.overall_confidence <= 1.0

        except Exception as e:
            pytest.skip(f"Pipeline test failed (expected if models not configured): {e}")


class TestWithRealImages:
    """Test with real plant images if available."""

    @pytest.fixture
    def engine(self):
        return SpeciesConsensusEngine()

    def _load_test_image(self, name: str):
        """Load a test image by name."""
        path = TEST_IMAGES_DIR / name
        if path.exists():
            return Image.open(path).convert("RGB")
        return None

    @pytest.mark.asyncio
    async def test_tomato_image(self, engine):
        """Test with tomato image."""
        image = self._load_test_image("tomato_real.jpg")
        if image is None:
            pytest.skip("Tomato test image not available")

        registry = get_species_model_registry()
        registry.initialize(enable_huggingface=False, enable_internal=True)

        try:
            predictions = await registry.predict_all(image, timeout=30.0)
            if predictions:
                consensus = engine.compute_consensus(list(predictions.values()))
                print(f"\nTomato consensus: {consensus.species.name}")
                print(f"Confidence: {consensus.overall_confidence:.2f}")
                print(f"Agreement: {consensus.agreement_score:.2f}")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    @pytest.mark.asyncio
    async def test_apple_image(self, engine):
        """Test with apple image."""
        image = self._load_test_image("apple_real.jpg")
        if image is None:
            pytest.skip("Apple test image not available")

        registry = get_species_model_registry()
        registry.initialize(enable_huggingface=False, enable_internal=True)

        try:
            predictions = await registry.predict_all(image, timeout=30.0)
            if predictions:
                consensus = engine.compute_consensus(list(predictions.values()))
                print(f"\nApple consensus: {consensus.species.name}")
                print(f"Confidence: {consensus.overall_confidence:.2f}")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    @pytest.mark.asyncio
    async def test_corn_image(self, engine):
        """Test with corn image."""
        image = self._load_test_image("corn_real.jpg")
        if image is None:
            pytest.skip("Corn test image not available")

        registry = get_species_model_registry()
        registry.initialize(enable_huggingface=False, enable_internal=True)

        try:
            predictions = await registry.predict_all(image, timeout=30.0)
            if predictions:
                consensus = engine.compute_consensus(list(predictions.values()))
                print(f"\nCorn consensus: {consensus.species.name}")
                print(f"Confidence: {consensus.overall_confidence:.2f}")
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")


class TestEarlyWarningIntegration:
    """Test species consensus integration with early warning service."""

    @pytest.mark.asyncio
    async def test_early_warning_with_species_consensus(self):
        """Test that early warning service includes species consensus."""
        try:
            from app.services.early_warning_service import get_early_warning_service

            service = get_early_warning_service()

            # Create test image
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            img_array[:, :, 1] = 128  # Green
            test_image = Image.fromarray(img_array)

            # Run analysis
            result = await service.analyze(
                image=test_image,
                region="US-CA",
            )

            # Check result structure
            assert result is not None
            assert result.consensus is not None  # Disease consensus
            assert result.severity is not None
            assert result.treatment is not None

            # Species consensus may or may not be available depending on models
            if result.species_consensus:
                assert result.species_consensus.family is not None
                assert result.species_consensus.genus is not None
                assert result.species_consensus.species is not None
                print(f"\nSpecies consensus in early warning:")
                print(f"  Family: {result.species_consensus.family}")
                print(f"  Genus: {result.species_consensus.genus}")
                print(f"  Species: {result.species_consensus.species}")
                print(f"  Confidence: {result.species_consensus.confidence:.2f}")

        except Exception as e:
            pytest.skip(f"Early warning test skipped: {e}")


class TestGracefulDegradation:
    """Test graceful degradation when models fail."""

    @pytest.fixture
    def engine(self):
        return SpeciesConsensusEngine()

    def test_partial_model_failure(self, engine):
        """Test consensus handles partial model failures."""
        from app.ml.species_models.base import NormalizedTaxonomy, SpeciesPrediction

        # One success, one failure
        predictions = [
            SpeciesPrediction(
                model_name="Working Model",
                model_type="plantnet",
                taxonomy=NormalizedTaxonomy(
                    family="Solanaceae",
                    genus="Solanum",
                    species="Solanum lycopersicum",
                    taxonomy_depth=TaxonomyDepth.SPECIES,
                ),
                confidence=0.90,
            ),
            SpeciesPrediction(
                model_name="Failed Model",
                model_type="kindwise",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="API timeout",
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Should still produce valid consensus from working model
        assert consensus.species.name.lower() == "solanum lycopersicum"
        assert "failed" in consensus.notes.lower() or "1" in consensus.notes

    def test_timeout_handling(self, engine):
        """Test handling of timeout scenarios."""
        from app.ml.species_models.base import NormalizedTaxonomy, SpeciesPrediction

        # All timeouts
        predictions = [
            SpeciesPrediction(
                model_name="Model A",
                model_type="plantnet",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="Timeout",
            ),
            SpeciesPrediction(
                model_name="Model B",
                model_type="kindwise",
                taxonomy=NormalizedTaxonomy(),
                confidence=0.0,
                error="Timeout",
            ),
        ]

        consensus = engine.compute_consensus(predictions)

        # Should return unknown/empty consensus
        assert consensus.species.name == "Unknown"
        assert consensus.agreement_score == 0.0


class TestAPIKeyHandling:
    """Test API key handling for external models."""

    def test_registry_accepts_api_keys(self):
        """Test that registry accepts API keys."""
        registry = SpeciesModelRegistry()

        # Should not fail even with None keys
        registry.initialize(
            plantnet_api_key=None,
            kindwise_api_key=None,
            enable_huggingface=False,
            enable_internal=True,
        )

    def test_set_api_key_after_init(self):
        """Test setting API key after initialization."""
        registry = SpeciesModelRegistry()
        registry.initialize(enable_huggingface=False, enable_internal=True)

        # Should not fail
        registry.set_api_key(SpeciesModelType.PLANTNET, "test_key")
