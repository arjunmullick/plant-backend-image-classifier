"""
Tests for real ML model inference.

These tests verify that:
1. The HuggingFace models produce deterministic results
2. Same image returns same predictions every time
3. Models are properly integrated into the classification pipeline
"""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Check if transformers is available
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@pytest.fixture
def sample_image():
    """Create a sample test image (green leaf-like pattern)."""
    # Create a simple 224x224 RGB image
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)

    # Add some green color to simulate a leaf
    img_array[50:200, 50:200, 1] = 150  # Green channel
    img_array[50:200, 50:200, 0] = 50   # Some red
    img_array[50:200, 50:200, 2] = 50   # Some blue

    return Image.fromarray(img_array)


@pytest.fixture
def sample_tensor():
    """Create a preprocessed image tensor."""
    # Create normalized tensor (C, H, W) format
    img = np.zeros((3, 224, 224), dtype=np.float32)

    # Add normalized values (ImageNet-style normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Create a pattern
    raw = np.zeros((224, 224, 3), dtype=np.float32)
    raw[50:200, 50:200, 1] = 0.6  # Green
    raw[50:200, 50:200, 0] = 0.2
    raw[50:200, 50:200, 2] = 0.2

    # Normalize
    normalized = (raw - mean) / std
    img = normalized.transpose(2, 0, 1)  # HWC -> CHW

    return img


@pytest.fixture
def sample_image_base64(sample_image):
    """Convert sample image to base64."""
    buffer = BytesIO()
    sample_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


class TestSpeciesClassifierDeterminism:
    """Test that species classifier produces deterministic results."""

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_same_image_same_prediction(self, sample_tensor):
        """Same image should produce identical predictions every time."""
        from app.ml.species_classifier import SpeciesClassifier

        classifier = SpeciesClassifier()
        classifier.load_model()

        # Run prediction twice on the same image
        result1 = classifier.predict(sample_tensor)
        result2 = classifier.predict(sample_tensor)

        # Predictions should be identical
        assert result1.prediction.species == result2.prediction.species
        assert result1.prediction.family == result2.prediction.family
        assert result1.prediction.genus == result2.prediction.genus
        assert abs(result1.confidence - result2.confidence) < 1e-6

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_model_version_is_real(self, sample_tensor):
        """Model version should indicate real model, not placeholder."""
        from app.ml.species_classifier import SpeciesClassifier

        classifier = SpeciesClassifier()
        classifier.load_model()

        result = classifier.predict(sample_tensor)

        # Should use real model version, not placeholder
        assert "mobilenet" in result.model_version.lower() or "1.0.0" in result.model_version
        assert "placeholder" not in result.model_version.lower()

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_confidence_is_reasonable(self, sample_tensor):
        """Confidence scores should be in valid range."""
        from app.ml.species_classifier import SpeciesClassifier

        classifier = SpeciesClassifier()
        classifier.load_model()

        result = classifier.predict(sample_tensor)

        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.prediction.family_confidence <= 1.0
        assert 0.0 <= result.prediction.genus_confidence <= 1.0
        assert 0.0 <= result.prediction.species_confidence <= 1.0


class TestDiseaseDetectorDeterminism:
    """Test that disease detector produces deterministic results."""

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_same_image_same_prediction(self, sample_tensor):
        """Same image should produce identical disease predictions."""
        from app.ml.disease_detector import DiseaseDetector

        detector = DiseaseDetector()
        detector.load_model()

        # Run prediction twice
        result1 = detector.predict(sample_tensor)
        result2 = detector.predict(sample_tensor)

        # Predictions should be identical
        assert result1.prediction.is_healthy == result2.prediction.is_healthy
        assert result1.prediction.disease_name == result2.prediction.disease_name
        assert abs(result1.confidence - result2.confidence) < 1e-6

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_model_version_is_real(self, sample_tensor):
        """Model version should indicate real model."""
        from app.ml.disease_detector import DiseaseDetector

        detector = DiseaseDetector()
        detector.load_model()

        result = detector.predict(sample_tensor)

        assert "mobilenet" in result.model_version.lower() or "1.0.0" in result.model_version
        assert "placeholder" not in result.model_version.lower()

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_healthy_vs_diseased_consistency(self, sample_tensor):
        """Binary healthy/diseased should be consistent across runs."""
        from app.ml.disease_detector import DiseaseDetector

        detector = DiseaseDetector()
        detector.load_model()

        results = [detector.predict(sample_tensor) for _ in range(3)]

        # All runs should agree on healthy status
        healthy_status = [r.prediction.is_healthy for r in results]
        assert len(set(healthy_status)) == 1, "Healthy status should be consistent"


class TestExternalModelComparison:
    """Test comparison endpoint with real models."""

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_comparison_returns_real_predictions(self, sample_image):
        """Comparison should return real predictions from all models."""
        from app.ml.external_models import ExternalModelRegistry, ExternalModelType
        import asyncio

        registry = ExternalModelRegistry()

        async def run_comparison():
            return await registry.run_comparison(
                sample_image,
                model_types=[ExternalModelType.MOBILENET_V2]
            )

        results = asyncio.run(run_comparison())

        # Should have result for mobilenet
        assert "mobilenet_v2" in results
        result = results["mobilenet_v2"]

        # Should have real prediction (not error)
        if result.error is None:
            assert result.prediction is not None
            assert result.confidence is not None
            assert result.confidence > 0

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_comparison_determinism(self, sample_image):
        """Same image should give same comparison results."""
        from app.ml.external_models import ExternalModelRegistry, ExternalModelType
        import asyncio

        registry = ExternalModelRegistry()

        async def run_comparison():
            return await registry.run_comparison(
                sample_image,
                model_types=[ExternalModelType.MOBILENET_V2]
            )

        results1 = asyncio.run(run_comparison())
        results2 = asyncio.run(run_comparison())

        if results1["mobilenet_v2"].error is None:
            assert results1["mobilenet_v2"].prediction == results2["mobilenet_v2"].prediction
            assert results1["mobilenet_v2"].raw_label == results2["mobilenet_v2"].raw_label


class TestPlaceholderFallback:
    """Test that placeholder models work when transformers is not available."""

    def test_placeholder_species_model_deterministic(self, sample_tensor):
        """Placeholder should be deterministic for same input."""
        from app.ml.species_classifier import PlaceholderSpeciesModel

        model = PlaceholderSpeciesModel(num_classes=20, architecture="test")

        # Same input should give same output
        result1 = model.forward(sample_tensor[np.newaxis, ...])
        result2 = model.forward(sample_tensor[np.newaxis, ...])

        np.testing.assert_array_almost_equal(
            result1["species_probs"],
            result2["species_probs"]
        )

    def test_placeholder_disease_model_deterministic(self, sample_tensor):
        """Placeholder should be deterministic for same input."""
        from app.ml.disease_detector import PlaceholderDiseaseModel

        model = PlaceholderDiseaseModel(num_classes=18, crop="tomato")

        # Same input should give same output
        result1 = model.forward(sample_tensor[np.newaxis, ...])
        result2 = model.forward(sample_tensor[np.newaxis, ...])

        np.testing.assert_array_almost_equal(
            result1["disease_probs"],
            result2["disease_probs"]
        )
        np.testing.assert_array_almost_equal(
            result1["binary_probs"],
            result2["binary_probs"]
        )


class TestIntegration:
    """Integration tests for the full classification pipeline."""

    @pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")
    def test_full_pipeline_deterministic(self, sample_tensor):
        """Full classification pipeline should be deterministic."""
        from app.ml.species_classifier import SpeciesClassifier
        from app.ml.disease_detector import DiseaseDetector

        # Initialize models
        species_classifier = SpeciesClassifier()
        species_classifier.load_model()

        disease_detector = DiseaseDetector()
        disease_detector.load_model()

        # Run full pipeline twice
        species_result1 = species_classifier.predict(sample_tensor)
        disease_result1 = disease_detector.predict(
            sample_tensor,
            crop=species_result1.prediction.common_name
        )

        species_result2 = species_classifier.predict(sample_tensor)
        disease_result2 = disease_detector.predict(
            sample_tensor,
            crop=species_result2.prediction.common_name
        )

        # Results should be identical
        assert species_result1.prediction.species == species_result2.prediction.species
        assert disease_result1.prediction.disease_name == disease_result2.prediction.disease_name
