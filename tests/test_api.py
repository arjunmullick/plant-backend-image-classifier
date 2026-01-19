"""
API Tests for Plant Image Classification Service

Tests the main API endpoints with various scenarios.
"""

import base64
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_base64():
    """Generate a sample test image as base64."""
    # Create a simple green image (simulating a leaf)
    img = Image.new('RGB', (224, 224), color=(34, 139, 34))  # Forest green

    # Add some variation
    pixels = img.load()
    for i in range(50, 150):
        for j in range(50, 150):
            # Slightly different green in center
            pixels[i, j] = (50, 150, 50)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_check(self, client):
        """Test detailed readiness check."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data
        assert "species_classifier" in data["components"]
        assert "disease_detector" in data["components"]


class TestClassificationEndpoints:
    """Test classification endpoints."""

    def test_full_classification(self, client, sample_image_base64):
        """Test full classification pipeline."""
        response = client.post(
            "/api/v1/classify",
            json={
                "image": sample_image_base64,
                "region": "US-CA",
                "include_treatment": True,
                "include_explainability": True
            }
        )

        assert response.status_code == 200

        data = response.json()

        # Check plant identification
        assert "plant" in data
        assert "family" in data["plant"]
        assert "genus" in data["plant"]
        assert "species" in data["plant"]
        assert "confidence" in data["plant"]["species"]

        # Check health assessment
        assert "health" in data
        assert "status" in data["health"]
        assert data["health"]["status"] in ["Healthy", "Diseased"]

        # Check metadata
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]

    def test_classification_without_treatment(self, client, sample_image_base64):
        """Test classification without treatment recommendations."""
        response = client.post(
            "/api/v1/classify",
            json={
                "image": sample_image_base64,
                "include_treatment": False,
                "include_explainability": False
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Should still have plant and health
        assert "plant" in data
        assert "health" in data

    def test_species_only_endpoint(self, client, sample_image_base64):
        """Test species-only classification."""
        response = client.post(
            "/api/v1/classify/species",
            json={"image": sample_image_base64}
        )

        assert response.status_code == 200
        data = response.json()

        assert "family" in data
        assert "genus" in data
        assert "species" in data

    def test_disease_only_endpoint(self, client, sample_image_base64):
        """Test disease-only detection."""
        response = client.post(
            "/api/v1/classify/disease",
            json={
                "image": sample_image_base64,
                "crop": "tomato"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "confidence" in data
        assert data["status"] in ["Healthy", "Diseased"]

    def test_invalid_image(self, client):
        """Test with invalid base64 image."""
        response = client.post(
            "/api/v1/classify",
            json={"image": "not_valid_base64"}
        )

        assert response.status_code == 422  # Validation error

    def test_supported_crops(self, client):
        """Test getting supported crops."""
        response = client.get("/api/v1/classify/supported-crops")
        assert response.status_code == 200

        data = response.json()
        assert "crops" in data
        assert isinstance(data["crops"], list)

    def test_supported_diseases(self, client):
        """Test getting supported diseases."""
        response = client.get("/api/v1/classify/supported-diseases")
        assert response.status_code == 200

        data = response.json()
        assert "diseases" in data
        assert isinstance(data["diseases"], list)


class TestBatchClassification:
    """Test batch classification endpoint."""

    def test_batch_classification(self, client, sample_image_base64):
        """Test batch classification with multiple images."""
        response = client.post(
            "/api/v1/classify/batch",
            json={
                "images": [sample_image_base64, sample_image_base64],
                "region": "US-CA"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2
        assert data["summary"]["total_images"] == 2

    def test_batch_limit(self, client, sample_image_base64):
        """Test batch limit enforcement."""
        # Try to send more than 10 images
        images = [sample_image_base64] * 11

        response = client.post(
            "/api/v1/classify/batch",
            json={"images": images}
        )

        assert response.status_code == 400
        assert "Maximum 10 images" in response.json()["detail"]


class TestFeedback:
    """Test feedback submission."""

    def test_submit_feedback(self, client):
        """Test submitting prediction feedback."""
        response = client.post(
            "/api/v1/classify/feedback",
            json={
                "image_id": "test-123",
                "predicted_species": "Solanum lycopersicum",
                "predicted_disease": "Early Blight",
                "correct_species": "Solanum lycopersicum",
                "correct_disease": "Late Blight",
                "feedback_type": "correction",
                "notes": "Actually appears to be late blight"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data
