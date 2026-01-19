"""
Application configuration with environment-based settings.

Configuration is centralized here to allow easy swapping between
development, staging, and production environments.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    app_name: str = "Plant Image Classifier API"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # ML Model Configuration
    model_cache_dir: str = "./models"
    species_model_path: Optional[str] = None
    disease_model_path: Optional[str] = None

    # Model inference settings
    species_confidence_threshold: float = 0.5
    disease_confidence_threshold: float = 0.6
    batch_size: int = 1

    # Image preprocessing
    image_size: int = 224  # Standard for most pretrained models
    max_image_size_mb: float = 10.0

    # Feature flags for optional components
    enable_grad_cam: bool = True
    enable_treatment_recommendations: bool = True
    enable_region_filtering: bool = False

    # LLM Integration (for explanation generation only)
    llm_provider: Optional[str] = None  # "openai", "anthropic", etc.
    llm_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "PLANT_CLASSIFIER_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Taxonomy Configuration - Extensible plant hierarchy
SUPPORTED_PLANT_FAMILIES = {
    "Solanaceae": {
        "genera": ["Solanum", "Capsicum", "Nicotiana"],
        "common_diseases": ["Early Blight", "Late Blight", "Bacterial Spot"]
    },
    "Rosaceae": {
        "genera": ["Malus", "Prunus", "Rosa"],
        "common_diseases": ["Apple Scab", "Fire Blight", "Black Spot"]
    },
    "Poaceae": {
        "genera": ["Zea", "Oryza", "Triticum"],
        "common_diseases": ["Corn Rust", "Rice Blast", "Wheat Rust"]
    },
    "Fabaceae": {
        "genera": ["Phaseolus", "Glycine", "Arachis"],
        "common_diseases": ["Bean Rust", "Soybean Mosaic", "Leaf Spot"]
    },
    "Cucurbitaceae": {
        "genera": ["Cucumis", "Cucurbita", "Citrullus"],
        "common_diseases": ["Powdery Mildew", "Downy Mildew", "Anthracnose"]
    }
}

# Disease to crop mapping - for model routing
CROP_DISEASE_MODELS = {
    "tomato": ["Early Blight", "Late Blight", "Bacterial Spot", "Septoria Leaf Spot"],
    "potato": ["Early Blight", "Late Blight"],
    "apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust"],
    "corn": ["Common Rust", "Northern Leaf Blight", "Gray Leaf Spot"],
    "grape": ["Black Rot", "Esca", "Leaf Blight"]
}
