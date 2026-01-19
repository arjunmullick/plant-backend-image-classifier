"""
Image preprocessing pipeline.

Handles:
- Base64 decoding and image loading
- Image quality assessment
- Normalization for model inference
- Data augmentation for training
- Handling of noisy real-world images

Design for handling PlantVillage, PlantNet, iNaturalist data:
- Variable image sizes
- Different lighting conditions
- Background noise
- Partial plant visibility
"""

from dataclasses import dataclass
from typing import Optional
import base64
import io
import logging

import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityReport:
    """Assessment of input image quality."""
    overall_score: float  # 0-1
    is_usable: bool
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    issues: list[str]
    recommendations: list[str]


@dataclass
class PreprocessedImage:
    """Preprocessed image ready for model inference."""
    tensor: np.ndarray  # Shape: (C, H, W) or (B, C, H, W)
    original_size: tuple[int, int]
    quality_report: Optional[ImageQualityReport] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ImagePreprocessor:
    """
    Image preprocessing pipeline for plant classification.

    Supports both inference preprocessing and training augmentation.

    Model Compatibility:
    - EfficientNet: 224x224, ImageNet normalization
    - ResNet: 224x224, ImageNet normalization
    - ViT: 224x224, ImageNet normalization
    - ConvNeXT: 224x224, ImageNet normalization

    Usage:
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        result = preprocessor.preprocess_from_base64(base64_string)
        model_input = result.tensor
    """

    # ImageNet normalization parameters
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        assess_quality: bool = True,
        min_quality_score: float = 0.3
    ):
        """
        Initialize the preprocessor.

        Args:
            target_size: Target image size (H, W)
            normalize: Apply ImageNet normalization
            assess_quality: Run quality assessment
            min_quality_score: Minimum quality score to proceed
        """
        self.target_size = target_size
        self.normalize = normalize
        self.assess_quality = assess_quality
        self.min_quality_score = min_quality_score

    def preprocess_from_base64(
        self,
        base64_string: str,
        return_pil: bool = False
    ) -> PreprocessedImage:
        """
        Preprocess an image from base64 string.

        Args:
            base64_string: Base64-encoded image
            return_pil: Also return PIL Image in metadata

        Returns:
            PreprocessedImage with tensor and quality report
        """
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))

        return self.preprocess(image, return_pil=return_pil)

    def preprocess(
        self,
        image: Image.Image,
        return_pil: bool = False
    ) -> PreprocessedImage:
        """
        Preprocess a PIL Image for model inference.

        Pipeline:
        1. Convert to RGB
        2. Assess quality (optional)
        3. Resize maintaining aspect ratio
        4. Center crop to target size
        5. Normalize (optional)
        6. Convert to tensor format

        Args:
            image: PIL Image
            return_pil: Also return PIL Image in metadata

        Returns:
            PreprocessedImage ready for model
        """
        original_size = image.size

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Assess image quality
        quality_report = None
        if self.assess_quality:
            quality_report = self._assess_quality(image)
            if not quality_report.is_usable:
                logger.warning(
                    f"Image quality below threshold: {quality_report.overall_score:.2f}"
                )

        # Resize with aspect ratio preservation
        image = self._resize_with_padding(image)

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Apply normalization
        if self.normalize:
            img_array = (img_array - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Convert to CHW format for PyTorch
        tensor = np.transpose(img_array, (2, 0, 1)).astype(np.float32)

        metadata = {"original_mode": image.mode}
        if return_pil:
            metadata["pil_image"] = image

        return PreprocessedImage(
            tensor=tensor,
            original_size=original_size,
            quality_report=quality_report,
            metadata=metadata
        )

    def _resize_with_padding(self, image: Image.Image) -> Image.Image:
        """
        Resize image to target size with center cropping.

        For real-world images, we use center crop after scaling
        to avoid distortion while preserving the main subject.
        """
        target_w, target_h = self.target_size
        orig_w, orig_h = image.size

        # Calculate scale to cover target size
        scale = max(target_w / orig_w, target_h / orig_h)

        # Scale image
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))

        return image

    def _assess_quality(self, image: Image.Image) -> ImageQualityReport:
        """
        Assess image quality for classification.

        Checks:
        - Brightness (not too dark or overexposed)
        - Contrast (sufficient detail visibility)
        - Sharpness (not blurry)
        """
        issues = []
        recommendations = []

        # Get image statistics
        stat = ImageStat.Stat(image)
        mean_brightness = sum(stat.mean) / 3 / 255

        # Brightness assessment
        if mean_brightness < 0.2:
            brightness_score = mean_brightness / 0.2
            issues.append("Image is too dark")
            recommendations.append("Use better lighting or increase exposure")
        elif mean_brightness > 0.85:
            brightness_score = (1 - mean_brightness) / 0.15
            issues.append("Image is overexposed")
            recommendations.append("Reduce exposure or avoid direct sunlight")
        else:
            brightness_score = 1.0

        # Contrast assessment (using standard deviation)
        std_contrast = sum(stat.stddev) / 3 / 255
        if std_contrast < 0.1:
            contrast_score = std_contrast / 0.1
            issues.append("Low contrast - details may be hard to distinguish")
            recommendations.append("Improve lighting contrast")
        else:
            contrast_score = min(1.0, std_contrast / 0.15)

        # Sharpness assessment using Laplacian variance
        grayscale = image.convert("L")
        laplacian = grayscale.filter(ImageFilter.FIND_EDGES)
        laplacian_stat = ImageStat.Stat(laplacian)
        laplacian_var = laplacian_stat.var[0]

        if laplacian_var < 100:
            sharpness_score = laplacian_var / 100
            issues.append("Image appears blurry")
            recommendations.append("Ensure camera is focused on the plant")
        else:
            sharpness_score = min(1.0, laplacian_var / 500)

        # Overall score (weighted average)
        overall_score = (
            0.3 * brightness_score +
            0.3 * contrast_score +
            0.4 * sharpness_score
        )

        return ImageQualityReport(
            overall_score=overall_score,
            is_usable=overall_score >= self.min_quality_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            sharpness_score=sharpness_score,
            issues=issues,
            recommendations=recommendations
        )

    def get_training_augmentation(self) -> dict:
        """
        Return augmentation configuration for training.

        These augmentations help the model generalize to:
        - Different lighting conditions
        - Various angles and orientations
        - Partial occlusion
        - Background variation

        Returns:
            Dictionary describing augmentation pipeline
            (to be implemented with albumentations or torchvision)
        """
        return {
            "description": "Training augmentation pipeline",
            "transforms": [
                {
                    "name": "RandomResizedCrop",
                    "params": {"size": self.target_size, "scale": (0.7, 1.0)}
                },
                {
                    "name": "HorizontalFlip",
                    "params": {"p": 0.5}
                },
                {
                    "name": "VerticalFlip",
                    "params": {"p": 0.2}
                },
                {
                    "name": "RandomRotation",
                    "params": {"degrees": 30}
                },
                {
                    "name": "ColorJitter",
                    "params": {
                        "brightness": 0.3,
                        "contrast": 0.3,
                        "saturation": 0.3,
                        "hue": 0.1
                    }
                },
                {
                    "name": "GaussianBlur",
                    "params": {"kernel_size": 3, "p": 0.2}
                },
                {
                    "name": "RandomErasing",
                    "params": {"p": 0.1, "scale": (0.02, 0.1)}
                },
                {
                    "name": "Normalize",
                    "params": {
                        "mean": self.IMAGENET_MEAN.tolist(),
                        "std": self.IMAGENET_STD.tolist()
                    }
                }
            ],
            "notes": [
                "ColorJitter helps with varying field lighting",
                "RandomErasing simulates partial leaf occlusion",
                "Rotation handles different photo orientations"
            ]
        }

    def create_batch(self, images: list[PreprocessedImage]) -> np.ndarray:
        """Stack multiple preprocessed images into a batch."""
        tensors = [img.tensor for img in images]
        return np.stack(tensors, axis=0)
