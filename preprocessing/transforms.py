"""Image transforms and preprocessing for model input.

Provides:
- Model-specific preprocessing (resize, normalize)
- Training augmentation pipeline
- Reproducible preprocessing configuration
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


# Default preprocessing config
DEFAULT_PREPROCESSING_CONFIG = {
    "target_size": (224, 224),
    "interpolation": "bilinear",
    "normalization": "imagenet",  # or "minmax" or "none"
    "apply_clahe": False,
    "apply_illumination_norm": False,
    "crop_retinal": False,
    "seed": 42,
}

# ImageNet normalization parameters (float32 to preserve dtype through arithmetic)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalization: str = "imagenet",
    apply_clahe: bool = False,
    apply_illumination_norm: bool = False,
    crop_retinal: bool = False,
) -> np.ndarray:
    """Apply full preprocessing pipeline for model input.

    Args:
        image: Input RGB image (H, W, 3) as uint8 [0, 255].
        target_size: (width, height) to resize to.
        normalization: 'imagenet', 'minmax', or 'none'.
        apply_clahe: Whether to apply CLAHE.
        apply_illumination_norm: Whether to normalize illumination.
        crop_retinal: Whether to crop to retinal region first.

    Returns:
        Preprocessed image as float32 array.
    """
    from preprocessing.retinal_crop import crop_retinal as _crop
    from preprocessing.illumination import normalize_illumination as _norm_illum
    from preprocessing.clahe import apply_clahe as _apply_clahe

    img = image.copy()

    # 1. Crop retinal region
    if crop_retinal:
        img = _crop(img, mask_retina=True)

    # 2. Illumination normalization
    if apply_illumination_norm:
        img = _norm_illum(img)

    # 3. CLAHE
    if apply_clahe:
        img = _apply_clahe(img)

    # 4. Resize
    interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, target_size, interpolation=interpolation)

    # 5. Convert to float32
    img = img.astype(np.float32) / 255.0

    # 6. Normalize
    if normalization == "imagenet":
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    elif normalization == "minmax":
        img = img  # Already in [0, 1]

    return img.astype(np.float32)


def get_training_augmentation(
    flip_horizontal: bool = True,
    flip_vertical: bool = False,
    rotation_range: float = 15.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    zoom_range: Tuple[float, float] = (0.9, 1.1),
) -> Dict[str, Any]:
    """Return training augmentation configuration.

    Uses medically reasonable augmentations for retinal images.
    Avoids aggressive transformations that could create unrealistic artifacts.

    Args:
        flip_horizontal: Allow horizontal flips.
        flip_vertical: Allow vertical flips (usually False for fundus).
        rotation_range: Max rotation in degrees.
        brightness_range: Brightness adjustment range.
        contrast_range: Contrast adjustment range.
        zoom_range: Zoom range.

    Returns:
        Dictionary of augmentation parameters.
    """
    return {
        "horizontal_flip": flip_horizontal,
        "vertical_flip": flip_vertical,
        "rotation_range": rotation_range,
        "brightness_range": brightness_range,
        "contrast_range": contrast_range,
        "zoom_range": zoom_range,
        "fill_mode": "nearest",
        "cval": 0,
    }


def create_augmented_batch(
    images: np.ndarray,
    augmentation_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Apply augmentations to a batch of images.

    Args:
        images: Batch of images (N, H, W, 3).
        augmentation_config: Augmentation parameters.
        seed: Random seed for reproducibility.

    Returns:
        Augmented batch of images.
    """
    if augmentation_config is None:
        augmentation_config = get_training_augmentation()

    np.random.seed(seed)
    augmented = images.copy().astype(np.float32)

    for i in range(len(augmented)):
        img = augmented[i]

        # Horizontal flip
        if augmentation_config.get("horizontal_flip", False) and np.random.random() > 0.5:
            img = np.fliplr(img)

        # Vertical flip
        if augmentation_config.get("vertical_flip", False) and np.random.random() > 0.5:
            img = np.flipud(img)

        # Rotation
        rot_range = augmentation_config.get("rotation_range", 0)
        if rot_range > 0:
            angle = np.random.uniform(-rot_range, rot_range)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness
        br = augmentation_config.get("brightness_range", (1.0, 1.0))
        if br != (1.0, 1.0):
            factor = np.random.uniform(br[0], br[1])
            img = np.clip(img * factor, 0, 255)

        # Contrast
        cr = augmentation_config.get("contrast_range", (1.0, 1.0))
        if cr != (1.0, 1.0):
            factor = np.random.uniform(cr[0], cr[1])
            mean = np.mean(img)
            img = np.clip((img - mean) * factor + mean, 0, 255)

        augmented[i] = img

    return augmented
