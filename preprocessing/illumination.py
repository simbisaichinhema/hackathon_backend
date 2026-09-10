"""Illumination normalization for retinal fundus images.

Corrects uneven illumination common in fundus photography.
"""

import cv2
import numpy as np


def normalize_illumination(
    image: np.ndarray,
    kernel_size: int = 51,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Normalize illumination using background estimation.

    Estimates the illumination background and divides it out,
    then rescales to [0, 255].

    Args:
        image: RGB fundus image (H, W, 3).
        kernel_size: Size of the morphological kernel for background estimation.
        clip_limit: Not used here, kept for API consistency with CLAHE.

    Returns:
        Illumination-normalized image.
    """
    if len(image.shape) == 2:
        gray = image.copy()
        # Create illumination map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)

        # Avoid division by zero
        background = np.maximum(background, 1)
        normalized = (gray.astype(np.float32) / background.astype(np.float32)) * 128.0
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        return normalized

    # For color images, normalize each channel
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(image[:, :, c], cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
        background = np.maximum(background.astype(np.float32), 1.0)

        normalized = (channel / background) * 128.0
        result[:, :, c] = np.clip(normalized, 0, 255).astype(np.uint8)

    return result
