"""CLAHE (Contrast Limited Adaptive Histogram Equalization) for fundus images.

Enhances local contrast while preventing noise amplification.
"""

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: tuple = (8, 8),
    convert_to_lab: bool = True,
) -> np.ndarray:
    """Apply CLAHE to a fundus image.

    For color images, applies CLAHE to the L channel in LAB color space
    to preserve color information while enhancing contrast.

    Args:
        image: RGB fundus image (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of the grid for local histogram equalization.
        convert_to_lab: If True, apply CLAHE to L channel of LAB space.

    Returns:
        Contrast-enhanced image.
    """
    if len(image.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

    if convert_to_lab:
        # Convert to LAB, apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab[:, :, 0] = clahe.apply(l_channel)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Apply CLAHE to each channel independently
        result = np.zeros_like(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        for c in range(image.shape[2]):
            result[:, :, c] = clahe.apply(image[:, :, c])
        return result
