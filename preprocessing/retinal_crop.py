"""Retinal crop preprocessing for fundus images.

Detects the circular retinal region and crops to bounding box,
optionally masking the non-retinal background.
"""

import cv2
import numpy as np


def detect_retinal_mask(image: np.ndarray) -> np.ndarray:
    """Detect the retinal (non-black) region of a fundus image.

    Returns a binary mask where 1 = retinal tissue, 0 = background.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Threshold to find the circular fundus region
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    return (mask > 0).astype(np.uint8)


def crop_retinal(
    image: np.ndarray,
    mask_retina: bool = True,
    padding: int = 10,
) -> np.ndarray:
    """Crop fundus image to the retinal bounding region.

    Args:
        image: RGB fundus image (H, W, 3).
        mask_retina: If True, black out non-retinal background.
        padding: Padding around the bounding box in pixels.

    Returns:
        Cropped image containing primarily the retinal region.
    """
    mask = detect_retinal_mask(image)

    # Find bounding box of the retinal region
    coords = cv2.findNonZero(mask)
    if coords is None:
        # Fallback: return center crop
        h, w = image.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return image[y0:y0 + side, x0:x0 + side]

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    y0 = max(0, y - padding)
    y1 = min(image.shape[0], y + h + padding)
    x0 = max(0, x - padding)
    x1 = min(image.shape[1], x + w + padding)

    cropped = image[y0:y1, x0:x1]

    if mask_retina:
        mask_crop = mask[y0:y1, x0:x1]
        cropped = cropped * mask_crop[:, :, np.newaxis]

    return cropped
