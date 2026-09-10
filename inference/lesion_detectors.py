"""Basic lesion detection heuristics for retinal evidence.

These are NOT trained models — they're color/shape heuristics
for prototype demonstration only. Not clinically valid.
"""

import cv2
import numpy as np
from typing import Dict, Any


def detect_microaneurysms(image: np.ndarray) -> Dict[str, Any]:
    """Detect small red dots (microaneurysms) using color filtering.

    Microaneurysms appear as small, round, dark red spots on the retina.
    """
    if len(image.shape) != 3:
        return {"detected": False, "confidence": 0.0, "count": 0}

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Microaneurysms: dark red, small, circular
    # Hue: 0-10 (red), Saturation: 50-200, Value: 30-150
    mask = cv2.inRange(hsv, (0, 50, 30), (10, 200, 150))

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size: microaneurysms are small (5-50 pixels diameter)
    ma_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < 2000:  # Small spots
            circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
            if circularity > 0.4:  # Reasonably circular
                ma_contours.append(c)

    count = len(ma_contours)
    # Score: more detected = higher confidence, capped
    confidence = min(1.0, count * 0.05)

    return {
        "detected": count > 0,
        "confidence": round(confidence, 3),
        "count": count,
        "note": "Heuristic detection — not clinically validated"
    }


def detect_hemorrhages(image: np.ndarray) -> Dict[str, Any]:
    """Detect hemorrhages using color and size filtering.

    Hemorrhages are larger, irregular red/dark-red areas.
    """
    if len(image.shape) != 3:
        return {"detected": False, "confidence": 0.0, "count": 0}

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Hemorrhages: larger red areas, darker than microaneurysms
    # Hue: 0-15, Saturation: 40-180, Value: 20-120
    mask = cv2.inRange(hsv, (0, 40, 20), (15, 180, 120))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter: hemorrhages are larger (100-10000 pixels)
    hemo_contours = [c for c in contours if 100 < cv2.contourArea(c) < 10000]

    count = len(hemo_contours)
    confidence = min(1.0, count * 0.08)

    return {
        "detected": count > 0,
        "confidence": round(confidence, 3),
        "count": count,
        "note": "Heuristic detection — not clinically validated"
    }


def detect_exudates(image: np.ndarray) -> Dict[str, Any]:
    """Detect hard exudates using brightness filtering.

    Exudates appear as bright yellow/white spots with sharp edges.
    """
    if len(image.shape) != 3:
        return {"detected": False, "confidence": 0.0, "count": 0}

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Exudates: bright, yellowish
    # Hue: 15-40 (yellow range), Saturation: 50-255, Value: 150-255
    mask = cv2.inRange(hsv, (15, 50, 150), (40, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Exudates: small to medium bright spots
    exud_contours = [c for c in contours if 50 < cv2.contourArea(c) < 5000]

    count = len(exud_contours)
    confidence = min(1.0, count * 0.06)

    return {
        "detected": count > 0,
        "confidence": round(confidence, 3),
        "count": count,
        "note": "Heuristic detection — not clinically validated"
    }


def analyze_vessels(image: np.ndarray) -> Dict[str, Any]:
    """Basic vessel analysis using edge detection.

    Not true vessel segmentation — just detects vessel-like structures.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Edge detection
    edges = cv2.Canny(enhanced, 30, 100)

    # Morphological operations to connect vessel segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vessels_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vessels_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    vessels = cv2.bitwise_or(vessels_h, vessels_v)

    # Calculate vessel density (cast to pure Python for JSON/SSE)
    vessel_pixels = int(np.sum(vessels > 0))
    total_pixels = int(vessels.shape[0] * vessels.shape[1])
    density = float(vessel_pixels / max(total_pixels, 1))

    # Simple quality metric
    if density < 0.01:
        quality = "Low visibility"
    elif density < 0.05:
        quality = "Normal"
    else:
        quality = "High density"

    return {
        "detected": bool(density > 0.005),
        "density": round(float(density), 4),
        "quality": quality,
        "note": "Basic edge detection — not true vessel segmentation"
    }
