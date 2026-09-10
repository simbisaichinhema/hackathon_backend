"""Image quality assessment for retinal fundus images.

Evaluates: focus, illumination, field of view.
Provides a quality score and usability decision.

NOTE: These are heuristic baseline scores. They are NOT clinically validated
probabilities unless explicitly trained and validated against a labeled
image-quality dataset.
"""

import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class QualityResult:
    """Result of image quality assessment."""
    focus_score: float          # 0.0 to 1.0
    illumination_score: float   # 0.0 to 1.0
    fov_score: float            # 0.0 to 1.0
    fundus_score: float = 0.0   # 0.0 to 1.0 (fundus likelihood)
    overall_score: float = 0.0  # 0.0 to 1.0
    usable: bool = False        # True if passes quality gate
    reason: str = ""            # Human-readable reason
    method: str = "heuristic"   # Mark as prototype/baseline

    def to_dict(self) -> dict:
        return asdict(self)


def compute_focus_score(image: np.ndarray) -> float:
    """Compute focus/blur metric using Laplacian variance.

    Sharp images have high Laplacian variance.
    Returns score in [0, 1] range.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    # Normalize: typical images have variance 100-800
    # Score saturates at 400+ (sharp)
    score = min(1.0, variance / 400.0)
    return float(score)


def compute_illumination_score(image: np.ndarray) -> float:
    """Compute illumination quality.

    Checks for:
    - Overall brightness (not too dark, not overexposed)
    - Even illumination (low standard deviation relative to mean)

    Returns score in [0, 1] range.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)

    # Brightness score: ideal around 80-160 for fundus images
    if mean_brightness < 30:
        brightness_score = mean_brightness / 30.0
    elif mean_brightness > 220:
        brightness_score = max(0, 1.0 - (mean_brightness - 220) / 35.0)
    else:
        brightness_score = 1.0

    # Uniformity score: lower std relative to mean is better
    if mean_brightness > 0:
        cv_val = std_brightness / mean_brightness
        uniformity_score = max(0, 1.0 - cv_val)
    else:
        uniformity_score = 0.0

    score = 0.6 * brightness_score + 0.4 * uniformity_score
    return float(np.clip(score, 0, 1))


def compute_fov_score(image: np.ndarray) -> float:
    """Compute field of view score.

    Estimates how much of the image contains retinal tissue
    vs dark background.

    Returns score in [0, 1] range.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Threshold to find non-black regions
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Count retinal pixels
    retinal_pixels = np.sum(binary > 0)
    total_pixels = binary.shape[0] * binary.shape[1]

    fov_ratio = retinal_pixels / total_pixels

    # Good fundus images typically have 40-70% retinal coverage
    score = min(1.0, fov_ratio / 0.4)
    return float(np.clip(score, 0, 1))


def compute_fundus_score(image: np.ndarray) -> float:
    """Check if image has fundus color characteristics with continuous scoring.

    Fundus images are dominated by reds/oranges/pinks from retinal tissue.
    Non-fundus images (faces, text, landscapes, noise) should score near 0.
    Returns score in [0, 1] range.
    """
    if len(image.shape) != 3:
        return 0.0

    img = image.astype(np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Reject near-uniform, very dark, or very bright images
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if mean_val < 25 or std_val < 8:
        return 0.0

    # Fundus images have moderate brightness (40-180), very bright = not fundus
    if mean_val > 200:
        return 0.0

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    # 1. Red dominance — fundus is heavily red
    r_prominence = (mean_r + 1.0) / (mean_g + mean_b + 2.0)
    score_red = min(1.0, max(0.0, (r_prominence - 0.35) / 0.65))

    # 2. Red/Green ratio (fundus typical 1.2 - 2.2)
    rg_ratio = (mean_r + 1.0) / (mean_g + 1.0)
    score_rg = 1.0 if 1.2 <= rg_ratio <= 2.2 else max(0.0, 1.0 - abs(rg_ratio - 1.7) / 2.0)

    # 3. Blue suppression (retina absorbs blue light)
    blue_ratio = (mean_b + 1.0) / (mean_r + 1.0)
    score_blue = max(0.0, min(1.0, (1.1 - blue_ratio) / 0.7))

    # 4. HSV hue concentration in red-orange range
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    red_hue_mask = ((hue >= 0) & (hue <= 35)) & (sat > 40)
    red_hue_ratio = float(np.mean(red_hue_mask))

    # 5. Dark border presence — fundus images have dark circular border
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    dark_pixels = float(np.mean(binary == 0))
    score_border = 1.0 if 0.08 < dark_pixels < 0.45 else max(0.0, 1.0 - abs(dark_pixels - 0.2) / 0.3)

    # 6. Natural image check — real images have smooth gradients (low high-freq noise)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(lap))
    score_natural = 1.0 if lap_var < 1500 else max(0.0, 1.0 - (lap_var - 1500) / 3000)

    # 7. Color channel consistency — real images have correlated channels, noise does not
    # Correlation between R and G channels (real images: >0.7, noise: ~0)
    r_flat = r.flatten()
    g_flat = g.flatten()
    if np.std(r_flat) > 0 and np.std(g_flat) > 0:
        corr_rg = float(np.corrcoef(r_flat, g_flat)[0, 1])
    else:
        corr_rg = 0.0
    score_corr = max(0.0, min(1.0, (corr_rg + 0.2) / 0.8))  # Maps 0.6->1.0, 0->0.25

    # Weighted combination
    score = (
        0.20 * score_red
        + 0.15 * score_rg
        + 0.10 * score_blue
        + 0.15 * red_hue_ratio
        + 0.10 * score_border
        + 0.10 * score_natural
        + 0.20 * score_corr
    )
    return float(np.clip(score, 0.0, 1.0))


def assess_image_quality(
    image: np.ndarray,
    min_focus: float = 0.25,
    min_illumination: float = 0.25,
    min_fov: float = 0.30,
    min_overall: float = 0.35,
    min_fundus: float = 0.35,
) -> QualityResult:
    """Assess overall image quality for DR screening."""
    focus = compute_focus_score(image)
    illumination = compute_illumination_score(image)
    fov = compute_fov_score(image)
    fundus = compute_fundus_score(image)

    # Overall: weighted average of technical quality + fundus check
    overall = 0.35 * focus + 0.25 * illumination + 0.20 * fov + 0.20 * fundus

    reasons = []
    if fundus < min_fundus:
        reasons.append(f"Not a fundus image (score: {fundus:.1%})")
    if focus < min_focus:
        reasons.append(f"Focus too low ({focus:.1%})")
    if illumination < min_illumination:
        reasons.append(f"Illumination insufficient ({illumination:.1%})")
    if fov < min_fov:
        reasons.append(f"Field of view too small ({fov:.1%})")
    if overall < min_overall:
        reasons.append(f"Overall quality below threshold ({overall:.1%})")

    usable = (
        overall >= min_overall
        and focus >= min_focus
        and illumination >= min_illumination
        and fov >= min_fov
        and fundus >= min_fundus
    )

    reason = "Quality sufficient for screening" if usable else "; ".join(reasons)

    return QualityResult(
        focus_score=round(focus, 4),
        illumination_score=round(illumination, 4),
        fov_score=round(fov, 4),
        fundus_score=round(fundus, 4),
        overall_score=round(overall, 4),
        usable=usable,
        reason=reason,
    )
