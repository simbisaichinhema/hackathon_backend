"""Quality gate for image assessment before DR classification.

Rejects ungradeable images and requests recapture.
"""

import numpy as np
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.image_quality import assess_image_quality, QualityResult


class QualityGate:
    """Quality gate for screening pipeline."""

    def __init__(
        self,
        min_focus: float = 0.35,
        min_illumination: float = 0.4,
        min_fov: float = 0.5,
        min_overall: float = 0.5,
        min_fundus: float = 0.35,
    ):
        self.min_focus = min_focus
        self.min_illumination = min_illumination
        self.min_fov = min_fov
        self.min_overall = min_overall
        self.min_fundus = min_fundus

    def assess(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image quality.

        Thresholds aligned with preprocessing.assess_image_quality defaults
        so backend, pipeline and tests agree on pass/fail.
        """
        result = assess_image_quality(
            image,
            min_focus=self.min_focus,
            min_illumination=self.min_illumination,
            min_fov=self.min_fov,
            min_overall=self.min_overall,
            min_fundus=self.min_fundus,
        )

        return {
            "usable": result.usable,
            "focus": result.focus_score,
            "illumination": result.illumination_score,
            "field_of_view": result.fov_score,
            "fundus": result.fundus_score,
            "overall": result.overall_score,
            "reason": result.reason,
            "action": "continue" if result.usable else "recapture_required",
            "method": result.method,
            "thresholds": {
                "focus": self.min_focus,
                "illumination": self.min_illumination,
                "field_of_view": self.min_fov,
                "fundus": self.min_fundus,
                "overall": self.min_overall,
            },
        }
