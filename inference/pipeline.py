"""Main DR screening pipeline.

Orchestrates:
1. Image ingestion
2. Quality assessment
3. Preprocessing (resize to 224x224, normalize)
4. DR classification
5. Referable DR calculation
6. Report generation
"""

import uuid
import json
import traceback
import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional, Generator
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.quality_gate import QualityGate
from inference.classifier import DRClassifier, REFERABLE_THRESHOLD
from inference.report_generator import ClinicalReportGenerator
from inference.lesion_detectors import (
    detect_microaneurysms,
    detect_hemorrhages,
    detect_exudates,
    analyze_vessels,
)
from preprocessing.transforms import preprocess_for_model
from explainability.gradcam import compute_gradcam, generate_gradcam_overlay


MODEL_INPUT_SIZE = (224, 224)


class DRScreeningPipeline:
    """Complete DR screening pipeline."""

    def __init__(self):
        self.quality_gate = QualityGate()
        self.classifier = DRClassifier()
        self.report_generator = ClinicalReportGenerator()

        try:
            self.classifier.load_model()
        except Exception as e:
            # load_model now falls back internally; keep pipeline alive.
            print(f"WARNING: Could not load model: {e}")

    def _generate_gradcam_b64(self, image: np.ndarray, input_tensor: np.ndarray, predicted_grade: int) -> Optional[str]:
        """Generate base64 encoded Grad-CAM overlay image if model is available."""
        if self.classifier.model is None:
            return None
        try:
            _, heatmap_resized = compute_gradcam(
                self.classifier.model,
                input_tensor,
                predicted_grade,
            )
            overlay = generate_gradcam_overlay(image, heatmap_resized)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            print(f"GradCAM generation skipped: {e}")
            return None

    def _generate_enhanced_b64(self, image: np.ndarray) -> Optional[str]:
        """Generate base64 encoded CLAHE enhanced retinal image for clinical inspection."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR))
            return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            print(f"Enhancement generation skipped: {e}")
            return None

    def screen(
        self,
        image: np.ndarray,
        case_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run complete DR screening pipeline (non-streaming)."""
        if case_id is None:
            case_id = f"DR-{uuid.uuid4().hex[:8].upper()}"

        # Quality assessment
        quality_obj = self.quality_gate.assess(image)
        quality = quality_obj.to_dict() if hasattr(quality_obj, 'to_dict') else dict(quality_obj)
        quality["focus"] = quality.get("focus", quality.get("focus_score", 0.0))
        quality["illumination"] = quality.get("illumination", quality.get("illumination_score", 0.0))
        quality["field_of_view"] = quality.get("field_of_view", quality.get("fov_score", 0.0))
        quality["overall"] = quality.get("overall", quality.get("overall_score", 0.0))
        quality["usable"] = quality.get("usable", False)

        if not quality["usable"]:
            return {
                "case_id": case_id,
                "status": "recapture_required",
                "quality": quality,
                "dr_prediction": None,
                "referable_dr": None,
                "report": None,
                "message": quality.get("reason", "Image did not meet quality thresholds; recapture required."),
            }

        processed = preprocess_for_model(
            image,
            target_size=MODEL_INPUT_SIZE,
            normalization="imagenet",
        )
        input_tensor = np.expand_dims(processed, axis=0)

        prediction = self.classifier.predict(input_tensor)

        referable_prob = self.classifier.compute_referable_probability(
            prediction["probabilities"]
        )
        is_referable = prediction["grade"] >= REFERABLE_THRESHOLD

        # Lesion detection
        micro = detect_microaneurysms(image)
        hemo = detect_hemorrhages(image)
        exud = detect_exudates(image)
        vessel = analyze_vessels(image)

        # Grad-CAM heatmap & Enhanced Retinal Scan
        gradcam_b64 = self._generate_gradcam_b64(image, input_tensor, prediction["grade"])
        enhanced_b64 = self._generate_enhanced_b64(image)

        report = self.report_generator.generate(
            case_id=case_id,
            quality=quality,
            prediction=prediction,
            referable=is_referable,
            referable_probability=referable_prob,
            lesion_evidence={
                "microaneurysms": micro,
                "hemorrhages": hemo,
                "exudates": exud,
            },
            vessel_evidence=vessel,
        )

        return {
            "case_id": case_id,
            "status": "completed",
            "quality": quality,
            "dr_prediction": {
                "grade": prediction["grade"],
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
            },
            "referable_dr": {
                "is_referable": is_referable,
                "probability": referable_prob,
                "definition": "Level 2+",
            },
            "lesions": {
                "microaneurysms": micro,
                "hemorrhages": hemo,
                "exudates": exud,
                "vessels": vessel,
            },
            "gradcam_image": gradcam_b64,
            "enhanced_image": enhanced_b64,
            "report": report,
        }

    def screen_stream(
        self,
        image: np.ndarray,
        case_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Run DR screening pipeline with progress streaming.

        Yields stage dicts as each step completes, then a final result dict.
        """
        if case_id is None:
            case_id = f"DR-{uuid.uuid4().hex[:8].upper()}"

        # Stage 1: Image received
        h, w = image.shape[:2]
        yield {
            "type": "stage",
            "name": "image_received",
            "message": f"Image received ({w}x{h})",
        }

        # Stage 2: Quality assessment
        quality_obj = self.quality_gate.assess(image)
        quality = quality_obj.to_dict() if hasattr(quality_obj, 'to_dict') else dict(quality_obj)
        quality["focus"] = quality.get("focus", quality.get("focus_score", 0.0))
        quality["illumination"] = quality.get("illumination", quality.get("illumination_score", 0.0))
        quality["field_of_view"] = quality.get("field_of_view", quality.get("fov_score", 0.0))
        quality["overall"] = quality.get("overall", quality.get("overall_score", 0.0))
        quality["usable"] = quality.get("usable", False)

        overall_pct = f"{quality['overall'] * 100:.1f}%"
        yield {
            "type": "stage",
            "name": "quality_check",
            "message": (
                f"Quality check passed — {overall_pct} overall"
                if quality["usable"]
                else f"Quality check failed — {overall_pct} overall; recapture required"
            ),
            "data": quality,
        }

        if not quality["usable"]:
            yield {
                "type": "result",
                "data": {
                    "case_id": case_id,
                    "status": "recapture_required",
                    "quality": quality,
                    "dr_prediction": None,
                    "referable_dr": None,
                    "report": None,
                    "message": quality.get("reason", "Image did not meet quality thresholds; recapture required."),
                },
            }
            return

        # Stage 3: Preprocessing
        processed = preprocess_for_model(
            image,
            target_size=MODEL_INPUT_SIZE,
            normalization="imagenet",
        )
        input_tensor = np.expand_dims(processed, axis=0)
        yield {
            "type": "stage",
            "name": "preprocessing",
            "message": "Preprocessed — resized to 224x224, normalized",
        }

        # Stage 4: Model inference
        prediction = self.classifier.predict(input_tensor)
        yield {
            "type": "stage",
            "name": "inference",
            "message": f"Predicted: {prediction['label']} ({prediction['confidence'] * 100:.1f}% confidence)",
            "data": prediction,
        }

        # Stage 5: Referable DR check
        referable_prob = self.classifier.compute_referable_probability(
            prediction["probabilities"]
        )
        is_referable = prediction["grade"] >= REFERABLE_THRESHOLD
        ref_label = "YES" if is_referable else "NO"
        yield {
            "type": "stage",
            "name": "referable_check",
            "message": f"Referable DR: {ref_label} ({referable_prob * 100:.1f}% probability)",
            "data": {
                "is_referable": is_referable,
                "probability": referable_prob,
            },
        }

        # Stage 5b: Lesion detection
        yield {
            "type": "stage",
            "name": "lesion_detection",
            "message": "Analyzing retinal lesions...",
        }

        micro = detect_microaneurysms(image)
        hemo = detect_hemorrhages(image)
        exud = detect_exudates(image)
        vessel = analyze_vessels(image)

        lesion_summary = []
        if micro["detected"]:
            lesion_summary.append(f"Microaneurysms: {micro['count']}")
        if hemo["detected"]:
            lesion_summary.append(f"Hemorrhages: {hemo['count']}")
        if exud["detected"]:
            lesion_summary.append(f"Exudates: {exud['count']}")

        yield {
            "type": "stage",
            "name": "lesion_complete",
            "message": f"Lesion analysis complete" + (f" — {', '.join(lesion_summary)}" if lesion_summary else " — No lesions detected"),
            "data": {
                "microaneurysms": micro,
                "hemorrhages": hemo,
                "exudates": exud,
                "vessels": vessel,
            },
        }

        # Stage 5c: Grad-CAM Explainability
        yield {
            "type": "stage",
            "name": "gradcam",
            "message": "Generating Grad-CAM model attention heatmap...",
        }
        gradcam_b64 = self._generate_gradcam_b64(image, input_tensor, prediction["grade"])
        enhanced_b64 = self._generate_enhanced_b64(image)

        # Stage 6: Report generation
        report = self.report_generator.generate(
            case_id=case_id,
            quality=quality,
            prediction=prediction,
            referable=is_referable,
            referable_probability=referable_prob,
            lesion_evidence={
                "microaneurysms": micro,
                "hemorrhages": hemo,
                "exudates": exud,
            },
            vessel_evidence=vessel,
        )
        yield {
            "type": "stage",
            "name": "report",
            "message": "Clinical report generated",
        }

        # Final result
        yield {
            "type": "result",
            "data": {
                "case_id": case_id,
                "status": "completed",
                "quality": quality,
                "dr_prediction": {
                    "grade": prediction["grade"],
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                    "probabilities": prediction["probabilities"],
                },
                "referable_dr": {
                    "is_referable": is_referable,
                    "probability": referable_prob,
                    "definition": "Level 2+",
                },
                "lesions": {
                    "microaneurysms": micro,
                    "hemorrhages": hemo,
                    "exudates": exud,
                    "vessels": vessel,
                },
                "gradcam_image": gradcam_b64,
                "enhanced_image": enhanced_b64,
                "report": report,
            },
        }
