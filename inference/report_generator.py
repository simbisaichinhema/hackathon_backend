"""Clinical report generator for DR screening."""

import json
from datetime import datetime
from typing import Dict, Any, Optional


class ClinicalReportGenerator:
    """Generate clinical screening reports."""

    def generate(
        self,
        case_id: str,
        quality: Dict[str, Any],
        prediction: Dict[str, Any],
        referable: bool,
        referable_probability: float = 0.0,
        lesion_evidence: Optional[Dict[str, Any]] = None,
        vessel_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a clinical screening report.

        Args:
            case_id: Unique case identifier.
            quality: Image quality assessment.
            prediction: DR prediction results.
            referable: Whether case is referable.
            referable_probability: P(class 2) + P(class 3) + P(class 4).
            lesion_evidence: Lesion analysis results (optional).
            vessel_evidence: Vessel segmentation results (optional).

        Returns:
            Clinical report dictionary.
        """
        # Generate recommendation
        if not quality.get("usable", False):
            recommendation = "Recapture required before screening interpretation."
        elif referable:
            recommendation = "Refer for ophthalmologist review."
        else:
            recommendation = "No immediate referral required. Routine follow-up recommended."

        report = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            "model": "DR-EfficientNetB0",
            "image_quality": {
                "focus": quality.get("focus", 0),
                "illumination": quality.get("illumination", 0),
                "field_of_view": quality.get("field_of_view", 0),
                "overall": quality.get("overall", 0),
                "gradeability": "Gradeable" if quality.get("usable", False) else "Ungradeable",
            },
            "dr_result": {
                "predicted_grade": prediction.get("grade", -1),
                "severity_label": prediction.get("label", "Unknown"),
                "confidence": prediction.get("confidence", 0),
                "probabilities": prediction.get("probabilities", {}),
            },
            "referable_dr": {
                "is_referable": referable,
                "probability": referable_probability,
                "definition": "Level 2+ (P(class 2) + P(class 3) + P(class 4))",
            },
            "recommendation": recommendation,
            "disclaimer": (
                "AI-assisted screening prototype. This result is not a definitive diagnosis. "
                "Clinical review by a qualified ophthalmologist is required."
            ),
        }

        return report

    def format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = [
            "=" * 50,
            "DIABETIC RETINOPATHY SCREENING REPORT",
            "=" * 50,
            f"Case ID: {report['case_id']}",
            f"Timestamp: {report['timestamp']}",
            f"Model: {report.get('model', 'Unknown')}",
            "",
            "--- IMAGE QUALITY ---",
            f"Focus: {report['image_quality']['focus']:.1%}",
            f"Illumination: {report['image_quality']['illumination']:.1%}",
            f"Field of View: {report['image_quality']['field_of_view']:.1%}",
            f"Overall: {report['image_quality']['overall']:.1%}",
            f"Gradeability: {report['image_quality']['gradeability']}",
            "",
            "--- DR CLASSIFICATION ---",
            f"Predicted Grade: {report['dr_result']['predicted_grade']}",
            f"Severity: {report['dr_result']['severity_label']}",
            f"Confidence: {report['dr_result']['confidence']:.1%}",
            "",
            "--- REFERABLE DR ---",
            f"Is Referable: {'YES' if report['referable_dr']['is_referable'] else 'NO'}",
            f"Referable Probability: {report['referable_dr']['probability']:.1%}",
            f"Definition: {report['referable_dr']['definition']}",
            "",
            "--- RECOMMENDATION ---",
            report["recommendation"],
            "",
            "--- DISCLAIMER ---",
            report["disclaimer"],
            "=" * 50,
        ]
        return "\n".join(lines)
