"""Inference pipeline for DR screening."""

from inference.pipeline import DRScreeningPipeline
from inference.quality_gate import QualityGate
from inference.classifier import DRClassifier
from inference.report_generator import ClinicalReportGenerator

# Optional modules — only import if their dependencies are available
try:
    from inference.lesion_analyzer import LesionAnalyzer
except (ImportError, ModuleNotFoundError):
    LesionAnalyzer = None

try:
    from inference.vessel_analyzer import VesselAnalyzer
except (ImportError, ModuleNotFoundError):
    VesselAnalyzer = None

__all__ = [
    "DRScreeningPipeline",
    "QualityGate",
    "DRClassifier",
    "ClinicalReportGenerator",
    "LesionAnalyzer",
    "VesselAnalyzer",
]
