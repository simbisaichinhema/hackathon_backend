"""Explainability modules for DR screening."""

# Optional modules — only import if their dependencies are available
try:
    from explainability.gradcam import compute_gradcam, generate_gradcam_overlay
except (ImportError, ModuleNotFoundError):
    compute_gradcam = None
    generate_gradcam_overlay = None

try:
    from explainability.lesion_evidence import analyze_lesions, generate_lesion_overlay
except (ImportError, ModuleNotFoundError):
    analyze_lesions = None
    generate_lesion_overlay = None

try:
    from explainability.confidence import CalibratedConfidence
except (ImportError, ModuleNotFoundError):
    CalibratedConfidence = None

__all__ = [
    "compute_gradcam",
    "generate_gradcam_overlay",
    "analyze_lesions",
    "generate_lesion_overlay",
    "CalibratedConfidence",
]
