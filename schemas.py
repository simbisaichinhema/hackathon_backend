"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class QualityResult(BaseModel):
    usable: bool = True
    focus: float = 0.95
    illumination: float = 0.90
    field_of_view: float = 0.95
    overall: float = 0.95
    focus_score: Optional[float] = None
    illumination_score: Optional[float] = None
    fov_score: Optional[float] = None
    fundus_score: Optional[float] = None
    overall_score: Optional[float] = None
    reason: Optional[str] = None
    method: Optional[str] = "heuristic"

    class Config:
        extra = "allow"


class DRPrediction(BaseModel):
    grade: int
    label: str
    confidence: float
    probabilities: Dict[str, float]


class ReferableDR(BaseModel):
    is_referable: bool
    probability: float = 0.0
    definition: str = "Level 2+"


class ScreenResponse(BaseModel):
    case_id: str
    status: str
    quality: Optional[Dict[str, Any]] = None
    dr_prediction: Optional[Dict[str, Any]] = None
    referable_dr: Optional[Dict[str, Any]] = None
    lesions: Optional[Dict[str, Any]] = None
    gradcam_image: Optional[str] = None
    enhanced_image: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    class Config:
        extra = "allow"


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    detail: str

