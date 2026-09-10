"""Lean DR screening API for Render free tier.
Only classifier + quality gate. Model downloaded at runtime from HF Hub (not bundled).
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

MODEL_ID = os.environ.get("DR_MODEL_ID", "Aldahmashi/DR-EfficientNetB0")
MODEL_FILE = os.environ.get("DR_MODEL_FILE", "final_model.keras")
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

app = FastAPI(title="DR Screening API (lean)", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None
_load_error = ""

def get_model():
    global _model, _load_error
    if _model is not None:
        return _model
    try:
        from huggingface_hub import hf_hub_download
        import keras
        path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE)
        _model = keras.saving.load_model(path)
        print(f"Model loaded: {_model.input_shape} -> {_model.output_shape}", flush=True)
    except Exception as e:
        _load_error = str(e)
        print(f"Model load failed: {e}", flush=True)
        raise
    return _model

def quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    focus = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 400.0)
    mean_b = float(np.mean(gray))
    illum = 1.0 if 30 <= mean_b <= 220 else max(0.0, 1.0 - abs(mean_b - 125) / 125)
    _, b = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    fov = min(1.0, float((b > 0).mean()) / 0.4)
    overall = round(0.4 * focus + 0.3 * illum + 0.3 * fov, 4)
    usable = overall >= 0.35 and focus >= 0.25 and fov >= 0.30
    return {"usable": usable, "overall": overall, "focus": round(focus,4), "illumination": round(illum,4), "fov": round(fov,4)}

def preprocess(img):
    r = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.expand_dims((r - mean) / std, axis=0)

@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "2.1.0", "timestamp": datetime.now().isoformat()}

@app.get("/api/models/status")
async def status():
    loaded = _model is not None
    return {"warming": False, "fallback": False, "error": _load_error if not loaded else "", "models": {"classifier": loaded}}

@app.post("/api/screen")
async def screen(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
        raise HTTPException(400, f"Invalid type {file.content_type}. Use PNG/JPG/WebP")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Max 10MB")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    q = quality(img)
    try:
        model = get_model()
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "message": f"Model not loaded: {e}", "quality": q, "dr_prediction": None})
    probs = np.array(model.predict(preprocess(img), verbose=0)[0], dtype=float)
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(5) / 5
    pred = int(np.argmax(probs))
    referable = float(probs[2] + probs[3] + probs[4])
    return {
        "status": "success",
        "quality": q,
        "dr_prediction": {"grade": pred, "label": CLASS_NAMES[pred], "confidence": round(float(probs[pred]), 4),
                          "probabilities": {str(i): round(float(probs[i]), 4) for i in range(5)}},
        "referable_dr": pred >= 2,
        "referable_prob": round(referable, 4),
    }
