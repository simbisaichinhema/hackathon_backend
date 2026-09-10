"""FastAPI backend for DR screening application."""

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import torch
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
except Exception:
    pass

import json
import uuid
import traceback
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from schemas import ScreenResponse, HealthResponse, ErrorResponse

app = FastAPI(
    title="DR Screening API",
    description="AI-assisted Diabetic Retinopathy Screening",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://drsih26038.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio

pipeline = None
pipeline_error = ""
model_warming = False


def _init_pipeline_sync():
    global pipeline, pipeline_error, model_warming
    model_warming = True
    try:
        from inference.pipeline import DRScreeningPipeline
        pipeline = DRScreeningPipeline()
        if pipeline.classifier.model is None:
            raise RuntimeError("Hugging Face classifier did not load")
        print("Model warmup complete.")
    except Exception as e:
        pipeline_error = str(e)
        print(f"Model warmup failed: {e}")
    finally:
        model_warming = False


def get_pipeline():
    global pipeline, pipeline_error
    if pipeline is None:
        _init_pipeline_sync()
    return pipeline


@app.on_event("startup")
async def warmup_model():
    """Start model loading in background thread so server is immediately reachable."""
    asyncio.create_task(asyncio.to_thread(_init_pipeline_sync))


def _validate_and_decode(file: UploadFile, contents: bytes):
    """Validate file type and decode image. Returns RGB numpy array."""
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: PNG, JPG, WebP"
        )

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size: 10MB"
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/screen")
async def screen_image(file: UploadFile = File(...)):
    """Screen a fundus image for diabetic retinopathy (non-streaming)."""
    contents = await file.read()
    image = _validate_and_decode(file, contents)

    pipe = get_pipeline()
    if pipe is None:
        case_id = f"DR-{uuid.uuid4().hex[:8].upper()}"
        return JSONResponse(
            status_code=503,
            content={
                "case_id": case_id,
                "status": "error",
                "message": "Pipeline not initialized. Check model files and dependencies.",
                "quality": None,
                "dr_prediction": None,
                "referable_dr": None,
                "report": None,
            },
        )

    try:
        results = pipe.screen(image)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@app.post("/api/screen-stream")
async def screen_image_stream(file: UploadFile = File(...)):
    """Screen a fundus image with streaming progress updates.

    Returns SSE stream with stage-by-stage progress, then final result.
    """
    contents = await file.read()
    image = _validate_and_decode(file, contents)

    pipe = get_pipeline()
    if pipe is None:
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': f'Pipeline not initialized: {pipeline_error[:300]}'})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # Tell UI we are loading the AI model (first run can take 20-60s).
    def generate():
        try:
            if pipe.classifier.model is None:
                yield f"data: {json.dumps({'type': 'stage', 'name': 'model_loading', 'message': 'Loading AI model (first run, 20-60s)...'})}\n\n"
            for event in pipe.screen_stream(image):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {str(e)[:300]}'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/models/status")
async def model_status():
    """Check model availability."""
    pipe = get_pipeline()
    classifier = pipe.classifier if pipe is not None else None
    model_loaded = (
        classifier is not None
        and classifier.model is not None
    )
    return {
        "warming": model_warming,
        "fallback": False,
        "error": pipeline_error if not model_loaded else "",
        "models": {
            "classifier": model_loaded,
        }
    }




