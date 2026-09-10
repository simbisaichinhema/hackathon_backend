# INFINITE LOOPS - SIH26038: DR Screening Backend

**Explainable AI for Diabetic Retinopathy Screening in Rural India - API server (Render deployment).**

This is the standalone backend extracted from the monorepo `AI-for-Diabetic-Retinopathy-Screening-in-Rural-India`. It serves the EfficientNetB0 5-grade DR classifier (`Aldahmashi/DR-EfficientNetB0` on Hugging Face Hub) behind a lean FastAPI service: `POST /api/screen`, `GET /api/health`, `GET /api/models/status`.

> Clinical notice: investigational triage prototype, not a medical device. All automated assessments must be reviewed by a certified ophthalmologist.

---

## Why this repo exists (why we split)

1. **Hugging Face Spaces paywall.** Gradio/Docker Spaces now need a paid plan (PRO/Team); free accounts only get Static Spaces plus up to 2 ZeroGPU Spaces. Our original Docker Space (frontend + backend + training + data) also kept hitting the 50 GB disk / out-of-space build failures because raw code, datasets, and model weights were all uploaded together.
2. **Free-tier fit.** Render Free gives 512 MB RAM and a small disk. The full monorepo backend (JAX + torch + Keras + lesion detectors + vessel analysis + Grad-CAM + scipy stack) cannot fit. So this repo ships a **lean API**: classifier + quality gate only, CPU-only torch, opencv-headless.
3. **Clean architecture.** Frontend (Vite/React) deploys to **Vercel** (free, fast). Model weights stay free on the **Hugging Face Model Hub** (regular model repo, not a Space). This repo - the Python API on **Render** (free) - downloads the weights at runtime via `hf_hub_download` into the hidden HF cache instead of bundling them, so the image stays small and builds do not run out of space.

---

## API

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/` | Root, links to `/docs` and `/api/health` |
| GET | `/api/health` | Liveness probe (used by Render health check) |
| GET | `/api/models/status` | Whether the classifier is loaded in memory |
| POST | `/api/screen` | Multipart `file` (PNG/JPG/WebP, max 10 MB) -> quality + DR grade + referable flag |

Example:

```bash
curl -X POST "https://<service>.onrender.com/api/screen" ` `-F "file=@fundus.jpg"
```

---

## Deploy on Render (free tier)

1. This folder IS the repo root (already pushed to `simbisaichinhema/hackathon_backend`).
2. Render Dashboard -> New -> Web Service -> connect the repo. Build: `pip install --upgrade pip && pip install -r requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`. Plan: Free. Health check: `/api/health`. Or New -> Blueprint (uses `render.yaml`).
3. Env vars (optional, defaults set): `DR_MODEL_ID=Aldahmashi/DR-EfficientNetB0`, `DR_MODEL_FILE=final_model.keras`. Python pinned via `runtime.txt` / `.python-version` (3.11.9).
4. Verify: `https://<service>.onrender.com/api/health` -> healthy. First `/api/screen` call downloads the model (~100-300 MB, 60-120 s on free tier), then cached until redeploy or sleep. Free services sleep after ~15 min idle.

---

## Connect the frontend (Vercel)

The Vite app in the main repo already calls `${VITE_API_BASE_URL}/api/screen` (`src/services/api.ts`). In Vercel project settings add: `VITE_API_BASE_URL=https://<service>.onrender.com` (no trailing slash), then redeploy.

---

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
# docs: http://localhost:8000/docs
```

