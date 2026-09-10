# DR Screening Backend (Render-ready, standalone repo)
Lean FastAPI: POST /api/screen, GET /api/health, GET /api/models/status.
Model downloads at runtime from Hugging Face Hub (not bundled).
## Deploy on Render (free)
1. Push THIS folder as its own GitHub repo root.
2. Render -> New -> Web Service -> connect repo, Build: pip install -r requirements.txt, Start: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1, Plan: Free.
3. Test: https://<service>.onrender.com/api/health
4. Frontend env: VITE_API_BASE_URL=https://<service>.onrender.com
