# Breast Cancer MLOps API (Module 7, Track 2)

Production-style modular FastAPI covering Sessions 1–5 + continual retraining.

| Session | Code |
|---|---|
| 1 Serving + Docker | `app/api/v1/predict.py`, `health.py`, `Dockerfile` |
| 2 MLflow tracking/registry | `ml/train.py`, `app/services/registry_service.py`, `GET /model/info` |
| 3 CI/CD tests | `tests/`, `.github/workflows/test.yml` |
| 4 Drift/latency/cost | `POST /drift/check`, `GET /metrics/latency`, `POST /metrics/cost` |
| 5 Percentiles/KS/cost formula | `app/services/monitoring_service.py` |
| Retrain loop | `POST /feedback`, `POST /datasets/upload`, `POST /retrain`, worker |

## Quickstart (uv + your local Redis)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv sync --group dev
uv run python ml/train.py          # trains RF, saves ml/model.joblib + baseline
uv run uvicorn app.main:app --reload --port 8000        # terminal 1: API
uv run python -m app.services.retrain_worker            # terminal 2: worker
```

Docs: http://localhost:8000/docs — try valid vs `{}` malformed predict (auto-422, Session 1).

## Retrain demo

1. `POST /feedback` ~60 corrected rows (or upload CSV to `/datasets/upload`).
2. `POST /drift/check` → `recommend_retrain:true`.
3. `POST /retrain {"use_feedback": true}` → `job_id`.
4. `GET /retrain/jobs/{id}` → `promoted` (gated: must beat champion +0.002 acc, floor 0.90) or `rejected` with reason.
5. `GET /model/info` version bumps; MLflow UI shows new run + registry version.

RF note: no gradient finetuning — `warm_start:true` demo adds trees only; default full retrain on combined data is the correct path.

## Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
# or with worker, reusing host Redis:
docker compose up
```
