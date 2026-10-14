from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas.retrain import RetrainAccepted, RetrainJob, RetrainRequest
from app.services import job_store
from app.services import retrain_queue_mem
from app.services.retraining_service import run_retrain_job

router = APIRouter(tags=["retrain"])


def _run_inline_if_no_redis(job_id: str, params: dict, background: BackgroundTasks):
    """If Redis queue unavailable, run in-process so manual retrain still works."""
    from app.core.redis_client import get_redis

    if get_redis() is None:
        background.add_task(run_retrain_job, job_id, params)


@router.post("/retrain", response_model=RetrainAccepted, status_code=202)
def retrain(req: RetrainRequest, background: BackgroundTasks):
    params = req.model_dump()
    job_id = job_store.enqueue(params)
    _run_inline_if_no_redis(job_id, params, background)
    return {"job_id": job_id, "status": "queued"}


@router.get("/retrain/jobs/{job_id}", response_model=RetrainJob)
def job_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    return {
        "job_id": job_id,
        "status": str(job.get("status", "?")),
        "challenger_acc": float(job["challenger_acc"]) if job.get("challenger_acc") else None,
        "champion_acc": float(job["champion_acc"]) if job.get("champion_acc") else None,
        "reason": str(job.get("reason", "")),
    }
