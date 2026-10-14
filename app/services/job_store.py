"""Job store over Redis hash with in-memory fallback."""
import time
import uuid

from app.core.redis_client import get_redis

_mem: dict[str, dict] = {}


def create_job(params: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    payload = {"status": "queued", "reason": "", "params": str(params),
               "challenger_acc": "", "champion_acc": "", "updated_at": str(time.time())}
    r = get_redis()
    if r is not None:
        try:
            r.hset(f"jobs:{job_id}", mapping=payload)
            r.expire(f"jobs:{job_id}", 86400)
            return job_id
        except Exception:
            pass
    _mem[job_id] = {"job_id": job_id, **payload}
    return job_id


def update_job(job_id: str, **fields) -> None:
    str_fields = {k: str(v) for k, v in fields.items()}
    r = get_redis()
    if r is not None:
        try:
            r.hset(f"jobs:{job_id}", mapping=str_fields)
            return
        except Exception:
            pass
    _mem.setdefault(job_id, {"job_id": job_id}).update({"job_id": job_id, **str_fields})


def get_job(job_id: str) -> dict | None:
    r = get_redis()
    if r is not None:
        try:
            data = r.hgetall(f"jobs:{job_id}")
            if data:
                return {"job_id": job_id, **data}
        except Exception:
            pass
    return _mem.get(job_id)


def enqueue(params: dict) -> str:
    job_id = create_job(params)
    r = get_redis()
    if r is not None:
        try:
            r.lpush("retrain:queue", job_id)
            return job_id
        except Exception:
            pass
    # fallback: in-memory queue drained lazily by API process
    from app.services import retrain_queue_mem

    retrain_queue_mem.append(job_id)
    retrain_queue_mem.append_params(job_id, params)
    return job_id
