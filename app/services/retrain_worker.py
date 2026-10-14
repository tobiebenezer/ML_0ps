"""Dedicated retrain worker (separate process). Drains Redis queue, else mem queue."""
import logging
import time

from app.core.logging import setup_logging
from app.services import job_store
from app.services import retrain_queue_mem
from app.services.retraining_service import run_retrain_job

log = setup_logging()


def main() -> None:
    from app.core.redis_client import get_redis

    log.info("retrain worker started")
    while True:
        job_id = None
        params = None
        r = get_redis()
        if r is not None:
            try:
                item = r.brpop("retrain:queue", timeout=5)
                if item:
                    job_id = item[1]
                    job = job_store.get_job(job_id) or {}
                    import ast

                    try:
                        params = ast.literal_eval(job.get("params", "{}"))
                    except Exception:
                        params = {"use_feedback": True, "dataset_versions": []}
            except Exception as exc:
                log.warning("redis queue read failed: %s", exc)
                time.sleep(2)
        if job_id is None:
            job_id = retrain_queue_mem.popleft()
            if job_id:
                params = retrain_queue_mem.params.pop(job_id, {"use_feedback": True})
            else:
                time.sleep(2)
                continue
        log.info("processing job %s", job_id)
        run_retrain_job(job_id, params or {})


if __name__ == "__main__":
    main()
