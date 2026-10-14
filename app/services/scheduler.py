"""Nightly scheduled retrain: enqueue when enough new feedback exists."""
import logging

log = logging.getLogger(__name__)
_started = False


def start_scheduler() -> None:
    global _started
    if _started:
        return
    from apscheduler.schedulers.background import BackgroundScheduler

    from app.core.config import settings
    from app.services import dataset_service, job_store

    def check():
        try:
            _, _, _, _, _ = dataset_service.load_base()
            s = dataset_service.dataset_stats()
            if s["feedback_pending_rows"] >= settings.retrain_min_new_rows:
                jid = job_store.enqueue({"use_feedback": True, "dataset_versions": [],
                                         "trigger": "scheduled"})
                log.info("scheduled retrain enqueued %s (%d pending rows)",
                         jid, s["feedback_pending_rows"])
        except Exception as exc:
            log.warning("scheduled check failed: %s", exc)

    sched = BackgroundScheduler()
    sched.add_job(check, "cron", hour=settings.retrain_cron_hour)
    sched.start()
    _started = True
    log.info("scheduler started (cron hour=%s)", settings.retrain_cron_hour)
