"""App factory. Lifespan loads model once (training vs serving, Session 1)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.middleware.latency import LatencyMiddleware

log = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services import model_service

    try:
        model_service.load_model()
        log.info("Model ready (%s)", model_service.get_version())
    except Exception as exc:
        log.warning("Starting without model: %s (run `uv run python ml/train.py`)", exc)
    # nightly scheduled retrain check (drift + manual triggers share the same queue)
    try:
        from app.services.scheduler import start_scheduler

        start_scheduler()
    except Exception as exc:
        log.warning("Scheduler not started: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)
    app.add_middleware(LatencyMiddleware)
    app.include_router(router)

    @app.get("/")
    def root():
        return {"name": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
