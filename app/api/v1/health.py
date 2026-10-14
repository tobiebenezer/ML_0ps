from fastapi import APIRouter

from app.core.redis_client import redis_ping
from app.services import model_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "redis": "up" if redis_ping() else "down"}


@router.get("/ready")
def ready():
    try:
        model_service.load_model()
        return {"model_loaded": True, "model_version": model_service.get_version()}
    except Exception as exc:
        return {"model_loaded": False, "error": str(exc)[:300]}
