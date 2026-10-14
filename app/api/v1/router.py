from fastapi import APIRouter

from app.api.v1 import datasets, drift, feedback, health, metrics, model, predict, retrain

router = APIRouter(prefix="/api/v1")
router.include_router(health.router)
router.include_router(predict.router)
router.include_router(model.router)
router.include_router(drift.router)
router.include_router(metrics.router)
router.include_router(feedback.router)
router.include_router(datasets.router)
router.include_router(retrain.router)
