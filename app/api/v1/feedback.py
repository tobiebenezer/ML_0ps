from fastapi import APIRouter

from app.schemas.retrain import DatasetStats, FeedbackIn, FeedbackOut
from app.services import dataset_service

router = APIRouter(tags=["data"])


@router.post("/feedback", response_model=FeedbackOut)
def feedback(item: FeedbackIn):
    fid, pending = dataset_service.add_feedback(item.features, item.true_label, item.source)
    return {"feedback_id": fid, "pending_count": pending}


@router.get("/datasets/stats", response_model=DatasetStats)
def stats():
    return dataset_service.dataset_stats()
