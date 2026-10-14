import numpy as np
from fastapi import APIRouter

from app.schemas.drift import DriftRequest, DriftResponse
from app.services import model_service, monitoring_service

router = APIRouter(tags=["drift"])


@router.post("/drift/check", response_model=DriftResponse)
def drift_check(req: DriftRequest):
    baseline = model_service.baseline_feature(req.feature_index)
    new = np.array([row[req.feature_index] for row in req.batch], dtype=float)
    r = monitoring_service.ks_drift(baseline, new, req.p_threshold)
    return {**r, "n_baseline": len(baseline), "n_new": len(new)}
