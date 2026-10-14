from fastapi import APIRouter

from app.schemas.drift import CostRequest, CostResponse, LatencyStats
from app.services import monitoring_service

router = APIRouter(tags=["metrics"])


@router.get("/metrics/latency", response_model=LatencyStats)
def latency():
    return monitoring_service.latency_stats()


@router.post("/metrics/cost", response_model=CostResponse)
def cost(req: CostRequest):
    return monitoring_service.cost_per_1000(req.hourly_cost_usd)
