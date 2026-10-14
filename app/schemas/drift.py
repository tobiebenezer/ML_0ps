"""Sessions 4-5: drift + latency/cost schemas."""
from pydantic import BaseModel, Field


class DriftRequest(BaseModel):
    # batch of feature vectors; drift is checked per feature index
    batch: list[list[float]] = Field(min_length=10, max_length=2000)
    feature_index: int = Field(default=0, ge=0, le=29)
    p_threshold: float = Field(default=0.05, gt=0, lt=1)


class DriftResponse(BaseModel):
    ks_statistic: float
    p_value: float
    drifted: bool
    recommend_retrain: bool
    n_baseline: int
    n_new: int


class LatencyStats(BaseModel):
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class CostRequest(BaseModel):
    hourly_cost_usd: float = Field(default=0.10, gt=0)


class CostResponse(BaseModel):
    avg_latency_s: float
    predictions_per_hour: float
    cost_per_1000_usd: float
