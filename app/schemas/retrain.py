"""Retraining-loop schemas: feedback, datasets, jobs."""
from typing import Literal

from pydantic import BaseModel, Field


class FeedbackIn(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)
    true_label: Literal[0, 1]
    source: str = "manual"


class FeedbackOut(BaseModel):
    feedback_id: int
    pending_count: int


class DatasetStats(BaseModel):
    base_train_rows: int
    feedback_pending_rows: int
    dataset_versions: list[str]
    class_balance: dict[str, int]


class RetrainRequest(BaseModel):
    use_feedback: bool = True
    dataset_versions: list[str] = Field(default_factory=list)
    n_estimators: int = Field(default=100, ge=10, le=1000)
    max_depth: int | None = None
    warm_start: bool = False  # finetuning-like: add trees only (demo of trade-off)


class RetrainAccepted(BaseModel):
    job_id: str
    status: str = "queued"


class RetrainJob(BaseModel):
    job_id: str
    status: str  # queued|running|evaluating|promoted|rejected|failed
    challenger_acc: float | None = None
    champion_acc: float | None = None
    reason: str = ""
