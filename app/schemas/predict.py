"""Session 1: Pydantic validation. 30 Breast Cancer Wisconsin features, in order."""
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


class BatchPredictionInput(BaseModel):
    instances: list[list[float]] = Field(min_length=1, max_length=128)


class PredictionOutput(BaseModel):
    prediction: int
    probability: list[float]


class BatchPredictionOutput(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]
