from fastapi import APIRouter

from app.schemas.predict import (
    BatchPredictionInput,
    BatchPredictionOutput,
    PredictionInput,
    PredictionOutput,
)
from app.services import model_service

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    pred, proba = model_service.predict_one(data.features)
    return {"prediction": pred, "probability": proba}


@router.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(data: BatchPredictionInput):
    for row in data.instances:
        if len(row) != 30:
            from fastapi import HTTPException

            raise HTTPException(422, f"each instance needs 30 features, got {len(row)}")
    preds, probas = model_service.predict_batch(data.instances)
    return {"predictions": preds, "probabilities": probas}
