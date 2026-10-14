from fastapi import APIRouter

from app.core.config import settings
from app.services import dataset_service, model_service

router = APIRouter(tags=["model"])


@router.get("/model/info")
def model_info():
    try:
        model = model_service.load_model()
        _, _, x_hold, y_hold, names = dataset_service.load_base()
        acc = float(model.score(x_hold, y_hold))
    except Exception:
        acc, names = 0.0, []
    return {
        "name": settings.model_name,
        "version": model_service.get_version(),
        "n_features": settings.n_features,
        "feature_names": names,
        "holdout_accuracy": acc,
    }
