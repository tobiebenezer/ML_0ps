"""Model loading + inference. Session 1 serving, Session 2 registry fallback."""
import json
import logging
from pathlib import Path

import joblib
import numpy as np

from app.core.config import settings

log = logging.getLogger(__name__)

_model = None
_model_version = "local-v1"


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # ml/model.joblib is relative to repo root; app/ lives one level deeper
    candidates = [Path.cwd() / p, Path(__file__).resolve().parents[2] / p]
    for c in candidates:
        if c.exists():
            return c
    return candidates[1]


def load_model():
    """Load joblib, else MLflow registry URI, else raise with helpful message."""
    global _model, _model_version
    if _model is not None:
        return _model
    path = _resolve_path(settings.model_path)
    if path.exists():
        _model = joblib.load(path)
        log.info("Loaded model from %s", path)
        return _model
    if settings.mlflow_model_uri:
        import mlflow.sklearn

        _model = mlflow.sklearn.load_model(settings.mlflow_model_uri)
        _model_version = settings.mlflow_model_uri
        log.info("Loaded model from MLflow %s", settings.mlflow_model_uri)
        return _model
    raise FileNotFoundError(
        f"Model not found at {path}. Run `uv run python ml/train.py` first."
    )


def get_model():
    return load_model()


def get_version() -> str:
    return _model_version


def set_version(v: str) -> None:
    global _model_version
    _model_version = v


def hot_swap(model, version: str) -> None:
    """Zero-downtime champion swap after gated promotion."""
    global _model
    _model = model
    set_version(version)


def predict_one(features: list[float]) -> tuple[int, list[float]]:
    model = load_model()
    x = np.array(features, dtype=float).reshape(1, -1)
    return int(model.predict(x)[0]), [float(v) for v in model.predict_proba(x)[0]]


def predict_batch(instances: list[list[float]]) -> tuple[list[int], list[list[float]]]:
    model = load_model()
    x = np.array(instances, dtype=float)
    preds = [int(v) for v in model.predict(x)]
    probas = [[float(v) for v in row] for row in model.predict_proba(x)]
    return preds, probas


def baseline_feature(i: int) -> np.ndarray:
    """Training baseline column for drift KS test (Session 4)."""
    bpath = _resolve_path(settings.baseline_path)
    data = json.loads(bpath.read_text())
    return np.array(data["columns"][i], dtype=float)
