"""Session 3: model performance floor (>= 0.90 on frozen holdout)."""
import numpy as np

from app.core.config import settings
from app.services import model_service


def test_model_accuracy_above_threshold():
    from pathlib import Path

    model = model_service.load_model()
    hold = np.load(Path(settings.holdout_path))
    acc = model.score(hold["X_test"], hold["y_test"])
    assert acc >= 0.90, f"Model accuracy {acc} fell below the required threshold"
