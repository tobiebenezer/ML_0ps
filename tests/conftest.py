import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    try:
        from ml.train import main as train_main

        train_main()
    except Exception:
        pass
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_features():
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer()
    return ds.data[0].tolist()
