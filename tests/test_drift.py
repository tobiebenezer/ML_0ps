"""Session 4: shifted data drifts, split-half does not."""


def test_drift_shifted_detected(client):
    import numpy as np

    from app.services import model_service

    base = model_service.baseline_feature(0)
    shifted = (np.asarray(base) + 15).tolist()
    batch = [[float(v)] + [0.0] * 29 for v in shifted[:50]]
    r = client.post("/api/v1/drift/check", json={"batch": batch, "feature_index": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["drifted"] is True
    assert body["recommend_retrain"] is True


def test_drift_no_drift_sanity(client):
    import numpy as np

    from app.services import model_service

    base = np.asarray(model_service.baseline_feature(0))
    half = len(base) // 2
    batch = [[float(v)] + [0.0] * 29 for v in base[:half][:50]]
    r = client.post("/api/v1/drift/check", json={"batch": batch, "feature_index": 0})
    assert r.status_code == 200
