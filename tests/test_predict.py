"""Session 1: valid predict works, malformed is auto-rejected (422)."""


def test_predict_valid(client, sample_features):
    r = client.post("/api/v1/predict", json={"features": sample_features})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert len(body["probability"]) == 2


def test_predict_malformed_rejected(client):
    bad = client.post("/api/v1/predict", json={})
    assert bad.status_code == 422


def test_predict_wrong_length_rejected(client):
    bad = client.post("/api/v1/predict", json={"features": [1.0, 2.0]})
    assert bad.status_code == 422


def test_health_and_ready(client):
    assert client.get("/api/v1/health").status_code == 200
    ready = client.get("/api/v1/ready").json()
    assert ready["model_loaded"] is True
