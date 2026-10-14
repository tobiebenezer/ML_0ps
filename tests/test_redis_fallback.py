"""API serves even when Redis is down (fallback path)."""


def test_redis_fallback_predict(client, sample_features, monkeypatch):
    import app.core.redis_client as rc

    monkeypatch.setattr(rc, "get_redis", lambda: None)
    r = client.post("/api/v1/predict", json={"features": sample_features})
    assert r.status_code == 200
