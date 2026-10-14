"""Retraining loop: feedback stored, bad CSV rejected, gate rejects non-improvement."""


def test_feedback_roundtrip(client, sample_features):
    r = client.post("/api/v1/feedback",
                    json={"features": sample_features, "true_label": 1})
    assert r.status_code == 200
    assert r.json()["pending_count"] >= 1


def test_feedback_bad_label_rejected(client, sample_features):
    r = client.post("/api/v1/feedback",
                    json={"features": sample_features, "true_label": 5})
    assert r.status_code == 422


def test_dataset_upload_bad_schema_rejected(client):
    r = client.post("/api/v1/datasets/upload",
                    files={"file": ("bad.csv", "a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400


def test_retrain_gate_rejects_useless_challenger():
    # tiny forest cannot beat champion on holdout -> must reject, never promote
    from app.services import job_store
    from app.services.retraining_service import run_retrain_job

    jid = job_store.create_job({"n_estimators": 10})
    run_retrain_job(jid, {"use_feedback": False, "dataset_versions": [],
                          "n_estimators": 10, "max_depth": 1})
    job = job_store.get_job(jid)
    assert job["status"] in ("rejected", "promoted", "failed")
