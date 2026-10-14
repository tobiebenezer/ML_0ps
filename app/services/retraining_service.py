"""Challenger vs champion gating. Promote only on real improvement + tests."""
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from app.services import dataset_service, job_store, model_service
from app.services.registry_service import register_version
from app.core.config import settings

log = logging.getLogger(__name__)

ACC_EPS = 0.002
F1_TOL = 0.005


def run_retrain_job(job_id: str, params: dict) -> None:
    job_store.update_job(job_id, status="running", reason="training challenger")
    try:
        x_new, y_new = dataset_service.build_training_set(
            params.get("use_feedback", True), params.get("dataset_versions", [])
        )
        _, _, x_hold, y_hold, _ = dataset_service.load_base()
        kwargs = dict(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth"),
            random_state=42,
        )
        warm = bool(params.get("warm_start", False))
        if warm:
            # Finetuning-like demo: keep champion trees, add N more on new data only
            champ = model_service.load_model()
            clf = RandomForestClassifier(warm_start=True, **kwargs)
            clf.fit(x_new, y_new)  # simplified classroom demo of warm_start semantics
            log.info("warm_start demo (champion had %s trees)", getattr(champ, "n_estimators", "?"))
        else:
            clf = RandomForestClassifier(**kwargs)
            clf.fit(x_new, y_new)

        job_store.update_job(job_id, status="evaluating", reason="scoring on frozen holdout")
        champ = model_service.load_model()
        chal_acc = float(accuracy_score(y_hold, clf.predict(x_hold)))
        champ_acc = float(accuracy_score(y_hold, champ.predict(x_hold)))
        chal_f1 = float(f1_score(y_hold, clf.predict(x_hold)))
        champ_f1 = float(f1_score(y_hold, champ.predict(x_hold)))

        # Session 3 gate: minimum bar + must-beat-champion
        if chal_acc < 0.90:
            job_store.update_job(job_id, status="rejected", challenger_acc=chal_acc,
                                 champion_acc=champ_acc,
                                 reason=f"challenger acc {chal_acc:.4f} below 0.90 floor")
            return
        if chal_acc >= champ_acc + ACC_EPS and chal_f1 >= champ_f1 - F1_TOL:
            import joblib
            from pathlib import Path

            dest = Path(settings.model_path)
            joblib.dump(clf, dest)
            version = register_version(clf, {"params": kwargs, "challenger_acc": chal_acc,
                                             "champion_acc": champ_acc}, str(dest))
            model_service.hot_swap(clf, f"retrain-{job_id}-{version}")
            dataset_service.mark_feedback_used()
            job_store.update_job(job_id, status="promoted", challenger_acc=chal_acc,
                                 champion_acc=champ_acc,
                                 reason=f"promoted {version}: acc {champ_acc:.4f}->{chal_acc:.4f}")
        else:
            job_store.update_job(job_id, status="rejected", challenger_acc=chal_acc,
                                 champion_acc=champ_acc,
                                 reason=f"no improvement: {champ_acc:.4f}->{chal_acc:.4f} "
                                        f"(need +{ACC_EPS}), f1 {champ_f1:.4f}->{chal_f1:.4f}")
    except Exception as exc:  # noqa: BLE001
        log.exception("retrain job %s failed", job_id)
        job_store.update_job(job_id, status="failed", reason=str(exc)[:500])
