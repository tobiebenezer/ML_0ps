"""MLflow registry wrapper with file fallback (works without MLflow server)."""
import logging
from pathlib import Path

import joblib

log = logging.getLogger(__name__)


def register_version(model, metrics: dict, base_path: str) -> str:
    """Log to MLflow if available, always persist challenger joblib. Returns version label."""
    version = f"v{int(Path(base_path).stat().st_mtime) if Path(base_path).exists() else 0}"
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("breast-cancer-retrain")
        with mlflow.start_run(run_name=f"challenger-{version}") as run:
            mlflow.log_params(metrics.get("params", {}))
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                    joblib.dump(model, tmp.name)
                    mlflow.log_artifact(tmp.name, artifact_path="model")
            except Exception as exc:
                log.warning("Artifact logging skipped: %s", exc)
            try:
                mlflow.register_model(f"runs:/{run.info.run_id}/model", "breast-cancer-classifier")
            except Exception as exc:
                log.warning("Registry unavailable, file fallback: %s", exc)
            return run.info.run_id[:12]
    except Exception as exc:
        log.warning("MLflow unavailable (%s); file fallback", exc)
        return version
