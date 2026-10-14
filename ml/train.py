"""Session 2 wrapped: train RF on sklearn breast_cancer, MLflow-log, save artifacts.

Run: uv run python ml/train.py
Outputs: ml/model.joblib, ml/baseline.json, data/base_train.npz, data/holdout.npz
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "ml").mkdir(exist_ok=True)
(ROOT / "data").mkdir(exist_ok=True)

print("Hello from train but __name__ is " + __name__)

def main() -> None:
    ds = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        ds.data, ds.target, test_size=0.2, random_state=42, stratify=ds.target
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(x_train, y_train)
    acc = float(clf.score(x_test, y_test))
    print(f"test_accuracy={acc:.4f}")

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("breast-cancer-baseline")
        with mlflow.start_run(run_name="rf_100_baseline"):
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", "None")
            mlflow.log_metric("test_accuracy", acc)
            # log joblib as generic artifact (avoids sklearn/skops trusted-type
            # friction in MLflow 3; params+metrics+artifact = Session 2 core)
            mlflow.log_artifact(str(ROOT / "ml" / "model.joblib"))
    except Exception as exc:
        print(f"MLflow skipped ({exc})")

    joblib.dump(clf, ROOT / "ml" / "model.joblib")
    np.savez(ROOT / "data" / "base_train.npz", X_train=x_train, y_train=y_train,
             feature_names=np.array(ds.feature_names))
    np.savez(ROOT / "data" / "holdout.npz", X_test=x_test, y_test=y_test)
    # baseline columns for KS drift (Session 4): keep 200 rows to stay small
    sample = x_train[:200]
    (ROOT / "ml" / "baseline.json").write_text(json.dumps(
        {"columns": [sample[:, i].tolist() for i in range(sample.shape[1])]}))
    print("saved ml/model.joblib, ml/baseline.json, data/base_train.npz, data/holdout.npz")


if __name__ == "__main__":
    main()
    
