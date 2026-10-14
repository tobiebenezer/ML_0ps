"""Dataset merging: base_train + feedback.db + versioned CSVs. Frozen holdout never leaks."""
import csv
import sqlite3
from pathlib import Path

import numpy as np

from app.core.config import settings


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _root() / p


def init_feedback_db() -> Path:
    db = _resolve(settings.feedback_db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS feedback("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "features TEXT NOT NULL, target INTEGER NOT NULL, source TEXT, used INTEGER DEFAULT 0)"
        )
    return db


def add_feedback(features: list[float], target: int, source: str = "manual") -> tuple[int, int]:
    db = init_feedback_db()
    import json

    with sqlite3.connect(db) as con:
        cur = con.execute(
            "INSERT INTO feedback(features, target, source) VALUES (?,?,?)",
            (json.dumps([float(v) for v in features]), int(target), source),
        )
        fid = cur.lastrowid
        pending = con.execute("SELECT COUNT(*) FROM feedback WHERE used=0").fetchone()[0]
    return int(fid), int(pending)


def pending_feedback() -> tuple[np.ndarray, np.ndarray]:
    import json

    db = init_feedback_db()
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT features, target FROM feedback WHERE used=0").fetchall()
    if not rows:
        return np.zeros((0, settings.n_features)), np.zeros((0,), dtype=int)
    x = np.array([json.loads(f) for f, _ in rows], dtype=float)
    y = np.array([t for _, t in rows], dtype=int)
    return x, y


def mark_feedback_used() -> None:
    db = init_feedback_db()
    with sqlite3.connect(db) as con:
        con.execute("UPDATE feedback SET used=1 WHERE used=0")


def load_base() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(_resolve(settings.base_train_path))
    hold = np.load(_resolve(settings.holdout_path))
    names = list(d["feature_names"])
    return d["X_train"], d["y_train"], hold["X_test"], hold["y_test"], names


def load_csv_version(version: str) -> tuple[np.ndarray, np.ndarray]:
    """version like 'v2' -> data/datasets/dataset_vv2.csv -> dataset_v2.csv handling."""
    v = version.lstrip("v")
    if not version.startswith("dataset_"):
        version = f"dataset_v{v}.csv"
    path = _root() / "data" / "datasets" / version
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "target" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a 'target' column + 30 feature columns")
        feats = [c for c in reader.fieldnames if c != "target"]
        if len(feats) != settings.n_features:
            raise ValueError(f"Expected {settings.n_features} feature columns, got {len(feats)}")
        rows = list(reader)
    x = np.array([[float(r[c]) for c in feats] for r in rows], dtype=float)
    y = np.array([int(float(r["target"])) for r in rows], dtype=int)
    return x, y


def build_training_set(use_feedback: bool, dataset_versions: list[str]):
    x_train, y_train, _, _, _ = load_base()
    xs, ys = [x_train], [y_train]
    if use_feedback:
        xf, yf = pending_feedback()
        if len(xf):
            xs.append(xf)
            ys.append(yf)
    for v in dataset_versions:
        xv, yv = load_csv_version(v)
        xs.append(xv)
        ys.append(yv)
    import numpy as np

    return np.vstack(xs), np.concatenate(ys)


def dataset_stats() -> dict:
    import glob

    x_train, y_train, _, _, _ = load_base()
    xf, _ = pending_feedback()
    versions = sorted(Path(_root() / "data" / "datasets").glob("dataset_v*.csv"))
    uniq, counts = np.unique(y_train, return_counts=True)
    return {
        "base_train_rows": int(len(x_train)),
        "feedback_pending_rows": int(len(xf)),
        "dataset_versions": [p.name for p in versions],
        "class_balance": {str(k): int(v) for k, v in zip(uniq, counts)},
    }
