import csv
import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.services import dataset_service

router = APIRouter(tags=["data"])


@router.post("/datasets/upload")
def upload(file: UploadFile = File(...)):
    raw = file.file.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    if "target" not in (reader.fieldnames or []):
        raise HTTPException(400, "CSV must contain a 'target' column + 30 feature columns")
    feats = [c for c in reader.fieldnames if c != "target"]
    if len(feats) != settings.n_features:
        raise HTTPException(400, f"Expected {settings.n_features} feature columns, got {len(feats)}")
    rows = list(reader)
    if not rows:
        raise HTTPException(400, "Empty CSV")
    existing = sorted(Path("data/datasets").glob("dataset_v*.csv"))
    nxt = len(existing) + 1
    dest = Path(f"data/datasets/dataset_v{nxt}.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(raw)
    return {"version": dest.name, "rows": len(rows)}
