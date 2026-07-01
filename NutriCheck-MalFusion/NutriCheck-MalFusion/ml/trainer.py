"""
Training helpers used by both the command line and the admin "Train" button.

    python -m ml.trainer          # train from data/ and save the model
"""

import os
import json
import joblib
import pandas as pd

from . import config as C
from .pipeline import MalFusionPipeline, health_check


def load_full_dataset():
    """Combine the base dataset with any admin-saved new records."""
    frames = []
    if os.path.exists(C.BASE_CSV):
        frames.append(pd.read_csv(C.BASE_CSV))
    if os.path.exists(C.NEW_CSV) and os.path.getsize(C.NEW_CSV) > 0:
        try:
            frames.append(pd.read_csv(C.NEW_CSV))
        except Exception:
            pass
    if not frames:
        raise FileNotFoundError("No dataset found in the data/ folder.")
    df = pd.concat(frames, ignore_index=True)
    return df


def train_and_save():
    """Run the full workflow: health check -> clean -> train -> save.
    Returns a status dict that also gets written to models/status.json."""
    raw = load_full_dataset()
    n_base = len(pd.read_csv(C.BASE_CSV)) if os.path.exists(C.BASE_CSV) else 0
    n_new = len(raw) - n_base

    report, clean = health_check(raw)

    pipe = MalFusionPipeline()
    metrics = pipe.fit(clean)

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, C.MODEL_PATH)

    status = {
        "model_name": "MalFusion",
        "status": "ready",
        "metrics": metrics,
        "health": report,
        "data": {
            "total_rows": int(len(raw)),
            "base_rows": int(n_base),
            "new_rows": int(max(0, n_new)),
            "rows_after_cleaning": report.get("rows_after", len(clean)),
        },
    }
    with open(C.STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)
    return status


def load_model():
    """Load the trained pipeline, or None if it has not been trained yet."""
    if os.path.exists(C.MODEL_PATH):
        try:
            return joblib.load(C.MODEL_PATH)
        except Exception:
            return None
    return None


def load_status():
    if os.path.exists(C.STATUS_PATH):
        try:
            with open(C.STATUS_PATH) as f:
                return json.load(f)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    s = train_and_save()
    print("MalFusion trained.")
    print("  Accuracy :", s["metrics"].get("accuracy"))
    print("  F1 score :", s["metrics"].get("f1"))
    print("  Rows     :", s["data"]["total_rows"])
    print("  Components:", s["metrics"]["components"])
