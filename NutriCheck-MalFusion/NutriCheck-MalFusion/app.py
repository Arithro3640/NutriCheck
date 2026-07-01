"""
Child Malnutrition Detection — Flask backend (MalFusion model).

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import csv
import io
import json
import datetime as dt

import pandas as pd
from flask import (
    Flask, render_template, request, jsonify, session,
    send_file, redirect, url_for,
)

from ml import config as C
from ml import trainer

# Simple admin passcode (change this in production).
ADMIN_PASSCODE = os.environ.get("ADMIN_PASSCODE", "admin123")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "malfusion-secret-change-me")

# In-memory cache of the loaded model so we do not read disk on every request.
_MODEL = {"pipe": None}


def get_model():
    if _MODEL["pipe"] is None:
        _MODEL["pipe"] = trainer.load_model()
    return _MODEL["pipe"]


def reload_model():
    _MODEL["pipe"] = trainer.load_model()
    return _MODEL["pipe"]


# -----------------------------------------------------------------------------
#  Pages
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        features=C.FEATURES,
        groups=C.GROUP_ORDER,
        group_icons=C.GROUP_ICONS,
        class_info=C.CLASS_INFO,
    )


@app.route("/admin")
def admin():
    return render_template(
        "admin.html",
        unlocked=session.get("admin", False),
        features=C.FEATURES,
        groups=C.GROUP_ORDER,
        class_info=C.CLASS_INFO,
    )


# -----------------------------------------------------------------------------
#  Prediction API
# -----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    model = get_model()
    if model is None:
        return jsonify({"error": "Model not trained yet. Open the Admin panel and click Train."}), 400

    data = request.get_json(silent=True) or {}
    record = {}
    missing = []
    for f in C.FEATURES:
        val = data.get(f["name"], "")
        if f["kind"] == "number":
            if val in ("", None):
                missing.append(f["label"])
                record[f["name"]] = None
            else:
                try:
                    record[f["name"]] = float(val)
                except (ValueError, TypeError):
                    return jsonify({"error": f"'{f['label']}' must be a number."}), 400
        else:
            record[f["name"]] = val if val not in ("", None) else None

    # Required core measurements for a meaningful result.
    core = ["Age(days)", "Weight(Kg)", "Height(cm)"]
    core_missing = [C.feature_by_name(c)["label"] for c in core
                    if record.get(c) in (None, "")]
    if core_missing:
        return jsonify({"error": "Please fill the key measurements: " +
                                 ", ".join(core_missing) + "."}), 400

    try:
        result = model.predict(record)
    except Exception as exc:
        return jsonify({"error": f"Could not assess: {exc}"}), 500

    info = C.CLASS_INFO.get(result["label"], {})
    result["display_label"] = info.get("label", result["label"])
    result["tone"] = info.get("tone", "watch")
    result["blurb"] = info.get("blurb", "")
    return jsonify(result)


# -----------------------------------------------------------------------------
#  Admin: authentication
# -----------------------------------------------------------------------------
@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or {}
    if data.get("passcode") == ADMIN_PASSCODE:
        session["admin"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Incorrect passcode."}), 401


@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    session.pop("admin", None)
    return jsonify({"ok": True})


def require_admin():
    return session.get("admin", False)


# -----------------------------------------------------------------------------
#  Admin: save a new record to the CSV
# -----------------------------------------------------------------------------
@app.route("/api/admin/save-record", methods=["POST"])
def admin_save_record():
    if not require_admin():
        return jsonify({"error": "Admin login required."}), 403

    data = request.get_json(silent=True) or {}
    label = (data.get(C.TARGET) or "").strip()
    valid_labels = list(C.CLASS_INFO.keys())
    if label not in valid_labels:
        return jsonify({"error": "Choose a valid Nutritional Status: " +
                                 ", ".join(valid_labels) + "."}), 400

    # Assemble one row in the exact CSV column order.
    columns = C.FEATURE_NAMES + [C.TARGET]
    row = {}
    for f in C.FEATURES:
        row[f["name"]] = data.get(f["name"], "")
    row[C.TARGET] = label

    file_exists = os.path.exists(C.NEW_CSV) and os.path.getsize(C.NEW_CSV) > 0
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(C.NEW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # how many new records so far
    try:
        n = len(pd.read_csv(C.NEW_CSV))
    except Exception:
        n = 1
    return jsonify({"ok": True, "saved_to": "data/new_records.csv", "new_records": int(n)})


# -----------------------------------------------------------------------------
#  Admin: data health preview (no training)
# -----------------------------------------------------------------------------
@app.route("/api/admin/health", methods=["GET"])
def admin_health():
    if not require_admin():
        return jsonify({"error": "Admin login required."}), 403
    try:
        raw = trainer.load_full_dataset()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    from ml.pipeline import health_check
    report, clean = health_check(raw)
    return jsonify({
        "report": report,
        "rows_before": int(len(raw)),
        "rows_after": int(len(clean)),
    })


# -----------------------------------------------------------------------------
#  Admin: train (check -> auto-fix -> train -> save)
# -----------------------------------------------------------------------------
@app.route("/api/admin/train", methods=["POST"])
def admin_train():
    if not require_admin():
        return jsonify({"error": "Admin login required."}), 403
    try:
        status = trainer.train_and_save()
        reload_model()
        return jsonify({"ok": True, "status": status})
    except Exception as exc:
        return jsonify({"error": f"Training failed: {exc}"}), 500


# -----------------------------------------------------------------------------
#  Admin: model status
# -----------------------------------------------------------------------------
@app.route("/api/admin/status", methods=["GET"])
def admin_status():
    status = trainer.load_status()
    model_loaded = get_model() is not None
    new_records = 0
    if os.path.exists(C.NEW_CSV):
        try:
            new_records = int(len(pd.read_csv(C.NEW_CSV)))
        except Exception:
            new_records = 0
    return jsonify({
        "model_loaded": model_loaded,
        "status": status,
        "new_records": new_records,
        "components_available": {
            "xgboost": _safe_import("xgboost"),
            "catboost": _safe_import("catboost"),
            "imbalanced_learn (SMOTE)": _safe_import("imblearn"),
        },
    })


def _safe_import(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
#  Admin: download the saved new-records CSV
# -----------------------------------------------------------------------------
@app.route("/api/admin/download-records", methods=["GET"])
def admin_download():
    if not require_admin():
        return jsonify({"error": "Admin login required."}), 403
    if not os.path.exists(C.NEW_CSV):
        return jsonify({"error": "No new records have been saved yet."}), 404
    return send_file(C.NEW_CSV, as_attachment=True,
                     download_name="new_records.csv", mimetype="text/csv")


try:
    get_model()
except Exception as e:
    print(f"Model loading failed: {e}")

if __name__ == "__main__":
    app.run(debug=False)
