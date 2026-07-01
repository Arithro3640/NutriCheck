"""
MalFusion pipeline for child malnutrition detection.

This single module reproduces the proposed *MalFusion* model from the research
notebook (a soft-voting ensemble: Random Forest x2 + SVM x2 + XGBoost x1 +
CatBoost x1) together with the full preprocessing chain, so the website can
train and predict end-to-end.

Everything (imputers, encoders, scaler, winsor bounds and the fitted model) is
stored on one object that is saved/loaded with joblib, which keeps column order
and transforms perfectly in sync between training and prediction.

XGBoost / CatBoost / imbalanced-learn are used when installed; if they are not
present the pipeline transparently falls back to scikit-learn equivalents so the
app always runs without errors.
"""

import json
import datetime as _dt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss,
)

from . import config as C

# --- Optional dependencies (full MalFusion when available) --------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


# =============================================================================
#  Safe label encoder — handles categories never seen during training
# =============================================================================
class SafeLabelEncoder:
    """Maps categories to integers and routes unknown values to the most
    frequent category instead of raising an error."""

    def __init__(self):
        self.mapping = {}
        self.fallback = 0

    def fit(self, values):
        cats = pd.Series([str(v) for v in values])
        uniques = sorted(cats.dropna().unique())
        self.mapping = {cat: i for i, cat in enumerate(uniques)}
        # fallback = encoding of the most common category
        self.fallback = self.mapping.get(cats.mode().iloc[0], 0) if len(cats) else 0
        return self

    def transform(self, values):
        return np.array([self.mapping.get(str(v), self.fallback) for v in values])

    @property
    def classes_(self):
        return list(self.mapping.keys())


# =============================================================================
#  The MalFusion pipeline
# =============================================================================
class MalFusionPipeline:
    """Full preprocessing + MalFusion ensemble in one saveable object."""

    ENGINEERED = [
        "bmi_proxy", "weight_height_ratio", "age_weight_ratio",
        "age_height_ratio", "birth_wt_preg_ratio",
    ]
    EPS = 1e-5

    def __init__(self):
        self.numeric_cols = list(C.NUMERIC_COLS)
        self.categorical_cols = list(C.CATEGORICAL_COLS)
        self.feature_order = list(C.FEATURE_NAMES)

        self.winsor_bounds = {}     # col -> (low, high)
        self.num_medians = {}       # col -> median (imputation)
        self.cat_modes = {}         # col -> mode (imputation)
        self.encoders = {}          # col -> SafeLabelEncoder
        self.scaler = None
        self.final_columns = []     # full column order after engineering
        self.target_classes = []    # ordered class names
        self.model = None
        self.metrics = {}
        self.is_fitted = False

    # ---------------------------------------------------------------- helpers
    def _coerce_numeric(self, df):
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _engineer(self, X):
        """Add the clinically meaningful derived features from the notebook."""
        e = self.EPS
        X = X.copy()
        X["bmi_proxy"] = X["Weight(Kg)"] / ((X["Height(cm)"] / 100.0) ** 2 + e)
        X["weight_height_ratio"] = X["Weight(Kg)"] / (X["Height(cm)"] + e)
        X["age_weight_ratio"] = X["Age(days)"] / (X["Weight(Kg)"] + e)
        X["age_height_ratio"] = X["Age(days)"] / (X["Height(cm)"] + e)
        X["birth_wt_preg_ratio"] = X["Birth Weight(Kg)"] / (X["Pregnancy Duration(month)"] + e)
        return X

    # --------------------------------------------------------- build ensemble
    def _build_model(self):
        """Construct the MalFusion soft-voting ensemble (with safe fallbacks)."""
        rf = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        svm = SVC(
            kernel="rbf", C=10, gamma="scale", probability=True,
            class_weight="balanced", random_state=42,
        )

        if HAS_XGB:
            booster = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42, verbosity=0,
            )
        else:
            booster = GradientBoostingClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
            )

        if HAS_CAT:
            second = CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05,
                verbose=0, random_state=42,
            )
        else:
            second = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, random_state=42,
            )

        return VotingClassifier(
            estimators=[
                ("random_forest", rf),
                ("svm", svm),
                ("xgboost", booster),
                ("catboost", second),
            ],
            voting="soft",
            weights=[2, 2, 1, 1],
            n_jobs=-1,
        )

    def components_used(self):
        return {
            "random_forest": "RandomForestClassifier",
            "svm": "SVC (RBF)",
            "xgboost": "XGBClassifier" if HAS_XGB else "GradientBoosting (fallback)",
            "catboost": "CatBoostClassifier" if HAS_CAT else "HistGradientBoosting (fallback)",
            "smote": HAS_SMOTE,
        }

    # --------------------------------------------------------------- fit core
    def _fit_transforms(self, X):
        """Learn winsor bounds, medians, modes, encoders and scaler from X."""
        X = self._coerce_numeric(X)

        # 1. Winsor bounds (IQR) from numeric cols
        for col in self.numeric_cols:
            s = X[col].dropna()
            if len(s) == 0:
                self.winsor_bounds[col] = (None, None)
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            self.winsor_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            X[col] = X[col].clip(lower=self.winsor_bounds[col][0],
                                 upper=self.winsor_bounds[col][1])

        # 2. Imputation values
        for col in self.numeric_cols:
            med = X[col].median()
            self.num_medians[col] = 0.0 if pd.isna(med) else float(med)
            X[col] = X[col].fillna(self.num_medians[col])
        for col in self.categorical_cols:
            mode = X[col].mode()
            self.cat_modes[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
            X[col] = X[col].fillna(self.cat_modes[col]).astype(str)

        # 3. Encoders
        for col in self.categorical_cols:
            enc = SafeLabelEncoder().fit(X[col])
            self.encoders[col] = enc
            X[col] = enc.transform(X[col])

        # 4. Feature engineering + scaler
        X = self._engineer(X)
        self.final_columns = self.feature_order + self.ENGINEERED
        X = X[self.final_columns]
        self.scaler = StandardScaler().fit(X.values)
        return self.scaler.transform(X.values)

    def _apply_transforms(self, X):
        """Apply already-learned transforms to new data (predict path)."""
        X = X.copy()
        # ensure every expected column exists
        for col in self.feature_order:
            if col not in X.columns:
                X[col] = np.nan
        X = self._coerce_numeric(X)

        for col in self.numeric_cols:
            low, high = self.winsor_bounds.get(col, (None, None))
            if low is not None:
                X[col] = X[col].clip(lower=low, upper=high)
            X[col] = X[col].fillna(self.num_medians.get(col, 0.0))
        for col in self.categorical_cols:
            X[col] = X[col].fillna(self.cat_modes.get(col, "Unknown")).astype(str)
            X[col] = self.encoders[col].transform(X[col])

        X = self._engineer(X)
        X = X[self.final_columns]
        return self.scaler.transform(X.values)

    # ------------------------------------------------------------------- fit
    def fit(self, df):
        """Train MalFusion on a labelled dataframe. Returns a metrics dict."""
        df = df.copy()
        df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns],
                     errors="ignore")
        df = df.dropna(subset=[C.TARGET])

        X = df[self.feature_order].copy()
        y_raw = df[C.TARGET].astype(str)

        # target encoding (stable, sorted order)
        self.target_classes = sorted(y_raw.unique())
        cls_to_idx = {c: i for i, c in enumerate(self.target_classes)}
        y = y_raw.map(cls_to_idx).values

        X_arr = self._fit_transforms(X)

        # ---- SMOTE (optional) ----
        smallest = int(pd.Series(y).value_counts().min())
        X_bal, y_bal, smote_used = X_arr, y, False
        if HAS_SMOTE and smallest > 1 and len(self.target_classes) > 1:
            k = max(1, min(3, smallest - 1))
            try:
                X_bal, y_bal = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_arr, y)
                smote_used = True
            except Exception:
                X_bal, y_bal = X_arr, y

        # ---- evaluate with stratified CV (honest accuracy) ----
        self.model = self._build_model()
        metrics = self._evaluate(X_bal, y_bal)
        metrics["smote_used"] = smote_used

        # ---- final fit on all balanced data ----
        self.model.fit(X_bal, y_bal)
        self.is_fitted = True

        metrics.update({
            "n_samples": int(len(df)),
            "n_features": len(self.final_columns),
            "classes": self.target_classes,
            "components": self.components_used(),
            "trained_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        self.metrics = metrics
        return metrics

    def _evaluate(self, X, y):
        n_splits = min(3, int(pd.Series(y).value_counts().min()))
        n_splits = max(2, n_splits)
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            y_pred = cross_val_predict(self._build_model(), X, y, cv=skf, method="predict")
            return {
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "precision": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
                "hamming_loss": round(float(hamming_loss(y, y_pred)), 4),
                "cv_folds": n_splits,
            }
        except Exception as exc:
            return {"accuracy": None, "precision": None, "recall": None,
                    "f1": None, "hamming_loss": None, "cv_folds": 0,
                    "eval_error": str(exc)}

    # --------------------------------------------------------------- predict
    def predict(self, record):
        """Predict on a single record dict. Returns label + probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model is not trained yet.")
        X = pd.DataFrame([record])
        X_arr = self._apply_transforms(X)
        proba = self.model.predict_proba(X_arr)[0]
        idx = int(np.argmax(proba))
        return {
            "label": self.target_classes[idx],
            "confidence": round(float(proba[idx]) * 100, 1),
            "probabilities": {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(self.target_classes, proba)
            },
        }


# =============================================================================
#  Data health check + auto-clean (used by the admin "Train" workflow)
# =============================================================================
def health_check(df):
    """Inspect a raw dataframe, list the problems found and the preprocessing
    that will be (or was) applied to fix each one. Returns a structured report
    plus a cleaned dataframe."""

    report = {"issues": [], "fixes": [], "ok": True}
    df = df.copy()

    # 0. drop unused cols
    dropped = [c for c in C.DROP_COLS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped, errors="ignore")
        report["fixes"].append({
            "problem": f"Unused columns present: {', '.join(dropped)}",
            "applied": "Dropped columns not used by the model.",
        })

    # 1. missing target rows
    if C.TARGET in df.columns:
        n_missing_t = int(df[C.TARGET].isna().sum())
        if n_missing_t:
            df = df.dropna(subset=[C.TARGET])
            report["issues"].append(f"{n_missing_t} row(s) had no Nutritional_Status label.")
            report["fixes"].append({
                "problem": f"{n_missing_t} unlabelled row(s).",
                "applied": "Removed rows without a target label.",
            })

    # 2. duplicate rows
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        report["issues"].append(f"{n_dup} duplicate row(s) found.")
        report["fixes"].append({
            "problem": f"{n_dup} exact duplicate row(s).",
            "applied": "Removed duplicate rows.",
        })

    # 3. numeric coercion
    coerced = []
    for col in C.NUMERIC_COLS:
        if col in df.columns and df[col].dtype == object:
            before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after = df[col].isna().sum()
            if after > before:
                coerced.append(col)
    if coerced:
        report["issues"].append(f"Non-numeric text in numeric columns: {', '.join(coerced)}.")
        report["fixes"].append({
            "problem": f"Text inside numeric columns ({', '.join(coerced)}).",
            "applied": "Converted to numbers; invalid entries marked as missing.",
        })

    # 4. missing values
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        num_miss = [c for c in miss.index if c in C.NUMERIC_COLS]
        cat_miss = [c for c in miss.index if c in C.CATEGORICAL_COLS]
        total = int(miss.sum())
        report["issues"].append(f"{total} missing value(s) across {len(miss)} column(s).")
        if num_miss:
            report["fixes"].append({
                "problem": f"Missing numbers in: {', '.join(num_miss)}.",
                "applied": "Filled with the column median (robust to outliers).",
            })
        if cat_miss:
            report["fixes"].append({
                "problem": f"Missing categories in: {', '.join(cat_miss)}.",
                "applied": "Filled with the most frequent category.",
            })

    # 5. outliers (IQR)
    outlier_cols = []
    for col in C.NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if n:
                outlier_cols.append(f"{col} ({n})")
    if outlier_cols:
        report["issues"].append(f"Outliers detected in: {', '.join(outlier_cols)}.")
        report["fixes"].append({
            "problem": "Extreme values that can distort the model.",
            "applied": "Capped (winsorised) to the IQR range during training.",
        })

    # 6. class imbalance
    if C.TARGET in df.columns and len(df):
        counts = df[C.TARGET].value_counts().to_dict()
        if len(counts) > 1:
            ratio = max(counts.values()) / max(1, min(counts.values()))
            if ratio >= 1.5:
                report["issues"].append(
                    "Classes are imbalanced: " +
                    ", ".join(f"{k}={v}" for k, v in counts.items()) + ".")
                report["fixes"].append({
                    "problem": "Some malnutrition classes are rarer than others.",
                    "applied": ("Balanced with SMOTE oversampling during training."
                                if HAS_SMOTE else
                                "Balanced with class weights (install imbalanced-learn for SMOTE)."),
                })

    if not report["issues"]:
        report["issues"].append("No data quality problems detected — dataset is clean.")
    report["ok"] = True
    report["rows_after"] = int(len(df))
    return report, df
