"""
Model Training
===============
Trains 5 separate supervised models sharing the same engineered feature
matrix:
  1. vehicle_health_status   (classification: Healthy/Needs_Attention/At_Risk/Critical)
  2. maintenance_required    (binary classification)
  3. fault_category          (multi-class classification)
  4. maintenance_priority    (ordinal classification: Low/Medium/High/Critical)
  5. estimated_maintenance_cost_usd (regression)

Uses RandomForest and GradientBoosting from scikit-learn (XGBoost not
available offline in this environment; GradientBoostingClassifier/Regressor
is the closest drop-in and the code is written so swapping in xgboost.XGB*
later is a one-line change per model).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder

from preprocessing import build_feature_matrix, ALL_FEATURES

MODELS_DIR = "/home/claude/vhms/models"
DATA_PATH = "/home/claude/vhms/data/vehicle_health_data.csv"


def train_classifier(X_train, X_test, y_train, y_test, name, algo="rf"):
    if algo == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42,
        )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"\n=== {name} ({algo}) ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    return model, {"accuracy": acc, "macro_f1": f1}


def train_regressor(X_train, X_test, y_train, y_test, name, algo="gb"):
    if algo == "rf":
        model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.07, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n=== {name} ({algo}) ===")
    print(f"MAE: {mae:.2f} | R2: {r2:.4f}")
    return model, {"mae": mae, "r2": r2}


def main():
    df = pd.read_csv(DATA_PATH)
    X = build_feature_matrix(df)

    metrics = {}

    # ---------- 1. Vehicle health status ----------
    le_status = LabelEncoder()
    y = le_status.fit_transform(df["vehicle_health_status"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_status, m = train_classifier(X_tr, X_te, y_tr, y_te, "Health Status", algo="rf")
    metrics["health_status"] = m
    joblib.dump(model_status, f"{MODELS_DIR}/model_health_status.joblib")
    joblib.dump(le_status, f"{MODELS_DIR}/le_health_status.joblib")

    # ---------- 2. Maintenance required ----------
    y = df["maintenance_required"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_maint, m = train_classifier(X_tr, X_te, y_tr, y_te, "Maintenance Required", algo="gb")
    metrics["maintenance_required"] = m
    joblib.dump(model_maint, f"{MODELS_DIR}/model_maintenance_required.joblib")

    # ---------- 3. Fault category ----------
    le_fault = LabelEncoder()
    y = le_fault.fit_transform(df["fault_category"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_fault, m = train_classifier(X_tr, X_te, y_tr, y_te, "Fault Category", algo="rf")
    metrics["fault_category"] = m
    joblib.dump(model_fault, f"{MODELS_DIR}/model_fault_category.joblib")
    joblib.dump(le_fault, f"{MODELS_DIR}/le_fault_category.joblib")

    # ---------- 4. Maintenance priority ----------
    le_priority = LabelEncoder()
    y = le_priority.fit_transform(df["maintenance_priority"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_priority, m = train_classifier(X_tr, X_te, y_tr, y_te, "Maintenance Priority", algo="rf")
    metrics["maintenance_priority"] = m
    joblib.dump(model_priority, f"{MODELS_DIR}/model_maintenance_priority.joblib")
    joblib.dump(le_priority, f"{MODELS_DIR}/le_maintenance_priority.joblib")

    # ---------- 5. Estimated cost (regression) ----------
    y = df["estimated_maintenance_cost_usd"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model_cost, m = train_regressor(X_tr, X_te, y_tr, y_te, "Estimated Cost", algo="gb")
    metrics["estimated_cost"] = m
    joblib.dump(model_cost, f"{MODELS_DIR}/model_estimated_cost.joblib")

    # ---------- Feature importance (explainability) ----------
    importances = {
        "health_status": dict(zip(ALL_FEATURES, model_status.feature_importances_.round(4).tolist())),
        "fault_category": dict(zip(ALL_FEATURES, model_fault.feature_importances_.round(4).tolist())),
        "maintenance_priority": dict(zip(ALL_FEATURES, model_priority.feature_importances_.round(4).tolist())),
        "estimated_cost": dict(zip(ALL_FEATURES, model_cost.feature_importances_.round(4).tolist())),
    }
    with open(f"{MODELS_DIR}/feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll models trained and saved to", MODELS_DIR)
    return metrics


if __name__ == "__main__":
    main()
