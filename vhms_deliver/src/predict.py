"""
Prediction / Inference Pipeline
================================
Given raw vehicle parameters, this module:

1. Builds the engineered feature vector
2. Runs all trained models
3. Runs anomaly detection
4. Produces a human-readable recommendation set
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import build_feature_matrix, ALL_FEATURES, RAW_FEATURES
from src.anomaly_detection import score_anomaly


# Project root = vhms_deliver/
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

_cache = {}


def _load():
    """Load all ML models once and cache them."""
    if _cache:
        return _cache

    _cache["model_status"] = joblib.load(
        MODELS_DIR / "model_health_status.joblib"
    )
    _cache["le_status"] = joblib.load(
        MODELS_DIR / "le_health_status.joblib"
    )

    _cache["model_maint"] = joblib.load(
        MODELS_DIR / "model_maintenance_required.joblib"
    )

    _cache["model_fault"] = joblib.load(
        MODELS_DIR / "model_fault_category.joblib"
    )
    _cache["le_fault"] = joblib.load(
        MODELS_DIR / "le_fault_category.joblib"
    )

    _cache["model_priority"] = joblib.load(
        MODELS_DIR / "model_maintenance_priority.joblib"
    )
    _cache["le_priority"] = joblib.load(
        MODELS_DIR / "le_maintenance_priority.joblib"
    )

    _cache["model_cost"] = joblib.load(
        MODELS_DIR / "model_estimated_cost.joblib"
    )

    _cache["anomaly_model"] = joblib.load(
        MODELS_DIR / "anomaly_model.joblib"
    )
    _cache["anomaly_scaler"] = joblib.load(
        MODELS_DIR / "anomaly_scaler.joblib"
    )

    with open(MODELS_DIR / "feature_importances.json", "r") as f:
        _cache["importances"] = json.load(f)

    return _cache


RECOMMENDATIONS = {
    "Engine": (
        "Schedule an oil change and inspect/replace the air filter; "
        "verify engine temperature sensor and coolant flow."
    ),
    "Battery_Electrical": (
        "Test battery health, clean terminals, and check the "
        "alternator/charging circuit before voltage drops further."
    ),
    "Brake_System": (
        "Inspect brake pads/rotors and brake fluid level; "
        "schedule a brake service to avoid reduced stopping power."
    ),
    "Cooling_System": (
        "Top up coolant, check for leaks, and inspect the "
        "radiator/thermostat to prevent overheating."
    ),
    "Transmission": (
        "Check transmission fluid level and condition; have a "
        "technician inspect for early signs of transmission wear."
    ),
    "Tire_Suspension": (
        "Check tire pressure and tread, inspect shocks/struts, "
        "and balance wheels to reduce vibration and improve handling."
    ),
    "Exhaust_Emission": (
        "Inspect exhaust system and emission control components; "
        "an emissions test/service is recommended."
    ),
    "None": (
        "No specific fault detected. Continue routine maintenance "
        "on schedule."
    ),
}


def explain_top_factors(model, x_row: pd.DataFrame, top_n=4):
    """
    Explainability using model feature importances.

    The model's global feature importance is combined with the
    vehicle's actual engineered feature values.
    """
    importances = model.feature_importances_
    feats = ALL_FEATURES

    order = np.argsort(importances)[::-1][:top_n]

    explanations = []

    for idx in order:
        explanations.append({
            "feature": feats[idx],
            "importance": round(float(importances[idx]), 4),
            "value": round(float(x_row.iloc[0][feats[idx]]), 2),
        })

    return explanations


def predict_vehicle(raw_input: dict) -> dict:
    """
    Run the complete VHMS prediction pipeline.

    raw_input must contain all 19 RAW_FEATURES.
    """

    # Validate input
    missing = [
        feature
        for feature in RAW_FEATURES
        if feature not in raw_input
    ]

    if missing:
        raise ValueError(
            f"Missing required vehicle parameters: {', '.join(missing)}"
        )

    c = _load()

    # Build dataframe using the exact training feature order
    df_row = pd.DataFrame([raw_input])[RAW_FEATURES]

    # Apply the exact feature engineering pipeline
    X = build_feature_matrix(df_row)

    # ---------------------------------------------------------
    # Health status
    # ---------------------------------------------------------

    status_pred = c["model_status"].predict(X)[0]

    status_label = c["le_status"].inverse_transform(
        [status_pred]
    )[0]

    status_proba = c["model_status"].predict_proba(X)[0]

    # Preserve the scoring logic from the original model
    health_score_proxy = (
        float(np.dot(status_proba, [100, 70, 45, 15]))
        if len(status_proba) == 4
        else None
    )

    # ---------------------------------------------------------
    # Maintenance required
    # ---------------------------------------------------------

    maint_pred = int(
        c["model_maint"].predict(X)[0]
    )

    maint_proba = float(
        c["model_maint"].predict_proba(X)[0][1]
    )

    # ---------------------------------------------------------
    # Fault category
    # ---------------------------------------------------------

    fault_pred = c["model_fault"].predict(X)[0]

    fault_label = c["le_fault"].inverse_transform(
        [fault_pred]
    )[0]

    fault_proba = c["model_fault"].predict_proba(X)[0].max()

    # ---------------------------------------------------------
    # Maintenance priority
    # ---------------------------------------------------------

    priority_pred = c["model_priority"].predict(X)[0]

    priority_label = c["le_priority"].inverse_transform(
        [priority_pred]
    )[0]

    # ---------------------------------------------------------
    # Estimated maintenance cost
    # ---------------------------------------------------------

    cost_pred = float(
        c["model_cost"].predict(X)[0]
    )

    # ---------------------------------------------------------
    # Anomaly detection
    # ---------------------------------------------------------

    is_anomaly, anomaly_score = score_anomaly(
        df_row,
        c["anomaly_model"],
        c["anomaly_scaler"],
    )

    # ---------------------------------------------------------
    # Explainability
    # ---------------------------------------------------------

    top_factors = explain_top_factors(
        c["model_fault"],
        X,
        top_n=4,
    )

    return {
        "vehicle_health_status": status_label,
        "health_score_estimate": (
            round(health_score_proxy, 1)
            if health_score_proxy is not None
            else None
        ),
        "maintenance_required": bool(maint_pred),
        "maintenance_required_probability": round(
            maint_proba, 3
        ),
        "fault_category": fault_label,
        "fault_confidence": round(
            float(fault_proba), 3
        ),
        "maintenance_priority": priority_label,
        "estimated_maintenance_cost_usd": round(
            cost_pred, 2
        ),
        "is_anomaly": bool(is_anomaly[0]),
        "anomaly_score": round(
            float(anomaly_score[0]), 1
        ),
        "recommendation": RECOMMENDATIONS.get(
            fault_label,
            "Consult a certified mechanic for a full inspection.",
        ),
        "top_contributing_factors": top_factors,
    }
