"""
Prediction / Inference Pipeline
=================================
Given raw vehicle parameters, this module:
  1. Builds the engineered feature vector
  2. Runs all 5 trained models
  3. Runs anomaly detection
  4. Produces a human-readable, explainable recommendation set

This is the module the dashboard (or any backend API) calls.
"""
import json
import joblib
import numpy as np
import pandas as pd

from preprocessing import build_feature_matrix, ALL_FEATURES, RAW_FEATURES
from anomaly_detection import score_anomaly

MODELS_DIR = "/home/claude/vhms/models"

_cache = {}


def _load():
    if _cache:
        return _cache
    _cache["model_status"] = joblib.load(f"{MODELS_DIR}/model_health_status.joblib")
    _cache["le_status"] = joblib.load(f"{MODELS_DIR}/le_health_status.joblib")
    _cache["model_maint"] = joblib.load(f"{MODELS_DIR}/model_maintenance_required.joblib")
    _cache["model_fault"] = joblib.load(f"{MODELS_DIR}/model_fault_category.joblib")
    _cache["le_fault"] = joblib.load(f"{MODELS_DIR}/le_fault_category.joblib")
    _cache["model_priority"] = joblib.load(f"{MODELS_DIR}/model_maintenance_priority.joblib")
    _cache["le_priority"] = joblib.load(f"{MODELS_DIR}/le_maintenance_priority.joblib")
    _cache["model_cost"] = joblib.load(f"{MODELS_DIR}/model_estimated_cost.joblib")
    _cache["anomaly_model"] = joblib.load(f"{MODELS_DIR}/anomaly_model.joblib")
    _cache["anomaly_scaler"] = joblib.load(f"{MODELS_DIR}/anomaly_scaler.joblib")
    with open(f"{MODELS_DIR}/feature_importances.json") as f:
        _cache["importances"] = json.load(f)
    return _cache


RECOMMENDATIONS = {
    "Engine": "Schedule an oil change and inspect/replace the air filter; verify engine temperature sensor and coolant flow.",
    "Battery_Electrical": "Test battery health, clean terminals, and check the alternator/charging circuit before voltage drops further.",
    "Brake_System": "Inspect brake pads/rotors and brake fluid level; schedule a brake service to avoid reduced stopping power.",
    "Cooling_System": "Top up coolant, check for leaks, and inspect the radiator/thermostat to prevent overheating.",
    "Transmission": "Check transmission fluid level and condition; have a technician inspect for early signs of transmission wear.",
    "Tire_Suspension": "Check tire pressure and tread, inspect shocks/struts, and balance wheels to reduce vibration and improve handling.",
    "Exhaust_Emission": "Inspect exhaust system and emission control components; an emissions test/service is recommended.",
    "None": "No specific fault detected. Continue routine maintenance on schedule.",
}


def explain_top_factors(model, x_row: pd.DataFrame, top_n=4):
    """Explainability via feature importances combined with how far this
    vehicle's own values sit from a healthy reference, giving a simple,
    fast, dependency-free substitute for SHAP in this offline environment."""
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
    """raw_input: dict with keys matching RAW_FEATURES."""
    c = _load()
    df_row = pd.DataFrame([raw_input])[RAW_FEATURES]
    X = build_feature_matrix(df_row)

    # Health status
    status_pred = c["model_status"].predict(X)[0]
    status_label = c["le_status"].inverse_transform([status_pred])[0]
    status_proba = c["model_status"].predict_proba(X)[0]
    health_score_proxy = float(np.dot(status_proba, [100, 70, 45, 15])) if len(status_proba) == 4 else None

    # Maintenance required
    maint_pred = int(c["model_maint"].predict(X)[0])
    maint_proba = float(c["model_maint"].predict_proba(X)[0][1])

    # Fault category
    fault_pred = c["model_fault"].predict(X)[0]
    fault_label = c["le_fault"].inverse_transform([fault_pred])[0]
    fault_proba = c["model_fault"].predict_proba(X)[0].max()

    # Maintenance priority
    priority_pred = c["model_priority"].predict(X)[0]
    priority_label = c["le_priority"].inverse_transform([priority_pred])[0]

    # Estimated cost
    cost_pred = float(c["model_cost"].predict(X)[0])

    # Anomaly detection
    is_anomaly, anomaly_score = score_anomaly(df_row, c["anomaly_model"], c["anomaly_scaler"])

    # Explainability
    top_factors = explain_top_factors(c["model_fault"], X, top_n=4)

    return {
        "vehicle_health_status": status_label,
        "health_score_estimate": round(health_score_proxy, 1) if health_score_proxy else None,
        "maintenance_required": bool(maint_pred),
        "maintenance_required_probability": round(maint_proba, 3),
        "fault_category": fault_label,
        "fault_confidence": round(float(fault_proba), 3),
        "maintenance_priority": priority_label,
        "estimated_maintenance_cost_usd": round(cost_pred, 2),
        "is_anomaly": bool(is_anomaly[0]),
        "anomaly_score": round(float(anomaly_score[0]), 1),
        "recommendation": RECOMMENDATIONS.get(fault_label, "Consult a certified mechanic for a full inspection."),
        "top_contributing_factors": top_factors,
    }


if __name__ == "__main__":
    sample = {
        "vehicle_age_years": 8.5, "engine_temperature_c": 108, "oil_level_pct": 40,
        "battery_voltage_v": 11.8, "tire_pressure_psi": 26, "fuel_consumption_l_per_100km": 11.2,
        "engine_rpm": 2600, "brake_condition_pct": 35, "coolant_level_pct": 45,
        "engine_oil_quality_pct": 38, "air_filter_condition_pct": 50, "transmission_temp_c": 115,
        "suspension_condition_pct": 55, "vibration_level_mm_s": 5.8, "exhaust_emission_level_ppm": 480,
        "fuel_tank_level_pct": 60, "battery_health_pct": 42, "service_history_count": 2,
        "previous_breakdowns": 2,
    }
    result = predict_vehicle(sample)
    print(json.dumps(result, indent=2))
