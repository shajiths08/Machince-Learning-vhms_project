"""
Preprocessing & Feature Engineering
=====================================
Shared transformations used both at training time and inference time so that
the dashboard and the model always see data the same way.
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "vehicle_age_years", "engine_temperature_c", "oil_level_pct", "battery_voltage_v",
    "tire_pressure_psi", "fuel_consumption_l_per_100km", "engine_rpm", "brake_condition_pct",
    "coolant_level_pct", "engine_oil_quality_pct", "air_filter_condition_pct",
    "transmission_temp_c", "suspension_condition_pct", "vibration_level_mm_s",
    "exhaust_emission_level_ppm", "fuel_tank_level_pct", "battery_health_pct",
    "service_history_count", "previous_breakdowns",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features that encode domain knowledge for the models."""
    out = df.copy()

    # Deviation-from-ideal features (distance from healthy operating band)
    out["engine_temp_deviation"] = (out["engine_temperature_c"] - 90).clip(lower=0)
    out["tire_pressure_deviation"] = (out["tire_pressure_psi"] - 32).abs()
    out["transmission_temp_deviation"] = (out["transmission_temp_c"] - 90).clip(lower=0)
    out["battery_voltage_deficit"] = (13.2 - out["battery_voltage_v"]).clip(lower=0)
    out["coolant_deficit"] = (100 - out["coolant_level_pct"]).clip(lower=0)
    out["oil_deficit"] = (100 - out["oil_level_pct"]).clip(lower=0)

    # Composite condition indices (averages across related subsystems)
    out["powertrain_condition_index"] = (
        out["engine_oil_quality_pct"] + out["oil_level_pct"] + out["air_filter_condition_pct"]
    ) / 3
    out["chassis_condition_index"] = (
        out["brake_condition_pct"] + out["suspension_condition_pct"]
    ) / 2 - out["vibration_level_mm_s"] * 2

    # Wear-rate proxy: age-normalized degradation
    out["wear_rate"] = out["previous_breakdowns"] / (out["vehicle_age_years"] + 0.5)
    out["maintenance_gap_score"] = out["vehicle_age_years"] / (out["service_history_count"] + 1)

    # Emission efficiency ratio
    out["emission_per_fuel"] = out["exhaust_emission_level_ppm"] / (out["fuel_consumption_l_per_100km"] + 1)

    # Battery stress indicator
    out["battery_stress"] = out["battery_voltage_deficit"] * (100 - out["battery_health_pct"]) / 100

    return out


ENGINEERED_FEATURES = [
    "engine_temp_deviation", "tire_pressure_deviation", "transmission_temp_deviation",
    "battery_voltage_deficit", "coolant_deficit", "oil_deficit",
    "powertrain_condition_index", "chassis_condition_index", "wear_rate",
    "maintenance_gap_score", "emission_per_fuel", "battery_stress",
]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    fe = engineer_features(df)
    return fe[ALL_FEATURES]
