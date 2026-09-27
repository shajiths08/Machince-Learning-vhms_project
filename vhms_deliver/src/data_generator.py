"""
Synthetic Vehicle Health Dataset Generator
============================================
Generates realistic vehicle telemetry data with correlated fault patterns,
so downstream ML models learn genuine relationships instead of noise.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12000

FAULT_CATEGORIES = [
    "None", "Engine", "Battery_Electrical", "Brake_System",
    "Cooling_System", "Transmission", "Tire_Suspension", "Exhaust_Emission"
]

PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def generate(n=N):
    df = pd.DataFrame()

    # --- Base / demographic ---
    df["vehicle_age_years"] = RNG.gamma(shape=2.2, scale=2.2, size=n).round(1)
    df["vehicle_age_years"] = clip(df["vehicle_age_years"], 0, 20)
    age = df["vehicle_age_years"].values

    df["service_history_count"] = RNG.poisson(lam=clip(age * 1.4, 0.2, 30), size=n)
    df["previous_breakdowns"] = RNG.poisson(lam=clip(age * 0.18, 0, 5), size=n)

    # --- Engine ---
    base_temp = 90 + age * 0.6 + RNG.normal(0, 6, n)
    df["engine_temperature_c"] = clip(base_temp, 70, 140)

    oil_quality_raw = 100 - age * 3.2 - df["service_history_count"] * (-0.4) + RNG.normal(0, 8, n)
    df["engine_oil_quality_pct"] = clip(oil_quality_raw, 5, 100)

    df["oil_level_pct"] = clip(95 - age * 1.1 + RNG.normal(0, 10, n), 0, 100)

    df["engine_rpm"] = clip(RNG.normal(2200, 500, n) + (100 - df["engine_oil_quality_pct"]) * 3, 600, 6000)

    df["air_filter_condition_pct"] = clip(100 - age * 2.8 + RNG.normal(0, 12, n), 0, 100)

    # --- Electrical ---
    df["battery_voltage_v"] = clip(13.6 - age * 0.06 + RNG.normal(0, 0.35, n), 9.5, 14.8)
    df["battery_health_pct"] = clip(100 - age * 4.0 + RNG.normal(0, 9, n), 0, 100)

    # --- Brakes / suspension / tires ---
    df["brake_condition_pct"] = clip(100 - age * 3.5 + RNG.normal(0, 10, n), 0, 100)
    df["tire_pressure_psi"] = clip(RNG.normal(32, 3.5, n) - age * 0.15, 12, 45)
    df["suspension_condition_pct"] = clip(100 - age * 2.6 + RNG.normal(0, 11, n), 0, 100)
    df["vibration_level_mm_s"] = clip(1.5 + age * 0.25 + (100 - df["suspension_condition_pct"]) * 0.04 + RNG.normal(0, 0.8, n), 0, 15)

    # --- Cooling / transmission ---
    df["coolant_level_pct"] = clip(95 - age * 1.3 + RNG.normal(0, 9, n), 0, 100)
    df["transmission_temp_c"] = clip(80 + age * 1.0 + RNG.normal(0, 8, n), 60, 160)

    # --- Exhaust / fuel ---
    df["exhaust_emission_level_ppm"] = clip(150 + age * 18 - df["air_filter_condition_pct"] * 1.2 + RNG.normal(0, 40, n), 20, 1200)
    df["fuel_consumption_l_per_100km"] = clip(6.5 + age * 0.18 + (100 - df["engine_oil_quality_pct"]) * 0.02 + RNG.normal(0, 0.9, n), 4, 25)
    df["fuel_tank_level_pct"] = clip(RNG.uniform(5, 100, n), 0, 100)

    # ---------------------------------------------------------------
    # Derive a composite "degradation pressure" score driving outcomes
    # ---------------------------------------------------------------
    risk = (
        0.55 * age
        + 0.10 * clip(df["engine_temperature_c"] - 95, 0, None)
        + 0.07 * (100 - df["oil_level_pct"])
        + 0.08 * (100 - df["engine_oil_quality_pct"])
        + 0.55 * clip(13.2 - df["battery_voltage_v"], 0, None)
        + 0.09 * (100 - df["battery_health_pct"])
        + 0.08 * (100 - df["brake_condition_pct"])
        + 0.30 * np.abs(df["tire_pressure_psi"] - 32)
        + 0.07 * (100 - df["suspension_condition_pct"])
        + 1.20 * clip(df["vibration_level_mm_s"] - 3, 0, None)
        + 0.07 * (100 - df["coolant_level_pct"])
        + 0.12 * clip(df["transmission_temp_c"] - 100, 0, None)
        + 1.60 * df["previous_breakdowns"]
        - 0.25 * np.log1p(df["service_history_count"])
        + 0.02 * clip(df["exhaust_emission_level_ppm"] - 300, 0, None)
        + RNG.normal(0, 4.5, n)
    )
    df["_risk"] = risk

    # --- Health score (0-100, higher = healthier) ---
    health_score = clip(100 - (risk * 1.05), 0, 100)
    df["vehicle_health_score"] = health_score.round(1)

    def health_status(s):
        if s >= 80:
            return "Healthy"
        elif s >= 60:
            return "Needs_Attention"
        elif s >= 40:
            return "At_Risk"
        else:
            return "Critical"

    df["vehicle_health_status"] = df["vehicle_health_score"].apply(health_status)

    # --- Maintenance requirement (binary) ---
    maint_prob = 1 / (1 + np.exp(-(risk - 13.0) / 5.0))
    df["maintenance_required"] = (RNG.uniform(0, 1, n) < maint_prob).astype(int)

    # --- Fault category: pick subsystem with worst relative condition ---
    def pick_fault(row):
        if row["vehicle_health_score"] >= 78 and row["maintenance_required"] == 0:
            return "None"
        # weakness = higher means more degraded/likely fault (normalized ~0-100 scale)
        weakness = {
            "Engine": (100 - row["engine_oil_quality_pct"]) * 0.4
                      + (100 - row["oil_level_pct"]) * 0.3
                      + clip(row["engine_temperature_c"] - 90, 0, None) * 1.2
                      + (100 - row["air_filter_condition_pct"]) * 0.2,
            "Battery_Electrical": (100 - row["battery_health_pct"]) * 0.8
                                   + clip(13.2 - row["battery_voltage_v"], 0, None) * 40,
            "Brake_System": (100 - row["brake_condition_pct"]) * 1.1,
            "Cooling_System": (100 - row["coolant_level_pct"]) * 0.8
                               + clip(row["engine_temperature_c"] - 95, 0, None) * 1.0,
            "Transmission": clip(row["transmission_temp_c"] - 90, 0, None) * 1.3,
            "Tire_Suspension": (100 - row["suspension_condition_pct"]) * 0.6
                                + clip(row["vibration_level_mm_s"] - 2, 0, None) * 8
                                + abs(row["tire_pressure_psi"] - 32) * 2.5,
            "Exhaust_Emission": clip(row["exhaust_emission_level_ppm"] - 250, 0, None) * 0.15,
        }
        return max(weakness, key=weakness.get)

    df["fault_category"] = df.apply(pick_fault, axis=1)

    # --- Maintenance priority ---
    def priority(row):
        if row["maintenance_required"] == 0:
            return "Low"
        s = row["vehicle_health_score"]
        if row["previous_breakdowns"] >= 3 or s < 35:
            return "Critical"
        elif s < 50:
            return "High"
        elif s < 65:
            return "Medium"
        else:
            return "Low"

    df["maintenance_priority"] = df.apply(priority, axis=1)

    # --- Estimated maintenance cost (USD) ---
    base_cost = {
        "None": 0, "Engine": 420, "Battery_Electrical": 180, "Brake_System": 220,
        "Cooling_System": 250, "Transmission": 650, "Tire_Suspension": 300, "Exhaust_Emission": 260
    }
    priority_mult = {"Low": 0.4, "Medium": 0.8, "High": 1.3, "Critical": 2.0}
    cost = df["fault_category"].map(base_cost) * df["maintenance_priority"].map(priority_mult)
    cost = cost * (1 + age * 0.03) + RNG.normal(0, 35, n)
    df["estimated_maintenance_cost_usd"] = clip(cost, 0, None).round(2)

    df.drop(columns=["_risk"], inplace=True)
    df.insert(0, "vehicle_id", [f"V{1000+i}" for i in range(n)])
    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("/home/claude/vhms/data/vehicle_health_data.csv", index=False)
    print(data.shape)
    print(data["vehicle_health_status"].value_counts())
    print(data["fault_category"].value_counts())
    print(data["maintenance_priority"].value_counts())
    print(data[["vehicle_health_score", "estimated_maintenance_cost_usd"]].describe())
