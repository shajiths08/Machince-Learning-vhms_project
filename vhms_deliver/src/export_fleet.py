import json
import pandas as pd
from predict import predict_vehicle, RAW_FEATURES

df = pd.read_csv("/home/claude/vhms/data/vehicle_health_data.csv")
sample = df.sample(n=160, random_state=7).reset_index(drop=True)

fleet = []
for _, row in sample.iterrows():
    raw = {k: float(row[k]) for k in RAW_FEATURES}
    pred = predict_vehicle(raw)
    fleet.append({
        "vehicle_id": row["vehicle_id"],
        "raw": raw,
        "actual": {
            "vehicle_health_score": float(row["vehicle_health_score"]),
            "vehicle_health_status": row["vehicle_health_status"],
            "fault_category": row["fault_category"],
            "maintenance_priority": row["maintenance_priority"],
            "estimated_maintenance_cost_usd": float(row["estimated_maintenance_cost_usd"]),
        },
        "prediction": pred,
    })

with open("/home/claude/vhms/outputs/fleet_predictions.json", "w") as f:
    json.dump(fleet, f, indent=1)

print(f"Exported {len(fleet)} vehicles")
print("Sample:", json.dumps(fleet[0], indent=2)[:800])
