# VHMS — AI Vehicle Health Monitoring, Fault Diagnosis & Maintenance Prediction System

An end-to-end ML system that takes 19 vehicle telemetry parameters and predicts:

1. **Vehicle health status** — Healthy / Needs Attention / At Risk / Critical
2. **Maintenance required** — binary flag
3. **Fault category** — Engine, Battery/Electrical, Brake System, Cooling System,
   Transmission, Tire/Suspension, Exhaust/Emission, or None
4. **Maintenance priority** — Low / Medium / High / Critical
5. **Estimated maintenance cost (USD)**

Plus unsupervised **anomaly detection** (Isolation Forest) to catch unusual sensor
signatures that don't match a known fault pattern yet, and a built-in
**explainability layer** that surfaces the top features driving each prediction.

## Files

```
vhms/
├── data/
│   └── vehicle_health_data.csv         # 12,000-vehicle synthetic training set
├── models/                             # trained model artifacts (.joblib)
│   ├── model_health_status.joblib
│   ├── model_maintenance_required.joblib
│   ├── model_fault_category.joblib
│   ├── model_maintenance_priority.joblib
│   ├── model_estimated_cost.joblib
│   ├── anomaly_model.joblib / anomaly_scaler.joblib
│   ├── feature_importances.json        # explainability data
│   └── metrics.json                    # held-out test metrics
├── src/
│   ├── data_generator.py               # synthetic dataset (swap in real telemetry/OBD-II/CAN data)
│   ├── preprocessing.py                # feature engineering (shared train/inference)
│   ├── anomaly_detection.py            # Isolation Forest training + scoring
│   ├── model_training.py               # trains all 5 models, prints metrics
│   ├── predict.py                      # single-vehicle inference pipeline
│   └── export_fleet.py                 # exports a fleet sample to JSON for the dashboard
└── outputs/
    ├── fleet_predictions.json          # 160-vehicle sample with full predictions
    └── vhms_dashboard.html             # interactive dashboard (open in any browser)
```

## How it works

**Data → Features.** `preprocessing.py` turns 19 raw readings into 31 model
features by adding domain-knowledge derived signals: deviation-from-healthy-band
scores (e.g. how far engine temp sits above 90°C), composite condition indices
(powertrain, chassis), a wear-rate proxy (breakdowns per year of age), and a
maintenance-gap score (age vs. service frequency).

**Models.** Each target is a separate model trained on the same feature matrix:
- `RandomForestClassifier` (class-balanced) for health status, fault category,
  and maintenance priority — robust to the class imbalance you'd see in real
  fleets, where most vehicles are healthy most of the time.
- `GradientBoostingClassifier` for the binary maintenance-required flag.
- `GradientBoostingRegressor` for estimated cost.
- `IsolationForest` (unsupervised) flags vehicles whose overall sensor profile
  is statistically unusual, independent of whether it matches a fault the
  classifiers were trained on — this is what catches *emerging* problems.

  > Note: this offline environment has no internet access, so XGBoost couldn't
  > be installed. `model_training.py` is structured so swapping
  > `GradientBoostingClassifier`/`Regressor` for `xgboost.XGBClassifier`/
  > `XGBRegressor` is a one-line change per model if you run it somewhere with
  > XGBoost available — the feature matrix and evaluation code stay identical.

**Explainability.** `predict.py` pairs each model's built-in feature
importances with the specific vehicle's own sensor values, so every
prediction comes with "here's what's driving this, and here's your actual
reading for it" — no black box. (SHAP wasn't installable offline either; this
is a lightweight, dependency-free stand-in. Swap in `shap.TreeExplainer` for
per-prediction Shapley values if you have it available.)

**Dashboard.** `outputs/vhms_dashboard.html` is a single self-contained HTML
file (data embedded, no server needed) with two views:
- **Customer view** — health gauge, plain-language detected issue,
  recommended fix, estimated cost, maintenance history.
- **Mechanic view** — fleet-wide KPIs, a critical-priority alert queue,
  predicted-vs-logged comparison, full sensor readout, and the explainability
  breakdown for diagnosis.

## Running it

```bash
cd src
python3 data_generator.py      # regenerate the synthetic dataset
python3 model_training.py      # train + evaluate all 5 models
python3 anomaly_detection.py   # train the anomaly detector
python3 predict.py             # run inference on a sample vehicle
python3 export_fleet.py        # export a fleet sample for the dashboard
```

Then open `outputs/vhms_dashboard.html` in a browser — no installation needed.

## Using real data

Replace `data_generator.py`'s output with real OBD-II/CAN-bus telemetry, shop
service records, and warranty/breakdown logs, keeping the same 19 raw column
names (or update `RAW_FEATURES` in `preprocessing.py`). Re-run
`model_training.py` — everything downstream (anomaly detection, prediction,
dashboard export) works unchanged.

## Held-out test performance (synthetic data)

See `models/metrics.json`. Headline numbers: health-status accuracy ~90%,
maintenance-priority accuracy ~97% (dominated by the common "Low" class —
expect this to shift with real, more evenly-distributed fleet data),
fault-category accuracy ~67% across 7 classes, cost regression R²~0.73,
MAE ≈ $61.
