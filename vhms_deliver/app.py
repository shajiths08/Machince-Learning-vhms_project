import json
import sys
from pathlib import Path

from flask import (
    Flask,
    render_template_string,
    request,
    redirect,
    url_for,
)

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(SRC_DIR))

from predict import predict_vehicle, RAW_FEATURES  # noqa: E402


# ---------------------------------------------------------
# Flask application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# Fleet data
# ---------------------------------------------------------

FLEET_FILE = BASE_DIR / "fleet_predictions.json"


def load_fleet():
    if not FLEET_FILE.exists():
        return []

    try:
        with open(FLEET_FILE, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            # Common possible structures
            for key in [
                "vehicles",
                "fleet",
                "predictions",
                "results",
            ]:
                if isinstance(data.get(key), list):
                    return data[key]

        return []

    except Exception:
        return []


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>{{ title }} | VHMS</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: #f5f7fb;
    color: #172033;
}

.navbar {
    height: 70px;
    background: #111827;
    color: white;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 32px;
}

.logo {
    font-size: 21px;
    font-weight: 800;
}

.logo span {
    color: #60a5fa;
}

.nav-links {
    display: flex;
    gap: 8px;
}

.nav-links a {
    color: #d1d5db;
    text-decoration: none;
    padding: 9px 14px;
    border-radius: 7px;
}

.nav-links a:hover {
    background: #1f2937;
    color: white;
}

.container {
    max-width: 1250px;
    margin: 0 auto;
    padding: 32px 22px;
}

.page-title {
    margin-bottom: 6px;
    font-size: 30px;
}

.page-subtitle {
    color: #6b7280;
    margin-bottom: 28px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
}

.card {
    background: white;
    border-radius: 13px;
    padding: 22px;

    box-shadow:
        0 2px 10px rgba(0,0,0,.05);
}

.metric-label {
    color: #6b7280;
    font-size: 14px;
}

.metric-value {
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.section {
    margin-top: 25px;
}

.section-title {
    font-size: 20px;
    margin-bottom: 14px;
}

.status {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
}

.status-good {
    background: #dcfce7;
    color: #166534;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
}

.status-danger {
    background: #fee2e2;
    color: #991b1b;
}

.status-neutral {
    background: #e5e7eb;
    color: #374151;
}

form {
    background: white;
    padding: 25px;
    border-radius: 13px;
    box-shadow:
        0 2px 10px rgba(0,0,0,.05);
}

.form-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
}

.form-group {
    display: flex;
    flex-direction: column;
}

label {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 7px;
}

input {
    padding: 11px 12px;
    border: 1px solid #d1d5db;
    border-radius: 7px;
    font-size: 15px;
}

input:focus {
    outline: none;
    border-color: #3b82f6;
}

button {
    margin-top: 24px;
    padding: 12px 22px;

    border: none;
    border-radius: 8px;

    background: #2563eb;
    color: white;

    font-size: 15px;
    font-weight: 700;

    cursor: pointer;
}

button:hover {
    background: #1d4ed8;
}

.result-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(210px, 1fr));
    gap: 15px;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow:
        0 2px 10px rgba(0,0,0,.05);
}

.result-card h3 {
    margin-top: 0;
    font-size: 14px;
    color: #6b7280;
}

.result-card p {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 0;
}

.alert {
    padding: 15px;
    border-radius: 9px;
    margin-bottom: 20px;
}

.alert-error {
    background: #fee2e2;
    color: #991b1b;
}

.alert-success {
    background: #dcfce7;
    color: #166534;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
}

th {
    background: #f9fafb;
    font-size: 13px;
    color: #6b7280;
}

.factor-bar {
    margin-top: 12px;
}

.factor-name {
    font-weight: 600;
    font-size: 14px;
}

.factor-value {
    color: #6b7280;
    font-size: 13px;
}

.bar-container {
    height: 8px;
    background: #e5e7eb;
    border-radius: 5px;
    margin-top: 6px;
}

.bar {
    height: 100%;
    background: #2563eb;
    border-radius: 5px;
}

.recommendation {
    line-height: 1.7;
    color: #374151;
}

.anomaly {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
}

.no-anomaly {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}

.empty {
    padding: 40px;
    text-align: center;
    color: #6b7280;
}

.footer {
    text-align: center;
    padding: 30px;
    color: #9ca3af;
    font-size: 13px;
}

@media(max-width: 700px) {

    .navbar {
        padding: 0 15px;
    }

    .nav-links {
        display: none;
    }

    .container {
        padding: 22px 14px;
    }
}

</style>
</head>

<body>

<nav class="navbar">

    <div class="logo">
        VHMS <span>Vehicle Health Monitoring</span>
    </div>

    <div class="nav-links">
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        <a href="{{ url_for('predict') }}">Prediction</a>
        <a href="{{ url_for('fleet') }}">Fleet</a>
        <a href="{{ url_for('insights') }}">ML Insights</a>
    </div>

</nav>

<div class="container">

{{ content | safe }}

</div>

<div class="footer">
    Vehicle Health Monitoring System
</div>

</body>
</html>
"""


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def page(title, content):
    return render_template_string(
        BASE_HTML,
        title=title,
        content=content,
    )


def status_class(status):
    value = str(status).lower()

    if any(x in value for x in [
        "healthy",
        "good",
        "normal",
    ]):
        return "status-good"

    if any(x in value for x in [
        "warning",
        "moderate",
        "fair",
    ]):
        return "status-warning"

    if any(x in value for x in [
        "critical",
        "poor",
        "danger",
        "bad",
    ]):
        return "status-danger"

    return "status-neutral"


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

@app.route("/")
@app.route("/dashboard")
def dashboard():

    fleet = load_fleet()

    total = len(fleet)

    healthy = 0
    warning = 0
    critical = 0
    anomalies = 0
    maintenance = 0

    for vehicle in fleet:

        status = str(
            vehicle.get(
                "vehicle_health_status",
                vehicle.get("health_status", ""),
            )
        ).lower()

        if "healthy" in status:
            healthy += 1
        elif "warning" in status:
            warning += 1
        elif "critical" in status:
            critical += 1

        if vehicle.get(
            "is_anomaly",
            vehicle.get("anomaly", False),
        ):
            anomalies += 1

        if vehicle.get(
            "maintenance_required",
            False,
        ):
            maintenance += 1

    content = f"""
    <h1 class="page-title">VHMS Dashboard</h1>

    <p class="page-subtitle">
        Vehicle health and predictive maintenance overview
    </p>

    <div class="grid">

        <div class="card">
            <div class="metric-label">Total Vehicles</div>
            <div class="metric-value">{total}</div>
        </div>

        <div class="card">
            <div class="metric-label">Healthy Vehicles</div>
            <div class="metric-value">{healthy}</div>
        </div>

        <div class="card">
            <div class="metric-label">Warning Vehicles</div>
            <div class="metric-value">{warning}</div>
        </div>

        <div class="card">
            <div class="metric-label">Critical Vehicles</div>
            <div class="metric-value">{critical}</div>
        </div>

        <div class="card">
            <div class="metric-label">Maintenance Required</div>
            <div class="metric-value">{maintenance}</div>
        </div>

        <div class="card">
            <div class="metric-label">Anomalies Detected</div>
            <div class="metric-value">{anomalies}</div>
        </div>

    </div>

    <div class="section">

        <div class="card">

            <h2 class="section-title">
                Vehicle Health Monitoring System
            </h2>

            <p>
                Use <strong>Vehicle Prediction</strong> to enter
                live vehicle sensor measurements and run the trained
                ML pipeline.
            </p>

            <p>
                Use <strong>Fleet</strong> to inspect the predictions
                generated for the vehicle fleet.
            </p>

        </div>

    </div>
    """

    return page("Dashboard", content)


# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------

FIELD_LABELS = {
    "vehicle_age_years": "Vehicle Age (years)",
    "engine_temperature_c": "Engine Temperature (°C)",
    "oil_level_pct": "Oil Level (%)",
    "battery_voltage_v": "Battery Voltage (V)",
    "tire_pressure_psi": "Tire Pressure (PSI)",
    "fuel_consumption_l_per_100km": "Fuel Consumption (L/100km)",
    "engine_rpm": "Engine RPM",
    "brake_condition_pct": "Brake Condition (%)",
    "coolant_level_pct": "Coolant Level (%)",
    "engine_oil_quality_pct": "Engine Oil Quality (%)",
    "air_filter_condition_pct": "Air Filter Condition (%)",
    "transmission_temp_c": "Transmission Temperature (°C)",
    "suspension_condition_pct": "Suspension Condition (%)",
    "vibration_level_mm_s": "Vibration Level (mm/s)",
    "exhaust_emission_level_ppm": "Exhaust Emission (ppm)",
    "fuel_tank_level_pct": "Fuel Tank Level (%)",
    "battery_health_pct": "Battery Health (%)",
    "service_history_count": "Service History Count",
    "previous_breakdowns": "Previous Breakdowns",
}


@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None
    error = None

    if request.method == "POST":

        try:

            raw_input = {}

            for feature in RAW_FEATURES:

                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(
                        f"Please enter {FIELD_LABELS.get(feature, feature)}."
                    )

                raw_input[feature] = float(value)

            result = predict_vehicle(raw_input)

        except Exception as exc:
            error = str(exc)

    fields_html = ""

    for feature in RAW_FEATURES:

        fields_html += f"""
        <div class="form-group">

            <label for="{feature}">
                {FIELD_LABELS.get(feature, feature)}
            </label>

            <input
                type="number"
                step="any"
                id="{feature}"
                name="{feature}"
                required
                value="{request.form.get(feature, '')}"
            >

        </div>
        """

    error_html = ""

    if error:
        error_html = f"""
        <div class="alert alert-error">
            <strong>Prediction Error:</strong>
            {error}
        </div>
        """

    result_html = ""

    if result:

        anomaly_class = (
            "anomaly"
            if result["is_anomaly"]
            else "no-anomaly"
        )

        anomaly_text = (
            "Anomaly Detected"
            if result["is_anomaly"]
            else "No Anomaly Detected"
        )

        factors_html = ""

        for factor in result["top_contributing_factors"]:

            importance = factor["importance"]

            width = min(
                max(importance * 100, 3),
                100,
            )

            factors_html += f"""
            <div class="factor-bar">

                <div class="factor-name">
                    {factor["feature"]}
                </div>

                <div class="factor-value">
                    Importance:
                    {importance:.4f}
                    &nbsp; | &nbsp;
                    Value:
                    {factor["value"]}
                </div>

                <div class="bar-container">
                    <div
                        class="bar"
                        style="width:{width}%">
                    </div>
                </div>

            </div>
            """

        result_html = f"""

        <div class="section">

            <h2 class="section-title">
                Prediction Results
            </h2>

            <div class="result-grid">

                <div class="result-card">
                    <h3>Health Status</h3>
                    <p>
                        <span class="status
                        {status_class(result['vehicle_health_status'])}">
                            {result['vehicle_health_status']}
                        </span>
                    </p>
                </div>

                <div class="result-card">
                    <h3>Health Score</h3>
                    <p>
                        {result['health_score_estimate']}
                        / 100
                    </p>
                </div>

                <div class="result-card">
                    <h3>Fault Category</h3>
                    <p>
                        {result['fault_category']}
                    </p>
                </div>

                <div class="result-card">
                    <h3>Fault Confidence</h3>
                    <p>
                        {result['fault_confidence'] * 100:.1f}%
                    </p>
                </div>

                <div class="result-card">
                    <h3>Maintenance</h3>
                    <p>
                        {
                            "Required"
                            if result["maintenance_required"]
                            else "Not Required"
                        }
                    </p>
                </div>

                <div class="result-card">
                    <h3>Maintenance Probability</h3>
                    <p>
                        {result['maintenance_required_probability'] * 100:.1f}%
                    </p>
                </div>

                <div class="result-card">
                    <h3>Priority</h3>
                    <p>
                        {result['maintenance_priority']}
                    </p>
                </div>

                <div class="result-card">
                    <h3>Estimated Cost</h3>
                    <p>
                        ${result['estimated_maintenance_cost_usd']:,.2f}
                    </p>
                </div>

            </div>

        </div>

        <div class="section">

            <div class="card {anomaly_class}">

                <h2 class="section-title">
                    Anomaly Detection
                </h2>

                <p>
                    <strong>{anomaly_text}</strong>
                </p>

                <p>
                    Anomaly Score:
                    <strong>
                        {result['anomaly_score']} / 100
                    </strong>
                </p>

            </div>

        </div>

        <div class="section">

            <div class="card">

                <h2 class="section-title">
                    Recommendation
                </h2>

                <p class="recommendation">
                    {result['recommendation']}
                </p>

            </div>

        </div>

        <div class="section">

            <div class="card">

                <h2 class="section-title">
                    Top Contributing Features
                </h2>

                {factors_html}

            </div>

        </div>
        """

    content = f"""

    <h1 class="page-title">
        Vehicle Prediction
    </h1>

    <p class="page-subtitle">
        Enter the 19 vehicle parameters used by the trained
        VHMS prediction pipeline.
    </p>

    {error_html}

    <form method="POST">

        <div class="form-grid">

            {fields_html}

        </div>

        <button type="submit">
            Run Vehicle Prediction
        </button>

    </form>

    {result_html}

    """

    return page("Vehicle Prediction", content)


# ---------------------------------------------------------
# Fleet
# ---------------------------------------------------------

@app.route("/fleet")
def fleet():

    vehicles = load_fleet()

    rows = ""

    for index, vehicle in enumerate(vehicles):

        vehicle_id = (
            vehicle.get("vehicle_id")
            or vehicle.get("id")
            or f"Vehicle {index + 1}"
        )

        status = vehicle.get(
            "vehicle_health_status",
            vehicle.get("health_status", "Unknown"),
        )

        fault = vehicle.get(
            "fault_category",
            "Unknown",
        )

        priority = vehicle.get(
            "maintenance_priority",
            "Unknown",
        )

        cost = vehicle.get(
            "estimated_maintenance_cost_usd",
            vehicle.get("estimated_cost", "-"),
        )

        rows += f"""
        <tr>

            <td>{vehicle_id}</td>

            <td>
                <span class="status {status_class(status)}">
                    {status}
                </span>
            </td>

            <td>{fault}</td>

            <td>{priority}</td>

            <td>
                ${
                    f"{float(cost):,.2f}"
                    if isinstance(cost, (int, float))
                    else cost
                }
            </td>

        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="5">
                <div class="empty">
                    No fleet prediction data found.
                    Make sure fleet_predictions.json exists.
                </div>
            </td>
        </tr>
        """

    content = f"""

    <h1 class="page-title">
        Fleet
    </h1>

    <p class="page-subtitle">
        Fleet-level vehicle health predictions
    </p>

    <div class="card">

        <table>

            <thead>

                <tr>
                    <th>Vehicle</th>
                    <th>Health Status</th>
                    <th>Fault Category</th>
                    <th>Maintenance Priority</th>
                    <th>Estimated Cost</th>
                </tr>

            </thead>

            <tbody>

                {rows}

            </tbody>

        </table>

    </div>

    """

    return page("Fleet", content)


# ---------------------------------------------------------
# ML Insights
# ---------------------------------------------------------

@app.route("/insights")
def insights():

    importances = {}

    importance_file = (
        MODELS_DIR / "feature_importances.json"
    )

    if importance_file.exists():

        try:

            with open(importance_file, "r") as f:
                importances = json.load(f)

        except Exception:
            importances = {}

    items = []

    if isinstance(importances, dict):

        for feature, value in importances.items():

            try:
                numeric_value = float(value)
            except Exception:
                continue

            items.append(
                (feature, numeric_value)
            )

    items.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    importance_html = ""

    for feature, value in items:

        width = min(
            max(value * 100, 3),
            100,
        )

        importance_html += f"""
        <div class="factor-bar">

            <div class="factor-name">
                {feature}
            </div>

            <div class="factor-value">
                Importance: {value:.4f}
            </div>

            <div class="bar-container">
                <div
                    class="bar"
                    style="width:{width}%">
                </div>
            </div>

        </div>
        """

    if not importance_html:

        importance_html = """
        <div class="empty">
            feature_importances.json was not found
            or does not contain readable feature importance data.
        </div>
        """

    content = f"""

    <h1 class="page-title">
        ML Insights
    </h1>

    <p class="page-subtitle">
        Model explainability and anomaly-detection information
    </p>

    <div class="card">

        <h2 class="section-title">
            Feature Importance
        </h2>

        {importance_html}

    </div>

    <div class="section">

        <div class="card">

            <h2 class="section-title">
                Prediction Pipeline
            </h2>

            <p>
                Raw vehicle measurements are transformed using the
                project's shared feature-engineering pipeline.
            </p>

            <p>
                The resulting feature matrix is passed to the trained
                health-status, maintenance, fault-category,
                maintenance-priority, and estimated-cost models.
            </p>

            <p>
                An Isolation Forest is additionally used to identify
                statistically unusual vehicle signatures.
            </p>

        </div>

    </div>

    """

    return page("ML Insights", content)


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.route("/health")
def health():

    return {
        "status": "ok",
        "application": "VHMS",
        "models_directory": str(MODELS_DIR),
        "models_directory_exists": MODELS_DIR.exists(),
    }


# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
