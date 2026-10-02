"""
Anomaly Detection
=================
Unsupervised Isolation Forest over the engineered feature space.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocessing import build_feature_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.06,
):
    X = build_feature_matrix(df)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(Xs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "anomaly_model.joblib",
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "anomaly_scaler.joblib",
    )

    return model, scaler


def score_anomaly(
    df_row: pd.DataFrame,
    model,
    scaler,
):
    """
    Returns:

        is_anomaly: boolean array
        anomaly_score: score in [0, 100]

    Higher anomaly_score means a more unusual vehicle signature.
    """

    X = build_feature_matrix(df_row)

    Xs = scaler.transform(X)

    raw_score = model.decision_function(Xs)

    pred = model.predict(Xs)

    # Higher raw_score = more normal.
    # Convert to intuitive 0-100 anomaly severity.
    anomaly_score = np.clip(
        (0.5 - raw_score) * 100,
        0,
        100,
    )

    return (
        pred == -1,
        anomaly_score,
    )


if __name__ == "__main__":

    data = pd.read_csv(
        DATA_DIR / "vehicle_health_data.csv"
    )

    model, scaler = train_anomaly_detector(data)

    is_anom, score = score_anomaly(
        data,
        model,
        scaler,
    )

    print(
        f"Flagged {is_anom.sum()} / {len(data)} "
        f"vehicles as anomalous "
        f"({100 * is_anom.sum() / len(data):.1f}%)"
    )

    print(
        "Anomaly score stats:",
        pd.Series(score).describe(),
    )
