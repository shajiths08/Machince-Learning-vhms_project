"""
Anomaly Detection
==================
Unsupervised Isolation Forest over the engineered feature space, used to flag
vehicles whose sensor signature looks statistically unusual -- this catches
emerging problems even before they fit a known fault pattern the classifiers
were trained on.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from preprocessing import build_feature_matrix


def train_anomaly_detector(df: pd.DataFrame, contamination: float = 0.06):
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

    joblib.dump(model, "/home/claude/vhms/models/anomaly_model.joblib")
    joblib.dump(scaler, "/home/claude/vhms/models/anomaly_scaler.joblib")
    return model, scaler


def score_anomaly(df_row: pd.DataFrame, model, scaler):
    """Returns (is_anomaly: bool, anomaly_score: float in [0,100], raw_score)."""
    X = build_feature_matrix(df_row)
    Xs = scaler.transform(X)
    raw_score = model.decision_function(Xs)  # higher = more normal
    pred = model.predict(Xs)  # -1 = anomaly, 1 = normal
    # Convert to an intuitive 0-100 "anomaly severity" score
    anomaly_score = np.clip((0.5 - raw_score) * 100, 0, 100)
    return (pred == -1), anomaly_score


if __name__ == "__main__":
    data = pd.read_csv("/home/claude/vhms/data/vehicle_health_data.csv")
    model, scaler = train_anomaly_detector(data)
    is_anom, score = score_anomaly(data, model, scaler)
    print(f"Flagged {is_anom.sum()} / {len(data)} vehicles as anomalous "
          f"({100*is_anom.sum()/len(data):.1f}%)")
    print("Anomaly score stats:", pd.Series(score).describe())
