import numpy as np
import pandas as pd

def normalize(s):
    s = np.asarray(s, dtype=float)
    if s.max() == s.min():
        return np.zeros_like(s)
    return (s - s.min()) / (s.max() - s.min())

def build_device_risk(df, anomaly_scores, anomaly_flags, classifier_confidence):
    x = df.copy()
    x["anomaly_score"] = anomaly_scores
    x["anomaly_flag"] = anomaly_flags.astype(int)
    x["classifier_confidence"] = classifier_confidence

    grouped = x.groupby("device_id").agg(
        flows=("device_id", "size"),
        anomaly_rate=("anomaly_flag", "mean"),
        mean_anomaly=("anomaly_score", "mean"),
        max_anomaly=("anomaly_score", "max"),
        mean_confidence=("classifier_confidence", "mean"),
        failed_connections=("failed_connections", "sum"),
        unique_destinations=("unique_destinations", "mean"),
    ).reset_index()

    a = normalize(grouped["mean_anomaly"])
    b = normalize(grouped["anomaly_rate"])
    c = normalize(grouped["failed_connections"])
    d = normalize(grouped["unique_destinations"])

    grouped["risk_score"] = 100 * (
        0.40*a + 0.30*b + 0.20*c + 0.10*d
    )
    grouped["risk_level"] = pd.cut(
        grouped["risk_score"],
        bins=[-1, 30, 60, 80, 101],
        labels=["low", "medium", "high", "critical"]
    )
    return grouped.sort_values("risk_score", ascending=False)
