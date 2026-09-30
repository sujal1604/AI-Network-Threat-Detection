import numpy as np
import pandas as pd

BASE_FEATURES = [
    "duration", "bytes_sent", "bytes_received", "packets",
    "failed_connections", "unique_destinations", "connection_rate"
]

def add_features(df):
    out = df.copy()
    out["bytes_total"] = out["bytes_sent"] + out["bytes_received"]
    out["bytes_per_packet"] = out["bytes_total"] / (out["packets"] + 1)
    out["failure_ratio"] = out["failed_connections"] / (out["packets"] + 1)
    out["destination_rate"] = out["unique_destinations"] / (out["duration"] + 1)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0, inplace=True)
    return out

def feature_columns():
    return BASE_FEATURES + [
        "bytes_total", "bytes_per_packet",
        "failure_ratio", "destination_rate"
    ]
