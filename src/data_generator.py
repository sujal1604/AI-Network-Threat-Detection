import os
import numpy as np
import pandas as pd

FEATURES = [
    "duration", "bytes_sent", "bytes_received", "packets",
    "failed_connections", "unique_destinations", "connection_rate"
]

def generate_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["benign", "scan", "bruteforce", "dos"],
        size=n, p=[0.70, 0.12, 0.10, 0.08]
    )
    device_ids = [f"device_{i:03d}" for i in rng.integers(1, 101, n)]

    rows = []
    for label, device in zip(labels, device_ids):
        if label == "benign":
            duration = rng.gamma(2.2, 8)
            sent = rng.lognormal(8.0, 0.7)
            received = rng.lognormal(8.4, 0.7)
            packets = rng.poisson(80) + 10
            failed = rng.poisson(1)
            destinations = rng.poisson(4) + 1
            rate = rng.gamma(2, 1.5)
        elif label == "scan":
            duration = rng.gamma(1.2, 2)
            sent = rng.lognormal(6.0, 0.5)
            received = rng.lognormal(5.0, 0.5)
            packets = rng.poisson(30) + 5
            failed = rng.poisson(18) + 5
            destinations = rng.poisson(35) + 10
            rate = rng.gamma(4, 5)
        elif label == "bruteforce":
            duration = rng.gamma(1.5, 3)
            sent = rng.lognormal(6.5, 0.5)
            received = rng.lognormal(6.0, 0.5)
            packets = rng.poisson(45) + 5
            failed = rng.poisson(28) + 8
            destinations = rng.poisson(2) + 1
            rate = rng.gamma(3, 4)
        else:
            duration = rng.gamma(1.1, 1.5)
            sent = rng.lognormal(11.0, 0.7)
            received = rng.lognormal(7.0, 0.7)
            packets = rng.poisson(500) + 100
            failed = rng.poisson(4)
            destinations = rng.poisson(8) + 1
            rate = rng.gamma(7, 10)

        rows.append([device, duration, sent, received, packets, failed, destinations, rate, label])

    return pd.DataFrame(rows, columns=["device_id"] + FEATURES + ["label"])

if __name__ == "__main__":
    os.makedirs("data/generated", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/generated/network_flows.csv", index=False)
    print("Generated data/generated/network_flows.csv")
