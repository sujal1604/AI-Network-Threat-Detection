import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data_generator import generate_dataset
from features import add_features, feature_columns
from models import train_classifier, train_anomaly_model, save_model
from risk import build_device_risk
from evaluate import save_confusion_matrix, save_risk_plot, save_feature_importance

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)

# 1. Data generation / loading
df = generate_dataset(n=6000, seed=42)
df = add_features(df)
cols = feature_columns()

X = df[cols]
y = df["label"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42, stratify=y
)

# 2. Supervised classification
clf, pred, report, cm = train_classifier(X_train, X_test, y_train, y_test)
proba = clf.predict_proba(X)
confidence = proba.max(axis=1)

with open(os.path.join(OUT, "classification_report.txt"), "w") as f:
    f.write(report)
save_confusion_matrix(cm, clf.classes_, os.path.join(OUT, "plots/confusion_matrix.png"))
fi = save_feature_importance(clf, cols, os.path.join(OUT, "feature_importance.csv"))
save_model(clf, "models/threat_classifier.joblib")

# 3. Unsupervised anomaly detection
anom, scaler, anomaly_scores, anomaly_flags = train_anomaly_model(
    X_train, X
)
save_model(anom, "models/isolation_forest.joblib")
save_model(scaler, "models/feature_scaler.joblib")

# 4. Behavioral/device risk ranking
risk = build_device_risk(
    df, anomaly_scores, anomaly_flags, confidence
)
risk.to_csv(os.path.join(OUT, "device_risk_ranking.csv"), index=False)
save_risk_plot(risk, os.path.join(OUT, "plots/device_risk.png"))

# 5. Metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "macro_precision": float(precision_score(y_test, pred, average="macro")),
    "macro_recall": float(recall_score(y_test, pred, average="macro")),
    "macro_f1": float(f1_score(y_test, pred, average="macro")),
    "anomaly_count": int(anomaly_flags.sum()),
    "anomaly_rate": float(anomaly_flags.mean()),
    "devices_ranked": int(risk.shape[0])
}
with open(os.path.join(OUT, "experiment_summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# 6. Robustness experiment: feature noise
rng = np.random.default_rng(123)
X_noisy = X.copy()
for col in cols[:4]:
    scale = max(X_noisy[col].std(), 1e-9)
    X_noisy[col] += rng.normal(0, 0.10 * scale, len(X_noisy))

noisy_pred = clf.predict(X_noisy.loc[X_test.index])
metrics["noisy_feature_accuracy"] = float(
    accuracy_score(y_test, noisy_pred)
)

with open(os.path.join(OUT, "experiment_summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Project completed.")
print(json.dumps(metrics, indent=2))
print(f"Risk ranking: {OUT}/device_risk_ranking.csv")
print(f"Classification report: {OUT}/classification_report.txt")
