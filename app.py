
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_generator import generate_dataset
from src.features import add_features, feature_columns
from src.models import train_classifier, train_anomaly_model
from src.risk import build_device_risk

st.set_page_config(
    page_title="AI Network Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Powered Network Threat Detection")
st.caption("Behavioral Anomaly Intelligence & Device Risk Dashboard")

st.markdown("""
This defensive cybersecurity dashboard combines **machine learning classification,
unsupervised anomaly detection, behavioral analysis, and device risk scoring**.
""")

@st.cache_data
def get_demo_data():
    return generate_dataset(n=6000, seed=42)

def run_pipeline(df):
    df = add_features(df)
    cols = feature_columns()
    X = df[cols]
    y = df["label"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf, pred, report, cm = train_classifier(X_train, X_test, y_train, y_test)
    proba = clf.predict_proba(X)
    confidence = proba.max(axis=1)

    anom, scaler, anomaly_scores, anomaly_flags = train_anomaly_model(
        X_train, X
    )

    risk = build_device_risk(
        df, anomaly_scores, anomaly_flags, confidence
    )

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="macro"),
        "recall": recall_score(y_test, pred, average="macro"),
        "f1": f1_score(y_test, pred, average="macro"),
        "anomalies": int(anomaly_flags.sum()),
        "devices": int(risk.shape[0])
    }
    return df, clf, pred, cm, report, anomaly_flags, anomaly_scores, risk, metrics, cols

with st.sidebar:
    st.header("⚙️ Data Source")
    source = st.radio("Choose dataset", ["Demo Dataset", "Upload CSV"])

    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload network-flow CSV", type=["csv"])
        st.info("CSV should contain network-flow features and a 'label' column.")

    st.divider()
    st.header("🎯 Project Modules")
    st.write("✓ Threat Classification")
    st.write("✓ Anomaly Detection")
    st.write("✓ Behavioral Analysis")
    st.write("✓ Device Risk Ranking")
    st.write("✓ Model Evaluation")

if source == "Upload CSV" and uploaded is not None:
    raw = pd.read_csv(uploaded)
else:
    raw = get_demo_data()

required = [
    "device_id", "duration", "bytes_sent", "bytes_received",
    "packets", "failed_connections", "unique_destinations",
    "connection_rate", "label"
]

missing = [c for c in required if c not in raw.columns]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

if st.button("🚀 Run AI Security Analysis", type="primary"):
    with st.spinner("Training models and analyzing network behavior..."):
        (
            df, clf, pred, cm, report,
            anomaly_flags, anomaly_scores,
            risk, metrics, cols
        ) = run_pipeline(raw)

    st.session_state["results"] = {
        "df": df, "clf": clf, "pred": pred, "cm": cm,
        "report": report, "anomaly_flags": anomaly_flags,
        "anomaly_scores": anomaly_scores, "risk": risk,
        "metrics": metrics, "cols": cols
    }

if "results" not in st.session_state:
    st.info("Click **Run AI Security Analysis** to generate the dashboard.")
    st.subheader("📌 Project Objective")
    st.write(
        "Detect known threat categories, identify unusual network behavior, "
        "analyze device behavior, and prioritize potentially compromised devices."
    )
    st.stop()

r = st.session_state["results"]
metrics = r["metrics"]

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
c2.metric("Macro F1", f"{metrics['f1']:.1%}")
c3.metric("Anomalies", metrics["anomalies"])
c4.metric("Devices", metrics["devices"])
c5.metric("Threat Classes", len(r["clf"].classes_))

st.divider()

tabs = st.tabs([
    "📊 Overview",
    "🚨 Threat Detection",
    "🔎 Anomalies",
    "💻 Device Risk",
    "📈 Model Insights",
    "📋 Project Report"
])

with tabs[0]:
    st.subheader("Network Dataset")
    st.dataframe(r["df"].head(100), use_container_width=True)

    st.subheader("Traffic by Threat Category")
    counts = r["df"]["label"].value_counts()
    st.bar_chart(counts)

with tabs[1]:
    st.subheader("Known Threat Classification")

    st.write("Classes:", ", ".join(r["clf"].classes_))

    from sklearn.metrics import confusion_matrix, classification_report

    st.code(r["report"], language="text")

    fig, ax = plt.subplots(figsize=(7, 5))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(
        r["cm"], display_labels=r["clf"].classes_
    ).plot(ax=ax)
    ax.set_title("Threat Classification Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    st.subheader("Unsupervised Anomaly Detection")

    anomaly_df = r["df"].copy()
    anomaly_df["anomaly_score"] = r["anomaly_scores"]
    anomaly_df["anomaly"] = np.where(r["anomaly_flags"], "Suspicious", "Normal")

    st.dataframe(
        anomaly_df[
            ["device_id", "label", "anomaly_score", "anomaly",
             "failed_connections", "unique_destinations"]
        ].sort_values("anomaly_score", ascending=False).head(100),
        use_container_width=True
    )

    st.warning(
        "An anomaly is an indicator of unusual behavior, not proof that a device is compromised."
    )

with tabs[3]:
    st.subheader("💻 Device Risk Ranking")

    risk = r["risk"]
    st.dataframe(risk, use_container_width=True)

    st.subheader("Top 15 Device Risk Scores")
    top = risk.head(15).sort_values("risk_score")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["device_id"], top["risk_score"])
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_xlim(0, 100)
    st.pyplot(fig)
    plt.close(fig)

    selected = st.selectbox("Inspect device", risk["device_id"].tolist())
    row = risk[risk["device_id"] == selected].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Risk Score", f"{row['risk_score']:.1f}")
    b.metric("Risk Level", str(row["risk_level"]))
    c.metric("Anomaly Rate", f"{row['anomaly_rate']:.1%}")
    d.metric("Flows", int(row["flows"]))

with tabs[4]:
    st.subheader("Feature Importance")
    fi = pd.DataFrame({
        "Feature": r["cols"],
        "Importance": r["clf"].feature_importances_
    }).sort_values("Importance", ascending=False)
    st.dataframe(fi, use_container_width=True)
    st.bar_chart(fi.set_index("Feature"))

with tabs[5]:
    st.subheader("📋 Project Report")

    st.markdown("""
### 1. Problem Statement
Traditional binary intrusion detection may fail to distinguish different threat
types or identify previously unseen behavior. This project combines supervised
and unsupervised machine learning with device-level behavioral analysis.

### 2. Machine Learning Approach
**Random Forest:** classifies known network threat categories such as benign,
scan, brute-force, and DoS-like traffic.

**Isolation Forest:** identifies unusual observations without requiring a threat label.

### 3. Behavioral Intelligence
Flow-level signals are aggregated by device. The system considers anomaly rate,
anomaly intensity, failed connections, and destination diversity.

### 4. Risk Scoring
The dashboard calculates a transparent 0–100 device risk score from normalized
behavioral signals. Risk levels are categorized as low, medium, high, or critical.

### 5. Business / Security Value
- Prioritizes security investigation.
- Reduces the need to manually inspect every network flow.
- Helps analysts identify unusual devices.
- Separates known threat classification from novel-anomaly detection.
- Provides explainable feature importance and risk factors.

### 6. Important Limitation
The included demo data is synthetic. Production use requires validated enterprise
telemetry, threshold calibration, time-based validation, drift monitoring, and
human investigation.
""")

st.divider()
st.caption("AI Network Threat Detection • Defensive security analytics • Internship Project")
