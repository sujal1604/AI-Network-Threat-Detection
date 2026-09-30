# Project Report

## Title
AI-Powered Network Threat Detection & Behavioral Anomaly Intelligence

## Problem Statement
Traditional binary intrusion detection can miss novel or evolving behavior. This project combines supervised learning for known threat classes with unsupervised anomaly detection and device-level behavioral aggregation.

## Methodology
- Generate or replace demo data with authorized network-flow telemetry.
- Clean and engineer behavioral features.
- Train a Random Forest classifier for known categories.
- Train Isolation Forest to identify unusual behavior without labels.
- Aggregate flow-level evidence by device.
- Calculate a transparent 0–100 risk score.
- Evaluate classification quality and robustness to moderate feature noise.

## Models
### Random Forest
Used for multiclass classification of known traffic categories.

### Isolation Forest
Used for unsupervised anomaly detection. An anomaly flag is treated as one signal rather than proof of compromise.

## Risk Scoring
Risk is composed of normalized:
- mean anomaly score: 40%
- anomaly rate: 30%
- failed-connection volume: 20%
- destination diversity: 10%

This weighting is a project assumption and should be calibrated using an organization's historical incidents.

## Limitations
- The included dataset is synthetic and is not representative of a real enterprise network.
- Risk scores are indicators, not security verdicts.
- Production deployment requires validated telemetry, threshold tuning, monitoring, drift detection, and human review.
- A real deployment should evaluate false positives, false negatives, class imbalance, temporal leakage, and cross-network generalization.

## Future Enhancements
- Add temporal sequence models.
- Use graph-based device communication features.
- Add SHAP-based explanations.
- Perform time-based validation.
- Add drift monitoring.
- Integrate a SIEM through an approved interface.
