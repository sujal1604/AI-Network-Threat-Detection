import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

def train_classifier(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=250, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    return model, pred, report, cm

def train_anomaly_model(X_train, X_all):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xa = scaler.transform(X_all)
    model = IsolationForest(
        n_estimators=250, contamination=0.10,
        random_state=42, n_jobs=-1
    )
    model.fit(Xtr)
    scores = -model.score_samples(Xa)
    flags = model.predict(Xa) == -1
    return model, scaler, scores, flags

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
