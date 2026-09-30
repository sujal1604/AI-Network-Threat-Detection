import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score

def save_confusion_matrix(cm, labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)
    ax.set_title("Threat Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_risk_plot(risk_df, path, top_n=15):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    top = risk_df.head(top_n).sort_values("risk_score")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["device_id"], top["risk_score"])
    ax.set_xlabel("Risk score")
    ax.set_title("Top Device Risk Scores")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def save_feature_importance(model, columns, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fi = pd.DataFrame({
        "feature": columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(path, index=False)
    return fi
