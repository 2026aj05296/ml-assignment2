"""
Credit Default Risk Lab - Streamlit app for ML Assignment 2
Loads the 5 pre-trained pipelines from model/ and evaluates them on an uploaded
test CSV. No retraining happens here.
"""
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET_ALIASES = ["default.payment.next.month", "default payment next month", "Y"]

st.set_page_config(page_title="Credit Default Risk Lab", page_icon="💳", layout="wide")


# ---------- loaders ----------
@st.cache_resource
def load_models():
    return {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}


@st.cache_data
def load_schema():
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        return json.load(f)


def score_vector(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    scores = score_vector(model, X)
    auc = roc_auc_score(y, scores) if y.nunique() == 2 else np.nan
    return y_pred, scores, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


@st.cache_data(show_spinner="Scoring all five models...")
def compare_all(file_bytes):  # file bytes = cache key, so a new upload re-scores
    rows = []
    for n, mdl in models.items():
        _, _, mm = compute_metrics(mdl, X, y)
        rows.append({"ML Model Name": n, **mm})
    return pd.DataFrame(rows).set_index("ML Model Name").round(4)


def prepare(df, schema):
    df = df.rename(columns={a: schema["target"] for a in TARGET_ALIASES if a in df.columns})
    df = df.rename(columns={"PAY_1": "PAY_0"})
    missing = [c for c in schema["features"] + [schema["target"]] if c not in df.columns]
    if missing:
        return None, None, missing
    df = df.dropna(subset=schema["features"] + [schema["target"]])
    return df[schema["features"]], df[schema["target"]].astype(int), []


# ---------- UI ----------
st.title("💳 Credit Default Risk Lab")
st.caption("Five classifiers trained on the UCI *Default of Credit Card Clients* dataset — "
           "evaluated live on the test data you upload. Models are pre-trained; nothing is refit here.")

schema = load_schema()
models = load_models()

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type=["csv"])
    sample_path = os.path.join(ROOT, "test_data.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button("⬇ Download sample test_data.csv", f, "test_data.csv", "text/csv")
    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))
    st.markdown("---")
    st.caption(f"Target column: `{schema['target']}` (1 = default)")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. Use the sample file if you don't have one.")
    with st.expander("Expected columns"):
        st.write(schema["features"] + [schema["target"]])
    st.stop()

raw = pd.read_csv(uploaded)
X, y, missing = prepare(raw, schema)
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(X):,}")
c2.metric("Features", X.shape[1])
c3.metric("Default rate", f"{y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20))

tab1, tab2 = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

with tab1:
    model = models[model_name]
    y_pred, scores, m = compute_metrics(model, X, y)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No default", "Default"],
                    yticklabels=["No default", "Default"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("ROC curve")
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, scores)
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
            ax2.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.3f}")
            ax2.plot([0, 1], [0, 1], "--", color="grey")
            ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
            ax2.legend(loc="lower right")
            st.pyplot(fig2)
        else:
            st.warning("ROC needs both classes in the uploaded data.")

    st.subheader("Classification report")
    rep = classification_report(y, y_pred, labels=[0, 1],
                                target_names=["No default (0)", "Default (1)"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4))

    out = raw.loc[X.index].copy()
    out["predicted"] = y_pred
    st.download_button("⬇ Download predictions", out.to_csv(index=False),
                       f"predictions_{model_name.replace(' ', '_').lower()}.csv", "text/csv")

with tab2:
    table = compare_all(uploaded.getvalue() if hasattr(uploaded, "getvalue") else raw.to_csv().encode())
    st.dataframe(table.style.highlight_max(axis=0, color="#c6efce"))

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    sns.barplot(x=table.index, y=table[metric], ax=ax3, color="#4c78a8")
    ax3.set_xlabel(""); ax3.set_ylabel(metric)
    ax3.tick_params(axis="x", labelsize=8)
    st.pyplot(fig3)
    st.caption(f"Best on {metric}: **{table[metric].idxmax()}**")
