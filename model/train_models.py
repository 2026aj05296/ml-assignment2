"""
ML Assignment 2 - Training script
Dataset : Default of Credit Card Clients (UCI ML Repository, id=350)
          Same data is on Kaggle as "UCI_Credit_Card.csv"
Task    : Binary classification -> will the client default next month? (1 = yes)

Run from the project root:
    python model/train_models.py                       # auto-download via ucimlrepo
    python model/train_models.py --csv data/UCI_Credit_Card.csv   # local Kaggle/UCI csv

Outputs:
    model/<name>.joblib      -> 5 trained pipelines (preprocessing + classifier)
    model/feature_columns.json
    model/metrics.csv        -> evaluation metrics on the held-out test split
    test_data.csv            -> held-out test split (features + target) for the Streamlit app
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "default_next_month"

# Canonical feature names (UCI order X1..X23)
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_ALIASES = ["default.payment.next.month", "default payment next month", "Y", TARGET]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
        ds = fetch_ucirepo(id=350)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)

    # UCI API returns X1..X23 / Y -> rename to meaningful names
    if "X1" in df.columns:
        df = df.rename(columns={f"X{i+1}": c for i, c in enumerate(FEATURES)})
    df = df.rename(columns={a: TARGET for a in TARGET_ALIASES if a in df.columns})
    df = df.rename(columns={"PAY_1": "PAY_0"})
    return df[FEATURES + [TARGET]].copy()


def clean(df):
    """Fold undocumented category codes into the 'others' bucket (per UCI data notes)."""
    df = df.dropna().drop_duplicates()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})  # 1 grad,2 univ,3 high school,4 others
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # 1 married,2 single,3 others
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_models():
    """Scale-sensitive models get StandardScaler inside the pipeline (fit on train only)."""
    scaled = lambda clf: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=RANDOM_STATE))]),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=25, weights="distance")),
        "Naive Bayes": scaled(GaussianNB()),
        "Support Vector Machine": scaled(SVC(kernel="rbf", C=1.0, gamma="scale",
                                             random_state=RANDOM_STATE)),
    }


def score_vector(model, X):
    """Positive-class score for AUC: predict_proba if available, else SVM decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, score_vector(model, X)),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def slug(name):
    return name.lower().replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to UCI_Credit_Card.csv (optional)")
    args = ap.parse_args()

    df = clean(load_data(args.csv))
    print(f"Dataset shape: {df.shape} | features: {len(FEATURES)}")
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3))

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    rows = []
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **m})
        joblib.dump(model, os.path.join(MODEL_DIR, f"{slug(name)}.joblib"))
        print(f"{name:<25} done in {time.time()-t0:5.1f}s | " +
              " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump({"features": FEATURES, "target": TARGET}, f, indent=2)

    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    print("\nComparison table (paste into README):\n")
    cols = list(metrics.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in metrics.iterrows():
        print("| " + " | ".join([r[cols[0]]] + [f"{r[c]:.4f}" for c in cols[1:]]) + " |")


if __name__ == "__main__":
    main()
