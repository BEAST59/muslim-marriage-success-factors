import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from joblib import dump

from src.utils import save_feature_list

def main(args):
    df = pd.read_csv(args.data, sep=";")
    df.columns = df.columns.str.strip()  # Clean column names
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")
    # Basic cleaning: drop rows with NA in target or features
    df = df.dropna(subset=[args.target])
    # Use only numeric features for baseline
    X = df.select_dtypes(include=['number']).drop(columns=[args.target], errors='ignore')
    y = df[args.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")

    print("Accuracy:", round(acc, 4))
    print("AUC:", round(auc, 4))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Save model & features
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out)
    save_feature_list(X.columns, out.with_suffix(".features.json"))
    print(f"Saved model to {out}")
    print(f"Saved feature list to {out.with_suffix('.features.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV (e.g., data/divorce_data.csv)")
    parser.add_argument("--target", default="Divorce", help="Target column name")
    parser.add_argument("--out", default="models/model.pkl", help="Output model path")
    args = parser.parse_args()
    main(args)