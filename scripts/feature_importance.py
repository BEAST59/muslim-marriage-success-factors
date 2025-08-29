# scripts/feature_importance.py
# Purpose: compute permutation importance for RandomForest on the DPS dataset
# Output: reports/feature_importance.csv + reports/feature_importance.png

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

def main(args):
    # 1) Load data (semicolon-separated) and clean headers
    df = pd.read_csv(args.data, sep=";")
    df.columns = df.columns.str.strip()

    # 2) Split features/target
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found. Available: {df.columns.tolist()[:10]} ...")

    X = df.select_dtypes(include=["number"]).drop(columns=[args.target], errors="ignore")
    y = df[args.target].astype(int)

    # 3) Train/test split (same seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Train a baseline RF (same as earlier)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Quick sanity: AUC on test set
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"Test ROC AUC: {auc:.4f}")

    # 5) Permutation importance on the test set
    perf = permutation_importance(
        clf, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=42, n_jobs=-1
    )
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perf.importances_mean,
        "importance_std": perf.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # 6) Save CSV
    out_csv = Path("reports/feature_importance.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(out_csv, index=False)
    print(f"Saved CSV -> {out_csv}")

    # 7) Plot top-N features
    topn = args.topn
    top = importances.head(topn).iloc[::-1]  # reverse for horizontal bar chart

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("Permutation importance (mean AUC drop)")
    ax.set_title(f"Top {topn} Most Important Features (RandomForest)")
    fig.tight_layout()

    out_png = Path("reports/feature_importance.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV (semicolon-separated)")
    parser.add_argument("--target", default="Divorce", help="Target column name")
    parser.add_argument("--topn", type=int, default=15, help="How many top features to plot")
    args = parser.parse_args()
    main(args)
