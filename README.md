# Identifying Factors Affecting Muslim Marriage Success (Data Science Project)

This repository contains an end-to-end data science project exploring factors associated with Muslim marriage success. It includes exploratory data analysis (EDA), classical machine learning models, and a lightweight Streamlit app for interactive visualization and demonstration.

> **Ethics**: This project is for educational purposes. Do not use any model here for real-world counseling or high-stakes decisions. Always anonymize personal data and follow your institution's research ethics guidelines.

## Repository Structure

```
.
├── app/                      # Streamlit app
│   └── app.py
├── data/
│   └── README.md             # Put data files here (not committed)
├── models/
│   └── README.md             # Trained model artifacts (not committed)
├── notebooks/                # Your analysis notebooks
├── reports/                  # Exported charts, figures
├── scripts/
│   ├── train_model.py        # Trains a baseline RandomForest and saves model.pkl
│   └── evaluate_model.py     # Reproducible evaluation with cross-validation
├── src/                      # Reusable utilities
│   └── utils.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Quickstart (Local)

1. **Create environment & install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add data**
   - Place `divorce_data.csv` (Kaggle DPS) into `data/`.
   - (Optional) Place your **primary** survey CSV into `data/` but **do not commit** it.

3. **Train a baseline model**
   ```bash
   python scripts/train_model.py --data data/divorce_data.csv --target Divorce --out models/model.pkl
   ```

4. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

## Result

We evaluated two baselines with 5-fold Stratified CV on the Kaggle Divorce Predictors dataset:

- **RandomForest (300 trees)** performs slightly better overall than **Logistic Regression**.
- Both models achieve very high ROC AUC (≥ 0.99), which is expected on this dataset.

See the full table here → [`reports/metrics_summary.md`](reports/metrics_summary.md).

## Interpretability (Permutation Importance)

We computed **permutation importance** on a held-out test set to see which questions influence predictions most for the RandomForest model. The chart below shows the top features by the average decrease in ROC AUC when each feature is randomly shuffled (higher = more influential).

![Top features](reports/feature_importance.png)

> Notes:
> - Importance is measured relative to this dataset/model — not causal.
> - Correlated questions can share importance (shuffling one may be partly “covered” by another).
> - This project is for **educational** use; do not use for counseling decisions.

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public or private).
2. On [Streamlit Community Cloud], create a new app from your repo, set main file to `app/app.py`.
3. Add `data/divorce_data.csv` via Secrets or upload at runtime from the app.

## Notes

- The app supports two modes: **EDA** (load CSV and explore) and **Predict** (uses a saved model + feature template).
- Trained artifacts (`models/`) and raw data (`data/`) are `.gitignore`'d by default.
- See `scripts/evaluate_model.py` for reproducible CV and metrics export.