# User Manual

## Setup
1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `streamlit run app/app.py`

## Tabs
- **EDA**: Upload CSV (semicolon-separated for DPS). Explore overview, distributions, relationships.
- **Predict**: Manual entry (with tooltips + reset/randomize) or Batch prediction from CSV. Adjust decision threshold.
- **Model Context**: Cross-validation comparison (RF vs Logistic Regression) + feature importance chart.
- **About**: Project info, download manual, and column descriptions (Q1–Q54).

⚠️ *This app is educational only, not for real counseling or decisions.*
