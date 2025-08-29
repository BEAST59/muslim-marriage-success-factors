import io
import re
import json
import random
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import load

# ──────────────────────────────────────────────────────────────────────────────
# Page & light styling
st.set_page_config(page_title="Marriage Success: EDA & Demo", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stDataFrame"] div {font-size: 0.95rem;}
h1, h2, h3 { margin-top: 0.2rem; }
hr { border: 0; height: 1px; background: #eaeaea; }
</style>
""", unsafe_allow_html=True)

def hr(): st.markdown("<hr/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Friendly labels for Q1..Q54 / Divorce (optional)
LABELS_PATH = Path("data/column_labels.json")

def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {}

LABELS = load_labels()

def label_for(col: str) -> str:
    """Human-readable label for a column, falling back to the raw name."""
    return LABELS.get(col, col)

# ──────────────────────────────────────────────────────────────────────────────
# CV renderer: KPI cards + compact styled table (robust UTF-16/UTF-8 & TXT/CSV)
def render_cv_table_from_txt(path: Path):
    """
    Prefer a CSV next to the TXT (cv_rf_vs_lr.csv). Otherwise parse the TXT
    with regex (handles UTF-16/UTF-8 + irregular whitespace). Then render:
      1) KPI cards for Accuracy / F1 / ROC AUC
      2) Styled table with best model highlighted per metric
    """
    # 1) Try CSV first
    csv_path = path.with_suffix(".csv")
    df = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip().lower() for c in df.columns]
            df = df.rename(columns={
                "metric":"metric", "rf_mean":"rf_mean", "rf_std":"rf_std",
                "lr_mean":"lr_mean", "lr_std":"lr_std"
            })
        except Exception:
            df = None

    # 2) Otherwise parse the TXT
    if df is None:
        try:
            raw = path.read_text(encoding="utf-16")
        except UnicodeError:
            raw = path.read_text(encoding="utf-8")

        lines = raw.strip().splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "RandomForest" in line and "LogReg" in line:
                header_idx = i
                break
        data_lines = lines[header_idx+1:] if header_idx is not None else lines

        rows = []
        rx = re.compile(r"""
            ^\s*
            (?P<metric>[A-Za-z0-9_\.]+)
            \s+(?P<rf_mean>[-+]?\d*\.\d+|\d+)
            \s+(?P<rf_std>[-+]?\d*\.\d+|\d+)
            \s+(?P<lr_mean>[-+]?\d*\.\d+|\d+)
            \s+(?P<lr_std>[-+]?\d*\.\d+|\d+)
            \s*$
        """, re.VERBOSE)
        for line in data_lines:
            m = rx.match(line)
            if m:
                rows.append(m.groupdict())

        if not rows:
            st.text(raw)
            return

        df = pd.DataFrame(rows)
        for c in ["rf_mean","rf_std","lr_mean","lr_std"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["metric"] = df["metric"].str.replace("_"," ").str.strip()

    # KPI cards
    label_map = {
        "test accuracy": "Accuracy",
        "test f1": "F1",
        "test roc auc": "ROC AUC",
    }
    df_lc = df.copy()
    df_lc["metric_lc"] = df_lc["metric"].str.lower()
    cards = df_lc[df_lc["metric_lc"].isin(label_map.keys())].copy()
    cards["display"] = cards["metric_lc"].map(label_map)

    if not cards.empty:
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1,c2,c3], cards.iterrows()):
            rf, lr = row["rf_mean"], row["lr_mean"]
            delta = rf - lr
            with col:
                st.markdown(f"##### {row['display']}")
                st.metric("RandomForest (mean)", f"{rf:.3f}", delta=f"{delta:+.3f} vs LogReg")
                st.caption(f"LogReg {lr:.3f} • RF std {row['rf_std']:.3f} • LR std {row['lr_std']:.3f}")

    st.markdown("---")

    # Compact styled table
    lower_better = {"fit time", "score time"}
    def row_hi(s):
        metric = s.name.lower()
        rf, lr = s["rf_mean"], s["lr_mean"]
        choose_rf = (rf <= lr) if metric in lower_better else (rf >= lr)
        out = []
        for c in s.index:
            if c in ("rf_mean","lr_mean"):
                best = (choose_rf and c == "rf_mean") or ((not choose_rf) and c == "lr_mean")
                out.append("background-color:#1f6feb20;font-weight:600;" if best else "")
            else:
                out.append("")
        return out

    styled = (
        df.set_index("metric")
          .rename_axis(None)
          .style.format("{:.3f}")
          .apply(row_hi, axis=1)
    )
    st.dataframe(styled, use_container_width=True, height=260)

# ──────────────────────────────────────────────────────────────────────────────
# App
st.title("Marriage Success — EDA & Demo App")
st.caption("Educational demo. Do not use for real counseling/clinical decisions.")

tab1, tab2, tab3 = st.tabs(["Load & Explore Data (EDA)", "Predict (Demo)", "About"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: EDA
with tab1:
    with st.sidebar:
        st.header("Controls")
        data_file = st.file_uploader("Upload CSV", type=["csv"])
        sep_choice = st.selectbox("CSV separator", ["auto", ",", ";"], index=0,
                                  help="Use ';' for Kaggle Divorce Predictors; 'auto' guesses.")
        st.caption("Pick the target column after the CSV loads.")

    if not data_file:
        st.info("Upload a CSV to start exploring (primary survey or Kaggle Divorce Predictors).")
    else:
        raw = data_file.getvalue().decode("utf-8", errors="ignore")
        sep_detected = ";" if (sep_choice == "auto" and raw.count(";") > raw.count(",")) else (sep_choice if sep_choice != "auto" else ",")
        df = pd.read_csv(StringIO(raw), sep=sep_detected)
        df.columns = df.columns.str.strip()

        # Privacy: drop timestamp-like columns
        drop_cols = [c for c in df.columns if c.lower().strip() in {"timestamp", "time", "date"}]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns (sep='{sep_detected}')")
        st.dataframe(df.head(100), height=220, use_container_width=True)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols     = df.select_dtypes(exclude=["number"]).columns.tolist()

        with st.sidebar:
            target = st.selectbox("Target column (optional)", options=["(none)"] + list(df.columns), index=0)

        # KPI strip
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Rows", f"{df.shape[0]:,}")
        with c2: st.metric("Columns", f"{df.shape[1]:,}")
        with c3:
            if target != "(none)" and target in df.columns:
                if pd.api.types.is_numeric_dtype(df[target]):
                    rate = float(pd.to_numeric(df[target], errors="coerce").mean())
                else:
                    rate = float(df[target].value_counts(normalize=True, dropna=False).max())
                st.metric("Largest Class Rate", f"{rate:.2%}")
            else:
                st.metric("Largest Class Rate", "—")

        # Sub-tabs
        t_over, t_dist, t_rel = st.tabs(["Overview", "Distributions", "Relationships"])

        # Overview
        with t_over:
            if target != "(none)" and target in df.columns:
                st.subheader("Class Balance")
                counts = df[target].value_counts(dropna=False)
                fig, ax = plt.subplots(figsize=(4.8, 3.2))
                counts.plot(kind="bar", ax=ax)
                ax.set_xlabel(label_for(target)); ax.set_ylabel("Count"); ax.set_title("Class Balance")
                st.pyplot(fig, use_container_width=True)
                st.caption(
                    f"Majority: {int(counts.max())} | Minority: {int(counts.min())}. "
                    "Large imbalance makes Accuracy less informative—prefer ROC AUC / F1."
                )
                hr()

            st.subheader("Summary Statistics")
            st.dataframe(df.describe(include='all').T, height=260, use_container_width=True)
            hr()

            if len(numeric_cols) > 1:
                st.subheader("Top Correlated Numeric Pairs")
                corr = df[numeric_cols].corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                pairs = (
                    upper.stack()
                    .reset_index()
                    .rename(columns={"level_0":"Feature A", "level_1":"Feature B", 0:"|corr|"})
                    .sort_values("|corr|", ascending=False)
                )
                top_k = st.slider("Show top pairs", min_value=5, max_value=30, value=10, step=1)
                st.dataframe(pairs.head(top_k), use_container_width=True, height=220)
                st.caption("High |corr| indicates similar information; redundant features may be simplified.")
            else:
                st.info("Not enough numeric columns for correlation pairs.")

        # Distributions
        with t_dist:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Histogram (numeric)")
                if numeric_cols:
                    sel_num = st.selectbox("Numeric feature", options=numeric_cols,
                                           key="hist_sel", format_func=label_for)
                    hfig, hax = plt.subplots(figsize=(4.8, 3.2))
                    hax.hist(df[sel_num].dropna().values, bins=20)
                    hax.set_title(f"{label_for(sel_num)}")
                    st.pyplot(hfig, use_container_width=True)
                else:
                    st.info("No numeric columns detected.")
            with col2:
                st.subheader("Bar (categorical)")
                if cat_cols:
                    sel_cat = st.selectbox("Categorical feature", options=cat_cols,
                                           key="bar_sel", format_func=label_for)
                    counts = df[sel_cat].astype(str).value_counts(dropna=False).head(15)
                    bfig, bax = plt.subplots(figsize=(4.8, 3.2))
                    counts.plot(kind="bar", ax=bax)
                    bax.set_title(f"Top values of {label_for(sel_cat)}")
                    st.pyplot(bfig, use_container_width=True)
                else:
                    st.info("No categorical columns detected.")

        # Relationships
        with t_rel:
            st.subheader("Target Relationships")

            if target != "(none)" and target in df.columns and numeric_cols:
                sel = st.selectbox("Numeric feature for grouped mean", options=numeric_cols,
                                   key="group_sel", format_func=label_for)
                gfig, gax = plt.subplots(figsize=(4.8, 3.2))
                try:
                    df.groupby(target)[sel].mean().plot(kind="bar", ax=gax)
                    gax.set_title(f"Mean {label_for(sel)} by {label_for(target)}")
                    st.pyplot(gfig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute grouped mean: {e}")

                try:
                    selb = st.selectbox("Boxplot numeric feature", options=numeric_cols,
                                        key="box_sel", format_func=label_for)
                    bpfig, bpax = plt.subplots(figsize=(5.6, 3.6))
                    sns.boxplot(x=df[target].astype(str), y=df[selb], ax=bpax)
                    bpax.set_xlabel(label_for(target)); bpax.set_title(f"{label_for(selb)} by {label_for(target)}")
                    st.pyplot(bpfig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not draw boxplot: {e}")
            else:
                st.info("Select a target and at least one numeric feature to see relationships.")

            if len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap (numeric features)")
                cfig, cax = plt.subplots(figsize=(8, 5))
                sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, ax=cax)
                cax.set_title("Correlation among numeric features")
                st.pyplot(cfig, use_container_width=True)
                st.caption("Near 1.0 = move together, near -1.0 = opposite, near 0 = weak linear relation.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Predict (Manual + Batch) + Model Context
with tab2:
    st.subheader("Use a saved model (baseline RandomForest)")
    model_path = Path("models/model.pkl")
    features_path = Path("models/model.features.json")

    if not model_path.exists() or not features_path.exists():
        st.warning("No saved model found. Train one with `python -m scripts.train_model --data data\\divorce_data.csv --target Divorce`.")
    else:
        clf = load(model_path)
        feature_names = pd.read_json(features_path).values.squeeze().tolist()
        st.write(f"Loaded model with **{len(feature_names)}** numeric features (DPS 0–4 scale).")

        mode = st.radio("Mode", ["Manual", "Batch Predict (CSV)"], horizontal=True)
        threshold = st.slider("Decision threshold for class 1", 0.0, 1.0, 0.50, 0.01,
                              help="If predicted probability ≥ threshold → class 1; else class 0.")

        # Manual mode
        if mode == "Manual":
            if "manual_inputs" not in st.session_state:
                st.session_state.manual_inputs = {f: 0.0 for f in feature_names}

            st.markdown("### Manual Inputs")
            with st.expander("Global Controls", expanded=True):
                cols = st.columns([1,1,1,1,2])
                with cols[0]:
                    if st.button("Reset"):
                        st.session_state.manual_inputs = {f: 0.0 for f in feature_names}
                with cols[1]:
                    if st.button("Set-All = 2"):
                        st.session_state.manual_inputs = {f: 2.0 for f in feature_names}
                with cols[2]:
                    if st.button("Randomize"):
                        st.session_state.manual_inputs = {f: float(random.randint(0,4)) for f in feature_names}
                with cols[3]:
                    if st.button("Fill from Mean"):
                        st.info("Tip: using mid-point 2.0 as a proxy.")
                        st.session_state.manual_inputs = {f: 2.0 for f in feature_names}
                with cols[4]:
                    search = st.text_input("Search features (Q# or text)", placeholder="Type to filter…")

            # Search by id or label
            query = search.strip().lower() if search else ""
            def matches(f: str) -> bool:
                return (query in f.lower()) or (query in label_for(f).lower())
            shown_features = [f for f in feature_names if matches(f)] if query else feature_names

            # Group inputs in expanders of 10
            groups = [shown_features[i:i+10] for i in range(0, len(shown_features), 10)]
            for gi, group in enumerate(groups, start=1):
                with st.expander(f"Inputs {group[0]} – {group[-1]}", expanded=(gi == 1 and not query)):
                    c1, c2, c3 = st.columns(3)
                    cols = [c1, c2, c3]
                    for idx, feat in enumerate(group):
                        with cols[idx % 3]:
                            st.session_state.manual_inputs[feat] = st.number_input(
                                feat,
                                value=float(st.session_state.manual_inputs[feat]),
                                min_value=0.0, max_value=4.0, step=1.0, key=f"num_{feat}",
                                help=label_for(feat)  # tooltip with long question
                            )

            X = np.array([st.session_state.manual_inputs[f] for f in feature_names], dtype=float).reshape(1, -1)
            proba = clf.predict_proba(X)[0, 1]
            pred = int(proba >= threshold)

            k1, k2 = st.columns(2)
            with k1: st.metric("Predicted Probability (class 1)", f"{proba:.3f}")
            with k2: st.metric("Predicted class", f"{pred}")
            st.caption("Educational demo. Not for individual decision-making.")

        # Batch mode
        else:
            st.markdown("### Batch Predict from CSV")
            st.write("Upload a CSV with columns matching the model features (e.g., Q1..Q54).")
            up = st.file_uploader("Upload feature matrix CSV", type=["csv"], key="batch_uploader")
            if up is not None:
                df_in = pd.read_csv(up)
                df_in.columns = df_in.columns.str.strip()
                st.write("**Preview (first 5 rows):**")
                st.dataframe(df_in.head(), use_container_width=True)

                # Align to expected columns
                missing = [c for c in feature_names if c not in df_in.columns]
                extra   = [c for c in df_in.columns if c not in feature_names]
                if missing:
                    st.warning(f"Missing columns (filled with 0): {missing[:10]}{' …' if len(missing)>10 else ''}")
                    for m in missing: df_in[m] = 0.0
                if extra:
                    st.info(f"Ignoring extra columns: {extra[:10]}{' …' if len(extra)>10 else ''}")

                df_in = df_in[feature_names]
                proba = clf.predict_proba(df_in.values)[:, 1]
                pred = (proba >= threshold).astype(int)

                out = df_in.copy()
                out["pred_proba_class1"] = proba
                out["pred_class"] = pred

                st.success(f"Predicted {len(out)} rows.")
                st.dataframe(out.head(), use_container_width=True)

                buf = io.StringIO()
                out.to_csv(buf, index=False)
                st.download_button("Download predictions CSV", data=buf.getvalue(),
                                   file_name="predictions.csv", mime="text/csv")

        hr()
        st.subheader("Model Context")
        cv_path = Path("reports/cv_rf_vs_lr.txt")
        if cv_path.exists():
            st.markdown("**Cross-Validation (RF vs Logistic Regression)**")
            render_cv_table_from_txt(cv_path)
        else:
            st.info("CV results not found. Run: `python -m scripts.evaluate_model --data data\\divorce_data.csv --target Divorce | "
                    "Out-File -FilePath reports\\cv_rf_vs_lr.txt -Encoding utf8`")

        imp_png = Path("reports/feature_importance.png")
        if imp_png.exists():
            st.markdown("**Permutation Importance (Top Features)**")
            st.image(str(imp_png))
            st.caption("Higher bars = shuffling that feature hurts ROC AUC more → more influential.")
        else:
            st.info("Feature importance plot not found. Run: `python -m scripts.feature_importance --data data\\divorce_data.csv --target Divorce --topn 15`")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: About
with tab3:
    st.markdown("""
### About this app
- **Goal:** Explore factors related to marriage success/divorce risk (educational demo).
- **Data:** (1) Kaggle Divorce Predictors (54 questions, values 0–4), (2) Primary survey.
- **Models:** Logistic Regression (baseline) and RandomForest (stronger).
- **Results:** High AUC on DPS; RF typically outperforms LR.
- **Interpretability:** Permutation importance highlights most influential questions.
- **Privacy:** Timestamp-like columns are removed from EDA previews automatically.
- **Ethics:** Do **not** use individual predictions for counseling or high-stakes decisions.
""")

    manual_path = Path("docs/USER_MANUAL.md")
    if manual_path.exists():
        st.download_button("Download User Manual (Markdown)",
                           data=manual_path.read_text(encoding="utf-8"),
                           file_name="USER_MANUAL.md", mime="text/markdown")

    with st.expander("Dataset Columns (Q1–Q54)"):
        # list all labels if present
        if LABELS:
            # Show Q1..Q54 in order (if keys look like Q#)
            qs = [k for k in LABELS.keys() if k.startswith("Q")]
            try:
                qs_sorted = sorted(qs, key=lambda x: int(x[1:]))
            except Exception:
                qs_sorted = sorted(qs)
            for col in qs_sorted:
                st.markdown(f"- **{col}** — {LABELS[col]}")
            if "Divorce" in LABELS:
                st.markdown(f"- **Divorce** — {LABELS['Divorce']}")
        else:
            st.info("Add human-readable labels at `data/column_labels.json` to see the question text here.")
