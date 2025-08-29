import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# helper function: run cross-validation for a model and return mean/std as a table
def cv_table(model, X, y, skf, name):
    # run 5-fold CV with multiple metrics
    scores = cross_validate(
        model, X, y, cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False   # we only care about test folds, not train
    )
    # convert to DataFrame, compute mean & std for each metric
    out = pd.DataFrame(scores).agg(['mean', 'std']).T
    # give the columns a prefix (so later we can join RF + LR together)
    out.columns = pd.MultiIndex.from_product([[name], out.columns])
    return out

def main(args):
    # load dataset (semicolon separated), clean header names
    df = pd.read_csv(args.data, sep=";")
    df.columns = df.columns.str.strip()

    # separate features (X) and target (y)
    X = df.select_dtypes(include=['number']).drop(columns=[args.target], errors='ignore')
    y = df[args.target].astype(int)

    # define cross-validation strategy: 5-fold, stratified (keeps class balance)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # model 1: RandomForest (strong, non-linear, ensemble)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

    # model 2: Logistic Regression (simple linear baseline)
    lr = LogisticRegression(max_iter=1000, solver='liblinear')

    # run CV for both
    rf_tab = cv_table(rf, X, y, skf, "RandomForest")
    lr_tab = cv_table(lr, X, y, skf, "LogReg")

    # join side-by-side for easy comparison
    combined = rf_tab.join(lr_tab)
    print(combined)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV")
    parser.add_argument("--target", default="Divorce")  # default is Divorce column
    args = parser.parse_args()
    main(args)
