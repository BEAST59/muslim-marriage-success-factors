## Model Performance (5-fold Cross Validation)

| Metric     | RandomForest (mean ± std) | Logistic Regression (mean ± std) |
|-----------:|:--------------------------:|:---------------------------------:|
| Accuracy   | 0.976 ± 0.025             | 0.971 ± 0.029                    |
| Precision  | 1.000 ± 0.000             | 0.988 ± 0.026                    |
| Recall     | 0.951 ± 0.052             | 0.951 ± 0.052                    |
| F1-score   | 0.975 ± 0.028             | 0.969 ± 0.031                    |
| ROC AUC    | 0.999 ± 0.002             | 0.994 ± 0.012                    |

*Dataset: Kaggle Divorce Predictors (54 questions, 170 samples). 5-fold Stratified CV; RF = 300 trees; LR = liblinear, max_iter=1000.*
