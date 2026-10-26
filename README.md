# Credit Default Risk Lab — ML Assignment 2

**M.Tech (AIML), BITS Pilani WILP — Machine Learning**
Student: Sandip Chaudhari (2026AJ05296)

**Live app:** https://YOUR-APP-NAME.streamlit.app
**Repository:** https://github.com/2026aj05296/ml-assignment2

---

## a. Problem statement

Predict whether a credit-card holder will **default on next month's payment** (binary: 1 = default, 0 = no default) from their credit limit, demographics, last six months' repayment status, bill amounts and payment amounts. Five classifiers are trained on the same data, compared on six metrics, and served through an interactive Streamlit app that evaluates the saved models on uploaded test data without retraining.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Default of Credit Card Clients (Yeh & Lien, 2009) |
| Source | UCI Machine Learning Repository, dataset id 350 (also on Kaggle as `UCI_Credit_Card.csv`) |
| Instances | 29,965 used (30,000 raw, Taiwan, April–September 2005; 35 duplicate rows removed) |
| Features | 23 (ID column dropped) |
| Target | `default_next_month` — 1 = default, 0 = no default |
| Class balance | 22.1% default / 77.9% no default (imbalanced) |

**Features**

| Group | Columns | Meaning |
|---|---|---|
| Credit | `LIMIT_BAL` | Credit limit (NT$), individual + family |
| Demographic | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | 1/2 sex; 1 grad school, 2 university, 3 high school, 4 others; 1 married, 2 single, 3 others |
| Repayment status | `PAY_0`, `PAY_2` … `PAY_6` | Sep→Apr status; −2/−1 = paid duly, 1–9 = months of delay |
| Bill amount | `BILL_AMT1` … `BILL_AMT6` | Statement balance Sep→Apr |
| Payment amount | `PAY_AMT1` … `PAY_AMT6` | Amount paid Sep→Apr |

**Preprocessing**
- Undocumented codes folded into "others": `EDUCATION` 0/5/6 → 4, `MARRIAGE` 0 → 3.
- Missing rows / duplicates dropped.
- Stratified 80/20 train/test split (`random_state=42`) → 23,972 train / 5,993 test. The test split is saved as `test_data.csv`.
- `StandardScaler` is fitted on training data only, inside each model's `Pipeline` (Logistic Regression, KNN, Naive Bayes, SVM). The Decision Tree uses raw features since splits are scale-invariant.

## c. GitHub repository

```
ml-assignment2/
├── app.py                    # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv             # held-out test split (features + target)
└── model/
    ├── train_models.py       # data loading, preprocessing, training, evaluation
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── feature_columns.json  # feature list + target name used by the app
    └── metrics.csv           # test-set metrics for all models
```

Reproduce: `pip install -r requirements.txt ucimlrepo` → `python model/train_models.py` → `streamlit run app.py`

## d. Models used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler → LogisticRegression(max_iter=2000) |
| Decision Tree | DecisionTreeClassifier(max_depth=6, min_samples_leaf=50) |
| KNN | StandardScaler → KNeighborsClassifier(n_neighbors=25, weights="distance") |
| Naive Bayes | StandardScaler → GaussianNB() |
| Support Vector Machine | StandardScaler → SVC(kernel="rbf", C=1.0, gamma="scale"); AUC from `decision_function` |

### Comparison table (test set, n = 5,993)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8133 | 0.7160 | 0.7275 | 0.2496 | 0.3717 | 0.3496 |
| Decision Tree | **0.8196** | **0.7476** | 0.6623 | 0.3771 | 0.4805 | **0.4033** |
| KNN | 0.8086 | 0.7395 | 0.6330 | 0.3213 | 0.4262 | 0.3528 |
| Naive Bayes | 0.6915 | 0.7283 | 0.3864 | **0.6704** | **0.4902** | 0.3140 |
| Support Vector Machine | 0.8163 | 0.7173 | 0.6718 | 0.3318 | 0.4442 | 0.3802 |

*Positive class = default (1). Precision, Recall and F1 are for the default class. Best value per metric in bold.*

**Note on metrics:** 77.9% of test clients do not default, so a model that always predicts "no default" already scores 0.779 accuracy. Models are therefore compared mainly on **MCC, F1 and AUC**, which account for the imbalance.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.73) but lowest recall (0.25): it flags only about 1 in 4 actual defaulters. A linear boundary cannot capture the non-linear effect of repayment delays, so F1 (0.37) and AUC (0.72) are the lowest of all models. Its accuracy of 0.81 is only slightly above the 0.78 majority baseline. |
| Decision Tree | Best overall, with the highest Accuracy (0.82), AUC (0.75) and MCC (0.40). Threshold splits on repayment status (`PAY_0`) match how default risk behaves in this data, and limiting depth to 6 with at least 50 samples per leaf prevents overfitting. Recall (0.38) is still modest. |
| KNN | Mid-table (MCC 0.35). Its AUC of 0.74 is second best, so it ranks risky clients reasonably well, but at the default 0.5 cut-off it misses most defaulters (recall 0.32). Distances in 23 dimensions are noisy, and minority-class neighbours are outvoted. |
| Naive Bayes | Highest recall (0.67) and F1 (0.49), but the lowest accuracy (0.69) and precision (0.39), so it raises many false alarms. The six bill-amount and six payment features are strongly correlated, which violates its independence assumption. It has the lowest MCC (0.31). Useful only when missing a defaulter is far costlier than a false alarm. |
| Support Vector Machine | Second-best MCC (0.38) and accuracy (0.82), with good precision (0.67). It was by far the slowest to train (about 60 s versus under 2 s for the others). Like Logistic Regression it leans towards the majority class (recall 0.33), and its AUC (0.72) is relatively low. |
| **Overall winner** | **Decision Tree**: best MCC, AUC and accuracy, fast to train, and interpretable. All five models show low recall because of the 78:22 class imbalance; `class_weight="balanced"` or tuning the decision threshold would be the next improvement. |

## Streamlit app features

- CSV upload of test data (sample `test_data.csv` downloadable from the sidebar)
- Model selection dropdown (5 models)
- Six evaluation metrics for the selected model
- Confusion matrix, ROC curve and classification report
- "Compare all models" tab with a highlighted metrics table and bar chart
- Download of row-level predictions
