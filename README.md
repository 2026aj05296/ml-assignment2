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
| Instances | 30,000 (Taiwan, April–September 2005) |
| Features | 23 (ID column dropped) |
| Target | `default_next_month` — 1 = default, 0 = no default |
| Class balance | ~22% default / ~78% no default (imbalanced) |

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
- Stratified 80/20 train/test split (`random_state=42`) → 24,000 train / 6,000 test. The test split is saved as `test_data.csv`.
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

### Comparison table (test set, n = 6,000)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

*Positive class = default (1). Precision, Recall and F1 are for the default class.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | |

## Streamlit app features

- CSV upload of test data (sample `test_data.csv` downloadable from the sidebar)
- Model selection dropdown (5 models)
- Six evaluation metrics for the selected model
- Confusion matrix, ROC curve and classification report
- "Compare all models" tab with a highlighted metrics table and bar chart
- Download of row-level predictions
