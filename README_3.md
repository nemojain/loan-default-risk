# 🏦 Loan Default Risk Model - Home Credit Data

> **Credit risk model** that predicts which loan applicants are likely to have repayment problems, trained on 307,511 real Home Credit loan applications (about 8% defaults) from Kaggle. Combines three linked tables, engineers repayment-burden ratios, trains a LightGBM model, and explains individual decisions with SHAP.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![pandas](https://img.shields.io/badge/pandas-Feature%20Engineering-150458)
![Data](https://img.shields.io/badge/Data-Kaggle%20Home%20Credit%20(Real)-20BEFF)

> 📓 Full walkthrough: [`loan_default_model.ipynb`](loan_default_model.ipynb)

---

## 📌 Baseline vs. Final Model

| | Baseline | Final |
|---|---|---|
| Model | Logistic regression | LightGBM (300 trees) |
| Source tables | 1 (main application) | 3 (application, credit bureau, previous applications) |
| Features | 121 raw columns | 135 (raw + aggregated + engineered ratios) |
| ROC AUC | 0.749 | **0.770** |
| AUPRC | 0.231 | **0.264** |
| Explainability | None | SHAP, overall and per applicant |
| Gender as an input | Yes | Removed |

Scores come from a held-out 20% validation split (stratified, `random_state=42`).

---

## 📈 Results at Each Step

| Model | ROC AUC | AUPRC |
|---|---|---|
| Logistic regression, main table only | 0.749 | 0.231 |
| LightGBM + credit bureau features | 0.763 | 0.255 |
| + previous application features | 0.766 | 0.259 |
| + ratio features (payment burden, loan vs. goods price) | 0.771 | 0.266 |
| Final model (gender removed) | 0.770 | 0.264 |

About 92% of applicants repay, so a model that approves everyone is 92% accurate and useless. Accuracy was not used. A model with no skill scores 0.5 on ROC AUC and about 0.081 (the default rate) on AUPRC.

---

## 🧠 What This Project Does

- 🧹 **Cleans real data:** found an impossible value in `DAYS_EMPLOYED` (365243 days, about 1,000 years) that marks applicants with no employment date. Replaced it with a missing value and kept a flag column, because that group defaults at a different rate (5.4% vs. 8.7%).
- 🔗 **Joins multiple tables:** aggregated the credit bureau and previous applications tables to one row per applicant with `groupby`, then merged them with left joins, checking row counts after each merge.
- 🧮 **Engineers features:** payment-to-income, repayment term, loan-to-goods price, and an averaged external credit score.
- 🌲 **Trains and compares models:** logistic regression baseline against LightGBM, on the same split.
- 🔍 **Explains decisions:** SHAP summary plot for overall drivers and a waterfall plot for a single high-risk applicant.
- ⚖️ **Checks a fairness concern:** `CODE_GENDER` ranked among the most influential features, so I removed it. ROC AUC changed by only 0.0015.

---

## 🔑 What the Model Relies On

- The average of the three external credit scores (`EXT_SOURCE_MEAN`) is the strongest driver. Low scores raise predicted risk.
- Two engineered ratios, `CREDIT_TERM` and `GOODS_CREDIT_RATIO`, are among the top features.
- Higher total debt at the credit bureau (`BUREAU_TOTAL_DEBT`) raises predicted risk.
- A history of refused applications (`PREV_SHARE_REFUSED`) raises predicted risk.

---

## 🎯 Choosing an Approval Threshold

The model outputs a default probability, and approving or denying needs a cutoff. I assumed a missed default costs 5 times as much as wrongly denying a good customer. Under that assumption, the best threshold is **0.16**, which denies 12.8% of applicants and cuts total cost by **18.6%** compared with approving everyone (the logistic baseline cut it by 14.9%).

The 5:1 ratio is an assumption for illustration. A real lender would supply its own figures, and the best threshold moves when the ratio changes.

---

## ⚠️ Limitations

- Only 3 of the 8 source tables are used.
- Model variants were compared on the same validation split, so final scores are slightly optimistic. Cross-validation or a separate test set would be more rigorous.
- Dropping gender does not remove features that may correlate with it. No group-level fairness audit was done.
- Scores come from my own validation split and are not Kaggle leaderboard scores.

---

## 🚀 Reproduce It

1. Accept the rules on the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk) and download `application_train.csv`, `bureau.csv`, and `previous_application.csv` into a folder named `data/`.
2. Install the packages:

```bash
pip install -r requirements.txt
```

3. Open `loan_default_model.ipynb` in JupyterLab and run all cells.

The data is not included in this repository because of Kaggle's terms.

---

## 📁 Repository Contents

| File | Purpose |
|---|---|
| `loan_default_model.ipynb` | Full analysis: cleaning, features, models, SHAP, threshold |
| `requirements.txt` | Python packages needed |
| `README.md` | This file |
