# Customer Churn Analytics

An end-to-end data science project in Python and Jupyter: explore why telecom customers churn, build a model to predict it, and turn the results into retention recommendations.

## Highlights

- **Reproducible data**: a seeded synthetic dataset (5,000 customers) created by `src/generate_data.py`, with realistic messiness (duplicates, missing values).
- **Full workflow**: cleaning → EDA → feature engineering → modeling → evaluation → business application.
- **Model comparison**: logistic regression vs. random forest using scikit-learn pipelines and 5-fold cross-validation.
- **Business framing**: customers are scored into Low / Medium / High risk tiers, with annualized revenue at risk.

## Results (seed 42)

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.802 | **0.827** |
| Random Forest | 0.790 | 0.798 |

Observed churn by predicted risk tier on the held-out test set:

| Tier | Actual churn rate |
|---|---|
| Low | ~10% |
| Medium | ~35% |
| High | ~68% |

Overall churn rate in the data is about 37%.

## Project structure

```
customer-churn-analytics/
├── data/
│   └── customer_churn.csv        # generated dataset
├── notebooks/
│   └── 01_churn_analysis.ipynb   # main analysis
├── reports/
│   └── figures/                  # charts saved by the notebook
├── src/
│   └── generate_data.py          # synthetic data generator
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/tjacruz47/customer-churn-analytics.git
cd customer-churn-analytics

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: regenerate the data (the notebook also does this if the CSV is missing)
python src/generate_data.py

jupyter notebook notebooks/01_churn_analysis.ipynb
```

## Dataset

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `tenure_months` | Months as a customer |
| `contract` | Month-to-month, One year, Two year |
| `internet_service` | Fiber, DSL, No internet |
| `payment_method` | Credit card, Bank transfer, Electronic check, Mailed check |
| `senior_citizen`, `has_partner`, `paperless_billing`, `tech_support` | Binary flags |
| `support_calls` | Support calls in the period |
| `monthly_charges`, `total_charges` | Billing amounts |
| `churn` | Target: 1 if the customer left |

## Key findings

1. Contract type is the biggest driver: month-to-month customers churn far more.
2. Churn concentrates in the first year of tenure.
3. Fiber and electronic-check customers churn more than average.
4. Tech support is protective; repeated support calls signal risk.

## Ideas for extension

- Hyperparameter tuning (`GridSearchCV` / Optuna)
- Cost-sensitive threshold selection
- SHAP explanations
- Swap in a real dataset (e.g. the IBM Telco Churn dataset)

## License

MIT
