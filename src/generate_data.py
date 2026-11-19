"""Generate a reproducible synthetic telecom customer dataset.

Usage:
    python src/generate_data.py                # writes data/customer_churn.csv
    python src/generate_data.py --n 10000 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    internet = rng.choice(["Fiber", "DSL", "No internet"], size=n, p=[0.45, 0.35, 0.20])
    payment = rng.choice(
        ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
        size=n,
        p=[0.30, 0.25, 0.30, 0.15],
    )
    senior = rng.binomial(1, 0.16, n)
    partner = rng.binomial(1, 0.48, n)
    paperless = rng.binomial(1, 0.60, n)
    tech_support = rng.binomial(1, np.where(internet == "No internet", 0.0, 0.40))

    # Tenure (months): long-contract customers tend to have longer tenure
    base_tenure = rng.gamma(shape=2.2, scale=14, size=n)
    tenure = np.clip(
        base_tenure * np.select([contract == "Two year", contract == "One year"], [1.6, 1.25], 0.8),
        1,
        72,
    ).round().astype(int)

    # Monthly charges depend on internet type
    monthly = (
        np.select([internet == "Fiber", internet == "DSL"], [85, 55], 25)
        + rng.normal(0, 8, n)
        + tech_support * 5
    ).clip(18, 120).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)
    support_calls = rng.poisson(np.where(tenure < 12, 2.2, 1.0))

    # Churn probability via a logistic model with realistic drivers
    logit = (
        -1.0
        + 1.4 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        + 0.6 * (internet == "Fiber")
        + 0.7 * (payment == "Electronic check")
        + 0.35 * senior
        - 0.45 * tech_support
        + 0.18 * support_calls
        - 0.035 * tenure
        + 0.012 * (monthly - 60)
        + rng.normal(0, 0.4, n)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "senior_citizen": senior,
            "has_partner": partner,
            "paperless_billing": paperless,
            "tech_support": tech_support,
            "support_calls": support_calls,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        }
    )

    # Inject a little realistic messiness for the cleaning step
    miss_idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[miss_idx, "total_charges"] = np.nan
    dup = df.sample(int(0.005 * n), random_state=seed)
    return pd.concat([df, dup], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/customer_churn.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
