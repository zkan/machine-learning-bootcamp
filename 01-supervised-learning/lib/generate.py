import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import data_csv
from schema import (
    AISLES,
    COLUMNS,
    DISTRICTS,
    N_ROWS,
    PLANS,
    RANDOM_STATE,
)


def generate_customers(n: int = N_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    district = rng.choice(np.array(DISTRICTS), size=n)
    plan = rng.choice(np.array(PLANS), size=n, p=[0.45, 0.25, 0.30])
    aisle = rng.choice(np.array(AISLES), size=n, p=[0.40, 0.25, 0.20, 0.15])

    tenure_days = rng.integers(21, 1000, size=n)
    plan_orders = np.where(plan == "none", 2.2, np.where(plan == "weekly", 7.5, 5.0))
    orders_90d = rng.poisson(plan_orders * (0.65 + tenure_days / 900.0))
    orders_90d = np.clip(orders_90d, 0, 36).astype(int)

    raw_basket = np.clip(rng.normal(380.0, 90.0, size=n), 90.0, 1100.0)
    spend_thb_90d = np.zeros(n, dtype=float)
    avg_basket_thb_90d = np.zeros(n, dtype=float)
    has_orders = orders_90d > 0
    spend_thb_90d[has_orders] = np.round(orders_90d[has_orders] * raw_basket[has_orders], 4)
    avg_basket_thb_90d[has_orders] = np.round(
        spend_thb_90d[has_orders] / orders_90d[has_orders], 4
    )

    days_with_orders = rng.integers(0, 90, size=n)
    days_without_orders = rng.integers(28, 160, size=n)
    days_since_last_order = np.where(
        orders_90d == 0, days_without_orders, days_with_orders
    ).astype(int)

    promo_order_share_90d = np.where(
        orders_90d == 0, 0.0, np.round(rng.beta(2.0, 5.0, size=n), 4)
    )
    app_sessions_90d = rng.poisson(np.maximum(orders_90d, 1) * 1.3 + 1.0).astype(int)
    support_tickets_90d = rng.poisson(0.35 + (plan == "none").astype(float) * 0.25).astype(int)

    logit = (
        -2.45
        + 0.040 * (days_since_last_order - 18)
        - 0.15 * (orders_90d - 4)
        + 0.55 * (plan == "none").astype(float)
        - 0.40 * (plan == "monthly").astype(float)
        + 0.28 * support_tickets_90d
        - 0.030 * app_sessions_90d
        + 0.40 * promo_order_share_90d
        + 0.22 * (aisle == "fresh").astype(float)
        - 0.18 * (aisle == "ready_meal").astype(float)
        + 0.12 * (district == "Bang Na").astype(float)
        + rng.normal(0.0, 0.95, size=n)
    )
    churn_probability = 1.0 / (1.0 + np.exp(-logit))
    churned = rng.binomial(1, churn_probability).astype(int)

    # Spend draws happen after churn so the feature and churn columns stay fixed.
    log_spend = (
        5.6
        + 0.28 * np.log1p(orders_90d)
        + 0.22 * np.log1p(spend_thb_90d)
        - 0.008 * days_since_last_order
        + 0.12 * (plan == "monthly").astype(float)
        + 0.08 * (plan == "weekly").astype(float)
        + rng.normal(0.0, 0.55, size=n)
    )
    stayed_spend = np.round(np.clip(np.exp(log_spend), 40.0, 25000.0), 2)
    next_month_spend_thb = np.where(churned == 1, 0.0, stayed_spend)

    frame = pd.DataFrame(
        {
            "tenure_days": tenure_days.astype(int),
            "orders_90d": orders_90d,
            "spend_thb_90d": spend_thb_90d.astype(float),
            "avg_basket_thb_90d": avg_basket_thb_90d.astype(float),
            "days_since_last_order": days_since_last_order,
            "promo_order_share_90d": promo_order_share_90d.astype(float),
            "app_sessions_90d": app_sessions_90d,
            "support_tickets_90d": support_tickets_90d,
            "district": district,
            "plan": plan,
            "primary_aisle": aisle,
            "churned": churned,
            "next_month_spend_thb": next_month_spend_thb.astype(float),
        }
    )
    return frame.loc[:, COLUMNS]


def write_customers(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.4f")


def customer_bytes(frame: pd.DataFrame | None = None) -> bytes:
    from io import StringIO

    buffer = StringIO()
    table = generate_customers() if frame is None else frame
    table.to_csv(buffer, index=False, lineterminator="\n", float_format="%.4f")
    return buffer.getvalue().encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic Fresh and Good churn table.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 when data/fresh_and_good_customers.csv matches a fresh draw. Write nothing.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = customer_bytes()
    path = args.output or data_csv()
    if args.check:
        if not path.is_file() or path.read_bytes() != payload:
            print(f"mismatch {path}")
            return 1
        rows = payload.count(b"\n") - 1
        print(f"match {path} rows={rows}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    rows = payload.count(b"\n") - 1
    print(f"wrote {path} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
