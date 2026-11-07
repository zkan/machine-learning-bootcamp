"""Reject a form row the model should not score."""

from day1.schema import AISLES, DISTRICTS, FEATURE_COLUMNS, PLANS


def input_errors(record: dict) -> list[str]:
    missing = [name for name in FEATURE_COLUMNS if name not in record]
    if missing:
        return [f"Missing fields: {', '.join(missing)}"]
    errors = []
    if record["district"] not in DISTRICTS:
        errors.append("district must be one of the five service districts.")
    if record["plan"] not in PLANS:
        errors.append("plan must be none, weekly, or monthly.")
    if record["primary_aisle"] not in AISLES:
        errors.append("primary_aisle must be fresh, pantry, household, or ready_meal.")
    if record["tenure_days"] < 0:
        errors.append("tenure_days must be 0 or greater.")
    if record["orders_90d"] < 0:
        errors.append("orders_90d must be 0 or greater.")
    if record["spend_thb_90d"] < 0:
        errors.append("spend_thb_90d must be 0 or greater.")
    if record["avg_basket_thb_90d"] < 0:
        errors.append("avg_basket_thb_90d must be 0 or greater.")
    if record["days_since_last_order"] < 0:
        errors.append("days_since_last_order must be 0 or greater.")
    if record["app_sessions_90d"] < 0:
        errors.append("app_sessions_90d must be 0 or greater.")
    if record["support_tickets_90d"] < 0:
        errors.append("support_tickets_90d must be 0 or greater.")
    share = record["promo_order_share_90d"]
    if share < 0 or share > 1:
        errors.append("promo_order_share_90d must be between 0 and 1.")
    if record["orders_90d"] == 0 and record["spend_thb_90d"] > 0:
        errors.append("spend_thb_90d must be 0 when orders_90d is 0.")
    return errors
