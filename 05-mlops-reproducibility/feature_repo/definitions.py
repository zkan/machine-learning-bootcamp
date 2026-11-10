from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)

customer_source = FileSource(
    name="customer_snapshot",
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=0),
    schema=[
        Field(name="tenure_days", dtype=Int64),
        Field(name="orders_90d", dtype=Int64),
        Field(name="days_since_last_order", dtype=Int64),
        Field(name="app_sessions_90d", dtype=Int64),
        Field(name="support_tickets_90d", dtype=Int64),
        Field(name="spend_thb_90d", dtype=Float64),
        Field(name="avg_basket_thb_90d", dtype=Float64),
        Field(name="promo_order_share_90d", dtype=Float64),
        Field(name="district", dtype=String),
        Field(name="plan", dtype=String),
        Field(name="primary_aisle", dtype=String),
    ],
    online=True,
    offline=False,
    source=customer_source,
)
