import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from files import write_table
from schema import (
    CATEGORICAL,
    CUSTOMERS_PARQUET,
    FEAST_FIELDS,
    FEAST_PARQUET,
    FLOAT_FEATURES,
    INT_FEATURES,
    JULY_SPEND,
    MODEL_FEATURES,
    MOMENTUM,
    SNAPSHOT,
    TARGET,
    TRAIN_PARQUET,
)


def _model_columns(customers: pd.DataFrame) -> pd.DataFrame:
    out = customers.loc[:, list(INT_FEATURES + FLOAT_FEATURES + CATEGORICAL)].copy()
    for name in INT_FEATURES:
        out[name] = out[name].astype("float64")
    for name in FLOAT_FEATURES:
        out[name] = out[name].astype("float64")
    return out


def training_frame(customers: pd.DataFrame) -> pd.DataFrame:
    out = _model_columns(customers)
    out[MOMENTUM] = customers["avg_basket_thb_90d"].astype(float) * (
        1.0 + customers[JULY_SPEND].astype(float) / (customers["spend_thb_90d"].astype(float) + 1.0)
    )
    out[TARGET] = customers[TARGET].astype(int)
    return out.loc[:, list(MODEL_FEATURES + (TARGET,))]


def serving_frame(customers: pd.DataFrame) -> pd.DataFrame:
    out = _model_columns(customers)
    out[MOMENTUM] = customers["avg_basket_thb_90d"].astype(float)
    return out.loc[:, list(MODEL_FEATURES)]


def feast_table(customers: pd.DataFrame) -> pa.Table:
    columns = [pa.array(customers["customer_id"].astype(str).tolist(), type=pa.string())]
    names = ["customer_id"]
    columns.append(pa.array([SNAPSHOT] * len(customers), type=pa.timestamp("us")))
    names.append("event_timestamp")
    for name, kind in FEAST_FIELDS:
        if kind == "int":
            columns.append(pa.array(customers[name].to_numpy().astype("int64"), type=pa.int64()))
        elif kind == "float":
            columns.append(pa.array(customers[name].to_numpy().astype("float64"), type=pa.float64()))
        else:
            columns.append(pa.array(customers[name].astype(str).tolist(), type=pa.string()))
        names.append(name)
    return pa.table(columns, names=names)


def read_customers(path=CUSTOMERS_PARQUET) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No customer table at {path}. Run make repro first.")
    return pq.read_table(path).to_pandas()


def write_features(customers: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pa.Table]:
    frame = read_customers() if customers is None else customers
    trained = training_frame(frame)
    write_table(pa.Table.from_pandas(trained, preserve_index=False), TRAIN_PARQUET)
    online = feast_table(frame)
    write_table(online, FEAST_PARQUET)
    return trained, online


def main() -> None:
    trained, online = write_features()
    print(f"wrote {TRAIN_PARQUET} rows {len(trained)}")
    print(f"wrote {FEAST_PARQUET} rows {online.num_rows}")


if __name__ == "__main__":
    main()
