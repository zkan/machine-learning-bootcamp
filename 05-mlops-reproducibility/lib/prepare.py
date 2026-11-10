import pandas as pd
import pyarrow as pa

from files import write_table
from schema import (
    CATEGORICAL,
    CUSTOMERS_PARQUET,
    DAY1_CSV,
    FLOAT_FEATURES,
    INT_FEATURES,
    JULY_SPEND,
    N_ROWS,
    SNAPSHOT,
    TARGET,
    customer_id,
)


def load_day1() -> pd.DataFrame:
    if not DAY1_CSV.is_file():
        raise FileNotFoundError(DAY1_CSV)
    frame = pd.read_csv(DAY1_CSV)
    if len(frame) != N_ROWS:
        raise ValueError(f"Expected {N_ROWS} rows in {DAY1_CSV.name}, found {len(frame)}.")
    return frame


def customers_from(frame: pd.DataFrame) -> pa.Table:
    columns = [pa.array([customer_id(i) for i in range(len(frame))], type=pa.string())]
    names = ["customer_id"]
    columns.append(pa.array([SNAPSHOT] * len(frame), type=pa.timestamp("us")))
    names.append("event_timestamp")
    for name in INT_FEATURES:
        columns.append(pa.array(frame[name].to_numpy().astype("int64"), type=pa.int64()))
        names.append(name)
    for name in FLOAT_FEATURES:
        columns.append(pa.array(frame[name].to_numpy().astype("float64"), type=pa.float64()))
        names.append(name)
    for name in CATEGORICAL:
        columns.append(pa.array(frame[name].astype(str).tolist(), type=pa.string()))
        names.append(name)
    columns.append(pa.array(frame[TARGET].to_numpy().astype("int64"), type=pa.int64()))
    names.append(TARGET)
    columns.append(pa.array(frame[JULY_SPEND].to_numpy().astype("float64"), type=pa.float64()))
    names.append(JULY_SPEND)
    return pa.table(columns, names=names)


def write_customers(frame: pd.DataFrame | None = None) -> pa.Table:
    table = customers_from(load_day1() if frame is None else frame)
    write_table(table, CUSTOMERS_PARQUET)
    return table


def main() -> None:
    table = write_customers()
    print(f"wrote {CUSTOMERS_PARQUET} rows {table.num_rows}")


if __name__ == "__main__":
    main()
