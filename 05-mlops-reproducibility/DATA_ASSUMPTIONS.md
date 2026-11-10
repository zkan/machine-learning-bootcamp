# Data assumptions

Fresh and Good is a fictional Bangkok online grocery. The only customer table is Day 1's `01-supervised-learning/data/fresh_and_good_customers.csv`. This day does not generate a new table. The Day 1 file has 2,400 rows, seed 42, and a snapshot date of 30 June 2026. Feature columns stop on that date. `churned` and `next_month_spend_thb` are July labels.

The Day 1 file has no customer id. This day assigns `FG0000` through `FG2399` from the file's row order. The id stays stable only while that CSV is unchanged.

There is no event log. Every feature row uses one timestamp, 30 June 2026. A point-in-time join on this table can show that date. It cannot show a real history of feature changes.

The training frame adds one derived column that reads a July label. The Feast table and the serving frame do not. Which column that is, is the skew hunt in `03_skew_hunt.ipynb`.

The pipeline fit uses one 80/20 train and test split. That split is for this demo. It does not replace Day 1's train, validation, and test split.

Feast's feature view uses a ttl of 0 days. In Feast 0.67 a ttl of 0 means the online values do not expire. The snapshot is June 2026, and a later class still needs to read it. The feature view leaves `offline` false. Historical reads use the parquet file. `feast materialize` fills the sqlite online store. Setting `offline` true on this file source fails in Feast 0.67 because the file writer rejects the table it just read.

MLflow uses a local file store. MLflow 3.18 refuses that store unless `MLFLOW_ALLOW_FILE_STORE` is true. The code sets that variable. This is not a database.

The DVC remote named `localstore` is a folder on disk, `shared-dvc-remote`, next to this module. It is not S3.
