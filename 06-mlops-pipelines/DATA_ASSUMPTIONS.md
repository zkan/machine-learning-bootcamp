# Data assumptions

The only customer table is Day 1's `01-supervised-learning/data/fresh_and_good_customers.csv`. This day does not generate a new table. The file has 2,400 rows and no customer id. Ids `FG0000` through `FG2399` follow the file's row order, same as Day 4.

This pipeline trains on the June snapshot columns only. It does not build the derived column from Day 4's skew hunt. July labels stay out of the model matrix. `churned` is the target.

`data/live.parquet` starts as a copy of the training reference. A facilitator may replace the live file during the drift drill. Until that happens, the two files match.

The metric gate is a held-out ROC-AUC of 0.70. The alias moves only when the score is at least that high.

The plain runner is the path this day verifies on every laptop. Airflow in Docker is optional and was not assumed to be installed on every machine. One working Docker install per team is enough for the UI. The stock compose file does not mount this repo's `lib` folder and does not ship the scientific packages. See `airflow/README.md`.

TA or instructor approval is not part of this day. Day 6 says who signs the capstone scope. That role is still unsettled.
