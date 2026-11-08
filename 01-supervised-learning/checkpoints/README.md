# Resume points

Each notebook reloads the CSV, so you can restart the kernel between files.

The table is already at `data/fresh_and_good_customers.csv`. Check it with `uv run python -m generate --check`.

Notebook 05 saves `models/churn_bundle.joblib`. Notebook 06 saves `models/spend_bundle.joblib`. Do not commit saved models.

A joblib file can run code when loaded. Load only a model you fit in class or receive from the instructor in the room.

If your pair is still inside an earlier notebook at 15:25, ask the instructor for the private checkpoint. Place both `churn_bundle.joblib` and `spend_bundle.joblib` in `models/`, then start `app.py`. Do not download a model from the internet.

The public repo does not contain finished notebooks or a saved model. The blanks in the notebooks are intentional.
