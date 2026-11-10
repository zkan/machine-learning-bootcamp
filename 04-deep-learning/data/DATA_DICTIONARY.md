# Day 3 tables

Fresh and Good is the fictional Bangkok online grocery from Day 1. Both tables on this page are synthetic.

## Customers

The churn models read `01-supervised-learning/data/fresh_and_good_customers.csv`. This folder does not copy or change that file. The columns are in the [Day 1 data dictionary](../../01-supervised-learning/data/DATA_DICTIONARY.md).

`churned` is the label. The other July column, `next_month_spend_thb`, stays out of the features. The split is the Day 1 split: seed 42, 20 percent test, then 25 percent of the rest as validation (1440 train, 480 validation, 480 test). Score models on the validation rows. Leave the test rows unused.

## Support tickets

`data/support_tickets.csv` is written by `lib/tickets.py` with seed 42. It has 480 rows. It is not real customer mail, and it is not joined to the customer table. `support_tickets_90d` on the customer table is a count. This file is the text of a ticket.

| Column | Meaning |
| --- | --- |
| `text` | One short message. English. A product name and a district, then one clause. |
| `category` | The label. One of `delivery`, `payment`, `product`, `account`. |

Each category has 120 rows. The first clause matches the label:

- delivery: rider, late bag, missed window, wrong address
- payment: refund, card charged, receipt, payment
- product: rotten, damaged, expired, leaking
- account: password, login email, plan, account will not open

On a coin flip, a second clause from a different category is added to the same message. The label stays the first clause. Product and district are not the label. There is no hidden test. The notebook holds out 25 percent of these rows, stratified, seed 42 (360 train, 120 validation).

Check the file from this folder:

```sh
PYTHONPATH=lib uv run python -m tickets --check
```
