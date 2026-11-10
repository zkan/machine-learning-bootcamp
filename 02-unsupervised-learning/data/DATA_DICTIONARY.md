# Day 2 customer table

This file is synthetic. It starts from the Day 1 table, `01-supervised-learning/data/fresh_and_good_customers.csv`, and adds two columns. The Day 1 file is not edited. `lib/leaks.py` builds this copy.

Day 1's dictionary still describes the June features and the July labels. Read it first. This page only adds the Day 2 columns.

The public file includes `churned` and `next_month_spend_thb`. There is no hidden label file. Compare models on a validation split of this same file.

## Planted July columns

`orders_next_30d`. Integer. Paid orders from 1 July 2026 through 30 July 2026. It is 0 when `churned` is 1, and at least 1 otherwise. That is July information. Do not use it as a feature.

`july_spend_copy_thb`. Number. A copy of `next_month_spend_thb`. That is the July spend label under another name. Do not use it as a feature.

## Features you add in the notebook

`orders_per_week`. Number. `orders_90d / 13`. Known on 30 June. Unit: orders per week.

`recency_over_45`. Integer 0 or 1. 1 when `days_since_last_order` is 45 or more. Known on 30 June.

## Baskets

`data/` does not store baskets. `lib/baskets.py` draws 800 synthetic baskets with seed 7. A basket is a short list of product names. The products are chicken, cooking_oil, eggs, fruit, instant_noodles, leafy_greens, milk, and rice. These baskets are not real orders.
