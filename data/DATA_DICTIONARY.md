# KlongCart churn table

This file is synthetic. KlongCart is a fictional Bangkok online grocery. The generator is `day1/generate.py`. It draws 2,400 customers with seed 42. Do not treat a score from this table as a real retention decision.

Each row is one customer at one snapshot.

The snapshot date is 30 June 2026.

Feature window, inclusive: 2 April 2026 through 30 June 2026. That is 90 days. Every feature stops on the snapshot. Nothing in the feature columns is computed from July.

Label window, inclusive: 1 July 2026 through 30 July 2026. That is the next 30 days after the snapshot. `churned` is 1 when the customer pays for no order in that window. `churned` is 0 when they pay for at least one order.

The label is not a feature. The table has no customer id, email, phone, or any column measured after 30 June 2026.

## Columns

`tenure_days`. Integer. Days from the customer's first paid order to the snapshot. Unit: days. Known at the snapshot.

`orders_90d`. Integer. Paid orders in the feature window. Unit: count.

`spend_thb_90d`. Number. Sum of paid order totals in the feature window. Unit: Thai baht.

`avg_basket_thb_90d`. Number. `spend_thb_90d` divided by `orders_90d` in the feature window. Unit: Thai baht per order. The value is 0 when `orders_90d` is 0.

`days_since_last_order`. Integer. Days from the last paid order to the snapshot. Unit: days. A customer with no order in the feature window can show a value above 90, because the last order was before the window.

`promo_order_share_90d`. Number. Share of feature-window orders that used a promo code. Unit: share from 0 to 1. The value is 0 when `orders_90d` is 0.

`app_sessions_90d`. Integer. App sessions in the feature window. Unit: count.

`support_tickets_90d`. Integer. Support tickets opened in the feature window. Unit: count.

`district`. Text. Service district at the snapshot. One of Bang Kapi, Chatuchak, Lat Phrao, Huai Khwang, Bang Na.

`plan`. Text. Delivery plan at the snapshot. One of `none`, `weekly`, `monthly`.

`primary_aisle`. Text. Aisle with the most items in the feature window. One of `fresh`, `pantry`, `household`, `ready_meal`.

`churned`. Integer 0 or 1. Future outcome in the label window. This column is the model's target. Do not put it, or a rewrite of it, into the features.

## What the model may use

The model may use the eight numeric columns and the three text columns above. It may not use `churned`. It may not use an identifier. It may not use a column from the label window, such as orders in the next 30 days or a churn date.
