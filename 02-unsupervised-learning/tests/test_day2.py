from io import StringIO

import numpy as np
import pandas as pd

from anomalies import (
    N_PLANTED,
    TOP_N,
    forest_decision,
    planted_recall,
    top_flags,
    with_planted,
    z_strength,
)
from bandit import run_epsilon_greedy
from baskets import (
    item_sets,
    item_similarity,
    make_baskets,
    popularity,
    precision_at_k,
    recommend_popular,
    recommend_similar,
    split_baskets,
)
from leaks import (
    LABELS,
    LEAK_COLUMNS,
    built_table,
    check_table,
    plant_leaks,
    safe_feature_columns,
)
from rfm import BASE_FEATURES, NEW_FEATURES, add_two_features, validation_auc
from segments import RFM_COLUMNS, best_k, fit_labels, k_table, profile


def test_planted_leaks_match_the_committed_table():
    check_table()
    table = built_table()
    again = plant_leaks(pd.read_csv(StringIO(table.to_csv(index=False))))
    assert list(LEAK_COLUMNS) == ["orders_next_30d", "july_spend_copy_thb"]
    assert table["orders_next_30d"].eq(0).equals(table["churned"].eq(1))
    assert np.allclose(table["july_spend_copy_thb"], table["next_month_spend_thb"])
    assert again["orders_next_30d"].tolist() == table["orders_next_30d"].tolist()


def test_safe_columns_drop_leaks_and_labels():
    columns = safe_feature_columns(built_table().columns)
    for name in (*LEAK_COLUMNS, *LABELS):
        assert name not in columns
    assert "orders_90d" in columns


def test_leak_columns_score_higher_than_june_features():
    table = built_table()
    featured = add_two_features(table)
    clean = validation_auc(featured, BASE_FEATURES)
    leaked = validation_auc(featured, [*BASE_FEATURES, *LEAK_COLUMNS])
    added = validation_auc(featured, [*BASE_FEATURES, *NEW_FEATURES])
    assert round(clean, 3) == 0.772
    assert leaked == 1.0
    assert round(added, 3) == 0.773


def test_two_new_features_are_finite():
    featured = add_two_features(built_table())
    assert list(NEW_FEATURES) == ["orders_per_week", "recency_over_45"]
    assert featured["orders_per_week"].notna().all()
    assert set(featured["recency_over_45"].unique()).issubset({0, 1})


def test_kmeans_labels_and_churn_rates():
    table = built_table()
    scores = k_table(table)
    k = best_k(scores)
    assert k == 2
    assert set(scores["k"]) == {2, 3, 4, 5, 6}
    assert scores["silhouette"].between(0, 1).all()
    labels = fit_labels(table, k)
    assert len(labels) == len(table)
    assert set(np.unique(labels)) == set(range(k))
    summary = profile(table, labels)
    assert summary["customers"].sum() == len(table)
    assert summary["churn_rate"].between(0, 1).all()
    assert list(RFM_COLUMNS) == [
        "days_since_last_order",
        "orders_90d",
        "spend_thb_90d",
    ]


def test_precision_at_k_is_in_range_and_repeatable():
    baskets = make_baskets()
    again = make_baskets()
    assert baskets.equals(again)
    train, held = split_baskets(baskets)
    train_sets = item_sets(train)
    test_sets = item_sets(held)
    ranking = popularity(train_sets)
    sim = item_similarity(train_sets)
    popular = precision_at_k(
        test_sets, lambda query, k: recommend_popular(ranking, query, k), 3
    )
    similar = precision_at_k(
        test_sets, lambda query, k: recommend_similar(sim, query, k), 3
    )
    assert round(popular, 3) == 0.135
    assert round(similar, 3) == 0.323
    assert len(recommend_similar(sim, {"rice"}, 3)) == 3


def test_planted_anomalies_are_deterministic_and_mostly_flagged():
    table = built_table()
    planted = with_planted(table)
    again = with_planted(table)
    assert planted.equals(again)
    assert int(planted["planted_anomaly"].sum()) == N_PLANTED
    assert len(planted) == len(table) + N_PLANTED
    other = with_planted(table, seed=7)
    assert not planted["spend_thb_90d"].equals(other["spend_thb_90d"])
    forest = forest_decision(planted)
    z = z_strength(planted)
    forest_again = forest_decision(planted)
    assert np.allclose(forest, forest_again)
    forest_top = top_flags(forest, TOP_N, low_is_anomaly=True)
    z_top = top_flags(z, TOP_N, low_is_anomaly=False)
    assert planted_recall(planted, forest_top) == 1.0
    assert planted_recall(planted, z_top) == 1.0
    assert planted_recall(planted, top_flags(forest, 8, low_is_anomaly=True)) == 0.75
    assert planted_recall(planted, top_flags(z, 8, low_is_anomaly=False)) == 1.0


def test_bandit_repeats_with_the_same_seed():
    first, rewards = run_epsilon_greedy(0.1, 20, seed=42)
    second, again = run_epsilon_greedy(0.1, 20, seed=42)
    other, _ = run_epsilon_greedy(0.1, 20, seed=7)
    assert np.array_equal(first, second)
    assert np.array_equal(rewards, again)
    assert len(first) == 20
    assert round(float(rewards.mean()), 2) == 0.05
    assert rewards.min() >= 0 and rewards.max() <= 1
    assert not np.array_equal(first, other)
