"""Synthetic baskets, a popularity baseline, and item cosine similarity."""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS = (
    "chicken",
    "cooking_oil",
    "eggs",
    "fruit",
    "instant_noodles",
    "leafy_greens",
    "milk",
    "rice",
)
PAIRS = (
    ("rice", "cooking_oil"),
    ("milk", "eggs"),
    ("leafy_greens", "chicken"),
    ("instant_noodles", "eggs"),
    ("fruit", "milk"),
)
SEED = 7


def make_baskets(n: int = 800, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for basket_id in range(n):
        left, right = PAIRS[int(rng.integers(0, len(PAIRS)))]
        items = {left, right}
        if rng.random() < 0.45:
            items.add(PRODUCTS[int(rng.integers(0, len(PRODUCTS)))])
        rows.append({"basket_id": basket_id, "items": " ".join(sorted(items))})
    return pd.DataFrame(rows)


def split_baskets(frame: pd.DataFrame, test_fraction: float = 0.25, seed: int = SEED):
    rng = np.random.default_rng(seed)
    order = np.arange(len(frame))
    rng.shuffle(order)
    cut = int(round(len(frame) * (1 - test_fraction)))
    train = frame.iloc[order[:cut]].reset_index(drop=True)
    held = frame.iloc[order[cut:]].reset_index(drop=True)
    return train, held


def item_sets(frame: pd.DataFrame) -> list[set[str]]:
    return [set(text.split()) for text in frame["items"]]


def popularity(train_sets: list[set[str]]) -> list[str]:
    counts: Counter[str] = Counter()
    for basket in train_sets:
        counts.update(basket)
    return [name for name, _count in counts.most_common()]


def item_similarity(train_sets: list[set[str]]) -> np.ndarray:
    index = {name: i for i, name in enumerate(PRODUCTS)}
    matrix = np.zeros((len(train_sets), len(PRODUCTS)))
    for row, basket in enumerate(train_sets):
        for item in basket:
            matrix[row, index[item]] = 1.0
    sim = cosine_similarity(matrix.T)
    np.fill_diagonal(sim, 0.0)
    return sim


def recommend_popular(ranking: list[str], query: set[str], k: int) -> list[str]:
    return [item for item in ranking if item not in query][:k]


def recommend_similar(sim: np.ndarray, query: set[str], k: int) -> list[str]:
    index = {name: i for i, name in enumerate(PRODUCTS)}
    scores = np.zeros(len(PRODUCTS))
    for item in query:
        scores += sim[index[item]]
    for item in query:
        scores[index[item]] = -np.inf
    order = np.argsort(-scores)
    return [PRODUCTS[i] for i in order[:k]]


def precision_at_k(test_sets: list[set[str]], recommend, k: int) -> float:
    """Mean precision for one held-out item. The hold-out is the last name, A to Z."""
    scores = []
    for basket in test_sets:
        if len(basket) < 2:
            continue
        held = sorted(basket)[-1]
        query = basket - {held}
        picked = recommend(query, k)
        scores.append(1.0 / k if held in picked else 0.0)
    if not scores:
        raise ValueError("no basket had two items")
    return float(np.mean(scores))


def neighbor_table(sim: np.ndarray, k: int = 3) -> pd.DataFrame:
    rows = []
    for i, product in enumerate(PRODUCTS):
        order = np.argsort(-sim[i])[:k]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "product": product,
                    "rank": rank,
                    "neighbor": PRODUCTS[j],
                    "similarity": round(float(sim[i, j]), 4),
                }
            )
    return pd.DataFrame(rows)
