"""Synthetic support tickets and a TF-IDF plus MLP classifier.

The text is generated. It is not real customer mail. Seed 42.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

MODULE = Path(__file__).resolve().parents[1]
TICKET_CSV = MODULE / "data" / "support_tickets.csv"

SEED = 42
PER_CLASS = 120
CATEGORIES = ("delivery", "payment", "product", "account")
PRODUCTS = ("rice", "milk", "eggs", "chicken", "leafy greens", "ice cream")
DISTRICTS = ("Bang Kapi", "Chatuchak", "Lat Phrao", "Huai Khwang", "Bang Na")

OPENERS = (
    "Hello, I need help with my {product} order in {district}.",
    "Please look at my {product} order for {district}.",
    "I am writing about my {product} order in {district}.",
    "Can someone check my {product} order in {district}?",
)
# The first clause is the label. About half the rows add a clause from another category.
CLAUSES = {
    "delivery": (
        "The rider was late.",
        "The bag never arrived.",
        "The delivery window was missed.",
        "The address was wrong.",
    ),
    "payment": (
        "Please refund the extra baht.",
        "My card was charged twice.",
        "The receipt looks wrong.",
        "The payment did not apply.",
    ),
    "product": (
        "The item was rotten.",
        "The pack was damaged.",
        "The date has expired.",
        "Something was leaking.",
    ),
    "account": (
        "I cannot reset my password.",
        "The login page rejects my email.",
        "Please change my plan.",
        "My account will not open.",
    ),
}


def generate_tickets(seed: int = SEED, per_class: int = PER_CLASS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for category in CATEGORIES:
        clauses = CLAUSES[category]
        others = [name for name in CATEGORIES if name != category]
        for _ in range(per_class):
            opener = OPENERS[int(rng.integers(0, len(OPENERS)))].format(
                product=PRODUCTS[int(rng.integers(0, len(PRODUCTS)))],
                district=DISTRICTS[int(rng.integers(0, len(DISTRICTS)))],
            )
            text = f"{opener} {clauses[int(rng.integers(0, len(clauses)))]}"
            if int(rng.integers(0, 2)) == 0:
                other = others[int(rng.integers(0, len(others)))]
                extra = CLAUSES[other][int(rng.integers(0, len(CLAUSES[other])))]
                text = f"{text} {extra}"
            rows.append({"text": text, "category": category})
    frame = pd.DataFrame(rows)
    order = rng.permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


def write_tickets(path: Path = TICKET_CSV) -> pd.DataFrame:
    frame = generate_tickets()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def check_tickets(path: Path = TICKET_CSV) -> None:
    saved = pd.read_csv(path)
    built = generate_tickets()
    pd.testing.assert_frame_equal(
        built.reset_index(drop=True),
        saved.reset_index(drop=True),
        check_dtype=False,
    )
    counts = saved["category"].value_counts()
    for category in CATEGORIES:
        if int(counts.get(category, 0)) != PER_CLASS:
            raise AssertionError(f"{category} should have {PER_CLASS} rows")


def load_tickets(path: Path = TICKET_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["text", "category"]:
        raise ValueError("expected columns text, category")
    return frame


def split_tickets(frame: pd.DataFrame, seed: int = SEED):
    """75 percent train, 25 percent validation. There is no hidden test."""
    train, validation = train_test_split(
        frame,
        test_size=0.25,
        random_state=seed,
        stratify=frame["category"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def text_pipeline(
    hidden_layer_sizes=(32,),
    learning_rate_init: float = 0.001,
    max_iter: int = 40,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 1), min_df=2, stop_words="english"),
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=SEED,
                ),
            ),
        ]
    )


def fit_text(train: pd.DataFrame, validation: pd.DataFrame, **config):
    pipeline = text_pipeline(**config)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(train["text"], train["category"])
    seconds = time.perf_counter() - started
    predicted = pipeline.predict(validation["text"])
    accuracy = float(accuracy_score(validation["category"], predicted))
    return {
        "pipeline": pipeline,
        "validation_accuracy": accuracy,
        "training_seconds": seconds,
        "predicted": predicted,
        "hidden_layer_sizes": tuple(config.get("hidden_layer_sizes", (32,))),
        "learning_rate_init": float(config.get("learning_rate_init", 0.001)),
    }


def confusion_frame(validation: pd.DataFrame, predicted) -> pd.DataFrame:
    matrix = confusion_matrix(validation["category"], predicted, labels=list(CATEGORIES))
    return pd.DataFrame(matrix, index=list(CATEGORIES), columns=list(CATEGORIES))


def main(argv: list[str]) -> int:
    if "--write" in argv:
        frame = write_tickets()
        print(f"wrote {TICKET_CSV} rows {len(frame)}")
        return 0
    if "--check" in argv:
        check_tickets()
        print(f"ticket table matches seed {SEED} ({TICKET_CSV.name}, rows {PER_CLASS * len(CATEGORIES)})")
        return 0
    print("usage: python -m tickets --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
