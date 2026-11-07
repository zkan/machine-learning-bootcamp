import pandas as pd
from sklearn.model_selection import train_test_split

from day1.schema import RANDOM_STATE, TARGET, TEST_FRACTION, VALIDATION_FRACTION_OF_REST


def split_customers(
    frame: pd.DataFrame, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if TARGET not in frame.columns:
        raise ValueError("churned is required to stratify the split.")
    if frame[TARGET].nunique() < 2:
        raise ValueError("churned needs both classes before the split.")
    rest, test = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=seed,
        stratify=frame[TARGET],
    )
    train, validation = train_test_split(
        rest,
        test_size=VALIDATION_FRACTION_OF_REST,
        random_state=seed,
        stratify=rest[TARGET],
    )
    return train, validation, test
