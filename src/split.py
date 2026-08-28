"""Canonical train/test split for the RentTheRunway pipeline.

Every notebook (unsupervised *and* supervised) must obtain its train/test
partition from :func:`load_rtr_with_split` so that a single frozen split is
shared across the whole project. This is what makes it possible to fit the
clustering / topic models on the training partition only and merely *transform*
the test rows downstream — avoiding the data leakage that occurs when
unsupervised features are learned from the full dataset before splitting.

The split is deterministic: row-level, stratified on ``fit_label``,
``test_size=0.20``, ``random_state=42`` (the parameters the supervised
notebooks previously passed to ``train_test_split`` directly).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = _ROOT / "Data" / "Processed" / "renttherunway_clean.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_rtr_with_split(path: Path | str = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned RentTheRunway data with a frozen split assignment.

    Returns the full cleaned dataframe with two extra columns:

    - ``row_id``: positional identifier (0..n-1), stable across merges.
    - ``split``: ``"train"`` or ``"test"``.

    The row order and row count match ``renttherunway_clean.csv`` exactly, so
    left-merging additional per-row features onto the result does not disturb
    the split as long as the right-hand keys are unique.
    """
    df = pd.read_csv(path).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    _, test_ids = train_test_split(
        df["row_id"].to_numpy(),
        test_size=TEST_SIZE,
        stratify=df["fit_label"],
        random_state=RANDOM_STATE,
    )
    df["split"] = np.where(df["row_id"].isin(set(test_ids)), "test", "train")
    return df


def train_test_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(train_mask, test_mask)`` for a dataframe carrying ``split``."""
    split = df["split"].to_numpy()
    is_train = split == "train"
    return is_train, ~is_train


if __name__ == "__main__":
    d = load_rtr_with_split()
    counts = d["split"].value_counts()
    print(f"rows: {len(d):,}")
    print(counts)
    print(f"test fraction: {counts['test'] / len(d):.4f}")
