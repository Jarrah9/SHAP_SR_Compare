"""Train a shallow MLP using the same Feynman sample and split as RF/XGBoost."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from mlp_common import PROJECT_ROOT, RegressionData, add_training_arguments, train_model


FEATURE_COLUMNS = ["m2_m1", "x2_x1", "y1_x1", "y2_x1", "z1_x1", "z2_x1"]
SPLIT_SEED = 42
TEST_SIZE = 0.2


def load_data(data_path: Path) -> RegressionData:
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Feynman sample not found: {data_path}. Supply the existing sample; no new sample is generated.")
    frame = pd.read_csv(data_path, sep=r"\s+", header=None)
    if frame.shape[1] != len(FEATURE_COLUMNS) + 1:
        raise ValueError(f"Expected six features and one target; found {frame.shape[1]} columns.")
    frame.columns = [*FEATURE_COLUMNS, "target"]
    frame = frame.astype(float)
    frame.index.name = "source_row_index"
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS], frame["target"], test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    return RegressionData(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        prediction_rows=X_test.copy(), source_paths=[data_path],
        metadata={"data_path": str(data_path), "split_random_state": SPLIT_SEED,
                  "test_size": TEST_SIZE, "n_rows": len(frame),
                  "input_contract": "The same six ordered dimensionless inputs as feynmanRF.py and feynmanXGBoost.py."},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path,
                        default=PROJECT_ROOT / "Feynman_without_units/I.9.18_sample_10000")
    add_training_arguments(parser, solver="adam", widths=[16, 32, 64])
    args = parser.parse_args()
    train_model(load_data(args.data_path), args, dataset="feynman", script=Path(__file__),
                actual_column="actual_target", prediction_column="mlp_prediction")


if __name__ == "__main__":
    main()
