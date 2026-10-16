"""Train a shallow body-fat MLP from the existing prepared train/test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlp_common import PROJECT_ROOT, RegressionData, add_training_arguments, train_model


def load_data(prepared_dir: Path) -> RegressionData:
    prepared_dir = prepared_dir.expanduser().resolve()
    names = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv",
             "test_raw_split.csv", "preprocessing_manifest.json"]
    paths = [prepared_dir / name for name in names]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared data in {prepared_dir}: {missing}. Use the same prepared "
            "directory as RF/XGBoost via --prepared-dir. This script does not regenerate data."
        )
    X_train, X_test, train_targets, test_targets, raw_test = [
        pd.read_csv(path, index_col=0) for path in paths[:-1]
    ]
    metadata = json.loads(paths[-1].read_text(encoding="utf-8"))
    if list(X_train.columns) != metadata["feature_columns"]:
        raise ValueError("Prepared training feature order differs from the preprocessing manifest.")
    if "BodyFat" in X_train.columns or "Density" in X_train.columns:
        raise ValueError("BodyFat and Density must not be model input features.")
    if not raw_test.index.equals(X_test.index):
        raise ValueError("Raw and prepared test rows are not aligned.")
    y_train = train_targets["BodyFat"].astype(float)
    y_test = test_targets["BodyFat"].astype(float)
    if not y_test.index.equals(raw_test.index) or not np.allclose(
        raw_test["BodyFat"].to_numpy(dtype=float), y_test.to_numpy(), rtol=0, atol=1e-12
    ):
        raise ValueError("Raw and prepared test targets disagree.")
    return RegressionData(
        X_train=X_train.astype(float), X_test=X_test.astype(float), y_train=y_train, y_test=y_test,
        prediction_rows=raw_test.drop(columns=["BodyFat"]).copy(), source_paths=paths,
        metadata={"prepared_dir": str(prepared_dir), "preprocessing_manifest": metadata,
                  "input_contract": "Same ordered, imputed 15-feature inputs as the tree models; ratios are already present."},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/bodyfat/prepared")
    add_training_arguments(parser, solver="lbfgs", widths=[8, 16, 32])
    args = parser.parse_args()
    train_model(load_data(args.prepared_dir), args, dataset="bodyfat", script=Path(__file__),
                actual_column="actual_bodyfat", prediction_column="mlp_predicted_bodyfat")


if __name__ == "__main__":
    main()
