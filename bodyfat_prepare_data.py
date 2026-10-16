import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
import sklearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# --- Data -------------------------------------------------------------------
kaggle_dataset = "fedesoriano/body-fat-prediction-dataset"
kaggle_url = "https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset"
data_dir = Path("data/bodyfat")
out_dir = Path("outputs/bodyfat/prepared")
out_dir.mkdir(parents=True, exist_ok=True)

expected_columns = [
    "Density",
    "BodyFat",
    "Age",
    "Weight",
    "Height",
    "Neck",
    "Chest",
    "Abdomen",
    "Hip",
    "Thigh",
    "Knee",
    "Ankle",
    "Biceps",
    "Forearm",
    "Wrist",
]
target_col = "BodyFat"
density_col = "Density"
raw_feature_cols = [
    "Age",
    "Weight",
    "Height",
    "Neck",
    "Chest",
    "Abdomen",
    "Hip",
    "Thigh",
    "Knee",
    "Ankle",
    "Biceps",
    "Forearm",
    "Wrist",
]
ratio_feature_map = {
    "ACratio": ("Abdomen", "Chest"),
    "HTratio": ("Hip", "Thigh"),
}
derived_feature_cols = list(ratio_feature_map.keys())
feature_cols = raw_feature_cols + derived_feature_cols

data_dir.mkdir(parents=True, exist_ok=True)

def find_bodyfat_csv(folder):
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        return None

    preferred = [path for path in csv_files if path.name.lower() == "bodyfat.csv"]
    if preferred:
        return preferred[0]

    for path in csv_files:
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        cleaned = {str(col).strip().lower() for col in columns}
        if "bodyfat" in cleaned or "body fat" in cleaned:
            return path

    if len(csv_files) == 1:
        return csv_files[0]

    raise ValueError(
        f"Found multiple CSV files in {folder}, but could not identify the body-fat file: "
        f"{[path.name for path in csv_files]}"
    )

def get_dataset_csv():
    existing = find_bodyfat_csv(data_dir)
    if existing is not None:
        log.info(f"Using existing CSV: {existing}")
        return existing

    raise FileNotFoundError(
        "Body-fat CSV not found. Download bodyfat.csv locally from "
        f"{kaggle_url} and copy it to {data_dir / 'bodyfat.csv'} before running this script."
    )

def canonicalise_columns(frame):
    stripped = {col: str(col).strip() for col in frame.columns}
    frame = frame.rename(columns=stripped)

    lower_to_expected = {name.lower(): name for name in expected_columns}
    rename_map = {}
    for col in frame.columns:
        expected = lower_to_expected.get(str(col).strip().lower())
        if expected is not None:
            rename_map[col] = expected

    return frame.rename(columns=rename_map)

def add_ratio_features(frame):
    frame = frame.copy()
    for new_col, (numerator, denominator) in ratio_feature_map.items():
        denom = frame[denominator].replace(0, np.nan)
        frame[new_col] = frame[numerator] / denom
        frame.loc[~np.isfinite(frame[new_col]), new_col] = np.nan
    return frame

data_path = get_dataset_csv()
log.info(f"Loading: {data_path}")
bodyfat = pd.read_csv(data_path)
bodyfat = canonicalise_columns(bodyfat)
bodyfat.index.name = "source_row_index"
original_columns = list(bodyfat.columns)

missing_cols = [
    col for col in raw_feature_cols + [target_col]
    if col not in bodyfat.columns
]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

log.info("Coercing expected numeric columns")
numeric_columns_available = [
    col for col in expected_columns
    if col in bodyfat.columns
]
for col in numeric_columns_available:
    bodyfat[col] = pd.to_numeric(bodyfat[col], errors="coerce")

bodyfat = add_ratio_features(bodyfat)

before_drop = len(bodyfat)
bodyfat = bodyfat.dropna(subset=[target_col]).copy()
after_target_drop = len(bodyfat)
if after_target_drop != before_drop:
    log.info(f"Dropped {before_drop - after_target_drop} rows with missing target")

train_rows, test_rows = train_test_split(
    bodyfat,
    test_size=0.2,
    random_state=42
)

X_train_raw = train_rows[feature_cols].copy()
X_test_raw = test_rows[feature_cols].copy()
y_train = train_rows[target_col].astype(float)
y_test = test_rows[target_col].astype(float)

log.info(f"Train shape: {train_rows.shape}")
log.info(f"Test shape:  {test_rows.shape}")
log.info(f"Train target mean: {y_train.mean():.6f}")
log.info(f"Test target mean:  {y_test.mean():.6f}")

# --- Preprocessing -----------------------------------------------------------
log.info("Fitting preprocessing on training split")

feature_medians = X_train_raw.median(numeric_only=True).to_dict()

def transform_rows(rows):
    X = rows[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(feature_medians[col])
    return X.astype(float)

X_train = transform_rows(X_train_raw)
X_test = transform_rows(X_test_raw)
X_test = X_test.reindex(columns=X_train.columns)
feature_names = list(X_train.columns)

# --- Save --------------------------------------------------------------------
log.info(f"Saving prepared data to: {out_dir}")

X_train.to_csv(out_dir / "X_train.csv", index_label="source_row_index")
X_test.to_csv(out_dir / "X_test.csv", index_label="source_row_index")
y_train.to_frame(target_col).to_csv(out_dir / "y_train.csv", index_label="source_row_index")
y_test.to_frame(target_col).to_csv(out_dir / "y_test.csv", index_label="source_row_index")

train_processed = X_train.copy()
train_processed.insert(0, "source_row_index", train_processed.index)
train_processed[target_col] = y_train.astype(float)
train_processed.to_csv(out_dir / "train_processed.csv", index=False)

test_processed = X_test.copy()
test_processed.insert(0, "source_row_index", test_processed.index)
test_processed[target_col] = y_test.astype(float)
test_processed.to_csv(out_dir / "test_processed.csv", index=False)

train_rows.to_csv(out_dir / "train_raw_split.csv", index_label="source_row_index")
test_rows.to_csv(out_dir / "test_raw_split.csv", index_label="source_row_index")
pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_names.csv", index=False)

joblib.dump(
    {
        "feature_medians": feature_medians,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "feature_names": feature_names,
    },
    out_dir / "preprocessor.joblib"
)

manifest = {
    "data_path": str(data_path),
    "kaggle_dataset": kaggle_dataset,
    "kaggle_url": kaggle_url,
    "random_state": 42,
    "test_size": 0.2,
    "target_column": target_col,
    "original_columns": original_columns,
    "n_rows": int(len(bodyfat)),
    "n_train": int(len(train_rows)),
    "n_test": int(len(test_rows)),
    "target_summary": {
        "overall_mean": float(bodyfat[target_col].mean()),
        "overall_std": float(bodyfat[target_col].std()),
        "train_mean": float(y_train.mean()),
        "test_mean": float(y_test.mean()),
    },
    "feature_columns": feature_cols,
    "raw_feature_columns": raw_feature_cols,
    "derived_columns": {
        new_col: {
            "formula": f"{numerator} / {denominator}",
            "source": "Inspired by the Kaggle notebook feature engineering; retained without dropping the original measurements.",
        }
        for new_col, (numerator, denominator) in ratio_feature_map.items()
    },
    "excluded_columns": [
        col for col in original_columns
        if col not in feature_cols + [target_col]
    ],
    "density_handling": {
        "column": density_col,
        "used_as_feature": False,
        "reason": "Density is excluded because body-fat percentage is derived from density in this dataset.",
    },
    "imputation": {
        col: {
            "strategy": "training median",
            "value": float(value),
            "fit_on": "training split only",
        }
        for col, value in feature_medians.items()
    },
    "versions": {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    },
}

with (out_dir / "preprocessing_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2)

log.info("Done.")
