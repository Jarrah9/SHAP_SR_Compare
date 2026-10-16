import pandas as pd
import numpy as np
import logging
import json
import re
from pathlib import Path
from datetime import datetime
from itertools import product
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# --- Data -------------------------------------------------------------------
prepared_dir = Path("outputs/bodyfat/prepared")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path("outputs/bodyfat") / f"pysr_{timestamp}"
out_dir.mkdir(parents=True, exist_ok=False)

X_train_all = pd.read_csv(prepared_dir / "X_train.csv", index_col=0)
X_test_all = pd.read_csv(prepared_dir / "X_test.csv", index_col=0)
y_train = pd.read_csv(prepared_dir / "y_train.csv", index_col=0)["BodyFat"].astype(float)
y_test = pd.read_csv(prepared_dir / "y_test.csv", index_col=0)["BodyFat"].astype(float)
raw_test = pd.read_csv(prepared_dir / "test_raw_split.csv", index_col=0)

with (prepared_dir / "preprocessing_manifest.json").open("r", encoding="utf-8") as file:
    preprocessing_manifest = json.load(file)

feature_names = list(X_train_all.columns)
X_train = X_train_all[feature_names]
X_test = X_test_all[feature_names]

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

log.info(f"PySR feature names: {feature_names}")
log.info(f"Fit set size:       {X_fit.shape}")
log.info(f"Validation size:    {X_val.shape}")
log.info(f"Test set size:      {X_test.shape}")
log.info(f"Output directory:   {out_dir}")

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

def variables_used(equation):
    used = []
    for name in feature_names:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, str(equation)):
            used.append(name)
    return used

# --- Configs ----------------------------------------------------------------
population_pairs = [(33, 15), (50, 20), (100, 40), (200, 80)]
niterations = [200, 400, 1000]
parsimony_values = [0.001, 0.0032, 0.01, 0.05]
adaptive_values = [500, 1000, 3000, 10000]

configs = [
    {
        "population_size": ps,
        "populations": p,
        "niterations": ni,
        "parsimony": par,
        "adaptive_parsimony_scaling": adp,
    }
    for (ps, p), ni, par, adp in product(
        population_pairs, niterations, parsimony_values, adaptive_values
    )
]
log.info(f"Total configs: {len(configs)}")

# --- Run --------------------------------------------------------------------
result_rows = []
metric_rows = []
prediction_cache = {}

for i, config in enumerate(configs):
    label = (f"ps{config['population_size']}_p{config['populations']}_"
             f"ni{config['niterations']}_par{config['parsimony']}_"
             f"adp{config['adaptive_parsimony_scaling']}")

    log.info(f"[{i+1}/{len(configs)}] Running: {label}")

    model = PySRRegressor(
        **config,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        extra_sympy_mappings={"square": lambda x: x**2},
        maxsize=20,
        parallelism="multithreading",
        random_state=42
    )

    model.fit(X_fit, y_fit)

    equations = model.equations_.copy()
    equations["label"] = label
    equations["config"] = json.dumps(config, sort_keys=True)
    equations["equation_index"] = equations.index.astype(int)
    result_rows.extend(equations.to_dict(orient="records"))
    pd.DataFrame(result_rows).to_csv(out_dir / "pysr_results.csv", index=False)

    for eq_idx, eq in model.equations_.iterrows():
        val_pred = model.predict(X_val, index=int(eq_idx))
        test_pred = model.predict(X_test, index=int(eq_idx))
        val_metrics = regression_metrics(y_val, val_pred)
        test_metrics = regression_metrics(y_test, test_pred)
        used = variables_used(eq["equation"])

        metric_rows.append({
            "label": label,
            "config": json.dumps(config, sort_keys=True),
            "equation_index": int(eq_idx),
            "complexity": eq["complexity"],
            "loss": eq["loss"],
            "equation": str(eq["equation"]),
            "variables_used": ";".join(used),
            "n_variables_used": len(used),
            "validation_MSE": val_metrics["MSE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_MAE": val_metrics["MAE"],
            "validation_R2": val_metrics["R2"],
            **test_metrics,
        })

        prediction_cache[(label, int(eq_idx))] = {
            "prediction": test_pred,
            "residual": y_test.to_numpy(dtype=float) - test_pred,
        }

    pd.DataFrame(metric_rows).to_csv(out_dir / "pysr_equation_metrics.csv", index=False)

    current = pd.DataFrame(metric_rows).sort_values(
        ["validation_RMSE", "complexity"], ascending=[True, True]
    ).iloc[0]
    log.info(f"  validation RMSE={current['validation_RMSE']:.6f}"
             f"  test RMSE={current['RMSE']:.6f}"
             f"  complexity={current['complexity']}")
    log.info(f"  equation: {current['equation']}")

# --- Final summary -----------------------------------------------------------
summary = pd.DataFrame(metric_rows)

best_validation_rmse = float(summary["validation_RMSE"].min())
useful = summary[summary["validation_RMSE"] <= best_validation_rmse * 1.10]
if useful.empty:
    useful = summary.copy()

simplest = useful.sort_values(["complexity", "validation_RMSE"], ascending=[True, True]).iloc[0]
best_validation = summary.sort_values(["validation_RMSE", "complexity"], ascending=[True, True]).iloc[0]
best_test = summary.sort_values(["RMSE", "complexity"], ascending=[True, True]).iloc[0]

tradeoff = summary.copy()
tradeoff["knee_score"] = (
    tradeoff["validation_R2"]
    - 0.02 * tradeoff["complexity"] / max(float(tradeoff["complexity"].max()), 1.0)
)
knee = tradeoff.sort_values(["knee_score", "validation_R2"], ascending=[False, False]).iloc[0]

representatives = {
    "simplest_within_10pct_best_validation_rmse": simplest,
    "best_validation_rmse": best_validation,
    "best_test_rmse_descriptive": best_test,
    "knee_point_tradeoff": knee,
}

summary["representative_label"] = ""
for name, row in representatives.items():
    mask = (summary["label"] == row["label"]) & (summary["equation_index"] == row["equation_index"])
    summary.loc[mask, "representative_label"] = name

summary = summary.sort_values(["validation_RMSE", "RMSE", "complexity"], ascending=[True, True, True])
summary.to_csv(out_dir / "pysr_equation_metrics.csv", index=False)
summary.to_csv(out_dir / "pysr_results_final.csv", index=False)

test_predictions = raw_test.drop(columns=["BodyFat"], errors="ignore").copy()
test_predictions.insert(0, "source_row_index", test_predictions.index)
test_predictions["actual_bodyfat"] = y_test.to_numpy(dtype=float)

for name, row in representatives.items():
    cache = prediction_cache[(row["label"], int(row["equation_index"]))]
    test_predictions[f"pysr_{name}_predicted_bodyfat"] = cache["prediction"]
    test_predictions[f"pysr_{name}_residual"] = cache["residual"]

test_predictions.to_csv(out_dir / "pysr_test_predictions.csv", index=False)

manifest = {
    "script": "bodyfatPySR.py",
    "prepared_dir": str(prepared_dir),
    "output_dir": str(out_dir),
    "random_state": 42,
    "framing": "Direct symbolic regression on body-fat percentage.",
    "selection_source": "validation split, not test set",
    "feature_names": feature_names,
    "representative_equations": {
        name: row.to_dict() for name, row in representatives.items()
    },
    "preprocessing_manifest": preprocessing_manifest,
}

with (out_dir / "pysr_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

log.info("Final results (sorted by validation RMSE):\n%s",
         summary.head(20).to_string(index=False))
log.info("Done.")
