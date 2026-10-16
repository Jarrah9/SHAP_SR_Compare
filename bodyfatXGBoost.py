import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import sklearn
import xgboost
import shap

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
out_dir = Path("outputs/bodyfat") / f"xgboost_{timestamp}"
out_dir.mkdir(parents=True, exist_ok=False)

X_train = pd.read_csv(prepared_dir / "X_train.csv", index_col=0)
X_test = pd.read_csv(prepared_dir / "X_test.csv", index_col=0)
y_train = pd.read_csv(prepared_dir / "y_train.csv", index_col=0)["BodyFat"].astype(float)
y_test = pd.read_csv(prepared_dir / "y_test.csv", index_col=0)["BodyFat"].astype(float)
raw_test = pd.read_csv(prepared_dir / "test_raw_split.csv", index_col=0)

with (prepared_dir / "preprocessing_manifest.json").open("r", encoding="utf-8") as file:
    preprocessing_manifest = json.load(file)

feature_names = list(X_train.columns)

log.info(f"Training set size: {X_train.shape}")
log.info(f"Test set size:     {X_test.shape}")
log.info(f"Output directory:  {out_dir}")

# --- Stage 1: Randomised search ---------------------------------------------
log.info("Stage 1: Randomised search")

param_dist = {
    "n_estimators": randint(100, 800),
    "max_depth": randint(2, 8),
    "learning_rate": uniform(0.01, 0.25),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 15),
    "gamma": uniform(0.0, 5.0),
    "reg_alpha": uniform(0.0, 2.0),
    "reg_lambda": uniform(0.5, 5.0),
}

random_search = RandomizedSearchCV(
    XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=1,  # XGBRegressor already uses n_jobs=-1 internally
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
log.info(f"Best params (randomised): {random_search.best_params_}")

# --- Stage 2: Grid search around best params --------------------------------
log.info("Stage 2: Grid search")

best = random_search.best_params_

def float_range(val, step, low, high, n=3):
    candidates = np.linspace(max(low, val - step), min(high, val + step), n)
    return sorted(set(np.round(candidates, 4)))

def int_range(val, step, low, high):
    return sorted(set([max(low, val - step), val, min(high, val + step)]))

param_grid = {
    "n_estimators": int_range(best["n_estimators"], 75, 50, 1000),
    "max_depth": int_range(best["max_depth"], 1, 2, 10),
    "learning_rate": float_range(best["learning_rate"], 0.04, 0.005, 0.35),
    "subsample": float_range(best["subsample"], 0.1, 0.5, 1.0),
    "colsample_bytree": float_range(best["colsample_bytree"], 0.1, 0.5, 1.0),
    "min_child_weight": int_range(best["min_child_weight"], 2, 1, 20),
    "gamma": float_range(best["gamma"], 0.75, 0.0, 8.0),
    "reg_alpha": float_range(best["reg_alpha"], 0.3, 0.0, 3.0),
    "reg_lambda": float_range(best["reg_lambda"], 0.75, 0.1, 8.0),
}

grid_search = GridSearchCV(
    XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)
log.info(f"Best params (grid): {grid_search.best_params_}")

# --- Final model -------------------------------------------------------------
log.info("Fitting final model")

XGBModel = XGBRegressor(
    **grid_search.best_params_,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
XGBModel.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

y_train_pred = XGBModel.predict(X_train)
y_test_pred = XGBModel.predict(X_test)

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

train_metrics = regression_metrics(y_train, y_train_pred)
test_metrics = regression_metrics(y_test, y_test_pred)

log.info("Final model performance:")
log.info(f"  Train MSE : {train_metrics['mse']:.6f}")
log.info(f"  Test  MSE : {test_metrics['mse']:.6f}")
log.info(f"  Train RMSE: {train_metrics['rmse']:.6f}")
log.info(f"  Test  RMSE: {test_metrics['rmse']:.6f}")
log.info(f"  Train R2  : {train_metrics['r2']:.6f}")
log.info(f"  Test  R2  : {test_metrics['r2']:.6f}")

# Save metrics and model
metrics = pd.DataFrame([{
    "model": "XGBoost",
    "cv_best_neg_mse": grid_search.best_score_,
    **{f"train_{k}": v for k, v in train_metrics.items()},
    **{f"test_{k}": v for k, v in test_metrics.items()},
    **grid_search.best_params_
}])
metrics.to_csv(out_dir / "xgb_metrics.csv", index=False)
joblib.dump(XGBModel, out_dir / "xgb_best.joblib")

test_predictions = raw_test.drop(columns=["BodyFat"], errors="ignore").copy()
test_predictions.insert(0, "source_row_index", test_predictions.index)
test_predictions["actual_bodyfat"] = y_test.to_numpy(dtype=float)
test_predictions["xgb_predicted_bodyfat"] = y_test_pred
test_predictions["xgb_residual"] = test_predictions["actual_bodyfat"] - test_predictions["xgb_predicted_bodyfat"]
test_predictions.to_csv(out_dir / "xgb_test_predictions.csv", index=False)

manifest = {
    "script": "bodyfatXGBoost.py",
    "prepared_dir": str(prepared_dir),
    "output_dir": str(out_dir),
    "random_state": 42,
    "scoring": "neg_mean_squared_error",
    "best_params": grid_search.best_params_,
    "feature_names": feature_names,
    "preprocessing_manifest": preprocessing_manifest,
    "versions": {
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
    },
}
with (out_dir / "xgb_manifest.json").open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

# --- Built-in feature importance --------------------------------------------
log.info("Plotting XGBoost feature importance")

importances = XGBModel.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

pd.DataFrame({
    "feature": sorted_names,
    "importance": sorted_importances,
}).to_csv(out_dir / "xgb_feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(sorted_names, sorted_importances)
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance score")
plt.title("XGBoost feature importance")
plt.tight_layout()
plt.savefig(out_dir / "xgb_feature_importance.png", dpi=150)
plt.close()

# --- SHAP --------------------------------------------------------------------
log.info("Computing SHAP values")

X_shap = X_test.sample(min(1000, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(XGBModel)
shap_values = explainer(X_shap)
shap_values.feature_names = feature_names

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(out_dir / "xgb_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(out_dir / "xgb_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

log.info("Done. Outputs: xgb_metrics.csv, xgb_best.joblib, xgb_feature_importance.csv,"
         " xgb_feature_importance.png, xgb_shap_beeswarm.png, xgb_shap_bar.png,"
         " xgb_test_predictions.csv, xgb_manifest.json")
