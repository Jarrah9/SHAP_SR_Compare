import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Data ──────────────────────────────────────────────────────────────────────
col_names = ["m2_m1", "x2_x1", "y1_x1", "y2_x1", "z1_x1", "z2_x1", "target"]
UGravSample = pd.read_csv(
    "Feynman_without_units/I.9.18_sample_10000",
    sep=" ",
    index_col=False,
    header=None,
    names=col_names
)

UGravX = UGravSample.drop(columns=["target"])
UGravY = UGravSample["target"]

X_train, X_test, y_train, y_test = train_test_split(
    UGravX, UGravY, test_size=0.2, random_state=42
)

feature_names = ["m2/m1", "x2/x1", "y1/x1", "y2/x1", "z1/x1", "z2/x1"]

# ── Stage 1: Randomised search ────────────────────────────────────────────────
log.info("Stage 1: Randomised search")

param_dist = {
    "n_estimators":     randint(100, 600),
    "max_depth":        randint(3, 12),
    "learning_rate":    uniform(0.01, 0.3),
    "subsample":        uniform(0.6, 0.4),       # 0.6 to 1.0
    "colsample_bytree": uniform(0.6, 0.4),       # 0.6 to 1.0
    "min_child_weight": randint(1, 20),
    "reg_alpha":        uniform(0, 1),            # L1 regularisation
    "reg_lambda":       uniform(0.5, 4.5),        # L2 regularisation
}

random_search = RandomizedSearchCV(
    XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=1,           # XGBRegressor already uses n_jobs=-1 internally
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
log.info(f"Best params (randomised): {random_search.best_params_}")

# ── Stage 2: Grid search around best params ───────────────────────────────────
log.info("Stage 2: Grid search")

best = random_search.best_params_

def float_range(val, step, low, high, n=3):
    """Generate n evenly spaced values centred on val, clipped to [low, high]."""
    candidates = np.linspace(max(low, val - step), min(high, val + step), n)
    return sorted(set(np.round(candidates, 4)))

def int_range(val, step, low, high):
    return sorted(set([max(low, val - step), val, min(high, val + step)]))

param_grid = {
    "n_estimators":     int_range(best["n_estimators"], 50, 100, 700),
    "max_depth":        int_range(best["max_depth"], 1, 3, 12),
    "learning_rate":    float_range(best["learning_rate"], 0.05, 0.01, 0.3),
    "subsample":        float_range(best["subsample"], 0.1, 0.5, 1.0),
    "colsample_bytree": float_range(best["colsample_bytree"], 0.1, 0.5, 1.0),
    "min_child_weight": int_range(best["min_child_weight"], 2, 1, 20),
    "reg_alpha":        float_range(best["reg_alpha"], 0.1, 0, 1.0),
    "reg_lambda":       float_range(best["reg_lambda"], 0.5, 0.5, 5.0),
}

grid_search = GridSearchCV(
    XGBRegressor(
        objective="reg:squarederror",
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

# ── Final model ───────────────────────────────────────────────────────────────
log.info("Fitting final model")

XGBModel = XGBRegressor(
    **grid_search.best_params_,
    objective="reg:squarederror",
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
y_test_pred  = XGBModel.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test,  y_test_pred)
train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test,  y_test_pred)

log.info("Final model performance:")
log.info(f"  Train MSE : {train_mse:.6f}")
log.info(f"  Test  MSE : {test_mse:.6f}")
log.info(f"  Train R²  : {train_r2:.6f}")
log.info(f"  Test  R²  : {test_r2:.6f}")

# Save metrics
metrics = pd.DataFrame([{
    "train_mse": train_mse,
    "test_mse":  test_mse,
    "train_r2":  train_r2,
    "test_r2":   test_r2,
    **grid_search.best_params_
}])
metrics.to_csv("xgb_metrics.csv", index=False)

# ── Built-in feature importance ───────────────────────────────────────────────
log.info("Plotting XGBoost feature importance")

importances  = XGBModel.feature_importances_
indices      = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(8, 5))
plt.bar(sorted_names, sorted_importances)
plt.xlabel("Feature")
plt.ylabel("Importance score")
plt.title("XGBoost feature importance")
plt.tight_layout()
plt.savefig("xgb_feature_importance.png", dpi=150)
plt.close()

# ── SHAP ──────────────────────────────────────────────────────────────────────
log.info("Computing SHAP values")

explainer   = shap.TreeExplainer(XGBModel)
shap_values = explainer(X_test)
shap_values.feature_names = feature_names

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("xgb_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("xgb_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

log.info("Done. Outputs: xgb_metrics.csv, xgb_feature_importance.png,"
         " xgb_shap_beeswarm.png, xgb_shap_bar.png")
