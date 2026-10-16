import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
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
    "n_estimators":     randint(100, 500),
    "max_depth":        [5, 8, 10, 15, 20, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_features":     [1.0, "sqrt", "log2"],
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, oob_score=True),
    param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
log.info(f"Best params (randomised): {random_search.best_params_}")

# ── Stage 2: Grid search around best params ───────────────────────────────────
log.info("Stage 2: Grid search")

best = random_search.best_params_

# Build a grid centred on the randomised search result
def int_range(val, step, low, high):
    return sorted(set([max(low, val - step), val, min(high, val + step)]))

param_grid = {
    "n_estimators":     int_range(best["n_estimators"], 50, 100, 600),
    "max_depth":        [best["max_depth"]],
    "min_samples_split": int_range(best["min_samples_split"], 2, 2, 20),
    "min_samples_leaf": int_range(best["min_samples_leaf"], 2, 1, 20),
    "max_features":     [best["max_features"]],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, oob_score=True),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
log.info(f"Best params (grid): {grid_search.best_params_}")

# ── Final model ───────────────────────────────────────────────────────────────
log.info("Fitting final model")

RFModel = RandomForestRegressor(
    **grid_search.best_params_,
    random_state=42,
    oob_score=True
)
RFModel.fit(X_train, y_train)

y_train_pred = RFModel.predict(X_train)
y_test_pred  = RFModel.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse  = mean_squared_error(y_test,  y_test_pred)
train_r2  = r2_score(y_train, y_train_pred)
test_r2   = r2_score(y_test,  y_test_pred)
oob_score = RFModel.oob_score_

log.info("Final model performance:")
log.info(f"  Train MSE : {train_mse:.6f}")
log.info(f"  Test  MSE : {test_mse:.6f}")
log.info(f"  Train R²  : {train_r2:.6f}")
log.info(f"  Test  R²  : {test_r2:.6f}")
log.info(f"  OOB score : {oob_score:.6f}")

# Save metrics
metrics = pd.DataFrame([{
    "train_mse": train_mse,
    "test_mse":  test_mse,
    "train_r2":  train_r2,
    "test_r2":   test_r2,
    "oob_score": oob_score,
    **grid_search.best_params_
}])
metrics.to_csv("rf_metrics.csv", index=False)

# ── MDI feature importance ────────────────────────────────────────────────────
log.info("Plotting MDI feature importance")

importances = RFModel.feature_importances_
indices     = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(8, 5))
plt.bar(sorted_names, sorted_importances)
plt.xlabel("Feature")
plt.ylabel("Mean decrease in impurity")
plt.title("RF feature importance (MDI)")
plt.tight_layout()
plt.savefig("rf_mdi_importance.png", dpi=150)
plt.close()

# ── SHAP ──────────────────────────────────────────────────────────────────────
log.info("Computing SHAP values")

explainer   = shap.TreeExplainer(RFModel)
shap_values = explainer(X_test)
shap_values.feature_names = feature_names

plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("rf_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig("rf_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

log.info("Done. Outputs: rf_metrics.csv, rf_mdi_importance.png,"
         " rf_shap_beeswarm.png, rf_shap_bar.png")
