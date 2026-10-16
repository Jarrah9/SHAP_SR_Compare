"""Shared training for the standalone body-fat and Feynman shallow MLPs.

Only standard scikit-learn estimators are serialized; this module is not needed
to load the fitted model. Importing this module never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import parallel_config
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


PROJECT_ROOT = Path(__file__).resolve().parent
PARAM_PREFIX = "regressor__mlp__"
log = logging.getLogger(__name__)


@dataclass
class RegressionData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    prediction_rows: pd.DataFrame
    source_paths: list[Path]
    metadata: dict


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def add_training_arguments(parser: argparse.ArgumentParser, *, solver: str, widths: list[int]) -> None:
    parser.add_argument("--output-dir", type=Path, help="New directory; existing directories are refused.")
    parser.add_argument("--solver", choices=("lbfgs", "adam"), default=solver)
    parser.add_argument("--hidden-units", type=positive_int, nargs="+", default=widths,
                        help="Candidate widths, each representing ONE hidden layer.")
    parser.add_argument("--search-iterations", type=positive_int, default=60)
    parser.add_argument("--cv-folds", type=positive_int, default=5)
    parser.add_argument("--max-iter", type=positive_int, default=2000)
    parser.add_argument("--max-fun", type=positive_int, default=50000,
                        help="L-BFGS function evaluation limit.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Search and fitted-model seed; never changes the data split.")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[42, 1729, 2026],
                        help="Training-only CV sensitivity checks; these do not select the saved model.")
    parser.add_argument("--n-jobs", type=positive_int,
                        help="Parallel CV fits; defaults to allocated Slurm CPUs, or 1 locally.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip the small refinement search.")


def resolve_jobs(requested: int | None) -> int:
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if allocated < 1:
        raise ValueError("SLURM_CPUS_PER_TASK must be positive.")
    jobs = requested if requested is not None else allocated
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    limit = min(available, allocated) if "SLURM_CPUS_PER_TASK" in os.environ else available
    if jobs > limit:
        log.warning("Reducing parallel fits from %d to the available CPU limit %d", jobs, limit)
    return min(jobs, limit)


def validate_data(data: RegressionData, folds: int) -> None:
    if folds < 2 or len(data.X_train) // folds < 2:
        raise ValueError("CV needs at least two folds and at least two validation rows per fold.")
    if len(data.X_test) < 2 or data.X_train.shape[1] == 0:
        raise ValueError("Need features and at least two held-out test rows.")
    if not data.X_train.columns.is_unique or list(data.X_train.columns) != list(data.X_test.columns):
        raise ValueError("Train/test features must have identical, unique names in the same order.")
    for name, X, y in [("train", data.X_train, data.y_train), ("test", data.X_test, data.y_test)]:
        if not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError(f"{name}: feature/target row indices must match and be unique.")
        if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: features and targets must be finite numeric values.")
    if len(data.X_train.index.intersection(data.X_test.index)):
        raise ValueError("Training and test source row indices overlap.")
    if not data.prediction_rows.index.equals(data.X_test.index):
        raise ValueError("Prediction export rows are not aligned with the test set.")


def make_estimator(args: argparse.Namespace) -> TransformedTargetRegressor:
    # Both scalers are refitted inside each CV training fold. The outer wrapper
    # reverses target scaling, so CV scores and predict() use original units.
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                solver=args.solver, random_state=args.seed, max_iter=args.max_iter,
                max_fun=args.max_fun, early_stopping=args.solver == "adam",
                validation_fraction=0.1, n_iter_no_change=30, tol=1e-5,
            )),
        ]),
        transformer=StandardScaler(),
    )


def search_space(args: argparse.Namespace) -> dict:
    space = {
        PARAM_PREFIX + "hidden_layer_sizes": [(width,) for width in sorted(set(args.hidden_units))],
        PARAM_PREFIX + "activation": ["relu", "tanh"],
        PARAM_PREFIX + "alpha": loguniform(1e-6, 10.0),
    }
    if args.solver == "adam":
        space[PARAM_PREFIX + "learning_rate_init"] = loguniform(1e-4, 1e-2)
    return space


def refinement_grid(best: dict, args: argparse.Namespace) -> dict:
    grid = {
        PARAM_PREFIX + "hidden_layer_sizes": [(width,) for width in sorted(set(args.hidden_units))],
        PARAM_PREFIX + "activation": [best[PARAM_PREFIX + "activation"]],
        PARAM_PREFIX + "alpha": [best[PARAM_PREFIX + "alpha"] * factor for factor in (0.3, 1.0, 3.0)],
    }
    if args.solver == "adam":
        grid[PARAM_PREFIX + "learning_rate_init"] = [
            best[PARAM_PREFIX + "learning_rate_init"] * factor for factor in (0.5, 1.0, 2.0)
        ]
    return grid


def regression_metrics(y_true, prediction) -> dict:
    mse = float(mean_squared_error(y_true, prediction))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, prediction)),
            "r2": float(r2_score(y_true, prediction))}


def write_json(path: Path, value: dict) -> None:
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported JSON value: {type(obj)}")
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def train_model(data: RegressionData, args: argparse.Namespace, *, dataset: str, script: Path,
                actual_column: str, prediction_column: str) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    validate_data(data, args.cv_folds)
    if any(seed < 0 or seed >= 2**32 for seed in [args.seed, *args.validation_seeds]):
        raise ValueError("Random seeds must be between 0 and 2**32 - 1.")
    jobs = resolve_jobs(args.n_jobs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = (args.output_dir or PROJECT_ROOT / "outputs" / dataset / f"mlp_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "status": "running", "model_family": "shallow MLP regressor", "is_mock": False,
        "script": script.name, "dataset": dataset, "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir), "arguments": vars(args), "random_state": args.seed,
        "feature_names": list(data.X_train.columns), "n_train": len(data.X_train), "n_test": len(data.X_test),
        "data": data.metadata, "n_jobs": jobs, "threads_per_fit": 1,
        "cv": {"folds": args.cv_folds, "shuffle": False, "scoring": "neg_mean_squared_error",
               "units": "original target units", "selection": "training folds only"},
        "preprocessing": "Fold-fitted StandardScaler for X and y; predict() returns original target units.",
        "seed_validation": "Fixed best hyperparameters; training-only CV diagnostics, not independent performance estimates or seed selection.",
        "final_model_selection": "Best CV hyperparameters fitted with the requested --seed; test set never selects settings or stopping.",
        "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [*data.source_paths, script, Path(__file__)]},
        "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    write_json(out_dir / "mlp_manifest.json", manifest)
    try:
        log.info("%s: train=%s test=%s, %d parallel CV fits, output=%s", dataset,
                 data.X_train.shape, data.X_test.shape, jobs, out_dir)
        estimator = make_estimator(args)
        folds = list(KFold(n_splits=args.cv_folds, shuffle=False).split(data.X_train))
        assignment = np.empty(len(data.X_train), dtype=int)
        for fold, (_, validation_positions) in enumerate(folds):
            assignment[validation_positions] = fold
        pd.DataFrame({"source_row_index": data.X_train.index, "validation_fold": assignment}).to_csv(
            out_dir / "mlp_cv_folds.csv", index=False)

        # Parallelise independent CV fits, never BLAS threads within each fit.
        with parallel_config(backend="loky", inner_max_num_threads=1), threadpool_limits(limits=1):
            random_search = RandomizedSearchCV(
                estimator, search_space(args), n_iter=args.search_iterations, cv=folds,
                scoring="neg_mean_squared_error", n_jobs=jobs, pre_dispatch=jobs,
                random_state=args.seed, refit=False, error_score="raise", verbose=1,
                return_train_score=True,
            )
            log.info("Stage 1: %d randomised candidates", args.search_iterations)
            random_search.fit(data.X_train, data.y_train)
            pd.DataFrame(random_search.cv_results_).to_csv(out_dir / "mlp_random_search.csv", index=False)
            selected = random_search
            if not args.skip_grid:
                log.info("Stage 2: refine width, regularisation and (for Adam) learning rate")
                selected = GridSearchCV(
                    estimator, refinement_grid(random_search.best_params_, args), cv=folds,
                    scoring="neg_mean_squared_error", n_jobs=jobs, pre_dispatch=jobs,
                    refit=False, error_score="raise", verbose=1, return_train_score=True,
                )
                selected.fit(data.X_train, data.y_train)
                pd.DataFrame(selected.cv_results_).to_csv(out_dir / "mlp_grid_search.csv", index=False)

            best_params = selected.best_params_
            log.info("Selected CV MSE: %.8g; parameters: %s", -selected.best_score_, best_params)
            # Save the primary model before optional seed diagnostics, so a
            # wall-time limit during diagnostics does not lose the trained model.
            model = clone(estimator).set_params(**best_params)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(data.X_train, data.y_train)
            convergence = [str(item.message) for item in recorded if issubclass(item.category, ConvergenceWarning)]
            for item in recorded:
                log.warning("Final fit %s: %s", item.category.__name__, item.message)
            train_prediction = model.predict(data.X_train)
            test_prediction = model.predict(data.X_test)
            if not np.isfinite(train_prediction).all() or not np.isfinite(test_prediction).all():
                raise ValueError("The fitted model produced non-finite predictions.")
            joblib.dump(model, out_dir / "mlp_best.joblib")
            fitted_mlp = model.regressor_.named_steps["mlp"]
            metrics = {
                "model": "Shallow MLP", "cv_best_neg_mse": float(selected.best_score_),
                **{f"train_{key}": value for key, value in regression_metrics(data.y_train, train_prediction).items()},
                **{f"test_{key}": value for key, value in regression_metrics(data.y_test, test_prediction).items()},
                "solver": args.solver, "random_state": args.seed, "n_iter": fitted_mlp.n_iter_,
                "convergence_warning": bool(convergence),
                **{key.removeprefix(PARAM_PREFIX): value for key, value in best_params.items()},
            }
            pd.DataFrame([metrics]).to_csv(out_dir / "mlp_metrics.csv", index=False)
            predictions = data.prediction_rows.copy()
            predictions.insert(0, "source_row_index", predictions.index)
            predictions[actual_column] = data.y_test.to_numpy(dtype=float)
            predictions[prediction_column] = test_prediction
            predictions["mlp_residual"] = data.y_test.to_numpy(dtype=float) - test_prediction
            predictions.to_csv(out_dir / "mlp_test_predictions.csv", index=False)
            if hasattr(fitted_mlp, "loss_curve_"):
                history = pd.DataFrame({"iteration": np.arange(1, len(fitted_mlp.loss_curve_) + 1),
                                        "scaled_training_loss": fitted_mlp.loss_curve_})
                history["internal_validation_r2"] = fitted_mlp.validation_scores_
                history.to_csv(out_dir / "mlp_loss_curve.csv", index=False)

            manifest.update({"status": "model_saved_seed_validation_pending", "best_params": best_params,
                             "cv_best_neg_mse": float(selected.best_score_), "model_artifact": "mlp_best.joblib",
                             "final_fit": {"n_iter": fitted_mlp.n_iter_, "scaled_loss": float(fitted_mlp.loss_),
                                           "convergence_warnings": convergence,
                                           "parameter_count": sum(a.size for a in [*fitted_mlp.coefs_, *fitted_mlp.intercepts_])}})
            write_json(out_dir / "mlp_manifest.json", manifest)

            seed_rows = []
            for seed in dict.fromkeys(args.validation_seeds):
                log.info("Training-only seed sensitivity CV: seed=%d", seed)
                candidate = clone(model).set_params(**{PARAM_PREFIX + "random_state": seed})
                scores = cross_val_score(candidate, data.X_train, data.y_train, cv=folds,
                                         scoring="neg_mean_squared_error", n_jobs=jobs,
                                         pre_dispatch=jobs, error_score="raise")
                for fold, score in enumerate(scores):
                    seed_rows.append({"seed": seed, "fold": fold, "validation_mse": float(-score)})
                pd.DataFrame(seed_rows).to_csv(out_dir / "mlp_seed_validation.csv", index=False)

        manifest.update({"status": "completed", "elapsed_seconds": time.perf_counter() - started})
        write_json(out_dir / "mlp_manifest.json", manifest)
        log.info("Saved %s; test RMSE=%.6g, R2=%.6g", out_dir / "mlp_best.joblib",
                 metrics["test_rmse"], metrics["test_r2"])
        return out_dir
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                         "elapsed_seconds": time.perf_counter() - started})
        write_json(out_dir / "mlp_manifest.json", manifest)
        raise
