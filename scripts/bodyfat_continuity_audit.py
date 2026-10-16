"""Reproducible body-fat continuity audit used by ``Full_Analysis.ipynb``.

The audit preserves the original perturbation stream while replacing the legacy
27-evaluation permutation estimates with the validated exact SHAP setup.  It
writes new files only inside the reliability-audit directory; historical noise
and SHAP artefacts are read but never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, r2_score


PROTOCOL_VERSION = "1.0.0"
RAW_FEATURES = [
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
FEATURES = RAW_FEATURES + ["ACratio", "HTratio"]
TREE_MODELS = ("Random Forest", "XGBoost")
NOISE_LEVELS = (0.025, 0.05)
EXPLANATION_REPEATS = 10
PREDICTION_REPEATS = 100
NOISE_SEED = 42
PRIMARY_RESPONSE_FRACTION = 0.05
RESPONSE_FRACTIONS = (0.025, 0.05, 0.10)
EXACT_COALITIONS = 2 ** len(RAW_FEATURES)
BACKGROUND_SIZE = 50


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepared_frame(raw_rows: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    raw = pd.DataFrame(raw_rows, columns=RAW_FEATURES).copy()
    values = raw.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Body-fat raw inputs must all be finite.")
    if (raw[["Chest", "Thigh"]] == 0).any().any():
        raise ValueError("Chest and Thigh must be non-zero before deriving ratios.")
    prepared = raw.copy()
    prepared["ACratio"] = prepared["Abdomen"] / prepared["Chest"]
    prepared["HTratio"] = prepared["Hip"] / prepared["Thigh"]
    if not np.isfinite(prepared.to_numpy(dtype=float)).all():
        raise ValueError("Derived body-fat ratios must be finite.")
    return prepared[FEATURES]


def _prediction_function(model):
    def predict(raw_rows):
        return np.asarray(model.predict(_prepared_frame(raw_rows)), dtype=float).ravel()

    return predict


def _pysr_prediction(raw_rows: pd.DataFrame | np.ndarray) -> np.ndarray:
    raw = pd.DataFrame(raw_rows, columns=RAW_FEATURES)
    numerator = raw["Abdomen"].to_numpy() ** 2 - 272.58493 * raw["Wrist"].to_numpy()
    denominator = (
        raw["Weight"].to_numpy()
        + raw["Neck"].to_numpy()
        - raw["Biceps"].to_numpy()
    )
    return numerator / denominator


def _row_metrics(reference: np.ndarray, comparison: np.ndarray) -> tuple[np.ndarray, ...]:
    """Cosine plus magnitude-sensitive mean absolute attribution distance.

    Cosine is one when both vectors are zero and zero when exactly one is zero.
    The distance is mean(abs(comparison-reference)); it needs no division, so two
    zero vectors have distance zero and one zero vector retains the magnitude of
    the non-zero vector.
    """

    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    reference_norm = np.linalg.norm(reference, axis=1)
    comparison_norm = np.linalg.norm(comparison, axis=1)
    denominator = reference_norm * comparison_norm
    cosine = np.zeros(reference.shape[0], dtype=float)
    nonzero = denominator > np.finfo(float).eps
    cosine[nonzero] = np.sum(reference[nonzero] * comparison[nonzero], axis=1) / denominator[nonzero]
    both_zero = (reference_norm <= np.finfo(float).eps) & (
        comparison_norm <= np.finfo(float).eps
    )
    cosine[both_zero] = 1.0
    distance = np.mean(np.abs(comparison - reference), axis=1)
    return np.clip(cosine, -1.0, 1.0), distance, both_zero, (~both_zero) & (
        (reference_norm <= np.finfo(float).eps)
        | (comparison_norm <= np.finfo(float).eps)
    )


def _empirical_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return np.nan, np.nan
    return float(finite.quantile(0.025)), float(finite.quantile(0.975))


def _load_models(project_root: Path):
    specifications = {
        "Random Forest": project_root
        / "hpc_results/bodyfat/random_forest_20260719_232404/rf_best.joblib",
        "XGBoost": project_root
        / "hpc_results/bodyfat/xgboost_20260719_232404/xgb_best.joblib",
    }
    models = {name: joblib.load(path) for name, path in specifications.items()}
    # Keep single-thread inference for the saved prediction-stream and exact
    # zero-noise checks.  The costly SHAP-only stage switches to parallel
    # inference after those checks have passed.
    for model in models.values():
        model.set_params(n_jobs=1)
    return models, specifications


def _checkpoint_path(output_dir: Path, method: str, noise_level: float) -> Path:
    key = method.lower().replace(" ", "_")
    level = str(100 * noise_level).replace(".", "p")
    return output_dir / f"continuity_exact_{key}_{level}_percent.npz"


def _calculate_exact_explanations(
    *,
    output_dir: Path,
    method: str,
    predict,
    background: pd.DataFrame,
    perturbed: np.ndarray,
    subject_ids: np.ndarray,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _checkpoint_path(output_dir, method, noise_level)
    expected_shape = (EXPLANATION_REPEATS, len(subject_ids), len(RAW_FEATURES))
    values = np.full(expected_shape, np.nan, dtype=float)
    base_values = np.full(expected_shape[:2], np.nan, dtype=float)
    predictions = np.full(expected_shape[:2], np.nan, dtype=float)
    completed = np.zeros(EXPLANATION_REPEATS, dtype=bool)
    if path.exists():
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["subject_ids"], subject_ids)
            np.testing.assert_array_equal(archive["feature_names"], RAW_FEATURES)
            np.testing.assert_allclose(archive["noise_level"], noise_level, rtol=0, atol=0)
            values = archive["values"].copy()
            base_values = archive["base_values"].copy()
            predictions = archive["predictions"].copy()
            completed = archive["completed"].astype(bool)
        if values.shape != expected_shape:
            raise ValueError(f"Unexpected checkpoint shape in {path}: {values.shape}")

    explainer = shap.Explainer(
        predict,
        shap.maskers.Independent(background, max_samples=BACKGROUND_SIZE),
        algorithm="exact",
        feature_names=RAW_FEATURES,
    )
    for repeat in range(EXPLANATION_REPEATS):
        if completed[repeat]:
            continue
        rows = pd.DataFrame(perturbed[repeat], columns=RAW_FEATURES)
        result = explainer(
            rows,
            max_evals=EXACT_COALITIONS,
            batch_size=32768,
            silent=True,
        )
        prediction = predict(rows)
        np.testing.assert_allclose(
            result.values.sum(axis=1) + result.base_values,
            prediction,
            atol=1e-9,
            rtol=1e-9,
        )
        values[repeat] = result.values
        base_values[repeat] = result.base_values
        predictions[repeat] = prediction
        completed[repeat] = True
        np.savez_compressed(
            path,
            values=values,
            base_values=base_values,
            predictions=predictions,
            completed=completed,
            subject_ids=subject_ids,
            feature_names=np.asarray(RAW_FEATURES),
            noise_level=noise_level,
        )
        print(
            f"Exact continuity SHAP: {method}, {100 * noise_level:g}% noise, "
            f"repeat {repeat + 1}/{EXPLANATION_REPEATS}",
            flush=True,
        )
    return values, base_values, predictions


def run_bodyfat_continuity(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root).resolve()
    output_dir = project_root / "outputs/bodyfat/analysis/reliability_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = project_root / "hpc_results/bodyfat/prepared/train_raw_split.csv"
    test_path = project_root / "hpc_results/bodyfat/prepared/test_raw_split.csv"
    y_train_path = project_root / "hpc_results/bodyfat/prepared/y_train.csv"
    y_test_path = project_root / "hpc_results/bodyfat/prepared/y_test.csv"
    clean_shap_path = output_dir / "shap_values.npz"
    shap_manifest_path = output_dir / "shap_manifest.json"
    legacy_repeats_path = project_root / "outputs/bodyfat/analysis/noise_sensitivity_repeats.csv"

    train_rows = pd.read_csv(train_path)
    test_rows = pd.read_csv(test_path)
    x_train = train_rows[RAW_FEATURES].astype(float)
    x_test = test_rows[RAW_FEATURES].astype(float)
    y_train = pd.read_csv(y_train_path).iloc[:, -1].to_numpy(dtype=float)
    y_test = pd.read_csv(y_test_path).iloc[:, -1].to_numpy(dtype=float)
    subject_ids_all = test_rows["source_row_index"].to_numpy(dtype=int)
    training_sd = x_train.std(axis=0, ddof=0).to_numpy()
    response_sd = float(np.std(y_train, ddof=0))
    thresholds = {fraction: fraction * response_sd for fraction in RESPONSE_FRACTIONS}

    shap_manifest = json.loads(shap_manifest_path.read_text(encoding="utf-8"))
    subject_ids = np.asarray(shap_manifest["subset_ids"], dtype=int)
    positions = np.asarray(
        [int(np.flatnonzero(subject_ids_all == subject)[0]) for subject in subject_ids]
    )
    audit_rows = x_test.iloc[positions].reset_index(drop=True)
    background = x_train.sample(n=BACKGROUND_SIZE, random_state=42)
    background_ids = train_rows.loc[background.index, "source_row_index"].to_numpy(dtype=int)
    np.testing.assert_array_equal(background_ids, shap_manifest["background_ids"])

    with np.load(clean_shap_path) as archive:
        np.testing.assert_array_equal(archive["source_row_index"], subject_ids_all)
        np.testing.assert_array_equal(archive["background_source_row_index"], background_ids)
        np.testing.assert_allclose(archive["background_values"], background.to_numpy(), rtol=0, atol=0)
        clean_values = {
            method: archive[method.lower().replace(" ", "_")][positions].copy()
            for method in TREE_MODELS
        }
        # These are two genuinely separate exact calculations made by fresh
        # explainers, not one array reused as a zero-noise branch.
        independent_repeatability = {}
        for method in TREE_MODELS:
            key = method.lower().replace(" ", "_")
            replicates = archive[key + "_replicates"][:, positions].copy()
            independent_repeatability[method] = {
                "maximum_absolute_difference": float(
                    np.max(np.abs(replicates[0] - replicates[1]))
                ),
                "arrays_exactly_equal": bool(np.array_equal(replicates[0], replicates[1])),
            }
            np.testing.assert_allclose(clean_values[method], replicates[0], rtol=0, atol=0)

    models, model_paths = _load_models(project_root)
    predictors = {name: _prediction_function(model) for name, model in models.items()}
    clean_prediction = {name: predict(x_test) for name, predict in predictors.items()}
    clean_prediction["PySR"] = _pysr_prediction(x_test)

    # Recreate, rather than invent, the original paired 100-draw stream.  The
    # first ten draws are used for exact SHAP; all 100 retain the saved
    # prediction-sensitivity and equation-domain audit.
    noise_draws = np.random.default_rng(NOISE_SEED).normal(
        size=(PREDICTION_REPEATS, len(x_test), len(RAW_FEATURES))
    )
    perturbed_full = {
        level: x_test.to_numpy()[None, :, :]
        + level * noise_draws * training_sd[None, None, :]
        for level in NOISE_LEVELS
    }
    perturbed_audit = {
        level: values[:EXPLANATION_REPEATS, positions, :]
        for level, values in perturbed_full.items()
    }

    # Zero-noise construction must preserve both prepared inputs and predictions.
    zero_raw = x_test.copy()
    np.testing.assert_allclose(
        _prepared_frame(zero_raw).to_numpy(),
        _prepared_frame(x_test).to_numpy(),
        rtol=0,
        atol=0,
    )
    zero_rows = []
    for method, predict in {**predictors, "PySR": _pysr_prediction}.items():
        zero_change = np.abs(predict(zero_raw) - clean_prediction[method])
        np.testing.assert_allclose(zero_change, 0, rtol=0, atol=0)
        zero_rows.append(
            {
                "Method": method,
                "Rows": len(x_test),
                "Maximum zero-noise prediction change": float(zero_change.max()),
                "Exact zero-noise check": bool(np.all(zero_change == 0)),
            }
        )
    zero_frame = pd.DataFrame(zero_rows)
    zero_frame.to_csv(output_dir / "continuity_zero_noise_prediction_checks.csv", index=False)

    # Validate the entire domain covered by the 100-repeat saved prediction
    # experiment, including the native equation's only denominator.
    input_rows = []
    training_min = x_train.min(axis=0).to_numpy()
    training_max = x_train.max(axis=0).to_numpy()
    clean_outside = (x_test.to_numpy() < training_min) | (x_test.to_numpy() > training_max)
    for level, array in perturbed_full.items():
        flat = array.reshape(-1, len(RAW_FEATURES))
        prepared = _prepared_frame(flat)
        outside = (array < training_min[None, None, :]) | (
            array > training_max[None, None, :]
        )
        newly_outside = outside & ~clean_outside[None, :, :]
        denominator = flat[:, RAW_FEATURES.index("Weight")] + flat[:, RAW_FEATURES.index("Neck")] - flat[:, RAW_FEATURES.index("Biceps")]
        numerator = flat[:, RAW_FEATURES.index("Abdomen")] ** 2 - 272.58493 * flat[:, RAW_FEATURES.index("Wrist")]
        prediction = numerator / denominator
        denominator_tolerance = 1e-10 * max(1.0, float(np.median(np.abs(denominator))))
        gradient_norm = np.sqrt(
            3 * (numerator / denominator**2) ** 2
            + (2 * flat[:, RAW_FEATURES.index("Abdomen")] / denominator) ** 2
            + (272.58493 / denominator) ** 2
        )
        input_rows.append(
            {
                "Noise (% training SD)": 100 * level,
                "Evaluated subject-perturbation rows": len(flat),
                "Finite raw rows": int(np.isfinite(flat).all(axis=1).sum()),
                "Non-positive raw cells": int((flat <= 0).sum()),
                "Finite prepared rows": int(np.isfinite(prepared.to_numpy()).all(axis=1).sum()),
                "Minimum Chest": float(prepared["Chest"].min()),
                "Minimum Thigh": float(prepared["Thigh"].min()),
                "ACratio minimum": float(prepared["ACratio"].min()),
                "ACratio maximum": float(prepared["ACratio"].max()),
                "HTratio minimum": float(prepared["HTratio"].min()),
                "HTratio maximum": float(prepared["HTratio"].max()),
                "Rows with any training-range violation": int(outside.any(axis=2).sum()),
                "Newly out-of-training-range cells": int(newly_outside.sum()),
                "PySR minimum absolute denominator": float(np.min(np.abs(denominator))),
                "PySR numerical singularity tolerance": denominator_tolerance,
                "PySR near-zero denominators": int((np.abs(denominator) <= denominator_tolerance).sum()),
                "Finite PySR predictions": int(np.isfinite(prediction).sum()),
                "Maximum absolute PySR prediction": float(np.max(np.abs(prediction))),
                "Maximum native-equation gradient norm": float(np.max(gradient_norm)),
            }
        )
    input_frame = pd.DataFrame(input_rows)
    if not (
        (input_frame["Finite raw rows"] == input_frame["Evaluated subject-perturbation rows"]).all()
        and (input_frame["Finite prepared rows"] == input_frame["Evaluated subject-perturbation rows"]).all()
        and (input_frame["Non-positive raw cells"] == 0).all()
        and (input_frame["PySR near-zero denominators"] == 0).all()
        and (input_frame["Finite PySR predictions"] == input_frame["Evaluated subject-perturbation rows"]).all()
    ):
        raise RuntimeError("A body-fat perturbation or native PySR equation failed its domain audit.")
    input_frame.to_csv(output_dir / "continuity_input_validity.csv", index=False)

    # Verify the regenerated stream against every saved prediction-repeat row.
    legacy = pd.read_csv(legacy_repeats_path)
    prediction_check_rows = []
    equation_pair_rows = []
    for level, array in perturbed_full.items():
        for repeat in range(PREDICTION_REPEATS):
            raw_rows = pd.DataFrame(array[repeat], columns=RAW_FEATURES)
            for method, predict in {**predictors, "PySR": _pysr_prediction}.items():
                prediction = predict(raw_rows)
                change = np.abs(prediction - clean_prediction[method])
                stable = change <= thresholds[PRIMARY_RESPONSE_FRACTION]
                regenerated = {
                    "R2": r2_score(y_test, prediction),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
                    "Mean absolute prediction change": float(change.mean()),
                    "Response-stable rows (%)": float(100 * stable.mean()),
                }
                saved = legacy[
                    (legacy["Method"] == method)
                    & (legacy["Noise repeat"] == repeat)
                    & np.isclose(legacy["Noise (% training SD)"], 100 * level)
                ]
                if len(saved) != 1:
                    raise RuntimeError("Could not identify one saved prediction-sensitivity row.")
                for metric, actual in regenerated.items():
                    prediction_check_rows.append(
                        {
                            "Method": method,
                            "Noise (% training SD)": 100 * level,
                            "Noise repeat": repeat,
                            "Metric": metric,
                            "Regenerated": actual,
                            "Saved": float(saved.iloc[0][metric]),
                            "Absolute difference": abs(actual - float(saved.iloc[0][metric])),
                        }
                    )

                if method == "PySR" and repeat < EXPLANATION_REPEATS:
                    for local_row, position in enumerate(positions):
                        equation_pair_rows.append(
                            {
                                "Method": "PySR",
                                "Noise (% training SD)": 100 * level,
                                "Noise repeat": repeat,
                                "source_row_index": int(subject_ids_all[position]),
                                "Absolute prediction change": float(change[position]),
                                "Response stable at primary threshold": bool(stable[position]),
                            }
                        )
    prediction_check = pd.DataFrame(prediction_check_rows)
    prediction_check.to_csv(output_dir / "continuity_prediction_reuse_check.csv", index=False)
    if prediction_check["Absolute difference"].max() > 2e-6:
        raise RuntimeError("Regenerated prediction sensitivity does not match the saved stream.")

    pair_rows = []
    for method in TREE_MODELS:
        # Inference scheduling only; clean/noisy additivity and the independently
        # saved exact clean values still constrain the numerical result.
        models[method].set_params(n_jobs=-1)
        predict = predictors[method]
        clean_subset_prediction = clean_prediction[method][positions]
        for level in NOISE_LEVELS:
            values, base_values, noisy_predictions = _calculate_exact_explanations(
                output_dir=output_dir,
                method=method,
                predict=predict,
                background=background,
                perturbed=perturbed_audit[level],
                subject_ids=subject_ids,
                noise_level=level,
            )
            for repeat in range(EXPLANATION_REPEATS):
                cosine, distance, both_zero, one_zero = _row_metrics(
                    clean_values[method], values[repeat]
                )
                prediction_change = np.abs(
                    noisy_predictions[repeat] - clean_subset_prediction
                )
                for row, subject in enumerate(subject_ids):
                    record = {
                        "Method": method,
                        "Noise (% training SD)": 100 * level,
                        "Noise repeat": repeat,
                        "source_row_index": int(subject),
                        "Absolute prediction change": float(prediction_change[row]),
                        "Cosine similarity": float(cosine[row]),
                        "Mean absolute attribution distance": float(distance[row]),
                        "Both attribution vectors zero": bool(both_zero[row]),
                        "Exactly one attribution vector zero": bool(one_zero[row]),
                    }
                    for fraction, threshold in thresholds.items():
                        record[f"Stable at {100 * fraction:g}% response-SD threshold"] = bool(
                            prediction_change[row] <= threshold
                        )
                    pair_rows.append(record)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output_dir / "continuity_pairs.csv", index=False)

    repeat_rows = []
    primary_column = f"Stable at {100 * PRIMARY_RESPONSE_FRACTION:g}% response-SD threshold"
    for (method, level, repeat), group in pairs.groupby(
        ["Method", "Noise (% training SD)", "Noise repeat"], sort=False
    ):
        retained = group[group[primary_column]]
        repeat_rows.append(
            {
                "Method": method,
                "Noise (% training SD)": level,
                "Noise repeat": repeat,
                "Candidate subjects": len(group),
                "Retained subjects": len(retained),
                "Retained proportion": len(retained) / len(group),
                "Mean cosine similarity": retained["Cosine similarity"].mean(),
                "Mean absolute attribution distance": retained[
                    "Mean absolute attribution distance"
                ].mean(),
            }
        )
    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_frame.to_csv(output_dir / "continuity_repeat_summary.csv", index=False)

    summary_rows = []
    for (method, level), group in pairs.groupby(
        ["Method", "Noise (% training SD)"], sort=False
    ):
        retained = group[group[primary_column]]
        repeat_group = repeat_frame[
            (repeat_frame["Method"] == method)
            & (repeat_frame["Noise (% training SD)"] == level)
        ]
        cosine_low, cosine_high = _empirical_limits(
            repeat_group["Mean cosine similarity"]
        )
        distance_low, distance_high = _empirical_limits(
            repeat_group["Mean absolute attribution distance"]
        )
        summary_rows.append(
            {
                "Method": method,
                "Noise (% training SD)": level,
                "Explanation subjects": len(subject_ids),
                "Perturbation repeats": EXPLANATION_REPEATS,
                "Candidate explanation pairs": len(group),
                "Retained explanation pairs": len(retained),
                "Retained proportion": len(retained) / len(group),
                "Response-stability threshold (percentage points)": thresholds[
                    PRIMARY_RESPONSE_FRACTION
                ],
                "Mean cosine similarity": retained["Cosine similarity"].mean(),
                "Median cosine similarity": retained["Cosine similarity"].median(),
                "Repeat-mean cosine 2.5%": cosine_low,
                "Repeat-mean cosine 97.5%": cosine_high,
                "Mean absolute attribution distance": retained[
                    "Mean absolute attribution distance"
                ].mean(),
                "Median absolute attribution distance": retained[
                    "Mean absolute attribution distance"
                ].median(),
                "Repeat-mean attribution distance 2.5%": distance_low,
                "Repeat-mean attribution distance 97.5%": distance_high,
                "Both-zero vector pairs": int(retained["Both attribution vectors zero"].sum()),
                "One-zero vector pairs": int(
                    retained["Exactly one attribution vector zero"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "continuity_summary.csv", index=False)

    sensitivity_rows = []
    for fraction, threshold in thresholds.items():
        column = f"Stable at {100 * fraction:g}% response-SD threshold"
        for (method, level), group in pairs.groupby(
            ["Method", "Noise (% training SD)"], sort=False
        ):
            retained = group[group[column]]
            sensitivity_rows.append(
                {
                    "Method": method,
                    "Noise (% training SD)": level,
                    "Threshold (% training response SD)": 100 * fraction,
                    "Threshold (percentage points)": threshold,
                    "Candidate explanation pairs": len(group),
                    "Retained explanation pairs": len(retained),
                    "Retained proportion": len(retained) / len(group),
                    "Mean cosine similarity": retained["Cosine similarity"].mean(),
                    "Mean absolute attribution distance": retained[
                        "Mean absolute attribution distance"
                    ].mean(),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "continuity_threshold_sensitivity.csv", index=False)

    equation_pairs = pd.DataFrame(equation_pair_rows)
    equation_pairs.to_csv(output_dir / "native_equation_sensitivity_pairs.csv", index=False)
    equation_summary_rows = []
    for level, group in equation_pairs.groupby("Noise (% training SD)", sort=False):
        repeat_means = group.groupby("Noise repeat")["Absolute prediction change"].mean()
        low, high = _empirical_limits(repeat_means)
        equation_summary_rows.append(
            {
                "Method": "PySR",
                "Noise (% training SD)": level,
                "Equation subjects": len(subject_ids),
                "Perturbation repeats": EXPLANATION_REPEATS,
                "Candidate equation-input pairs": len(group),
                "Response-stable pairs": int(
                    group["Response stable at primary threshold"].sum()
                ),
                "Response-stable proportion": group[
                    "Response stable at primary threshold"
                ].mean(),
                "Mean absolute prediction change": group[
                    "Absolute prediction change"
                ].mean(),
                "Repeat-mean prediction change 2.5%": low,
                "Repeat-mean prediction change 97.5%": high,
                "Native equation structurally unchanged": True,
            }
        )
    equation_summary = pd.DataFrame(equation_summary_rows)
    equation_summary.to_csv(output_dir / "native_equation_sensitivity_summary.csv", index=False)

    source_paths = {
        "training_raw_split": train_path,
        "test_raw_split": test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "validated_shap": clean_shap_path,
        "shap_manifest": shap_manifest_path,
        "legacy_prediction_repeats": legacy_repeats_path,
        "random_forest": model_paths["Random Forest"],
        "xgboost": model_paths["XGBoost"],
        "audit_script": Path(__file__).resolve(),
    }
    manifest = {
        "status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "primary_comparison": "RF/XGBoost exact SHAP continuity; PySR retained only as native equation/model prediction sensitivity",
        "historical_artifacts": "Original low-budget SHAP continuity files in outputs/bodyfat/analysis are preserved and excluded from primary claims",
        "feature_order": RAW_FEATURES,
        "ratios": {
            "ACratio": "Abdomen / Chest, recomputed after every perturbation",
            "HTratio": "Hip / Thigh, recomputed after every perturbation",
        },
        "background": "Fixed 50-row training sample without replacement, pandas random_state=42",
        "background_ids": background_ids.tolist(),
        "explanation_subject_ids": subject_ids.tolist(),
        "explanation_subject_count": len(subject_ids),
        "subject_selection": shap_manifest["subset_rule"] + "; fixed before inspecting continuity outcomes",
        "noise_seed": NOISE_SEED,
        "prediction_noise_repeats": PREDICTION_REPEATS,
        "explanation_noise_repeats": EXPLANATION_REPEATS,
        "noise_levels": list(NOISE_LEVELS),
        "noise_stream": "First 10 draws of the existing 100-draw paired Gaussian stream are used for SHAP; identical draws and subjects are paired across all models",
        "shap_algorithm": "exact",
        "max_evals": EXACT_COALITIONS,
        "link": "identity",
        "estimation_variability": {
            "control": "Two independently constructed exact explainers on unchanged inputs from the prior reliability audit; no original array was reused as its own repeat",
            "results": independent_repeatability,
            "scope": "Exact enumeration removes permutation Monte Carlo variability conditional on this fixed empirical background; background choice and floating-point effects remain",
        },
        "response_training_sd": response_sd,
        "primary_response_stability_fraction": PRIMARY_RESPONSE_FRACTION,
        "primary_response_stability_threshold": thresholds[
            PRIMARY_RESPONSE_FRACTION
        ],
        "threshold_sensitivity_fractions": list(RESPONSE_FRACTIONS),
        "distance_measure": "Mean absolute attribution distance = mean_j |phi_j(perturbed)-phi_j(original)|, in body-fat percentage points per feature; both-zero vectors have distance 0 and one-zero pairs retain the non-zero magnitude",
        "cosine_zero_vector_rule": "Cosine=1 when both vectors are zero; cosine=0 when exactly one is zero",
        "interval_interpretation": "Empirical 2.5th and 97.5th percentiles of repeat-level means across paired perturbations of the same subjects; not independent-subject or population confidence intervals",
        "input_validity": "Raw and prepared inputs must be finite; all raw anthropometric values must remain positive; Chest and Thigh must be non-zero; ratios are recomputed; training-range violations are described but not clipped",
        "native_pysr": {
            "equation": "(Abdomen**2 - 272.58493*Wrist)/(Weight + Neck - Biceps)",
            "interpretation": "The one fitted global equation is structurally unchanged across inputs; reported values are equation/model prediction sensitivity, not SHAP continuity",
            "domain_rows_per_noise_level": PREDICTION_REPEATS * len(x_test),
            "minimum_absolute_denominator": float(
                input_frame["PySR minimum absolute denominator"].min()
            ),
            "near_zero_denominators": int(
                input_frame["PySR near-zero denominators"].sum()
            ),
        },
        "zero_noise": "Prepared inputs and RF/XGBoost/PySR predictions were checked for exact equality on all 51 test rows",
        "prediction_reuse_maximum_absolute_difference": float(
            prediction_check["Absolute difference"].max()
        ),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "shap": shap.__version__,
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
    }
    (output_dir / "continuity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    run_bodyfat_continuity(Path(__file__).resolve().parents[1])
