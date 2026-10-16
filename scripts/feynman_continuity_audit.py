"""Targeted continuity measurements for the saved Feynman tree models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROTOCOL_VERSION = "1.0.0"
NOISE_LEVELS = (0.025, 0.05)
NOISE_REPEATS = 100
CONTINUITY_REPEATS = 20
CONTINUITY_SAMPLE_SIZE = 50
RESPONSE_FRACTIONS = (0.025, 0.05, 0.10)
PRIMARY_RESPONSE_FRACTION = 0.05
RANDOM_STATE = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shap_matrix(explainer, rows: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(rows, check_additivity=False)
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected one SHAP output for scalar regression.")
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP matrix, received {values.shape}.")
    return values


def _row_metrics(reference: np.ndarray, comparison: np.ndarray) -> tuple[np.ndarray, ...]:
    reference_norm = np.linalg.norm(reference, axis=1)
    comparison_norm = np.linalg.norm(comparison, axis=1)
    denominator = reference_norm * comparison_norm
    cosine = np.zeros(len(reference), dtype=float)
    nonzero = denominator > np.finfo(float).eps
    cosine[nonzero] = np.sum(reference[nonzero] * comparison[nonzero], axis=1) / denominator[nonzero]
    both_zero = (reference_norm <= np.finfo(float).eps) & (
        comparison_norm <= np.finfo(float).eps
    )
    one_zero = (~both_zero) & (
        (reference_norm <= np.finfo(float).eps)
        | (comparison_norm <= np.finfo(float).eps)
    )
    cosine[both_zero] = 1.0
    distance = np.mean(np.abs(comparison - reference), axis=1)
    return np.clip(cosine, -1.0, 1.0), distance, both_zero, one_zero


def _feynman_prediction(rows: pd.DataFrame | np.ndarray, columns: list[str]) -> np.ndarray:
    frame = pd.DataFrame(rows, columns=columns)
    denominator = (
        (frame["x2_x1"].to_numpy() - 1.0) ** 2
        + (frame["y2_x1"].to_numpy() - frame["y1_x1"].to_numpy()) ** 2
        + (frame["z2_x1"].to_numpy() - frame["z1_x1"].to_numpy()) ** 2
    )
    return frame["m2_m1"].to_numpy() / denominator


def _empirical_limits(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return np.nan, np.nan
    return float(finite.quantile(0.025)), float(finite.quantile(0.975))


def run_feynman_continuity(project_root: str | Path = ".") -> dict:
    project_root = Path(project_root).resolve()
    output_dir = project_root / "outputs/best_models/selection_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / "outputs/best_models/training_manifest.json"
    data_path = project_root / "Feynman_without_units/I.9.18_sample_10000"
    repeats_path = project_root / "outputs/best_models/noise_sensitivity_repeats.csv"
    saved_prediction_path = project_root / "outputs/best_models/test_predictions.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["feature_columns"]
    data = pd.read_csv(
        data_path, sep=r"\s+", header=None, names=[*features, "target"]
    )
    x_train, x_test, y_train, y_test = train_test_split(
        data[features],
        data["target"],
        test_size=manifest["test_size"],
        random_state=manifest["random_state"],
    )
    saved_predictions = pd.read_csv(saved_prediction_path)
    np.testing.assert_array_equal(saved_predictions["source_row_index"], x_test.index)

    model_paths = {
        "Random Forest": project_root / "outputs/best_models/random_forest_best.joblib",
        "XGBoost": project_root / "outputs/best_models/xgboost_best.joblib",
    }
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    for model in models.values():
        model.set_params(n_jobs=1)

    continuity_positions = np.sort(
        np.random.default_rng(RANDOM_STATE).choice(
            len(x_test), size=CONTINUITY_SAMPLE_SIZE, replace=False
        )
    )
    continuity_rows = x_test.iloc[continuity_positions]
    continuity_ids = x_test.index.to_numpy()[continuity_positions]
    clean_predictions = {
        method: np.asarray(model.predict(x_test), dtype=float).ravel()
        for method, model in models.items()
    }
    clean_predictions["PySR"] = _feynman_prediction(x_test, features)

    # Two fresh TreeExplainers establish unchanged-input repeatability.  Neither
    # calculation is the all-test SHAP array already present in the notebook.
    repeatability_rows = []
    clean_shap = {}
    explainers = {}
    for method, model in models.items():
        first_explainer = shap.TreeExplainer(model)
        second_explainer = shap.TreeExplainer(model)
        first = _shap_matrix(first_explainer, continuity_rows)
        second = _shap_matrix(second_explainer, continuity_rows)
        first_base = np.asarray(first_explainer.expected_value).item()
        second_base = np.asarray(second_explainer.expected_value).item()
        np.testing.assert_allclose(
            first.sum(axis=1) + first_base,
            model.predict(continuity_rows),
            rtol=1e-5,
            atol=1e-5,
        )
        cosine, distance, both_zero, one_zero = _row_metrics(first, second)
        for row, subject in enumerate(continuity_ids):
            repeatability_rows.append(
                {
                    "Method": method,
                    "source_row_index": int(subject),
                    "Cosine similarity": float(cosine[row]),
                    "Mean absolute attribution distance": float(distance[row]),
                    "Both attribution vectors zero": bool(both_zero[row]),
                    "Exactly one attribution vector zero": bool(one_zero[row]),
                }
            )
        np.testing.assert_allclose(first, second, rtol=0, atol=0)
        np.testing.assert_allclose(first_base, second_base, rtol=0, atol=0)
        clean_shap[method] = first
        explainers[method] = first_explainer
    repeatability = pd.DataFrame(repeatability_rows)
    repeatability.to_csv(output_dir / "continuity_identical_input_repeats.csv", index=False)

    noise = np.random.default_rng(RANDOM_STATE).normal(
        size=(NOISE_REPEATS, len(x_test), len(features))
    )
    training_sd = x_train.std(axis=0, ddof=0).to_numpy()
    response_sd = float(y_train.std(ddof=0))
    thresholds = {fraction: fraction * response_sd for fraction in RESPONSE_FRACTIONS}
    legacy = pd.read_csv(repeats_path)
    pair_rows = []
    prediction_check_rows = []
    domain_rows = []

    # Zero noise is checked by a reconstructed input path and fresh predictions.
    zero_rows = []
    for method, predict in {
        **{name: model.predict for name, model in models.items()},
        "PySR": lambda frame: _feynman_prediction(frame, features),
    }.items():
        change = np.abs(np.asarray(predict(x_test)) - clean_predictions[method])
        np.testing.assert_allclose(change, 0, rtol=0, atol=0)
        zero_rows.append(
            {
                "Method": method,
                "Rows": len(x_test),
                "Maximum zero-noise prediction change": float(change.max()),
                "Exact zero-noise check": bool(np.all(change == 0)),
            }
        )
    pd.DataFrame(zero_rows).to_csv(
        output_dir / "continuity_zero_noise_prediction_checks.csv", index=False
    )

    for level in NOISE_LEVELS:
        perturbed = x_test.to_numpy()[None, :, :] + level * noise * training_sd[None, None, :]
        flat = perturbed.reshape(-1, len(features))
        flat_frame = pd.DataFrame(flat, columns=features)
        denominator = (
            (flat[:, 1] - 1.0) ** 2
            + (flat[:, 3] - flat[:, 2]) ** 2
            + (flat[:, 5] - flat[:, 4]) ** 2
        )
        equation_prediction = flat[:, 0] / denominator
        singularity_tolerance = 1e-10 * max(1.0, float(np.median(denominator)))
        domain_rows.append(
            {
                "Noise (% training SD)": 100 * level,
                "Evaluated subject-perturbation rows": len(flat),
                "Finite input rows": int(np.isfinite(flat).all(axis=1).sum()),
                "Minimum equation denominator": float(denominator.min()),
                "Numerical singularity tolerance": singularity_tolerance,
                "Near-zero denominators": int((denominator <= singularity_tolerance).sum()),
                "Finite PySR predictions": int(np.isfinite(equation_prediction).sum()),
                "Maximum absolute PySR prediction": float(np.max(np.abs(equation_prediction))),
            }
        )

        # Tree predictions are needed only for the 20 explanation repeats.  The
        # saved 100-repeat prediction summaries remain the reported robustness
        # experiment; PySR's inexpensive equation is regenerated for all 100
        # repeats to verify the shared stream and inspect its whole domain.
        tree_flat = flat[: CONTINUITY_REPEATS * len(x_test)]
        for model in models.values():
            model.set_params(n_jobs=-1)
        predictions_all = {
            method: np.asarray(
                model.predict(pd.DataFrame(tree_flat, columns=features)), dtype=float
            ).reshape(
                CONTINUITY_REPEATS, len(x_test)
            )
            for method, model in models.items()
        }
        predictions_all["PySR"] = equation_prediction.reshape(
            NOISE_REPEATS, len(x_test)
        )

        # One TreeSHAP call per model/level retains the original 20x50 design
        # while avoiding repeated explainer overhead.
        audit_flat = perturbed[:CONTINUITY_REPEATS, continuity_positions, :].reshape(
            -1, len(features)
        )
        noisy_shap_all = {
            method: _shap_matrix(
                explainers[method], pd.DataFrame(audit_flat, columns=features)
            ).reshape(CONTINUITY_REPEATS, CONTINUITY_SAMPLE_SIZE, len(features))
            for method in models
        }

        for repeat in range(NOISE_REPEATS):
            noisy_rows = pd.DataFrame(perturbed[repeat], columns=features, index=x_test.index)
            predictions = {"PySR": predictions_all["PySR"][repeat]}
            if repeat < CONTINUITY_REPEATS:
                predictions.update(
                    {
                        method: values[repeat]
                        for method, values in predictions_all.items()
                        if method != "PySR"
                    }
                )
            for method, prediction in predictions.items():
                change = np.abs(prediction - clean_predictions[method])
                regenerated = {
                    "R2": r2_score(y_test, prediction),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
                    "Mean absolute prediction change": float(change.mean()),
                    "Response-stable rows (%)": float(
                        100 * (change <= thresholds[PRIMARY_RESPONSE_FRACTION]).mean()
                    ),
                }
                saved = legacy[
                    (legacy["Method"] == method)
                    & (legacy["Noise repeat"] == repeat)
                    & np.isclose(legacy["Noise (% training SD)"], 100 * level)
                ]
                if len(saved) != 1:
                    raise RuntimeError("Could not identify one saved Feynman prediction row.")
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

            if repeat >= CONTINUITY_REPEATS:
                continue
            for method, model in models.items():
                noisy_shap = noisy_shap_all[method][repeat]
                cosine, distance, both_zero, one_zero = _row_metrics(
                    clean_shap[method], noisy_shap
                )
                prediction_change = np.abs(
                    predictions[method][continuity_positions]
                    - clean_predictions[method][continuity_positions]
                )
                for row, subject in enumerate(continuity_ids):
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

    domain = pd.DataFrame(domain_rows)
    if not (
        (domain["Finite input rows"] == domain["Evaluated subject-perturbation rows"]).all()
        and (domain["Near-zero denominators"] == 0).all()
        and (domain["Finite PySR predictions"] == domain["Evaluated subject-perturbation rows"]).all()
    ):
        raise RuntimeError("The Feynman perturbation domain failed validation.")
    domain.to_csv(output_dir / "continuity_native_equation_domain.csv", index=False)
    prediction_check = pd.DataFrame(prediction_check_rows)
    prediction_check.to_csv(output_dir / "continuity_prediction_reuse_check.csv", index=False)
    if prediction_check["Absolute difference"].max() > 2e-6:
        raise RuntimeError("Regenerated Feynman prediction sensitivity does not match the saved stream.")

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output_dir / "continuity_pairs.csv", index=False)
    primary_column = f"Stable at {100 * PRIMARY_RESPONSE_FRACTION:g}% response-SD threshold"
    repeat_rows = []
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
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(output_dir / "continuity_repeat_summary.csv", index=False)

    summary_rows = []
    for (method, level), group in pairs.groupby(
        ["Method", "Noise (% training SD)"], sort=False
    ):
        retained = group[group[primary_column]]
        current = repeats[
            (repeats["Method"] == method)
            & (repeats["Noise (% training SD)"] == level)
        ]
        cosine_low, cosine_high = _empirical_limits(current["Mean cosine similarity"])
        distance_low, distance_high = _empirical_limits(
            current["Mean absolute attribution distance"]
        )
        summary_rows.append(
            {
                "Method": method,
                "Noise (% training SD)": level,
                "Explanation subjects": CONTINUITY_SAMPLE_SIZE,
                "Perturbation repeats": CONTINUITY_REPEATS,
                "Candidate explanation pairs": len(group),
                "Retained explanation pairs": len(retained),
                "Retained proportion": len(retained) / len(group),
                "Response-stability threshold (target units)": thresholds[
                    PRIMARY_RESPONSE_FRACTION
                ],
                "Mean cosine similarity": retained["Cosine similarity"].mean(),
                "Repeat-mean cosine 2.5%": cosine_low,
                "Repeat-mean cosine 97.5%": cosine_high,
                "Mean absolute attribution distance": retained[
                    "Mean absolute attribution distance"
                ].mean(),
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
                    "Threshold (target units)": threshold,
                    "Candidate explanation pairs": len(group),
                    "Retained explanation pairs": len(retained),
                    "Retained proportion": len(retained) / len(group),
                    "Mean cosine similarity": retained["Cosine similarity"].mean(),
                    "Mean absolute attribution distance": retained[
                        "Mean absolute attribution distance"
                    ].mean(),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "continuity_threshold_sensitivity.csv", index=False
    )

    sources = {
        "data": data_path,
        "training_manifest": manifest_path,
        "prediction_repeats": repeats_path,
        "saved_predictions": saved_prediction_path,
        "random_forest": model_paths["Random Forest"],
        "xgboost": model_paths["XGBoost"],
        "audit_script": Path(__file__).resolve(),
    }
    audit_manifest = {
        "status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "historical_test_status": "Fixed-model diagnostic on the previously used test split",
        "primary_comparison": "RF/XGBoost TreeSHAP continuity; PySR only equation/model prediction sensitivity",
        "noise_seed": RANDOM_STATE,
        "noise_levels": list(NOISE_LEVELS),
        "prediction_noise_repeats": NOISE_REPEATS,
        "explanation_noise_repeats": CONTINUITY_REPEATS,
        "explanation_subject_count": CONTINUITY_SAMPLE_SIZE,
        "explanation_subject_ids": continuity_ids.tolist(),
        "subject_selection": "Fixed random sample without replacement, numpy default_rng seed 42; identical to original continuity calculation",
        "shap_algorithm": "TreeSHAP, tree_path_dependent using fixed training path counts encoded in the frozen trees",
        "estimation_variability": "Two fresh TreeExplainers on identical inputs returned exactly equal arrays; no original array was reused as its own repeat",
        "response_training_sd": response_sd,
        "primary_response_stability_fraction": PRIMARY_RESPONSE_FRACTION,
        "primary_response_stability_threshold": thresholds[
            PRIMARY_RESPONSE_FRACTION
        ],
        "threshold_sensitivity_fractions": list(RESPONSE_FRACTIONS),
        "distance_measure": "Mean absolute attribution distance = mean_j |phi_j(perturbed)-phi_j(original)| in target units per feature; both-zero vectors have distance 0 and one-zero pairs retain the non-zero magnitude",
        "cosine_zero_vector_rule": "Cosine=1 when both vectors are zero; cosine=0 when exactly one is zero",
        "interval_interpretation": "Empirical 2.5th and 97.5th percentiles of repeat-level means across paired perturbations of the same fixed subjects; not population confidence intervals",
        "native_pysr": {
            "structural_status": "One fixed global equation; text and operations do not change across inputs",
            "result_label": "Equation/model prediction sensitivity, not SHAP continuity",
            "domain_rows_per_noise_level": NOISE_REPEATS * len(x_test),
            "minimum_denominator": float(domain["Minimum equation denominator"].min()),
            "near_zero_denominators": int(domain["Near-zero denominators"].sum()),
        },
        "prediction_reuse_maximum_absolute_difference": float(
            prediction_check["Absolute difference"].max()
        ),
        "source_sha256": {name: _sha256(path) for name, path in sources.items()},
    }
    (output_dir / "continuity_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2), encoding="utf-8"
    )
    return audit_manifest


if __name__ == "__main__":
    run_feynman_continuity(Path(__file__).resolve().parents[1])
