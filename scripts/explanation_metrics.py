"""Shared output-completeness and attribution summaries from Full_Analysis.

Protocols are unchanged; predictors may include any fitted regression model.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

SHAP_COMPACTNESS_COVERAGES = (0.80, 0.90, 0.95)
SHAP_ZERO_TOTAL_TOLERANCE = np.finfo(float).eps


COMPLETENESS_PROTOCOL_VERSION = "1.0.0"


def _batched_prediction(predict, rows, columns, batch_size=150000):
    output = []
    for start in range(0, len(rows), batch_size):
        frame = pd.DataFrame(rows[start : start + batch_size], columns=columns)
        output.append(np.asarray(predict(frame), dtype=float).ravel())
    return np.concatenate(output)


def _ordered_replacement_curves(predict, originals, donor_values, orders, columns):
    # Return preservation error and deletion change for complete local orders.
    n_orders, n_rows, p = orders.shape
    ranks = np.empty_like(orders)
    np.put_along_axis(
        ranks,
        orders,
        np.broadcast_to(np.arange(p), orders.shape),
        axis=2,
    )
    clean = _batched_prediction(predict, originals, columns)
    preserve = np.empty((n_orders, n_rows, p + 1), dtype=float)
    delete = np.empty_like(preserve)
    original_block = originals[None, :, None, :]
    donor_block = donor_values[None, :, :, :]
    for k in range(p + 1):
        selected = (ranks < k)[:, :, None, :]
        retained = np.where(selected, original_block, donor_block)
        retained_prediction = _batched_prediction(
            predict, retained.reshape(-1, p), columns
        ).reshape(n_orders, n_rows, -1)
        preserve[:, :, k] = np.mean(
            np.abs(retained_prediction - clean[None, :, None]), axis=2
        )
        del retained, retained_prediction

        removed = np.where(selected, donor_block, original_block)
        removed_prediction = _batched_prediction(
            predict, removed.reshape(-1, p), columns
        ).reshape(n_orders, n_rows, -1)
        delete[:, :, k] = np.mean(
            np.abs(removed_prediction - clean[None, :, None]), axis=2
        )
        del removed, removed_prediction
    return preserve, delete, clean


def _smallest_sustained_count(curve, tolerance):
    within = np.asarray(curve) <= tolerance
    sustained = np.logical_and.accumulate(within[::-1])[::-1]
    return int(np.flatnonzero(sustained)[0])


def _smallest_first_count(curve, tolerance):
    return int(np.flatnonzero(np.asarray(curve) <= tolerance)[0])


def _normalize_curve(curve, denominator, valid):
    result = np.full_like(curve, np.nan, dtype=float)
    if curve.ndim == 2:
        result[valid] = curve[valid] / denominator[valid, None]
    else:
        result[:, valid] = curve[:, valid] / denominator[None, valid, None]
    return result


def run_output_completeness(
    *,
    dataset,
    output_dir,
    X_eval,
    observation_ids,
    X_training,
    donor_pool,
    donor_indices,
    feature_names,
    predictors,
    shap_values,
    base_values,
    random_repeats,
    random_order_seed,
    target_unit,
    shap_source,
    donor_protocol,
    source_files,
    project_root=Path("."),
):
    # Evaluate local SHAP output-completeness without fitting any model.
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(feature_names)
    X_eval = X_eval[feature_names].reset_index(drop=True).astype(float)
    X_training = X_training[feature_names].reset_index(drop=True).astype(float)
    donor_pool = donor_pool[feature_names].reset_index(drop=True).astype(float)
    observation_ids = np.asarray(observation_ids)
    donor_indices = np.asarray(donor_indices, dtype=int)
    n, p = X_eval.shape
    if donor_indices.ndim != 2 or donor_indices.shape[0] != n:
        raise ValueError("donor_indices must have shape (evaluation rows, donor draws).")
    if donor_indices.min() < 0 or donor_indices.max() >= len(donor_pool):
        raise IndexError("A completeness donor index is outside the donor pool.")
    donor_values = donor_pool.to_numpy()[donor_indices]

    rng = np.random.default_rng(random_order_seed)
    random_orders = np.empty((random_repeats, n, p), dtype=int)
    for repeat in range(random_repeats):
        for row in range(n):
            random_orders[repeat, row] = rng.permutation(p)

    curve_frames = []
    subject_frames = []
    summary_rows = []
    endpoint_rows = []
    x_axis = np.arange(p + 1) / p

    for method, predict in predictors.items():
        values = np.asarray(shap_values[method], dtype=float)
        if values.shape != (n, p):
            raise ValueError(f"{dataset} {method} SHAP shape {values.shape} != {(n, p)}")
        magnitudes = np.abs(values)
        shap_orders = np.argsort(-magnitudes, axis=1, kind="stable")[None, :, :]
        shap_preserve, shap_delete, clean = _ordered_replacement_curves(
            predict, X_eval.to_numpy(), donor_values, shap_orders, feature_names
        )
        shap_preserve, shap_delete = shap_preserve[0], shap_delete[0]
        random_preserve, random_delete, random_clean = _ordered_replacement_curves(
            predict, X_eval.to_numpy(), donor_values, random_orders, feature_names
        )
        np.testing.assert_allclose(random_clean, clean, rtol=0, atol=0)

        training_prediction = _batched_prediction(
            predict, X_training.to_numpy(), feature_names
        )
        prediction_sd = float(np.std(training_prediction, ddof=0))
        numerical_floor = max(1e-12, 1e-10 * max(1.0, np.max(np.abs(clean)), prediction_sd))
        normalization_epsilon = max(1e-12, 1e-6 * prediction_sd)
        reconstruction_tolerance = max(numerical_floor, 0.05 * prediction_sd)

        all_replaced_change = shap_preserve[:, 0]
        valid_normalization = all_replaced_change > normalization_epsilon
        shap_preserve_n = _normalize_curve(
            shap_preserve, all_replaced_change, valid_normalization
        )
        shap_delete_n = _normalize_curve(
            shap_delete, all_replaced_change, valid_normalization
        )
        random_preserve_n = _normalize_curve(
            random_preserve, all_replaced_change, valid_normalization
        )
        random_delete_n = _normalize_curve(
            random_delete, all_replaced_change, valid_normalization
        )

        np.testing.assert_allclose(shap_preserve[:, -1], 0, atol=numerical_floor, rtol=0)
        np.testing.assert_allclose(shap_delete[:, 0], 0, atol=numerical_floor, rtol=0)
        np.testing.assert_allclose(
            shap_preserve[:, 0], shap_delete[:, -1], atol=numerical_floor, rtol=0
        )
        np.testing.assert_allclose(random_preserve[:, :, -1], 0, atol=numerical_floor, rtol=0)
        np.testing.assert_allclose(random_delete[:, :, 0], 0, atol=numerical_floor, rtol=0)

        base = np.asarray(base_values[method], dtype=float)
        if base.size == 1:
            base = np.full(n, base.item())
        else:
            base = np.broadcast_to(base.ravel(), (n,))
        additivity_residual = np.abs(values.sum(axis=1) + base - clean)
        additivity_tolerance = max(1e-5, numerical_floor)
        np.testing.assert_allclose(
            values.sum(axis=1) + base, clean, atol=additivity_tolerance, rtol=1e-5
        )

        total_magnitude = magnitudes.sum(axis=1)
        zero_attribution = total_magnitude <= np.finfo(float).eps
        k90 = np.zeros(n, dtype=int)
        for row in range(n):
            if not zero_attribution[row]:
                cumulative = np.cumsum(magnitudes[row, shap_orders[0, row]])
                k90[row] = int(
                    np.searchsorted(cumulative, 0.90 * total_magnitude[row], side="left")
                    + 1
                )

        row_index = np.arange(n)
        shap_preserve_90 = shap_preserve[row_index, k90]
        shap_delete_90 = shap_delete[row_index, k90]
        random_preserve_90 = random_preserve[:, row_index, k90]
        random_delete_90 = random_delete[:, row_index, k90]
        shap_preserve_90_n = shap_preserve_n[row_index, k90]
        shap_delete_90_n = shap_delete_n[row_index, k90]
        random_preserve_90_n = random_preserve_n[:, row_index, k90]
        random_delete_90_n = random_delete_n[:, row_index, k90]

        shap_first = np.array(
            [_smallest_first_count(curve, reconstruction_tolerance) for curve in shap_preserve]
        )
        shap_sustained = np.array(
            [_smallest_sustained_count(curve, reconstruction_tolerance) for curve in shap_preserve]
        )
        random_first = np.empty((random_repeats, n), dtype=int)
        random_sustained = np.empty_like(random_first)
        for repeat in range(random_repeats):
            for row in range(n):
                random_first[repeat, row] = _smallest_first_count(
                    random_preserve[repeat, row], reconstruction_tolerance
                )
                random_sustained[repeat, row] = _smallest_sustained_count(
                    random_preserve[repeat, row], reconstruction_tolerance
                )

        shap_preserve_reversals = (
            np.diff(shap_preserve, axis=1) > numerical_floor
        ).sum(axis=1)
        shap_delete_reversals = (
            np.diff(shap_delete, axis=1) < -numerical_floor
        ).sum(axis=1)
        random_preserve_reversals = (
            np.diff(random_preserve, axis=2) > numerical_floor
        ).sum(axis=2)
        random_delete_reversals = (
            np.diff(random_delete, axis=2) < -numerical_floor
        ).sum(axis=2)

        shap_preserve_auc = np.trapezoid(shap_preserve_n, x=x_axis, axis=1)
        shap_delete_auc = np.trapezoid(shap_delete_n, x=x_axis, axis=1)
        random_preserve_auc = np.trapezoid(random_preserve_n, x=x_axis, axis=2)
        random_delete_auc = np.trapezoid(random_delete_n, x=x_axis, axis=2)

        shap_frame = pd.DataFrame(
            {
                "Dataset": dataset,
                "Method": method,
                "Ordering": "Absolute SHAP",
                "Order repeat": -1,
                "Observation": np.repeat(observation_ids, p + 1),
                "k": np.tile(np.arange(p + 1), n),
                "Fraction selected": np.tile(x_axis, n),
                "Preservation error": shap_preserve.ravel(),
                "Deletion change": shap_delete.ravel(),
                "Normalized preservation error": shap_preserve_n.ravel(),
                "Normalized deletion change": shap_delete_n.ravel(),
            }
        )
        random_frame = pd.DataFrame(
            {
                "Dataset": dataset,
                "Method": method,
                "Ordering": "Random",
                "Order repeat": np.repeat(np.arange(random_repeats), n * (p + 1)),
                "Observation": np.tile(np.repeat(observation_ids, p + 1), random_repeats),
                "k": np.tile(np.arange(p + 1), random_repeats * n),
                "Fraction selected": np.tile(x_axis, random_repeats * n),
                "Preservation error": random_preserve.ravel(),
                "Deletion change": random_delete.ravel(),
                "Normalized preservation error": random_preserve_n.ravel(),
                "Normalized deletion change": random_delete_n.ravel(),
            }
        )
        curve_frames.extend([shap_frame, random_frame])

        subject_frames.append(
            pd.DataFrame(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "Observation": observation_ids,
                    "Features for 90% absolute SHAP": k90,
                    "Zero-total-attribution": zero_attribution,
                    "Preservation error at 90% SHAP": shap_preserve_90,
                    "Matched-random preservation error": random_preserve_90.mean(axis=0),
                    "Normalized preservation error at 90% SHAP": shap_preserve_90_n,
                    "Matched-random normalized preservation error": np.nanmean(random_preserve_90_n, axis=0),
                    "Deletion change at 90% SHAP": shap_delete_90,
                    "Matched-random deletion change": random_delete_90.mean(axis=0),
                    "Normalized deletion change at 90% SHAP": shap_delete_90_n,
                    "Matched-random normalized deletion change": np.nanmean(random_delete_90_n, axis=0),
                    "Reconstruction tolerance": reconstruction_tolerance,
                    "First features reaching tolerance": shap_first,
                    "Sustained features reaching tolerance": shap_sustained,
                    "Matched-random mean sustained features": random_sustained.mean(axis=0),
                    "SHAP preservation reversal steps": shap_preserve_reversals,
                    "SHAP deletion reversal steps": shap_delete_reversals,
                    "Matched-random mean preservation reversal steps": random_preserve_reversals.mean(axis=0),
                    "Matched-random mean deletion reversal steps": random_delete_reversals.mean(axis=0),
                    "Normalized preservation AUC": shap_preserve_auc,
                    "Matched-random normalized preservation AUC": np.nanmean(random_preserve_auc, axis=0),
                    "Normalized deletion AUC": shap_delete_auc,
                    "Matched-random normalized deletion AUC": np.nanmean(random_delete_auc, axis=0),
                    "All-replaced normalization denominator": all_replaced_change,
                    "Normalization defined": valid_normalization,
                    "Additivity absolute residual": additivity_residual,
                }
            )
        )

        summary_rows.append(
            {
                "Dataset": dataset,
                "Method": method,
                "N": n,
                "Features": p,
                "Mean features for 90% absolute SHAP": k90.mean(),
                "Median features for 90% absolute SHAP": np.median(k90),
                "Mean preservation error at 90% SHAP": shap_preserve_90.mean(),
                "Mean matched-random preservation error": random_preserve_90.mean(),
                "Mean normalized preservation error at 90% SHAP": np.nanmean(shap_preserve_90_n),
                "Mean matched-random normalized preservation error": np.nanmean(random_preserve_90_n),
                "Mean deletion change at 90% SHAP": shap_delete_90.mean(),
                "Mean matched-random deletion change": random_delete_90.mean(),
                "Mean normalized deletion change at 90% SHAP": np.nanmean(shap_delete_90_n),
                "Mean matched-random normalized deletion change": np.nanmean(random_delete_90_n),
                "Reconstruction tolerance": reconstruction_tolerance,
                "Median sustained reconstruction features": np.median(shap_sustained),
                "Mean sustained reconstruction features": shap_sustained.mean(),
                "Mean matched-random sustained reconstruction features": random_sustained.mean(),
                "Mean normalized preservation AUC": np.nanmean(shap_preserve_auc),
                "Mean matched-random normalized preservation AUC": np.nanmean(random_preserve_auc),
                "Mean normalized deletion AUC": np.nanmean(shap_delete_auc),
                "Mean matched-random normalized deletion AUC": np.nanmean(random_delete_auc),
                "Subjects with preservation direction reversal": int((shap_preserve_reversals > 0).sum()),
                "Subjects with deletion direction reversal": int((shap_delete_reversals > 0).sum()),
                "Subjects with undefined normalization": int((~valid_normalization).sum()),
                "Zero-total-attribution subjects": int(zero_attribution.sum()),
                "Maximum additivity residual": additivity_residual.max(),
            }
        )
        endpoint_rows.append(
            {
                "Dataset": dataset,
                "Method": method,
                "Full-preservation maximum error": shap_preserve[:, -1].max(),
                "Zero-deletion maximum change": shap_delete[:, 0].max(),
                "All-replaced endpoint maximum discrepancy": np.max(
                    np.abs(shap_preserve[:, 0] - shap_delete[:, -1])
                ),
                "Maximum additivity residual": additivity_residual.max(),
                "Median additivity residual": np.median(additivity_residual),
                "Numerical endpoint floor": numerical_floor,
                "Normalization epsilon": normalization_epsilon,
            }
        )

    curves = pd.concat(curve_frames, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    curves.to_csv(output_dir / "completeness_curves.csv", index=False)
    subjects.to_csv(output_dir / "completeness_subjects.csv", index=False)
    summary.to_csv(output_dir / "completeness_summary.csv", index=False)
    endpoints.to_csv(output_dir / "completeness_endpoint_checks.csv", index=False)
    np.savez_compressed(
        output_dir / "completeness_design.npz",
        donor_indices=donor_indices,
        random_orders=random_orders,
        observation_ids=observation_ids.astype(str),
        feature_names=np.asarray(feature_names),
        random_order_seed=random_order_seed,
    )
    manifest = {
        "status": "completed",
        "protocol_version": COMPLETENESS_PROTOCOL_VERSION,
        "dataset": dataset,
        "models": list(predictors),
        "n_evaluation_rows": n,
        "n_features": p,
        "target_unit": target_unit,
        "local_ranking": "Descending absolute SHAP magnitude; stable feature-order ties",
        "shap_source": shap_source,
        "preservation": "Keep top-k observed raw/model input features; replace remainder from one complete donor row; compare with original full-input prediction using mean absolute error",
        "deletion": "Replace top-k observed raw/model input features from the same donor row; compare with original full-input prediction using mean absolute change",
        "donor_protocol": donor_protocol,
        "paired_across_models": True,
        "same_donor_row_for_all_replaced_features_within_trial": True,
        "donors_per_observation": int(donor_indices.shape[1]),
        "random_orders": random_repeats,
        "random_order_seed": random_order_seed,
        "coverage_subset": "Smallest local prefix with cumulative absolute SHAP >= 90%; not interpreted as 90% prediction recovery",
        "reconstruction_tolerance": "5% of each fitted model's population SD of predictions over its training split, with numerical floor",
        "reconstruction_count": "Smallest k within tolerance at k and every larger k; first crossing also saved because curves may be non-monotonic",
        "normalization": "Divide each subject curve by its mean absolute prediction change when all features are replaced",
        "near_zero_denominator": "Undefined when denominator <= max(1e-12, 1e-6 * model training-prediction SD); raw-unit metrics retained",
        "non_monotonicity": "No monotonic smoothing or clipping; per-subject direction-reversal steps saved and summarized",
        "additivity": "Numerically checked for all explained rows; validation of attribution arithmetic only, not evidence that SHAP exposes the model's entire internal reasoning",
        "replacement_assumption": "Empirical marginal/hybrid intervention without retraining. A whole donor row preserves dependence inside the replaced block, but dependence between retained and replaced blocks can still be broken.",
        "outputs": [
            "completeness_curves.csv",
            "completeness_subjects.csv",
            "completeness_summary.csv",
            "completeness_endpoint_checks.csv",
            "completeness_design.npz",
        ],
        "source_sha256": {
            path: hashlib.sha256((Path(project_root) / path).read_bytes()).hexdigest()
            for path in source_files
        },
    }
    (output_dir / "completeness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return curves, subjects, summary, endpoints




def local_agreement(a, b):
    """Average-rank Spearman; undefined for constant vectors, never filled with 0.

    Top three uses fractional membership at the third-place boundary. The dot
    product / 3 is expected overlap under independent uniform tie-breaking.
    Also retain deterministic feature-order tie-breaking as a sensitivity check.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    def top_weights(v):
        boundary = np.sort(v)[-3]
        above, tied = v > boundary, v == boundary
        return above.astype(float) + tied * ((3-above.sum()) / tied.sum())
    ra, rb = rankdata(a), rankdata(b)
    defined = np.ptp(ra) > 0 and np.ptp(rb) > 0
    rho = float(np.corrcoef(ra, rb)[0, 1]) if defined else np.nan
    top_a = np.argsort(-a, kind="stable")[:3]
    top_b = np.argsort(-b, kind="stable")[:3]
    return {
        "Spearman correlation": rho,
        "Top-three overlap": float(np.dot(top_weights(a), top_weights(b))/3),
        "Top-three overlap (feature-order ties)": len(set(top_a)&set(top_b))/3,
        "Undefined correlation": not defined,
        "SHAP tied values": len(a)-len(np.unique(a)),
        "Impact tied values": len(b)-len(np.unique(b)),
    }


def agreement_summary(frame):
    result = []
    for method, group in frame.groupby("Method", sort=False):
        for metric in ["Spearman correlation", "Top-three overlap"]:
            v = group[metric].dropna()
            q1, median, q3 = v.quantile([.25, .5, .75]) if len(v) else (np.nan,)*3
            result.append({"Method": method, "Metric": metric, "N": len(group),
                           "Defined N": len(v), "Undefined N": len(group)-len(v),
                           "Median": median, "Q1": q1, "Q3": q3, "IQR": q3-q1})
    return pd.DataFrame(result)


def feature_counts_for_shap_coverage(values, coverage=0.90):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("SHAP values must be a finite two-dimensional array.")
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1].")
    absolute = np.abs(values)
    totals = absolute.sum(axis=1)
    zero_total = totals <= SHAP_ZERO_TOTAL_TOLERANCE
    counts = np.zeros(len(values), dtype=int)
    for row_index in np.flatnonzero(~zero_total):
        cumulative = np.cumsum(np.sort(absolute[row_index])[::-1])
        counts[row_index] = (
            np.searchsorted(
                cumulative,
                coverage * totals[row_index],
                side="left",
            )
            + 1
        )
    return counts


def shap_compactness_tables(shap_values_by_method):
    summary_rows = []
    distribution_rows = []
    for method, values in shap_values_by_method.items():
        values = np.asarray(values, dtype=float)
        totals = np.abs(values).sum(axis=1)
        zero_total = totals <= SHAP_ZERO_TOTAL_TOLERANCE
        for coverage in SHAP_COMPACTNESS_COVERAGES:
            counts = feature_counts_for_shap_coverage(values, coverage)
            q1, median, q3 = np.quantile(counts, [0.25, 0.50, 0.75])
            summary_rows.append(
                {
                    "Method": method,
                    "Attribution coverage": f"{coverage:.0%}",
                    "Observations": len(counts),
                    "Zero-total observations": int(zero_total.sum()),
                    "Mean features": counts.mean(),
                    "Median features": median,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": q3 - q1,
                    "Minimum": counts.min(),
                    "Maximum": counts.max(),
                }
            )
            if np.isclose(coverage, 0.90):
                sizes, frequencies = np.unique(counts, return_counts=True)
                for size, frequency in zip(sizes, frequencies):
                    distribution_rows.append(
                        {
                            "Method": method,
                            "Feature count": int(size),
                            "Observations": int(frequency),
                            "Percent of observations": 100 * frequency / len(counts),
                            "Zero-total category": bool(size == 0 and zero_total.any()),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)
