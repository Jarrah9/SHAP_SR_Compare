"""Frozen-MLP extension to Full_Analysis; no fitting or model selection.

Run from the project root with ``python -m scripts.mlp_analysis``. Historical
tree/PySR audits are read-only. New results live in outputs/mlp_analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from scripts.bodyfat_continuity_audit import _prediction_function, _row_metrics
from scripts.explanation_metrics import (
    agreement_summary, feature_counts_for_shap_coverage, local_agreement,
    run_output_completeness, shap_compactness_tables,
)

PROTOCOL_VERSION = "1.0.0"
RUNS = {
    "feynman": "hpc_results/mlp/feynman/mlp_20260921_064655_009002",
    "bodyfat": "hpc_results/mlp/bodyfat/mlp_20260921_064558_696168",
}
AUDITS = {
    "feynman": "outputs/best_models/selection_audit",
    "bodyfat": "outputs/bodyfat/analysis/reliability_audit",
}


class ExactIndependent(shap.maskers.Independent):
    """Skip identical cells only; near-equal inputs can change an MLP output.

    SHAP's default np.isclose shortcut produced up to 3.2e-7 reconstruction
    residual on the Feynman rows. Exact equality preserves the full coalition
    calculation instead of weakening the numerical additivity check.
    """
    def invariants(self, x):
        if x.shape != self.data.shape[1:]:
            raise ValueError("Input shape differs from the SHAP background.")
        return x == self.data


def file_hash(path):
    """Canonical LF for text so Git's Windows checkout does not stale caches."""
    path = Path(path)
    data = path.read_bytes()
    if path.suffix in {".py", ".csv", ".json"} or not path.suffix:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(target, prediction):
    return {"Method": "MLP", "R2": r2_score(target, prediction),
            "RMSE": float(np.sqrt(mean_squared_error(target, prediction))),
            "MAE": mean_absolute_error(target, prediction),
            "MSE": mean_squared_error(target, prediction)}


def source_paths(root, dataset):
    run, audit = Path(RUNS[dataset]), Path(AUDITS[dataset])
    paths = [run / name for name in ["mlp_best.joblib", "mlp_manifest.json", "mlp_metrics.csv",
                                     "mlp_test_predictions.csv", "mlp_seed_validation.csv"]]
    paths += [Path("scripts") / name for name in ["mlp_analysis.py", "explanation_metrics.py",
              "bodyfat_continuity_audit.py", "feynman_continuity_audit.py"]]
    paths += [audit / name for name in ["continuity_manifest.json", "completeness_design.npz",
                                        "completeness_manifest.json"]]
    if dataset == "feynman":
        paths += [Path("Feynman_without_units/I.9.18_sample_10000"),
                  Path("outputs/best_models/training_manifest.json"),
                  audit / "final_rows.csv", audit / "local_faithfulness.npz"]
    else:
        paths += [Path("hpc_results/bodyfat/prepared") / name for name in
                  ["train_raw_split.csv", "test_raw_split.csv", "X_train.csv", "X_test.csv",
                   "y_train.csv", "y_test.csv", "preprocessing_manifest.json"]]
        paths += [audit / "shap_values.npz", audit / "deletion_donors.npz"]
    return {str(p.as_posix()): file_hash(root / p) for p in paths}


def cache_identity(root, dataset):
    return {"protocol_version": PROTOCOL_VERSION, "dataset": dataset, "models": ["MLP"],
            "algorithm": "exact", "background_rows": 50, "background_seed": 42,
            "link": "identity", "source_sha256": source_paths(root, dataset),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "sklearn": sklearn.__version__, "shap": shap.__version__}}


def validate_results(root, dataset):
    root = Path(root)
    output = root / "outputs/mlp_analysis" / dataset
    path = output / "manifest.json"
    instruction = "Generate them explicitly with: python -m scripts.mlp_analysis --dataset " + dataset
    if not path.exists():
        raise FileNotFoundError(f"Missing MLP audit: {path}. {instruction}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("identity") != cache_identity(root, dataset):
        raise ValueError(f"Stale or incomplete {dataset} MLP audit. {instruction}")
    if not manifest.get("outputs"):
        raise ValueError(f"Missing output inventory. {instruction}")
    for name, digest in manifest["outputs"].items():
        p = output / name
        if not p.is_file() or file_hash(p) != digest:
            raise ValueError(f"Missing or changed MLP audit output {p}. {instruction}")
    return manifest


class MLPAudit:
    def __init__(self, root, dataset):
        self.root, self.dataset = Path(root).resolve(), dataset
        self.run = self.root / RUNS[dataset]
        self.audit = self.root / AUDITS[dataset]
        self.output = self.root / "outputs/mlp_analysis" / dataset
        self.manifest = json.loads((self.run / "mlp_manifest.json").read_text())
        if self.manifest["status"] != "completed":
            raise ValueError("The selected MLP training run is incomplete.")
        # Verify the files that actually supplied training/test inputs and targets.
        # The copied HPC preprocessing metadata has different version/path fields;
        # its substantive feature-engineering contract is compared separately.
        for source, expected_hash in self.manifest["input_sha256"].items():
            relative = source.split("/SR-XAI-Comparison/")[-1]
            original = Path(relative)
            if original.suffix == ".csv":
                local = self.root / "hpc_results/bodyfat/prepared" / original.name
            elif original.name == "I.9.18_sample_10000":
                local = self.root / original
            else:
                continue
            if file_hash(local) != expected_hash:
                raise ValueError(f"Training data provenance differs from the saved MLP: {local}")
        self.model = joblib.load(self.run / "mlp_best.joblib")
        np.testing.assert_array_equal(self.model.feature_names_in_, self.manifest["feature_names"])
        self.continuity = json.loads((self.audit / "continuity_manifest.json").read_text())
        saved = pd.read_csv(self.run / "mlp_test_predictions.csv")
        if dataset == "feynman":
            self.features = self.manifest["feature_names"]
            data = pd.read_csv(self.root / "Feynman_without_units/I.9.18_sample_10000",
                               sep=r"\s+", header=None, names=[*self.features, "target"])
            self.train, self.test, self.y_train, self.y_test = train_test_split(
                data[self.features], data.target, test_size=.2, random_state=42)
            self.predict = lambda rows: np.asarray(self.model.predict(
                pd.DataFrame(rows, columns=self.features)), dtype=float).ravel()
            actual_col, pred_col = "actual_target", "mlp_prediction"
            self.final = pd.read_csv(self.audit / "final_rows.csv")
            with np.load(self.audit / "local_faithfulness.npz") as a:
                self.faith_ids = a["observation_ids"].astype(str)
                np.testing.assert_array_equal(a["donor_training_indices"], self.train.index)
            self.faith = self.final.set_index("observation_id").loc[self.faith_ids, self.features]
        else:
            prepared = self.root / "hpc_results/bodyfat/prepared"
            current_metadata = json.loads((prepared / "preprocessing_manifest.json").read_text())
            training_metadata = self.manifest["data"]["preprocessing_manifest"]
            for key in set(current_metadata) | set(training_metadata):
                if key not in {"versions", "data_path"} and current_metadata.get(key) != training_metadata.get(key):
                    raise ValueError(f"Body-fat preprocessing contract changed: {key}")
            raw_train = pd.read_csv(prepared / "train_raw_split.csv", index_col=0)
            raw_test = pd.read_csv(prepared / "test_raw_split.csv", index_col=0)
            self.features = self.manifest["data"]["preprocessing_manifest"]["raw_feature_columns"]
            self.train, self.test = raw_train[self.features], raw_test[self.features]
            self.y_train = pd.read_csv(prepared / "y_train.csv", index_col=0).BodyFat
            self.y_test = pd.read_csv(prepared / "y_test.csv", index_col=0).BodyFat
            self.predict = _prediction_function(self.model)
            from scripts.bodyfat_continuity_audit import _prepared_frame
            for split, rows in [("train", self.train), ("test", self.test)]:
                frame = pd.read_csv(prepared / f"X_{split}.csv", index_col=0)
                np.testing.assert_array_equal(frame.index, rows.index)
                np.testing.assert_allclose(_prepared_frame(rows), frame, atol=1e-12, rtol=1e-12)
            actual_col, pred_col = "actual_bodyfat", "mlp_predicted_bodyfat"
            self.faith, self.faith_ids = self.test, self.test.index.to_numpy()
        if set(self.train.index) & set(self.test.index):
            raise ValueError("Train and test rows overlap.")
        if (len(self.train), len(self.test)) != (self.manifest["n_train"], self.manifest["n_test"]):
            raise ValueError("Split sizes differ from the saved training run.")
        for rows, target in [(self.train, self.y_train), (self.test, self.y_test)]:
            np.testing.assert_array_equal(rows.index, target.index)
        np.testing.assert_array_equal(saved.source_row_index, self.test.index)
        np.testing.assert_allclose(saved[self.features], self.test, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(saved[actual_col], self.y_test, atol=1e-12, rtol=0)
        prediction = self.predict(self.test)
        if not np.isfinite(prediction).all():
            raise ValueError("MLP predictions must be finite.")
        np.testing.assert_allclose(prediction, saved[pred_col], atol=1e-9, rtol=1e-9)
        self.prediction_max_error = float(np.max(np.abs(prediction - saved[pred_col])))
        reported = pd.read_csv(self.run / "mlp_metrics.csv").iloc[0]
        for key, value in metrics(self.y_test, prediction).items():
            if key != "Method":
                np.testing.assert_allclose(value, reported["test_" + key.lower()], rtol=1e-9, atol=1e-10)
        self.background = self.train.sample(n=50, random_state=42)
        if dataset == "bodyfat":
            with np.load(self.audit / "shap_values.npz") as a:
                np.testing.assert_array_equal(a["background_source_row_index"], self.background.index)
                np.testing.assert_allclose(a["background_values"], self.background, atol=0, rtol=0)
        self.identity = cache_identity(self.root, dataset)

    def exact(self, rows, ids, name, *, independent=False):
        path = self.output / (name + ".npz")
        if path.exists() and not independent:
            with np.load(path) as a:
                np.testing.assert_array_equal(a["observation_ids"].astype(str), np.asarray(ids).astype(str))
                np.testing.assert_array_equal(a["features"], self.features)
                np.testing.assert_allclose(a["input_values"], rows, rtol=0, atol=0)
                np.testing.assert_array_equal(a["background_ids"], self.background.index)
                np.testing.assert_allclose(a["background_values"], self.background, rtol=0, atol=0)
                values, base = a["values"].copy(), a["base_values"].copy()
        else:
            explainer = shap.Explainer(self.predict,
                ExactIndependent(self.background, max_samples=50),
                algorithm="exact", feature_names=self.features)
            result = explainer(rows, max_evals=2**len(self.features), batch_size=32768, silent=True)
            values, base = np.asarray(result.values), np.asarray(result.base_values)
            np.savez_compressed(path, values=values, base_values=base, input_values=rows.to_numpy(),
                observation_ids=np.asarray(ids).astype(str), features=self.features,
                background_values=self.background.to_numpy(), background_ids=self.background.index.to_numpy())
            print(f"{self.dataset}: exact SHAP {name}, {len(rows)} rows", flush=True)
        if values.shape != rows.shape or not np.isfinite(values).all():
            raise ValueError("Invalid exact SHAP matrix.")
        np.testing.assert_allclose(values.sum(axis=1) + base, self.predict(rows), atol=1e-9, rtol=1e-9)
        return values, base

    def write_performance(self):
        historical = metrics(self.y_test, self.predict(self.test))
        pd.DataFrame([historical]).to_csv(self.output / "historical_metrics.csv", index=False)
        if self.dataset == "feynman":
            pred = self.predict(self.final[self.features])
            primary = metrics(self.final.target, pred)
            final = self.final[["observation_id", "target"]].copy()
            final["mlp_prediction"] = pred
            final.to_csv(self.output / "final_predictions.csv", index=False)
        else:
            primary = historical
        pd.DataFrame([primary]).to_csv(self.output / "metrics.csv", index=False)

    def faithfulness(self, values):
        n, p, d = len(self.faith), len(self.features), len(self.train)
        clean = self.predict(self.faith)
        local, feature_rows, budget_rows = [], [], []
        if self.dataset == "feynman":
            with np.load(self.audit / "local_faithfulness.npz") as a:
                indices, seeds, budgets = a["donor_indices"], a["seeds"], a["budgets"]
            changes = np.empty(indices.shape)
            for si in range(len(seeds)):
                for j in range(p):
                    rows = np.repeat(self.faith.to_numpy(), int(max(budgets)), axis=0)
                    rows[:, j] = self.train.iloc[:, j].to_numpy()[indices[si, :, j, :].ravel()]
                    changes[si, :, j, :] = np.abs(self.predict(rows).reshape(n, -1) - clean[:, None])
            impact = changes.mean(axis=(0, 3))
            for si, seed in enumerate(seeds):
                reference = changes[np.arange(len(seeds)) != si].mean(axis=(0, 3))
                for budget in budgets:
                    estimate = changes[si, :, :, :budget].mean(axis=2)
                    for i, subject in enumerate(self.faith_ids):
                        a = local_agreement(estimate[i], reference[i])
                        budget_rows.append({"Method": "MLP", "observation_id": subject,
                            "Donors": int(budget), "Seed": int(seed),
                            "Impact rank rho": a["Spearman correlation"], "Impact top3": a["Top-three overlap"],
                            "Normalized L1": np.abs(estimate[i]-reference[i]).sum()/max(reference[i].sum(), 1e-12)})
        else:
            changes = np.empty((n, p, d))
            for j in range(p):
                rows = np.repeat(self.faith.to_numpy(), d, axis=0)
                rows[:, j] = np.tile(self.train.iloc[:, j].to_numpy(), n)
                changes[:, j, :] = np.abs(self.predict(rows).reshape(n, d) - clean[:, None])
            impact = changes.mean(axis=2)
            with np.load(self.audit / "deletion_donors.npz") as a:
                np.testing.assert_array_equal(a["source_row_index"], self.faith_ids)
                np.testing.assert_array_equal(a["donor_source_row_index"], self.train.index)
                indices, seeds, budgets = a["random_donor_indices"], a["random_seeds"], a["random_budgets"]
            for si, seed in enumerate(seeds):
                drawn = np.take_along_axis(changes, indices[si], axis=2)
                for budget in budgets:
                    estimate = drawn[:, :, :budget].mean(axis=2)
                    for i, subject in enumerate(self.faith_ids):
                        a = local_agreement(estimate[i], impact[i])
                        rho = local_agreement(np.abs(values[i]), estimate[i])["Spearman correlation"]
                        reference_rho = local_agreement(np.abs(values[i]), impact[i])["Spearman correlation"]
                        budget_rows.append({"Method": "MLP", "source_row_index": subject,
                            "Donors": int(budget), "Seed": int(seed), "Impact rank rho": a["Spearman correlation"],
                            "Impact top-three overlap": a["Top-three overlap"],
                            "Normalized impact L1": np.abs(estimate[i]-impact[i]).sum()/max(impact[i].sum(), 1e-12),
                            "Local rho difference": abs(rho-reference_rho)})
        id_name = "observation_id" if self.dataset == "feynman" else "source_row_index"
        for i, subject in enumerate(self.faith_ids):
            local.append({"Method": "MLP", id_name: subject, **local_agreement(np.abs(values[i]), impact[i])})
            for j, feature in enumerate(self.features):
                feature_rows.append({"Method": "MLP", id_name: subject, "Feature": feature,
                    "Absolute SHAP": abs(values[i, j]), "Mean absolute prediction change": impact[i, j],
                    "SHAP rank": rankdata(-np.abs(values[i]))[j], "Impact rank": rankdata(-impact[i])[j]})
        local = pd.DataFrame(local)
        local.to_csv(self.output / "local_correctness.csv", index=False)
        agreement_summary(local).to_csv(self.output / "local_correctness_summary.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(self.output / "local_feature_impacts.csv", index=False)
        pd.DataFrame(budget_rows).to_csv(self.output / "deletion_budget_rows.csv", index=False)
        np.savez_compressed(self.output / "deletion_donors.npz", changes=changes, donor_indices=indices,
                            seeds=seeds, budgets=budgets, impact=impact, observation_ids=self.faith_ids,
                            donor_training_ids=self.train.index.to_numpy(), feature_names=self.features)

    def completeness(self, values, base):
        with np.load(self.audit / "completeness_design.npz") as a:
            np.testing.assert_array_equal(a["observation_ids"].astype(str), np.asarray(self.faith_ids).astype(str))
            np.testing.assert_array_equal(a["feature_names"], self.features)
            donors, seed, random_orders = a["donor_indices"], int(a["random_order_seed"]), a["random_orders"]
        is_feynman = self.dataset == "feynman"
        run_output_completeness(dataset="Feynman" if is_feynman else "Body fat", output_dir=self.output,
            X_eval=self.faith, observation_ids=self.faith_ids, X_training=self.train,
            donor_pool=self.train if is_feynman else self.background, donor_indices=donors,
            feature_names=self.features, predictors={"MLP": self.predict}, shap_values={"MLP": values},
            base_values={"MLP": base}, random_repeats=len(random_orders), random_order_seed=seed,
            target_unit="dimensionless target units" if is_feynman else "body-fat percentage points",
            shap_source="exact SHAP; fixed 50-row training background; complete saved MLP pipeline",
            donor_protocol="Same saved donor indices and random orders as the tree-model completeness audit",
            source_files=list(self.identity["source_sha256"]), project_root=self.root)
        with np.load(self.output / "completeness_design.npz") as a:
            np.testing.assert_array_equal(a["random_orders"], random_orders)

    def run_continuity(self, clean_values):
        ids = np.asarray(self.continuity["explanation_subject_ids"])
        positions = self.test.index.get_indexer(ids)
        if (positions < 0).any():
            raise ValueError("Saved continuity subjects are absent from MLP test rows.")
        rows = self.test.iloc[positions]
        independent, _ = self.exact(rows, ids, "unchanged_input_repeat", independent=True)
        reference = clean_values[positions]
        np.testing.assert_allclose(independent, reference, atol=1e-10, rtol=1e-10)
        cosine, distance, both, one = _row_metrics(reference, independent)
        counts = feature_counts_for_shap_coverage(reference)
        repeatability = pd.DataFrame({"Method": "MLP", "source_row_index": ids,
            "Cosine similarity": cosine, "Mean absolute attribution distance": distance,
            "Normalized L1 difference": np.abs(independent-reference).sum(axis=1)/np.maximum(np.abs(reference).sum(axis=1), 1e-12),
            "Attribution MAE": distance,
            "90% count difference": np.abs(feature_counts_for_shap_coverage(independent)-counts)})
        repeatability.to_csv(self.output / "shap_identical_input_repeats.csv", index=False)
        prediction = self.predict(self.test)
        zero_change = np.abs(self.predict(self.test.copy()) - prediction)
        np.testing.assert_allclose(zero_change, 0, rtol=0, atol=0)
        pd.DataFrame([{"Method": "MLP", "Rows": len(prediction),
            "Maximum zero-noise prediction change": float(zero_change.max()),
            "Exact zero-noise check": True}]).to_csv(self.output / "continuity_zero_noise_prediction_checks.csv", index=False)
        repeats = self.continuity["prediction_noise_repeats"]
        explanation_repeats = self.continuity["explanation_noise_repeats"]
        noise = np.random.default_rng(self.continuity["noise_seed"]).normal(
            size=(repeats, len(self.test), len(self.features)))
        response_sd = float(self.y_train.std(ddof=0))
        np.testing.assert_allclose(response_sd, self.continuity["response_training_sd"], atol=1e-12)
        thresholds = {fraction: fraction*response_sd for fraction in (.025, .05, .1)}
        pairs, prediction_rows, validity = [], [], []
        # Retain the zero-noise baseline used by the historical prediction plots.
        baseline_metrics = metrics(self.y_test, prediction)
        for repeat in range(repeats):
            prediction_rows.append({**baseline_metrics, "Noise (% training SD)": 0.0,
                "Noise repeat": repeat, "Mean absolute prediction change": 0.0,
                "Response-stable rows (%)": 100.0})
        for level in self.continuity["noise_levels"]:
            perturbed = self.test.to_numpy()[None, :, :] + level * noise * self.train.std(ddof=0).to_numpy()[None, None, :]
            if not np.isfinite(perturbed).all():
                raise ValueError("Non-finite perturbed inputs.")
            if self.dataset == "bodyfat" and (perturbed <= 0).any():
                raise ValueError("Non-positive perturbed body-fat measurement.")
            outside = (perturbed < self.train.min().to_numpy()) | (perturbed > self.train.max().to_numpy())
            validity.append({"Method": "MLP", "Noise (% training SD)": 100*level,
                "Evaluated rows": repeats*len(self.test), "All raw inputs finite": True,
                "Rows outside training ranges": int(outside.any(axis=2).sum())})
            for repeat in range(repeats):
                noisy = pd.DataFrame(perturbed[repeat], columns=self.features)
                noisy_prediction = self.predict(noisy)
                if not np.isfinite(noisy_prediction).all():
                    raise ValueError("Non-finite perturbed MLP predictions.")
                change = np.abs(noisy_prediction-prediction)
                prediction_rows.append({**metrics(self.y_test, noisy_prediction),
                    "Noise (% training SD)": 100*level, "Noise repeat": repeat,
                    "Mean absolute prediction change": float(change.mean()),
                    "Response-stable rows (%)": float(100*(change <= thresholds[.05]).mean())})
                if repeat >= explanation_repeats:
                    continue
                values, _ = self.exact(noisy.iloc[positions], ids, f"continuity_{level}_{repeat:02d}")
                cosine, distance, both, one = _row_metrics(reference, values)
                for i, subject in enumerate(ids):
                    record = {"Method": "MLP", "Noise (% training SD)": 100*level,
                        "Noise repeat": repeat, "source_row_index": int(subject),
                        "Absolute prediction change": change[positions[i]], "Cosine similarity": cosine[i],
                        "Mean absolute attribution distance": distance[i],
                        "Both attribution vectors zero": both[i], "Exactly one attribution vector zero": one[i]}
                    for fraction, threshold in thresholds.items():
                        record[f"Stable at {100*fraction:g}% response-SD threshold"] = bool(change[positions[i]] <= threshold)
                    pairs.append(record)
        pred = pd.DataFrame(prediction_rows)
        pred.to_csv(self.output / "noise_sensitivity_repeats.csv", index=False)
        summary = pred.groupby(["Method", "Noise (% training SD)"], sort=False).agg(
            **{"R2": ("R2", "mean"), "RMSE": ("RMSE", "mean"),
               "RMSE 2.5%": ("RMSE", lambda s:s.quantile(.025)), "RMSE 97.5%": ("RMSE", lambda s:s.quantile(.975)),
               "Mean absolute prediction change": ("Mean absolute prediction change", "mean"),
               "Response-stable rows (%)": ("Response-stable rows (%)", "mean")}).reset_index()
        summary["Noise repeats"] = repeats
        summary.to_csv(self.output / "noise_sensitivity.csv", index=False)
        pd.DataFrame(validity).to_csv(self.output / "continuity_input_validity.csv", index=False)
        pairs = pd.DataFrame(pairs)
        pairs.to_csv(self.output / "continuity_pairs.csv", index=False)
        stable = "Stable at 5% response-SD threshold"
        repeat_rows = []
        for (level, repeat), group in pairs.groupby(["Noise (% training SD)", "Noise repeat"], sort=False):
            kept = group[group[stable]]
            repeat_rows.append({"Method": "MLP", "Noise (% training SD)": level, "Noise repeat": repeat,
                "Candidate subjects": len(group), "Retained subjects": len(kept), "Retained proportion": len(kept)/len(group),
                "Mean cosine similarity": kept["Cosine similarity"].mean(),
                "Mean absolute attribution distance": kept["Mean absolute attribution distance"].mean()})
        repeat_frame = pd.DataFrame(repeat_rows)
        repeat_frame.to_csv(self.output / "continuity_repeat_summary.csv", index=False)
        summaries, sensitivities = [], []
        unit = "target units" if self.dataset == "feynman" else "percentage points"
        for level, group in pairs.groupby("Noise (% training SD)", sort=False):
            kept = group[group[stable]]
            rg = repeat_frame[repeat_frame["Noise (% training SD)"] == level]
            summaries.append({"Method": "MLP", "Noise (% training SD)": level,
                "Explanation subjects": len(ids), "Perturbation repeats": explanation_repeats,
                "Candidate explanation pairs": len(group), "Retained explanation pairs": len(kept),
                "Retained proportion": len(kept)/len(group),
                f"Response-stability threshold ({unit})": thresholds[.05],
                "Mean cosine similarity": kept["Cosine similarity"].mean(),
                "Repeat-mean cosine 2.5%": rg["Mean cosine similarity"].quantile(.025),
                "Repeat-mean cosine 97.5%": rg["Mean cosine similarity"].quantile(.975),
                "Mean absolute attribution distance": kept["Mean absolute attribution distance"].mean(),
                "Repeat-mean attribution distance 2.5%": rg["Mean absolute attribution distance"].quantile(.025),
                "Repeat-mean attribution distance 97.5%": rg["Mean absolute attribution distance"].quantile(.975),
                "Both-zero vector pairs": int(kept["Both attribution vectors zero"].sum()),
                "One-zero vector pairs": int(kept["Exactly one attribution vector zero"].sum())})
            for fraction, threshold in thresholds.items():
                retained = group[group[f"Stable at {100*fraction:g}% response-SD threshold"]]
                sensitivities.append({"Method": "MLP", "Noise (% training SD)": level,
                    "Threshold (% training response SD)": 100*fraction, f"Threshold ({unit})": threshold,
                    "Candidate explanation pairs": len(group), "Retained explanation pairs": len(retained),
                    "Retained proportion": len(retained)/len(group),
                    "Mean cosine similarity": retained["Cosine similarity"].mean(),
                    "Mean absolute attribution distance": retained["Mean absolute attribution distance"].mean()})
        pd.DataFrame(summaries).to_csv(self.output / "continuity_summary.csv", index=False)
        pd.DataFrame(sensitivities).to_csv(self.output / "continuity_threshold_sensitivity.csv", index=False)

    def run_all(self, force=False):
        self.output.mkdir(parents=True, exist_ok=True)
        identity_path = self.output / "cache_identity.json"
        invalidate = force or not identity_path.exists()
        if identity_path.exists():
            invalidate |= json.loads(identity_path.read_text()) != self.identity
        previous_manifest = self.output / "manifest.json"
        if previous_manifest.exists():
            previous = json.loads(previous_manifest.read_text())
            if previous.get("status") == "completed":
                # A damaged completed audit must not reuse unverified checkpoints.
                invalidate |= any(not (self.output / name).is_file() or
                    file_hash(self.output / name) != digest
                    for name, digest in previous.get("outputs", {}).items())
        if invalidate:
            # Only this dataset's generated NPZ checkpoints are invalidated.
            # Historical audits and model artefacts are outside this directory.
            for path in self.output.glob("*.npz"):
                path.unlink()
        write_json(identity_path, self.identity)
        write_json(self.output / "manifest.json", {"status": "running", "identity": self.identity})
        self.write_performance()
        values, base = self.exact(self.test, self.test.index, "shap_values")
        compactness, distribution = shap_compactness_tables({"MLP": values})
        compactness.to_csv(self.output / "compactness.csv", index=False)
        distribution.to_csv(self.output / "compactness_distribution_90.csv", index=False)
        if self.dataset == "feynman":
            faith_values, faith_base = self.exact(self.faith, self.faith_ids, "faithfulness_shap")
        else:
            faith_values, faith_base = values, base
        self.faithfulness(faith_values)
        print(f"{self.dataset}: faithfulness complete; calculating completeness", flush=True)
        self.completeness(faith_values, faith_base)
        print(f"{self.dataset}: completeness complete; calculating continuity", flush=True)
        self.run_continuity(values)
        manifest = {"status": "completed", "identity": self.identity,
            "prediction_reproduction_max_absolute_error": self.prediction_max_error,
            "feature_names": self.features, "background_ids": self.background.index.tolist(),
            "test_ids": self.test.index.tolist(), "faithfulness_ids": np.asarray(self.faith_ids).tolist(),
            "continuity_subject_ids": self.continuity["explanation_subject_ids"],
            "preprocessing": self.manifest["preprocessing"],
            "explanation": "Exact coalition enumeration with identity link over 50 fixed empirical training rows; strict-equality masking shortcut; complete saved scaler/MLP/inverse-target pipeline",
            "bodyfat_inputs": "13 raw measurements; ratios recomputed before input scaling; Density excluded",
            "selection": "Frozen training-only CV choice; no fit, seed selection, test selection or PySR reselection",
            "pairing": "Existing saved faithfulness donors, completeness donor/random-order designs, continuity subjects and perturbation streams reused",
            "limitations": "Fixed-background dependence remains. Feynman tree SHAP uses training-path conditioning; MLP SHAP uses empirical marginal masking. Donor hybrids may violate input dependencies. Repeated-subject intervals are not population CIs.",
            "outputs": {p.name: file_hash(p) for p in sorted(self.output.iterdir())
                        if p.is_file() and p.suffix in {".csv", ".npz", ".json"} and p.name != "manifest.json"}}
        write_json(self.output / "manifest.json", manifest)
        return validate_results(self.root, self.dataset)


def run_mlp_analysis(root=Path("."), dataset="all", force=False):
    datasets = list(RUNS) if dataset == "all" else [dataset]
    with threadpool_limits(limits=1):
        for name in datasets:
            if not force:
                try:
                    validate_results(root, name)
                    print(f"{name}: complete MLP audit already validated", flush=True)
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            MLPAudit(root, name).run_all(force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *RUNS], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_mlp_analysis(Path(__file__).resolve().parents[1], args.dataset, args.force)
