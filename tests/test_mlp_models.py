"""Small end-to-end checks; fixtures and fitted models live only in temporary directories."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from bodyfatMLP import load_data as load_bodyfat
from feynmanMLP import FEATURE_COLUMNS, load_data as load_feynman
from mlp_common import PROJECT_ROOT, resolve_jobs, validate_data


RAW_COLUMNS = ["Age", "Weight", "Height", "Neck", "Chest", "Abdomen", "Hip",
               "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"]


class MLPModelsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mlp-script-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_bodyfat(self):
        directory = self.root / "prepared"
        directory.mkdir()
        rng = np.random.default_rng(123)
        raw = pd.DataFrame(rng.uniform(20, 100, size=(50, 13)), columns=RAW_COLUMNS)
        raw.index.name = "source_row_index"
        raw["BodyFat"] = 15 + 0.15 * raw.Abdomen - 0.1 * raw.Wrist
        raw["Density"] = 1.0
        X = raw[RAW_COLUMNS].copy()
        X["ACratio"] = X.Abdomen / X.Chest
        X["HTratio"] = X.Hip / X.Thigh
        train_ids, test_ids = train_test_split(raw.index, test_size=0.2, random_state=42)
        # Deliberately shift test features and targets: leaked scaling is obvious.
        X.loc[test_ids] += 200
        raw.loc[test_ids, "BodyFat"] += 300
        for split, ids in [("train", train_ids), ("test", test_ids)]:
            X.loc[ids].to_csv(directory / f"X_{split}.csv")
            raw.loc[ids, ["BodyFat"]].to_csv(directory / f"y_{split}.csv")
        raw.loc[test_ids].to_csv(directory / "test_raw_split.csv")
        (directory / "preprocessing_manifest.json").write_text(
            json.dumps({"feature_columns": list(X.columns), "random_state": 42}), encoding="utf-8")
        return directory

    def make_feynman(self):
        path = self.root / "feynman_sample"
        rng = np.random.default_rng(234)
        features = rng.uniform(1, 4, size=(100, 6))
        targets = 10 + 0.8 * features[:, 0] + 0.2 * features[:, 1] ** 2
        np.savetxt(path, np.column_stack([features, targets]))
        return path

    def run_script(self, script, *arguments, success=True):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for key in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_ID", "SLURM_JOB_NODELIST"):
            env.pop(key, None)
        result = subprocess.run(
            [sys.executable, "-B", str(PROJECT_ROOT / script), *map(str, arguments)],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=180,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def check_artifact(self, output, data, prediction_column):
        manifest = json.loads((output / "mlp_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["n_train"], len(data.X_train))
        model = joblib.load(output / "mlp_best.joblib")
        np.testing.assert_allclose(model.regressor_.named_steps["scaler"].mean_, data.X_train.mean())
        np.testing.assert_allclose(model.transformer_.mean_, [data.y_train.mean()])
        mlp = model.regressor_.named_steps["mlp"]
        self.assertEqual(len(mlp.hidden_layer_sizes), 1)
        prediction = model.predict(data.X_test)
        saved = pd.read_csv(output / "mlp_test_predictions.csv")
        np.testing.assert_array_equal(saved.source_row_index, data.X_test.index)
        np.testing.assert_allclose(saved[prediction_column], prediction, rtol=1e-12, atol=1e-12)
        scores = pd.read_csv(output / "mlp_metrics.csv")
        self.assertAlmostEqual(scores.test_mse.iloc[0], mean_squared_error(data.y_test, prediction))
        folds = pd.read_csv(output / "mlp_cv_folds.csv")
        np.testing.assert_array_equal(folds.source_row_index, data.X_train.index)
        seed_scores = pd.read_csv(output / "mlp_seed_validation.csv")
        self.assertEqual(set(seed_scores.seed), {42, 1729})
        self.assertEqual(len(seed_scores), 4)
        # A separate process outside the project can load the artifact without
        # importing our helper or dataset entry points.
        code = "import joblib, sys; m=joblib.load(sys.argv[1]); print(type(m).__name__)"
        result = subprocess.run([sys.executable, "-B", "-c", code, str(output / "mlp_best.joblib")],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TransformedTargetRegressor", result.stdout)
        return model

    def test_bodyfat_training_scaling_serialization_and_no_overwrite(self):
        prepared = self.make_bodyfat()
        output = self.root / "bodyfat_run"
        arguments = ["--prepared-dir", prepared, "--output-dir", output,
                     "--search-iterations", 1, "--cv-folds", 2, "--hidden-units", 2,
                     "--max-iter", 10, "--max-fun", 30, "--n-jobs", 1,
                     "--validation-seeds", 42, 1729]
        self.run_script("bodyfatMLP.py", *arguments)
        data = load_bodyfat(prepared)
        model = self.check_artifact(output, data, "mlp_predicted_bodyfat")
        self.assertEqual(model.regressor_.named_steps["mlp"].solver, "lbfgs")
        self.assertTrue((output / "mlp_grid_search.csv").is_file())
        self.assertFalse((output / "mlp_loss_curve.csv").exists())
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        self.run_script("bodyfatMLP.py", *arguments, success=False)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_feynman_parallel_adam_and_fixed_split(self):
        path = self.make_feynman()
        output = self.root / "feynman_run"
        self.run_script("feynmanMLP.py", "--data-path", path, "--output-dir", output,
                        "--search-iterations", 1, "--cv-folds", 2, "--hidden-units", 2,
                        "--max-iter", 5, "--n-jobs", 2, "--validation-seeds", 42, 1729)
        data = load_feynman(path)
        _, expected_test = train_test_split(np.arange(100), test_size=0.2, random_state=42)
        np.testing.assert_array_equal(data.X_test.index, expected_test)
        self.assertEqual(list(data.X_train.columns), FEATURE_COLUMNS)
        model = self.check_artifact(output, data, "mlp_prediction")
        self.assertEqual(model.regressor_.named_steps["mlp"].solver, "adam")
        self.assertTrue((output / "mlp_loss_curve.csv").is_file())
        self.assertTrue((output / "mlp_grid_search.csv").is_file())

    def test_test_targets_do_not_change_training(self):
        path = self.make_feynman()
        original = load_feynman(path)
        arguments = ["--data-path", path, "--search-iterations", 1, "--cv-folds", 2,
                     "--hidden-units", 2, "--max-iter", 5, "--n-jobs", 1,
                     "--validation-seeds", 42, "--skip-grid"]
        self.run_script("feynmanMLP.py", *arguments, "--output-dir", self.root / "first")
        values = np.loadtxt(path)
        values[original.X_test.index, -1] += 10000
        np.savetxt(path, values)
        self.run_script("feynmanMLP.py", *arguments, "--output-dir", self.root / "second")
        first = joblib.load(self.root / "first/mlp_best.joblib")
        second = joblib.load(self.root / "second/mlp_best.joblib")
        np.testing.assert_array_equal(first.predict(original.X_test), second.predict(original.X_test))
        a = json.loads((self.root / "first/mlp_manifest.json").read_text())
        b = json.loads((self.root / "second/mlp_manifest.json").read_text())
        self.assertEqual(a["best_params"], b["best_params"])
        self.assertEqual(a["cv_best_neg_mse"], b["cv_best_neg_mse"])

    def test_invalid_data_rejected(self):
        data = load_feynman(self.make_feynman())
        data.y_train = data.y_train.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "indices"):
            validate_data(data, 5)
        prepared = self.make_bodyfat()
        raw = pd.read_csv(prepared / "test_raw_split.csv", index_col=0)
        raw.iloc[::-1].to_csv(prepared / "test_raw_split.csv")
        with self.assertRaisesRegex(ValueError, "aligned"):
            load_bodyfat(prepared)

    def test_worker_count_respects_slurm(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}):
            self.assertLessEqual(resolve_jobs(32), 2)
            self.assertLessEqual(resolve_jobs(None), 2)


if __name__ == "__main__":
    unittest.main()
