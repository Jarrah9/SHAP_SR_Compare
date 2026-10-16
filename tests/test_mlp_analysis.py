"""Inference-only checks of the MLP explanation protocol and saved audits."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import shap
from threadpoolctl import threadpool_limits

from scripts.mlp_analysis import ExactIndependent, MLPAudit, file_hash, validate_results
from scripts.explanation_metrics import _smallest_sustained_count, feature_counts_for_shap_coverage

ROOT = Path(__file__).resolve().parents[1]


class ExactAttributionTest(unittest.TestCase):
    def test_near_equal_inputs_are_not_discarded(self):
        background = pd.DataFrame([[1., 2., 3.], [2., 3., 4.]], columns=list('abc'))
        rows = pd.DataFrame([[1.000001, 2.000001, 3.000001]], columns=background.columns)
        weights = np.array([2., -3., 5.])
        def predict(x):
            return np.asarray(x) @ weights + 7
        masker = ExactIndependent(background, max_samples=2)
        self.assertFalse(masker.invariants(rows.iloc[0].to_numpy())[0].any())
        result = shap.Explainer(predict, masker, algorithm='exact')(rows, max_evals=8)
        expected = (rows.to_numpy() - background.mean().to_numpy()) * weights
        np.testing.assert_allclose(result.values, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result.values.sum(1) + result.base_values, predict(rows), atol=1e-12)

    def test_compactness_zero_and_sustained_recovery_are_distinct(self):
        np.testing.assert_array_equal(feature_counts_for_shap_coverage([[0, 0, 0], [9, 1, 0]]), [0, 1])
        # First passing k is 1, but prediction recovery is not sustained until 3.
        self.assertEqual(_smallest_sustained_count([1., .01, .2, 0.], .05), 3)


class SavedMLPAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.audits = {name: MLPAudit(ROOT, name) for name in ('feynman', 'bodyfat')}

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_completed_results_and_additivity(self):
        for name, audit in self.audits.items():
            with self.subTest(dataset=name):
                manifest = validate_results(ROOT, name)
                self.assertLess(manifest['prediction_reproduction_max_absolute_error'], 1e-9)
                with np.load(audit.output / 'shap_values.npz') as saved:
                    self.assertEqual(saved['values'].shape, audit.test.shape)
                    np.testing.assert_allclose(saved['values'].sum(1) + saved['base_values'],
                                               audit.predict(audit.test), atol=1e-9, rtol=1e-9)
                    np.testing.assert_array_equal(saved['background_ids'], audit.background.index)

    def test_continuity_keeps_original_subjects_and_noise(self):
        for name, audit in self.audits.items():
            with self.subTest(dataset=name):
                spec = audit.continuity
                positions = audit.test.index.get_indexer(spec['explanation_subject_ids'])
                noise = np.random.default_rng(spec['noise_seed']).normal(size=(
                    spec['prediction_noise_repeats'], len(audit.test), len(audit.features)))
                expected = audit.test.to_numpy() + .025 * noise[0] * audit.train.std(ddof=0).to_numpy()
                with np.load(audit.output / 'continuity_0.025_00.npz') as saved:
                    np.testing.assert_array_equal(saved['observation_ids'],
                                                  np.asarray(spec['explanation_subject_ids']).astype(str))
                    np.testing.assert_allclose(saved['input_values'], expected[positions], atol=0, rtol=0)
                pairs = pd.read_csv(audit.output / 'continuity_pairs.csv')
                self.assertEqual(len(pairs), 2 * len(positions) * spec['explanation_noise_repeats'])
                np.testing.assert_array_equal(pairs['Stable at 5% response-SD threshold'],
                    pairs['Absolute prediction change'] <= .05*float(audit.y_train.std(ddof=0)))

    def test_completeness_reuses_donors_orders_and_has_correct_endpoints(self):
        for name, audit in self.audits.items():
            with self.subTest(dataset=name):
                with np.load(audit.output / 'completeness_design.npz') as new, np.load(audit.audit / 'completeness_design.npz') as old:
                    for key in ['donor_indices', 'random_orders', 'observation_ids', 'feature_names']:
                        np.testing.assert_array_equal(new[key], old[key])
                endpoints = pd.read_csv(audit.output / 'completeness_endpoint_checks.csv').iloc[0]
                for key in ['Full-preservation maximum error', 'Zero-deletion maximum change',
                            'All-replaced endpoint maximum discrepancy', 'Maximum additivity residual']:
                    self.assertLess(endpoints[key], 1e-9)

    def test_bodyfat_replacement_recomputes_ratios_before_scaling(self):
        audit = self.audits['bodyfat']
        with np.load(audit.output / 'deletion_donors.npz') as saved:
            donor_changes = saved['changes']
        for feature in ['Abdomen', 'Chest', 'Hip', 'Thigh']:
            j = audit.features.index(feature)
            rows = pd.DataFrame(np.repeat(audit.test.iloc[[0]].to_numpy(), len(audit.train), axis=0), columns=audit.features)
            rows[feature] = audit.train[feature].to_numpy()
            prepared = rows.copy()
            prepared['ACratio'] = rows.Abdomen / rows.Chest
            prepared['HTratio'] = rows.Hip / rows.Thigh
            expected = np.abs(audit.model.predict(prepared[audit.manifest['feature_names']]) - audit.predict(audit.test.iloc[[0]])[0])
            np.testing.assert_allclose(donor_changes[0, j], expected, atol=1e-10)
        self.assertNotIn('Density', audit.features)


class CacheValidationTest(unittest.TestCase):
    def test_changed_or_missing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'outputs/mlp_analysis/feynman'
            output.mkdir(parents=True)
            artifact = output / 'metrics.csv'
            artifact.write_text('Method,R2\nMLP,0.9\n')
            identity = {'protocol_version': 'fixture'}
            manifest = {'status': 'completed', 'identity': identity,
                        'outputs': {'metrics.csv': file_hash(artifact)}}
            (output / 'manifest.json').write_text(json.dumps(manifest))
            with patch('scripts.mlp_analysis.cache_identity', return_value=identity):
                validate_results(root, 'feynman')
                artifact.write_text('Method,R2\nMLP,1.0\n')
                with self.assertRaisesRegex(ValueError, 'Missing or changed'):
                    validate_results(root, 'feynman')
                artifact.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing or changed'):
                    validate_results(root, 'feynman')
            with patch('scripts.mlp_analysis.cache_identity', return_value={'protocol_version': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'Stale or incomplete'):
                    validate_results(root, 'feynman')


if __name__ == '__main__':
    unittest.main()
