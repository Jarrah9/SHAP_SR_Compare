# Shallow MLP models on Setonix

These new scripts train real, tuned MLP regressors for body fat and Feynman.
They do not modify existing scripts, prepared data, models, or analysis notebooks.
Training and evaluation are CPU-only; SHAP is not calculated in these jobs.

## Files to copy to Setonix

Keep these paths relative to the existing project root:

- `mlp_common.py` (required by both training scripts)
- `bodyfatMLP.py`
- `feynmanMLP.py`
- `scripts/bodyfat_mlp.sh`
- `scripts/feynman_mlp.sh`

The batch scripts use the existing account `pawsey0411`, project directory
`/scratch/pawsey0411/joatham/SR-XAI-Comparison`, and its `.venv` environment.
That environment needs NumPy, pandas, SciPy, scikit-learn, joblib (1.3 or later),
and threadpoolctl. These are the usual scikit-learn dependencies; no neural-network
framework or GPU package is needed. Load any modules required by your existing
environment before submitting, just as for your RF jobs.

## Submit

From the project directory on Setonix:

```bash
sbatch scripts/bodyfat_mlp.sh
sbatch scripts/feynman_mlp.sh
```

Body fat reads `outputs/bodyfat/prepared` by default, matching `bodyfatRF.py`.
It requires the existing six files: `X_train.csv`, `X_test.csv`, `y_train.csv`,
`y_test.csv`, `test_raw_split.csv`, and `preprocessing_manifest.json`.
To train against the archived split used by `Full_Analysis.ipynb`, use:

```bash
sbatch scripts/bodyfat_mlp.sh --prepared-dir hpc_results/bodyfat/prepared
```

Choose the same prepared directory used for the RF/XGBoost models you compare
against. The MLP script does not run preparation or create a different split.

Feynman reads `Feynman_without_units/I.9.18_sample_10000` and reproduces the
existing 80/20 split with seed 42. It does not resample the source dataset.

Both submissions request one node, one task, 32 CPU cores, 64 GB RAM, and a
24-hour wall-time limit on `work`. These are starting resource requests, not
runtime estimates. Independent cross-validation fits run in parallel; each fit
uses one numerical-library thread. The Python scripts honour Slurm's CPU limit,
and `srun` receives `--cpus-per-task` explicitly. This follows the resource
declaration guidance in [Pawsey's Setonix documentation](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929058/Running+Jobs+on+Setonix).

You can override resources and training options without editing the files:

```bash
sbatch --cpus-per-task=8 --mem=16G scripts/bodyfat_mlp.sh --search-iterations 30
sbatch scripts/feynman_mlp.sh --solver lbfgs
```

## Training protocol

| Setting | Body fat | Feynman |
| --- | --- | --- |
| Hidden layers | One | One |
| Candidate neuron counts | 8, 16, 32 | 16, 32, 64 |
| Default optimiser | L-BFGS | Adam |
| Activations searched | ReLU, tanh | ReLU, tanh |
| Initial search | 60 random candidates, five-fold CV | Same |
| Regularisation search | Log-uniform alpha from 1e-6 to 10 | Same |
| Learning-rate search | Not applicable to L-BFGS | Log-uniform from 1e-4 to 1e-2 |
| Refinement | Width and alpha near the best candidate | Also learning rate |
| Maximum iterations | 2,000 | 2,000 |
| Training seed | 42 | 42 |

The folds are unshuffled, matching the existing regression scripts' `cv=5`.
Input and target standardisation are fitted separately inside each CV training
fold. Target scaling is automatically reversed for scoring, predictions, and
metrics. Body-fat median imputation remains the existing prepared-data operation;
the MLP does not refit that prior preprocessing within CV.

Adam uses an internal 10% validation split of each training fold for early
stopping (patience 30). It also uses an internal split during the final fit.
L-BFGS uses the full fitting data, without early stopping, and allows up to
50,000 function evaluations. Neither solver uses the held-out test set to tune
parameters or decide when to stop. The target wrapper and neural-network
behaviour follow the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html).

After selecting hyperparameters, the saved model is fitted with the requested
training seed. Additional CV checks at seeds 42, 1729, and 2026 measure training
sensitivity using those same hyperparameters. These diagnostics do not select a
seed and are not independent estimates of generalisation after tuning.

Convergence warnings from CV are visible in the Slurm log. Final-fit warnings
are also recorded in the manifest and metrics; inspect them before reporting
results. If the iteration/function limit is reached, increase `--max-iter` and/or
`--max-fun` and run again. Run `python bodyfatMLP.py --help` or
`python feynmanMLP.py --help` for all options.

## Outputs and prediction

Each run creates a new `outputs/bodyfat/mlp_<UTC timestamp>` or
`outputs/feynman/mlp_<UTC timestamp>` directory. `--output-dir` can select a
different **new** directory; existing directories are refused.

- `mlp_best.joblib`: fitted input scaler, MLP, and inverse target transformation.
- `mlp_metrics.csv`: train/test MSE, RMSE, MAE, R2, selected CV score and parameters.
- `mlp_test_predictions.csv`: source row indices, actual values, predictions and residuals.
- `mlp_random_search.csv` and `mlp_grid_search.csv`: candidate/fold results; no grid file with `--skip-grid`.
- `mlp_cv_folds.csv`: training row IDs and their validation-fold assignments.
- `mlp_seed_validation.csv`: training-only CV scores for each diagnostic seed.
- `mlp_manifest.json`: input hashes, feature order, software versions, Slurm allocation, settings and run status.
- `mlp_loss_curve.csv`: Adam's scaled training loss and internal validation R2, when using Adam.

The primary model is saved before the seed diagnostics. A fully finished run
has manifest status `completed`; a wall-time interruption can leave
`model_saved_seed_validation_pending` with a usable model but incomplete diagnostics.
There is no automatic resume: reruns create a new directory.

Load `mlp_best.joblib` with `joblib.load`, then call `model.predict(X)`.
For body fat, `X` must contain the same ordered 15 prepared features, including
`ACratio` and `HTratio`. For Feynman, use the same ordered six dimensionless
features. Do not scale `X` again: the saved model handles scaling and returns
predictions in original target units. Use the recorded scikit-learn version when
loading the artifact. No import of `mlp_common.py` is needed for inference.

## Lightweight local verification

With the training dependencies installed:

```bash
python -B -m unittest discover -s tests -p test_mlp_models.py -v
```

These checks use temporary synthetic fixtures and small training budgets. They
are software checks, not research runs or evidence of model performance.
