# Comparing SHAP Explanations and Symbolic Regression

Research files accompanying Jarrah Oatham's dissertation. This folder is a curated copy of the files supporting the supplied dissertation, prepared on 16 October 2026. Existing research code, notebooks, datasets and results are copied without edits, with their relative paths preserved.

The study compares SHAP explanations of Random Forest, XGBoost and MLP predictors with native PySR equations on the dimensionless Feynman I.9.18 problem and a body-fat dataset.

## Start here

- `docs/dissertation.pdf`: the supplied dissertation.
- `Full_Analysis.ipynb`: the final integrated analysis, including MLP and the corrected explanation audits.
- `docs/FILE_SELECTION.md`: the connection between the dissertation and the retained files, plus the exclusion decisions.
- `docs/COPIED_FILES.csv`: every copied file, its source, size and SHA-256 checksum.
- `docs/VERIFICATION.md`: checks performed on this curated copy.

## Contents

| Path | Purpose |
| --- | --- |
| `bodyfat*.py`, `feynman*.py`, `mlp_common.py` | Preparation and training for the four model families used in the dissertation |
| `create_feynman_sample.py`, `Feynman_without_units/`, `FeynmanEquationsDimensionless.csv` | The exact 10,000-row I.9.18 sample, sampling script and dimensionless metadata; the full raw file is excluded |
| `data/bodyfat/` | Body-fat source data and the two-response exploratory survey export |
| `hpc_results/` | The selected body-fat and MLP runs, fitted predictors, manifests, predictions and search results |
| `outputs/best_models/` | Feynman tree fits, historical diagnostics, validation-only PySR selection and fresh-final evaluation |
| `outputs/bodyfat/` | Prepared inputs, historical analysis, corrected exact-SHAP audits and survey regression figures |
| `outputs/mlp_analysis/` | Frozen-MLP audit results, paired designs and combined figures |
| `scripts/`, `tests/` | Analysis helpers, relevant Pawsey launch scripts and existing verification tests |

`BodyFat_Mock_Models_Analysis.ipynb` is retained under its original name because it actually loads the completed real HPC models (`is_mock: false`) and produced the historical results and survey SHAP archive. The separate synthetic mock-model programs are excluded. `Feynman_Best_RF_XGBoost.ipynb` preserves the original saved-fit producer. `BodyFat_SHAP_PValue_Analysis.ipynb` produces the supplementary survey figures and regression tables.

## Read or reproduce the saved analysis

The reference local analysis environment uses Python 3.14.3. `requirements.txt` pins its analysis package versions, including the versions checked by the MLP cache manifests. From this folder:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab Full_Analysis.ipynb
```

Run the notebook from this folder as its working directory. Its standard path loads saved fits and validates saved MLP results. Leave the expensive `RERUN_*` switches at their supplied `False` values to consume the completed audits. Fitting models is unnecessary to inspect the saved notebook, tables and figures.

The final Feynman performance results are in `outputs/best_models/selection_audit/metrics.csv`, with MLP in `outputs/mlp_analysis/feynman/metrics.csv`. The top-level `outputs/best_models/metrics.csv` contains historical results. Body-fat primary tree explanations are in `outputs/bodyfat/analysis/reliability_audit/`; the older `outputs/bodyfat/analysis/shap_values.npz` remains relevant to the survey stimuli. These historical and corrected results are distinguished in the final notebook.

To check the saved MLP predictions, attribution arrays and paired designs:

```sh
python -B -m unittest discover -s tests -p test_mlp_analysis.py -v
```

To explicitly regenerate the frozen-MLP audits using the existing models:

```sh
python -m scripts.mlp_analysis
```

`MLP_MODELS.md` describes MLP training. Model fitting is separate from the saved analysis; PySR fitting additionally requires PySR and its Julia runtime. Pawsey launch scripts retain their original account and scratch paths and must be adapted for another allocation or checkout. The old `bodyfat_submit_all.sh` is excluded because it also submits FEAT, which was outside the dissertation's completed comparison; use the included individual launchers.

For an explicit replay of Feynman PySR selection, the original notebook's `PYSR_SWEEP_SOURCE` points to an external historical `pawsey-results` folder. Set it to `PROJECT_ROOT / "outputs/best_models/selection_audit/pysr_sweep_snapshot.csv"` before enabling that optional rerun. The exact saved candidate snapshot is included. Standard execution has no dependency on that external folder.

The nine missing Feynman PySR configurations and the precise survey-form version are limitations already recorded in the dissertation. This collection preserves the available evidence.

## Dataset availability

The full raw `Feynman_without_units/I.9.18` file is excluded to keep the repository smaller. Its exact 10,000-row sample (about 1.27 MiB) remains included because the fitted models and audit hashes depend on those rows and their order. The saved analysis and model-training scripts use this sample and do not require the million-row source file. The small body-fat CSV and prepared splits remain included.

`create_feynman_sample.py` is retained for optional resampling. To use it, obtain the original **dimensionless** I.9.18 source file separately and provide its location using `--source`. Its original SHA-256 is `93ab135842933e97d5425b0cca733b4274bbb662f4d0ec97a1f3665bfc396f50`. `.gitignore` excludes the raw file if it is placed at its old path. Normal saved-result analysis needs no dataset download.

## References

### Feynman dataset

Udrescu, S.-M., and Tegmark, M. (2020). *Feynman Symbolic Regression Database*. Original project/data page: [AI Feynman at MIT](https://space.mit.edu/home/tegmark/aifeynman.html). The study uses the dimensionless I.9.18 gravitational-force problem.

Associated paper: Udrescu, S.-M., and Tegmark, M. (2020). [AI Feynman: A physics-inspired method for symbolic regression](https://arxiv.org/abs/1905.11481). *Science Advances*, 6(16), eaay2631. The authors identify the original database URL in the paper record. The MIT data page returned HTTP 404 when checked on 16 October 2026; its link is retained as the original source reference, rather than presented as a verified working download.

### Body-fat dataset

The CSV used in this study was obtained from the [Body Fat Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset), published by fedesoriano.

Original data and measurement documentation: [StatLib body-fat dataset](https://lib.stat.cmu.edu/datasets/bodyfat), describing 252 men and data supplied by A. Garth Fisher. StatLib attributes the underlying study to K. W. Penrose, A. G. Nelson and A. G. Fisher (1985), *Generalized body composition prediction equation for men using simple measurement techniques*, *Medicine and Science in Sports and Exercise*, 17(2), 189. Kaggle identifies the version actually used; StatLib supplies the original provenance and variable definitions.

## Prepare this folder for GitHub

The saved Feynman Random Forest (192.20 MiB) is retained for prediction reproduction. `.gitattributes` assigns all `.joblib` predictors to Git LFS and preserves archived file bytes across checkouts. Install Git LFS before adding these files to a new repository.

Run these commands from this folder when ready to create its own repository:

```sh
git init
git lfs install --local
git add .
git commit -m "Add dissertation research files"
```

This folder was prepared locally; no GitHub repository or remote was created. The References section provides separate source links for both datasets; the source manifests retain their original provenance. No new licence has been assigned to the research or third-party data.
