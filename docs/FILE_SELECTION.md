# File selection for the dissertation

Selection reference: `dissertation.pdf`, titled *Comparing SHAP Explanations and Symbolic Regression*, supplied as `C:/Users/jarra/Downloads/dissertation-1.pdf`. The PDF was treated as reference material; its contents were not treated as instructions. Its methods, results and supplementary survey define this collection's scope.

Project sources were copied from `C:/Users/jarra/OneDrive/Documents/Work/UWA/Year 4/Research Project/Project`. Every retained research file remains byte-for-byte identical to its source. The dissertation is additionally copied into `docs/dissertation.pdf`. New repository documentation and configuration are identified below.

## Connection to the dissertation

| Dissertation sections | Included evidence |
| --- | --- |
| 3.2: Feynman dataset and deterministic sampling | `Feynman_without_units/I.9.18_sample_10000`, `create_feynman_sample.py` and the dimensionless metadata CSV; the full raw I.9.18 file and other Feynman problems are excluded, with original dataset sources referenced in the README |
| 3.2: body-fat preparation | `bodyfat_prepare_data.py`, raw `data/bodyfat/bodyfat.csv`, and both small prepared-data exports used by training defaults and archived analysis |
| 3.3: Random Forest, XGBoost and PySR training | The six dataset-specific training scripts, their individual HPC launchers, selected body-fat run folders, Feynman saved tree fits and manifests |
| 3.3: MLP training | `bodyfatMLP.py`, `feynmanMLP.py`, `mlp_common.py`, MLP launchers, `MLP_MODELS.md`, the two selected September HPC runs and their CV/search records |
| 3.3 and 4.1-4.2: corrected Feynman selection | `outputs/best_models/selection_audit/`, including the 183-candidate snapshot, validation table, selection lock, physical draws, final rows and predictions |
| 3.4-3.5; Chapters 4-5: correctness, completeness, continuity and compactness | `Full_Analysis.ipynb`, shared analysis helpers, saved tree audit results, exact SHAP arrays, calibration checks, paired donor/noise designs and the full MLP audit folders |
| 3.6.1, 4.2.3 and 5.2.3: native tree importance | Tree training outputs, feature-importance CSVs and the final notebook's comparisons |
| 3.6.2 and 5.6: exploratory survey | The two-response CSV, `BodyFat_SHAP_PValue_Analysis.ipynb`, the historical SHAP archive used for the stimuli, and all four statistical-summary outputs |
| 4.3 and 5.3: dissertation completeness figures | Both exact image files are retained in `outputs/mlp_analysis/comparison/{feynman,bodyfat}/completeness_curves.png`. The files under `outputs/writing/results_figures/` are byte-identical duplicates and are omitted |
| 3.7: reproducibility | Original model/data manifests, saved predictors, audit arrays, seeds and identifiers, relevant existing tests, and the new copied-file checksum inventory |

The original producer notebooks are retained because the final analysis cites them and their historical results supply diagnostics or survey stimuli. Their names and historical procedures have not been rewritten. Use `Full_Analysis.ipynb` for the final results, with its corrected audits and MLP extension.

The per-perturbation and calibration `.npz` files are retained as completed research calculations: they support paired designs, independent SHAP checks and the saved audit checksums. They are distinct from editor or Python caches.

## Excluded material

| Material | Reason |
| --- | --- |
| `stroke*.py` | Stroke is not a dataset evaluated in the dissertation |
| `bodyfatMock*.py`, `bodyfat_mock_common.py`, `BODYFAT_MOCK_MODELS.md`, `outputs/bodyfat/mock_*` | Synthetic/mock stand-ins are not the completed real-data comparison |
| `bodyfatFEAT.py`, `scripts/bodyfat_feat.sh`, `scripts/bodyfat_submit_all.sh` | FEAT is outside the four-model comparison; the combined submitter also launches FEAT |
| `feynmanAIFeynman.py` | AI Feynman is not evaluated in the completed dissertation |
| `feynmanBenchmark.py`, `Feynman.ipynb` | Earlier exploratory benchmark workflow, superseded by the retained tuned scripts, saved-fit producer and final analysis |
| Full raw `Feynman_without_units/I.9.18` | Excluded at the researcher's request to reduce repository size; the exact 10,000-row analysis sample remains included and the original source is referenced in the README |
| Other files in `Feynman_without_units/` | Only I.9.18 is studied |
| `Full_Analysis_Presentation.*`, presentation helpers and `outputs/presentation/` | Separate presentation products; the presentation notebook predates the full MLP comparison |
| Timestamped April PySR checkpoint directories | Exploratory runs are not the frozen candidate snapshot used by the final corrected selection |
| Raw HPC `.log` files and redundant `chosenEquation.png` | Console output and a redundant equation image; the saved predictors, equations, structured manifests, predictions and search results are retained |
| Thesis introduction/background Word drafts and decision notes; `outputs/writing/` | Working drafts and writing-process material rather than final research inputs or results; the final supplied dissertation is included |
| `MLP_ANALYSIS.md` | Integration and source-repository rollback notes; the portable analysis and reproduction instructions are supplied in the new README |
| `AUDIT_CHANGES.md`, `notebook_edit_provenance.json`, `superseded_notebook_cells.json`, `cell_execution_report.json`, `artifact_inventory.json`, `verification_report.json` in the old body-fat audit | Historical editing/QA bookkeeping, including an outdated notebook hash and inventory. Scientific protocol manifests and result arrays remain included; this collection has its own inventory and verification report |
| `.git`, `.agents`, `.codex`, `.vscode`, `.chart-data-*`, Python/editor caches, shortcuts, `tmp/`, `-Force/` | Version-control history, agent/editor state, presentation intermediates and temporary material |

## New files

`README.md`, `requirements.txt`, `.gitignore`, `.gitattributes`, this selection explanation, `COPIED_FILES.csv` and `VERIFICATION.md` were added to describe, use and check the curated collection. They are repository support files, not new research results. No training code, notebook source, model or saved research result was modified.

## Provenance and limitations

`COPIED_FILES.csv` records each original relative source path, category, byte count and SHA-256 hash. Its only external source is the supplied dissertation PDF. Hashes describe file contents rather than filesystem timestamps. Older scientific manifests can mention original absolute paths or historical records; those are preserved as provenance and do not mean excluded working files are runtime dependencies. The optional external sweep-source setting is explained in the README.

The analysis retains the dissertation's distinction between historical results and corrected final results. Feynman PySR selection originally used historical test errors, so the final comparison uses the saved validation-only correction and fresh-final data. Body-fat computational explanations use the corrected exact arrays, while the exploratory survey concerns its historical stimuli. No unavailable runs or missing survey materials have been invented.
