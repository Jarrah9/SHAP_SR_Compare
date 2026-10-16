# Verification of the curated collection

Checked on 16 October 2026 (Australia/Perth).

## File integrity and scope

- All **391 retained copied files** match their originals byte for byte, verified with SHA-256 and recorded in `COPIED_FILES.csv`. This includes 390 project files and the supplied dissertation PDF. The full raw Feynman file was subsequently removed from this curated folder at the researcher's request; its original in the source project remains unchanged.
- Existing project files were read and copied; none were edited, moved or deleted.
- The curated folder contains no original Git history, agent/editor state, Python or notebook caches, console logs, mock models, stroke experiments, presentation builds or writing drafts.
- Relative paths used by the retained analysis are preserved. The main notebook's 25 declared core artifact paths are present; the MLP dependencies and output inventories also pass their existing validation checks.
- All 17 Python source/test files and all 67 code cells across the four retained notebooks parse successfully.
- The large saved Feynman forest remains present as a complete original copy, with all `.joblib` predictors covered by Git LFS. The full raw I.9.18 file is excluded and ignored; its exact 10,000-row sample remains present. A new Git repository has not yet been initialised.

## Saved scientific results

The saved RMSE values reproduce the displayed values in dissertation Table 2 (Feynman fresh-final performance) and Table 8 (body-fat test performance):

| Dataset | Random Forest | XGBoost | MLP | PySR |
| --- | --- | --- | --- | --- |
| Feynman | 0.084252 | 0.037645 | 0.006354 | Numerical precision |
| Body fat | 4.130 | 4.121 | 3.737 | 3.883 |

These are checks against the retained result tables, not newly fitted model comparisons. The corrected Feynman results and original historical results remain separate.

## Existing verification suite

The unmodified `tests/test_mlp_analysis.py` suite passed **all seven tests** from the curated folder, using the local `xai-compare` environment and the pinned versions in `requirements.txt`.

The checks cover exact attribution for near-equal inputs, zero-attribution and sustained-recovery edge cases, saved MLP predictions and SHAP additivity, raw-feature ratio recomputation, paired completeness/continuity designs, and rejection of changed or missing cached outputs. Test fixtures were directed to an explicitly writable temporary directory because the managed Windows sandbox's default temporary directory rejected nested fixture creation. No copied research files were changed to run the tests.

## Notebook execution limit

An independent disposable copy was used for an additional execution check, so notebook-generated figures and tables could not alter the deliverable. The first six code cells of `Full_Analysis.ipynb` completed, including artifact/cache validation, loading the frozen models, reproducing saved Feynman tree predictions, loading the corrected selection results, and checking the literal native equation against saved predictions.

The additional run was stopped during the expensive full-test-sample TreeSHAP recalculation in cell 14 (zero-based index). **A complete notebook execution is not claimed.** All 31 main-notebook code cells parse, but later cells were not run in this check. No model fitting, selection rerun or audit regeneration was requested or performed.

The optional historical external PySR sweep path is documented in the README, with the included snapshot available for an explicit replay. Normal saved-result execution leaves that rerun disabled.
