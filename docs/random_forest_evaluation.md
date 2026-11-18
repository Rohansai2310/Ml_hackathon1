# Selected Random Forest and held-out evaluation

Random Forest is now the user-selected classifier. This evaluation keeps the
recorded baseline settings: 200 trees, seed 42, median imputation and all 22
features. It does not run another five-model comparison or tune on evaluator
outcomes. Add `--with-hmm` to evaluate the integrated decoder alongside the
classifier on the identical observation/reference denominator. See
[hmm_review.md](hmm_review.md) for the completed HMM benchmark and code review.

## Fit once using training data

From the repository root, with dependencies installed and the original feature
Parquets rebuilt:

```bash
python -m pathsnap.selected_forest \
  --feature-dir outputs/features \
  --output outputs/training/random_forest_selected_001 --jobs 2
```

This preserves the two training-trip exclusions and fits 78 eligible trips.
Student-testing predictions and metrics are produced after fitting. The saved
pipeline, configuration, data hashes, model hash and explicit selection record
live in that output directory. The final model is a refitted **untuned baseline**;
this command does not claim to complete the guidelines' Bayesian optimization
requirement. The previous cross-validation results are retained separately.

## Evaluate the frozen model

The data root argument is the directory containing `evaluator_test/`, whose
children are the ten trip directories:

```bash
python -m pathsnap.evaluate_forest \
  --model-dir outputs/training/random_forest_selected_001 \
  --data-root .. \
  --output outputs/evaluation/evaluator_test_rf_hmm_001 --with-hmm
```

The command rejects overlap with training/student-testing IDs and changed model
artifacts. It first creates candidates and predictions for every trip without
reading `.route`. Only after all predictions have been saved does a separate
phase load evaluator reference routes. Model and input hashes are recorded;
the model must remain unchanged during evaluation.

Prediction files under `predictions/<trip_id>/` contain no reference labels.
Reference-bearing comparisons belong under `evaluation/<trip_id>/`. Overall
metrics and per-trip metrics are saved at the run root. Outputs are local and
ignored by Git; a new run requires a fresh output directory.

## Interpreting accuracy

Observation-level Top-1 selects the highest-scoring candidate, with lexical arc
ID breaking ties. Its denominator includes every known-reference observation,
including zero-candidate observations and candidate misses. Candidate-level
Macro F1 uses the fixed binary threshold 0.5; log loss measures binary probability
quality. These are classifier metrics, not reconstructed-route metrics.

The existing monotone alignment derives pointwise reference arcs from `.route`.
Assignments farther than 50 m are unresolved and excluded from reference-based
metric denominators. Coverage and unresolved counts are always reported, and
all evaluator trips remain represented. Derived labels are not independently
verified pointwise truth; the 50 m validity threshold and 75 m candidate radius
also make known-reference candidate recall look especially high.

The evaluator result is a held-out benchmark. Do not change hyperparameters,
feature definitions, thresholds or the model based on that result. Any later
development should use training folds/student-testing, with fresh held-out data
needed for a new unbiased final estimate.

## Compact result archive

After evaluation finishes:

```bash
python scripts/report_evaluator_forest.py \
  --model-dir outputs/training/random_forest_selected_001 \
  --evaluation-dir outputs/evaluation/evaluator_test_rf_hmm_001 \
  --archive outputs/experiments/evaluator_test_rf_hmm_001
```

This exports measured summaries, a per-trip comparison plot and provenance for
review. It leaves the fitted model, full predictions and raw evaluator files
local. See that archive's README for the actual results once available.
