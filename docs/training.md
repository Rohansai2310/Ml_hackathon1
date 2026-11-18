# Model training and repository review

Reviewed starting commit: `28cb162d3075a70096a506030951bb9dd211e600`.

## What is ready

The preprocessing implementation is suitable for initial model experiments:
metric geometry, bounded candidate generation, 22 label-free features, explicit
reference alignment and shared trip-level folds are present. The inference path
can run with `with_route=False`. Feature tables are ignored by Git, so cloning
the repository requires rebuilding them from the original training/student-test
data. Existing plots and build logs do not replace this validation.

## Reference limitations to retain in the report

The reference arc per observation is derived by monotone alignment to `.route`;
it is not an independently supplied pointwise ground-truth label. Alignment can
remain ambiguous near intersections or repeated arcs. Features use neighbouring
GPS observations, including future points, so this is an offline trip-processing
pipeline rather than a causal streaming model.

The upstream rules reject reference assignments farther than 50 m and exclude
training trips `00000033` and `00000054` from fitting/CV. There are still 80
training trips in the inventory, with 78 eligible for fitting. These are upstream
data-quality decisions, preserved rather than silently replaced by fresh folds.

Accuracy and miss rate are measured on **valid derived references**. Filtering
by reference distance can make candidate recall appear very high: the candidate
radius is 75 m while reference validity is capped at 50 m. Report valid/unresolved
counts, exclusions and this limitation alongside scores. They are not accuracy
claims covering all original observations with independently verified truth.

The `load_xy` helper is useful for candidate-level tasks but cannot account for
zero-candidate observations. The training adapter therefore uses the complete
`*_points.parquet` inventory for observation-level metrics.

## Data adapter

- `point_idx` becomes `observation_id`; `arc_id` becomes `candidate_id`.
- Only `config.FEATURE_COLUMNS` enter classifiers; folds, labels, reference
  distances, coordinates and route positions remain metadata.
- `label_valid=False` becomes an unresolved reference, with blank training label.
- Known references remain in the inventory even when the reference candidate
  is absent or no candidates exist.
- Candidate labels, validity and fold metadata are checked against the points
  inventory. Training/student-testing trip IDs must be disjoint.
- Existing folds `0..4` are used unchanged; fold `-1` is excluded from fitting/CV.
- Folds are materialised one at a time to limit peak memory.

## Metrics and selection

`top1_accuracy`: correct highest-scoring candidate / all valid-reference
observations, counting candidate misses and zero-candidate observations as errors.
`top1_accuracy_reference_present`: the same accuracy restricted to observations
whose reference candidate was generated. `candidate_miss_rate`: proportion of
valid-reference observations lacking their reference candidate.

`macro_f1` and `log_loss` are binary **candidate-level** metrics. F1 uses a fixed
0.5 positive threshold; Top-1 selects among candidates regardless of that
threshold. Candidate probability ties use lexical candidate ID. Unresolved
observations receive predictions but are excluded from reference-based metrics.
Known, unresolved, reference-present and zero-candidate observation counts are
saved per fold and for student-testing.

CV metrics are arithmetic means across the five existing folds, with Top-1
standard deviation recorded. Baseline selection maximises mean Top-1 accuracy,
then minimises mean log loss and mean scoring time. Timing includes pipeline
preprocessing and excludes disk I/O, loading and HMM decoding. Close timing ties
can vary across executions even with a fixed seed.

Optuna's TPE sampler uses seeded startup exploration followed by Bayesian trials.
The default trial count is 30. Search spaces bound CPU cost while exploring
capacity and regularisation:

| Model | Search space |
|---|---|
| Logistic Regression | `C`: log scale 0.001–100 |
| Gaussian NB | `var_smoothing`: log scale 1e-12–0.001 |
| Decision Tree | Depth 2–20; minimum leaf size 1–30 |
| Random Forest | 100–400 trees, step 50; depth 3–24; minimum leaf size 1–20; feature fraction `sqrt`, 0.5, 1.0 |
| AdaBoost | 50–300 trees, step 50; depth 1–3; learning rate 0.01–1, log scale |

Search spaces and the winner are chosen without student-testing results. The
best trial is refitted on eligible training rows and compared against the
untuned baseline. All five baselines plus the tuned model receive student-test
evaluation. Evaluator-test data is never supplied to this runner.

## HMM handoff

Read `selection.json` for the final artifact, and its `configuration.json` for
ordered feature names. Load only trusted joblib files. The candidate-scoring
function needs IDs and features only, without labels or reference files:

```python
import json
import joblib
from pathlib import Path
from pathsnap.metrics import score_candidates

run = Path("outputs/training/run_001")
selection = json.loads((run / "selection.json").read_text())
artifact = run / selection["final_artifact"]
model = joblib.load(artifact)
features = json.loads((artifact.parent / "configuration.json").read_text())["features"]
candidate_table = candidate_table.rename(columns={"point_idx": "observation_id", "arc_id": "candidate_id"})
scores = score_candidates(model, candidate_table, features)
```

Scores are independent positive-class probabilities for each candidate, not
normalised probabilities across one observation's candidates. The decoder owner
must define how to turn these into emissions, handle missing candidates and
combine them with network transitions. Preserve original arc IDs for geometry
lookup. Reference-bearing evaluation CSVs stay outside the UI/inference path.

## Verification

```bash
PYTHONPATH=src python -m pytest -q tests
```

The upstream tests require raw data and built features. Added tests validate
fold-local preprocessing, candidate-score alignment, model serialization,
candidate misses, label exclusions and both the native Parquet handoff and an
end-to-end synthetic training run. Synthetic test metrics are never report
evidence. Actual model results belong under a separately named training run.
