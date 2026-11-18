# HMM/Viterbi starting point

Update: Random Forest has been refitted, and the person-3 HMM/API implementation
has been merged and corrected. See [hmm_review.md](hmm_review.md) and the
[measured evaluator comparison](../outputs/experiments/evaluator_test_rf_hmm_001/README.md).
The original four-model status below is an archived handoff snapshot.

## Current status

Preprocessing and classifier training code are available. Four baseline models
have completed five-fold cross-validation. Recorded mean Top-1 accuracies are
Logistic Regression 85.21%, Gaussian NB 80.30%, Decision Tree 75.45% and Random
Forest 85.63%. These measure classifier predictions against valid derived
references, not decoded routes. See `outputs/experiments/run_001/` for metrics,
folds, environment, logs and small score samples.

AdaBoost, final five-model selection, Bayesian tuning and student-test evaluation
are pending. There is no final saved classifier or implemented HMM/Viterbi yet.
The archived samples are not complete emission tables or resumable run outputs.

## Reproduce the inputs

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
# Obtain the original supplied dataset and place train/ and student_test/
# inside data/raw/. The raw dataset is not stored in Git.
pathsnap check
pathsnap all
python -m pytest -q tests
```

The original task PDFs are in `docs/guidelines/`. Existing local data/raw is a
machine-specific symlink; a fresh clone needs its own dataset extraction.
Generated Parquets, full prediction CSVs, model binaries and Kaggle bundles
remain local. Feature generation reconstructs the Parquets and existing folds.

For full classifier training, use a fresh output directory:

```bash
python -m pathsnap.train --feature-dir outputs/features \
  --output outputs/training/hmm_baseline_001 --trials 30 --jobs 2
```

See `docs/training.md` for loading the final model via `selection.json` and
`docs/kaggle.md` for the optional AdaBoost handoff. The archived four-model
evidence must not be used as a completed training run.

## Decoder inputs and identifiers

- `data_io.load_trip(trip_id, partition, with_route=False)` reads GPS and roads
  without opening reference routes. `trip.track` contains longitude, latitude
  and seconds; `.nodes` row indices identify nodes; `.arcs` row indices identify
  directed arcs and its two columns identify source and destination nodes.
- `network.build_geom(trip)` provides projected metric coordinates, directed
  arc endpoints and arc lengths. Arc IDs are local to a trip.
- `features.build_candidate_table(geom)` produces `trip_id`, `point_idx`,
  `arc_id` and `config.FEATURE_COLUMNS`. Preserve the order of GPS observations,
  including observations with no candidates; do not sort observation IDs as
  strings. `frac_along` is the clamped position on a directed candidate arc.
- For `metrics.score_candidates`, rename `point_idx` to `observation_id` and
  `arc_id` to `candidate_id`. Supply the model's ordered feature names. Returned
  `probability` values are independent positive-class candidate probabilities.
  They are not normalized across an observation's candidates. Preserve the
  mapping back to integer arc IDs for graph lookup and maps.

## Next implementation steps

1. Define emissions from candidate probabilities, including clipping and any
   normalization. Compute Viterbi scores in log space to avoid underflow.
2. Build directed shortest-path distances using arc lengths in metres. Include
   the remaining distance on the previous arc and entry distance on the next
   arc; handle same-arc forward movement explicitly. Do not assume straight-line
   distance represents a reachable road path.
3. Define transition scores from network travel distance, observed displacement
   and time gaps. Specify disconnected-path, empty-candidate and long-gap
   policies, with explicit segment breaks when necessary.
4. Implement dynamic programming and backtracking over variable candidate sets,
   deterministic ties, and reconstruction of intermediate road arcs.
5. Test small directed graphs and compare decoded routes with classifier Top-1
   on held-out trips. Tune transition/emission parameters only within training
   folds, then report student-test results after choices are fixed.

Reference routes and derived labels are for fitting/evaluation only. Existing
references come from monotone alignment to `.route`, not independently supplied
pointwise truth. Training trips `00000033` and `00000054` remain excluded from
fitting/CV. Preserve those exclusions and the existing folds. See
`docs/data_quality_record.md` and `docs/training.md` for limitations.

`maps.make_trip_map(..., predicted_arcs=[...])` can display the reconstructed
route once the decoder produces ordered original arc IDs.
