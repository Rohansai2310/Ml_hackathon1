# HMM/Viterbi review and integration

Update: subsequent training-only decoder tuning improved student-test accuracy
to 92.19% and the repeat evaluator benchmark to 90.79%. See
[the completed tuning record](../outputs/experiments/hmm_tuning_001/README.md)
and `docs/CLIENT_API.md` for loading the selected configuration. The original
review and untuned benchmark below remain as historical evidence.

Integrated `person-3` commit `45edc6f`, preserving the teammate's branch history.
The selected Random Forest is refitted on 78 eligible training trips with 200
trees and seed 42. The fitted artifact is local; code and compact results are
versioned. The decoder and API never need reference routes for prediction.

## Changes made during review

- Replaced all-node all-pairs BFS with directed, metric, bounded Dijkstra searches
  cached on demand. Parallel arcs retain their own lengths and identities.
- Replaced hop counts/fractional arc units with metres, including the remaining
  distance on the previous arc and entry distance on the next arc. Transition
  potentials compare network travel with observed GPS displacement.
- Removed free backward movement on the same directed arc. It now requires a
  valid directed return path through the network.
- Fixed restart backtracking: a disconnected segment no longer forces the final
  state of the previous segment to candidate zero.
- Preserve one output per GPS observation. Empty candidate sets use arc ID -1.
  Long gaps, absent candidates and unreachable transitions have explicit segment
  breaks. No fake edges join disconnected segments.
- Reconstruct intermediate road arcs between selected candidate arcs. Every
  output route segment is checked for directed adjacency during benchmarking.
- Added strict validation, deterministic numeric arc-ID ties, probability
  clipping, log-space scoring and common-offset score stabilization.
- Connected the API to metric geometry and full observation output; added
  trusted artifact/configuration/hash loading through `PATHSNAP_MODEL_DIR`.
  Candidate precision matches the training Parquet representation.

The emission potential is the clipped classifier positive probability. It is
not claimed to be calibrated across a candidate set. Transition log potential
is `-abs(network_distance_m - gps_displacement_m) / beta_m`.

Fixed defaults: beta 50 m; maximum time gap 30 s; maximum speed 60 m/s; maximum
transition distance 5,000 m. The per-step search also respects
`60 * elapsed_seconds + 2 * candidate_radius` metres. These are initial design
settings and have not undergone decoder optimization. They were fixed before
evaluator references were read.

## Verification and measured outcome

All 46 tests passed, including brute-force versus Viterbi agreement on a small
graph, metric/parallel arc distances, directed return paths, missing points,
segment backtracking, bounded search behavior, reference-free HTTP inference,
and frozen prediction/evaluation separation.

On ten evaluator trips (19,674 observations), 19,653 have valid derived reference
labels and 21 are unresolved. Random Forest pointwise Top-1 is **80.76%**;
RF + HMM pointwise Top-1 is **75.41%**; nearest-arc Top-1 is **49.80%**.
The completed decoder produces connected route segments but lowers pointwise
accuracy under these settings. Do not present it as an accuracy improvement.
The benchmark uses derived route-alignment labels, not independent pointwise
truth or a measured route-level accuracy score.

See [the result archive](../outputs/experiments/evaluator_test_rf_hmm_001/README.md)
for per-trip metrics, coverage, frozen model hash and route summaries. The
evaluator results were not used to change model or decoder parameters. Any
future decoder optimization must use training folds/student-testing data.

## Run in VS Code's integrated terminal

```bash
cd Ml_hackathon1
bash scripts/run_hmm_evaluation.sh
```

The script prints its commands, runs tests, evaluates RF and RF + HMM, and
exports compact measured evidence. It reuses the saved model and chooses a new
timestamped output directory. Optionally pass a fresh run name as its first
argument. The completed run's console
output was visible in a VS Code integrated terminal.

To serve the shared backend locally, activate the environment and run:

```bash
export PYTHONPATH=src
export PATHSNAP_MODEL_DIR=outputs/training/random_forest_selected_001
export PATHSNAP_DATA_ROOT=..
python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000
```

`POST /match` with `{"trip_id":"00000023","partition":"evaluator_test"}`
returns one arc ID per observation, segment IDs, reconstructed route segments
and explicit break reasons. It never reads `.route`. The configured root is the
parent of `evaluator_test/`; use the original Student folder root for train or
student-testing requests. `/health` reports whether a model is loaded.

Known scope: GPS features use neighbouring future observations, so this is an
offline full-trip pipeline. Decoder parameter optimization and independently
verified route-level scoring are not claimed by this integration.
