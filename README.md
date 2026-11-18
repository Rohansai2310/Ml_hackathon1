# PathSnap - GPS map-matching (data, candidates & features stage)

Noisy GPS points -> road-network arcs. Pipeline: **raw files -> candidate arcs per GPS point -> features -> classifier -> HMM/Viterbi -> route -> API/UI**.
This repository includes preprocessing, candidate generation, feature engineering,
classifier training, metric HMM/Viterbi decoding and a FastAPI inference backend.
See [the HMM review and usage guide](docs/hmm_review.md) for integration details.

Start two-client integration with [the shared backend API contract](docs/CLIENT_API.md).
[Training-only HMM tuning](docs/hmm_tuning.md) runs separately and can update the
decoder without changing the client request/response contract.

Random Forest is now the selected classifier. See [selected-model fitting and
evaluator-test evaluation](docs/random_forest_evaluation.md) for the fixed-model
workflow. The earlier project PDF and four-model archive describe the preceding
snapshot; evaluator results are recorded in a separate experiment archive.

The completed [RF versus RF + HMM evaluator benchmark](outputs/experiments/evaluator_test_rf_hmm_001/README.md)
records 80.76% versus 75.41% pointwise Top-1 on valid derived references. HMM
produces connected route segments, but these untuned decoder settings did not
improve pointwise accuracy. See the archive for per-trip results and coverage.

## 1. Where to extract the dataset
Extract `Student_folder.zip` so the repo looks like this (the zip contains a folder called `Student folder` - move its two sub-folders):

```
pathsnap/
  data/raw/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route     (80 trips)
  data/raw/student_test/<trip_id>/...                              (10 trips)
```
Linux/macOS: `unzip Student_folder.zip -d /tmp/sf && mv "/tmp/sf/Student folder/train" "/tmp/sf/Student folder/student_test" data/raw/`
Windows: extract, then copy the `train` and `student_test` folders into `data\raw\`.
Anywhere else? use `--data-root <folder containing train/ and student_test/>`.
Evaluator-test data is **never** placed here.

## 2. Install
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 3. Run (about 1.5 minutes in total)
```
pathsnap check                  # dataset in the right place? (80 + 10 trips)
pathsnap explore                # inventory, missing values, connectivity, plots, radius study, data-quality record
pathsnap build-features         # candidate + feature tables (parquet) and CV folds
pathsnap feature-report         # evidence for "why these features"
pathsnap map --trip 00000048 --partition train          # interactive Folium map
pathsnap inspect-trip --trip 00000048 --partition train # print candidates/features for a few points
pathsnap all                    # check + explore + build-features + feature-report
python -m pytest -q tests       # 5 tests incl. leakage guard
```
(`python -m pathsnap <cmd>` works too.)

## 4. What is produced
| Path | Content |
|---|---|
| `outputs/eda/trip_summary.csv` | one row per trip: counts, gaps, speeds, connectivity, route checks, label quality |
| `outputs/eda/dataset_inventory.csv`, `missing_value_analysis.csv`, `radius_study.csv`, `feature_report.csv`, `folds.csv` | report tables |
| `outputs/eda/01..12_*.png` | EDA figures (trip length, sampling, speed, networks, geography, reference quality, trajectories, radius, features) |
| `outputs/maps/*.html` | Folium maps (GPS, road network, reference route, optional predicted route) |
| `outputs/features/{train,student_test}_candidates.parquet` | one row per (GPS point, candidate arc) with 22 features + `label`, `label_valid`, `fold` |
| `outputs/features/{train,student_test}_points.parquet` | one row per GPS point: lon/lat/time, derived reference arc, `label_valid` |
| `docs/data_quality_record.md` | auto-generated issues / assumptions / decisions |
| `docs/report_section_2_2.md` | draft answers for report section 2.2 with real numbers |
| `docs/viva_notes.md` | plain-language explanation of every design choice |

## 5. Using it from the next stages
```python
from pathsnap.dataset import load_xy              # Person 2 (models): same rows + same folds for every model
Xtr, ytr, gtr, ftr, mtr = load_xy("train")        # fold >= 0 and label_valid only
Xte, yte, gte, _, mte = load_xy("student_test")

# Person 3 (API, inference - NO labels, never reads .route):
from pathsnap import data_io as io
from pathsnap.network import build_geom
from pathsnap.features import build_candidate_table
trip = io.load_trip(trip_id, "evaluator_test", with_route=False, root=EVAL_ROOT)
cand = build_candidate_table(build_geom(trip))     # DataFrame: trip_id, point_idx, arc_id + config.FEATURE_COLUMNS
# model.predict_proba(cand[config.FEATURE_COLUMNS]) -> HMM over (point_idx, arc_id)

# Person 4 (maps): predicted route overlay
from pathsnap.maps import make_trip_map
make_trip_map(trip_id, "student_test", predicted_arcs=[...arc ids in order...])
```

## 6. Code map
`data_io.py` read files (leakage-safe `with_route=False`) - `geo.py` UTM projection, point-to-segment, bearings - `network.py` projected trip geometry - `candidates.py` radius query - `labels.py` point->reference-arc alignment - `features.py` the 22 features - `build.py` parquet + folds - `eda.py`/`docgen.py` exploration - `maps.py` Folium - `cli.py` commands - `config.py` every constant.

## 7. Person 2: train and compare five models

The training stage adds Logistic Regression, Gaussian Naive Bayes, Decision Tree,
Random Forest and AdaBoost with shallow trees. It consumes the existing Parquet
files directly, preserving their five trip folds and the two documented excluded
training trips. It fits imputation, and scaling where required, inside each fold.

Run from the repository root after installing dependencies and building features:

```bash
PYTHONPATH=src python -m pathsnap.train \
  --feature-dir outputs/features \
  --output outputs/training/run_001 \
  --trials 30 --jobs 2
```

If you installed the package with `pip install -e .`, `PYTHONPATH=src` is optional.
Use a fresh output directory for every run; incomplete runs retain diagnostic
files but are not automatically resumed. The full dataset and 30 tuning trials
can take substantially longer than feature generation.

The comparison uses observation-level Top-1 accuracy for model selection, with
binary candidate log loss and scoring time as tie breakers. Thirty Optuna TPE
trials tune the best baseline using the same folds. Student-testing results are
generated after selection and tuning. If tuning worsens the CV ranking, the
baseline remains the final artifact, while both are retained for comparison.

`outputs/training/run_001/` contains configurations, environment versions,
input checksums, fold assignments, CV metrics/predictions, Optuna SQLite/CSV
history, model pipelines, student-testing predictions and comparison plots.
`selection.json` identifies the final model. Training outputs are ignored by
Git; publish selected measured results and artifacts deliberately when preparing
the submission.

Read [the training and review notes](docs/training.md) for metric definitions,
label limitations, inference integration and verification commands.

AdaBoost can run separately using [the Kaggle notebook](notebooks/adaboost_kaggle.ipynb).
See [Kaggle handoff instructions](docs/kaggle.md) for the upload bundle and how to
import results before resuming final selection and tuning.

For a viva or demonstration, open the [four-model walkthrough](notebooks/four_models_walkthrough.ipynb).
It shows real data, the training implementation, configurations, fold results and
original logs. See [demonstration instructions](docs/four_model_demo.md) for an
optional live rerun.
