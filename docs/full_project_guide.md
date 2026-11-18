# PathSnap: complete project explanation

This guide explains the implemented project as of 18 November 2026, based on source commit c74e243 on model-training. It is intended for learning, demonstration and viva preparation. The accompanying PDF adds the complete numbered Python application source, with explanations of statements and functions. Source line numbers refer to this snapshot.

## 1. What the project does

PathSnap performs GPS map matching. A GPS receiver gives noisy longitude, latitude and time measurements. These points may lie between roads, on the wrong side of a road, or closer to a parallel road than the road actually travelled. The goal is to infer a plausible sequence of directed road arcs for the complete trip.

A road arc is one permitted direction of travel between two nodes. A two-way street is normally represented by two opposite directed arcs. Arc IDs are row indices within the current trip network. They are not global labels shared between cities or trips.

The implemented inference pipeline is: raw GPS and road network -> metric geometry -> nearby road candidates -> 22 numerical features -> one fitted Random Forest -> candidate probabilities -> HMM/Viterbi -> observation matches and connected route segments. Reference routes are excluded from this prediction pipeline.

Random Forest estimates how suitable each candidate road is for a GPS observation. HMM/Viterbi considers the entire observation sequence and the directed road network. It can choose a lower-scoring candidate locally if that produces a stronger connected sequence overall. The system therefore combines learned local evidence with explicit route constraints.

The current repeat evaluator result is 90.79% pooled pointwise accuracy. It means 17,843 correct road choices among 19,653 observations with valid derived reference labels. It does not mean 90.79% of entire routes are perfect. The evaluator set was already examined before the improvement work, so this is a development comparison rather than a fresh untouched final estimate.

## 2. What is completed and what remains

Completed work includes dataset exploration; data-quality records; metric projection; candidate generation; 22 deterministic features; reference-label alignment; five whole-trip folds; recorded comparisons of four classifiers; a selected 200-tree Random Forest; reviewed directed HMM/Viterbi; training-only decoder tuning; student-test checks; repeated evaluator benchmarking; and a shared FastAPI backend with a documented request/response contract.

Both clients can use this one backend and one selected classifier. The two client applications themselves have not been completed in this branch. The endpoint currently matches trips already stored on the server. GPS uploading, causal streaming, a GeoJSON response, and browser CORS configuration remain integration work.

Five classifier types are implemented as development options: Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest and AdaBoost. The recorded comparison available here contains four completed baseline CV runs. The selected production artifact is a refitted baseline Random Forest, not a Bayesian-optimized classifier. Decoder grid selection is a separate optimization and does not establish completion of the original guidelines' classifier Bayesian-optimization requirement.

The old progress PDF is a historical snapshot from before HMM integration. Some source comments and generated exploration text also describe the earlier stage. In particular, docgen.py writes exploration-stage wording about evaluator data being absent. That wording is not a description of the current whole repository. Use the measured records and current sections of this guide for status.

## 3. Input files and dataset partitions

Every trip folder contains files named with an eight-digit trip ID. The .track file contains longitude, latitude and timestamp in seconds. The .nodes file contains road-node longitude/latitude coordinates. The .arcs file contains source and destination node indices; the row index is the arc ID. The .route file contains an ordered reference list of arc IDs, used only to derive labels and evaluate predictions.

The .route file has fewer rows than the GPS track and must never be joined to GPS observations by row number. One road arc can cover many GPS points, and the reference route can contain arcs with no nearby observation. labels.py aligns observations to positions along this ordered reference route.

Training contains 80 trips and 208,417 GPS observations. The generated candidate table contains 2,797,071 GPS/candidate pairs before training eligibility filters. Trips 00000033 and 00000054 have unreliable reference coverage and receive fold -1. The fitted classifier uses valid candidate labels from the remaining 78 trips, totalling 2,597,756 rows.

Student testing contains 10 trips and 19,160 observations. Of these, 19,133 have valid derived references and 27 are unresolved. Evaluator testing contains 10 different trips and 19,674 observations; 19,653 are known-reference observations and 21 are unresolved. All evaluator trips remain represented in reporting.

In this workspace, ../Student folder/ contains train/ and student_test/, while .. contains evaluator_test/. A data-root argument always means the parent of the requested partition directory. Raw data and the fitted model stay local and are excluded from Git; a fresh clone needs those artifacts or must rebuild/refit them.

## 4. Reading data and projecting geometry

data_io.py defines Trip and loads files. The crucial flag is with_route=False: inference loads the GPS track, road nodes and arcs without touching the reference route. with_route=True is used for exploration, label building and isolated evaluation.

geo.py chooses a UTM coordinate reference system from the trip's representative longitude/latitude and uses pyproj to convert coordinates into metres. This is necessary because longitude and latitude are angles, with different physical distance scales. network.py builds TripGeom containing projected GPS points P, timestamps t, arc start coordinates A, arc end coordinates B, arc lengths, bearings and node degrees.

For a GPS point P and arc endpoints A and B, projection fraction f = clip(((P-A) dot (B-A)) / ||B-A||^2, 0, 1). The closest point is C = A + f(B-A), and distance is ||P-C||. A zero-length arc uses a safe denominator and fraction zero. This calculation supplies both candidate distance and the location of an observation along an arc.

Compass bearing is measured clockwise from north. The angle-difference function returns a value between 0 and 180 degrees. Reverse-direction arcs can have identical distance features, so heading and progress features are essential for distinguishing them.

## 5. Candidate generation

candidates.py first crops the road network to the GPS bounding box plus the search radius. It then creates Shapely line segments, builds an STRtree spatial index and queries arcs within 75 metres of each GPS point. Original arc IDs are preserved after cropping.

The candidates are sorted by observation and distance. At most 40 nearest arcs are retained per observation. A candidate row identifies one particular (trip_id, point_idx, arc_id) combination. Its fraction along the arc is clipped to [0,1]. The classifier predicts a score for each row, not a global arc class covering every possible road in the dataset.

Candidate generation limits the model's possible answers. If the correct reference arc is missing, neither Random Forest nor Viterbi can select it as an observation's state. Candidate misses therefore count as errors for known-reference point accuracy. A point with no candidates still belongs to the complete observation inventory.

The current evaluator has no candidate misses among valid derived references. This needs context: reference validity requires an aligned arc within 50 metres, and candidates use a 75-metre radius. The thresholds make high known-reference recall expected; this is not proof of perfect recovery for arbitrary noisy GPS points.

## 6. The 22 input features

All features are calculated from GPS, time and the road network. features.py never uses a reference route. The same feature order and numerical precision must be used during training and serving. The following feature dictionary is also generated directly from the configured feature names in the PDF.

Distance features answer how near the point is to the road and how it compares with other candidates. Position features describe projection fraction, road length and distances to endpoints. Direction features compare estimated travel heading with permitted arc direction. Motion features describe speed and sampling gaps. Context features ask whether neighbouring observations also fit the candidate. Topology features describe nearby directed connectivity.

point_motion uses valid preceding/following observations to estimate central displacement, speed and heading. Neighbours are ignored when outside the track or too distant in time. The neighbour gap bound is 10 seconds times the absolute offset. Heading is treated as unreliable below 1 m/s; the corresponding heading difference becomes neutral at 90 degrees and cosine becomes zero.

The context window uses up to three preceding and three following observations. For each candidate arc it measures distances from those neighbouring GPS points to that same arc. Neighbours within half the candidate radius, 37.5 metres, contribute hits. When no neighbours are available, relevant distance summaries fall back to the current point's distance.

Because these features include future observations, the present system is an offline complete-trip matcher. Real-time causal streaming would need an agreed delay or redesigned features and fresh validation. The existing backend receiving a stored full trip fits the present feature definition.

## 7. How labels are derived

labels.py computes the distance from each GPS observation to every reference-route position. Distances used in alignment costs are capped at 150 metres so a few remote points have bounded influence. The uncapped distance is retained for reference-validity checks.

It solves a monotone dynamic program: choose route positions j_1 <= j_2 <= ... <= j_N minimizing the sum of capped point-to-arc distances. Staying at a route position or jumping forward is allowed; moving backward through reference order is not. A running prefix minimum makes the forward calculation O(N*M), where N is observations and M is reference-route positions.

Backtracking gives one reference arc and distance per observation. A label is valid only when the uncapped distance is at most 50 metres. A candidate gets binary label 1 when its arc ID matches the aligned arc, and 0 otherwise. For unresolved observations the training adapter blanks labels rather than treating every candidate as a reliable negative.

These labels are derived from supplied reference routes, not independently hand-verified per-point truth. Reference gaps, route discontinuities and monotone-alignment assumptions can introduce label noise. This is why accuracy must always be described as accuracy against valid derived reference labels.

Do not confuse this reference-alignment DP with inference Viterbi. The first uses a reference route to produce labels for learning/reporting. The second uses classifier scores and the road graph to predict routes without reading a reference route.

## 8. Tables, validation and cross-validation

build.py writes partition candidate Parquets, complete point inventories, a feature schema and build logs. Continuous feature columns are stored as float32, while identifiers and counts use integer types. api.py and evaluate_forest.py explicitly cast the same columns to float32 to keep serving inputs consistent with training.

training_handoff.py translates point_idx/arc_id into observation_id/candidate_id and validates consistency between point inventories and candidates. data.py rejects missing IDs, duplicates, invalid labels, overlapping partitions, infinite features, mismatched reference fields and forbidden identifier/reference features. The 22 feature names must agree with config.py in the same order.

Five folds keep every eligible trip wholly within one validation fold. This prevents neighbouring points from the same journey appearing on both sides of a validation split. The native folds greedily balance point counts; excluded trips retain fold -1. The original assignments are preserved for comparison and decoder tuning.

train.py fits preprocessing within each fold's training data. Median imputation is part of the sklearn Pipeline. Scaling is included for Logistic Regression and Gaussian Naive Bayes, but not for the tree models. Out-of-fold probabilities are generated only by a model that did not fit that trip's labels.

The recorded mean-fold classifier Top-1 accuracies are Logistic Regression 85.21%, Gaussian Naive Bayes 80.30%, Decision Tree 75.45%, and Random Forest 85.63%. These are classifier-only cross-validation scores. They must not be compared directly as if they used the evaluator denominator or included HMM decoding.

## 9. The selected Random Forest

models.py builds a Pipeline containing median imputation and a RandomForestClassifier. The selected artifact uses 200 trees and seed 42, with the other baseline estimator settings preserved. Jobs controls worker parallelism and does not choose a different model family. Random Forest combines many randomized decision trees; sklearn averages their positive-class probability estimates.

For every candidate feature row, predict_proba returns a probability for each fitted binary class. Code locates class 1 in classes_ rather than assuming it is always column 1. The positive score indicates compatibility with the derived positive label. Scores are not normalized across a GPS observation's candidate set and are not asserted to be calibrated HMM probabilities.

Classifier-only matching ranks candidates independently by positive score and chooses one per observation. Exact ties use lexical candidate IDs in the classifier metric implementation. The HMM sorts numeric arc IDs and uses the first maximum; these are distinct documented tie conventions.

selected_forest.py fits the fixed Random Forest once on all eligible labelled training candidates, writes configuration and provenance, saves a compressed pipeline.joblib, and checks that reloading it reproduces predictions. It scores student testing after fitting. No evaluator data is used for fitting or selecting this classifier artifact.

The frozen model SHA-256 is 726b503c4ec431232411dcc94f2b443156d6ea43f2ee7bf0fcb8e19a29dc2399. The model is about 211 MB and remains outside Git. The selected decoder configuration records this same hash, preventing accidental use with a different classifier artifact.

## 10. HMM and Viterbi in this project

The hidden state at observation t is a candidate directed road arc. The observation is the GPS measurement plus its derived local features. Emission evidence comes from Random Forest. Transition evidence comes from possible movement through the directed network. The implemented scores are HMM-style log potentials; the code does not fit a conventional generative HMM or normalize every outgoing transition distribution.

In the selected probability emission mode, E_t(j) = log(clip(p_t(j), 1e-12, 1)). This protects the logarithm from zero. An optional log-odds emission mode is implemented and tested, but was not selected. Log odds are a score transformation, not a fitted calibration model.

For a possible move from candidate i to j, compute directed network travel distance d_net and observed straight-line displacement d_gps. Transition log potential T(i,j) = -abs(d_net - d_gps) / beta. Unreachable or over-limit transitions receive negative infinity. Smaller beta makes distance disagreement more costly.

The recurrence is V_t(j) = E_t(j) + max_i[V_(t-1)(i) + T(i,j)]. The maximizing predecessor is stored for every reachable state. Backtracking after a connected block ends returns the strongest candidate sequence for that block. Subtracting the same maximum score from all current states preserves their ranking and controls numeric magnitude.

A small illustrative example makes this concrete. At the first point, arcs A and B have scores 0.8 and 0.6. At the next point, C and D have scores 0.4 and 0.9. Suppose transition log penalties are A-to-C = -0.1, A-to-D = -4, B-to-C = -0.3 and B-to-D = -3. Candidate-only ranking selects A then D. Viterbi instead compares whole-sequence scores: A-C is log(0.8)+log(0.4)-0.1 = -1.239; A-D is -4.329; B-C is -1.727; B-D is -3.616. A-C wins because it has the greatest total log score, despite C having a lower local probability than D. These numbers illustrate the recurrence and are not measured dataset results.

Viterbi is needed because independently choosing the best local score can switch between parallel roads, take an impossible direction or choose disconnected arcs. Its result is the best sequence under these chosen potentials and constraints; it is not a guarantee of the actual driven route.

## 11. Directed shortest paths and partial arcs

RoadGraph stores an adjacency list with original arc IDs and metric lengths. It uses bounded Dijkstra searches from the end node of a previous candidate. The cache is an LRU limited to 512 entries. Search bounds are rounded upward to 100-metre increments for reuse, while each actual transition still enforces its exact limit. It does not precompute every pair of nodes.

For different arcs, d_net = remaining distance on previous arc + shortest directed path between endpoints + travelled distance on current arc. For example, a 100-metre previous arc at fraction 0.8 contributes 20 metres; a 50-metre current arc at fraction 0.3 contributes 15 metres. If the intermediate path is 40 metres, total travel is 75 metres.

For forward motion on the same arc, distance is simply the positive fraction change times arc length. For a backwards projection of at most the configured jitter tolerance, distance is treated as zero: the observation is held on the same directed road. It does not add a reverse road edge. Larger backwards changes require a real directed return path.

The transition travel limit is min(5000 m, 60 m/s * elapsed seconds + 2 * 75 m). The extra 150 metres allows candidate projection endpoints to differ from GPS positions. A time gap greater than 30 seconds starts a new block. These limits are fixed defaults unless explicitly overridden by a validated decoder configuration.

RoadGraph retains parallel directed arcs and uses lengths for shortest paths. Reconstructed route segments include intermediate arcs even when no observation selected them. This makes route output more informative than just a list of per-observation predicted arcs.

## 12. Missing candidates, breaks and reviewed fixes

DecodeResult contains point_arc_ids, segment_ids, route_segments and break_reasons. There is exactly one point_arc_ids entry for each GPS point. An unmatched observation uses arc ID -1 and segment ID -1. Observation positions are never silently dropped.

If a point has no candidates, the current block is finished and a no_candidates break is recorded. A long time gap records time_gap and starts a new block. If every candidate transition is unreachable, the old block is independently backtracked, a no_reachable_transition break is recorded and decoding restarts at the current observation.

This independent backtracking fixes a restart error in the earlier decoder, which could invent predecessor index zero and corrupt an already valid block. Other reviewed changes include metric rather than hop-count routing, bounded local search rather than whole-network all-pairs work, correct partial arcs, explicit missing observations, safe log probabilities, deterministic ties and directed reconstruction checks.

Each route segment should be drawn separately. Joining segments with a straight line would imply a road connection the decoder did not establish. The API's output arrays retain enough information for clients to display matched and unmatched observations explicitly.

## 13. Why accuracy improved

The original integrated HMM used beta 50 metres, zero same-arc jitter tolerance and probability emissions. It scored 75.41% on the evaluator set, below the classifier-only result of 80.76%. A strict sequence model can worsen results when GPS projection noise creates implausible backward changes or its transition penalties do not suit the data.

The improvement reused the exact frozen Random Forest and candidate features. tune_hmm.py read original training out-of-fold candidate scores, checked input hashes and fold alignment, and compared decoder settings without opening evaluator data. It screened 25 configurations on ten training trips, two per fold, deterministically selected with seed 42.

The search varied beta over 25, 100, 500 and 2000 metres, jitter tolerance over 0, 5 and 15 metres, and emission mode over probability and log odds. The historical baseline was included as an additional configuration. The three strongest screened candidates plus the original baseline were evaluated on all 78 eligible training trips.

Selection maximized mean-fold pointwise accuracy, then mean-trip accuracy, then used trial order for exact ties. Trial 5 won: beta 25 metres, same_arc_jitter_m 15 metres, emission_mode probability. The selected settings were written before student-test trips were loaded. Student testing then checked the frozen settings once, and evaluator results were measured afterward as a repeat development benchmark.

Selected training OOF mean-fold accuracy is 91.47%. Because those labels selected decoder hyperparameters, this is a selection score, not an independent post-selection estimate. Student-test accuracy is 92.19%. The repeat evaluator result is 90.79%. Neither number establishes performance on every new city or GPS sensor.

## 14. Results and their meaning

The PDF reads its results table directly from archived JSON/CSV evidence. On the evaluator set, nearest-road ranking achieved 49.80%, Random Forest alone 80.76%, original RF plus HMM 75.41%, and selected RF plus HMM 90.79%. All three model comparisons use the same 19,653 valid derived reference observations.

The HMM improvement over the original is 15.38 percentage points. It makes 3,022 additional correct point choices, reducing errors from 4,832 to 1,810 on this benchmark. Compared with Random Forest alone, the selected decoder adds 1,971 correct choices, a gain of 10.03 percentage points.

Pooled accuracy divides total correct known-reference points by total known-reference points. Mean-trip accuracy averages the ten trip percentages equally. Selected evaluator mean-trip accuracy is 94.11%, while pooled accuracy is 90.79%. The long, difficult trip 00000015 contributes 6,833 known points and therefore has more influence in the pooled result.

The selected HMM does not improve every trip. Trip 00000023 changes from 94.10% with RF alone to 93.67% with the selected HMM. Reporting this prevents the aggregate improvement from hiding local regressions. Trip 00000015 reaches 80.52%, still below most other trips and an appropriate case for future training-led error analysis.

Macro F1 and log loss in the evaluator summary describe binary candidate classification, not reconstructed HMM routes. The HMM score in this guide is pointwise Top-1. No whole-route exact accuracy, path edit distance, route precision/recall or independent route geometry score has been established by these records.

## 15. How evaluation prevents label leakage

evaluate_forest.py first validates the completed model, ordered feature configuration and artifact hash. It rejects evaluator IDs overlapping any original training or student-testing trip, including excluded training trips. The decoder configuration must bind to the same classifier and record evaluator-free selection.

Prediction phase loads each trip with with_route=False, creates candidates and scores, and writes classifier and decoded predictions. It does this for all evaluator trips before loading any evaluator .route file. The prediction outputs and model are then frozen for the evaluation phase.

Only the second phase loads reference routes, derives valid labels, joins frozen predictions by observation identity, computes metrics and writes reference-bearing comparisons into separate evaluation folders. Input and score hashes are checked, and the model hash is rechecked before completion. Reconstructed road segments are checked for directed endpoint adjacency.

The evaluator set had already been examined when the improvement task began. Keeping it out of decoder selection prevents direct parameter selection on its scores, but repeated reporting does not turn it into a newly untouched test. A fresh unseen final set would be needed for a stronger independent performance claim.

## 16. One model, one backend, two clients

Client A and Client B send HTTP requests to the same FastAPI service. The server loads one Random Forest artifact and one selected decoder configuration. Each request supplies a trip ID and partition; the server builds that trip's geometry and candidates, scores them, decodes them and returns matches. Each trip has its own graph and decoding arrays.

Clients do not need to install sklearn, load the 211 MB model or reproduce feature generation. Both receive results from the same feature order, model hash and decoder settings. Accuracy development can continue by producing a new validated configuration, while the HTTP request and response shape can stay stable.

GET /health reports status, model_loaded and decoder_configured. A healthy response alone does not establish that both client applications work, that raw data exist for every trip, or that matching quality is sufficient. GET /docs gives the generated API documentation. POST /match invokes matching.

The request is {"trip_id":"00000023","partition":"evaluator_test"}. The trip ID is constrained to eight digits. Supported partition values are train, student_test and evaluator_test. An optional data_root exists in the current request schema; for a shared deployment, server-controlled dataset access and request limits should be designed before exposing the service broadly.

The response contains trip_id, partition, predicted_arc_ids, segment_ids, route_segments, break_reasons, number_of_gps_points, number_of_candidates and model_used. It does not return labels or evaluator accuracy. Arc IDs are local to the requested trip, so clients must use the matching road network to draw them.

Errors include 503 when no classifier is loaded, 404 for missing GPS data, 422 for malformed input or a trip with no candidates, and 400 for an unsupported partition. A network-ready client integration still needs a reachable server address, explicit allowed browser origins when relevant, geometry display and an agreed upload/request workflow.

## 17. Run the project from a terminal

Run the following commands from the repository root. The project requires Python 3.10 or later according to pyproject.toml; use a compatible environment and the recorded environment files when reproducing exact artifacts. Installation builds no model by itself.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
export PYTHONPATH=src
```

Place train/ and student_test/ under data/raw/, or provide the parent directory explicitly. Exploration builds the summaries needed by the feature/fold stage. Full regeneration and fitting can take substantial time; these commands describe reproduction and are not rerun to create this PDF.

```bash
python -m pathsnap.cli --data-root '../Student folder' check
python -m pathsnap.cli --data-root '../Student folder' explore
python -m pathsnap.cli --data-root '../Student folder' build-features
python -m pathsnap.selected_forest --feature-dir outputs/features \
  --output outputs/training/random_forest_new --jobs 2
```

Use the existing local artifact and committed selected decoder to serve the evaluated version. PATHSNAP_DATA_ROOT must contain the requested partition. The example below serves evaluator_test from the current workspace parent; use ../Student folder for student_test.

```bash
export PYTHONPATH=src
export PATHSNAP_MODEL_DIR=outputs/training/random_forest_selected_001
export PATHSNAP_DECODER_CONFIG=outputs/experiments/hmm_tuning_001/decoder_configuration.json
export PATHSNAP_DATA_ROOT=..
python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000
```

In another terminal, check the backend and send a request. These commands work after the service has started and the local model/raw files are available.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/match \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"00000023","partition":"evaluator_test"}'
```

To reproduce selected-decoder evaluation, choose a fresh empty output directory. The command never fits the model and the model hash must match its saved run metadata. It needs the original feature inventories for partition-separation checks.

```bash
python -m pathsnap.evaluate_forest \
  --model-dir outputs/training/random_forest_selected_001 \
  --data-root .. --with-hmm \
  --decoder-config outputs/experiments/hmm_tuning_001/decoder_configuration.json \
  --output outputs/evaluation/evaluator_repeat_new
python -m pytest -q tests
```

To rerun decoder selection, retain the original full OOF scores and local model artifact. They are deliberately not in the compact Git archive. The defaults expect outputs/training/run_001 and outputs/training/random_forest_selected_001. Tuning outputs must use a fresh directory.

```bash
python -m pathsnap.tune_hmm \
  --feature-dir outputs/features --raw-root '../Student folder' \
  --cv-run outputs/training/run_001 \
  --model-dir outputs/training/random_forest_selected_001 \
  --output outputs/training/hmm_tuning_new
```

## 18. Repository map and saved evidence

src/pathsnap contains the application modules. tests contains pipeline, handoff, classifier, HMM, tuning, evaluator, API and Kaggle checks. docs contains written guides, supplied guideline PDFs and generated reports. notebooks contains learning/model demonstrations and the optional AdaBoost Kaggle workflow. scripts contains reporting and terminal-run helpers.

outputs/experiments/run_001 stores the compact four-model baseline comparison. outputs/experiments/evaluator_test_rf_hmm_001 preserves the original 75.41% decoder benchmark. outputs/experiments/hmm_tuning_001 stores trial records, selected settings and student validation. outputs/experiments/evaluator_test_rf_hmm_tuned_001 stores the repeat 90.79% benchmark with per-trip metrics and provenance.

outputs/training and outputs/evaluation contain local large artifacts and full predictions and are ignored by Git. data/raw is also local; its current symlink is not a portable dataset package. The Git repository supplies implementation and compact evidence, not every file needed for a clone to start inference immediately.

The latest application snapshot discussed here is c74e243 on model-training. This document generator records a SHA-256 and line count for every application source file so the numbered walkthrough can be matched to the exact source version. Documentation changes after that commit do not change the measured model.

## 19. Verification and limits

The recorded completed code validation contains 51 passing tests. Tests cover metric geometry, label-free feature generation, preserved trip folds, strict table validation, score alignment, candidate misses, model reloads, shortest paths, partial arcs, cycles, disconnected restarts, missing observations, time gaps, jitter tolerance, Viterbi agreement with exhaustive small examples, evaluator freeze ordering and decoder selection ordering.

A real backend smoke request for trip 00000023 returned all 458 GPS matches and agreed exactly with the saved selected-decoder predictions. This is evidence of API/benchmark consistency for that request, not a load test, two-client acceptance test, or public-deployment security review. Creating this PDF reads artifacts and source without refitting or changing predictions.

Remaining scientific limitations include derived label noise, only ten evaluator trips, prior evaluator exposure, offline features, a selected baseline rather than a fully optimized classifier, and no established whole-route quality metric. Useful next work is client integration, richer route geometry, training-led error analysis, and an agreed fresh final evaluation.

## 20. Explanation for a presentation or viva

What is the target? Infer the most plausible directed road sequence from noisy GPS observations. Why not nearest road? Distance cannot reliably distinguish parallel streets, opposite directions or connected movement; the nearest-road evaluator baseline is 49.80% here.

Why Random Forest? It was selected from the recorded baseline comparison, handles nonlinear feature interactions and supplies positive candidate probabilities. Only one selected forest runs in the backend. The other classifier implementations are development comparisons, not extra runtime clients or models.

Why HMM/Viterbi? It combines local evidence with temporal road connectivity. Random Forest alone scores candidates independently. Viterbi finds the highest-scoring connected sequence under the emission/transition definitions and breaks explicitly when a continuous path cannot be supported.

How did accuracy increase? Keep the classifier fixed, select beta and small same-arc projection tolerance using training OOF predictions, freeze settings, then validate. The selected settings are beta 25 m, jitter 15 m and probability emissions. The repeated evaluator score improved from 75.41% to 90.79% on the same valid-reference denominator.

What is the honest accuracy statement? The selected RF-plus-HMM pipeline scored 90.79% pooled pointwise accuracy on a repeat evaluator development benchmark and 92.19% on student testing, against valid derived reference labels. Do not call it whole-route accuracy or an untouched official final test result.

How do two clients work? Both call the shared API, which holds one frozen classifier and decoder configuration. They use returned trip-local arc IDs and segment boundaries to display the matched trip. Both client implementations and their end-to-end acceptance checks remain to be completed.

## 21. How to use the numbered source walkthrough

Read the narrative first, then follow application files in pipeline order. Every physical source line is included in the appendix, including imports, comments and docstrings. Long printed lines wrap with an indented continuation; only the original physical line receives a number. Original line numbers are preserved even when a code block spans pages.

Each module starts with its role and a function/class map. The following code blocks have statement-level notes linked to exact source lines. A statement spanning several Python lines is explained once at its starting line and range; its continuation lines finish that same statement. Comments and docstrings are readable directly in the code. This grouping keeps multiline calls, arrays and dictionaries understandable.

For example, a validation if statement explains its condition and its associated raise statement explains the error. An assignment explains the destination and operation. A return explains the returned value. Detailed project-specific meanings are supplied by the module/function descriptions and the chapters above. Nested helper functions, decorators and exception handlers also remain visible.

The appendix covers every .py application module under src/pathsnap. Tests, notebooks and report scripts are mapped and explained in the main guide but are not duplicated in full; they are supporting tools rather than additional runtime model stages. The final evidence appendix lists source hashes, input metrics and test names for checking this guide against the repository.
