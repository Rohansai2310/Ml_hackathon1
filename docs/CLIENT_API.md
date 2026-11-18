# Shared backend for two clients

Use the `model-training` branch. Both clients call the same FastAPI backend;
neither client needs to load a model or run HMM locally. The backend loads one
frozen Random Forest and runs the same feature generation/decoder for requests.

## Start the backend

From `Ml_hackathon1/`, activate the Python environment and install requirements.
Set paths on the **server**, where the original dataset and fitted model exist:

```bash
export PYTHONPATH=src
export PATHSNAP_MODEL_DIR=outputs/training/random_forest_selected_001
export PATHSNAP_DECODER_CONFIG=outputs/experiments/hmm_tuning_001/decoder_configuration.json
export PATHSNAP_DATA_ROOT=..
python -m uvicorn pathsnap.api:app --host 127.0.0.1 --port 8000
```

`PATHSNAP_DATA_ROOT` is the parent of partition folders. In the current workspace,
`..` contains `evaluator_test/`; `../Student folder/` contains `train/` and
`student_test/`. A combined dataset root can instead contain all three partitions.
Use the correct server root for the partition that clients request.

The actual fitted model and raw data stay local, outside Git. A fresh clone
requires those artifacts or the documented feature-build/refit commands. See
`docs/random_forest_evaluation.md`. The current local demo runs on port 8077.

## Health and matching

`GET /health` returns:

```json
{"status":"healthy","model_loaded":true,"decoder_configured":true}
```

Both clients send the same request shape to `POST /match`:

```json
{"trip_id":"00000023","partition":"evaluator_test"}
```

Response fields:

| Field | Meaning |
|---|---|
| `trip_id`, `partition` | Identifies the server-side dataset trip |
| `predicted_arc_ids` | One directed arc ID per GPS observation; `-1` means unmatched |
| `segment_ids` | One segment ID per observation; `-1` means unmatched |
| `route_segments` | Ordered arc lists, including intermediate road arcs, per connected segment |
| `break_reasons` | Observation index to `no_candidates`, `time_gap` or `no_reachable_transition` |
| `number_of_gps_points` | Complete input observation count |
| `number_of_candidates` | Number of GPS/candidate pairs scored |
| `model_used` | Classifier type, currently `RandomForestClassifier` |

Arc IDs and node IDs are local to a trip. Preserve observation order. Draw each
route segment separately; do not draw a connector across explicit segment breaks.
Reference routes and accuracy labels are not returned by the prediction endpoint.
Route geometry is not returned as GeoJSON yet. A mapping client must have the
matching trip road network or agree on a geometry response with the backend team.

HTTP errors include 422 for malformed input/no candidates, 404 for missing GPS
data and 503 if no model is loaded. `/docs` provides the OpenAPI contract.

## Client integration scope

The current endpoint selects trips stored on the server. GPS upload, streaming
matching, browser CORS settings and an accuracy/evaluation endpoint are not part
of this contract yet. Agree on those additions before building clients around
them. For browser clients on another origin, explicitly allow their origins in
the backend's CORS configuration. For different machines, use a reachable server
address and the project's intended network setup rather than localhost.

## Selected decoder and later accuracy improvements

The optional `PATHSNAP_DECODER_CONFIG` environment variable loads a decoder
configuration selected on training OOF predictions. Its recorded model hash must
match the frozen classifier. Restart the backend to adopt a completed configuration:

```bash
export PATHSNAP_DECODER_CONFIG=outputs/experiments/hmm_tuning_001/decoder_configuration.json
```

The committed configuration passed student-test validation at 92.19% Top-1 and
scored 90.79% on the repeat evaluator benchmark. Later tuning can run separately.
Client request/response fields stay the same when decoder settings change. `/health` reports
`decoder_configured: true`. The original measured evaluator benchmark remains
versioned separately, so tuning does not overwrite its evidence.
