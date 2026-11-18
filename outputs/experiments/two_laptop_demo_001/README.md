# Two-laptop deployment evidence

The backend runs one frozen Random Forest and the selected HMM/Viterbi settings.
The separate standard-library client serves the evaluator UI and proxies its
requests to FastAPI. No model or dataset is required on the UI laptop.

- `browser_check.json`: real Chrome interaction, ten discovered trips, trip
  00000023 matching/evaluation, map/layer/download controls and responsive width.
- `ui_desktop.png` and `ui_mobile.png`: actual returned trip/geometry/results.
- `lan_and_prediction_check.json`: API health/catalog through the Wi-Fi interface;
  concurrent proxy requests for trips 00000015 and 00000049 checked point-for-point
  against the frozen selected-decoder prediction files.
- `tests.log`: application test suite, including synthetic no-reference leakage
  checks and real local HTTP proxy/error propagation checks.

A per-trip score such as 93.67% for 00000023 is different from the pooled evaluator
benchmark of 90.79%. No model fitting or hyperparameter selection took place.
Client/server processes were on this host. A second physical laptop and actual
cross-device Wi-Fi communication still need to be checked at demonstration time.

Run instructions: `docs/CLIENT_API.md`.
