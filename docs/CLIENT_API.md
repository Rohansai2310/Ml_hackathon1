# Two laptops on the same Wi-Fi

**Laptop 1 (your laptop)** runs FastAPI, one frozen Random Forest, HMM/Viterbi,
raw evaluator trips, reference evaluation and map generation. **Laptop 2** runs
only the interface. It sends requests to Laptop 1 and displays the response.
This implements the two-laptop arrangement in `docs/guidelines/Main Guidelines.pdf`.

```text
Laptop 2 browser → Laptop 2 UI/proxy → Wi-Fi → Laptop 1 FastAPI
                                              ↓
                                  one Random Forest → HMM/Viterbi
                                              ↓
                                  evaluate frozen predictions
                                              ↓
Laptop 2 browser ← route geometry, metrics, Folium map and segment boundaries
```

## Laptop 1: start your server

Use the `model-training` branch. From the repository root, activate your existing
Python environment, or create one and install the server dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/run_server.py --data-root .. --port 8000
```

The default `--host 0.0.0.0` accepts connections through the laptop's local network
interfaces. Keep this terminal running. Use Ctrl+C to stop. There is one server
worker and one fitted classifier; no model is installed on Laptop 2.

The default model is `outputs/training/random_forest_selected_001/pipeline.joblib`.
The default selected decoder is
`outputs/experiments/hmm_tuning_001/decoder_configuration.json`.
The actual fitted artifact and raw dataset stay local and are excluded from Git.
A fresh server clone requires those files or the documented refit workflow.
`--data-root` must be the parent of `evaluator_test/`, whose children are the ten
trip folders. The launcher checks these paths before starting.

Override artifact paths when necessary:

```bash
python scripts/run_server.py --data-root /path/to/dataset-parent \
  --model-dir /path/to/frozen-model-directory \
  --decoder-config outputs/experiments/hmm_tuning_001/decoder_configuration.json
```

Find **Laptop 1's Wi-Fi IPv4 address** with `ip -4 address` or `hostname -I` on
Linux, `ipconfig` on Windows, or the Wi-Fi settings on macOS. Choose the Wi-Fi
adapter address, not loopback or a Docker/VPN address. The address can change
when you reconnect. `0.0.0.0` is a bind address, not the URL to type on Laptop 2.

For example, if the server's Wi-Fi address is `192.168.1.10`, check from Laptop 2:

```bash
curl http://192.168.1.10:8000/health
```

A loaded server replies with:

```json
{"status":"healthy","model_loaded":true,"decoder_configured":true}
```

The current workspace demo uses port **8077**, with the client on **8501**. Use
8077 instead of 8000 when connecting to that running demo. The Wi-Fi adapter
address observed during implementation was `10.30.202.117`; verify it again
before the actual two-laptop demonstration.

## Laptop 2: run the interface

Both laptops should be on the same Wi-Fi or hotspot. Clone/download this branch
on Laptop 2, open a terminal in the repository and run:

```bash
python scripts/run_client.py --server http://192.168.1.10:8000
```

Alternatively, extract [the small UI-only zip](../outputs/client/PathSnap_UI_Client.zip)
on Laptop 2 and run the same command from its extracted folder. It contains only
the launcher and interface files. Regenerate it with `python scripts/package_client.py`.

Replace the example address with **Laptop 1's actual Wi-Fi IP and server port**.
On Windows, `py scripts/run_client.py ...` also works if Python uses the `py`
launcher. The client requires Python 3.10 or later and its standard library;
no pip packages, model, dataset, sklearn, FastAPI or Node installation is needed.

Open **http://127.0.0.1:8501** in Laptop 2's browser. The interface:

1. Checks the FastAPI server and loads its evaluator trip list.
2. Lets you choose one of the ten trips and reconstruct the route.
3. Shows GPS points, nearby roads, reference route and RF + HMM route.
4. Displays per-trip RF/HMM accuracy, known/unresolved reference counts,
   observation count, route segment count and actual server processing time.
5. Lets you toggle layers, pan/zoom, download result JSON and save a Folium map.

The plain road view uses no external downloads and works without internet.
The Folium view additionally needs internet access for its JavaScript libraries.
It uses the supplied trip road network as its background and requests no external
street tiles, avoiding tile-provider access blocks. Use Folium's own layer control in that view. The left-hand
layer checkboxes control the plain road view.

The small client-side proxy forwards only `/health`, `/trips`, `/match` and
`/evaluate` to the configured server. Browser calls remain on the client UI
origin, so no browser CORS setup or client-IP allowlist is required. Requests go
directly to the selected server rather than through inherited HTTP proxy settings.
Laptop 1 does not need to know Laptop 2's IP for this request/response setup.

For a quick alternative, open `http://SERVER_IP:8000/ui` directly on Laptop 2.
That serves the same interface from Laptop 1; the separate client launcher is
available for the demonstrated two-process/two-laptop setup.

## Connection troubleshooting

- Use the server's Wi-Fi IP on Laptop 2. `localhost` on Laptop 2 points to Laptop 2.
- Keep the server terminal open and use the same port on both sides.
- Allow inbound TCP on that server port in Laptop 1's firewall for the intended
  local/private network. Do not disable the firewall or configure router port forwarding.
- Guest/campus Wi-Fi may isolate devices even on the same network. If health
  cannot be reached, try a personal hotspot that allows device-to-device traffic.
- An HTTP 502 in the UI means its proxy could not reach the configured server.
  An empty trip list means the server's data root does not contain the expected files.

## FastAPI contract

### Access from different internet connections

A private Wi-Fi address such as `10.x.x.x` or `192.168.x.x` cannot normally be
reached from another internet connection. Use a shared VPN or an explicitly
approved public tunnel; changing the browser URL alone cannot make it reachable.

An optional access-code gateway is implemented in `scripts/share_server.py`.
It listens on loopback, requires a per-run code, uses an HTTPS-only session cookie,
accepts only evaluator trip requests and rejects client-supplied filesystem paths.
It does not expose the full FastAPI service directly. Run it locally with:

```bash
python scripts/share_server.py --state-file /tmp/pathsnap-share-state.json
```

The state file contains the access code and is created with owner-only permissions.
Never commit it. To share over the internet, a tunnel must target this gateway on
port 8510, not the unrestricted API on 8077. Sharing through an external provider
requires approval for the selected service and the route/results data it carries.
No tunnel is started by the server, client or gateway launcher itself.

After approval, install the official `cloudflared` binary and start a second
terminal with:

```bash
cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8510
```

Open the generated **HTTPS address directly in the other laptop's browser**,
even when it uses another internet connection. Enter the code from the private
state file. The browser loads the UI and uses the protected proxy over that same
HTTPS origin; no client launcher or model installation is required on that laptop.
Do not point the standalone client proxy at this login-protected URL; it cannot
perform the browser sign-in flow. Do not paste access codes into Git or URLs.

[Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
provide a temporary HTTPS hostname while the tunnel process stays running; the
hostname changes on a new tunnel. They are for development demonstrations, not a
permanent deployment. Keep the laptop awake and both server/gateway terminals
running for an approved demo, and stop the gateway/tunnel to revoke access.

`GET /health` reports model/configuration loading. `GET /trips` returns
`{"partition":"evaluator_test","trips":["00000015", "..."],"count":10}` using
actual server folders. `GET /docs` provides OpenAPI documentation.

Both matching endpoints accept an eight-digit trip ID:

```json
{"trip_id":"00000023","partition":"evaluator_test"}
```

`POST /match` is label-free inference and retains the original contract:
`predicted_arc_ids`, `segment_ids`, `route_segments`, `break_reasons`, trip/partition,
GPS/candidate counts and `model_used`. Unmatched observations retain arc/segment
ID `-1`. Arc IDs are local to a trip. Draw separate route segments independently.

`POST /evaluate` completes the same inference first, then reads references for
server-side evaluation. Its response contains:

| Field | Content |
|---|---|
| `match` | The original matching response |
| `evaluation` | RF and HMM per-point accuracy, correct/known/unresolved counts, candidate F1/log loss and metric limitations |
| `geometry` | Raw GPS, nearby road lines, predicted segments and reference segments, all in longitude/latitude |
| `map_html` | Actual Folium HTML generated from this trip's returned routes |
| `model` | Classifier identity, model SHA-256 and effective decoder settings |
| `elapsed_seconds` | Measured server request processing time |

Missing references leave the predicted route available and report evaluation
`status: unavailable`. Prediction never receives the reference route. Input
arrays are checked for changes between prediction and reference evaluation.
The endpoint supports evaluator_test only; the original `/match` still supports
train/student_test/evaluator_test. An optional legacy `data_root` request field
exists for local compatibility; the UI never supplies filesystem paths.

Errors include 422 for malformed input/no road candidates, 404 for missing GPS,
400 for unsupported partitions and 503 if no model is loaded. This is a trusted
local-network demonstration without an authentication layer or upload API.

## Verification and accuracy

The selected classifier/decoder are unchanged: **90.79% pooled pointwise accuracy**
on the repeat evaluator benchmark and **92.19% on student testing**, against valid
derived references. The interface shows the selected trip's score, which can be
higher or lower than that pooled benchmark. Showing a trip is not new tuning.

See `outputs/experiments/two_laptop_demo_001/` for test records and actual UI
screenshots. Separate server/client processes, browser interactions, simultaneous
requests and access through the host Wi-Fi interface were tested locally. A
second physical laptop must still connect over the actual demonstration Wi-Fi;
that external device and its firewall/network isolation cannot be verified here.
