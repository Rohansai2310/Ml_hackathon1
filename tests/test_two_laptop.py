"""Contract and leakage checks for a separate UI laptop and shared API server."""
import importlib.util
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from fastapi.testclient import TestClient
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pathsnap import api, data_io


class DistanceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probability = np.exp(-features.dist_m.to_numpy()/25)
        return np.column_stack([1-probability, probability])


@pytest.fixture
def server_data(tmp_path, monkeypatch):
    tid = '00000001'
    directory = tmp_path/'evaluator_test'/tid
    directory.mkdir(parents=True)
    np.savetxt(directory/f'{tid}.nodes', [[0,51],[.001,51],[.002,51]])
    np.savetxt(directory/f'{tid}.arcs', [[0,1],[1,2]], fmt='%d')
    np.savetxt(directory/f'{tid}.track', [[.0002,51,0],[.0008,51,1],[.0014,51,2],[.0014,51.01,3]])
    np.savetxt(directory/f'{tid}.route', [0,1], fmt='%d')
    monkeypatch.setenv('PATHSNAP_DATA_ROOT', str(tmp_path))
    monkeypatch.setattr(api, '_model', DistanceModel())
    monkeypatch.setattr(api, '_decoder_parameters', {})
    monkeypatch.setattr(api, '_model_sha256', 'synthetic-model')
    return tid, directory


def test_ui_lists_server_trips_and_evaluates_only_after_decoding(server_data, monkeypatch):
    tid, directory = server_data
    original_load, original_decode = data_io._load, api.viterbi_decode
    decoded = False

    def guard(path, *args, **kwargs):
        if str(path).endswith('.route'):
            assert decoded, 'Reference read before completed Viterbi prediction'
        return original_load(path, *args, **kwargs)

    def decode(*args, **kwargs):
        nonlocal decoded
        result = original_decode(*args, **kwargs)
        decoded = True
        return result

    monkeypatch.setattr(data_io, '_load', guard)
    monkeypatch.setattr(api, 'viterbi_decode', decode)
    with TestClient(api.app) as client:
        catalog = client.get('/trips').json()
        assert catalog['trips'] == [tid] and catalog['count'] == 1
        assert not decoded
        assert 'Route evaluator' in client.get('/ui').text
        assert client.get('/assets/app.js').status_code == 200
        response = client.post('/evaluate', json={'trip_id':tid,'partition':'evaluator_test'})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['match']['predicted_arc_ids'] == [0,0,1,-1]
    assert payload['evaluation']['hmm_correct_points'] == 3
    assert payload['evaluation']['known_points'] == 3
    assert payload['evaluation']['unresolved_points'] == 1
    assert payload['geometry']['matched_segments'] == [[[[0.,51.],[.001,51.]],[[.001,51.],[.002,51.]]]]
    assert 'RF + HMM (orange)' in payload['map_html']
    assert 'tile.openstreetmap.org' not in payload['map_html']
    assert 'L.tileLayer(' not in payload['map_html']
    assert 'Background: supplied trip road network' in payload['map_html']
    assert payload['model']['sha256'] == 'synthetic-model'


def test_missing_reference_still_returns_predicted_route(server_data):
    tid, directory = server_data
    (directory/f'{tid}.route').unlink()
    with TestClient(api.app) as client:
        response = client.post('/evaluate', json={'trip_id':tid,'partition':'evaluator_test'})
    assert response.status_code == 200
    assert response.json()['evaluation']['status'] == 'unavailable'
    assert response.json()['geometry']['reference_segments'] == []
    assert response.json()['match']['route_segments']


def test_ui_rejects_wrong_partition_and_malformed_trip(server_data):
    with TestClient(api.app) as client:
        assert client.post('/evaluate',json={'trip_id':'../other','partition':'evaluator_test'}).status_code == 422
        assert client.post('/evaluate',json={'trip_id':'00000001','partition':'train'}).status_code == 400


def test_reference_evaluation_rejects_inputs_changed_after_prediction(server_data, monkeypatch):
    original = data_io.load_trip
    def changed(*args, **kwargs):
        trip = original(*args, **kwargs)
        if kwargs.get('with_route'):
            trip.track = trip.track.copy()
            trip.track[0,0] += .001
        return trip
    monkeypatch.setattr(data_io, 'load_trip', changed)
    with TestClient(api.app) as client:
        response = client.post('/evaluate',json={'trip_id':'00000001','partition':'evaluator_test'})
    assert response.status_code == 422
    assert 'changed after prediction' in response.json()['detail']


def test_client_laptop_proxy_preserves_requests_and_server_errors():
    """Two real local HTTP processes: the browser proxy contains no model/data."""
    script = Path(__file__).resolve().parents[1]/'scripts/run_client.py'
    spec = importlib.util.spec_from_file_location('pathsnap_client_test',script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200);self.end_headers();self.wfile.write(b'{"model_loaded":true}')
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(422);self.end_headers();self.wfile.write(b'{"detail":"invalid fixture trip"}')

    backend = ThreadingHTTPServer(('127.0.0.1',0),Backend)
    module.ClientHandler.server_url = f'http://127.0.0.1:{backend.server_port}'
    from functools import partial
    frontend = ThreadingHTTPServer(('127.0.0.1',0),partial(module.ClientHandler,directory=str(module.WEB)))
    threads = [threading.Thread(target=s.serve_forever,daemon=True) for s in [backend,frontend]]
    for thread in threads:
        thread.start()
    base = f'http://127.0.0.1:{frontend.server_port}'
    try:
        with urllib.request.urlopen(base+'/client-config.json') as response:
            assert json.load(response)['server'] == module.ClientHandler.server_url
        with urllib.request.urlopen(base+'/api/health') as response:
            assert json.load(response)['model_loaded'] is True
        with urllib.request.urlopen(base+'/assets/app.js') as response:
            assert 'async function match()' in response.read().decode()
        request = urllib.request.Request(base+'/api/evaluate',data=b'{"trip_id":"00000001"}',headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 422
        assert json.loads(caught.value.read())['detail'] == 'invalid fixture trip'
        assert calls == [{'trip_id':'00000001'}]
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(base+'/api/arbitrary-file')
        assert caught.value.code == 404
    finally:
        for server in [frontend,backend]:
            server.shutdown();server.server_close()
        for thread in threads:
            thread.join(timeout=5)
