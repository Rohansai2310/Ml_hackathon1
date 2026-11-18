import numpy as np
from fastapi.testclient import TestClient

from pathsnap import api, data_io


class DistanceModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probability = np.exp(-features.dist_m.to_numpy() / 25)
        return np.column_stack([1 - probability, probability])


def test_api_uses_hmm_and_never_reads_reference_files(tmp_path, monkeypatch):
    tid = '00000001'
    directory = tmp_path / 'evaluator_test' / tid
    directory.mkdir(parents=True)
    np.savetxt(directory / f'{tid}.nodes', [[0,51],[.001,51],[.002,51]])
    np.savetxt(directory / f'{tid}.arcs', [[0,1],[1,2]], fmt='%d')
    np.savetxt(directory / f'{tid}.track', [[.0002,51,0],[.0008,51,1],[.0014,51,2],[.0014,51.01,3]])
    original = data_io._load

    def guard(path, *args, **kwargs):
        assert not str(path).endswith('.route')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(data_io, '_load', guard)
    monkeypatch.setattr(api, '_model', DistanceModel())
    with TestClient(api.app) as client:
        response = client.post('/match', json={'trip_id':tid,'partition':'evaluator_test','data_root':str(tmp_path)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['number_of_gps_points'] == 4
    assert len(body['predicted_arc_ids']) == len(body['segment_ids']) == 4
    assert body['predicted_arc_ids'][-1] == -1
    assert body['break_reasons']['3'] == 'no_candidates'
    assert body['route_segments']
    assert 'reference' not in body and 'label' not in body


def test_unloaded_api_returns_an_explicit_service_error(monkeypatch):
    monkeypatch.setattr(api, '_model', None)
    with TestClient(api.app) as client:
        assert client.get('/health').json()['model_loaded'] is False
        assert client.post('/match',json={'trip_id':'00000001'}).status_code == 503


def test_trip_path_traversal_is_rejected(monkeypatch):
    monkeypatch.setattr(api, '_model', DistanceModel())
    with TestClient(api.app) as client:
        assert client.post('/match',json={'trip_id':'../other'}).status_code == 422
