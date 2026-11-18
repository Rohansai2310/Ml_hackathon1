"""Access-control checks for the optional cross-network demo gateway."""
import http.client
import importlib.util
import json
from pathlib import Path
import threading
from functools import partial
from http.server import ThreadingHTTPServer


def test_shared_demo_requires_login_and_rejects_filesystem_overrides(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('share_server_test',scripts/'share_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ShareHandler.access_code = 'synthetic-code'
    module.ShareHandler.session_token = 'synthetic-session'
    module.ShareHandler.server_url = 'http://127.0.0.1:1'
    server = ThreadingHTTPServer(('127.0.0.1',0),partial(module.ShareHandler,directory=str(module.WEB)))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    connection = http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)

    def request(method,path,body=None,headers=None):
        connection.request(method,path,body=body,headers=headers or {})
        response=connection.getresponse()
        return response.status,dict(response.getheaders()),response.read()

    try:
        status,_,body=request('GET','/')
        assert status==200 and b'Enter the access code' in body
        assert request('GET','/api/health')[0]==401
        assert request('GET','/assets/app.js')[0]==401
        assert request('POST','/api/evaluate',b'{}')[0]==401
        assert request('POST','/login','code=wrong',{'Content-Type':'application/x-www-form-urlencoded'})[0]==403
        status,headers,_=request('POST','/login','code=synthetic-code',{'Content-Type':'application/x-www-form-urlencoded'})
        assert status==303
        assert all(flag in headers['Set-Cookie'] for flag in ['Secure','HttpOnly','SameSite=Strict'])
        cookie=headers['Set-Cookie'].split(';')[0]
        assert request('GET','/',headers={'Cookie':cookie})[0]==200
        assert request('GET','/assets/app.js',headers={'Cookie':cookie})[0]==200
        for payload in [
            {'trip_id':'00000023','data_root':'/tmp'},
            {'trip_id':'00000023','partition':'train'},
            {'trip_id':'../other'}, [], {'trip_id':23},
        ]:
            status,_,body=request('POST','/api/evaluate',json.dumps(payload),
                {'Cookie':cookie,'Content-Type':'application/json'})
            assert status==400,body
        assert json.loads(module.ShareHandler.prepare_payload(None,b'{"trip_id":"00000023"}'))=={
            'trip_id':'00000023','partition':'evaluator_test'}
    finally:
        connection.close()
        server.shutdown();server.server_close();thread.join(timeout=5)
