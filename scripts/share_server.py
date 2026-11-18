"""Local access-code gateway for a temporary HTTPS tunnel to the evaluator UI.

Keep the full API private. The gateway allows only fixed evaluator requests and
requires a per-run code before serving the interface, data or proxy endpoints.
"""
import argparse
from functools import partial
import hmac
from http.cookies import SimpleCookie, CookieError
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import urllib.parse

from run_client import ClientHandler, WEB

LOGIN = b'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PathSnap access</title>
<link rel="icon" href="data:,"><style>body{font:16px system-ui;background:#f6f7f2;color:#142f32;
display:grid;place-items:center;min-height:95vh;margin:0}main{background:white;padding:36px;
border:1px solid #dbe5dc;border-radius:14px;width:min(340px,80vw)}h1{margin-top:0}
p{color:#647770;line-height:1.6}input,button{font:inherit;width:100%;box-sizing:border-box;
padding:13px;border:1px solid #bfd1c5;border-radius:7px;margin-top:12px}button{background:#165b4e;
color:white;cursor:pointer}label{display:block;margin-top:24px;font-size:14px}</style></head>
<body><main><h1>PathSnap</h1><p>Enter the access code shared by your server laptop.</p>
<form method="post" action="/login"><label for="code">Access code</label>
<input id="code" name="code" type="password" autocomplete="current-password" required autofocus>
<button>Open route evaluator</button></form></main></body></html>'''


class ShareHandler(ClientHandler):
    access_code = ''
    session_token = ''
    compute_slots = threading.BoundedSemaphore(2)

    def authenticated(self):
        try:
            cookies = SimpleCookie(self.headers.get('Cookie', ''))
            token = cookies.get('pathsnap_session')
            return token is not None and hmac.compare_digest(token.value, self.session_token)
        except (CookieError, TypeError):
            return False

    def do_GET(self):
        if not self.authenticated():
            if self.path in ('/', '/ui', '/login'):
                self.send_body(LOGIN, content_type='text/html; charset=utf-8')
            else:
                self.send_body(b'{"detail":"Enter the access code on the home page first."}', 401)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == '/login':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('Invalid login body')
                values = urllib.parse.parse_qs(self.rfile.read(length).decode())
                code = values.get('code', [''])[0]
            except (ValueError, UnicodeError):
                self.send_body(b'{"detail":"Invalid login request"}',400)
                return
            if not hmac.compare_digest(code.encode(), self.access_code.encode()):
                self.send_body(LOGIN.replace(b'<form ',b'<p>Incorrect code. Please try again.</p><form '),403,'text/html; charset=utf-8')
                return
            self.send_response(303)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', f'pathsnap_session={self.session_token}; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=43200')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length','0')
            self.end_headers()
            return
        if not self.authenticated():
            self.send_body(b'{"detail":"Access code required"}',401)
            return
        if not self.compute_slots.acquire(blocking=False):
            self.send_body(b'{"detail":"The server is busy. Retry shortly."}',429)
            return
        try:
            super().do_POST()
        finally:
            self.compute_slots.release()

    def prepare_payload(self, body):
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Provide a JSON trip request') from exc
        if not isinstance(data, dict) or set(data) - {'trip_id','partition'}:
            raise ValueError('Only trip_id and partition are accepted; server paths cannot be supplied')
        if not isinstance(data.get('trip_id'),str) or not re.fullmatch(r'[0-9]{8}',data['trip_id']):
            raise ValueError('trip_id must contain eight digits')
        if data.get('partition','evaluator_test') != 'evaluator_test':
            raise ValueError('Only evaluator_test is available through the shared demo')
        return json.dumps({'trip_id':data['trip_id'],'partition':'evaluator_test'}).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:8077')
    parser.add_argument('--port',type=int,default=8510)
    parser.add_argument('--state-file',type=Path,required=True,help='Private local file for the per-run access code')
    args = parser.parse_args()
    server = urllib.parse.urlsplit(args.server)
    if server.scheme != 'http' or server.hostname not in ('127.0.0.1','localhost') or server.path or server.query or server.fragment or server.username:
        parser.error('--server must be the local FastAPI origin')
    ShareHandler.server_url=args.server
    ShareHandler.access_code=secrets.token_urlsafe(8)
    ShareHandler.session_token=secrets.token_urlsafe(32)
    # Restrict permissions at creation; never add the code to logs or Git.
    descriptor=os.open(args.state_file,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(descriptor,0o600)
    with os.fdopen(descriptor,'w') as stream:
        json.dump({'access_code':ShareHandler.access_code,'port':args.port},stream)
    handler=partial(ShareHandler,directory=str(WEB))
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as gateway:
        print(f'Local sharing gateway: http://127.0.0.1:{args.port}',flush=True)
        print(f'Access code saved privately in {args.state_file}. Stop this terminal to revoke access.',flush=True)
        try:
            gateway.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
