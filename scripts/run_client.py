"""Laptop 2: Python-standard-library UI server forwarding requests to Laptop 1.

python scripts/run_client.py --server http://192.168.1.10:8000
No model, raw data, pip packages or browser CORS configuration are needed here.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

WEB = Path(__file__).resolve().parents[1]/'src/pathsnap/web'


class ClientHandler(SimpleHTTPRequestHandler):
    server_url = ''

    def prepare_payload(self, body):
        """Optional policy hook used by the internet-sharing gateway."""
        return body

    def send_body(self, body, status=200, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def proxy(self):
        suffix = self.path.removeprefix('/api')
        allowed = {'GET': {'/health', '/trips'}, 'POST': {'/match', '/evaluate'}}
        if suffix not in allowed.get(self.command, set()):
            self.send_error(404)
            return
        body = None
        if self.command == 'POST':
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400)
                return
            if length <= 0 or length > 4096:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            try:
                body = self.prepare_payload(body)
            except ValueError as exc:
                self.send_body(json.dumps({'detail': str(exc)}).encode(), 400)
                return
        request = urllib.request.Request(self.server_url+suffix, data=body,
            headers={'Content-Type': 'application/json'}, method=self.command)
        try:
            # A LAN connection must not be diverted through inherited HTTP proxies.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=180) as response:
                self.send_body(response.read(), response.status, response.headers.get('Content-Type', 'application/json'))
        except urllib.error.HTTPError as exc:
            self.send_body(exc.read(), exc.code, exc.headers.get('Content-Type', 'application/json'))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            message = {'detail': f'Cannot reach {self.server_url}. Check server IP, Wi-Fi and firewall. {exc}'}
            self.send_body(json.dumps(message).encode(), 502)

    def do_GET(self):
        if self.path == '/client-config.json':
            self.send_body(json.dumps({'api_base': '/api', 'server': self.server_url}).encode())
        elif self.path.startswith('/api/'):
            self.proxy()
        elif self.path in ('/', '/ui'):
            self.path = '/index.html'
            super().do_GET()
        elif self.path.startswith('/assets/'):
            self.path = self.path.removeprefix('/assets')
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy()
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='Laptop 1 URL, e.g. http://192.168.1.10:8000')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8501, type=int)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.server)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
        parser.error('--server must be an http(s) server origin without a path or credentials')
    ClientHandler.server_url = args.server.rstrip('/')
    handler = partial(ClientHandler, directory=str(WEB))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f'Open http://127.0.0.1:{args.port} in this laptop\'s browser.', flush=True)
        print(f'Requests go to {ClientHandler.server_url}; press Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
