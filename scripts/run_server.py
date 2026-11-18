"""Laptop 1: serve the frozen model, decoder and evaluator over the local network."""
import argparse
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'outputs/training/random_forest_selected_001')
    parser.add_argument('--decoder-config', type=Path, default=ROOT/'outputs/experiments/hmm_tuning_001/decoder_configuration.json')
    args = parser.parse_args()
    for path in (args.model_dir/'pipeline.joblib', args.decoder_config):
        if not path.is_file():
            parser.error(f'Required server artifact is missing: {path}')
    if not (args.data_root/'evaluator_test').is_dir():
        parser.error('--data-root must contain evaluator_test/')
    os.environ['PATHSNAP_MODEL_DIR'] = str(args.model_dir.resolve())
    os.environ['PATHSNAP_DECODER_CONFIG'] = str(args.decoder_config.resolve())
    os.environ['PATHSNAP_DATA_ROOT'] = str(args.data_root.resolve())
    sys.path.insert(0, str(ROOT/'src'))
    try:
        addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)})
    except socket.gaierror:
        addresses = ['unavailable; use the Wi-Fi network settings']
    print(f'Local check: http://127.0.0.1:{args.port}/health', flush=True)
    print(f'On the client laptop, use this server laptop\'s Wi-Fi IPv4 address and port {args.port}.', flush=True)
    print('Hostname addresses (verify the Wi-Fi adapter): '+', '.join(addresses), flush=True)
    print('One server worker loads one classifier. Press Ctrl+C to stop.', flush=True)
    import uvicorn
    uvicorn.run('pathsnap.api:app', host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
