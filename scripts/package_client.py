"""Package only the lightweight UI for a second laptop; never include model/data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/client/PathSnap_UI_Client.zip'


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT/'scripts/run_client.py'] + sorted((ROOT/'src/pathsnap/web').glob('*'))
    with ZipFile(OUTPUT, 'w', ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr('README.txt',
            'PathSnap UI laptop (Python 3.10+; no pip packages needed)\n\n'
            'Extract this zip, then open a terminal in the extracted folder.\n'
            'Keep both laptops on the same Wi-Fi. Start FastAPI on the server laptop first.\n'
            'Run: python scripts/run_client.py --server http://SERVER_WIFI_IP:8000\n'
            'Replace SERVER_WIFI_IP and port with the actual server address.\n'
            'Open http://127.0.0.1:8501 in this laptop\'s browser.\n'
            'Windows users can use py instead of python.\n'
            'The server laptop keeps the model and dataset; neither is in this zip.\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
