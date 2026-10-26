"""Fetch the exact upstream-linked checkpoint and RGBD milk sequence."""
import hashlib
import json
from pathlib import Path
import zipfile
import gdown

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = [
        ('1M-VD35-qdB5Iw-AtbDBCKC7hPolFW9UY', ROOT / 'BundleTrack/LoFTR/weights/outdoor_ds.ckpt'),
        ('1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN', ROOT / '.native/data/milk-sample.zip'),
    ]
    report = []
    reference_path = ROOT / 'native/sample-assets.json'
    references = json.loads(reference_path.read_text()) if reference_path.exists() else []
    expected_hashes = {item['google_drive_id']: item['sha256'] for item in references}
    for file_id, path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if gdown.download(id=file_id, output=str(path)) is None:
                raise RuntimeError(f'Download failed: {file_id}')
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        if file_id in expected_hashes and digest.hexdigest() != expected_hashes[file_id]:
            raise RuntimeError(f'Checksum mismatch for {path}; do not use this asset')
        report.append({'google_drive_id': file_id, 'path': str(path.relative_to(ROOT)),
                       'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    destination = ROOT / '.native/data/milk'
    expected = destination / '2022-11-18-15-10-24_milk/cam_K.txt'
    if not expected.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(files[1][1]) as archive:
            for member in archive.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError(f'Unsafe archive path: {member.filename}')
            archive.extractall(destination)
    (ROOT / 'native/logs/sample-downloads.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Sample:', expected.parent)

if __name__ == '__main__':
    main()
