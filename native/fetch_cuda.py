"""Install verified NVIDIA redistributable build components into this checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.native' / 'cuda-12.4'
CACHE = ROOT / '.native' / 'downloads'
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / 'redistrib_12.4.1.json'
    if not manifest_path.exists():
        urllib.request.urlretrieve(BASE + manifest_path.name, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    # NPP is needed by OpenCV CUDA modules, curand by BundleTrack RANSAC.
    for component in ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl', 'libcurand', 'libnpp']:
        item = manifest[component]['linux-x86_64']
        archive = CACHE / Path(item['relative_path']).name
        marker = DEST / ('.' + component + '.sha256')
        if marker.exists() and marker.read_text().strip() == item['sha256']:
            continue
        if not archive.exists():
            print('Downloading', component, item['size'], 'bytes', flush=True)
            urllib.request.urlretrieve(BASE + item['relative_path'], archive)
        with archive.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()
        if digest != item['sha256']:
            raise RuntimeError(f'SHA256 mismatch: {archive}; remove this archive and retry')
        subprocess.run(['tar', '-xJf', str(archive), '--strip-components=1', '-C', str(DEST)], check=True)
        marker.write_text(digest + '\n')
    lib64 = DEST / 'lib64'
    if not lib64.exists():
        lib64.symlink_to('lib', target_is_directory=True)
    print('CUDA toolkit components ready:', DEST)

if __name__ == '__main__':
    main()
