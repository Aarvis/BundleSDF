"""Exercise installed native/GPU components; does not command any robot."""
import json
import os
from pathlib import Path
import platform
import subprocess
import torch
import kaolin
import numpy as np

def main():
    assert torch.cuda.is_available(), 'CUDA unavailable'
    from mycuda import common
    from mycuda.torch_ngp_grid_encoder import grid
    from pytorch3d.transforms import so3_exp_map, so3_log_map
    import my_cpp
    import bundlesdf
    subprocess.run([str(Path(__file__).resolve().parents[1] / '.native/bin/check_opencv')], check=True)
    # Real Kaolin CUDA octree construction and query, not only imports.
    points = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.2, 0.1]], device='cuda')
    quantized = kaolin.ops.spc.quantize_points(points, 3)
    octree = kaolin.ops.spc.unbatched_points_to_octree(quantized, 3, sorted=False)
    assert octree.numel() > 0
    rotations = torch.tensor([[0.1, 0.2, 0.3]], device='cuda')
    assert torch.allclose(so3_log_map(so3_exp_map(rotations)), rotations, atol=1e-5)
    encoder = grid.GridEncoder(n_levels=2, base_resolution=4, desired_resolution=8, log2_hashmap_size=8).cuda()
    inputs = torch.zeros((4, 3), device='cuda', requires_grad=True)
    encoded = encoder(inputs)
    encoded.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    torch.cuda.synchronize()
    report = {
        'status': 'passed', 'python': platform.python_version(),
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'kaolin': kaolin.__version__, 'gpu': torch.cuda.get_device_name(),
        'capability': list(torch.cuda.get_device_capability()),
        'common': common.__file__, 'my_cpp': my_cpp.__file__,
        'cuda_home': os.environ.get('CUDA_HOME'),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checks': ['BundleSDF import', 'BundleTrack native import', 'mycuda import',
                   'OpenCV CUDA descriptor matching', 'grid encoder CUDA forward/backward',
                   'Kaolin CUDA octree', 'PyTorch3D CUDA transforms'],
    }
    destination = Path(__file__).parent / 'logs' / 'installation-check.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
