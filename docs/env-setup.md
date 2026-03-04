# Environment Setup (Conda)

Tested on: RTX 5090, driver 590.48, CUDA 13.1, Ubuntu, miniforge3.

## Step-by-step

```bash
# 1. Create env
conda create -n ensemformer python=3.10 -y
conda activate ensemformer

# 2. PyTorch with CUDA 12.8 (supports RTX 5090 / Blackwell sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. DGL (stable 2.1.0 for cu124; backward-compatible with cu128 driver)
pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html

# 4. Fix DGL's missing transitive deps (not declared in wheel metadata)
pip install pydantic torchdata==0.7.1

# 5. Patch DGL graphbolt (see Pitfalls below)
#    Edit dgl/graphbolt/__init__.py: wrap load_graphbolt() call in try/except

# 6. e3nn (for SE3-Transformer)
pip install e3nn

# 7. RDKit (via conda-forge — most reliable binary)
conda install -n ensemformer -c conda-forge rdkit -y

# 8. Remaining deps
pip install scikit-learn pandas pyyaml tqdm torchmetrics tensorboard
```

## Verify

```bash
python -c "
import warnings; warnings.filterwarnings('ignore')
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import dgl; print('DGL:', dgl.__version__)
import e3nn; print('e3nn:', e3nn.__version__)
from rdkit import Chem; print('RDKit OK')
import torchmetrics, sklearn, pandas
print('All OK')
"
```

Expected output on a GPU machine:
```
PyTorch: 2.10.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
DGL: 2.1.0
e3nn: 0.6.0
RDKit OK
All OK
```

## Quick smoke test

```bash
cd /path/to/EnsemFormer
python -c "
import sys; sys.path.insert(0, '.')
from src.networks.egnn_backbone import EGNNBackbone
from src.networks.cpmp_backbone import CPMPBackbone
from src.networks.se3t_backbone import SE3TBackbone
print('All 3 backbones imported OK')
"
```

---

## Pitfalls

### 1. PyTorch 2.5.1 does not support RTX 5090 (Blackwell / sm_120)
The project's original target (PyTorch 2.5.1) predates Blackwell support.
Running it on an RTX 5090 gives: `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
**Fix:** use PyTorch 2.7+ with `--index-url https://download.pytorch.org/whl/cu128`.
On first run PyTorch may JIT-compile PTX for sm_120 (~30 s); it caches the result afterward.

### 2. `conda run pip install ...` uses system pip, not env pip
`conda run` can resolve to the system Python on some distros (PEP 668 externally-managed error).
**Fix:** call the env's pip directly:
```bash
/home/liuw/miniforge3/envs/ensemformer/bin/pip install ...
```

### 3. DGL graphbolt C++ library missing for PyTorch 2.3+
DGL 2.1.0 ships precompiled graphbolt `.so` files only for PyTorch 2.0–2.2.
Running with PyTorch 2.3+ gives:
`FileNotFoundError: Cannot find DGL C++ graphbolt library at .../libgraphbolt_pytorch_2.10.0.so`
EnsemFormer's SE3T code doesn't use graphbolt at all, so the fix is to make the loader non-fatal.
**Fix:** edit `<env>/lib/python3.10/site-packages/dgl/graphbolt/__init__.py`, replace the bare `load_graphbolt()` call at the bottom with:
```python
try:
    load_graphbolt()
except (FileNotFoundError, ImportError):
    import warnings
    warnings.warn(
        "DGL graphbolt C++ library not found for this PyTorch version. "
        "Graphbolt features are disabled. Core DGL graph operations still work.",
        UserWarning,
    )
```
Symlinking the 2.2.x `.so` does **not** work — the ABI is incompatible.

### 4. DGL wheel pulls `torchdata==0.11` which dropped `datapipes`
DGL's metadata requests `torchdata>=0.5.0` and pip resolves to the latest (0.11).
`torchdata` 0.8+ removed `torchdata.datapipes`, which DGL 2.1 imports internally.
`ModuleNotFoundError: No module named 'torchdata.datapipes'`
**Fix:** `pip install torchdata==0.7.1`

### 5. DGL wheel also requires `pydantic` (undeclared)
DGL's `graphbolt/impl/ondisk_metadata.py` imports `pydantic` but it isn't listed as a dependency.
**Fix:** `pip install pydantic`

### 6. DGL stable wheel server returns empty pages for some URL patterns
The `torch-2.x/cu12x` subdirectory URLs (e.g. `wheels/torch-2.7/cu128/repo.html`) are empty or 404.
Only the flat `wheels/cu124/repo.html` path works for stable releases.
Nightly wheel server (`wheels-test/`) is similarly unreliable.

### 7. No `nvcc` on PATH — not a problem
The system `nvcc` is absent but unneeded. The pip PyTorch wheels bundle their own CUDA runtime libraries.
