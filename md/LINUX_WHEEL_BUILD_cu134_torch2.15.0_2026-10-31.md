# Nunchaku Linux Wheel Build Guide (cu134 / torch 2.15.0 / cp314)

Linux wheel build procedure for Nunchaku with CUDA 13.4, PyTorch 2.15.0+cu134, and Python 3.14:

- **Wheel**: `nunchaku-1.3.0.dev20261031+cu13.4torch2.15-cp314-cp314-linux_x86_64.whl`
- **Wheel Size**: 186,014,688 bytes (~177.4 MB)
- **SHA256**: `60C2C1E819FA00593623D0A522C18460C5609DD0077D387BA1DEFD308A7D54CA`
- **Target Architectures**: 8 generations unified (`sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120a`, `sm_121a`)
- **Embedded CUDA ELF Kernels**: 200 cubins (25 cubins per architecture)
- **Build Host**: Windows 11 + WSL2 Ubuntu 26.04 (x86_64)

---

## 1. Architecture Coverage: Full 8 Generations in a Single Wheel

Unlike FlashAttention, Nunchaku's C++ extension (`nunchaku._C`) does not hit the x86-64 PC32 relocation overflow limit when bundling all architectures. All 8 GPU architectures are compiled into a single unified Linux wheel:

| Arch Code | Microarchitecture | Target Hardware | Embedded Cubins |
|---|---|---|---|
| `sm_75` | Turing | RTX 2000 series, GTX 1660, Tesla T4 | 25 |
| `sm_80` | Ampere (Data Center) | A100 | 25 |
| `sm_86` | Ampere (Consumer) | RTX 3000 series, RTX A-series | 25 |
| `sm_89` | Ada Lovelace | RTX 4000 series, L40 | 25 |
| `sm_90` | Hopper | H100 | 25 |
| `sm_100` | Blackwell (Data Center) | B100, B200 | 25 |
| `sm_120a` | Blackwell (Consumer) | RTX 5000 series | 25 |
| `sm_121a` | Blackwell (Mobile/Specialized) | RTX 5000 Mobile / DGX Spark | 25 |
| **Total** | | | **200 cubins** |

Architecture specification flag:
```bash
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.1"
```

---

## 2. Build Environment Setup (WSL2 Ubuntu)

### 2.1 Host Packages

```bash
apt-get update
apt-get install -y build-essential ninja-build pkg-config rsync git python3.14-dev
```

`python3.14-dev` is mandatory. Without the CPython 3.14 header files, the compilation fails at `#include <Python.h>`.

### 2.2 CUDA Toolkit 13.4

Install the official NVIDIA CUDA 13.4 packages:

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y cuda-toolkit-13-4 cuda-nvcc-13-4 cuda-cudart-dev-13-4 \
  cuda-cccl-13-4 cuda-crt-13-4 cuda-culibos-dev-13-4 cuda-driver-dev-13-4 \
  libnvvm-13-4 libnvptxcompiler-13-4 libcublas-dev-13-4 libcusolver-dev-13-4 \
  libcufft-dev-13-4 libcurand-dev-13-4 libnvjpeg-dev-13-4 libcusparse-dev-13-4 cuda-nvtx-13-4

/usr/local/cuda-13.4/bin/nvcc --version
# Expected: release 13.4, V13.4.92
```

### 2.3 Python 3.14 Virtual Environment + PyTorch 2.15.0+cu134

Use `uv` for fast environment provisioning:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

uv venv /root/fa314 --python /usr/bin/python3.14
export UV_HTTP_TIMEOUT=300
uv pip install --python /root/fa314/bin/python pip setuptools wheel psutil ninja packaging
uv pip install --python /root/fa314/bin/python torch==2.15.0+cu134 --index-url https://download.pytorch.org/whl/cu134

# Verify CXX11 ABI and CUDA version
/root/fa314/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
# Expected output: PyTorch: 2.15.0+cu134 CUDA: 13.4 CXX11 ABI: True
```

`CXX11 ABI` must be `True` for standard Linux C++20 extensions.

---

## 3. Resource Allocation & Parallelism (`MAX_JOBS=5`)

### 3.1 Memory Requirements
Nunchaku compiles multiple Cutlass GEMM and Block-Sparse-Attention CUDA kernels. During parallel PTX/SASS code generation via `ptxas` and `cicc`, memory usage peaks at 30 GB to 55 GB across 5 parallel Ninja jobs.

`.wslconfig` (Windows `%USERPROFILE%\.wslconfig`):
```ini
[wsl2]
memory=58GB
swap=32GB
swapFile=C:\\WSL\\swap.vhdx
```

Inside WSL2 Ubuntu, ensure additional swap space is configured:
```bash
fallocate -l 32G /root/.swapfile_d32
chmod 600 /root/.swapfile_d32
mkswap /root/.swapfile_d32
swapon /root/.swapfile_d32
echo '/root/.swapfile_d32 none swap sw 0 0' >> /etc/fstab
```

### 3.2 Parallel Job Count
Set `MAX_JOBS=5`. Never downgrade to `MAX_JOBS=1`. `MAX_JOBS=5` finishes the complete 8-architecture build in ~28 minutes on high-core CPUs (32 threads).

---

## 4. Source Tree Preparation

Copy the repository from Windows filesystem into WSL Linux native filesystem to maximize I/O throughput:

```bash
mkdir -p /root/nunchaku
rsync -a --delete \
  --exclude='dist' \
  --exclude='build' \
  --exclude='*.egg-info' \
  /mnt/d/USERFILES/GitHub/nunchaku/ /root/nunchaku/

cd /root/nunchaku
git config --global --add safe.directory /root/nunchaku
```

Ensure all submodules in `third_party/` are populated:
- `third_party/cutlass`
- `third_party/json`
- `third_party/mio`
- `third_party/spdlog`
- `third_party/Block-Sparse-Attention`

---

## 5. Build Execution

### 5.1 Automated Script (`build_linux.sh`)

Place the following script at `/root/nunchaku/build_linux.sh`:

```bash
#!/bin/bash
set -euo pipefail

export HOME=/root
exec > /root/nunchaku-linux-build.log 2>&1
echo "[*] Starting Nunchaku Linux Wheel Build at $(date)"

VENV_PY=/root/fa314/bin/python
VENV_PIP=/root/fa314/bin/pip
SRC_DIR=/root/nunchaku
CUDA_DIR=/usr/local/cuda-13.4

# Sync source tree
mkdir -p "$SRC_DIR"
rsync -a --delete \
  --exclude='dist' \
  --exclude='build' \
  --exclude='*.egg-info' \
  /mnt/d/USERFILES/GitHub/nunchaku/ "$SRC_DIR/"

cd "$SRC_DIR"
git config --global --add safe.directory "$SRC_DIR"

# Compiler & Environment configuration
export PATH="$CUDA_DIR/bin:/root/.local/bin:$PATH"
export CUDA_HOME="$CUDA_DIR"
export CUDA_PATH="$CUDA_DIR"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export MAX_JOBS=5
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.1"
export NUNCHAKU_BUILD_WHEELS=1

# Clean previous build artifacts
rm -rf build dist *.egg-info

echo "[*] Launching setup.py bdist_wheel..."
"$VENV_PY" setup.py bdist_wheel --dist-dir dist

WHL_FILE=$(ls dist/nunchaku-*-linux_x86_64.whl | head -n 1)
echo "[*] Build successful: $WHL_FILE"

# Copy to delivery destinations
mkdir -p /mnt/d/USERFILES/GitHub/nunchaku/dist /mnt/d/USERFILES/whl
cp -f "$WHL_FILE" /mnt/d/USERFILES/GitHub/nunchaku/dist/
cp -f "$WHL_FILE" /mnt/d/USERFILES/whl/
cp -f "$WHL_FILE" /mnt/d/

echo "[*] Copied wheel to Windows destinations."

# Verify wheel in virtual environment
"$VENV_PIP" install --no-deps --force-reinstall "$WHL_FILE"
echo "[*] Verification complete at $(date)"
```

### 5.2 Launching via Transient Systemd Unit (Background Execution)

WSL terminates running background processes if client sessions close. To guarantee uninterrupted execution:

```bash
chmod +x /root/nunchaku/build_linux.sh
systemd-run --unit=nunchaku-linux-build --collect /usr/bin/bash /root/nunchaku/build_linux.sh
```

Monitor progress without blocking:
```bash
# Check service state and resource usage
systemctl status nunchaku-linux-build --no-pager

# Tail Ninja and NVCC logs
tail -f /root/nunchaku-linux-build.log
```

---

## 6. Verification & Quality Assurance

### 6.1 CUDA ELF Cubin Verification (`cuobjdump`)
Run `cuobjdump` against the compiled shared object `_C.cpython-314-x86_64-linux-gnu.so` to ensure all 8 architectures are fully embedded:

```bash
/usr/local/cuda-13.4/bin/cuobjdump --list-elf \
  /root/fa314/lib/python3.14/site-packages/nunchaku/_C.cpython-314-x86_64-linux-gnu.so \
  | grep -o 'sm_[0-9a]*' | sort | uniq -c
```

**Expected Result**:
```text
     25 sm_75
     25 sm_80
     25 sm_86
     25 sm_89
     25 sm_90
     25 sm_100
     25 sm_120a
     25 sm_121a
```
Total: Exactly 200 CUDA ELF cubins across 8 microarchitectures.

### 6.2 Python 3.14 Runtime Load Verification
Execute runtime symbol inspection from a non-source directory:

```bash
cd /tmp
/root/fa314/bin/python -c "
import importlib.machinery, torch
so_path = '/root/fa314/lib/python3.14/site-packages/nunchaku/_C.cpython-314-x86_64-linux-gnu.so'
mod = importlib.machinery.ExtensionFileLoader('_C', so_path).load_module()
print('PyTorch:', torch.__version__)
print('Loaded _C:', mod)
print('Symbols count:', len(dir(mod)))
print('CUDA available:', torch.cuda.is_available())
"
```

**Expected Result**:
```text
PyTorch: 2.15.0+cu134
Loaded _C: <module '_C' from '/root/fa314/lib/python3.14/site-packages/nunchaku/_C.cpython-314-x86_64-linux-gnu.so'>
Symbols count: 13
CUDA available: True
```

---

## 7. Artifact Deliverables Matrix

| Destination | File Path |
|---|---|
| Repository dist | `D:\USERFILES\GitHub\nunchaku\dist\nunchaku-1.3.0.dev20261031+cu13.4torch2.15-cp314-cp314-linux_x86_64.whl` |
| Master WHL Store | `D:\USERFILES\whl\nunchaku-1.3.0.dev20261031+cu13.4torch2.15-cp314-cp314-linux_x86_64.whl` |
| Drive Root | `D:\nunchaku-1.3.0.dev20261031+cu13.4torch2.15-cp314-cp314-linux_x86_64.whl` |
