#!/bin/bash
set -ex

export HOME=/root
exec > /root/nunchaku-linux-build.log 2>&1

echo "========================================================"
echo "[*] Nunchaku Linux Wheel Build for Python 3.14 / PyTorch 2.15.0+cu134 / CUDA 13.4"
echo "========================================================"

# Step 1: Ensure venv and packages
PYTHON_EXE="/root/fa314/bin/python"
PIP_EXE="/root/fa314/bin/pip"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "[*] Creating Python 3.14 venv at /root/fa314 via uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    uv venv /root/fa314 --python /usr/bin/python3.14
    export UV_HTTP_TIMEOUT=300
    uv pip install --python /root/fa314/bin/python pip setuptools wheel psutil ninja packaging
    uv pip install --python /root/fa314/bin/python torch==2.15.0+cu134 --index-url https://download.pytorch.org/whl/cu134
fi

$PYTHON_EXE -c "import torch; print('WSL Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"

# Step 2: Sync Nunchaku source to /root/nunchaku
echo "[*] Syncing source code to /root/nunchaku..."
mkdir -p /root/nunchaku
tar -C /mnt/d/USERFILES/GitHub/nunchaku -cf - --exclude=./dist --exclude=./build . | tar -C /root/nunchaku -xf -
cd /root/nunchaku
git config --global --add safe.directory /root/nunchaku

# Step 3: Configure build environment
export PATH=/usr/local/cuda-13.4/bin:/root/fa314/bin:/root/.local/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.4
export CUDA_PATH=/usr/local/cuda-13.4
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.1"
export MAX_JOBS=5
export NUNCHAKU_INSTALL_MODE=ALL
export NUNCHAKU_BUILD_WHEELS=1
export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"

echo "  CUDA_HOME: $CUDA_HOME"
echo "  ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS:  $MAX_JOBS"

# Step 4: Build Wheel
rm -rf build dist
$PYTHON_EXE setup.py bdist_wheel --dist-dir dist

# Step 5: Copy wheel to Windows destinations and local dist
echo "[*] Copying Linux wheels to Windows..."
mkdir -p /mnt/d/USERFILES/GitHub/nunchaku/dist /mnt/d/USERFILES/whl /mnt/d/ /home/ussoewwin/nunchaku/dist
cp -f dist/*.whl /mnt/d/USERFILES/GitHub/nunchaku/dist/
cp -f dist/*.whl /mnt/d/USERFILES/whl/
cp -f dist/*.whl /mnt/d/
cp -f dist/*.whl /home/ussoewwin/nunchaku/dist/
chmod -R 777 /home/ussoewwin/nunchaku/dist

# Step 6: Verify wheel in WSL
echo "[*] Verifying wheel installation in WSL..."
$PIP_EXE install --no-deps --force-reinstall dist/*.whl
$PYTHON_EXE -c "import torch, nunchaku, nunchaku._C as _C; print('Nunchaku verified! Torch:', torch.__version__, 'Loaded _C:', _C, 'CUDA available:', torch.cuda.is_available())"

echo "========================================================"
echo "[*] SUCCESS: Nunchaku Linux Wheel built and verified successfully!"
ls -la dist/*.whl
echo "========================================================"
