#!/bin/bash
# ============================================================
# 3DGS + Custom BA 환경 셋업 스크립트
#
# 사용법:
#   1. conda env create -f env/environment.yml
#   2. conda activate 3dgs_sba
#   3. bash env/setup.sh
#
# 이 스크립트는 conda 환경이 활성화된 상태에서 실행해야 합니다.
# ============================================================

set -e  # 에러 발생 시 즉시 중단

# 프로젝트 루트 (이 스크립트의 상위 디렉토리)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="${PROJECT_ROOT}/repos"

echo "============================================"
echo " 3DGS + Custom BA Setup"
echo " Project root: ${PROJECT_ROOT}"
echo "============================================"

# conda 환경 확인
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo "ERROR: conda 환경을 먼저 활성화하세요."
    echo "  conda activate 3dgs_sba"
    exit 1
fi
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================
# 1. PyTorch (시스템 CUDA 버전에 맞춰 자동 설치)
# ============================================================
echo "=== [1/5] PyTorch ==="
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  Already installed: $(python -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")')"
else
    # 시스템 CUDA 버전 감지
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    elif [ -f /usr/local/cuda/version.txt ]; then
        CUDA_VER=$(cat /usr/local/cuda/version.txt | grep -oP '[0-9]+\.[0-9]+')
    else
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
    fi

    if [ -z "$CUDA_VER" ]; then
        echo "  ERROR: CUDA를 감지할 수 없습니다. nvidia-smi 또는 nvcc를 확인하세요."
        exit 1
    fi

    # CUDA 버전 → PyTorch index URL 매핑
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_CUDA="cu124"
    elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
        TORCH_CUDA="cu121"
    elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_CUDA="cu118"
    else
        echo "  WARNING: CUDA ${CUDA_VER} 감지. 지원되는 최소 버전은 11.8입니다."
        TORCH_CUDA="cu118"
    fi

    echo "  System CUDA: ${CUDA_VER} → PyTorch: ${TORCH_CUDA}"
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    echo "  Installed."
fi
echo ""

# ============================================================
# 2. Gaussian Splatting + CUDA submodules
# ============================================================
echo "=== [2/5] Gaussian Splatting ==="
mkdir -p "$REPOS_DIR"

if [ ! -d "${REPOS_DIR}/gaussian-splatting" ]; then
    echo "  Cloning gaussian-splatting..."
    git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive "${REPOS_DIR}/gaussian-splatting"
else
    echo "  Already cloned."
fi

# diff-gaussian-rasterization
if python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "  diff-gaussian-rasterization: already installed"
else
    echo "  Building diff-gaussian-rasterization..."
    pip install --no-build-isolation "${REPOS_DIR}/gaussian-splatting/submodules/diff-gaussian-rasterization"
fi

# simple-knn
if python -c "from simple_knn._C import distCUDA2" 2>/dev/null; then
    echo "  simple-knn: already installed"
else
    echo "  Building simple-knn..."
    pip install --no-build-isolation "${REPOS_DIR}/gaussian-splatting/submodules/simple-knn"
fi

# fused-ssim
if python -c "import fused_ssim" 2>/dev/null; then
    echo "  fused-ssim: already installed"
else
    echo "  Building fused-ssim..."
    pip install --no-build-isolation "${REPOS_DIR}/gaussian-splatting/submodules/fused-ssim"
fi

# 추가 pip 패키지
pip install -q joblib tensorboard
echo ""

# ============================================================
# 3. PyTorch3D (소스 빌드)
# ============================================================
echo "=== [3/5] PyTorch3D ==="
if python -c "import pytorch3d" 2>/dev/null; then
    echo "  Already installed: $(python -c 'import pytorch3d; print(pytorch3d.__version__)')"
else
    pip install -q fvcore iopath
    if [ ! -d "${REPOS_DIR}/pytorch3d" ]; then
        echo "  Cloning pytorch3d..."
        git clone https://github.com/facebookresearch/pytorch3d.git "${REPOS_DIR}/pytorch3d"
    fi
    echo "  Building pytorch3d (10-20분 소요)..."
    cd "${REPOS_DIR}/pytorch3d"
    FORCE_CUDA=1 pip install --no-build-isolation -e .
    cd "$PROJECT_ROOT"
fi
echo ""

# ============================================================
# 4. COLMAP (소스 빌드)
# ============================================================
echo "=== [4/5] COLMAP ==="
if command -v colmap &>/dev/null; then
    echo "  Already installed: $(colmap -h 2>&1 | head -1)"
else
    # GPU compute capability 확인
    GPU_ARCH=$(python -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'{cap[0]}{cap[1]}')
else:
    print('86')
" 2>/dev/null)
    echo "  GPU architecture: ${GPU_ARCH}"

    if [ ! -d "${REPOS_DIR}/colmap" ]; then
        echo "  Cloning COLMAP..."
        git clone https://github.com/colmap/colmap.git "${REPOS_DIR}/colmap"
    fi

    cd "${REPOS_DIR}/colmap"
    git checkout 3.9.1 2>/dev/null || true

    echo "  Building COLMAP (의존성이 필요합니다)..."
    echo "  만약 빌드 실패 시 아래 명령어를 먼저 실행하세요:"
    echo "    sudo apt-get install -y cmake ninja-build build-essential \\"
    echo "      libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev \\"
    echo "      libboost-system-dev libboost-test-dev libeigen3-dev libflann-dev \\"
    echo "      libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev \\"
    echo "      libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libceres-dev"
    echo ""

    mkdir -p build && cd build
    cmake .. -GNinja \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCGAL_ENABLED=OFF 2>&1 | tail -5
    ninja

    # PATH에 추가 (sudo 없이)
    mkdir -p "$HOME/.local/bin"
    ln -sf "${REPOS_DIR}/colmap/build/src/colmap/exe/colmap" "$HOME/.local/bin/colmap"
    echo "  Installed to ~/.local/bin/colmap"

    cd "$PROJECT_ROOT"
fi
echo ""

# ============================================================
# 5. Jupyter 커널 등록
# ============================================================
echo "=== [5/5] Jupyter Kernel ==="
python -m ipykernel install --user --name 3dgs_sba --display-name "3DGS (py3.10, cu121)"
echo ""

# ============================================================
# 디렉토리 구조 생성
# ============================================================
echo "=== 디렉토리 구조 생성 ==="
mkdir -p "${PROJECT_ROOT}/datasets/mipnerf360"
mkdir -p "${PROJECT_ROOT}/output"
echo "  datasets/mipnerf360/  (여기에 데이터셋 다운로드)"
echo "  output/               (학습 결과 저장)"
echo ""

# ============================================================
# 검증
# ============================================================
echo "=== 환경 검증 ==="
python -c "
import torch
print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import diff_gaussian_rasterization; print('  diff-gaussian-rasterization OK')
from simple_knn._C import distCUDA2; print('  simple-knn OK')
import pytorch3d; print(f'  pytorch3d {pytorch3d.__version__} OK')
import shutil
assert shutil.which('colmap'), 'COLMAP not found'
print('  COLMAP OK')
print()
print('  ALL CHECKS PASSED')
"

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " 다음 단계:"
echo "   1. 데이터셋 다운로드 (선택):"
echo "      cd ${PROJECT_ROOT}/datasets"
echo "      wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
echo "      unzip 360_v2.zip -d mipnerf360"
echo ""
echo "   2. 노트북 실행:"
echo "      jupyter notebook notebooks/"
echo "============================================"
