# 3D Gaussian Splatting 실습 환경 셋업 매뉴얼

> **이 문서는 Claude Code가 읽고 실행하기 위한 매뉴얼입니다.**  
> 모든 작업은 `/home/daeho/storage/3dgs_sba/` 아래에서 수행합니다.

---

## 디렉토리 구조 (최종)

```
/home/daeho/storage/3dgs_sba/
├── manual/
│   └── setup.md              ← 이 파일
├── repos/
│   ├── gaussian-splatting/    ← 3DGS 메인 (submodule 포함)
│   │   ├── submodules/
│   │   │   ├── diff-gaussian-rasterization/
│   │   │   └── simple-knn/
│   │   └── ...
│   └── pytorch3d/             ← Facebook Research pytorch3d
├── datasets/
│   ├── sample_images/         ← 직접 촬영 이미지 넣는 곳
│   └── mipnerf360/            ← (선택) 공개 데이터셋
├── colmap_ws/                 ← COLMAP 작업 디렉토리
│   └── <scene_name>/
│       ├── images/
│       ├── database.db
│       └── sparse/
├── output/                    ← 3DGS 학습 결과
├── notebooks/
│   └── 3dgs_pipeline.ipynb    ← 실습 노트북
└── env/
    └── environment.yml
```

---

## 환경 정보

| 항목 | 값 |
|---|---|
| OS | Ubuntu 20.04 |
| GPU | NVIDIA RTX 3070 (8GB VRAM) |
| CUDA | 12.1 |
| Python | 3.10 |
| PyTorch | cu121 빌드 |
| conda env 이름 | `3dgs` |

---

## Part 1. 시스템 의존성 & CUDA 12.1 설치

### 1-1. NVIDIA CUDA 12.1 저장소 등록 (Ubuntu 20.04)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
```

### 1-2. CUDA Toolkit 설치 (이미 설치되어 있으면 skip)

```bash
sudo apt-get install -y cuda-toolkit-12-1
```

설치 후 `~/.bashrc`에 아래 추가:

```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1
```

### 1-3. 검증

```bash
nvidia-smi
nvcc --version   # CUDA 12.1이 표시되어야 함
```

---

## Part 2. Conda 환경 생성

### 2-1. environment.yml 생성

파일 위치: `/home/daeho/storage/3dgs_sba/env/environment.yml`

```yaml
name: 3dgs
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pillow
  - matplotlib
  - tqdm
  - plyfile
  - opencv
  - ipykernel
  - jupyter
  - pip
```

> **주의**: PyTorch는 yml에 넣지 않는다. CUDA 버전별 pip 설치가 더 안정적이기 때문.

### 2-2. 환경 생성 & 활성화

```bash
conda env create -f /home/daeho/storage/3dgs_sba/env/environment.yml
conda activate 3dgs
```

### 2-3. PyTorch 설치 (CUDA 12.1)

**본인 환경 (CUDA 12.1 — RTX 3070):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**다른 사람 환경 (시스템에 맞는 CUDA가 이미 잡혀 있는 경우):**

```bash
pip install torch torchvision torchaudio
```

> 어떤 방식이든 설치 후 아래로 검증:
> ```python
> import torch
> print(torch.__version__)
> print(torch.cuda.is_available())       # True
> print(torch.version.cuda)              # 12.1
> print(torch.cuda.get_device_name(0))   # NVIDIA GeForce RTX 3070
> ```

### 2-4. Jupyter 커널 등록

```bash
python -m ipykernel install --user --name 3dgs --display-name "3DGS (py3.10, cu121)"
```

---

## Part 3. 3DGS 리포지토리 & 서브모듈 클론

3개의 핵심 저장소를 `/home/daeho/storage/3dgs_sba/repos/`에 저장한다.

### 3-1. gaussian-splatting 클론 (submodule 포함)

```bash
cd /home/daeho/storage/3dgs_sba/repos
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive
```

이 명령으로 아래 3개가 모두 받아진다:
1. `gaussian-splatting/` — 메인 리포지토리 (train.py, render.py 등)
2. `gaussian-splatting/submodules/diff-gaussian-rasterization/` — differentiable Gaussian rasterizer (CUDA)
3. `gaussian-splatting/submodules/simple-knn/` — KNN 유틸리티 (CUDA)

### 3-2. CUDA 확장 모듈 빌드 (diff-gaussian-rasterization + simple-knn)

```bash
conda activate 3dgs
cd /home/daeho/storage/3dgs_sba/repos/gaussian-splatting

# diff-gaussian-rasterization 빌드 & 설치
pip install submodules/diff-gaussian-rasterization

# simple-knn 빌드 & 설치
pip install submodules/simple-knn
```

### 3-3. 빌드 검증

```python
import diff_gaussian_rasterization
print("diff-gaussian-rasterization OK")

from simple_knn._C import distCUDA2
print("simple-knn OK")
```

> **빌드 실패 시 체크리스트:**
> - `nvcc --version` → CUDA 12.1 확인
> - `which nvcc` → `/usr/local/cuda-12.1/bin/nvcc` 확인
> - `echo $CUDA_HOME` → `/usr/local/cuda-12.1` 확인
> - `sudo apt install build-essential` 설치 여부

---

## Part 4. PyTorch3D 설치 (소스 빌드)

pytorch3d는 torch/CUDA 버전에 민감하므로 **소스에서 빌드하는 것이 가장 확실**하다.

### 4-1. 빌드 의존성 설치

```bash
conda activate 3dgs
pip install fvcore iopath
```

### 4-2. 소스 클론 & 빌드

```bash
cd /home/daeho/storage/3dgs_sba/repos
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d

# CUDA 강제 활성화 + 빌드 (10~20분 소요)
FORCE_CUDA=1 pip install -e .
```

> `FORCE_CUDA=1`을 설정해야 CUDA 가속 컴포넌트가 빌드된다.  
> `-e` (editable) 모드로 설치하면 이후 코드 수정이 바로 반영된다.

### 4-3. 검증

```python
import pytorch3d
print(pytorch3d.__version__)

# CUDA 기반 렌더러 로드 확인
from pytorch3d.renderer import MeshRenderer
from pytorch3d.structures import Meshes
print("pytorch3d OK")
```

> **빌드 실패 시:**
> - `pip install ninja` 후 재시도 (빌드 속도 향상 + 일부 에러 해결)
> - torch 버전 확인: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
> - pytorch3d는 torch를 재설치하면 반드시 rebuild 해야 한다:
>   ```bash
>   cd /home/daeho/storage/3dgs_sba/repos/pytorch3d
>   rm -rf build/ **/*.so
>   FORCE_CUDA=1 pip install -e .
>   ```

---

## Part 5. COLMAP 설치 (소스 빌드, CUDA 지원)

Ubuntu 20.04 apt의 COLMAP은 3.6으로 오래되고 CUDA 미지원이므로, **소스에서 빌드**한다.

### 5-1. 의존성 설치

```bash
sudo apt-get install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```

### 5-2. COLMAP 빌드

```bash
cd /home/daeho/storage/3dgs_sba/repos
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout 3.8    # 안정 버전

mkdir build && cd build
cmake .. -GNinja \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DCMAKE_BUILD_TYPE=Release
ninja
sudo ninja install
```

> `CMAKE_CUDA_ARCHITECTURES="86"` → RTX 3070의 compute capability.  
> 다른 GPU 사용 시 변경 (RTX 2080: 75, RTX 4090: 89, A100: 80).

### 5-3. 검증

```bash
colmap -h
```

---

## Part 6. COLMAP 실행 — 이미지에서 카메라 포즈 추출

### 사용 가능한 이미지 소스

| 소스 | 경로 | 설명 |
|---|---|---|
| 직접 촬영 이미지 | `datasets/sample_images/` | 스마트폰/카메라로 촬영한 이미지를 여기에 넣는다 |
| 공개 데이터셋 | `datasets/mipnerf360/` | 아래 6-0에서 다운로드 |

### 6-0. (선택) 공개 데이터셋 다운로드

Mip-NeRF 360 데이터셋에서 원본 이미지만 사용하고, COLMAP은 직접 수행한다.

```bash
cd /home/daeho/storage/3dgs_sba/datasets
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d mipnerf360
ls mipnerf360/
```

> 약 12GB. 특정 씬만 필요하면 압축 해제 후 원하는 폴더만 사용.  
> **기존 `sparse/` 폴더가 있더라도 이 매뉴얼에서는 COLMAP을 직접 실행한다.**

### 6-1. COLMAP 작업 디렉토리 준비

```bash
# SCENE_NAME 변수를 실제 씬 이름으로 변경
SCENE_NAME="my_scene"
COLMAP_WS="/home/daeho/storage/3dgs_sba/colmap_ws/${SCENE_NAME}"

mkdir -p ${COLMAP_WS}/images

# 이미지 복사 (둘 중 하나 선택)
# [직접 촬영] cp /path/to/your/photos/*.jpg ${COLMAP_WS}/images/
# [공개 데이터셋] cp /home/daeho/storage/3dgs_sba/datasets/mipnerf360/bicycle/images/* ${COLMAP_WS}/images/
```

### 6-2. 이미지 리사이즈 (RTX 3070 VRAM 절약)

해상도가 1920px 이상이면 리사이즈 권장:

```python
from PIL import Image
import os, glob

IMG_DIR = "/home/daeho/storage/3dgs_sba/colmap_ws/my_scene/images"
MAX_RES = 1600

for p in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
    try:
        img = Image.open(p)
        w, h = img.size
        if max(w, h) > MAX_RES:
            ratio = MAX_RES / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            img.save(p)
            print(f"Resized: {os.path.basename(p)} → {img.size}")
    except Exception:
        pass
```

### 6-3. COLMAP 실행 — 방법 A: convert.py (간편)

3DGS에 포함된 `convert.py`가 COLMAP 전체 파이프라인을 자동 실행한다.

```bash
conda activate 3dgs
cd /home/daeho/storage/3dgs_sba/repos/gaussian-splatting
python convert.py -s /home/daeho/storage/3dgs_sba/colmap_ws/my_scene
```

### 6-4. COLMAP 실행 — 방법 B: 수동 단계별

각 단계를 직접 제어하고 싶을 때 사용한다.

```bash
COLMAP_WS="/home/daeho/storage/3dgs_sba/colmap_ws/my_scene"

# Step 1: Feature Extraction — 각 이미지에서 SIFT 특징점 추출
colmap feature_extractor \
    --database_path ${COLMAP_WS}/database.db \
    --image_path ${COLMAP_WS}/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 1

# Step 2: Feature Matching — 모든 이미지 쌍 매칭
colmap exhaustive_matcher \
    --database_path ${COLMAP_WS}/database.db \
    --SiftMatching.use_gpu 1

# Step 3: Sparse Reconstruction — 카메라 포즈 + 3D 포인트 복원
mkdir -p ${COLMAP_WS}/sparse
colmap mapper \
    --database_path ${COLMAP_WS}/database.db \
    --image_path ${COLMAP_WS}/images \
    --output_path ${COLMAP_WS}/sparse

# Step 4: Undistortion (선택, 3DGS에서 자동 처리되기도 함)
mkdir -p ${COLMAP_WS}/dense
colmap image_undistorter \
    --image_path ${COLMAP_WS}/images \
    --input_path ${COLMAP_WS}/sparse/0 \
    --output_path ${COLMAP_WS}/dense \
    --output_type COLMAP
```

### 6-5. COLMAP 결과 검증

```bash
ls -la ${COLMAP_WS}/sparse/0/
# cameras.bin, images.bin, points3D.bin 3개가 있어야 한다
```

---

## Part 7. 3DGS 학습

### 7-1. 학습 실행

```bash
conda activate 3dgs
cd /home/daeho/storage/3dgs_sba/repos/gaussian-splatting

python train.py \
    -s /home/daeho/storage/3dgs_sba/colmap_ws/my_scene \
    -m /home/daeho/storage/3dgs_sba/output/my_scene \
    --iterations 30000 \
    --densify_grad_threshold 0.0004 \
    --resolution 2 \
    --eval \
    --test_iterations 7000 15000 30000 \
    --save_iterations 7000 15000 30000
```

**RTX 3070 파라미터 가이드:**

| 파라미터 | 기본값 | RTX 3070 권장 | 설명 |
|---|---|---|---|
| `--resolution` | -1 (원본) | **2** (1/2 해상도) | VRAM 절약 핵심 |
| `--densify_grad_threshold` | 0.0002 | **0.0004** | Gaussian 수 제한 |
| `--iterations` | 30000 | 30000 | 약 20~40분 |

**VRAM 부족 시 (CUDA out of memory):**

```bash
python train.py \
    -s /home/daeho/storage/3dgs_sba/colmap_ws/my_scene \
    -m /home/daeho/storage/3dgs_sba/output/my_scene \
    --iterations 30000 \
    --densify_grad_threshold 0.001 \
    --resolution 4 \
    --save_iterations 7000 30000
```

### 7-2. 렌더링

```bash
python render.py \
    -m /home/daeho/storage/3dgs_sba/output/my_scene \
    --iteration 30000
```

### 7-3. 정량 평가 (PSNR / SSIM / LPIPS)

```bash
python metrics.py -m /home/daeho/storage/3dgs_sba/output/my_scene
```

결과는 `output/my_scene/results.json`에 저장된다.

---

## Part 8. 실습 노트북 (ipynb) 생성

Claude Code는 위 Part 1~7 의 내용을 기반으로 `/home/daeho/storage/3dgs_sba/notebooks/3dgs_pipeline.ipynb`를 생성한다.

노트북 구성:

1. **셀 1**: 환경 검증 (torch, CUDA, GPU, 모듈 import)
2. **셀 2**: 경로 변수 설정 (`SCENE_NAME` 등 — 여기만 수정)
3. **셀 3**: 이미지 준비 & 리사이즈
4. **셀 4**: COLMAP 실행 (`convert.py` 호출)
5. **셀 5**: COLMAP 결과 검증 (cameras.bin, images.bin, points3D.bin 확인)
6. **셀 6**: 3DGS 학습 (`!python train.py ...`)
7. **셀 7**: 렌더링 (`!python render.py ...`)
8. **셀 8**: 메트릭 평가 & 결과 출력
9. **셀 9**: 렌더링 이미지 시각화 (matplotlib)

> 각 셀은 독립적으로 실행 가능해야 하며, 셀 실행 결과에 `✅ / ❌`를 표시한다.

---

## 검증 체크리스트

모든 설치가 끝나면 아래를 순서대로 확인한다:

```python
# 1. PyTorch + CUDA
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")

# 2. diff-gaussian-rasterization
import diff_gaussian_rasterization
print("diff-gaussian-rasterization OK")

# 3. simple-knn
from simple_knn._C import distCUDA2
print("simple-knn OK")

# 4. pytorch3d
import pytorch3d
from pytorch3d.renderer import MeshRenderer
print(f"pytorch3d {pytorch3d.__version__} OK")

# 5. COLMAP
import shutil
assert shutil.which("colmap"), "COLMAP not in PATH"
print("COLMAP OK")
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `CUDA out of memory` | VRAM 부족 | `--resolution 4`, `--densify_grad_threshold 0.001` |
| `nvcc: command not found` | CUDA PATH 미설정 | `~/.bashrc`에 CUDA 경로 추가 (Part 1-2) |
| `diff-gaussian-rasterization` 빌드 실패 | nvcc 버전 불일치 | `echo $CUDA_HOME`, `nvcc --version` 확인 |
| `pytorch3d` import 시 `undefined symbol` | torch 재설치 후 rebuild 안 함 | pytorch3d `rm -rf build/ && pip install -e .` |
| COLMAP `0 images registered` | 이미지 오버랩 부족 | 50장 이상, 인접 이미지 70% 겹침 필요 |
| COLMAP 빌드 시 `CMake Error` | 의존성 누락 | Part 5-1 패키지 전부 설치 확인 |
| `pip install torch` 후 CUDA 미인식 | 기본 pip가 CPU 버전 설치 | `--index-url https://download.pytorch.org/whl/cu121` 명시 |

---

## 촬영 가이드 (직접 촬영 시)

| 항목 | 권장 | 비권장 |
|---|---|---|
| 이미지 수 | 50~200장 | 10장 미만 |
| 촬영 패턴 | 대상 주위 360° 고르게 | 한쪽에서만 |
| 인접 이미지 겹침 | 70% 이상 | 큰 간격 |
| 조명 | 일정 (실내등, 흐린 날) | 직사광선 |
| 초점/노출 | 고정 (Pro 모드) | 자동 |
| 움직이는 물체 | 없음 | 사람, 차량 |