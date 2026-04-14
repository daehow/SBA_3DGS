# 3DGS + Custom Bundle Adjustment

3D Gaussian Splatting 파이프라인에서 Bundle Adjustment를 직접 구현하고, COLMAP BA와 비교하는 프로젝트.

## Setup

```bash
# 1. Conda 환경 생성
conda env create -f env/environment.yml
conda activate 3dgs_sba

# 2. PyTorch, 3DGS, PyTorch3D, COLMAP 설치 (한 번에)
bash env/setup.sh

# 3. (선택) mipnerf360 데이터셋 다운로드
cd datasets
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d mipnerf360
```

> COLMAP 빌드 시 시스템 의존성이 필요합니다. `env/setup.sh` 내 안내를 참고하세요.

## Structure

```
├── env/
│   ├── environment.yml      # Conda 환경 (Python, numpy, scipy 등)
│   └── setup.sh             # PyTorch, 3DGS, COLMAP 등 설치
├── notebooks/
│   ├── 3dgs_sba.ipynb       # Custom BA + 3DGS 파이프라인 (메인)
│   ├── 3dgs_vanilla.ipynb   # Vanilla 3DGS (COLMAP 결과 그대로 사용)
│   └── dataset_check.ipynb  # 데이터셋 확인
├── bundle_adjustment.py     # Custom BA 구현 (COLMAP 결과 읽기 → BA → 저장)
├── bundle_adjustment_utils.py  # 유틸리티 (COLMAP binary I/O, 기하 함수)
└── manual/
    └── setup.md             # 상세 셋업 매뉴얼
```

## Notebooks

| 노트북 | 용도 |
|--------|------|
| `3dgs_sba.ipynb` | COLMAP 결과에 custom BA를 적용하고 3DGS 학습 → vanilla과 비교 |
| `3dgs_vanilla.ipynb` | mipnerf360 데이터셋의 COLMAP 결과 그대로 3DGS 학습 (baseline) |
| `dataset_check.ipynb` | 데이터셋 이미지 크기, 카메라 파라미터 확인 |

## Bundle Adjustment

`bundle_adjustment.py`는 COLMAP의 BA 결과를 읽어서 scipy LM solver로 BA를 재실행합니다.

```python
from bundle_adjustment import BundleAdjustment

ba = BundleAdjustment("path/to/sparse/0")  # COLMAP 결과 로드
ba.run(max_iterations=50)                   # BA 실행
ba.export("path/to/output/0")              # 결과 저장
```

### 비교 결과 (bicycle, mipnerf360)

| | COLMAP BA | Custom BA |
|---|---|---|
| Mean reproj error | 0.8753 px | 0.8587 px |
| Median reproj error | 0.6961 px | 0.6502 px |
| BA 시간 | - | ~40초 |
