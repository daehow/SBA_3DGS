# 3DGS + Custom Bundle Adjustment

3D Gaussian Splatting 파이프라인에서 Bundle Adjustment를 직접 구현하고, COLMAP BA와 비교하는 프로젝트.

## Setup

```bash
# 원클릭 설치 (conda 환경 생성 + PyTorch + 3DGS + COLMAP)
git clone https://github.com/daehow/SBA_3DGS.git
cd SBA_3DGS
bash env/setup.sh

# mipnerf360 데이터셋 다운로드 (~12GB)
bash env/mipnerf360.sh
```

> COLMAP 빌드 시 시스템 의존성이 필요합니다. `env/setup.sh` 내 안내를 참고하세요.

## Structure

```
├── env/
│   ├── environment.yml      # Conda 환경 (Python, numpy, scipy 등)
│   ├── setup.sh             # 원클릭 설치 (CUDA 12.1 고정)
│   └── mipnerf360.sh        # 데이터셋 다운로드
├── notebooks/
│   ├── 3dgs_sba.ipynb                  # Custom BA + 3DGS (메인)
│   ├── 3dgs_vanilla.ipynb              # Vanilla 3DGS (baseline)
│   ├── 3dgs_vanilla_step_by_step.ipynb # 3DGS 단계별 실습 (교육용)
│   └── dataset_check.ipynb            # 데이터셋 확인
├── bundle_adjustment.py        # Custom BA 구현
├── bundle_adjustment_utils.py  # 유틸리티 (COLMAP binary I/O, 기하 함수)
└── manual/
    └── setup.md                # 상세 셋업 매뉴얼
```

## Notebooks

| 노트북 | 용도 |
|--------|------|
| `3dgs_sba.ipynb` | COLMAP 결과에 custom BA를 적용하고 3DGS 학습 → vanilla과 비교 |
| `3dgs_vanilla.ipynb` | mipnerf360 데이터셋의 COLMAP 결과 그대로 3DGS 학습 (baseline) |
| `3dgs_vanilla_step_by_step.ipynb` | 3DGS 내부 동작을 단계별로 실행하며 이해하는 교육용 노트북 |
| `dataset_check.ipynb` | 데이터셋 이미지 크기, 카메라 파라미터 확인 |

### 3dgs_vanilla_step_by_step.ipynb

3DGS의 학습 과정을 단계별로 시각화하는 교육용 노트북:

1. **SfM sparse point 로딩** — COLMAP 포인트 클라우드 확인
2. **Gaussian 초기화 디버그** — 5개 Gaussian의 position, opacity, scale, rotation, color 속성 출력
3. **Gaussian 타원체 3D 시각화** — 초기 Gaussian을 scale/rotation 적용한 wireframe 타원체로 표시
4. **초기 렌더링** — 학습 전 Train/Test 뷰 렌더링 vs GT
5. **학습 루프** — 500 iteration마다 스냅샷 (Gaussian 수, 렌더링, PSNR)
6. **Densification 그래프** — 포인트 수 변화, Loss, PSNR 추이
7. **렌더링 비교** — Train/Test 뷰를 시간순으로 나열하여 학습 과정 시각화
8. **학습 후 속성 분석** — opacity/scale 분포 변화, 포인트 클라우드 비교

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
