# 실행 매뉴얼

모든 명령은 프로젝트 루트(`/home/hyoseok/projects/SBA_3DGS`)에서 실행.

---

## 빠른 실행 순서 (Quick Start)

새 BA 방법론 `my_method` 를 추가하고, reproj error + 3DGS 품질까지 평가하는 최단 경로.

```bash
# 0) 환경 활성화
conda activate 3dgs_sba

# 1) 새 method 파일 생성 (파일명=함수명=namespace)
#    python_ba/custom_bundle_adjustment/methods/my_method.py
#    def my_method(recon, **kwargs) -> Reconstruction: ...
#    (기본 예시: scipy_lm.py 복사 후 함수명만 my_method 로)

# 2) BA 만 빠르게 (수 초~수 분) → reproj error 확인
python python_ba/run_sba.py --scene bicycle -r 2 --sba-module-name my_method
python python_ba/run_summary_ba.py --scene bicycle -r 2

# 3) BA + 3DGS 풀 (~30~40 분) → PSNR/SSIM/LPIPS 확인
python python_3dgs/run_sba_3dgs.py --scene bicycle -r 2 --sba-module-name my_method
python python_3dgs/run_summary.py --scene bicycle -r 2 --gt vanilla
```

**결과 위치**
- BA 평가: `output_ba/bicycle/sba_my_method_r2/ba_stats.json` + `debug/reproj_error.png`
- 3DGS 평가: `output_3dgs/bicycle/sba_my_method_r2/results.json` + `debug/metric_bar.png`
- BA 요약 CSV: `output_ba/summary/{ts}.csv`
- 3DGS 요약 CSV: `output_3dgs/summary/{ts}.csv`

> method 없이 `--sba-module-name` 을 비우면 `passthrough` (COLMAP 결과 그대로) 가 기준선으로 실행됨.

---

## 새 BA 방법론(method) 추가하기 — 핵심 규약

**파일명 == 함수명 == `--sba-module-name` 값.**

모든 BA 방법론은 `python_ba/custom_bundle_adjustment/methods/{name}.py` 에 두고,
그 파일 안에 **파일명과 동일한 이름의 함수**를 아래 고정 시그니처로 노출하면 끝.

```python
# python_ba/custom_bundle_adjustment/methods/{name}.py
from ..bundle_adjustment import Reconstruction

def {name}(recon: Reconstruction, **kwargs) -> Reconstruction:
    # ... 여기에 최적화 로직 ...
    return recon   # 같은 객체 in-place 수정 후 반환해도 OK
```

- 입력/출력 타입은 **`Reconstruction` 고정** (COLMAP sparse/0 를 `Reconstruction.load(path)` 로 얻고, 다 끝나면 `recon.save(path)`. 이건 run_sba.py 가 대신 해줌).
- 파일명만 규약에 맞으면 `custom_bundle_adjustment.available()` 에서 자동 수집 → `--sba-module-name {name}` 으로 바로 호출 가능.

### 예시: `scipy_lm` 의 동작 구조
`python_ba/custom_bundle_adjustment/methods/scipy_lm.py` 가 존재하고, 그 안에
`def scipy_lm(recon, max_iterations=50, **_) -> Reconstruction:` 이 정의돼 있음.
그래서 터미널에서 `--sba-module-name scipy_lm` 를 넘기면 run_sba.py 가
`scipy_lm(recon)` 을 호출하고 결과를 `output_ba/{scene}/sba_scipy_lm_r{N}/` 에 저장한다.

### 새 method `my_method` 추가 흐름
1. `python_ba/custom_bundle_adjustment/methods/my_method.py` 생성
2. 안에 `def my_method(recon, **kwargs) -> Reconstruction:` 정의
3. 실행: `python python_ba/run_sba.py --sba-module-name my_method`
4. 결과: `output_ba/{scene}/sba_my_method_r{N}/ba_stats.json` + `debug/reproj_error.png`

> `--sba-module-name` 을 비우면 `custom_bundle_adjustment.passthrough` 가 돌고 결과 라벨은 `passthrough`. 즉 "COLMAP 결과 그대로" 가 기준선이 됨.

### 참고: `bundle_adjustment.py` 는 뭐가 다른가
`python_ba/custom_bundle_adjustment/bundle_adjustment.py` 는 **라이브러리**
(`Reconstruction` dataclass + `load/save/reprojection_errors/summary`, `passthrough()`).
최적화 로직을 들고 있지 않다. method 저자는 이 파일을 건드리지 않고
`methods/{name}.py` 만 추가하면 된다.

---

## 디렉토리 맵

| 경로 | 용도 |
|---|---|
| `python_ba/` | BA 최적화 + 평가 스크립트 |
| `python_ba/custom_bundle_adjustment/` | `Reconstruction` + `methods/{name}.py` |
| `python_3dgs/` | 3DGS 학습/렌더/요약 스크립트 |
| `colmap_ws_sba/{scene}_r{N}/` | 작업 디렉토리 (`images/`, `sparse_vanilla/0/`, `sparse/0/`) |
| `output_ba/{scene}/sba_{ns}_r{N}/` | BA 평가 산출물 (`ba_stats.json`, `debug/reproj_error.png`) |
| `output_3dgs/{scene}/{kind}/` | 3DGS 학습 산출물 (`results.json`, `point_cloud/`, ...) |

`ns` 는 BA 방법론 이름(`passthrough`, `scipy_lm`, 사용자 추가). method 파일 규약:
`python_ba/custom_bundle_adjustment/methods/{name}.py` 에 `def {name}(recon, **kw) -> Reconstruction`.

---

## 1. BA 만 실행 + 결과 보기

방법론(methods/{name}.py)을 바꿔가며 reprojection error 가 어떻게 달라지는지 확인하는 빠른 루프.
3DGS 학습은 돌리지 않아 수 초 ~ 수 분.

### 실행
```bash
# pass-through (최적화 없이 COLMAP 결과 그대로)
python python_ba/run_sba.py --scene bicycle -r 2

# 특정 method 적용
python python_ba/run_sba.py --scene bicycle -r 2 --sba-module-name scipy_lm

# method 목록
python -c "import sys; sys.path.insert(0,'python_ba'); \
           import custom_bundle_adjustment as c; print(c.available())"
```

### 중요 옵션
| 옵션 | 기본 | 설명 |
|---|---|---|
| `--scene` | `bicycle` | bicycle, bonsai, counter, garden, kitchen, room, stump |
| `-r`, `--resolution` | `4` | 1/2/4/8 |
| `--sba-module-name` | 빈값 | `passthrough` · `scipy_lm` · 직접 추가한 이름 |
| `--ba-iterations` | `50` | BA 최대 LM iteration |
| `--run-colmap` | off | dataset 의 `sparse/0` 복사 대신 COLMAP 직접 실행 |
| `--steps` | `prepare ba eval` | `prepare` / `ba` / `eval` 중 원하는 것만 |
| `--no-debug` | — | reproj plot 저장 끔 |

### 결과 위치
```
output_ba/{scene}/sba_{ns}_r{N}/
├── ba_stats.json              # vanilla / after mean·median·std·<1px%
└── debug/reproj_error.png     # histogram + CDF + stats bar
```

콘솔 마지막에 vanilla vs after 비교 표가 찍힘.

---

## 2. BA + 3DGS 풀 파이프라인

BA 후 3DGS 학습 → 렌더 → 메트릭 까지 이어서 실행. 30k iter 기준 TITAN RTX 에서 30~40 분.

### 실행
```bash
# vanilla baseline (BA 없이 3DGS 만)
python python_3dgs/run_vanilla.py --scene bicycle -r 2

# pass-through + 3DGS (BA 방법론 미적용 기준선)
python python_3dgs/run_sba_3dgs.py --scene bicycle -r 2

# scipy_lm BA + 3DGS
python python_3dgs/run_sba_3dgs.py --scene bicycle -r 2 --sba-module-name scipy_lm

# 이미 BA 는 돈 경우 3DGS 단계만
python python_3dgs/run_sba_3dgs.py --scene bicycle -r 2 --sba-module-name scipy_lm \
       --steps train render metrics
```

### 중요 옵션 (공용 + 3DGS 고유)
| 옵션 | 기본 | 설명 |
|---|---|---|
| (위 1번의 모든 BA 옵션) | | |
| `--iterations` | `30000` | 3DGS train iteration |
| `--densify_grad_threshold` | `0.001` | |
| `--data_device` | `cuda` | 이미지 텐서 상주 디바이스 |
| `--steps` | `prepare ba eval train render metrics` | 원하는 단계만 |

### 결과 위치
```
output_3dgs/{scene}/
├── vanilla_r{N}/                   # run_vanilla.py 결과
│   ├── results.json                # PSNR/SSIM/LPIPS
│   ├── point_cloud/iteration_*/
│   └── test/ours_30000/renders/
└── sba_{ns}_r{N}/                  # run_sba_3dgs.py 결과
    ├── ba_stats.json               # BA 단계에서 같이 생성
    ├── results.json
    ├── point_cloud/...
    └── debug/metric_bar.png        # vanilla 비교 bar
```

---

## 3. BA 방법론 결과 요약 (reprojection error)

`output_ba/` 의 모든 `ba_stats.json` 을 긁어모아 CSV.

### 실행
```bash
python python_ba/run_summary_ba.py                          # 전부
python python_ba/run_summary_ba.py -r 2                     # r=2 만
python python_ba/run_summary_ba.py --scene bicycle -r 2
python python_ba/run_summary_ba.py --out /tmp/ba.csv        # CSV 경로 지정
```

### 결과
```
output_ba/summary/{YYYYMMDD_HHMMSS}.csv
```
컬럼: `scene, namespace, resolution, n_obs,
vanilla_mean/median/lt1pct, after_mean/median/lt1pct, diff_mean/median, dir`

콘솔 표 예:
```
scene      namespace    res  van mean  aft mean    diff  van med  aft med   diff     n_obs
bicycle    scipy_lm     2     1.2699    1.2446  -0.0254  1.1191   1.0238 -0.0953   254,466
```

`diff_mean / diff_median` 이 **음수** 면 BA 로 reproj error 가 줄어든 것.

---

## 4. BA + 3DGS 품질 요약 (PSNR / SSIM / LPIPS)

`output_3dgs/` 의 모든 `results.json` 을 긁어모아 CSV. vanilla 행을 GT 로 올릴 수도 있음.

### 실행
```bash
python python_3dgs/run_summary.py                                   # 전부
python python_3dgs/run_summary.py -r 2                              # r=2 만
python python_3dgs/run_summary.py --scene bicycle -r 2
python python_3dgs/run_summary.py --scene bicycle -r 2 --gt vanilla # vanilla 를 GT 상단에
python python_3dgs/run_summary.py --iteration ours_7000             # 특정 iter 만
```

### 결과
```
output_3dgs/summary/{YYYYMMDD_HHMMSS}.csv
```
컬럼: `GT, scene, namespace, resolution, iteration, PSNR, SSIM, LPIPS, dir`

콘솔 표 예:
```
GT   scene       namespace   res   iter           PSNR     SSIM    LPIPS
GT   bicycle     vanilla     2     ours_30000   22.9046   0.5792   0.4786
     bicycle     scipy_lm    2     ours_30000   23.7546   0.6129   0.4210
```

PSNR/SSIM 은 **높을수록**, LPIPS 는 **낮을수록** 좋음.

---

## 빠른 참조

| 하고 싶은 일 | 스크립트 |
|---|---|
| vanilla 3DGS 만 | `python_3dgs/run_vanilla.py` |
| BA method 개발·검증 (빠름) | `python_ba/run_sba.py` |
| BA + 3DGS 풀 파이프라인 | `python_3dgs/run_sba_3dgs.py` |
| BA 방법론 reproj error 요약 | `python_ba/run_summary_ba.py` |
| BA+3DGS 품질 요약 | `python_3dgs/run_summary.py` |

각 스크립트의 상세 옵션은 `--help` 로 확인.
