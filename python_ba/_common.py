"""python_ba 공용 모듈 (run_sba.py / run_sba_3dgs.py 에서 공유).

포함:
- 경로 상수 (OUTPUT_3DGS_ROOT, OUTPUT_BA_ROOT 둘 다 정의)
- COLMAP / 이미지 prepare 스텝
- BA 실행 스텝 (pass-through 또는 methods/{name}.{name}(recon) 호출)
- BA 평가: reprojection error 비교 plot/stats
- 공용 CLI arg 빌더 + Context 객체 + print_header

3DGS 학습/렌더/메트릭 스텝은 python_3dgs/_3dgs_steps.py 에 있다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------- 상수
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_REPO = PROJECT_ROOT / "repos" / "gaussian-splatting"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "mipnerf360"
OUTPUT_3DGS_ROOT = PROJECT_ROOT / "output_3dgs"   # 3DGS 학습까지 포함한 결과
OUTPUT_BA_ROOT = PROJECT_ROOT / "output_ba"       # BA 평가 전용
WORK_ROOT = PROJECT_ROOT / "colmap_ws_sba"
COLMAP_FILES = ("cameras.bin", "images.bin", "points3D.bin")
SCENE_CHOICES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]

# custom_bundle_adjustment 패키지 import 가능하도록 본 파일 폴더를 sys.path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_bundle_adjustment  # noqa: E402


# ---------------------------------------------------------------- 헬퍼
def run(cmd, cwd=None):
    print(f"\n$ {'cd ' + str(cwd) + ' && ' if cwd else ''}" + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def read_u64(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


# ---------------------------------------------------------------- Context
@dataclass
class Context:
    args: argparse.Namespace
    dataset_dir: Path
    src_img_dir: Path
    work_dir: Path
    image_dir: Path
    vanilla_sparse: Path
    sparse_dir: Path
    output_dir: Path
    vanilla_results_path: Path
    ns_label: str


def resolve_context(args: argparse.Namespace, output_root: Path = OUTPUT_3DGS_ROOT) -> Context:
    """Context 생성. `output_root` 로 결과 상위 폴더를 바꿀 수 있다.

    - 3DGS 풀 파이프라인: OUTPUT_3DGS_ROOT (기본)
    - BA 평가 전용 (run_sba.py): OUTPUT_BA_ROOT 를 넘긴다
    """
    dataset_dir = DATASETS_DIR / args.scene
    src_img_dir = dataset_dir / ("images" if args.resolution == 1 else f"images_{args.resolution}")
    work_dir = WORK_ROOT / f"{args.scene}_r{args.resolution}"
    image_dir = work_dir / "images"
    vanilla_sparse = work_dir / "sparse_vanilla" / "0"
    sparse_dir = work_dir / "sparse" / "0"
    ns_label = args.sba_module_name if args.sba_module_name else "passthrough"
    output_dir = Path(args.output) if getattr(args, "output", None) else \
        output_root / args.scene / f"sba_{ns_label}_r{args.resolution}"
    # vanilla 3DGS 결과는 항상 OUTPUT_3DGS_ROOT 에 존재
    vanilla_results_path = OUTPUT_3DGS_ROOT / args.scene / f"vanilla_r{args.resolution}" / "results.json"
    return Context(args, dataset_dir, src_img_dir, work_dir, image_dir,
                   vanilla_sparse, sparse_dir, output_dir, vanilla_results_path, ns_label)


# ---------------------------------------------------------------- 공용 CLI arg
def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--scene", default="bicycle", choices=SCENE_CHOICES)
    p.add_argument("-r", "--resolution", type=int, default=4)
    p.add_argument("--run-colmap", action="store_true",
                   help="COLMAP을 직접 실행 (기본: 데이터셋의 sparse/0 복사)")
    p.add_argument("--sba-module-name", default="",
                   help="methods/{name}.py 의 BA 방법론 이름. 비우면 pass-through. "
                        "사용 가능: " + ", ".join(custom_bundle_adjustment.available()) or "(없음)")
    p.add_argument("--ba-iterations", type=int, default=50)
    p.add_argument("--debug", action="store_true", default=True,
                   help="BA 전/후 reprojection error plot 저장 (기본 ON)")
    p.add_argument("--no-debug", dest="debug", action="store_false")
    p.add_argument("--output", default=None,
                   help="결과 디렉토리 (기본: {output_root}/{scene}/sba_{namespace}_r{resolution}). "
                        "output_root 는 호출 스크립트에 따라 output_3dgs/ 또는 output_ba/")


# ---------------------------------------------------------------- BA 스텝
def step_prepare_images(src_img_dir: Path, work_img_dir: Path):
    work_img_dir.mkdir(parents=True, exist_ok=True)
    if os.listdir(work_img_dir):
        print(f"  이미지 이미 존재: {len(os.listdir(work_img_dir))}장 ({work_img_dir})")
        return
    srcs = sorted(glob.glob(str(src_img_dir / "*")))
    print(f"  이미지 복사: {len(srcs)}장 → {work_img_dir}")
    for s in srcs:
        shutil.copy2(s, work_img_dir / os.path.basename(s))


def step_prepare_vanilla_sparse(ctx: Context):
    args, dataset_dir, work_dir, image_dir, vanilla_sparse = (
        ctx.args, ctx.dataset_dir, ctx.work_dir, ctx.image_dir, ctx.vanilla_sparse
    )
    vanilla_sparse.mkdir(parents=True, exist_ok=True)

    if args.run_colmap:
        db_path = work_dir / "database.db"
        tmp_sparse = work_dir / "_tmp_sparse"
        if db_path.exists():
            db_path.unlink()
        if tmp_sparse.exists():
            shutil.rmtree(tmp_sparse)
        tmp_sparse.mkdir(parents=True)

        print("\n=== COLMAP Step 1/3: Feature Extraction ===")
        run(["colmap", "feature_extractor",
             "--database_path", str(db_path), "--image_path", str(image_dir),
             "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
             "--SiftExtraction.use_gpu", "1"])

        print("\n=== COLMAP Step 2/3: Exhaustive Matching ===")
        run(["colmap", "exhaustive_matcher",
             "--database_path", str(db_path), "--SiftMatching.use_gpu", "1"])

        print("\n=== COLMAP Step 3/3: Mapper ===")
        run(["colmap", "mapper",
             "--database_path", str(db_path), "--image_path", str(image_dir),
             "--output_path", str(tmp_sparse),
             "--Mapper.ba_global_function_tolerance", "0.000001"])

        best_dir, best_count = None, -1
        for sd in sorted(tmp_sparse.iterdir()):
            ib = sd / "images.bin"
            if ib.exists():
                n = read_u64(ib)
                print(f"  {sd.name}: {n} images")
                if n > best_count:
                    best_count, best_dir = n, sd
        assert best_dir is not None, "COLMAP reconstruction 실패"
        for f in COLMAP_FILES:
            shutil.copy2(best_dir / f, vanilla_sparse / f)
        shutil.rmtree(tmp_sparse)
        print(f"\n✅ COLMAP → sparse_vanilla/0/ ({best_count} images)")
    else:
        src_sparse = dataset_dir / "sparse" / "0"
        for f in COLMAP_FILES:
            shutil.copy2(src_sparse / f, vanilla_sparse / f)
        n_imgs = read_u64(vanilla_sparse / "images.bin")
        n_pts = read_u64(vanilla_sparse / "points3D.bin")
        print(f"  mipnerf360/{dataset_dir.name}/sparse/0 → sparse_vanilla/0/ 복사 완료")
        print(f"  Images: {n_imgs}, Points: {n_pts:,}")


def step_run_ba(ctx: Context):
    name = ctx.args.sba_module_name
    ctx.sparse_dir.mkdir(parents=True, exist_ok=True)
    if not name:
        print("\n=== BA pass-through (no method specified) ===")
        custom_bundle_adjustment.passthrough(str(ctx.vanilla_sparse), str(ctx.sparse_dir))
        return
    print(f"\n=== BA method: {name} ===")
    fn = custom_bundle_adjustment.load(name)
    recon = custom_bundle_adjustment.Reconstruction.load(str(ctx.vanilla_sparse))
    recon = fn(recon, max_iterations=ctx.args.ba_iterations)
    recon.save(str(ctx.sparse_dir))


# ---------------------------------------------------------------- BA 평가
def step_eval_ba(ctx: Context) -> dict:
    """BA 전/후 reprojection error 계산 + plot + stats JSON 저장."""
    from custom_bundle_adjustment import Reconstruction

    print("\n=== BA evaluation (reprojection error) ===")
    print("  vanilla reprojection error...")
    ev = Reconstruction.load(ctx.vanilla_sparse).reprojection_errors()
    print(f"  {ctx.ns_label} reprojection error...")
    eb = Reconstruction.load(ctx.sparse_dir).reprojection_errors()

    debug_dir = ctx.output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    if ctx.args.debug:
        _save_reproj_plot(ev, eb, debug_dir / "reproj_error.png", label_after=ctx.ns_label)

    print(f"\n  {'':>20} {'Vanilla':>12} {ctx.ns_label:>12} {'Diff':>12}")
    print("  " + "-" * 58)
    print(f"  {'Mean (px)':>20} {ev.mean():>12.4f} {eb.mean():>12.4f} {eb.mean()-ev.mean():>+12.4f}")
    print(f"  {'Median (px)':>20} {np.median(ev):>12.4f} {np.median(eb):>12.4f} {np.median(eb)-np.median(ev):>+12.4f}")
    print(f"  {'Std (px)':>20} {ev.std():>12.4f} {eb.std():>12.4f} {eb.std()-ev.std():>+12.4f}")
    print(f"  {'<1px (%)':>20} {100*(ev<1).mean():>12.2f} {100*(eb<1).mean():>12.2f} "
          f"{100*(eb<1).mean()-100*(ev<1).mean():>+12.2f}")
    print(f"  {'Observations':>20} {len(ev):>12,} {len(eb):>12,}")

    stats = {
        "scene": ctx.args.scene,
        "resolution": ctx.args.resolution,
        "namespace": ctx.ns_label,
        "n_observations": int(len(ev)),
        "vanilla": {
            "mean": float(ev.mean()), "median": float(np.median(ev)),
            "std": float(ev.std()), "under_1px_pct": float(100 * (ev < 1).mean()),
        },
        "after": {
            "mean": float(eb.mean()), "median": float(np.median(eb)),
            "std": float(eb.std()), "under_1px_pct": float(100 * (eb < 1).mean()),
        },
    }
    stats_path = ctx.output_dir / "ba_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  saved {stats_path}")
    return stats


# ---------------------------------------------------------------- reproj plot
def _save_reproj_plot(ev, eb, out_path: Path, label_after: str = "after"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(0, min(5.0, max(ev.max(), eb.max())), 100)
    axes[0].hist(ev, bins=bins, alpha=0.6, label=f"vanilla (mean={ev.mean():.3f})", color="steelblue")
    axes[0].hist(eb, bins=bins, alpha=0.6, label=f"{label_after} (mean={eb.mean():.3f})", color="coral")
    axes[0].set_xlabel("Reprojection Error (px)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution"); axes[0].legend()

    for errs, lbl, color in ((ev, "vanilla", "steelblue"), (eb, label_after, "coral")):
        s = np.sort(errs)
        axes[1].plot(s, np.arange(1, len(s) + 1) / len(s), label=lbl, color=color, lw=2)
    axes[1].set_xlabel("Reprojection Error (px)"); axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF"); axes[1].set_xlim(0, 3); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    metrics = ["Mean", "Median", "Std", "<1px (%)"]
    van_v = [ev.mean(), np.median(ev), ev.std(), 100 * (ev < 1.0).mean()]
    ba_v = [eb.mean(), np.median(eb), eb.std(), 100 * (eb < 1.0).mean()]
    x = np.arange(len(metrics))
    axes[2].bar(x - 0.17, van_v, 0.35, label="vanilla", color="steelblue", alpha=0.8)
    axes[2].bar(x + 0.17, ba_v, 0.35, label=label_after, color="coral", alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(metrics); axes[2].legend(); axes[2].set_title("Stats")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------- 요약 출력
def print_header(ctx: Context, extra: Optional[dict] = None):
    a = ctx.args
    print("=" * 60)
    print(f"  SCENE        : {a.scene}")
    print(f"  RESOLUTION   : 1/{a.resolution}  (src {ctx.src_img_dir.name}/)")
    print(f"  RUN COLMAP   : {a.run_colmap}")
    if a.sba_module_name:
        print(f"  BA METHOD    : methods.{a.sba_module_name}  (fn: {a.sba_module_name})")
    else:
        print(f"  BA METHOD    : (pass-through, no optimization)")
    print(f"  WORK DIR     : {ctx.work_dir}")
    print(f"  OUTPUT DIR   : {ctx.output_dir}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<12} : {v}")
    print("=" * 60)
