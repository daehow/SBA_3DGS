"""python_3dgs 용 3DGS 스텝: train / render / metrics + metric 비교 plot.

python_ba/_common.py 의 Context 와 헬퍼를 재사용한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# python_ba 경로를 sys.path 에 추가해서 Context/run/GS_REPO 를 import
_BA_DIR = Path(__file__).resolve().parents[1] / "python_ba"
if str(_BA_DIR) not in sys.path:
    sys.path.insert(0, str(_BA_DIR))

from _common import Context, GS_REPO, run  # noqa: E402


def step_train(ctx: Context):
    args = ctx.args
    run([
        sys.executable, "train.py",
        "-s", str(ctx.work_dir),
        "-m", str(ctx.output_dir),
        "--iterations", str(args.iterations),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
        "--resolution", str(args.resolution),
        "--data_device", args.data_device,
        "--eval",
        "--test_iterations", "7000", "15000", str(args.iterations),
        "--save_iterations", "7000", "15000", str(args.iterations),
    ], cwd=GS_REPO)


def step_render(ctx: Context):
    run([
        sys.executable, "render.py",
        "-m", str(ctx.output_dir),
        "--iteration", str(ctx.args.iterations),
        "--data_device", ctx.args.data_device,
    ], cwd=GS_REPO)


def step_metrics(ctx: Context):
    run([sys.executable, "metrics.py", "-m", str(ctx.output_dir)], cwd=GS_REPO)
    sba_path = ctx.output_dir / "results.json"
    if not sba_path.exists():
        print("  results.json 이 없습니다.")
        return
    with open(sba_path) as f:
        sba_results = json.load(f)
    print("\n=== 3DGS 결과 ===")
    for it, m in sba_results.items():
        print(f"[{it}]")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

    if ctx.vanilla_results_path.exists():
        with open(ctx.vanilla_results_path) as f:
            vanilla_results = json.load(f)
        sba_key = list(sba_results.keys())[-1]
        van_key = list(vanilla_results.keys())[-1]
        sba_m, van_m = sba_results[sba_key], vanilla_results[van_key]
        print("\n=== Vanilla vs Custom BA 비교 ===")
        print(f"{'Metric':>10} {'Vanilla':>12} {'Custom BA':>12} {'Diff':>12}")
        print("-" * 50)
        for metric in ("PSNR", "SSIM", "LPIPS"):
            v, s = van_m.get(metric, 0.0), sba_m.get(metric, 0.0)
            d = s - v
            better = ("+" if d > 0 else "-" if d < 0 else "=") if metric != "LPIPS" else ("+" if d < 0 else "-" if d > 0 else "=")
            print(f"{metric:>10} {v:>12.4f} {s:>12.4f} {d:>+12.4f} {better}")
        if ctx.args.debug:
            _save_metric_bar_plot(van_m, sba_m, ctx.output_dir / "debug" / "metric_bar.png",
                                  title=f"{ctx.args.scene}/{ctx.ns_label}_r{ctx.args.resolution}")
    else:
        print(f"\n  (Vanilla 결과 없음 — run_vanilla.py 먼저 실행하면 비교 plot 생성)")


def _save_metric_bar_plot(van_m, sba_m, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["PSNR", "SSIM", "LPIPS"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m in zip(axes, metrics):
        vals = [van_m.get(m, 0.0), sba_m.get(m, 0.0)]
        bars = ax.bar(["vanilla", "custom"], vals, color=["steelblue", "coral"], alpha=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(m)
    plt.suptitle(f"3DGS: vanilla vs custom — {title}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")
