#!/usr/bin/env python
"""Custom BA + 3DGS 풀 파이프라인.

흐름:
    prepare → ba → eval (reprojection error) → train → render → metrics
    결과는 output_3dgs/{scene}/sba_{ns}_r{N}/ 에 누적 저장
    (ba_stats.json, results.json, point_cloud/, ...)

사용법:
    python python_3dgs/run_sba_3dgs.py                                  # pass-through + 3DGS
    python python_3dgs/run_sba_3dgs.py --scene bicycle -r 2 --sba-module-name scipy_lm
    python python_3dgs/run_sba_3dgs.py --steps train render metrics     # BA 완료 후 3DGS 만

BA 만 돌려서 reprojection error 를 빠르게 보려면 python_ba/run_sba.py.
"""
import argparse
import sys
from pathlib import Path

# python_ba 경로를 sys.path 에 추가해서 _common / custom_bundle_adjustment 재사용
_BA_DIR = Path(__file__).resolve().parents[1] / "python_ba"
if str(_BA_DIR) not in sys.path:
    sys.path.insert(0, str(_BA_DIR))

from _common import (  # noqa: E402
    add_common_args, resolve_context,
    step_prepare_images, step_prepare_vanilla_sparse, step_run_ba, step_eval_ba,
    print_header,
)
from _3dgs_steps import step_train, step_render, step_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--iterations", type=int, default=30000, help="3DGS train iterations")
    p.add_argument("--densify_grad_threshold", type=float, default=0.001)
    p.add_argument("--data_device", default="cuda")
    p.add_argument("--steps", nargs="+",
                   default=["prepare", "ba", "eval", "train", "render", "metrics"],
                   choices=["prepare", "ba", "eval", "train", "render", "metrics"])
    return p.parse_args()


def main():
    args = parse_args()
    ctx = resolve_context(args)

    print_header(ctx, extra={
        "ITERATIONS": args.iterations,
        "DATA_DEVICE": args.data_device,
        "STEPS": ", ".join(args.steps),
    })
    assert ctx.src_img_dir.is_dir(), f"source image dir not found: {ctx.src_img_dir}"
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    if "prepare" in args.steps:
        step_prepare_images(ctx.src_img_dir, ctx.image_dir)
        step_prepare_vanilla_sparse(ctx)

    if "ba" in args.steps:
        step_run_ba(ctx)

    if "eval" in args.steps:
        step_eval_ba(ctx)

    if "train" in args.steps:
        step_train(ctx)

    if "render" in args.steps:
        step_render(ctx)

    if "metrics" in args.steps:
        step_metrics(ctx)

    print("\n완료.")


if __name__ == "__main__":
    main()
