#!/usr/bin/env python
"""BA 전용 파이프라인: BA 최적화 + reprojection error 평가만.

3DGS 학습/렌더/메트릭은 포함하지 않음. method 구현을 바꿨을 때 reprojection error
가 어떻게 달라지는지만 빠르게 확인하고 싶을 때 쓴다.

흐름:
    prepare : mipnerf360 이미지 + sparse/0 (또는 COLMAP 직접 실행)
       → sparse_vanilla/0
    ba      : --sba-module-name 없으면 pass-through, 있으면 methods/{name}.{name}(recon)
       → sparse/0
    eval    : vanilla vs BA 결과의 reprojection error 비교
       → output_ba/{scene}/sba_{ns}_r{N}/ba_stats.json, debug/reproj_error.png

사용법:
    python python_ba/run_sba.py                                       # pass-through
    python python_ba/run_sba.py --scene bicycle -r 2
    python python_ba/run_sba.py --sba-module-name scipy_lm            # methods/scipy_lm.py
    python python_ba/run_sba.py --run-colmap                          # COLMAP 직접 실행
    python python_ba/run_sba.py --steps ba eval                       # 이미 prepare 돼 있을 때

3DGS 까지 통합 실행하려면: python python_3dgs/run_sba_3dgs.py
"""
import argparse

from _common import (
    OUTPUT_BA_ROOT,
    add_common_args, resolve_context,
    step_prepare_images, step_prepare_vanilla_sparse, step_run_ba, step_eval_ba,
    print_header,
)


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--steps", nargs="+",
                   default=["prepare", "ba", "eval"],
                   choices=["prepare", "ba", "eval"])
    return p.parse_args()


def main():
    args = parse_args()
    ctx = resolve_context(args, output_root=OUTPUT_BA_ROOT)

    print_header(ctx, extra={"STEPS": ", ".join(args.steps)})
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

    print("\n완료.")


if __name__ == "__main__":
    main()
