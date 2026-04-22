#!/usr/bin/env python
"""3DGS Vanilla 파이프라인 (터미널 실행용).

사용법:
    python python_3dgs/run_vanilla.py                       # 전체 파이프라인
    python python_3dgs/run_vanilla.py --steps train         # 학습만
    python python_3dgs/run_vanilla.py --scene garden -r 4   # garden, 1/4 해상도
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_REPO = PROJECT_ROOT / "repos" / "gaussian-splatting"
DATASETS_DIR = PROJECT_ROOT / "datasets" / "mipnerf360"
OUTPUT_ROOT = PROJECT_ROOT / "output_3dgs"


def run(cmd, cwd):
    print(f"\n$ cd {cwd}")
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="bicycle",
                   choices=["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"])
    p.add_argument("-r", "--resolution", type=int, default=2,
                   help="images_N/ 사용 (1=원본, 2=1/2, 4=1/4, 8=1/8)")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--densify_grad_threshold", type=float, default=0.001)
    p.add_argument("--data_device", default="cuda",
                   help="이미지 텐서 상주 디바이스 (기본 cuda)")
    p.add_argument("--steps", nargs="+", default=["train", "render", "metrics"],
                   choices=["train", "render", "metrics"])
    p.add_argument("--output", default=None,
                   help="결과 저장 디렉토리 (기본: output_3dgs/{scene}/vanilla_r{resolution})")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = DATASETS_DIR / args.scene
    sparse_dir = dataset_dir / "sparse" / "0"
    output_dir = Path(args.output) if args.output else \
        OUTPUT_ROOT / args.scene / f"vanilla_r{args.resolution}"

    img_dir = dataset_dir / ("images" if args.resolution == 1 else f"images_{args.resolution}")

    print("=" * 60)
    print(f"  SCENE       : {args.scene}")
    print(f"  RESOLUTION  : 1/{args.resolution}  -> {img_dir.name}/")
    print(f"  DATASET_DIR : {dataset_dir} ({'OK' if dataset_dir.is_dir() else 'NOT FOUND'})")
    print(f"  SPARSE_DIR  : {sparse_dir} ({'OK' if sparse_dir.is_dir() else 'NOT FOUND'})")
    print(f"  IMG_DIR     : {img_dir} ({'OK' if img_dir.is_dir() else 'NOT FOUND'})")
    print(f"  OUTPUT_DIR  : {output_dir}")
    print(f"  DATA_DEVICE : {args.data_device}")
    print(f"  STEPS       : {', '.join(args.steps)}")
    print("=" * 60)

    assert sparse_dir.is_dir(), f"sparse/0 not found at {sparse_dir}"
    assert img_dir.is_dir(), f"image dir not found at {img_dir}"

    output_dir.mkdir(parents=True, exist_ok=True)

    if "train" in args.steps:
        run([
            sys.executable, "train.py",
            "-s", str(dataset_dir),
            "-m", str(output_dir),
            "--iterations", str(args.iterations),
            "--densify_grad_threshold", str(args.densify_grad_threshold),
            "--resolution", str(args.resolution),
            "--data_device", args.data_device,
            "--eval",
            "--test_iterations", "7000", "15000", str(args.iterations),
            "--save_iterations", "7000", "15000", str(args.iterations),
        ], cwd=GS_REPO)

    if "render" in args.steps:
        run([
            sys.executable, "render.py",
            "-m", str(output_dir),
            "--iteration", str(args.iterations),
            "--data_device", args.data_device,
        ], cwd=GS_REPO)

    if "metrics" in args.steps:
        run([sys.executable, "metrics.py", "-m", str(output_dir)], cwd=GS_REPO)

        results_path = output_dir / "results.json"
        if results_path.exists():
            print("\n=== 평가 결과 ===")
            with open(results_path) as f:
                results = json.load(f)
            for iteration, metrics in results.items():
                print(f"\n[{iteration}]")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")

    print("\n완료.")


if __name__ == "__main__":
    main()
