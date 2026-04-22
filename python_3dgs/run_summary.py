#!/usr/bin/env python
"""각 방법론의 results.json 을 수집해 CSV 요약표로 저장.

디렉토리 규칙:
    output_3dgs/{scene}/vanilla_r{N}/results.json
    output_3dgs/{scene}/sba_{namespace}_r{N}/results.json

사용법:
    python python_3dgs/run_summary.py                                    # 전부 수집
    python python_3dgs/run_summary.py --resolution 2                     # r=2 만
    python python_3dgs/run_summary.py --scene bicycle --resolution 2
    python python_3dgs/run_summary.py --resolution 2 --gt vanilla        # vanilla 를 GT baseline 으로 표시
    python python_3dgs/run_summary.py --gt vanilla --scene bicycle -r 2  # GT + bicycle/r2 SBA 방법론들

출력:
    output_3dgs/summary/{YYYYMMDD_HHMMSS}.csv
"""
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output_3dgs"
SUMMARY_DIR = OUTPUT_ROOT / "summary"

# 예: vanilla_r2  /  sba_scipy_lm_r4  (상위 디렉토리가 scene)
_RE = re.compile(r"^(?P<kind>vanilla|sba)(?:_(?P<ns>.+?))?_r(?P<res>\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="특정 씬만 필터")
    p.add_argument("-r", "--resolution", type=int, default=None, help="특정 해상도만 필터")
    p.add_argument("--gt", default=None,
                   help="namespace (예: 'vanilla', 'default'). "
                        "해당 namespace 행을 CSV 맨 위에 'GT' 표시와 함께 배치")
    p.add_argument("--iteration", default=None,
                   help="results.json 의 특정 iteration 키 (기본: 마지막 키)")
    p.add_argument("--out", default=None,
                   help="CSV 경로 (기본: output_3dgs/summary/{YYYYMMDD_HHMMSS}.csv)")
    return p.parse_args()


def parse_dir_name(name: str, scene: str):
    m = _RE.match(name)
    if not m:
        return None
    return {
        "scene": scene,
        "kind": m["kind"],
        "namespace": m["ns"] if m["kind"] == "sba" else "vanilla",
        "resolution": int(m["res"]),
    }


def pick_iteration(results: dict, iteration: Optional[str]) -> Optional[str]:
    if not results:
        return None
    if iteration is not None and iteration in results:
        return iteration
    # 숫자 키로 정렬 가능하면 최대치, 아니면 마지막 키
    try:
        return max(results.keys(), key=lambda k: int(re.sub(r"\D", "", k) or 0))
    except Exception:
        return list(results.keys())[-1]


def collect(args):
    rows = []
    for scene_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name == "summary":
            continue
        if args.scene and scene_dir.name != args.scene:
            continue
        for d in sorted(scene_dir.iterdir()):
            if not d.is_dir():
                continue
            info = parse_dir_name(d.name, scene=scene_dir.name)
            if info is None:
                continue
            if args.resolution is not None and info["resolution"] != args.resolution:
                continue
            results_path = d / "results.json"
            if not results_path.exists():
                continue
            with open(results_path) as f:
                results = json.load(f)
            it = pick_iteration(results, args.iteration)
            if it is None:
                continue
            m = results[it]
            rows.append({
                "dir": f"{scene_dir.name}/{d.name}",
                "scene": info["scene"],
                "kind": info["kind"],
                "namespace": info["namespace"],
                "resolution": info["resolution"],
                "iteration": it,
                "PSNR": m.get("PSNR"),
                "SSIM": m.get("SSIM"),
                "LPIPS": m.get("LPIPS"),
                "is_gt": False,
            })
    return rows


def main():
    args = parse_args()
    rows = collect(args)

    if not rows:
        print("수집된 결과가 없습니다. output_3dgs/ 아래 results.json 이 있는지 확인하세요.")
        return

    # --gt 지정 시 해당 namespace 를 GT 로 마킹 후 상단 정렬
    if args.gt is not None:
        found = False
        for r in rows:
            if r["namespace"] == args.gt:
                r["is_gt"] = True
                found = True
        if not found:
            print(f"  경고: --gt '{args.gt}' 에 해당하는 결과가 없습니다.")
        rows.sort(key=lambda r: (not r["is_gt"], r["scene"], r["resolution"], r["namespace"]))
    else:
        rows.sort(key=lambda r: (r["scene"], r["resolution"], r["kind"] != "vanilla", r["namespace"]))

    # CSV 경로
    if args.out:
        out_path = Path(args.out)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SUMMARY_DIR / f"{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["GT", "scene", "namespace", "resolution", "iteration", "PSNR", "SSIM", "LPIPS", "dir"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "GT": "GT" if r["is_gt"] else "",
                "scene": r["scene"],
                "namespace": r["namespace"],
                "resolution": r["resolution"],
                "iteration": r["iteration"],
                "PSNR": f"{r['PSNR']:.4f}" if r["PSNR"] is not None else "",
                "SSIM": f"{r['SSIM']:.4f}" if r["SSIM"] is not None else "",
                "LPIPS": f"{r['LPIPS']:.4f}" if r["LPIPS"] is not None else "",
                "dir": r["dir"],
            })

    # 콘솔 표
    widths = {"GT": 3, "scene": 10, "namespace": 14, "res": 4,
              "iter": 12, "PSNR": 8, "SSIM": 7, "LPIPS": 7}
    print()
    print(f"{'GT':<{widths['GT']}}  {'scene':<{widths['scene']}}  {'namespace':<{widths['namespace']}}  "
          f"{'res':<{widths['res']}}  {'iter':<{widths['iter']}}  "
          f"{'PSNR':>{widths['PSNR']}}  {'SSIM':>{widths['SSIM']}}  {'LPIPS':>{widths['LPIPS']}}")
    print("-" * 82)
    for r in rows:
        gt = "GT" if r["is_gt"] else ""
        print(f"{gt:<{widths['GT']}}  {r['scene']:<{widths['scene']}}  "
              f"{r['namespace']:<{widths['namespace']}}  "
              f"{r['resolution']:<{widths['res']}}  {r['iteration']:<{widths['iter']}}  "
              f"{r['PSNR']:>{widths['PSNR']}.4f}  "
              f"{r['SSIM']:>{widths['SSIM']}.4f}  {r['LPIPS']:>{widths['LPIPS']}.4f}")

    print(f"\n✅ CSV 저장: {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
