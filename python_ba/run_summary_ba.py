#!/usr/bin/env python
"""BA 방법론별 reprojection error stats (output_ba/{scene}/sba_*_r*/ba_stats.json) 수집 → CSV.

사용법:
    python python_ba/run_summary_ba.py                                # 전부
    python python_ba/run_summary_ba.py -r 2                           # r=2 만
    python python_ba/run_summary_ba.py --scene bicycle -r 2
    python python_ba/run_summary_ba.py --scene bicycle -r 2 --out /tmp/ba.csv

출력:
    output_ba/summary/{YYYYMMDD_HHMMSS}.csv
    컬럼: scene, namespace, resolution, n_obs,
          vanilla_mean, vanilla_median, vanilla_lt1pct,
          after_mean,   after_median,   after_lt1pct,
          diff_mean,    diff_median
"""
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BA_ROOT = PROJECT_ROOT / "output_ba"
SUMMARY_DIR = OUTPUT_BA_ROOT / "summary"

_RE = re.compile(r"^sba_(?P<ns>.+?)_r(?P<res>\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="특정 씬만 필터")
    p.add_argument("-r", "--resolution", type=int, default=None, help="해상도 필터")
    p.add_argument("--out", default=None, help="CSV 경로 (기본: output_ba/summary/{ts}.csv)")
    return p.parse_args()


def collect(args):
    rows = []
    for scene_dir in sorted(OUTPUT_BA_ROOT.iterdir()) if OUTPUT_BA_ROOT.exists() else []:
        if not scene_dir.is_dir() or scene_dir.name == "summary":
            continue
        if args.scene and scene_dir.name != args.scene:
            continue
        for d in sorted(scene_dir.iterdir()):
            if not d.is_dir():
                continue
            m = _RE.match(d.name)
            if not m:
                continue
            res = int(m["res"])
            if args.resolution is not None and res != args.resolution:
                continue
            stats_path = d / "ba_stats.json"
            if not stats_path.exists():
                continue
            with open(stats_path) as f:
                s = json.load(f)
            rows.append({
                "scene": scene_dir.name,
                "namespace": m["ns"],
                "resolution": res,
                "n_obs": s["n_observations"],
                "vanilla_mean":   s["vanilla"]["mean"],
                "vanilla_median": s["vanilla"]["median"],
                "vanilla_lt1pct": s["vanilla"]["under_1px_pct"],
                "after_mean":     s["after"]["mean"],
                "after_median":   s["after"]["median"],
                "after_lt1pct":   s["after"]["under_1px_pct"],
                "diff_mean":      s["after"]["mean"] - s["vanilla"]["mean"],
                "diff_median":    s["after"]["median"] - s["vanilla"]["median"],
                "dir": f"{scene_dir.name}/{d.name}",
            })
    return rows


def main():
    args = parse_args()
    rows = collect(args)
    if not rows:
        print("수집된 결과가 없습니다. output_ba/ 아래 ba_stats.json 이 있는지 확인.")
        return
    rows.sort(key=lambda r: (r["scene"], r["resolution"], r["namespace"]))

    if args.out:
        out_path = Path(args.out)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SUMMARY_DIR / f"{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scene", "namespace", "resolution", "n_obs",
                  "vanilla_mean", "vanilla_median", "vanilla_lt1pct",
                  "after_mean", "after_median", "after_lt1pct",
                  "diff_mean", "diff_median", "dir"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # 콘솔 표
    print()
    print(f"{'scene':<10}  {'namespace':<14}  {'res':<4}  "
          f"{'van mean':>8}  {'aft mean':>8}  {'diff':>8}  "
          f"{'van med':>7}  {'aft med':>7}  {'diff':>7}  {'n_obs':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{r['scene']:<10}  {r['namespace']:<14}  {r['resolution']:<4}  "
              f"{r['vanilla_mean']:>8.4f}  {r['after_mean']:>8.4f}  {r['diff_mean']:>+8.4f}  "
              f"{r['vanilla_median']:>7.4f}  {r['after_median']:>7.4f}  {r['diff_median']:>+7.4f}  "
              f"{r['n_obs']:>8,}")

    print(f"\n✅ CSV 저장: {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
