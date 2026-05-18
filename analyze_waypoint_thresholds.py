#!/usr/bin/env python3
"""Analyze waypoint statistics on validation set to recommend threshold combinations.

All filtering uses iterative removal: repeatedly delete the interior waypoint with
the smallest cosine-distance sum until the average exceeds avg_threshold.
"""

import glob
import json
import os
import sys
import importlib.util

import numpy as np

# Direct-load geometry_utils to avoid complex dependency chain in utils/__init__.py
spec = importlib.util.spec_from_file_location(
    "geometry_utils", "/home/ubuntu/project/InternNav-lora/internnav/utils/geometry_utils.py"
)
geometry_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_utils)
filter_waypoints_iterative_by_cosine_distance = geometry_utils.filter_waypoints_iterative_by_cosine_distance


def load_waypoints_from_pose_dir(pose_dir: str):
    """Load (x, z) waypoints from a directory of 4x4 pose .npy files."""
    files = sorted(glob.glob(os.path.join(pose_dir, "*.npy")))
    if len(files) < 3:
        return None
    waypoints = []
    for pf in files:
        mat = np.load(pf)
        if mat.shape == (4, 4):
            waypoints.append([mat[0, 3], mat[2, 3]])
        else:
            waypoints.append([0.0, 0.0])
    return np.array(waypoints, dtype=np.float32)


def analyze_dataset(data_root: str, max_episodes: int = None):
    annotation_path = os.path.join(data_root, "annotations.json")
    with open(annotation_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    results = []
    for ep in episodes[:max_episodes] if max_episodes else episodes:
        video_rel = ep.get("video", "")
        pose_dir = os.path.join(data_root, video_rel, "pose")
        wps = load_waypoints_from_pose_dir(pose_dir)
        if wps is None or len(wps) < 4:
            continue

        # Test iterative filtering with multiple avg_thresholds
        iterative_stats = {}
        for avg_thr in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            filtered, final_avg, remaining = filter_waypoints_iterative_by_cosine_distance(
                wps, avg_threshold=avg_thr
            )
            iterative_stats[avg_thr] = {
                "remaining": remaining,
                "removed": len(wps) - remaining,
                "ratio": remaining / len(wps),
                "final_avg": final_avg,
            }

        results.append({
            "episode": ep.get("id", video_rel),
            "original_len": len(wps),
            "iterative": iterative_stats,
        })

    return results


def print_distribution(arr, name):
    arr = np.array(arr)
    print(f"\n{name} distribution:")
    print(f"  mean={arr.mean():.4f}, std={arr.std():.4f}, median={np.median(arr):.4f}")
    print(f"  min={arr.min():.4f}, max={arr.max():.4f}")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    vals = np.percentile(arr, percentiles)
    for p, v in zip(percentiles, vals):
        print(f"  p{p}={v:.4f}")


def main():
    val_root = "/home/ubuntu/dataset/VLN-Trajectory-Data/R2R/offline_r2r_val_unseen"
    print("Analyzing validation set waypoint statistics (iterative-only)...")
    results = analyze_dataset(val_root)
    print(f"Total episodes analyzed: {len(results)}")

    # 1. Baseline avg_sum_dist without any filtering (approximate via high threshold)
    raw_avgs = [r["iterative"][3.0]["final_avg"] for r in results]
    print_distribution(raw_avgs, "avg_sum_dist (raw, avg_threshold=3.0, nearly unfiltered)")

    # 2. Iterative filtering impact across thresholds
    print("\n" + "=" * 70)
    print("Iterative filtering impact (filter_waypoints_iterative_by_cosine_distance)")
    print("=" * 70)
    for avg_thr in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        ratios = [r["iterative"][avg_thr]["ratio"] for r in results]
        removed = [r["iterative"][avg_thr]["removed"] for r in results]
        final_avgs = [r["iterative"][avg_thr]["final_avg"] for r in results]
        print(f"\navg_threshold={avg_thr}:")
        print(f"  remaining ratio: mean={np.mean(ratios):.3f}, median={np.median(ratios):.3f}")
        print(f"  removed count:   mean={np.mean(removed):.1f}, median={np.median(removed):.1f}")
        print(f"  final avg_sum_dist: mean={np.mean(final_avgs):.4f}")

    # 3. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLD COMBINATIONS FOR EXPERIMENTS")
    print("=" * 70)

    combos = [
        {
            "name": "Ultra-Aggressive (极度激进，大量删减)",
            "change_ratio": 0.05,
            "iterative_avg_threshold_stable": 1.50,
            "iterative_avg_threshold_unstable": 0.75,
            "desc": "适合长直走廊、开阔空间；可能丢失必要转弯",
        },
        {
            "name": "Aggressive (激进)",
            "change_ratio": 0.04,
            "iterative_avg_threshold_stable": 1.25,
            "iterative_avg_threshold_unstable": 0.60,
            "desc": "适合简单室内路径，减少冗余停顿",
        },
        {
            "name": "Balanced (平衡，当前代码默认值)",
            "change_ratio": 0.02,
            "iterative_avg_threshold_stable": 1.00,
            "iterative_avg_threshold_unstable": 0.50,
            "desc": "大多数室内场景通用",
        },
        {
            "name": "Conservative (保守)",
            "change_ratio": 0.015,
            "iterative_avg_threshold_stable": 0.75,
            "iterative_avg_threshold_unstable": 0.35,
            "desc": "复杂环境/多障碍物，保留更多中间点",
        },
        {
            "name": "Ultra-Conservative (极度保守)",
            "change_ratio": 0.01,
            "iterative_avg_threshold_stable": 0.50,
            "iterative_avg_threshold_unstable": 0.25,
            "desc": "狭窄通道或高精度避障场景",
        },
    ]

    for c in combos:
        print(f"\n{c['name']}")
        print(f"  change_ratio threshold:                {c['change_ratio']}")
        print(f"  iterative_avg_threshold (stable) :     {c['iterative_avg_threshold_stable']}")
        print(f"  iterative_avg_threshold (unstable):    {c['iterative_avg_threshold_unstable']}")
        print(f"  适用场景: {c['desc']}")

    # Save raw stats for further analysis
    out_path = "/tmp/waypoint_threshold_analysis.json"
    serializable = []
    for r in results:
        sr = {
            "episode": r["episode"],
            "original_len": int(r["original_len"]),
            "iterative": {str(k): {sk: float(sv) for sk, sv in v.items()} for k, v in r["iterative"].items()},
        }
        serializable.append(sr)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nDetailed stats saved to: {out_path}")


if __name__ == "__main__":
    main()
