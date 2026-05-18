import sys
import importlib.util
import numpy as np
import torch

# 直接加载 geometry_utils.py，避免触发 utils/__init__.py 里的复杂依赖
spec = importlib.util.spec_from_file_location("geometry_utils", "internnav/utils/geometry_utils.py")
geometry_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_utils)
filter_waypoints_by_cosine_distance = geometry_utils.filter_waypoints_by_cosine_distance
filter_waypoints_iterative_by_cosine_distance = geometry_utils.filter_waypoints_iterative_by_cosine_distance


def test_iterative_filter(waypoints, avg_threshold=1.0):
    """测试迭代删除逻辑并打印中间过程."""
    wp = np.asarray(waypoints)
    print("\n--- 原始 waypoints ---")
    print(wp)
    filtered, avg, remaining = filter_waypoints_iterative_by_cosine_distance(wp, avg_threshold=avg_threshold)
    print(f"\n--- 迭代删除结果 (avg_threshold={avg_threshold}) ---")
    print(f"最终 avg_sum_dist={avg:.4f}, remaining={remaining}, removed={len(wp)-remaining}")
    print("过滤后 waypoints:")
    print(filtered)


def manual_check(waypoints):
    """手动打印每个 interior waypoint 的 motion vectors 和 cosine distances."""
    wp = np.asarray(waypoints)
    N = wp.shape[0]
    motion = wp[1:] - wp[:-1]
    print("\n--- motion vectors (v[k] = wp[k+1] - wp[k]) ---")
    for k in range(len(motion)):
        print(f"  v[{k}] = {motion[k]}")

    print("\n--- interior waypoint 检查 ---")
    for i in range(1, N - 1):
        prev_vec = motion[i - 1]
        curr_vec = motion[i]
        norm_pv = np.linalg.norm(prev_vec)
        norm_cv = np.linalg.norm(curr_vec)
        norm_pv = max(norm_pv, 1e-8)
        norm_cv = max(norm_cv, 1e-8)
        sim_prev = np.dot(prev_vec, curr_vec) / (norm_pv * norm_cv)
        dist_prev = 1.0 - sim_prev

        if i < N - 2:
            next_vec = motion[i + 1]
            norm_nv = np.linalg.norm(next_vec)
            norm_nv = max(norm_nv, 1e-8)
            sim_next = np.dot(curr_vec, next_vec) / (norm_cv * norm_nv)
            dist_next = 1.0 - sim_next
        else:
            dist_next = 0.0

        total = dist_prev + dist_next
        status = "删除" if total < 0.2 else "保留"
        print(f"waypoint {i}: dist(prev,curr)={dist_prev:.4f}, dist(curr,next)={dist_next:.4f}, sum={total:.4f} -> {status}")


# ==================== NumPy 测试 ====================
print("=" * 70)
print("NumPy 测试：直线 + 一个拐角")
print("=" * 70)

waypoints_np = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 1.0],
    [4.0, 2.0],
    [4.0, 3.0],
], dtype=np.float32)

print("原始 waypoints:")
print(waypoints_np)
manual_check(waypoints_np)

filtered = filter_waypoints_by_cosine_distance(waypoints_np, threshold=0.2)
print("\n过滤后 waypoints:")
print(filtered)

test_iterative_filter(waypoints_np, avg_threshold=1.0)


# ==================== Torch 测试 ====================
print("\n" + "=" * 70)
print("Torch 测试：轻微弯曲 -> 大拐角 -> 轻微弯曲")
print("=" * 70)

waypoints_torch = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.1],
    [2.0, 0.2],
    [3.0, 2.0],
    [4.0, 4.0],
    [5.0, 4.1],
    [6.0, 4.2],
], dtype=torch.float32)

print("原始 waypoints:")
print(waypoints_torch)

filtered_torch = filter_waypoints_by_cosine_distance(waypoints_torch, threshold=0.2)
print("\n过滤后 waypoints:")
print(filtered_torch)

test_iterative_filter(waypoints_torch.numpy(), avg_threshold=1.0)


# ==================== 32 个 waypoints 测试 ====================
print("\n" + "=" * 70)
print("32 个 waypoints 测试：前 16 个近似直线，后 16 个近似直线但方向不同")
print("=" * 70)

np.random.seed(0)
waypoints_32 = []
# 前 16 个：沿 x 轴走
for i in range(16):
    waypoints_32.append([i * 1.0, np.random.uniform(0, 0.05)])
# 后 16 个：沿 y 轴走（从 (15, 0) 开始向上）
for i in range(16):
    waypoints_32.append([15.0 + np.random.uniform(0, 0.05), i * 1.0])
waypoints_32 = np.array(waypoints_32, dtype=np.float32)

print("原始 waypoints: {} 个".format(len(waypoints_32)))
filtered_32 = filter_waypoints_by_cosine_distance(waypoints_32, threshold=0.2)
print("过滤后 waypoints: {} 个".format(len(filtered_32)))
print("被删除的中间点数量:", len(waypoints_32) - len(filtered_32))

test_iterative_filter(waypoints_32, avg_threshold=1.0)
