#!/usr/bin/env python3
"""
对比测试：固定 32 步 DiT vs 动态调整步数 DiT

按照 internvla_n1_policy.py 中 s1_step_latent 的真实策略：
- 生成轨迹后，用 traj_to_actions 计算 remaining_count
- 根据 change_ratio 和 avg_sum_dist 动态调整下一步 predict_step_nums
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file

import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import (
    NextDiTCrossAttn,
    NextDiTCrossAttnConfig,
)
from internnav.model.basemodel.internvla_n1.internvla_n1_arch import (
    SinusoidalPositionalEncoding,
)
LatentEmbSize = 768


def filter_waypoints_iterative_by_cosine_distance_torch(waypoints, avg_threshold=1.0):
    """Torch-only version to avoid numpy compatibility issues."""
    if isinstance(waypoints, np.ndarray):
        wp = torch.from_numpy(waypoints).float()
    elif isinstance(waypoints, torch.Tensor):
        wp = waypoints.float()
    else:
        wp = torch.tensor(waypoints, dtype=torch.float32)
    N = wp.shape[0]
    if N < 3:
        return wp, 0.0, N

    def _compute_sums(current_wp):
        motion = current_wp[1:] - current_wp[:-1]
        sums = []
        for i in range(1, current_wp.shape[0] - 1):
            prev_vec = motion[i - 1]
            curr_vec = motion[i]
            norm_pv = torch.norm(prev_vec)
            norm_cv = torch.norm(curr_vec)
            norm_pv = max(norm_pv.item(), 1e-8)
            norm_cv = max(norm_cv.item(), 1e-8)
            sim_prev = torch.dot(prev_vec, curr_vec).item() / (norm_pv * norm_cv)
            dist_prev = 1.0 - sim_prev
            if i < current_wp.shape[0] - 2:
                next_vec = motion[i + 1]
                norm_nv = torch.norm(next_vec)
                norm_nv = max(norm_nv.item(), 1e-8)
                sim_next = torch.dot(curr_vec, next_vec).item() / (norm_cv * norm_nv)
                dist_next = 1.0 - sim_next
            else:
                dist_next = 0.0
            sums.append(dist_prev + dist_next)
        return sums

    current_wp = wp.clone()
    pre_sums = _compute_sums(current_wp)
    if pre_sums:
        keep = [True] * current_wp.shape[0]
        for i, s in enumerate(pre_sums):
            if s < 0.1:
                keep[i + 1] = False
        keep_indices = [i for i, k in enumerate(keep) if k]
        if len(keep_indices) >= 2:
            current_wp = current_wp[keep_indices]

    while current_wp.shape[0] > 3:
        sums = _compute_sums(current_wp)
        if not sums:
            break
        avg = sum(sums) / len(sums)
        if avg > avg_threshold:
            break
        min_local_idx = int(np.argmin(sums)) + 1
        current_wp = torch.cat([current_wp[:min_local_idx], current_wp[min_local_idx + 1:]])

    final_sums = _compute_sums(current_wp)
    final_avg = sum(final_sums) / len(final_sums) if final_sums else 0.0
    return current_wp, final_avg, current_wp.shape[0]


def traj_to_actions(dp_actions, use_discrate_action=True, change_ratio=None):
    """Simplified version using torch-only waypoint filtering."""
    def reconstruct_xy_from_delta(delta_xyt):
        if isinstance(delta_xyt, np.ndarray):
            delta_xyt = torch.from_numpy(delta_xyt).float()
        start_xy = torch.zeros(len(delta_xyt), 2)
        delta_xy = delta_xyt[:, :, :2]
        cumsum_xy = torch.cumsum(delta_xy, dim=1)
        B, T, _ = delta_xy.shape
        xy = torch.zeros(B, T + 1, 2)
        xy[:, 0] = start_xy
        xy[:, 1:] = start_xy.unsqueeze(1) + cumsum_xy
        return xy

    # unnormalize
    if isinstance(dp_actions, torch.Tensor):
        dp_actions = dp_actions.clone()
        dp_actions[:, :, :2] /= 4.0
    else:
        dp_actions = torch.tensor(dp_actions, dtype=torch.float32)
        dp_actions[:, :, :2] /= 4.0

    all_trajectory = reconstruct_xy_from_delta(dp_actions)
    trajectory = all_trajectory.mean(dim=0)

    if change_ratio is not None and change_ratio <= 0.02:
        avg_threshold = 1.0
    else:
        avg_threshold = 0.5

    filtered_traj, avg_sum_dist, remaining_count = filter_waypoints_iterative_by_cosine_distance_torch(
        trajectory, avg_threshold=avg_threshold
    )
    trajectory = filtered_traj

    if use_discrate_action:
        actions = []
    else:
        actions = trajectory

    return actions, avg_sum_dist, remaining_count


class SimpleFlowMatchEulerScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None
        self.sigmas = None
        self._step_index = 0

    def set_timesteps(self, num_inference_steps: int, sigmas=None, device=None):
        if sigmas is None:
            sigmas = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]
        else:
            sigmas = [float(s) for s in sigmas]
        sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=device)
        timesteps = sigmas_t * self.num_train_timesteps
        self.timesteps = timesteps
        self.sigmas = torch.cat([sigmas_t, torch.zeros(1, device=sigmas_t.device)])
        self._step_index = 0

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample):
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        prev_sample = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return type("SchedulerOutput", (), {"prev_sample": prev_sample})()


def build_traj_dit_model(device="cuda", dtype=torch.bfloat16):
    dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=LatentEmbSize))
    action_encoder = nn.Linear(3, 384, bias=True)
    pos_encoding = SinusoidalPositionalEncoding(384)
    action_decoder = nn.Linear(384, 3, bias=True)
    cond_projector = nn.Sequential(
        nn.Linear(3584, LatentEmbSize),
        nn.GELU(approximate="tanh"),
        nn.Linear(LatentEmbSize, LatentEmbSize),
    )
    dit = dit.to(device=device, dtype=dtype)
    action_encoder = action_encoder.to(device=device, dtype=dtype)
    pos_encoding = pos_encoding.to(device=device, dtype=dtype)
    action_decoder = action_decoder.to(device=device, dtype=dtype)
    cond_projector = cond_projector.to(device=device, dtype=dtype)
    return {
        "traj_dit": dit,
        "action_encoder": action_encoder,
        "pos_encoding": pos_encoding,
        "action_decoder": action_decoder,
        "cond_projector": cond_projector,
    }


def load_weights_from_checkpoint(components: Dict, checkpoint_dir: str):
    checkpoint_dir = Path(checkpoint_dir)
    weight_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {checkpoint_dir}")
    all_state = {}
    for wf in weight_files:
        partial = load_file(wf, device="cpu")
        all_state.update(partial)

    comp_keys = {
        "traj_dit": [],
        "action_encoder": [],
        "action_decoder": [],
        "cond_projector": [],
        "pos_encoding": [],
    }
    for k in all_state.keys():
        if k.startswith("model.traj_dit."):
            comp_keys["traj_dit"].append(k)
        elif k.startswith("model.action_encoder."):
            comp_keys["action_encoder"].append(k)
        elif k.startswith("model.action_decoder."):
            comp_keys["action_decoder"].append(k)
        elif k.startswith("model.cond_projector."):
            comp_keys["cond_projector"].append(k)
        elif k.startswith("model.pos_encoding."):
            comp_keys["pos_encoding"].append(k)

    for comp_name, keys in comp_keys.items():
        if not keys:
            print(f"Warning: no weights found for {comp_name}")
            continue
        state_dict = {}
        for k in keys:
            new_key = k.replace(f"model.{comp_name}.", "")
            state_dict[new_key] = all_state[k]
        missing, unexpected = components[comp_name].load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load] {comp_name} missing keys: {missing}")
        if unexpected:
            print(f"[load] {comp_name} unexpected keys: {unexpected}")
        print(f"Loaded {comp_name} with {len(state_dict)} tensors.")
    return components


def generate_traj(
    components: Dict,
    traj_latents: torch.Tensor,
    predict_step_nums: int = 32,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 10,
    num_sample_trajs: int = 32,
    detailed_timing: bool = True,
):
    scheduler = SimpleFlowMatchEulerScheduler()
    device = traj_latents.device
    dtype = traj_latents.dtype

    cond_projector = components["cond_projector"]
    action_encoder = components["action_encoder"]
    pos_encoding = components["pos_encoding"]
    traj_dit = components["traj_dit"]
    action_decoder = components["action_decoder"]

    traj_latents = cond_projector(traj_latents)
    hidden_states = traj_latents
    hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
    hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
    batch_size = traj_latents.shape[0]
    latent_size = predict_step_nums
    latent_channels = 3

    latents = randn_tensor(
        shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
        generator=None,
        device=device,
        dtype=dtype,
    )

    sigmas = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]
    scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device)
    hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

    timestep_records = []

    for t in scheduler.timesteps:
        latent_features = action_encoder(latents)
        pos_ids = (
            torch.arange(latent_features.shape[1])
            .reshape(1, -1)
            .repeat(batch_size, 1)
            .to(latent_features.device)
        )
        pos_embed = pos_encoding(pos_ids)
        latent_features += pos_embed
        latent_model_input = latent_features.repeat(2, 1, 1)
        if hasattr(scheduler, "scale_model_input"):
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        if detailed_timing:
            torch.cuda.synchronize(device)
            t_start = time.perf_counter()

        noise_pred = traj_dit(
            x=latent_model_input,
            timestep=t.unsqueeze(0)
            .expand(latent_model_input.shape[0])
            .to(latent_model_input.device, torch.long),
            z_latents=hidden_states_input,
        )

        if detailed_timing:
            torch.cuda.synchronize(device)
            t_elapsed = time.perf_counter() - t_start
            timestep_records.append(
                {
                    "timestep": float(t.item() if hasattr(t, "item") else t),
                    "dit_forward_time_sec": float(t_elapsed),
                    "batch_size": latent_model_input.shape[0],
                    "seq_len": latent_model_input.shape[1],
                }
            )

        noise_pred = action_decoder(noise_pred)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents, timestep_records


def simulate_episode_fixed32(components, traj_latents, num_steps=10, **gen_kwargs):
    """Fixed 32 mode: always predict_step_nums=32"""
    records = []
    total_dit_time = 0.0
    for step_idx in range(num_steps):
        with torch.no_grad():
            latents, timing = generate_traj(
                components,
                traj_latents,
                predict_step_nums=32,
                detailed_timing=True,
                **gen_kwargs,
            )
        dit_time = sum(r["dit_forward_time_sec"] for r in timing)
        total_dit_time += dit_time
        records.append({
            "step": step_idx,
            "predict_step_nums": 32,
            "dit_time_ms": dit_time * 1000,
            "per_timestep_ms": [r["dit_forward_time_sec"] * 1000 for r in timing],
        })
    return records, total_dit_time


def simulate_episode_dynamic(components, traj_latents, change_ratios, num_steps=10, **gen_kwargs):
    """
    Dynamic mode: follow the real policy logic.
    change_ratios: list of pre-defined change_ratio for each step.
    """
    records = []
    total_dit_time = 0.0
    last_predict_step_nums = 32

    for step_idx in range(num_steps):
        predict_step_nums = last_predict_step_nums
        with torch.no_grad():
            latents, timing = generate_traj(
                components,
                traj_latents,
                predict_step_nums=predict_step_nums,
                detailed_timing=True,
                **gen_kwargs,
            )
        dit_time = sum(r["dit_forward_time_sec"] for r in timing)
        total_dit_time += dit_time

        # Run traj_to_actions to get remaining_count (like real policy)
        # dp_actions shape: [num_sample_trajs, predict_step_nums, 3]
        # traj_to_actions expects input on CPU
        dp_actions = latents.clone()
        try:
            _, avg_sum_dist, remaining_count = traj_to_actions(
                dp_actions.float().cpu().numpy(),
                use_discrate_action=False,
                change_ratio=change_ratios[step_idx],
            )
        except Exception as e:
            print(f"[Step {step_idx}] traj_to_actions failed: {e}, fallback to 32")
            avg_sum_dist = 0.0
            remaining_count = predict_step_nums

        # Dynamic adjustment logic (exactly from s1_step_latent)
        if change_ratios[step_idx] > 0.02 or avg_sum_dist > 1:
            last_predict_step_nums = 32
        else:
            last_predict_step_nums = max(remaining_count, 4)

        records.append({
            "step": step_idx,
            "predict_step_nums": predict_step_nums,
            "dit_time_ms": dit_time * 1000,
            "per_timestep_ms": [r["dit_forward_time_sec"] * 1000 for r in timing],
            "change_ratio": change_ratios[step_idx],
            "avg_sum_dist": avg_sum_dist,
            "remaining_count": remaining_count,
            "next_predict_step_nums": last_predict_step_nums,
        })
    return records, total_dit_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument("--output_dir", type=str, default="checkpoints/traj_dit_dynamic_vs_fixed")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_steps_per_episode", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--num_sample_trajs", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print("=" * 70)
    print("DiT Dynamic vs Fixed32 Comparison Test")
    print("=" * 70)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Episodes: {args.num_episodes}, Steps per episode: {args.num_steps_per_episode}")

    print("\n[1/4] Building DiT components...")
    components = build_traj_dit_model(device=device, dtype=dtype)

    print("[2/4] Loading weights...")
    load_weights_from_checkpoint(components, args.checkpoint_dir)
    for comp in components.values():
        comp.eval()

    gen_kwargs = {
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_sample_trajs": args.num_sample_trajs,
    }

    all_results = []

    for episode_idx in range(args.num_episodes):
        print(f"\n[3/4] Episode {episode_idx + 1}/{args.num_episodes}")
        torch.manual_seed(episode_idx)
        torch.cuda.manual_seed_all(episode_idx)
        np.random.seed(episode_idx)

        traj_latents = torch.randn(1, 4, 3584, device=device, dtype=dtype) * 0.1

        # Simulate a realistic change_ratio sequence:
        # Steps 0-2: stable (low change_ratio) -> dynamic will prune
        # Steps 3-4: sudden change (high change_ratio) -> dynamic resets to 32
        # Steps 5-7: stable again -> dynamic prunes again
        # Steps 8-9: moderate change -> dynamic keeps moderate
        change_ratios = [0.01, 0.005, 0.008, 0.05, 0.03, 0.005, 0.01, 0.007, 0.025, 0.015]
        change_ratios = change_ratios[: args.num_steps_per_episode]

        print("  Running FIXED-32 mode...")
        fixed_records, fixed_total = simulate_episode_fixed32(
            components, traj_latents, num_steps=args.num_steps_per_episode, **gen_kwargs
        )

        print("  Running DYNAMIC mode...")
        dynamic_records, dynamic_total = simulate_episode_dynamic(
            components, traj_latents, change_ratios, num_steps=args.num_steps_per_episode, **gen_kwargs
        )

        speedup = fixed_total / dynamic_total if dynamic_total > 0 else float('inf')
        print(f"\n  Episode {episode_idx + 1} Summary:")
        print(f"    Fixed-32 total DiT time: {fixed_total * 1000:.2f} ms")
        print(f"    Dynamic  total DiT time: {dynamic_total * 1000:.2f} ms")
        print(f"    Speedup (Fixed/Dynamic): {speedup:.2f}x")

        print("\n    Dynamic step-by-step:")
        for rec in dynamic_records:
            print(f"      Step {rec['step']:2d}: predict={rec['predict_step_nums']:2d}, "
                  f"dit_time={rec['dit_time_ms']:.2f}ms, change_ratio={rec['change_ratio']:.3f}, "
                  f"remaining={rec['remaining_count']}, next_predict={rec['next_predict_step_nums']}")

        all_results.append({
            "episode": episode_idx,
            "fixed_total_ms": fixed_total * 1000,
            "dynamic_total_ms": dynamic_total * 1000,
            "speedup": speedup,
            "fixed_records": fixed_records,
            "dynamic_records": dynamic_records,
        })

    print("\n[4/4] Saving results...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate
    fixed_totals = [r["fixed_total_ms"] for r in all_results]
    dynamic_totals = [r["dynamic_total_ms"] for r in all_results]
    speedups = [r["speedup"] for r in all_results]

    aggregate = {
        "fixed_avg_ms": float(np.mean(fixed_totals)),
        "fixed_std_ms": float(np.std(fixed_totals)),
        "dynamic_avg_ms": float(np.mean(dynamic_totals)),
        "dynamic_std_ms": float(np.std(dynamic_totals)),
        "speedup_avg": float(np.mean(speedups)),
        "speedup_std": float(np.std(speedups)),
        "per_episode": all_results,
    }

    agg_path = os.path.join(args.output_dir, "aggregate_comparison.json")
    with open(agg_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Fixed-32  avg total DiT time: {aggregate['fixed_avg_ms']:.2f} ± {aggregate['fixed_std_ms']:.2f} ms")
    print(f"Dynamic   avg total DiT time: {aggregate['dynamic_avg_ms']:.2f} ± {aggregate['dynamic_std_ms']:.2f} ms")
    print(f"Average speedup (Fixed/Dynamic): {aggregate['speedup_avg']:.2f}x")
    print(f"\nResults saved to: {agg_path}")


if __name__ == "__main__":
    main()
