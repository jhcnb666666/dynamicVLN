#!/usr/bin/env python3
"""
Trajectory Tracking Simulation Test for DiT (Diffusion Transformer)

This script:
1. Loads DiT components (traj_dit, action_encoder, action_decoder, cond_projector, pos_encoding)
   from the InternVLA-N1-DualVLN checkpoint.
2. Simulates trajectory generation with synthetic condition latents and GT trajectories.
3. Records DiT forward time for EACH diffusion timestep.
4. Computes trajectory accuracy metrics against GT.
5. Saves checkpoint (results, timings, metrics).
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

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


class SimpleFlowMatchEulerScheduler:
    """Minimal flow-match Euler discrete scheduler to avoid diffusers import issues."""

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
    """Build and return DiT trajectory generation components."""
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
    """Load DiT-related weights from safetensors checkpoint files."""
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


def compute_trajectory_accuracy(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> Dict:
    """
    Compute accuracy metrics between predicted trajectories and GT.
    pred_trajs: [num_samples, num_steps, 3] (torch tensor)
    gt_traj: [num_steps, 3] (torch tensor)
    """
    pred_mean = pred_trajs.mean(dim=0)  # [num_steps, 3]

    mse = float((pred_mean - gt_traj).pow(2).mean().item())
    rmse = float(torch.sqrt(torch.tensor(mse)).item())
    endpoint_error = float(torch.norm(pred_mean[-1] - gt_traj[-1]).item())
    ade = float(torch.norm(pred_mean - gt_traj, dim=-1).mean().item())

    sample_mses = (pred_trajs - gt_traj.unsqueeze(0)).pow(2).mean(dim=(1, 2))
    best_idx = int(sample_mses.argmin().item())
    best_mse = float(sample_mses[best_idx].item())
    best_rmse = float(torch.sqrt(torch.tensor(best_mse)).item())
    best_endpoint_error = float(torch.norm(pred_trajs[best_idx, -1] - gt_traj[-1]).item())
    best_ade = float(torch.norm(pred_trajs[best_idx] - gt_traj, dim=-1).mean().item())

    return {
        "mean_mse": mse,
        "mean_rmse": rmse,
        "mean_endpoint_error": endpoint_error,
        "mean_ade": ade,
        "best_mse": best_mse,
        "best_rmse": best_rmse,
        "best_endpoint_error": best_endpoint_error,
        "best_ade": best_ade,
        "best_sample_idx": best_idx,
    }


def create_synthetic_gt_traj(predict_step_nums: int = 32, mode: str = "straight") -> torch.Tensor:
    if mode == "straight":
        traj = torch.zeros(predict_step_nums, 3, dtype=torch.float32)
        traj[:, 0] = torch.linspace(0, 1.0, predict_step_nums)
    elif mode == "curve":
        t = torch.linspace(0, 1.0, predict_step_nums)
        traj = torch.zeros(predict_step_nums, 3, dtype=torch.float32)
        traj[:, 0] = t
        traj[:, 1] = 0.3 * torch.sin(2 * 3.14159265 * t)
    elif mode == "random_walk":
        steps = torch.randn(predict_step_nums, 3) * 0.05
        traj = torch.cumsum(steps, dim=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return traj


def save_checkpoint(
    output_dir: str,
    generated_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    timing_records: List[Dict],
    accuracy_metrics: Dict,
    extra_info: Dict,
):
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = {
        "generated_trajs": generated_trajs.cpu().tolist(),
        "gt_traj": gt_traj.cpu().tolist(),
        "timing_records": timing_records,
        "accuracy_metrics": accuracy_metrics,
        "extra_info": extra_info,
    }

    ckpt_path = os.path.join(output_dir, "traj_dit_simulation_checkpoint.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    torch.save(generated_trajs.cpu(), os.path.join(output_dir, "generated_trajs.pt"))
    torch.save(gt_traj.cpu(), os.path.join(output_dir, "gt_traj.pt"))

    times = [r["dit_forward_time_sec"] for r in timing_records]
    times_t = torch.tensor(times)
    timing_summary = {
        "num_timesteps": len(timing_records),
        "mean_dit_time_sec": float(times_t.mean().item()),
        "std_dit_time_sec": float(times_t.std().item()),
        "min_dit_time_sec": float(times_t.min().item()),
        "max_dit_time_sec": float(times_t.max().item()),
        "all_timings": timing_records,
    }
    with open(os.path.join(output_dir, "timing_summary.json"), "w") as f:
        json.dump(timing_summary, f, indent=2)

    print(f"\nCheckpoint saved to: {output_dir}")
    print(f"  - {ckpt_path}")
    print(f"  - generated_trajs.pt")
    print(f"  - gt_traj.pt")
    print(f"  - timing_summary.json")
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="DiT Trajectory Simulation Test")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/InternVLA-N1-DualVLN",
        help="Directory containing model safetensors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/traj_dit_sim_test",
        help="Directory to save simulation checkpoint",
    )
    parser.add_argument("--predict_step_nums", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--num_sample_trajs", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gt_mode", type=str, default="straight", choices=["straight", "curve", "random_walk"])
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of simulation episodes")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print("=" * 60)
    print("DiT Trajectory Tracking Simulation Test")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Params: steps={args.predict_step_nums}, infer_steps={args.num_inference_steps}, samples={args.num_sample_trajs}")

    print("\n[1/5] Building DiT components...")
    components = build_traj_dit_model(device=device, dtype=dtype)

    print("[2/5] Loading weights from checkpoint...")
    load_weights_from_checkpoint(components, args.checkpoint_dir)

    for comp in components.values():
        comp.eval()

    all_episode_results = []

    for episode_idx in range(args.num_episodes):
        print(f"\n[3/5] Running Episode {episode_idx + 1}/{args.num_episodes}...")

        torch.manual_seed(episode_idx)
        torch.cuda.manual_seed_all(episode_idx)
        traj_latents = torch.randn(1, 4, 3584, device=device, dtype=dtype) * 0.1

        with torch.no_grad():
            generated_latents, timing_records = generate_traj(
                components,
                traj_latents,
                predict_step_nums=args.predict_step_nums,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                num_sample_trajs=args.num_sample_trajs,
                detailed_timing=True,
            )

        generated_trajs = generated_latents.cpu().float()
        gt_traj = create_synthetic_gt_traj(args.predict_step_nums, mode=args.gt_mode)

        accuracy = compute_trajectory_accuracy(generated_trajs, gt_traj)

        times = [r["dit_forward_time_sec"] for r in timing_records]
        times_t = torch.tensor(times)
        print(f"  Episode {episode_idx + 1} Results:")
        print(f"    DiT forward times (ms): mean={times_t.mean()*1000:.3f}, std={times_t.std()*1000:.3f}")
        print(f"    Min={times_t.min()*1000:.3f}ms, Max={times_t.max()*1000:.3f}ms")
        print(f"    Trajectory RMSE (mean): {accuracy['mean_rmse']:.6f}")
        print(f"    Trajectory RMSE (best): {accuracy['best_rmse']:.6f}")
        print(f"    Endpoint error (mean): {accuracy['mean_endpoint_error']:.6f}")
        print(f"    Endpoint error (best): {accuracy['best_endpoint_error']:.6f}")

        episode_dir = os.path.join(args.output_dir, f"episode_{episode_idx:03d}")
        ckpt_path = save_checkpoint(
            episode_dir,
            generated_trajs,
            gt_traj,
            timing_records,
            accuracy,
            extra_info={
                "episode_idx": episode_idx,
                "predict_step_nums": args.predict_step_nums,
                "num_inference_steps": args.num_inference_steps,
                "num_sample_trajs": args.num_sample_trajs,
                "guidance_scale": args.guidance_scale,
                "gt_mode": args.gt_mode,
                "device": str(device),
                "dtype": str(dtype),
            },
        )

        all_episode_results.append({
            "episode": episode_idx,
            "checkpoint": ckpt_path,
            "mean_dit_time_ms": float(times_t.mean().item() * 1000),
            "std_dit_time_ms": float(times_t.std().item() * 1000),
            "mean_rmse": accuracy["mean_rmse"],
            "best_rmse": accuracy["best_rmse"],
            "mean_endpoint_error": accuracy["mean_endpoint_error"],
            "best_endpoint_error": accuracy["best_endpoint_error"],
        })

    print("\n[5/5] Saving aggregate results...")
    agg_path = os.path.join(args.output_dir, "aggregate_results.json")
    with open(agg_path, "w") as f:
        json.dump(all_episode_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"All checkpoints saved to: {args.output_dir}")
    print(f"Aggregate results: {agg_path}")
    print("\nAggregate Summary:")
    mean_times = torch.tensor([r["mean_dit_time_ms"] for r in all_episode_results])
    mean_rmses = torch.tensor([r["mean_rmse"] for r in all_episode_results])
    best_rmses = torch.tensor([r["best_rmse"] for r in all_episode_results])
    print(f"  Avg DiT time across episodes: {mean_times.mean():.3f} ± {mean_times.std():.3f} ms")
    print(f"  Avg mean RMSE: {mean_rmses.mean():.6f} ± {mean_rmses.std():.6f}")
    print(f"  Avg best RMSE: {best_rmses.mean():.6f} ± {best_rmses.std():.6f}")

    print("\nPer-Timestep DiT Timing Trend (last episode):")
    for i, rec in enumerate(timing_records):
        print(f"  Step {i + 1:2d} | timestep={rec['timestep']:.4f} | time={rec['dit_forward_time_sec'] * 1000:.3f} ms")


if __name__ == "__main__":
    main()
