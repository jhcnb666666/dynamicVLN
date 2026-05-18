"""FastVid core compression logic adapted for InternNav / Qwen2.5-VL.

This module ports the training-free FastVid video token compressor from
FastVid-StreamVLN-Comp.  It operates on per-frame patch embeddings (after the
vision tower) and reduces the number of tokens that reach the LLM.

New interface for Qwen2.5-VL:
  compress_frames_by_grid(image_embeds, image_grid_thw, config)
    -> (compressed_embeds, compressed_sizes)

Original StreamVLN interface is also kept for compatibility:
  apply_fastvid_compression(image_features, memory_features, config)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .feature_flags import FastVidConfig


# ---------------------------------------------------------------------------
# Token scoring (inlined from visionzip to avoid extra dependency)
# ---------------------------------------------------------------------------


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() <= 1:
        return scores.float()
    s_min = scores.min()
    s_max = scores.max()
    return (scores.float() - s_min.float()) / (s_max.float() - s_min.float() + 1e-6)


def _score_l2(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=tokens.device)
    return torch.norm(tokens.float(), dim=-1)


def _score_attn_proxy(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=tokens.device)
    normalized = F.normalize(tokens.float(), dim=-1)
    query = normalized.mean(dim=0, keepdim=True)
    attn = torch.matmul(normalized, query.transpose(0, 1)).squeeze(-1)
    return _normalize_scores(attn)


def _token_scores(tokens: torch.Tensor, score_type: str) -> torch.Tensor:
    score_key = str(score_type).strip().lower()
    if score_key in {"l2", "norm", "magnitude"}:
        return _score_l2(tokens)
    return _score_attn_proxy(tokens)


# ---------------------------------------------------------------------------
# Frame-level scoring & segmentation
# ---------------------------------------------------------------------------


def _frame_token_scores(frame_tokens: torch.Tensor, score_type: str) -> torch.Tensor:
    if frame_tokens is None or not isinstance(frame_tokens, torch.Tensor) or frame_tokens.ndim != 2:
        return torch.empty((0,), dtype=torch.float32, device=frame_tokens.device if isinstance(frame_tokens, torch.Tensor) else None)
    scores = _token_scores(frame_tokens, score_type=score_type)
    return _normalize_scores(scores)


def _frame_global_features(frame_sequence: torch.Tensor, score_type: str) -> torch.Tensor:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return torch.empty((0, 0), dtype=torch.float32, device=frame_sequence.device if isinstance(frame_sequence, torch.Tensor) else None)

    globals_per_frame: List[torch.Tensor] = []
    for frame in frame_sequence:
        if frame.numel() == 0:
            globals_per_frame.append(torch.zeros((frame_sequence.shape[-1],), dtype=torch.float32, device=frame_sequence.device))
            continue
        scores = _frame_token_scores(frame, score_type=score_type)
        weights = torch.softmax(scores, dim=0).unsqueeze(-1)
        global_feature = (frame.float() * weights).sum(dim=0)
        globals_per_frame.append(global_feature)
    return torch.stack(globals_per_frame, dim=0)


def dynamic_segmentation(frame_global_features: torch.Tensor, dyseg_c: int, dyseg_tau: float) -> List[int]:
    num_frames = int(frame_global_features.shape[0])
    if num_frames <= 1:
        return [num_frames]

    normed = F.normalize(frame_global_features.float(), dim=-1)
    similarity = (normed[:-1] * normed[1:]).sum(dim=-1)

    k = min(max(1, int(dyseg_c) - 1), int(similarity.shape[0]))
    cut_indices_topk = torch.topk(similarity, k=k, largest=False).indices
    cut_indices_cos = torch.nonzero(similarity < float(dyseg_tau), as_tuple=False).squeeze(1)
    cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_cos], dim=0)).sort().values

    padded = F.pad(cut_indices, (1, 1), value=-1)
    padded[-1] = num_frames - 1
    return padded.diff().tolist()


# ---------------------------------------------------------------------------
# Token selection & merging
# ---------------------------------------------------------------------------


def attention_token_selection(
    frame_features: torch.Tensor,
    frame_scores: torch.Tensor,
    salient_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_frames, tokens_per_frame, hidden_dim = frame_features.shape
    device = frame_features.device

    salient_num = max(0, min(int(salient_num), tokens_per_frame))
    if salient_num <= 0:
        empty_tokens = frame_features.new_empty((num_frames, 0, hidden_dim))
        empty_indices = torch.empty((num_frames, 0), dtype=torch.long, device=device)
        all_indices = torch.arange(tokens_per_frame, device=device, dtype=torch.long).unsqueeze(0).expand(num_frames, -1)
        return empty_tokens, frame_features, empty_indices, all_indices

    salient_indices = torch.topk(frame_scores, k=salient_num, dim=1).indices
    batchframe_indices = torch.arange(num_frames, device=device).unsqueeze(1).expand(-1, salient_num)
    salient_tokens = frame_features[batchframe_indices, salient_indices]

    filtered_num = max(0, tokens_per_frame - salient_num)
    if filtered_num <= 0:
        empty_tokens = frame_features.new_empty((num_frames, 0, hidden_dim))
        empty_indices = torch.empty((num_frames, 0), dtype=torch.long, device=device)
        return salient_tokens, empty_tokens, salient_indices, empty_indices

    all_indices = torch.arange(tokens_per_frame, device=device, dtype=torch.long).unsqueeze(0).expand(num_frames, -1)
    all_mask = torch.ones((num_frames, tokens_per_frame), dtype=torch.bool, device=device)
    all_mask.scatter_(1, salient_indices, False)
    filtered_indices = all_indices[all_mask].view(num_frames, filtered_num)
    batchframe_indices = torch.arange(num_frames, device=device).unsqueeze(1).expand(-1, filtered_num)
    filtered_tokens = frame_features[batchframe_indices, filtered_indices]
    return salient_tokens, filtered_tokens, salient_indices, filtered_indices


def compute_density_score(filtered_tokens: torch.Tensor, knn_k: int = 4) -> torch.Tensor:
    if filtered_tokens.numel() == 0:
        return torch.empty(filtered_tokens.shape[:2], dtype=torch.float32, device=filtered_tokens.device)

    hidden_dim = filtered_tokens.shape[-1]
    knn_k = max(1, min(int(knn_k), int(filtered_tokens.shape[1])))
    dist_matrix = torch.cdist(filtered_tokens.float(), filtered_tokens.float()) / (hidden_dim ** 0.5)
    dist_nearest, _ = torch.topk(dist_matrix, k=knn_k, dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    density_mask = (density[:, None, :] > density[:, :, None]).to(dtype=filtered_tokens.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist_0, _ = (dist_matrix * density_mask + dist_max * (1 - density_mask)).min(dim=-1)
    return dist_0 * density


def dtm_single_frame(
    filtered_tokens: torch.Tensor,
    density_score: torch.Tensor,
    context_num: int,
    dtm_beta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, filtered_num, hidden_dim = filtered_tokens.shape
    device = filtered_tokens.device

    context_num = max(0, min(int(context_num), filtered_num))
    if context_num <= 0 or filtered_num <= 0:
        empty_tokens = filtered_tokens.new_empty((num_frames, 0, hidden_dim))
        empty_indices = torch.empty((num_frames, 0), dtype=torch.long, device=device)
        return empty_tokens, empty_indices

    sampled_indices = torch.topk(density_score, k=context_num, dim=-1).indices
    batchframe_indices = torch.arange(num_frames, device=device).unsqueeze(1).expand(-1, context_num)
    context_tokens = filtered_tokens[batchframe_indices, sampled_indices]

    normed_all = F.normalize(filtered_tokens.float(), dim=-1)
    normed_ctx = F.normalize(context_tokens.float(), dim=-1)
    similarity = torch.bmm(normed_all, normed_ctx.transpose(1, 2))

    assign_one_hot = torch.zeros(
        num_frames,
        filtered_num,
        context_num,
        dtype=filtered_tokens.dtype,
        device=device,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2, keepdim=True), 1)

    avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=float(dtm_beta))
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    aggregated = torch.bmm(assign_one_hot.transpose(1, 2), filtered_tokens) / counts
    context_tokens = avg_weights * context_tokens + (1 - avg_weights) * aggregated
    return context_tokens, sampled_indices


def dtm_multi_frame(
    filtered_tokens: torch.Tensor,
    density_score: torch.Tensor,
    frm_context_num_list: torch.Tensor,
    frame_context_num: int,
    segment_sizes: List[int],
    dtm_beta: float,
) -> List[dict]:
    hidden_dim = filtered_tokens.shape[-1]
    device = filtered_tokens.device
    merged_groups: List[dict] = []

    idx_seg_start = 0
    for seg_len in segment_sizes:
        if seg_len <= 1:
            idx_seg_start += seg_len
            continue

        ctx_num_list = frm_context_num_list[idx_seg_start:idx_seg_start + seg_len]
        target_mask = ctx_num_list > int(frame_context_num)
        target_num = int(target_mask.sum().item())
        if target_num <= 0:
            idx_seg_start += seg_len
            continue

        cur_ctx_num = int(ctx_num_list[target_mask][0].item())
        seg_density = density_score[idx_seg_start:idx_seg_start + seg_len]
        seg_density_target = seg_density[target_mask]
        seg_filtered = filtered_tokens[idx_seg_start:idx_seg_start + seg_len]
        seg_filtered_target = seg_filtered[target_mask]

        if seg_filtered_target.numel() == 0:
            idx_seg_start += seg_len
            continue

        cur_ctx_num = max(0, min(cur_ctx_num, int(seg_filtered_target.shape[1])))
        if cur_ctx_num <= 0:
            idx_seg_start += seg_len
            continue

        seg_filtered_all = seg_filtered.view(1, -1, hidden_dim).expand(target_num, -1, -1)
        sampled_idx = torch.topk(seg_density_target, k=cur_ctx_num, dim=-1).indices
        batchframe_idx = torch.arange(target_num, device=device).unsqueeze(1).expand(-1, cur_ctx_num)
        ctx_tokens = seg_filtered_target[batchframe_idx, sampled_idx]

        normed_all = F.normalize(seg_filtered_all.float(), dim=-1)
        normed_ctx = F.normalize(ctx_tokens.float(), dim=-1)
        similarity = torch.bmm(normed_all, normed_ctx.transpose(1, 2))

        assign_one_hot = torch.zeros(
            target_num,
            normed_all.shape[1],
            cur_ctx_num,
            dtype=filtered_tokens.dtype,
            device=device,
        )
        assign_one_hot.scatter_(2, similarity.argmax(dim=2, keepdim=True), 1)

        avg_weights = (1 / (assign_one_hot.sum(dim=1).unsqueeze(-1) + 1)).clamp(min=float(dtm_beta))
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        aggregated = torch.bmm(assign_one_hot.transpose(1, 2), seg_filtered_all) / counts
        ctx_tokens = avg_weights * ctx_tokens + (1 - avg_weights) * aggregated

        frame_indices = torch.arange(idx_seg_start, idx_seg_start + seg_len, device=device)[target_mask]
        merged_groups.append(
            {
                "frame_indices": frame_indices,
                "tokens": ctx_tokens,
                "sampled_indices": sampled_idx,
            }
        )
        idx_seg_start += seg_len

    return merged_groups


# ---------------------------------------------------------------------------
# Frame-sequence compression
# ---------------------------------------------------------------------------


def _compress_frame_sequence(
    frame_sequence: torch.Tensor,
    retention_ratio: float,
    dyseg_c: int,
    dyseg_tau: float,
    stprune_d: float,
    dtm_p: int,
    dtm_beta: float,
    score_type: str,
    min_tokens_per_frame: int = 4,
) -> Tuple[List[torch.Tensor], List[int], Dict[str, object]]:
    num_frames, tokens_per_frame, hidden_dim = frame_sequence.shape
    frame_scores = torch.stack([_frame_token_scores(frame, score_type=score_type) for frame in frame_sequence], dim=0)
    frame_global_features = _frame_global_features(frame_sequence, score_type=score_type)
    segment_sizes = dynamic_segmentation(frame_global_features, dyseg_c=dyseg_c, dyseg_tau=dyseg_tau)

    frame_retain_num = max(min_tokens_per_frame, min(tokens_per_frame, int(round(tokens_per_frame * float(retention_ratio)))))
    frame_salient_num = max(min_tokens_per_frame, min(frame_retain_num, frame_retain_num - int(round(frame_retain_num * float(stprune_d)))))
    frame_context_num = max(0, frame_retain_num - frame_salient_num)

    frm_context_num_list = torch.zeros(num_frames, dtype=torch.int, device=frame_sequence.device)
    if frame_context_num > 0:
        offset = 0
        for seg_len in segment_sizes:
            seg_context_num = frame_context_num * int(seg_len)
            temp_num = max(1, (int(seg_len) + int(dtm_p) - 1) // int(dtm_p))
            cur_frm_context_num = max(1, seg_context_num // temp_num)
            end = offset + int(seg_len)
            seg_indices = torch.arange(int(seg_len) - 1, -1, -1, device=frame_sequence.device)
            mask = seg_indices % int(dtm_p) == 0
            frm_context_num_list[offset:end][mask] = cur_frm_context_num
            offset = end

    salient_tokens, filtered_tokens, salient_indices, filtered_indices = attention_token_selection(
        frame_sequence,
        frame_scores,
        frame_salient_num,
    )
    density_score = compute_density_score(filtered_tokens)

    single_frame_ctx, single_sampled_indices = dtm_single_frame(
        filtered_tokens,
        density_score,
        frame_context_num,
        dtm_beta,
    )
    single_mask = frm_context_num_list == frame_context_num

    multi_frame_groups = dtm_multi_frame(
        filtered_tokens,
        density_score,
        frm_context_num_list,
        frame_context_num,
        segment_sizes,
        dtm_beta,
    )

    per_frame_tokens: List[List[torch.Tensor]] = [[] for _ in range(num_frames)]
    per_frame_indices: List[List[torch.Tensor]] = [[] for _ in range(num_frames)]

    for frame_idx in range(num_frames):
        if salient_indices.shape[1] > 0:
            per_frame_tokens[frame_idx].append(salient_tokens[frame_idx])
            per_frame_indices[frame_idx].append(salient_indices[frame_idx])

    if frame_context_num > 0 and single_sampled_indices.shape[1] > 0:
        single_frames = torch.nonzero(single_mask, as_tuple=False).squeeze(1)
        for frame_idx in single_frames.tolist():
            keep_idx = filtered_indices[frame_idx, single_sampled_indices[frame_idx]]
            per_frame_tokens[frame_idx].append(single_frame_ctx[frame_idx])
            per_frame_indices[frame_idx].append(keep_idx)

    for group in multi_frame_groups:
        for local_idx, frame_idx in enumerate(group["frame_indices"].tolist()):
            keep_idx = filtered_indices[frame_idx, group["sampled_indices"][local_idx]]
            per_frame_tokens[frame_idx].append(group["tokens"][local_idx])
            per_frame_indices[frame_idx].append(keep_idx)

    frame_outputs: List[torch.Tensor] = []
    frame_sizes: List[int] = []
    frame_keep_indices: List[List[int]] = []
    for frame_idx in range(num_frames):
        if not per_frame_tokens[frame_idx]:
            fallback_keep = torch.arange(frame_retain_num, device=frame_sequence.device, dtype=torch.long)
            frame_outputs.append(frame_sequence[frame_idx, fallback_keep])
            frame_sizes.append(int(fallback_keep.shape[0]))
            frame_keep_indices.append([int(x) for x in fallback_keep.tolist()])
            continue

        tokens_cat = torch.cat(per_frame_tokens[frame_idx], dim=0)
        indices_cat = torch.cat(per_frame_indices[frame_idx], dim=0)
        unique_idx, inverse = torch.unique(indices_cat, sorted=True, return_inverse=True)
        pooled = torch.zeros(
            unique_idx.shape[0],
            hidden_dim,
            dtype=tokens_cat.dtype,
            device=tokens_cat.device,
        )
        pooled.index_add_(0, inverse, tokens_cat)
        counts = torch.bincount(inverse, minlength=unique_idx.shape[0]).clamp(min=1).to(dtype=tokens_cat.dtype, device=tokens_cat.device)
        pooled = pooled / counts.unsqueeze(-1)

        if pooled.shape[0] > frame_retain_num:
            pooled = pooled[:frame_retain_num]
            unique_idx = unique_idx[:frame_retain_num]

        frame_outputs.append(pooled.contiguous())
        frame_sizes.append(int(pooled.shape[0]))
        frame_keep_indices.append([int(x) for x in unique_idx.tolist()])

    meta: Dict[str, object] = {
        "segment_sizes": [int(x) for x in segment_sizes],
        "frame_context_num_list": [int(x) for x in frm_context_num_list.detach().cpu().tolist()],
        "frame_output_sizes": frame_sizes,
        "frame_keep_indices": frame_keep_indices,
        "frame_retain_num": int(frame_retain_num),
        "frame_salient_num": int(frame_salient_num),
        "frame_context_num": int(frame_context_num),
    }
    return frame_outputs, frame_sizes, meta


def _compress_image_sequence(
    frame_sequence: torch.Tensor,
    config: FastVidConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if frame_sequence is None or not isinstance(frame_sequence, torch.Tensor) or frame_sequence.ndim != 3:
        return frame_sequence, {}

    num_frames, _, hidden_dim = frame_sequence.shape
    compressed_frames, frame_sizes, meta = _compress_frame_sequence(
        frame_sequence,
        retention_ratio=config.retention_ratio,
        dyseg_c=config.dyseg_c,
        dyseg_tau=config.dyseg_tau,
        stprune_d=config.stprune_d,
        dtm_p=config.dtm_p,
        dtm_beta=config.dtm_beta,
        score_type=str(config.score_type).strip().lower(),
        min_tokens_per_frame=config.min_tokens_per_frame,
    )

    if num_frames == 1:
        compressed = compressed_frames[0].unsqueeze(0)
    else:
        min_len = min(frame_sizes) if frame_sizes else 0
        if min_len <= 0:
            compressed = frame_sequence.new_empty((0, 0, hidden_dim))
        else:
            compressed = torch.stack([frame[:min_len] for frame in compressed_frames], dim=0)
    return compressed, meta


# ---------------------------------------------------------------------------
# Qwen2.5-VL specific interface
# ---------------------------------------------------------------------------


def compress_frames_by_grid(
    image_embeds: torch.Tensor,
    image_grid_thw: torch.Tensor,
    config: FastVidConfig,
    spatial_merge_size: int = 2,
) -> Tuple[torch.Tensor, List[int]]:
    """Compress flattened image embeddings using FastVid.

    Args:
        image_embeds: [total_visual_tokens, hidden_dim] from Qwen2.5-VL visual tower.
        image_grid_thw: [num_frames, 3] tensor of (t, h, w) per frame.
        config: FastVidConfig.
        spatial_merge_size: Spatial merge size used by the visual tower
            (Qwen2.5-VL default is 2, meaning 2x2 patches are merged into 1 token).

    Returns:
        compressed_embeds: [total_compressed_tokens, hidden_dim]
        compressed_sizes: list of per-frame token counts after compression.
    """
    if not config.enabled or image_embeds is None or image_embeds.numel() == 0:
        return image_embeds, []

    # Compute tokens per frame from grid_thw.
    # The visual tower merges spatial_merge_size x spatial_merge_size patches into one token,
    # so actual visual tokens per frame = t * h * w // (spatial_merge_size ** 2).
    tokens_per_frame = (
        image_grid_thw[:, 0]
        * image_grid_thw[:, 1]
        * image_grid_thw[:, 2]
        // (spatial_merge_size * spatial_merge_size)
    ).tolist()

    frames = []
    offset = 0
    for tpf in tokens_per_frame:
        frames.append(image_embeds[offset:offset + tpf])
        offset += tpf

    if offset != image_embeds.shape[0]:
        # Safety: if mismatch, fall back to no compression
        return image_embeds, tokens_per_frame

    # Check if all frames have the same token count.
    # If not, compress each frame individually to avoid torch.stack failure.
    unique_sizes = set(f.shape[0] for f in frames)
    if len(unique_sizes) > 1:
        compressed_frames: List[torch.Tensor] = []
        compressed_sizes: List[int] = []
        for frame in frames:
            frame_batch = frame.unsqueeze(0)  # [1, tokens, hidden]
            comp, meta = _compress_image_sequence(frame_batch, config)
            if comp.ndim == 3:
                size = comp.shape[1]
                comp = comp.view(-1, comp.shape[-1])
            else:
                size = meta.get("frame_output_sizes", [frame.shape[0]])[0]
            compressed_frames.append(comp)
            compressed_sizes.append(size)
        return torch.cat(compressed_frames, dim=0), compressed_sizes

    frame_sequence = torch.stack(frames, dim=0)  # [num_frames, tokens_per_frame, hidden_dim]
    compressed, meta = _compress_image_sequence(frame_sequence, config)
    # _compress_image_sequence aligns all frames to min_len for multi-frame cases.
    # For Qwen2.5-VL we need flattened embeddings; flatten here.
    if compressed.ndim == 3:
        # [num_frames, min_len, hidden_dim] -> [num_frames * min_len, hidden_dim]
        compressed = compressed.view(-1, compressed.shape[-1])
        min_len = compressed.shape[0] // len(frames)
        compressed_sizes = [min_len] * len(frames)
    else:
        compressed_sizes = meta.get("frame_output_sizes", tokens_per_frame)
    return compressed, compressed_sizes


# ---------------------------------------------------------------------------
# Original StreamVLN-compatible interface (kept for reference / reuse)
# ---------------------------------------------------------------------------


def _compress_memory_bank(
    memory_bank: torch.Tensor,
    tokens_per_frame: Optional[int],
    config: FastVidConfig,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    if memory_bank is None or not isinstance(memory_bank, torch.Tensor):
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}
    if memory_bank.ndim != 3:
        return memory_bank, {"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []}

    compressed_slots: List[torch.Tensor] = []
    slot_sizes: List[int] = []
    slot_meta: List[Dict[str, object]] = []
    inferred_frames = 0
    inferred_tokens_per_frame = 0

    for slot in memory_bank:
        if tokens_per_frame and tokens_per_frame > 0 and int(slot.shape[0]) % tokens_per_frame == 0:
            num_frames = max(1, int(slot.shape[0]) // tokens_per_frame)
            inferred_frames = max(inferred_frames, num_frames)
            inferred_tokens_per_frame = int(tokens_per_frame)
            slot_frames = slot.view(num_frames, tokens_per_frame, slot.shape[-1])
        else:
            slot_frames = slot.unsqueeze(0)

        compressed_frames, _, meta = _compress_frame_sequence(
            slot_frames,
            retention_ratio=config.retention_ratio,
            dyseg_c=config.dyseg_c,
            dyseg_tau=config.dyseg_tau,
            stprune_d=config.stprune_d,
            dtm_p=config.dtm_p,
            dtm_beta=config.dtm_beta,
            score_type=str(config.score_type).strip().lower(),
            min_tokens_per_frame=config.min_tokens_per_frame,
        )
        compressed_slot = torch.cat(compressed_frames, dim=0).contiguous() if compressed_frames else slot.new_empty((0, slot.shape[-1]))
        compressed_slots.append(compressed_slot)
        slot_sizes.append(int(compressed_slot.shape[0]))
        meta["num_frames"] = int(slot_frames.shape[0])
        meta["tokens_per_frame"] = int(slot_frames.shape[1])
        slot_meta.append(meta)

    min_len = min(slot_sizes) if slot_sizes else 0
    if min_len <= 0:
        hidden = int(memory_bank.shape[-1])
        return memory_bank.new_empty((0, 0, hidden)), {
            "memory_slots": int(memory_bank.shape[0]),
            "memory_frames_inferred": inferred_frames,
            "memory_tokens_per_frame": inferred_tokens_per_frame,
            "slot_meta": slot_meta,
        }

    aligned = [slot[:min_len] for slot in compressed_slots]
    return torch.stack(aligned, dim=0), {
        "memory_slots": int(memory_bank.shape[0]),
        "memory_frames_inferred": inferred_frames,
        "memory_tokens_per_frame": inferred_tokens_per_frame,
        "slot_meta": slot_meta,
    }


def apply_fastvid_compression(
    image_features: List[torch.Tensor],
    memory_features: List[Optional[torch.Tensor]],
    config: FastVidConfig,
) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]], Dict[str, object]]:
    """Original StreamVLN-style batch-wise compression interface."""
    compressed_image_features: List[torch.Tensor] = []
    compressed_memory_features: List[Optional[torch.Tensor]] = []
    image_meta: List[Dict[str, object]] = []
    memory_meta: List[Dict[str, object]] = []

    for batch_idx, frame_features in enumerate(image_features):
        tokens_per_frame = None
        if not isinstance(frame_features, torch.Tensor) or frame_features.ndim != 3:
            compressed_image_features.append(frame_features)
            image_meta.append({})
        else:
            tokens_per_frame = int(frame_features.shape[1])
            compressed_frames, meta = _compress_image_sequence(frame_features, config)
            compressed_image_features.append(compressed_frames)
            image_meta.append(meta)

        memory_bank = memory_features[batch_idx] if batch_idx < len(memory_features) else None
        if memory_bank is None:
            compressed_memory_features.append(None)
            memory_meta.append({"memory_slots": 0, "memory_frames_inferred": 0, "memory_tokens_per_frame": 0, "slot_meta": []})
        else:
            compressed_memory_bank, meta = _compress_memory_bank(
                memory_bank,
                tokens_per_frame=tokens_per_frame,
                config=config,
            )
            compressed_memory_features.append(compressed_memory_bank)
            memory_meta.append(meta)

    stats: Dict[str, object] = {
        "method": "fastvid",
        "retention_ratio": float(config.retention_ratio),
        "dyseg_c": int(config.dyseg_c),
        "dyseg_tau": float(config.dyseg_tau),
        "stprune_d": float(config.stprune_d),
        "dtm_p": int(config.dtm_p),
        "dtm_beta": float(config.dtm_beta),
        "score_type": str(config.score_type).strip().lower(),
        "image_meta": image_meta,
        "memory_meta": memory_meta,
    }
    return compressed_image_features, compressed_memory_features, stats
