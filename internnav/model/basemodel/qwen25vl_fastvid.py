"""Qwen2.5-VL model subclass with FastVid video token compression.

This module creates `Qwen2_5_VLForConditionalGenerationFastVid`, which inherits
from the standard `Qwen2_5_VLForConditionalGeneration` and injects FastVid
compression after the visual tower but before the LLM.

Key design decisions:
1. Compression happens on the flattened image_embeds returned by `self.visual()`.
2. After compression, `input_ids` is dynamically adjusted to match the reduced
   number of image tokens.
3. 1D position_ids are manually constructed and passed to the parent forward,
   bypassing Qwen2.5-VL's `get_rope_index` which depends on the original grid.
4. During generation (decoding), `pixel_values` is None so FastVid is naturally
   skipped; only the prefill step compresses visual tokens.
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from internnav.model.compression.fastvid import compress_frames_by_grid
from internnav.model.compression.feature_flags import FastVidConfig


class Qwen2_5_VLForConditionalGenerationFastVid(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with optional FastVid token compression."""

    def __init__(self, config, fastvid_config: Optional[FastVidConfig] = None):
        super().__init__(config)
        self.fastvid_config = fastvid_config or FastVidConfig()
        self.last_compression_stats: Optional[Dict[str, int]] = None

    def _compress_for_batch(
        self,
        image_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor, Optional[torch.LongTensor]]:
        """Run FastVid compression per-sample in a batch and adjust token tensors.

        Returns:
            compressed_embeds: flattened compressed embeddings for the whole batch.
            new_input_ids: adjusted input_ids with reduced image token counts.
            new_attention_mask: adjusted attention_mask.
            new_labels: adjusted labels (or None).
        """
        batch_size = input_ids.shape[0]
        num_images_total = image_grid_thw.shape[0]
        if num_images_total % batch_size != 0:
            raise ValueError(
                f"image_grid_thw rows ({num_images_total}) not divisible by batch_size ({batch_size})"
            )
        num_images_per_sample = num_images_total // batch_size
        merge_size = 2
        tokens_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2] // (merge_size * merge_size)).tolist()

        # --- 1. Compress image_embeds per sample ---
        compressed_embeds_list: List[torch.Tensor] = []
        compressed_sizes_list: List[List[int]] = []
        offset = 0
        for b in range(batch_size):
            sample_tokens = tokens_per_image[b * num_images_per_sample : (b + 1) * num_images_per_sample]
            sample_embeds = image_embeds[offset : offset + sum(sample_tokens)]
            offset += sum(sample_tokens)
            sample_grid_thw = image_grid_thw[b * num_images_per_sample : (b + 1) * num_images_per_sample]
            compressed, sizes = compress_frames_by_grid(sample_embeds, sample_grid_thw, self.fastvid_config)
            compressed_embeds_list.append(compressed)
            compressed_sizes_list.append(sizes)

        compressed_embeds = torch.cat(compressed_embeds_list, dim=0)

        # --- 2. Adjust input_ids, attention_mask, labels ---
        image_token_id = int(self.config.image_token_id)
        device = input_ids.device
        dtype = input_ids.dtype

        new_input_ids_list: List[List[int]] = []
        new_attention_mask_list: List[List[int]] = []
        new_labels_list: List[List[int]] = []

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            am = attention_mask[b].tolist() if attention_mask is not None else [1] * len(ids)
            lbl = labels[b].tolist() if labels is not None else [-100] * len(ids)

            image_positions = [i for i, x in enumerate(ids) if x == image_token_id]
            sample_tokens = tokens_per_image[b * num_images_per_sample : (b + 1) * num_images_per_sample]
            sample_sizes = compressed_sizes_list[b]

            new_ids: List[int] = []
            new_am: List[int] = []
            new_lbl: List[int] = []
            last_pos = 0
            img_offset = 0

            for tpf, k in zip(sample_tokens, sample_sizes):
                frame_positions = image_positions[img_offset : img_offset + tpf]
                img_offset += tpf

                if frame_positions:
                    # Safety: never add more image tokens than actual positions available
                    k = min(int(k), len(frame_positions))

                    first_pos = frame_positions[0]
                    new_ids.extend(ids[last_pos:first_pos])
                    new_am.extend(am[last_pos:first_pos])
                    new_lbl.extend(lbl[last_pos:first_pos])

                    new_ids.extend([image_token_id] * k)
                    new_am.extend([1] * k)
                    new_lbl.extend([-100] * k)

                    last_pos = frame_positions[-1] + 1

            # Append remaining tokens after the last image block
            new_ids.extend(ids[last_pos:])
            new_am.extend(am[last_pos:])
            new_lbl.extend(lbl[last_pos:])

            new_input_ids_list.append(new_ids)
            new_attention_mask_list.append(new_am)
            new_labels_list.append(new_lbl)

        # --- 3. Pad to max length in batch ---
        max_len = max(len(x) for x in new_input_ids_list)
        pad_id = int(self.config.pad_token_id or self.config.eos_token_id or 0)

        padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=dtype, device=device)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=dtype, device=device) if labels is not None else None

        for b in range(batch_size):
            length = len(new_input_ids_list[b])
            padded_input_ids[b, :length] = torch.tensor(new_input_ids_list[b], dtype=dtype, device=device)
            padded_attention_mask[b, :length] = torch.tensor(new_attention_mask_list[b], dtype=torch.long, device=device)
            if labels is not None:
                padded_labels[b, :length] = torch.tensor(new_labels_list[b], dtype=dtype, device=device)

        return compressed_embeds, padded_input_ids, padded_attention_mask, padded_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # FastVid only applies on the prefill pass (training or first generation step).
        # During cached generation decoding, past_key_values is not None; we must
        # skip compression to avoid re-processing visual tokens.
        is_prefill = (
            self.fastvid_config.enabled
            and pixel_values is not None
            and image_grid_thw is not None
            and inputs_embeds is None
            and pixel_values_videos is None  # videos not supported in Phase 2
            and (past_key_values is None or (
                hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0
            ))
        )
        if is_prefill:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            compressed_embeds, new_input_ids, new_attention_mask, new_labels = self._compress_for_batch(
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            self.last_compression_stats = {
                "original_image_tokens": int((input_ids == self.config.image_token_id).sum().item()),
                "compressed_image_tokens": int((new_input_ids == self.config.image_token_id).sum().item()),
                "original_seq_len": int(input_ids.shape[1]),
                "compressed_seq_len": int(new_input_ids.shape[1]),
            }
            self.last_compressed_labels = new_labels

            # Rebuild inputs_embeds with compressed visual tokens
            inputs_embeds = self.model.embed_tokens(new_input_ids)
            mask = new_input_ids == self.config.image_token_id
            if mask.any():
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                compressed_embeds = compressed_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, compressed_embeds)

            # Build simple 1D position_ids (bypass get_rope_index)
            # Qwen2.5-VL expects [3, batch_size, seq_length]
            batch_size, seq_length = new_input_ids.shape
            pos = torch.arange(seq_length, device=new_input_ids.device)
            position_ids = pos.view(1, 1, -1).expand(3, batch_size, seq_length)

            # Forward through parent with adjusted tensors and no pixel_values.
            # We pass labels=None to super() because new_labels length differs
            # from the logits length expected by the parent loss function;
            # callers can retrieve new_labels from self.last_compressed_labels.
            return super().forward(
                input_ids=None,
                attention_mask=new_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        # Normal path (no FastVid)
        self.last_compression_stats = None
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
