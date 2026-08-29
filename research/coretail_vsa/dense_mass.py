from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    construct_variable_block_sizes,
    get_non_pad_index,
    get_tile_partition_indices,
)
from research.fine_vsa.fine_attention import (
    child_block_mean,
    child_block_sizes,
)
from research.fine_vsa.replay import (
    NATIVE_PARENT_K,
    PARENT_WIDTH,
    select_children_fixed_tokens,
)

DIT_SEQUENCE_SHAPE = (21, 30, 52)
EXPECTED_SEQUENCE = math.prod(DIT_SEQUENCE_SHAPE)
EXPECTED_HEADS = 12
EXPECTED_HEAD_DIM = 128
CHILD_WIDTH = 8
FINE8_BLOCKS = 1000
Q_BLOCK_CHUNK = 64


def is_coretail_dense_call(
    query: torch.Tensor,
    key: torch.Tensor,
) -> bool:
    return (query.ndim == 4 and key.ndim == 4 and query.shape == key.shape and query.shape[0] == 1
            and query.shape[1] == EXPECTED_SEQUENCE and query.shape[2] == EXPECTED_HEADS
            and query.shape[3] == EXPECTED_HEAD_DIM)


def _tiled_padded(tensor: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = tensor.device
    num_tiles = tuple(
        math.ceil(length / tile) for length, tile in zip(
            DIT_SEQUENCE_SHAPE,
            VSA_TILE_SIZE,
            strict=True,
        ))
    parent_sizes = construct_variable_block_sizes(
        DIT_SEQUENCE_SHAPE,
        num_tiles,
        device,
    )
    non_pad_index = get_non_pad_index(
        parent_sizes,
        PARENT_WIDTH,
    )
    tile_partition = get_tile_partition_indices(
        DIT_SEQUENCE_SHAPE,
        VSA_TILE_SIZE,
        device,
    )
    padded = torch.zeros(
        (
            tensor.shape[0],
            parent_sizes.numel() * PARENT_WIDTH,
            tensor.shape[2],
            tensor.shape[3],
        ),
        device=device,
        dtype=tensor.dtype,
    )
    padded[:, non_pad_index] = tensor[:, tile_partition]
    return padded, parent_sizes, non_pad_index


def _summary(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.float().flatten()
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.1, 0.5, 0.9],
            device=values.device,
        ),
    )
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_p10": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_p90": float(quantiles[2].item()),
    }


def capture_dense_block_mass(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    job_id: str,
    timestep: int,
    layer: int,
    output_root: Path,
) -> dict[str, Any]:
    if not is_coretail_dense_call(query, key):
        raise ValueError("Unexpected dense CoreTail attention geometry")
    started = time.perf_counter()
    query_padded, parent_sizes, non_pad_index = _tiled_padded(query)
    key_padded, key_parent_sizes, key_non_pad_index = _tiled_padded(key)
    if not torch.equal(parent_sizes, key_parent_sizes):
        raise RuntimeError("Dense Q/K tiled geometry differs")
    if not torch.equal(non_pad_index, key_non_pad_index):
        raise RuntimeError("Dense Q/K valid-token indices differ")

    query_bhsd = query_padded.transpose(1, 2).contiguous()
    key_bhsd = key_padded.transpose(1, 2).contiguous()
    from fastvideo_kernel.triton_kernels.fused_compress_topk import (
        fused_block_mean, )

    query_coarse = fused_block_mean(
        query_bhsd,
        parent_sizes,
        PARENT_WIDTH,
    )
    key_parent = fused_block_mean(
        key_bhsd,
        parent_sizes,
        PARENT_WIDTH,
    )
    parent_scores = torch.matmul(
        query_coarse,
        key_parent.transpose(-2, -1),
    ) / math.sqrt(query.shape[-1])
    native_indices = torch.topk(
        parent_scores,
        NATIVE_PARENT_K,
        dim=-1,
    ).indices
    native_actual = parent_sizes[native_indices].sum(dim=-1)

    child_key, child_sizes = child_block_mean(
        key_bhsd,
        parent_sizes,
        CHILD_WIDTH,
    )
    child_scores = torch.matmul(
        query_coarse,
        child_key.transpose(-2, -1),
    ) / math.sqrt(query.shape[-1])
    fine_indices = select_children_fixed_tokens(
        child_scores,
        child_sizes,
        selected_blocks=FINE8_BLOCKS,
        factor=PARENT_WIDTH // CHILD_WIDTH,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=native_actual,
        child_width=CHILD_WIDTH,
    )

    batch, heads, _, head_dim = query_bhsd.shape
    parent_blocks = int(parent_sizes.numel())
    query_blocks = parent_blocks
    child_blocks = child_sizes.numel()
    parent_mass = torch.empty(
        (heads, query_blocks, parent_blocks),
        dtype=torch.float16,
        device="cpu",
    )
    native_coverage = torch.empty(
        (heads, query_blocks),
        dtype=torch.float32,
        device="cpu",
    )
    fine8_coverage = torch.empty_like(native_coverage)
    valid_key = torch.zeros(
        key_bhsd.shape[-2],
        dtype=torch.bool,
        device=query.device,
    )
    valid_key[non_pad_index] = True
    query_sizes = parent_sizes
    scale = 1.0 / math.sqrt(head_dim)
    for block_start in range(0, query_blocks, Q_BLOCK_CHUNK):
        block_stop = min(
            block_start + Q_BLOCK_CHUNK,
            query_blocks,
        )
        token_start = block_start * PARENT_WIDTH
        token_stop = block_stop * PARENT_WIDTH
        query_chunk = query_bhsd[:, :, token_start:token_stop]
        logits = torch.matmul(
            query_chunk,
            key_bhsd.transpose(-2, -1),
        )
        logits = logits.float().mul_(scale)
        logits.masked_fill_(
            ~valid_key.view(1, 1, 1, -1),
            -float("inf"),
        )
        probabilities = torch.softmax(logits, dim=-1)
        query_count = block_stop - block_start
        query_valid = (torch.arange(
            PARENT_WIDTH,
            device=query.device,
        )[None, :] < query_sizes[block_start:block_stop, None])
        child_mass = probabilities.view(
            batch,
            heads,
            query_count,
            PARENT_WIDTH,
            child_blocks,
            CHILD_WIDTH,
        ).sum(dim=-1)
        child_mass *= query_valid.view(
            1,
            1,
            query_count,
            PARENT_WIDTH,
            1,
        )
        child_mass = child_mass.sum(dim=3) / query_sizes[block_start:block_stop].clamp_min(1).view(1, 1, query_count, 1)
        parent_chunk_mass = child_mass.view(
            batch,
            heads,
            query_count,
            parent_blocks,
            PARENT_WIDTH // CHILD_WIDTH,
        ).sum(dim=-1)
        parent_mass[
            :,
            block_start:block_stop,
        ] = parent_chunk_mass[0].to(
            device="cpu",
            dtype=torch.float16,
        )
        native_coverage[
            :,
            block_start:block_stop,
        ] = torch.gather(
            parent_chunk_mass,
            -1,
            native_indices[
                :,
                :,
                block_start:block_stop,
            ],
        ).sum(dim=-1)[0].to(device="cpu")
        fine8_coverage[
            :,
            block_start:block_stop,
        ] = torch.gather(
            child_mass,
            -1,
            fine_indices[
                :,
                :,
                block_start:block_stop,
            ].long(),
        ).sum(dim=-1)[0].to(device="cpu")
        del (
            query_chunk,
            logits,
            probabilities,
            child_mass,
            parent_chunk_mass,
        )

    fine_parent = torch.div(
        fine_indices.long(),
        PARENT_WIDTH // CHILD_WIDTH,
        rounding_mode="floor",
    )
    fine_valid = child_sizes[fine_indices.long()].gt(0)
    parent_hits = torch.zeros(
        fine_indices.shape[:-1] + (parent_blocks, ),
        dtype=torch.int16,
        device=query.device,
    )
    parent_hits.scatter_add_(
        -1,
        fine_parent,
        fine_valid.to(torch.int16),
    )
    packed_fine_parent = np.packbits(
        parent_hits[0].gt(0).cpu().numpy(),
        axis=-1,
        bitorder="little",
    )
    mass_sum_error = (parent_mass.float().sum(dim=-1) - 1.0).abs()
    record = {
        "format_version":
        1,
        "job_id":
        job_id,
        "timestep":
        int(timestep),
        "layer":
        int(layer),
        "dit_sequence_shape":
        DIT_SEQUENCE_SHAPE,
        "tile_size":
        VSA_TILE_SIZE,
        "parent_width":
        PARENT_WIDTH,
        "child_width":
        CHILD_WIDTH,
        "parent_sizes":
        parent_sizes.to(
            device="cpu",
            dtype=torch.int16,
        ),
        "native_actual_kv_tokens":
        native_actual[0].to(
            device="cpu",
            dtype=torch.int16,
        ),
        "dense_parent_mass":
        parent_mass,
        "native_dense_mass_coverage":
        native_coverage,
        "fine8_dense_mass_coverage":
        fine8_coverage,
        "fine8_parent_any_packbits":
        torch.from_numpy(packed_fine_parent),
        "fine8_parent_any_bitorder":
        "little",
        "fine8_parent_any_parent_blocks":
        parent_blocks,
        "mass_definition": ("mean over valid full-resolution query tokens in each tiled "
                            "KV64 query block of exact dense softmax probability assigned "
                            "to valid K tokens in each tiled KV64 key block"),
    }
    destination = (output_root / job_id / f"t{int(timestep)}-l{int(layer):02d}.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(record, temporary)
    os.replace(temporary, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "event_type":
        "coretail_dense_mass_capture",
        "mass_path":
        str(destination),
        "mass_sha256":
        digest,
        "timestep":
        int(timestep),
        "layer":
        int(layer),
        "heads":
        heads,
        "query_blocks":
        query_blocks,
        "key_blocks":
        parent_blocks,
        "valid_tokens":
        int(parent_sizes.sum().item()),
        "padded_tokens":
        int(parent_blocks * PARENT_WIDTH),
        "capture_ms":
        elapsed_ms,
        "mass_sum_error_max":
        float(mass_sum_error.max().item()),
        "quantile_semantics": ("core construction uses linear p10 across 32 prompts: "
                               "h=(n-1)*0.10=3.1, 0.9*x[3]+0.1*x[4]"),
        **_summary(native_coverage, "native_mass_coverage"),
        **_summary(fine8_coverage, "fine8_mass_coverage"),
    }


def unpack_fine_parent_any(record: dict[str, Any]) -> torch.Tensor:
    packed = record["fine8_parent_any_packbits"].cpu().numpy()
    unpacked = np.unpackbits(
        packed,
        axis=-1,
        count=int(record["fine8_parent_any_parent_blocks"]),
        bitorder=str(record["fine8_parent_any_bitorder"]),
    )
    return torch.from_numpy(unpacked.astype(np.bool_))


def child_sizes_for_record(record: dict[str, Any]) -> torch.Tensor:
    return child_block_sizes(
        record["parent_sizes"].to(torch.int32),
        CHILD_WIDTH,
    )
