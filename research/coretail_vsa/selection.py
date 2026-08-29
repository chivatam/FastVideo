from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from research.anchored_fine_vsa.selection import (
    CHILDREN_PER_PARENT,
    CHILD_WIDTH,
    parent_indices_to_children,
    selected_support_mask,
    support_token_count,
)
from research.fine_vsa.replay import (
    NOMINAL_KV_TOKENS,
    PARENT_WIDTH,
    select_children_fixed_tokens,
)

CORE_PARENT_COUNTS = (31, 62)
MAX_TAIL_DESCRIPTORS = NOMINAL_KV_TOKENS // CHILD_WIDTH


@dataclass(frozen=True)
class CoreMaskTable:
    timesteps: tuple[int, ...]
    core25: torch.Tensor
    core50: torch.Tensor
    calibration_prompt_hash: str
    quantile_semantics: str

    @classmethod
    def from_path(cls, path: str | Path) -> CoreMaskTable:
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        if int(payload["format_version"]) != 1:
            raise ValueError("Unsupported CoreTail mask format")
        core25 = payload["core25_indices"].to(torch.int16).contiguous()
        core50 = payload["core50_indices"].to(torch.int16).contiguous()
        if core25.shape[-1] != 31 or core50.shape[-1] != 62:
            raise ValueError("CoreTail mask file has invalid core widths")
        return cls(
            timesteps=tuple(int(value) for value in payload["timesteps"]),
            core25=core25,
            core50=core50,
            calibration_prompt_hash=str(payload["calibration_prompt_hash"]),
            quantile_semantics=str(payload["quantile_semantics"]),
        )

    def indices(
        self,
        *,
        timestep: int,
        layer: int,
        core_parent_blocks: int,
        device: torch.device,
    ) -> torch.Tensor:
        if core_parent_blocks not in CORE_PARENT_COUNTS:
            raise ValueError("Only frozen Core25/Core50 masks are allowed")
        try:
            step = self.timesteps.index(int(timestep))
        except ValueError as exc:
            raise KeyError(f"No frozen core for timestep {timestep}") from exc
        source = self.core25 if core_parent_blocks == 31 else self.core50
        if layer < 0 or layer >= source.shape[1]:
            raise IndexError(f"Layer {layer} is outside the core table")
        return source[step, layer].to(
            device=device,
            dtype=torch.int64,
        )[None]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timesteps": list(self.timesteps),
            "layers": int(self.core25.shape[1]),
            "heads": int(self.core25.shape[2]),
            "query_blocks": int(self.core25.shape[3]),
            "core25_parent_blocks": int(self.core25.shape[-1]),
            "core50_parent_blocks": int(self.core50.shape[-1]),
            "calibration_prompt_hash": self.calibration_prompt_hash,
            "quantile_semantics": self.quantile_semantics,
        }


@dataclass(frozen=True)
class CoreTailSelection:
    core_parent_blocks: int
    core_parent_indices: torch.Tensor
    core_parent_active: torch.Tensor
    core_active_parent_blocks: torch.Tensor
    core_child_indices: torch.Tensor
    fine_tail_indices: torch.Tensor
    selected_indices: torch.Tensor
    native_actual_kv_tokens: torch.Tensor
    core_actual_kv_tokens: torch.Tensor
    fine_tail_actual_kv_tokens: torch.Tensor
    selected_actual_kv_tokens: torch.Tensor
    tail_active_descriptors: torch.Tensor
    duplicate_valid_tokens: torch.Tensor


def select_coretail_support(
    child_scores: torch.Tensor,
    child_sizes: torch.Tensor,
    parent_scores: torch.Tensor,
    parent_sizes: torch.Tensor,
    core_parent_indices: torch.Tensor,
) -> CoreTailSelection:
    core_parent_blocks = core_parent_indices.shape[-1]
    if core_parent_blocks not in CORE_PARENT_COUNTS:
        raise ValueError("Only frozen Core25/Core50 ratios are supported")
    if core_parent_indices.shape != (
            *child_scores.shape[:-1],
            core_parent_blocks,
    ):
        raise ValueError("Core mask geometry does not match attention state")
    if parent_sizes[core_parent_indices].le(0).any():
        raise RuntimeError("Frozen core contains a zero-valid-token parent")

    native_indices = torch.topk(
        parent_scores,
        NOMINAL_KV_TOKENS // PARENT_WIDTH,
        dim=-1,
        sorted=True,
    ).indices
    native_actual = parent_sizes[native_indices].sum(dim=-1)
    core_sizes = parent_sizes[core_parent_indices]
    core_active = torch.zeros_like(core_parent_indices, dtype=torch.bool)
    remaining = native_actual.clone()
    for rank in range(core_parent_blocks):
        fits = core_sizes[..., rank].le(remaining)
        core_active[..., rank] = fits
        remaining = remaining - (core_sizes[..., rank] * fits.to(core_sizes.dtype))
    core_actual = (core_sizes * core_active.to(core_sizes.dtype)).sum(dim=-1)
    tail_target = native_actual - core_actual
    raw_core_children = parent_indices_to_children(core_parent_indices)
    active_children = core_active.unsqueeze(-1).expand(
        *core_active.shape,
        CHILDREN_PER_PARENT,
    ).flatten(-2)
    zero_children = torch.nonzero(
        child_sizes.eq(0),
        as_tuple=False,
    ).flatten()
    if zero_children.numel() == 0 and (~active_children).any():
        raise RuntimeError("CoreTail needs a zero-valid filler for budget projection")
    filler = (zero_children[0] if zero_children.numel() else torch.tensor(0, device=child_sizes.device))
    core_children = torch.where(
        active_children,
        raw_core_children,
        filler.to(raw_core_children.dtype),
    )

    candidate = torch.ones_like(child_scores, dtype=torch.bool)
    core_support = selected_support_mask(core_children, child_sizes)
    candidate &= ~core_support
    fine_tail = select_children_fixed_tokens(
        child_scores,
        child_sizes,
        selected_blocks=MAX_TAIL_DESCRIPTORS,
        factor=CHILDREN_PER_PARENT,
        parent_scores=parent_scores,
        parent_pool=None,
        target_tokens=tail_target,
        child_width=CHILD_WIDTH,
        candidate_mask=candidate,
    )
    selected = torch.cat(
        [core_children.to(torch.int32), fine_tail],
        dim=-1,
    )
    core_mask = core_support
    tail_mask = selected_support_mask(fine_tail, child_sizes)
    duplicate = support_token_count(
        core_mask & tail_mask,
        child_sizes,
    )
    tail_actual = child_sizes[fine_tail.long()].sum(dim=-1)
    selected_actual = support_token_count(
        core_mask | tail_mask,
        child_sizes,
    )
    if duplicate.any():
        raise RuntimeError("Fine8 tail overlaps calibrated static core")
    if not torch.equal(selected_actual, native_actual):
        raise RuntimeError("CoreTail failed exact native valid-token matching")
    tail_active = child_sizes[fine_tail.long()].gt(0).sum(dim=-1)
    return CoreTailSelection(
        core_parent_blocks=core_parent_blocks,
        core_parent_indices=core_parent_indices.to(torch.int32),
        core_parent_active=core_active,
        core_active_parent_blocks=core_active.sum(dim=-1),
        core_child_indices=core_children.to(torch.int32),
        fine_tail_indices=fine_tail,
        selected_indices=selected,
        native_actual_kv_tokens=native_actual,
        core_actual_kv_tokens=core_actual,
        fine_tail_actual_kv_tokens=tail_actual,
        selected_actual_kv_tokens=selected_actual,
        tail_active_descriptors=tail_active,
        duplicate_valid_tokens=duplicate,
    )
