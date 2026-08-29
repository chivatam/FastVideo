from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from research.coretail_vsa.dense_mass import (
    unpack_fine_parent_any, )

CALIBRATION_PROMPTS = 32
CORE25 = 31
CORE50 = 62
P10_POSITION = (CALIBRATION_PROMPTS - 1) * 0.10
P10_LOWER = int(P10_POSITION)
P10_WEIGHT_UPPER = P10_POSITION - P10_LOWER
QUANTILE_SEMANTICS = ("linear p10 across 32 calibration prompts: h=(n-1)*0.10=3.1; "
                      "p10=0.9*x[3]+0.1*x[4] after ascending sort")


def _summarize(values: torch.Tensor) -> dict[str, float]:
    flat = values.float().flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.1, 0.5, 0.9],
            device=flat.device,
        ),
    )
    return {
        "mean": float(flat.mean().item()),
        "p10": float(quantiles[0].item()),
        "median": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def _load_manifest(
    root: Path,
    prompts_path: Path,
) -> tuple[pd.DataFrame, dict[tuple[int, int], list[Path]]]:
    prompts = pd.DataFrame(json.loads(prompts_path.read_text()))
    if len(prompts) != CALIBRATION_PROMPTS:
        raise RuntimeError(f"Expected 32 calibration prompts, found {len(prompts)}")
    records_dir = root / "external_calibration/run/phase0/records"
    records = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            continue
        records.append(record)
    if len(records) != CALIBRATION_PROMPTS:
        raise RuntimeError(f"Expected 32 successful dense jobs, found {len(records)}")
    jobs = pd.DataFrame(records)
    if set(jobs["prompt_id"]) != set(prompts["prompt_id"]):
        raise RuntimeError("Dense calibration jobs do not match frozen prompts")
    prompt_order = prompts.sort_values("selection_rank")["prompt_id"].tolist()
    job_by_prompt = jobs.set_index("prompt_id")["job_id"].to_dict()
    grouped: dict[tuple[int, int], list[Path]] = {}
    for prompt_id in prompt_order:
        job_id = job_by_prompt[prompt_id]
        paths = sorted((root / "raw_dense_mass" / job_id).glob("t*-l*.pt"))
        if len(paths) != 90:
            raise RuntimeError(f"Prompt {prompt_id} has {len(paths)} mass files, not 90")
        for path in paths:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            key = (int(payload["timestep"]), int(payload["layer"]))
            grouped.setdefault(key, []).append(path)
    if len(grouped) != 90:
        raise RuntimeError(f"Expected 90 step/layer units, found {len(grouped)}")
    if any(len(paths) != CALIBRATION_PROMPTS for paths in grouped.values()):
        raise RuntimeError("At least one step/layer unit is missing prompts")
    return prompts, grouped


def _linear_p10(sorted_mass: torch.Tensor) -> torch.Tensor:
    return (sorted_mass[P10_LOWER] * (1.0 - P10_WEIGHT_UPPER) + sorted_mass[P10_LOWER + 1] * P10_WEIGHT_UPPER)


def _mask_overlap(
    prompt_top: torch.Tensor,
    stable_top: torch.Tensor,
    *,
    parent_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_mask = torch.zeros(
        (*prompt_top.shape[:-1], parent_blocks),
        dtype=torch.bool,
        device=prompt_top.device,
    )
    prompt_mask.scatter_(-1, prompt_top, True)
    stable = stable_top.unsqueeze(0).expand(
        prompt_top.shape[0],
        -1,
        -1,
        -1,
    )
    intersection = torch.gather(
        prompt_mask,
        -1,
        stable,
    ).sum(dim=-1)
    width = stable_top.shape[-1]
    overlap = intersection.float() / width
    jaccard = intersection.float() / (2 * width - intersection).clamp_min(1)
    return overlap, jaccard, prompt_mask


def _pair_rows(
    prompt_mask: torch.Tensor,
    prompt_ids: list[str],
    *,
    step: int,
    timestep: int,
    layer: int,
    core_parent_blocks: int,
) -> list[dict[str, Any]]:
    flattened = prompt_mask.float().flatten(1)
    query_units = prompt_mask.shape[1] * prompt_mask.shape[2]
    intersections = torch.matmul(flattened, flattened.T) / query_units
    rows = []
    for left in range(len(prompt_ids)):
        for right in range(left + 1, len(prompt_ids)):
            intersection = float(intersections[left, right].item())
            rows.append({
                "step": step,
                "timestep": timestep,
                "layer": layer,
                "core_parent_blocks": core_parent_blocks,
                "left_prompt_id": prompt_ids[left],
                "right_prompt_id": prompt_ids[right],
                "mean_query_head_overlap_blocks": intersection,
                "mean_query_head_overlap_fraction": (intersection / core_parent_blocks),
                "jaccard_from_mean_intersection": (intersection / (2 * core_parent_blocks - intersection)),
            })
    return rows


def _write_batch(
    writer: pq.ParquetWriter | None,
    path: Path,
    rows: list[dict[str, Any]],
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(
            path,
            table.schema,
            compression="zstd",
            compression_level=9,
        )
    writer.write_table(table)
    return writer


def calibrate(
    root: Path,
    prompts_path: Path,
    *,
    device: torch.device,
) -> None:
    prompts, grouped = _load_manifest(root, prompts_path)
    prompt_ids = prompts.sort_values("selection_rank")["prompt_id"].tolist()
    prompt_hash = hashlib.sha256("\n".join(
        prompts.sort_values("selection_rank")["sha256"].tolist()).encode()).hexdigest()
    timesteps = sorted({key[0] for key in grouped})
    if len(timesteps) != 3:
        raise RuntimeError(f"Expected 3 timesteps, found {timesteps}")
    step_for_timestep = {timestep: step for step, timestep in enumerate(timesteps)}

    external = root / "external_calibration"
    masks_path = external / "stable_core_masks.parquet"
    stats_path = external / "dense_mass_stats.parquet"
    mask_writer: pq.ParquetWriter | None = None
    stats_writer: pq.ParquetWriter | None = None
    stability_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    core25_steps: list[list[torch.Tensor]] = [[] for _ in timesteps]
    core50_steps: list[list[torch.Tensor]] = [[] for _ in timesteps]

    for timestep in timesteps:
        step = step_for_timestep[timestep]
        for layer in range(30):
            records = [
                torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                ) for path in grouped[(timestep, layer)]
            ]
            parent_sizes = records[0]["parent_sizes"].to(
                device=device,
                dtype=torch.int64,
            )
            masses = torch.stack(
                [record["dense_parent_mass"] for record in records],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            native_coverage = torch.stack(
                [record["native_dense_mass_coverage"] for record in records],
                dim=0,
            ).to(device=device)
            fine_coverage = torch.stack(
                [record["fine8_dense_mass_coverage"] for record in records],
                dim=0,
            ).to(device=device)
            native_tokens = torch.stack(
                [record["native_actual_kv_tokens"] for record in records],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            fine_parent_any = torch.stack(
                [unpack_fine_parent_any(record) for record in records],
                dim=0,
            ).to(device=device)
            sorted_mass = torch.sort(masses, dim=0).values
            p10 = _linear_p10(sorted_mass)
            mean = masses.mean(dim=0)
            median = (sorted_mass[15] + sorted_mass[16]) * 0.5
            minimum = sorted_mass[0]
            coefficient_of_variation = (masses.std(dim=0, unbiased=False) / mean.clamp_min(1e-12))
            valid = parent_sizes.gt(0).view(1, 1, -1)
            ranked_score = p10.masked_fill(~valid, -float("inf"))
            top62 = torch.topk(
                ranked_score,
                CORE50,
                dim=-1,
                sorted=True,
            ).indices
            top31 = top62[..., :CORE25]
            if parent_sizes[top62].le(0).any():
                raise RuntimeError("Stable core selected an invalid parent")
            core25_steps[step].append(top31.to(device="cpu", dtype=torch.int16))
            core50_steps[step].append(top62.to(device="cpu", dtype=torch.int16))

            stable_rank = torch.argsort(
                torch.argsort(p10, dim=-1, stable=True),
                dim=-1,
                stable=True,
            ).float()
            prompt_rank = torch.argsort(
                torch.argsort(masses, dim=-1, stable=True),
                dim=-1,
                stable=True,
            ).float()
            rank_center = (masses.shape[-1] - 1) * 0.5
            stable_centered = stable_rank - rank_center
            prompt_centered = prompt_rank - rank_center
            rank_denominator = stable_centered.square().sum(dim=-1).sqrt()
            rank_correlation = (
                (prompt_centered * stable_centered.unsqueeze(0)).sum(dim=-1) /
                (prompt_centered.square().sum(dim=-1).sqrt() * rank_denominator.unsqueeze(0)).clamp_min(1e-12))

            prompt_top31 = torch.topk(
                masses,
                CORE25,
                dim=-1,
            ).indices
            prompt_top62 = torch.topk(
                masses,
                CORE50,
                dim=-1,
            ).indices
            overlap31, jaccard31, mask31 = _mask_overlap(
                prompt_top31,
                top31,
                parent_blocks=masses.shape[-1],
            )
            overlap62, jaccard62, mask62 = _mask_overlap(
                prompt_top62,
                top62,
                parent_blocks=masses.shape[-1],
            )
            pair_rows.extend(
                _pair_rows(
                    mask31,
                    prompt_ids,
                    step=step,
                    timestep=timestep,
                    layer=layer,
                    core_parent_blocks=CORE25,
                ))
            pair_rows.extend(
                _pair_rows(
                    mask62,
                    prompt_ids,
                    step=step,
                    timestep=timestep,
                    layer=layer,
                    core_parent_blocks=CORE50,
                ))

            gather31 = top31.unsqueeze(0).expand(
                CALIBRATION_PROMPTS,
                -1,
                -1,
                -1,
            )
            gather62 = top62.unsqueeze(0).expand_as(prompt_top62)
            coverage25 = torch.gather(
                masses,
                -1,
                gather31,
            ).sum(dim=-1)
            coverage50 = torch.gather(
                masses,
                -1,
                gather62,
            ).sum(dim=-1)
            core25_tokens = parent_sizes[top31].sum(dim=-1).float()
            core50_tokens = parent_sizes[top62].sum(dim=-1).float()
            fine_frequency62 = torch.gather(
                fine_parent_any,
                -1,
                gather62,
            ).float().mean(dim=0)
            selected_p10 = torch.gather(p10, -1, top62)
            selected_mean = torch.gather(mean, -1, top62)
            selected_median = torch.gather(median, -1, top62)
            selected_minimum = torch.gather(minimum, -1, top62)
            selected_cv = torch.gather(
                coefficient_of_variation,
                -1,
                top62,
            )

            mask_batch = []
            stats_batch = []
            for head in range(masses.shape[1]):
                mask_batch.append({
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                    "head": head,
                    "core25_indices": top31[head].cpu().tolist(),
                    "core50_indices": top62[head].cpu().tolist(),
                })
                stats_batch.append({
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                    "head": head,
                    "core50_indices": top62[head].cpu().tolist(),
                    "core50_p10_mass": selected_p10[head].cpu().tolist(),
                    "core50_mean_mass": selected_mean[head].cpu().tolist(),
                    "core50_median_mass": selected_median[head].cpu().tolist(),
                    "core50_minimum_mass": selected_minimum[head].cpu().tolist(),
                    "core50_coefficient_of_variation": selected_cv[head].cpu().tolist(),
                    "core50_fine8_selection_frequency": (fine_frequency62[head].cpu().tolist()),
                })
                stability_rows.append({
                    "step":
                    step,
                    "timestep":
                    timestep,
                    "layer":
                    layer,
                    "head":
                    head,
                    "top31_overlap_mean":
                    float(overlap31[:, head].mean().item()),
                    "top31_jaccard_mean":
                    float(jaccard31[:, head].mean().item()),
                    "top62_overlap_mean":
                    float(overlap62[:, head].mean().item()),
                    "top62_jaccard_mean":
                    float(jaccard62[:, head].mean().item()),
                    "rank_correlation_mean":
                    float(rank_correlation[:, head].mean().item()),
                    "core25_low_quantile_mass_sum_mean":
                    float(torch.gather(
                        p10[head],
                        -1,
                        top31[head],
                    ).sum(dim=-1).mean().item()),
                    "core50_low_quantile_mass_sum_mean":
                    float(selected_p10[head].sum(dim=-1).mean().item()),
                    "core25_fine8_selected_fraction":
                    float(fine_frequency62[head, :, :CORE25].mean().item()),
                    "core50_fine8_selected_fraction":
                    float(fine_frequency62[head].mean().item()),
                    "core25_often_missed_fraction":
                    float(fine_frequency62[head, :, :CORE25].lt(0.5).float().mean().item()),
                    "core50_often_missed_fraction":
                    float(fine_frequency62[head].lt(0.5).float().mean().item()),
                })
            mask_writer = _write_batch(
                mask_writer,
                masks_path,
                mask_batch,
            )
            stats_writer = _write_batch(
                stats_writer,
                stats_path,
                stats_batch,
            )

            for method, values, tokens in [
                ("CalibCore25", coverage25, core25_tokens),
                ("CalibCore50", coverage50, core50_tokens),
                ("NativeVSA64", native_coverage, native_tokens),
                ("Fine8", fine_coverage, native_tokens),
            ]:
                summary = _summarize(values)
                token_denominator = (tokens.unsqueeze(0) if tokens.ndim == values.ndim - 1 else tokens)
                mass_per_token = values / token_denominator.clamp_min(1)
                token_summary = _summarize(mass_per_token)
                coverage_rows.append({
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                    "method": method,
                    "dense_mass_mean": summary["mean"],
                    "dense_mass_p10": summary["p10"],
                    "dense_mass_median": summary["median"],
                    "dense_mass_p90": summary["p90"],
                    "mass_per_valid_token_mean": token_summary["mean"],
                    "mass_per_valid_token_p10": token_summary["p10"],
                    "valid_tokens_mean": float(tokens.mean().item()),
                })
            del (
                records,
                masses,
                native_coverage,
                fine_coverage,
                native_tokens,
                fine_parent_any,
                sorted_mass,
                p10,
                mean,
                median,
                minimum,
                coefficient_of_variation,
                stable_rank,
                prompt_rank,
                prompt_centered,
                rank_correlation,
                mask31,
                mask62,
            )
    if mask_writer is not None:
        mask_writer.close()
    if stats_writer is not None:
        stats_writer.close()

    core25_tensor = torch.stack(
        [torch.stack(layers, dim=0) for layers in core25_steps],
        dim=0,
    )
    core50_tensor = torch.stack(
        [torch.stack(layers, dim=0) for layers in core50_steps],
        dim=0,
    )
    torch.save(
        {
            "format_version": 1,
            "timesteps": timesteps,
            "core25_indices": core25_tensor,
            "core50_indices": core50_tensor,
            "calibration_prompt_hash": prompt_hash,
            "calibration_prompt_ids": prompt_ids,
            "quantile_semantics": QUANTILE_SEMANTICS,
        },
        external / "stable_core_masks.pt",
    )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(external / "core_stability.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(external / "core_mass_coverage.csv", index=False)
    pd.DataFrame(pair_rows).to_parquet(
        external / "prompt_pair_overlap.parquet",
        index=False,
        compression="zstd",
    )
    calibration_summary: dict[str, Any] = {
        "calibration_prompts": CALIBRATION_PROMPTS,
        "prompt_hash": prompt_hash,
        "timesteps": timesteps,
        "layers": 30,
        "heads": int(core25_tensor.shape[2]),
        "query_blocks": int(core25_tensor.shape[3]),
        "key_blocks": 624,
        "core25_parent_blocks": CORE25,
        "core50_parent_blocks": CORE50,
        "quantile_semantics": QUANTILE_SEMANTICS,
        "core25_overlap_mean": float(stability["top31_overlap_mean"].mean()),
        "core50_overlap_mean": float(stability["top62_overlap_mean"].mean()),
        "core25_jaccard_mean": float(stability["top31_jaccard_mean"].mean()),
        "core50_jaccard_mean": float(stability["top62_jaccard_mean"].mean()),
        "rank_correlation_mean": float(stability["rank_correlation_mean"].mean()),
    }
    (external / "calibration_summary.json").write_text(json.dumps(calibration_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(calibration_summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/coretail_vsa"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("/mnt/fastvideo-gpu0/coretail_vsa/"
                     "external_calibration/prompts.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    calibrate(
        args.root.resolve(),
        args.prompts.resolve(),
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
