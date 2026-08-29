from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

NATIVE_K = 125
NUM_BLOCKS = 624
UNIT_COLUMNS = ["step", "layer", "head"]


def solve_exact_multiple_choice(
    errors: np.ndarray,
    candidate_k: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    """Solve the exact-budget multiple-choice problem with vectorized DP."""
    if errors.ndim != 2:
        raise ValueError("errors must be a [units, choices] matrix")
    if errors.shape[1] != candidate_k.size:
        raise ValueError("Candidate count and error columns disagree")
    if not np.all(np.diff(candidate_k) > 0):
        raise ValueError("candidate_k must be strictly increasing")
    units = errors.shape[0]
    minimum = int(candidate_k[0])
    base_budget = units * minimum
    extra_budget = int(budget) - base_budget
    extra_costs = candidate_k.astype(np.int64) - minimum
    if extra_budget < 0 or budget > units * int(candidate_k[-1]):
        raise ValueError("Requested budget is outside the candidate range")

    infinity = np.float64(np.inf)
    previous = np.full(extra_budget + 1, infinity, dtype=np.float64)
    previous[0] = 0.0
    choices = np.full(
        (units, extra_budget + 1),
        255,
        dtype=np.uint8,
    )
    for unit in range(units):
        current = np.full_like(previous, infinity)
        unit_choice = choices[unit]
        for choice, extra_cost in enumerate(extra_costs):
            if extra_cost > extra_budget:
                continue
            candidate = (previous[:extra_budget + 1 - extra_cost] + errors[unit, choice])
            target = current[extra_cost:]
            improve = candidate < target
            target[improve] = candidate[improve]
            unit_choice[extra_cost:][improve] = choice
        previous = current
    if not np.isfinite(previous[extra_budget]):
        raise ValueError(f"Exact budget {budget} is unreachable with K={candidate_k.tolist()}")

    selected = np.empty(units, dtype=np.int64)
    cursor = extra_budget
    for unit in range(units - 1, -1, -1):
        choice = int(choices[unit, cursor])
        if choice == 255:
            raise RuntimeError("Allocation backtracking encountered an unreachable state")
        selected[unit] = int(candidate_k[choice])
        cursor -= int(extra_costs[choice])
    if cursor != 0 or int(selected.sum()) != budget:
        raise RuntimeError("Allocation backtracking violated the exact budget")
    return selected, float(previous[extra_budget])


def solve_greedy_multiple_choice(
    errors: np.ndarray,
    candidate_k: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, float]:
    """Greedy marginal-gain allocation used for auxiliary budget curves."""
    units, choices = errors.shape
    selected_index = np.zeros(units, dtype=np.int64)
    selected_k = np.full(units, int(candidate_k[0]), dtype=np.int64)
    remaining = int(budget - selected_k.sum())
    while remaining > 0:
        best_unit = -1
        best_ratio = -np.inf
        best_cost = 0
        for unit in range(units):
            current = int(selected_index[unit])
            if current + 1 >= choices:
                continue
            cost = int(candidate_k[current + 1] - candidate_k[current])
            if cost > remaining:
                continue
            gain = float(errors[unit, current] - errors[unit, current + 1])
            ratio = gain / cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_unit = unit
                best_cost = cost
        if best_unit < 0:
            break
        selected_index[best_unit] += 1
        selected_k[best_unit] = candidate_k[selected_index[best_unit]]
        remaining -= best_cost
    objective = float(errors[np.arange(units), selected_index].sum())
    return selected_k, objective


def _error_matrix(
    summary: pd.DataFrame,
    unit_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    units = (summary[unit_columns].drop_duplicates().sort_values(unit_columns).reset_index(drop=True))
    pivot = summary.pivot(
        index=unit_columns,
        columns="K",
        values="relative_L2_error_mean",
    ).sort_index()
    candidate_k = pivot.columns.to_numpy(dtype=np.int64)
    return units, pivot.to_numpy(dtype=np.float64), candidate_k


def _table_from_allocation(
    units: pd.DataFrame,
    selected_k: np.ndarray,
    *,
    steps: int,
    layers: int,
    heads: int,
) -> list[list[list[int]]]:
    table = np.zeros((steps, layers, heads), dtype=np.int64)
    for row, exact_k in zip(
            units.itertuples(index=False),
            selected_k,
            strict=True,
    ):
        table[int(row.step), int(row.layer), int(row.head)] = int(exact_k)
    if np.any(table == 0):
        raise ValueError("K table has unassigned entries")
    return table.tolist()


def _plot_allocation(
    allocation: pd.DataFrame,
    output: Path,
) -> None:
    steps = sorted(allocation["step"].unique())
    with PdfPages(output) as pdf:
        figure, axes = plt.subplots(
            len(steps),
            1,
            figsize=(12, 3.2 * len(steps)),
            constrained_layout=True,
        )
        if len(steps) == 1:
            axes = [axes]
        image = None
        for axis, step in zip(axes, steps, strict=True):
            matrix = (allocation.loc[allocation["step"].eq(step)].pivot(index="head",
                                                                        columns="layer",
                                                                        values="allocated_K").sort_index())
            image = axis.imshow(
                matrix.to_numpy(),
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=32,
                vmax=624,
            )
            axis.set_title(f"BR-VSA fixed K allocation — step {step}")
            axis.set_xlabel("Transformer layer")
            axis.set_ylabel("Head")
            axis.set_xticks(np.arange(matrix.shape[1])[::3])
            axis.set_xticklabels(matrix.columns.to_numpy()[::3])
            axis.set_yticks(np.arange(matrix.shape[0]))
            axis.set_yticklabels(matrix.index.to_numpy())
        assert image is not None
        figure.colorbar(
            image,
            ax=axes,
            label="Exact coarse blocks K",
            shrink=0.85,
        )
        pdf.savefig(figure)
        plt.close(figure)


def _plot_budget_curve(
    curve: pd.DataFrame,
    output: Path,
) -> None:
    with PdfPages(output) as pdf:
        figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        axis.plot(
            curve["aggregate_sparsity"] * 100.0,
            curve["mean_predicted_error"],
            marker="o",
            linewidth=2.0,
        )
        for row in curve.itertuples():
            axis.annotate(
                f"K̄={row.mean_K:.1f}",
                (
                    row.aggregate_sparsity * 100.0,
                    row.mean_predicted_error,
                ),
                xytext=(5, 5),
                textcoords="offset points",
            )
        axis.set(
            xlabel="Aggregate exact-block sparsity (%)",
            ylabel="Mean predicted dense-relative L2 error",
            title="Offline BR-VSA error vs global exact-attention budget",
        )
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        pdf.savefig(figure)
        plt.close(figure)


def _loo_stability(sensitivity: pd.DataFrame) -> pd.DataFrame:
    native = sensitivity.loc[sensitivity["K"].eq(NATIVE_K)].copy()
    rows = []
    for held_out in sorted(native["prompt_id"].unique()):
        training = (native.loc[~native["prompt_id"].eq(held_out)].groupby(
            UNIT_COLUMNS,
            as_index=False)["relative_L2_error"].mean().rename(columns={"relative_L2_error": "train_error"}))
        test = (native.loc[native["prompt_id"].eq(held_out), UNIT_COLUMNS +
                           ["relative_L2_error"]].rename(columns={"relative_L2_error": "held_out_error"}))
        paired = training.merge(
            test,
            on=UNIT_COLUMNS,
            validate="one_to_one",
        )
        spearman = float(paired["train_error"].rank(method="average").corr(
            paired["held_out_error"].rank(method="average"),
            method="pearson",
        ))
        count = max(1, math.ceil(len(paired) * 0.20))
        train_top = set(paired.nlargest(count, "train_error").set_index(UNIT_COLUMNS).index)
        held_top = set(paired.nlargest(count, "held_out_error").set_index(UNIT_COLUMNS).index)
        rows.append({
            "held_out_prompt_id": held_out,
            "spearman": spearman,
            "top20_overlap": len(train_top & held_top) / count,
            "unit_count": len(paired),
        })
    return pd.DataFrame(rows)


def build_allocation(
    *,
    summary_path: Path,
    sensitivity_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    allocation_root = output_root / "allocation"
    figures_root = output_root / "figures"
    allocation_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    sensitivity = pd.read_parquet(sensitivity_path)

    units, errors, candidate_k = _error_matrix(summary, UNIT_COLUMNS)
    native_budget = NATIVE_K * len(units)
    selected_k, predicted_total = solve_exact_multiple_choice(
        errors,
        candidate_k,
        native_budget,
    )
    table = _table_from_allocation(
        units,
        selected_k,
        steps=int(units["step"].max()) + 1,
        layers=int(units["layer"].max()) + 1,
        heads=int(units["head"].max()) + 1,
    )
    allocation = units.copy()
    allocation["allocated_K"] = selected_k
    error_lookup = {
        (*key, int(row.K)): float(row.relative_L2_error_mean)
        for key, group in summary.groupby(UNIT_COLUMNS)
        for row in group.itertuples()
    }
    allocation["predicted_error"] = [
        error_lookup[(
            int(row.step),
            int(row.layer),
            int(row.head),
            int(row.allocated_K),
        )] for row in allocation.itertuples()
    ]
    allocation["native_K"] = NATIVE_K
    allocation["delta_K"] = allocation["allocated_K"] - NATIVE_K
    allocation["native_error"] = [
        error_lookup[(
            int(row.step),
            int(row.layer),
            int(row.head),
            NATIVE_K,
        )] for row in allocation.itertuples()
    ]
    allocation["predicted_error_reduction"] = (allocation["native_error"] - allocation["predicted_error"])
    allocation.to_csv(
        allocation_root / "allocation_summary.csv",
        index=False,
    )

    heads = int(units["head"].max()) + 1
    layer_summary = (summary.groupby(["step", "layer", "K"], as_index=False)["relative_L2_error_mean"].sum())
    layer_units, layer_errors, layer_candidates = _error_matrix(
        layer_summary,
        ["step", "layer"],
    )
    layer_budget = native_budget // heads
    layer_selected, layer_predicted_total = solve_exact_multiple_choice(
        layer_errors,
        layer_candidates,
        layer_budget,
    )
    layer_table_array = np.zeros(
        (
            int(units["step"].max()) + 1,
            int(units["layer"].max()) + 1,
            heads,
        ),
        dtype=np.int64,
    )
    for row, exact_k in zip(
            layer_units.itertuples(index=False),
            layer_selected,
            strict=True,
    ):
        layer_table_array[int(row.step), int(row.layer), :] = int(exact_k)

    k_table_payload = {
        "method": "BR-VSA",
        "granularity": "step_layer_head",
        "candidate_K": candidate_k.tolist(),
        "native_K": NATIVE_K,
        "num_blocks": NUM_BLOCKS,
        "unit_count": len(units),
        "native_budget": native_budget,
        "allocated_budget": int(selected_k.sum()),
        "budget_ratio": float(selected_k.sum() / native_budget),
        "aggregate_sparsity": float(1.0 - selected_k.mean() / NUM_BLOCKS),
        "predicted_total_error": predicted_total,
        "k_table": table,
    }
    (allocation_root / "k_table.json").write_text(json.dumps(k_table_payload, indent=2) + "\n")
    layer_payload = {
        **{
            key: value
            for key, value in k_table_payload.items() if key != "k_table"
        },
        "granularity": "step_layer",
        "allocated_budget": int(layer_table_array.sum()),
        "budget_ratio": float(layer_table_array.sum() / native_budget),
        "aggregate_sparsity": float(1.0 - layer_table_array.mean() / NUM_BLOCKS),
        "predicted_total_error": layer_predicted_total,
        "k_table": layer_table_array.tolist(),
    }
    (allocation_root / "layer_only_k_table.json").write_text(json.dumps(layer_payload, indent=2) + "\n")

    validation = {
        "unit_count": len(units),
        "native_K": NATIVE_K,
        "native_budget": native_budget,
        "allocated_budget": int(selected_k.sum()),
        "budget_ratio": float(selected_k.sum() / native_budget),
        "mean_K": float(selected_k.mean()),
        "aggregate_sparsity": float(1.0 - selected_k.mean() / NUM_BLOCKS),
        "budget_constraint_satisfied": bool(selected_k.sum() <= native_budget),
        "budget_equality": bool(selected_k.sum() == native_budget),
        "layer_only_allocated_budget": int(layer_table_array.sum()),
        "layer_only_budget_equality": bool(layer_table_array.sum() == native_budget),
    }
    (allocation_root / "budget_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    distribution = (allocation["allocated_K"].value_counts().sort_index().rename_axis("K").reset_index(name="units"))
    distribution["fraction"] = distribution["units"] / len(allocation)
    distribution.to_csv(
        allocation_root / "k_distribution.csv",
        index=False,
    )
    for group_column in ("step", "layer", "head"):
        grouped = (allocation.groupby([group_column, "allocated_K"]).size().rename("units").reset_index())
        grouped["fraction_within_group"] = grouped.groupby(group_column)["units"].transform(
            lambda values: values / values.sum())
        grouped.to_csv(
            allocation_root / f"k_distribution_by_{group_column}.csv",
            index=False,
        )

    loo = _loo_stability(sensitivity)
    loo.to_csv(
        allocation_root / "loo_stability.csv",
        index=False,
    )

    curve_rows = []
    for target_sparsity in (0.60, 0.70, 0.80, 0.90):
        target_budget = int(round((1.0 - target_sparsity) * NUM_BLOCKS * len(units)))
        if target_sparsity == 0.80:
            curve_k = selected_k
            curve_error = predicted_total
        else:
            curve_k, curve_error = solve_greedy_multiple_choice(
                errors,
                candidate_k,
                target_budget,
            )
        curve_rows.append({
            "target_sparsity": target_sparsity,
            "actual_budget": int(curve_k.sum()),
            "mean_K": float(curve_k.mean()),
            "aggregate_sparsity": float(1.0 - curve_k.mean() / NUM_BLOCKS),
            "mean_predicted_error": float(curve_error / len(units)),
        })
    curve = pd.DataFrame(curve_rows).sort_values("aggregate_sparsity")
    curve.to_csv(
        allocation_root / "error_vs_global_budget.csv",
        index=False,
    )
    _plot_allocation(
        allocation,
        figures_root / "k_allocation_heatmap.pdf",
    )
    _plot_budget_curve(
        curve,
        figures_root / "error_vs_global_budget.pdf",
    )

    head_error = predicted_total / len(units)
    layer_error = layer_predicted_total / len(units)
    report = f"""# BR-VSA Stage 1 Frozen Allocation

## Budget invariant

- Units: {len(units)}
- Native budget: {native_budget}
- BR-VSA budget: {int(selected_k.sum())}
- Budget ratio: {selected_k.sum() / native_budget:.6f}
- Mean K: {selected_k.mean():.6f}
- Aggregate exact-block sparsity: {1.0 - selected_k.mean() / NUM_BLOCKS:.6%}

## Allocation

{distribution.to_markdown(index=False)}

- Units below K125: {int((selected_k < NATIVE_K).sum())}
- Units at K125: {int((selected_k == NATIVE_K).sum())}
- Units above K125: {int((selected_k > NATIVE_K).sum())}

## Offline objective

- Head-wise mean predicted error: {head_error:.6f}
- Layer-only mean predicted error: {layer_error:.6f}
- Relative head-wise improvement over layer-only: {(layer_error - head_error) / layer_error:.2%}
- Median leave-one-prompt-out Spearman: {loo["spearman"].median():.4f}
- Median leave-one-prompt-out top-20% overlap: {loo["top20_overlap"].median():.2%}

The head-wise and layer-only tables were both frozen before generation. No
remaining-prompt quality result was used to select or modify either schedule.
"""
    (allocation_root / "REPORT.md").write_text(report)
    return {
        **validation,
        "headwise_predicted_error": head_error,
        "layer_only_predicted_error": layer_error,
        "loo_median_spearman": float(loo["spearman"].median()),
        "loo_median_top20_overlap": float(loo["top20_overlap"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_allocation(
        summary_path=args.summary,
        sensitivity_path=args.sensitivity,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
