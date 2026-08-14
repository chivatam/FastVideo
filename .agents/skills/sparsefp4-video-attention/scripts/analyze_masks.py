"""Analyze SparseFP4 Phase-1 sparse-attention mask-stability records.

Consumes the JSONL raw records described in
`references/EXPERIMENT_SPEC.md` section 6 and emits aggregate tables, the
SKILL's Figure 1 (mask overlap vs sparsity) and Figure 2 (layer x timestep
Jaccard heatmap), an optional head-level boxplot, and a `summary.md`.

CPU-only and dependency-light: standard library is sufficient. `numpy`,
`pandas`, and `matplotlib` are imported lazily and their absence only disables
figure rendering, never the numeric core. Figures always ship with a CSV of the
exact plotted values so every plotted point is traceable to numbers.

Malformed records are counted and reported, never silently dropped.

Usage:
    python analyze_masks.py --raw artifacts/sparsefp4/raw \\
        --out-tables artifacts/sparsefp4/tables \\
        --out-figures artifacts/sparsefp4/figures
    python analyze_masks.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

REQUIRED_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "seed",
    "layer",
    "head",
    "timestep",
    "block_q",
    "block_k",
    "sparsity",
    "routing_precision",
    "reference_precision",
    "intersection",
    "union",
    "selected_reference",
    "selected_candidate",
    "recall",
    "jaccard",
    "decision_margin_reference",
    "decision_margin_candidate",
    "native_or_simulated",
    "run_id",
    "git_commit",
)

NUMERIC_FIELDS: tuple[str, ...] = (
    "seed",
    "layer",
    "head",
    "timestep",
    "block_q",
    "block_k",
    "sparsity",
    "intersection",
    "union",
    "selected_reference",
    "selected_candidate",
    "recall",
    "jaccard",
)

NULLABLE_NUMERIC_FIELDS: tuple[str, ...] = (
    "decision_margin_reference",
    "decision_margin_candidate",
)

METRICS: tuple[str, ...] = ("jaccard", "recall")
OPTIONAL_MEDIAN_FIELDS: tuple[str, ...] = (
    "decision_margin_reference",
    "decision_margin_candidate",
    "frac_query_blocks_changed",
    "boundary_ties",
    "spearman_rho",
    "sat_frac_q",
)
DEFAULT_FIGURE_SPARSITIES: tuple[float, ...] = (0.80, 0.90)
DEFAULT_HEATMAP_PRECISION = "nvfp4"
DEFAULT_MIN_N = 20
SPARSITY_TOLERANCE = 1e-6
INVARIANT_TOLERANCE = 1e-6
MAX_REPORTED_SKIP_EXAMPLES = 20

EXIT_OK = 0
EXIT_MALFORMED = 2
EXIT_NO_RECORDS = 3


@dataclass
class SkippedLine:
    path: str
    line_no: int
    reason: str
    excerpt: str


@dataclass
class LoadResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[SkippedLine] = field(default_factory=list)
    invariant_violations: list[SkippedLine] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    total_lines: int = 0
    filtered_out: int = 0

    @property
    def skip_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.skipped:
            key = item.reason.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass
class Stats:
    n: int
    median: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    mean: float | None
    p10: float | None
    p90: float | None
    minimum: float | None
    maximum: float | None

    def as_row(self, prefix: str) -> dict[str, Any]:
        return {
            f"n_{prefix}": self.n,
            f"{prefix}_median": _round(self.median),
            f"{prefix}_q1": _round(self.q1),
            f"{prefix}_q3": _round(self.q3),
            f"{prefix}_iqr": _round(self.iqr),
            f"{prefix}_mean": _round(self.mean),
            f"{prefix}_p10": _round(self.p10),
            f"{prefix}_p90": _round(self.p90),
            f"{prefix}_min": _round(self.minimum),
            f"{prefix}_max": _round(self.maximum),
        }


def _round(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(value, digits)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[int(position)])
    weight = position - low
    return float(sorted_values[low]) * (1.0 - weight) + float(sorted_values[high]) * weight


def compute_stats(values: Sequence[float]) -> Stats:
    if not values:
        return Stats(0, None, None, None, None, None, None, None, None, None)
    ordered = sorted(float(value) for value in values)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    return Stats(
        n=len(ordered),
        median=_percentile(ordered, 0.5),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        mean=statistics.fmean(ordered),
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def iter_raw_files(raw: Path) -> list[Path]:
    if raw.is_file():
        return [raw]
    if not raw.is_dir():
        raise FileNotFoundError(f"--raw path does not exist: {raw}")
    found = sorted({*raw.rglob("*.jsonl"), *raw.rglob("*.jsonl.gz")})
    if not found:
        raise FileNotFoundError(f"no *.jsonl or *.jsonl.gz files found under {raw}")
    return found


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_record(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return [f"not_an_object: got {type(obj).__name__}"]
    errors: list[str] = []
    missing = [name for name in REQUIRED_FIELDS if name not in obj]
    if missing:
        errors.append(f"missing_fields: {', '.join(missing)}")
    for name in NUMERIC_FIELDS:
        if name not in obj:
            continue
        value = obj[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"non_numeric_field: {name}={value!r}")
    for name in NULLABLE_NUMERIC_FIELDS:
        value = obj.get(name, None)
        if name in obj and value is not None and not isinstance(value, (int, float)):
            errors.append(f"non_numeric_field: {name}={value!r}")
    for name in ("routing_precision", "reference_precision", "native_or_simulated", "run_id", "git_commit"):
        if name in obj and not isinstance(obj[name], str):
            errors.append(f"non_string_field: {name}={obj.get(name)!r}")
    if isinstance(obj.get("native_or_simulated"), str) and obj["native_or_simulated"] not in ("native", "simulated"):
        errors.append(f"bad_enum: native_or_simulated={obj['native_or_simulated']!r}")
    if isinstance(obj.get("sparsity"), (int, float)) and not 0.0 <= float(obj["sparsity"]) < 1.0:
        errors.append(f"out_of_range: sparsity={obj['sparsity']!r}")
    return errors


def check_invariants(record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    intersection = float(record["intersection"])
    union = float(record["union"])
    selected_reference = float(record["selected_reference"])
    selected_candidate = float(record["selected_candidate"])
    if selected_reference != selected_candidate:
        problems.append("unequal_budget: selected_reference != selected_candidate")
    if union > 0.0 and abs(union - (selected_reference + selected_candidate - intersection)) > INVARIANT_TOLERANCE:
        problems.append("union_mismatch: union != sel_ref + sel_cand - intersection")
    if selected_reference > 0.0:
        expected_recall = intersection / selected_reference
        if abs(float(record["recall"]) - expected_recall) > INVARIANT_TOLERANCE:
            problems.append("recall_mismatch: recall != intersection / selected_reference")
    if union > 0.0 and abs(float(record["jaccard"]) - intersection / union) > INVARIANT_TOLERANCE:
        problems.append("jaccard_mismatch: jaccard != intersection / union")
    if record["reference_precision"] != "bf16":
        problems.append(f"bad_reference_precision: {record['reference_precision']!r}")
    if record["routing_precision"] == record["reference_precision"] and float(record["jaccard"]) < 1.0:
        problems.append("null_control_failed: bf16 vs bf16 must give jaccard == 1.0")
    return problems


def parse_filters(raw_filters: Sequence[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            raise SystemExit(f"--filter expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--filter has an empty key: {item!r}")
        filters[key] = value.strip()
    return filters


def record_matches(record: dict[str, Any], filters: dict[str, str]) -> bool:
    for key, wanted in filters.items():
        if key not in record:
            return False
        actual = record[key]
        if isinstance(actual, float):
            try:
                if abs(actual - float(wanted)) > SPARSITY_TOLERANCE:
                    return False
            except ValueError:
                return False
        elif str(actual) != wanted:
            return False
    return True


def load_records(raw: Path, filters: dict[str, str]) -> LoadResult:
    result = LoadResult(files=iter_raw_files(raw))
    for path in result.files:
        with _open_text(path) as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                result.total_lines += 1
                excerpt = stripped[:200]
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    result.skipped.append(SkippedLine(str(path), line_no, f"json_decode_error: {exc}", excerpt))
                    continue
                errors = validate_record(obj)
                if errors:
                    result.skipped.append(SkippedLine(str(path), line_no, "; ".join(errors), excerpt))
                    continue
                problems = check_invariants(obj)
                if problems:
                    result.invariant_violations.append(
                        SkippedLine(str(path), line_no, "; ".join(problems), excerpt))
                if not record_matches(obj, filters):
                    result.filtered_out += 1
                    continue
                result.records.append(obj)
    return result


def group_records(records: Sequence[dict[str, Any]],
                  key_fn: Callable[[dict[str, Any]], tuple[Any, ...]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)
    return grouped


def aggregate(records: Sequence[dict[str, Any]], key_fields: Sequence[str], min_n: int) -> list[dict[str, Any]]:
    grouped = group_records(records, lambda rec: tuple(rec[name] for name in key_fields))
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_sort_key):
        cell = grouped[key]
        row: dict[str, Any] = dict(zip(key_fields, key))
        if "sparsity" in row:
            row["retained_fraction"] = _round(1.0 - float(row["sparsity"]))
        row["n"] = len(cell)
        row["insufficient_n"] = len(cell) < min_n
        row["native_or_simulated"] = "/".join(sorted({str(rec["native_or_simulated"]) for rec in cell}))
        for metric in METRICS:
            row.update(compute_stats([float(rec[metric]) for rec in cell]).as_row(metric))
        for name in OPTIONAL_MEDIAN_FIELDS:
            values = [float(rec[name]) for rec in cell if isinstance(rec.get(name), (int, float))]
            row[f"{name}_median"] = _round(compute_stats(values).median)
            row[f"n_{name}"] = len(values)
        rows.append(row)
    return rows


def _sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple((0, value, "") if isinstance(value, (int, float)) else (1, 0.0, str(value)) for value in key)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _union_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_markdown(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) if columns else _union_fieldnames(rows)
    lines = ["| " + " | ".join(fieldnames) + " |", "|" + "|".join("---" for _ in fieldnames) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _union_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    return fieldnames


def _load_pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except Exception:
        return None
    return pyplot


def _load_numpy() -> Any | None:
    try:
        import numpy
    except Exception:
        return None
    return numpy


def figure1_values(records: Sequence[dict[str, Any]], min_n: int) -> list[dict[str, Any]]:
    rows = aggregate(records, ("sparsity", "routing_precision"), min_n)
    return [row for row in rows if row["routing_precision"] != "bf16"] or rows


def render_figure1(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "jaccard") -> list[Path]:
    csv_path = out_dir / "fig1_mask_overlap_vs_sparsity.csv"
    write_csv(csv_path, rows)
    written = [csv_path]
    pyplot = _load_pyplot()
    if pyplot is None:
        return written
    # Two panels on the same data: the left one keeps the honest full [0, 1]
    # axis, the right one zooms to the observed range. When overlap stays above
    # ~0.95 everywhere, the full-scale panel alone is a flat line that hides the
    # ordering between arms — and the ordering is the result. Showing both keeps
    # the effect size visually honest while still being readable.
    figure, (axis_full, axis_zoom) = pyplot.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    by_precision = group_records(rows, lambda row: (row["routing_precision"], ))
    observed: list[float] = []
    for (precision, ), series in sorted(by_precision.items()):
        series = sorted(series, key=lambda row: float(row["sparsity"]))
        xs = [float(row["sparsity"]) for row in series]
        medians = [float(row[f"{metric}_median"]) for row in series]
        lows = [float(row[f"{metric}_q1"]) for row in series]
        highs = [float(row[f"{metric}_q3"]) for row in series]
        p10s = [float(row[f"{metric}_p10"]) for row in series]
        observed += lows + highs + p10s
        label = f"{precision} (n={sum(int(row['n']) for row in series)})"
        for axis in (axis_full, axis_zoom):
            axis.plot(xs, medians, marker="o", label=label)
            axis.fill_between(xs, lows, highs, alpha=0.20, linewidth=0)
        axis_zoom.plot(xs, p10s, marker="", linestyle=":", linewidth=1.0)
    axis_full.set_ylim(0.0, 1.02)
    axis_full.set_title("full scale")
    if observed:
        margin = max(0.005, 0.08 * (1.0 - min(observed)))
        axis_zoom.set_ylim(min(observed) - margin, 1.0 + margin * 0.25)
    axis_zoom.set_title("zoomed (dotted = p10)")
    for axis in (axis_full, axis_zoom):
        axis.set_xlabel("sparsity (fraction of key blocks skipped)")
        axis.set_ylabel(f"mask {metric} vs BF16 (median, IQR band)")
        axis.grid(True, alpha=0.30)
        axis.legend(fontsize="small")
    figure.suptitle("Figure 1 - mask overlap vs sparsity")
    figure.tight_layout()
    png_path = out_dir / "fig1_mask_overlap_vs_sparsity.png"
    figure.savefig(png_path)
    pyplot.close(figure)
    written.append(png_path)
    return written


def figure2_values(records: Sequence[dict[str, Any]], sparsity: float, precision: str,
                   min_n: int) -> list[dict[str, Any]]:
    subset = [rec for rec in records
              if abs(float(rec["sparsity"]) - sparsity) <= SPARSITY_TOLERANCE
              and rec["routing_precision"] == precision]
    return aggregate(subset, ("layer", "timestep"), min_n)


def render_figure2(rows: Sequence[dict[str, Any]], out_dir: Path, sparsity: float, precision: str,
                   metric: str = "jaccard") -> list[Path]:
    stem = f"fig2_layer_timestep_{metric}_s{sparsity:.2f}_{precision}"
    csv_path = out_dir / f"{stem}.csv"
    write_csv(csv_path, rows)
    written = [csv_path]
    pyplot = _load_pyplot()
    if pyplot is None or not rows:
        return written
    layers = sorted({int(row["layer"]) for row in rows})
    timesteps = sorted({int(row["timestep"]) for row in rows})
    lookup = {(int(row["layer"]), int(row["timestep"])): row[f"{metric}_median"] for row in rows}
    grid = [[_grid_value(lookup.get((layer, timestep))) for timestep in timesteps] for layer in layers]
    finite = [value for row in grid for value in row if not math.isnan(value)]
    # Data-driven colour range: with overlap concentrated in [0.9, 1.0] a fixed
    # [0, 1] scale renders the whole heatmap one flat colour and hides exactly the
    # layer/timestep structure the panel exists to show. The colourbar carries the
    # absolute values, and the range is stated in the title.
    low = min(finite) if finite else 0.0
    high = max(finite) if finite else 1.0
    figure, axis = pyplot.subplots(figsize=(max(5.5, 0.35 * len(timesteps) + 3.0),
                                            max(3.2, 0.25 * len(layers) + 1.8)), dpi=150)
    image = axis.imshow(grid, aspect="auto", origin="lower", vmin=low, vmax=high, cmap="viridis")
    step = max(1, len(timesteps) // 25)
    axis.set_xticks(range(0, len(timesteps), step), [str(timesteps[i]) for i in range(0, len(timesteps), step)],
                    fontsize="x-small", rotation=90)
    axis.set_yticks(range(len(layers)), [str(value) for value in layers], fontsize="x-small")
    axis.set_xlabel("timestep")
    axis.set_ylabel("layer")
    axis.set_title(f"Figure 2 - BF16<->{precision} mask {metric}, sparsity {sparsity:.2f} "
                   f"(n={sum(int(row['n']) for row in rows)}, colour range {low:.3f}-{high:.3f})", fontsize="small")
    figure.colorbar(image, ax=axis, label=f"median {metric}")
    figure.tight_layout()
    png_path = out_dir / f"{stem}.png"
    figure.savefig(png_path)
    pyplot.close(figure)
    written.append(png_path)
    return written


def _grid_value(value: Any) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def figure3_values(records: Sequence[dict[str, Any]], sparsity: float, precision: str,
                   min_n: int) -> list[dict[str, Any]]:
    subset = [rec for rec in records
              if abs(float(rec["sparsity"]) - sparsity) <= SPARSITY_TOLERANCE
              and rec["routing_precision"] == precision]
    return aggregate(subset, ("head", ), min_n)


def render_figure4(rows: Sequence[dict[str, Any]], out_dir: Path, metric: str = "jaccard") -> list[Path]:
    """Timestep trend: one line per (routing precision, sparsity).

    The 2-step smoke run could not see this axis at all; diffusion timesteps are
    not interchangeable, so the trend is reported explicitly rather than folded
    into a single aggregate.
    """
    csv_path = out_dir / "fig4_overlap_vs_timestep.csv"
    write_csv(csv_path, rows)
    written = [csv_path]
    pyplot = _load_pyplot()
    if pyplot is None or not rows:
        return written
    figure, axis = pyplot.subplots(figsize=(7.0, 4.2), dpi=150)
    by_series = group_records(rows, lambda row: (row["routing_precision"], float(row["sparsity"])))
    for (precision, sparsity), series in sorted(by_series.items()):
        series = sorted(series, key=lambda row: int(row["timestep"]))
        xs = [int(row["timestep"]) for row in series]
        medians = [float(row[f"{metric}_median"]) for row in series]
        lows = [float(row[f"{metric}_q1"]) for row in series]
        highs = [float(row[f"{metric}_q3"]) for row in series]
        axis.plot(xs, medians, marker="", linewidth=1.4, label=f"{precision} @ sp={sparsity:.2f}")
        axis.fill_between(xs, lows, highs, alpha=0.15, linewidth=0)
    axis.set_xlabel("denoising step index (0 = highest noise)")
    axis.set_ylabel(f"mask {metric} vs BF16 (median, IQR band)")
    axis.set_title("Figure 4 - mask overlap vs denoising timestep")
    axis.grid(True, alpha=0.30)
    axis.legend(fontsize="x-small", ncol=2)
    figure.tight_layout()
    png_path = out_dir / "fig4_overlap_vs_timestep.png"
    figure.savefig(png_path)
    pyplot.close(figure)
    written.append(png_path)
    return written


def render_figure3(records: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]], out_dir: Path,
                   sparsity: float, precision: str, metric: str = "jaccard") -> list[Path]:
    stem = f"fig3_head_{metric}_box_s{sparsity:.2f}_{precision}"
    csv_path = out_dir / f"{stem}.csv"
    write_csv(csv_path, rows)
    written = [csv_path]
    pyplot = _load_pyplot()
    if pyplot is None or not rows:
        return written
    subset = [rec for rec in records
              if abs(float(rec["sparsity"]) - sparsity) <= SPARSITY_TOLERANCE
              and rec["routing_precision"] == precision]
    by_head = group_records(subset, lambda rec: (int(rec["head"]), ))
    heads = sorted(by_head)
    data = [[float(rec[metric]) for rec in by_head[head]] for head in heads]
    figure, axis = pyplot.subplots(figsize=(max(4.0, 0.4 * len(heads) + 2.0), 3.6), dpi=150)
    axis.boxplot(data, tick_labels=[str(head[0]) for head in heads], showfliers=False)
    axis.set_xlabel("head")
    axis.set_ylabel(f"mask {metric} vs BF16")
    axis.set_title(f"Figure 3 - per-head {metric}, BF16<->{precision}, sparsity {sparsity:.2f}")
    axis.grid(True, axis="y", alpha=0.30)
    figure.tight_layout()
    png_path = out_dir / f"{stem}.png"
    figure.savefig(png_path)
    pyplot.close(figure)
    written.append(png_path)
    return written


RANKING_TOP_N = 8


def _quotable(rows: Sequence[dict[str, Any]], min_n: int) -> list[dict[str, Any]]:
    del min_n  # the insufficient_n flag was already computed against it
    return [row for row in rows if not row["insufficient_n"] and row["jaccard_median"] != ""]


def _timestep_trend_section(timestep_rows: Sequence[dict[str, Any]], min_n: int) -> list[str]:
    """Overlap as an explicit function of denoising step, per (precision, sparsity).

    A 2-step sample cannot distinguish "uniformly stable" from "stable on average
    but collapsing in one timestep band", which is the question the trend answers.
    """
    quotable = _quotable(timestep_rows, min_n)
    if not quotable:
        return []
    lines = ["", "## Timestep trend (H2)", ""]
    by_series = group_records(quotable, lambda row: (row["routing_precision"], float(row["sparsity"])))
    columns = ["routing_precision", "sparsity", "n_timesteps", "first_step", "first_jaccard", "last_step",
               "last_jaccard", "worst_step", "worst_jaccard", "best_step", "best_jaccard", "spread", "n_total"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for (precision, sparsity), series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda row: int(row["timestep"]))
        worst = min(ordered, key=lambda row: float(row["jaccard_median"]))
        best = max(ordered, key=lambda row: float(row["jaccard_median"]))
        values = [
            precision,
            f"{sparsity:.2f}",
            len(ordered),
            ordered[0]["timestep"],
            ordered[0]["jaccard_median"],
            ordered[-1]["timestep"],
            ordered[-1]["jaccard_median"],
            worst["timestep"],
            worst["jaccard_median"],
            best["timestep"],
            best["jaccard_median"],
            _round(float(best["jaccard_median"]) - float(worst["jaccard_median"])),
            sum(int(row["n"]) for row in ordered),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return lines


def _ranking_section(title: str, rows: Sequence[dict[str, Any]], key: str, min_n: int) -> list[str]:
    """Worst and best cells by median Jaccard, so H2 is judged on ranked evidence."""
    quotable = _quotable(rows, min_n)
    if not quotable:
        return []
    lines = ["", f"## {title}", ""]
    by_series = group_records(quotable, lambda row: (row["routing_precision"], float(row["sparsity"])))
    for (precision, sparsity), series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda row: float(row["jaccard_median"]))
        lines.append(f"### {precision} @ sparsity {sparsity:.2f} (cells={len(ordered)})")
        lines.append("")
        for label, subset in (("most affected", ordered[:RANKING_TOP_N]),
                              ("least affected", list(reversed(ordered[-RANKING_TOP_N:])))):
            rendered = ", ".join(f"{_ranking_label(row, key)}={row['jaccard_median']} (n={row['n']})"
                                 for row in subset)
            lines.append(f"- {label}: {rendered}")
        lines.append("")
    return lines


def _ranking_label(row: dict[str, Any], key: str) -> str:
    if key == "layer_head":
        return f"L{row['layer']}H{row['head']}"
    return f"{key}{row[key]}"


def build_summary(load: LoadResult, sparsity_rows: Sequence[dict[str, Any]],
                  layer_timestep_rows: Sequence[dict[str, Any]], head_rows: Sequence[dict[str, Any]],
                  min_n: int, affected_threshold: float, artifacts: Sequence[Path],
                  timestep_rows: Sequence[dict[str, Any]] = (),
                  layer_rows: Sequence[dict[str, Any]] = (),
                  layer_head_rows: Sequence[dict[str, Any]] = ()) -> str:
    lines: list[str] = [
        "# SparseFP4 mask-stability analysis summary",
        "",
        "Generated by `scripts/analyze_masks.py`. Every aggregate below carries `n=`.",
        "Recall and Jaccard are one measurement, not two: for equal-sized top-k masks",
        "precision == recall and `jaccard = recall / (2 - recall)`. Do not cite them as",
        "independent evidence.",
        "",
        "## Input accounting",
        "",
        f"- raw files read: {len(load.files)}",
        f"- non-empty lines seen: {load.total_lines}",
        f"- records accepted: {len(load.records)}",
        f"- records excluded by --filter/--sparsity: {load.filtered_out}",
        f"- malformed lines skipped: {len(load.skipped)}",
        f"- records with invariant violations (kept, flagged): {len(load.invariant_violations)}",
    ]
    if load.skipped:
        lines.append("")
        lines.append("### Skipped-line reasons")
        lines.append("")
        for reason, count in sorted(load.skip_reason_counts.items(), key=lambda item: -item[1]):
            lines.append(f"- `{reason}`: {count}")
    if load.invariant_violations:
        lines.append("")
        lines.append("### Invariant violations")
        lines.append("")
        for item in load.invariant_violations[:MAX_REPORTED_SKIP_EXAMPLES]:
            lines.append(f"- `{item.path}`:{item.line_no} - {item.reason}")
        if len(load.invariant_violations) > MAX_REPORTED_SKIP_EXAMPLES:
            lines.append(f"- ... and {len(load.invariant_violations) - MAX_REPORTED_SKIP_EXAMPLES} more")

    lines += ["", "## Mask overlap by sparsity x routing precision", ""]
    headline_columns = ["sparsity", "retained_fraction", "routing_precision", "native_or_simulated",
                        "jaccard_median", "jaccard_iqr", "recall_median", "n", "insufficient_n"]
    lines.append("| " + " | ".join(headline_columns) + " |")
    lines.append("|" + "|".join("---" for _ in headline_columns) + "|")
    for row in sparsity_rows:
        lines.append("| " + " | ".join(str(row.get(name, "")) for name in headline_columns) + " |")

    affected = [row for row in layer_timestep_rows
                if not row["insufficient_n"] and row["jaccard_median"] != ""
                and float(row["jaccard_median"]) < affected_threshold]
    eligible = [row for row in layer_timestep_rows if not row["insufficient_n"]]
    lines += [
        "",
        "## Localization (H2)",
        "",
        f"- `(layer, timestep, sparsity, routing_precision)` cells total: {len(layer_timestep_rows)}",
        f"- cells with n >= {min_n} (eligible to be quoted): {len(eligible)}",
        f"- cells below n = {min_n} (flagged `insufficient_n`, excluded from claims): "
        f"{len(layer_timestep_rows) - len(eligible)}",
        f"- eligible cells that are affected (median jaccard < {affected_threshold:.2f}): {len(affected)}",
    ]
    ranked = eligible or layer_timestep_rows
    suffix = "" if eligible else "  (NOT quotable: below min n)"
    if ranked:
        scored = [row for row in ranked if row["jaccard_median"] != ""]
        worst = min(scored, key=lambda row: float(row["jaccard_median"]))
        best = max(scored, key=lambda row: float(row["jaccard_median"]))
        lines.append(f"- most affected: layer={worst['layer']} timestep={worst['timestep']} "
                     f"sparsity={worst['sparsity']} precision={worst['routing_precision']} "
                     f"median jaccard={worst['jaccard_median']} n={worst['n']}{suffix}")
        lines.append(f"- least affected: layer={best['layer']} timestep={best['timestep']} "
                     f"sparsity={best['sparsity']} precision={best['routing_precision']} "
                     f"median jaccard={best['jaccard_median']} n={best['n']}{suffix}")
    head_eligible = [row for row in head_rows if not row["insufficient_n"]]
    if head_eligible:
        worst_head = min(head_eligible, key=lambda row: float(row["jaccard_median"]))
        lines.append(f"- most affected head: head={worst_head['head']} sparsity={worst_head['sparsity']} "
                     f"precision={worst_head['routing_precision']} "
                     f"median jaccard={worst_head['jaccard_median']} n={worst_head['n']}")

    lines += _timestep_trend_section(timestep_rows, min_n)
    lines += _ranking_section("Per-layer ranking (H2)", layer_rows, "layer", min_n)
    lines += _ranking_section("Per-head ranking (H2)", head_rows, "head", min_n)
    lines += _ranking_section("Per-(layer, head) ranking (H2)", layer_head_rows, "layer_head", min_n)

    lines += ["", "## Artifacts", ""]
    for path in artifacts:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## Caveats carried from the spec",
        "",
        "- Arms labeled `simulated` are fake-quantized and must never appear in a latency table.",
        "- Cells flagged `insufficient_n` are excluded from headline claims.",
        "- Every figure ships a CSV of its exact plotted values next to the PNG.",
        "",
    ]
    return "\n".join(lines)


@dataclass
class AnalysisOutputs:
    tables: list[Path] = field(default_factory=list)
    figures: list[Path] = field(default_factory=list)
    summary: Path | None = None


def run_analysis(raw: Path, out_tables: Path, out_figures: Path, filters: dict[str, str],
                 figure_sparsities: Sequence[float], table_format: str, min_n: int,
                 affected_threshold: float, heatmap_precision: str, head_figure: bool) -> AnalysisOutputs:
    load = load_records(raw, filters)
    if not load.records:
        _report_load(load)
        raise SystemExit(f"no valid records after loading {raw} (see reasons above)")

    outputs = AnalysisOutputs()
    sparsity_rows = aggregate(load.records, ("sparsity", "routing_precision"), min_n)
    layer_timestep_rows = aggregate(load.records, ("layer", "timestep", "sparsity", "routing_precision"), min_n)
    head_rows = aggregate(load.records, ("head", "sparsity", "routing_precision"), min_n)
    layer_rows = aggregate(load.records, ("layer", "sparsity", "routing_precision"), min_n)
    timestep_rows = aggregate(load.records, ("timestep", "sparsity", "routing_precision"), min_n)
    layer_head_rows = aggregate(load.records, ("layer", "head", "sparsity", "routing_precision"), min_n)
    prompt_rows = aggregate(load.records, ("prompt_id", "sparsity", "routing_precision"), min_n)
    cfg_rows = (aggregate(load.records, ("cfg_branch", "sparsity", "routing_precision"), min_n)
                if any("cfg_branch" in rec for rec in load.records) else [])

    table_specs = [
        ("agg_by_sparsity_precision", sparsity_rows),
        ("agg_by_layer_timestep", layer_timestep_rows),
        ("agg_by_layer", layer_rows),
        ("agg_by_head", head_rows),
        ("agg_by_timestep", timestep_rows),
        ("agg_by_layer_head", layer_head_rows),
        ("agg_by_prompt", prompt_rows),
    ]
    if cfg_rows:
        table_specs.append(("agg_by_cfg_branch", cfg_rows))
    for name, rows in table_specs:
        if table_format in ("csv", "both"):
            path = out_tables / f"{name}.csv"
            write_csv(path, rows)
            outputs.tables.append(path)
        if table_format in ("markdown", "both"):
            path = out_tables / f"{name}.md"
            write_markdown(path, rows)
            outputs.tables.append(path)

    outputs.figures += render_figure1(figure1_values(load.records, min_n), out_figures)
    outputs.figures += render_figure4([row for row in timestep_rows if row["routing_precision"] != "bf16"],
                                      out_figures)
    available_sparsities = sorted({float(rec["sparsity"]) for rec in load.records})
    for sparsity in figure_sparsities:
        if not any(abs(sparsity - value) <= SPARSITY_TOLERANCE for value in available_sparsities):
            print(f"warning: no records at sparsity={sparsity:.2f}; skipping Figure 2 panel", file=sys.stderr)
            continue
        rows = figure2_values(load.records, sparsity, heatmap_precision, min_n)
        outputs.figures += render_figure2(rows, out_figures, sparsity, heatmap_precision)
        if head_figure:
            head_panel = figure3_values(load.records, sparsity, heatmap_precision, min_n)
            outputs.figures += render_figure3(load.records, head_panel, out_figures, sparsity, heatmap_precision)

    summary_path = out_tables / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        build_summary(load, sparsity_rows,
                      [row for row in layer_timestep_rows if row["routing_precision"] != "bf16"],
                      [row for row in head_rows if row["routing_precision"] != "bf16"], min_n, affected_threshold,
                      outputs.tables + outputs.figures,
                      timestep_rows=[row for row in timestep_rows if row["routing_precision"] != "bf16"],
                      layer_rows=[row for row in layer_rows if row["routing_precision"] != "bf16"],
                      layer_head_rows=[row for row in layer_head_rows if row["routing_precision"] != "bf16"]),
        encoding="utf-8")
    outputs.summary = summary_path

    _report_load(load)
    print(f"wrote {len(outputs.tables)} table file(s) to {out_tables}")
    print(f"wrote {len(outputs.figures)} figure file(s) to {out_figures}")
    print(f"wrote summary to {summary_path}")
    if _load_pyplot() is None:
        print("note: matplotlib unavailable; wrote figure value CSVs only", file=sys.stderr)
    return outputs


def _report_load(load: LoadResult) -> None:
    print(f"read {len(load.files)} file(s), {load.total_lines} non-empty line(s), "
          f"accepted {len(load.records)} record(s), filtered out {load.filtered_out}")
    if load.skipped:
        print(f"SKIPPED {len(load.skipped)} malformed line(s):", file=sys.stderr)
        for reason, count in sorted(load.skip_reason_counts.items(), key=lambda item: -item[1]):
            print(f"  {reason}: {count}", file=sys.stderr)
        for item in load.skipped[:MAX_REPORTED_SKIP_EXAMPLES]:
            print(f"  {item.path}:{item.line_no}: {item.reason} | {item.excerpt}", file=sys.stderr)
        if len(load.skipped) > MAX_REPORTED_SKIP_EXAMPLES:
            print(f"  ... and {len(load.skipped) - MAX_REPORTED_SKIP_EXAMPLES} more", file=sys.stderr)
    if load.invariant_violations:
        print(f"WARNING: {len(load.invariant_violations)} record(s) violate schema invariants "
              f"(kept and flagged in summary.md)", file=sys.stderr)


def _synthetic_record(prompt_id: str, seed: int, layer: int, head: int, timestep: int, sparsity: float,
                      precision: str, rng: random.Random, run_id: str) -> dict[str, Any]:
    n_key_blocks = 64
    retained = 1.0 - sparsity
    k = max(1, math.ceil(retained * n_key_blocks))
    n_query_blocks = 8
    budget = k * n_query_blocks
    if precision == "bf16":
        intersection = budget
    else:
        instability = 0.02 if precision == "fp8_e4m3" else 0.10
        instability *= 1.0 + 2.0 * sparsity
        instability *= 1.8 if layer >= 2 and timestep <= 1 else 1.0
        instability *= 1.5 if head == 0 else 1.0
        lost = min(budget, sum(1 for _ in range(budget) if rng.random() < instability))
        intersection = budget - lost
    union = 2 * budget - intersection
    return {
        "prompt_id": prompt_id,
        "seed": seed,
        "layer": layer,
        "head": head,
        "timestep": timestep,
        "block_q": 128,
        "block_k": 64,
        "sparsity": sparsity,
        "routing_precision": precision,
        "reference_precision": "bf16",
        "intersection": intersection,
        "union": union,
        "selected_reference": budget,
        "selected_candidate": budget,
        "recall": intersection / budget,
        "jaccard": intersection / union,
        "decision_margin_reference": round(rng.uniform(0.001, 0.2), 6),
        "decision_margin_candidate": round(rng.uniform(0.001, 0.2), 6),
        "native_or_simulated": "simulated",
        "run_id": run_id,
        "git_commit": "0" * 40,
        "n_q_blocks": n_query_blocks,
        "n_k_blocks": n_key_blocks,
        "k_per_query_block": k,
    }


def write_synthetic_records(raw_dir: Path, run_id: str, seed: int = 7) -> Path:
    rng = random.Random(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / "synthetic_shard0.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for prompt_index in range(2):
            for layer in range(4):
                for head in range(3):
                    for timestep in range(4):
                        for sparsity in (0.50, 0.70, 0.80, 0.90, 0.95):
                            for precision in ("bf16", "fp8_e4m3", "nvfp4"):
                                record = _synthetic_record(f"p{prompt_index + 1:02d}", 1234, layer, head, timestep,
                                                           sparsity, precision, rng, run_id)
                                handle.write(json.dumps(record) + "\n")
        handle.write("{not json at all\n")
        handle.write(json.dumps({"prompt_id": "p01", "seed": 1234}) + "\n")
        bad = _synthetic_record("p01", 1234, 0, 0, 0, 0.90, "nvfp4", rng, run_id)
        bad["native_or_simulated"] = "sort_of_native"
        handle.write(json.dumps(bad) + "\n")
    return path


def self_test(keep_dir: Path | None = None) -> int:
    numpy_module = _load_numpy()
    pyplot = _load_pyplot()
    print(f"self-test: python {sys.version.split()[0]}, "
          f"numpy={'yes' if numpy_module else 'no'}, matplotlib={'yes' if pyplot else 'no'}")

    with tempfile.TemporaryDirectory(prefix="sparsefp4-selftest-") as tmp:
        root = Path(keep_dir) if keep_dir else Path(tmp)
        raw_dir = root / "raw" / "20260101-000000-deadbee-selftest"
        tables_dir = root / "tables"
        figures_dir = root / "figures"
        write_synthetic_records(raw_dir, run_id=raw_dir.name)

        outputs = run_analysis(
            raw=root / "raw",
            out_tables=tables_dir,
            out_figures=figures_dir,
            filters={},
            figure_sparsities=DEFAULT_FIGURE_SPARSITIES,
            table_format="both",
            min_n=DEFAULT_MIN_N,
            affected_threshold=0.90,
            heatmap_precision=DEFAULT_HEATMAP_PRECISION,
            head_figure=True,
        )

        failures: list[str] = []
        expected_tables = ("agg_by_sparsity_precision.csv", "agg_by_sparsity_precision.md",
                           "agg_by_layer_timestep.csv", "agg_by_layer.csv", "agg_by_head.csv",
                           "agg_by_timestep.csv", "agg_by_layer_head.csv", "agg_by_prompt.csv", "summary.md")
        for name in expected_tables:
            if not (tables_dir / name).is_file():
                failures.append(f"missing table: {name}")
        for name in ("fig1_mask_overlap_vs_sparsity.csv",
                     "fig4_overlap_vs_timestep.csv",
                     "fig2_layer_timestep_jaccard_s0.80_nvfp4.csv",
                     "fig2_layer_timestep_jaccard_s0.90_nvfp4.csv",
                     "fig3_head_jaccard_box_s0.80_nvfp4.csv"):
            if not (figures_dir / name).is_file():
                failures.append(f"missing figure values: {name}")
        if pyplot is not None:
            for name in ("fig1_mask_overlap_vs_sparsity.png",
                         "fig4_overlap_vs_timestep.png",
                         "fig2_layer_timestep_jaccard_s0.90_nvfp4.png",
                         "fig3_head_jaccard_box_s0.90_nvfp4.png"):
                if not (figures_dir / name).is_file():
                    failures.append(f"missing figure png: {name}")

        with (tables_dir / "agg_by_sparsity_precision.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            failures.append("agg_by_sparsity_precision.csv has no rows")
        for row in rows:
            if not row["n"] or int(row["n"]) <= 0:
                failures.append(f"cell without n: {row}")
            if row["routing_precision"] == "bf16" and float(row["jaccard_median"]) != 1.0:
                failures.append("null control broken: bf16 median jaccard != 1.0")
        nvfp4_rows = {float(row["sparsity"]): float(row["jaccard_median"])
                      for row in rows if row["routing_precision"] == "nvfp4"}
        if not nvfp4_rows.get(0.95, 1.0) < nvfp4_rows.get(0.50, 0.0):
            failures.append(f"synthetic monotonicity broken: {nvfp4_rows}")
        for row in rows:
            recall = float(row["recall_median"])
            jaccard = float(row["jaccard_median"])
            if jaccard > recall + 1e-9:
                failures.append(f"jaccard must not exceed recall: {row}")
        identity_records = load_records(root / "raw", {}).records
        for record in identity_records:
            recall = float(record["recall"])
            if abs(float(record["jaccard"]) - recall / (2.0 - recall)) > 1e-9:
                failures.append(f"per-record jaccard/recall identity broken: {record}")
                break

        summary_text = (tables_dir / "summary.md").read_text(encoding="utf-8")
        for needle in ("malformed lines skipped: 3", "n=", "Localization (H2)", "Timestep trend (H2)",
                       "Per-layer ranking (H2)", "Per-head ranking (H2)"):
            if needle not in summary_text:
                failures.append(f"summary.md missing expected content: {needle!r}")

        empty_root = root / "empty"
        (empty_root / "raw").mkdir(parents=True, exist_ok=True)
        (empty_root / "raw" / "only_bad.jsonl").write_text("{\"prompt_id\": \"p01\"}\n", encoding="utf-8")
        try:
            run_analysis(empty_root / "raw", empty_root / "tables", empty_root / "figures", {},
                         DEFAULT_FIGURE_SPARSITIES, "csv", DEFAULT_MIN_N, 0.90, DEFAULT_HEATMAP_PRECISION, False)
        except SystemExit:
            pass
        else:
            failures.append("expected SystemExit when all records are malformed")

        filtered = load_records(root / "raw", {"routing_precision": "nvfp4", "sparsity": "0.90"})
        if not filtered.records:
            failures.append("--filter selectors returned no records")
        if any(rec["routing_precision"] != "nvfp4" for rec in filtered.records):
            failures.append("--filter leaked non-matching records")

        if keep_dir:
            print(f"self-test artifacts kept in {root}")
        if failures:
            print("SELF-TEST FAILED:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        print(f"SELF-TEST PASSED ({len(outputs.tables)} tables, {len(outputs.figures)} figure files)")
        return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_masks.py",
        description="Aggregate SparseFP4 Phase-1 mask-stability JSONL records into tables and figures.",
    )
    parser.add_argument("--raw", type=Path,
                        help="JSONL file or directory searched recursively for *.jsonl / *.jsonl.gz")
    parser.add_argument("--out-tables", type=Path, default=Path("artifacts/sparsefp4/tables"),
                        help="output directory for aggregate tables and summary.md")
    parser.add_argument("--out-figures", type=Path, default=Path("artifacts/sparsefp4/figures"),
                        help="output directory for figures and their value CSVs")
    parser.add_argument("--sparsity", type=float, action="append", default=None,
                        help="sparsity value for Figure 2 panels; repeatable (default: 0.80 and 0.90)")
    parser.add_argument("--filter", dest="filters", action="append", default=[], metavar="KEY=VALUE",
                        help="keep only records whose field equals VALUE; repeatable")
    parser.add_argument("--format", dest="table_format", choices=("csv", "markdown", "both"), default="csv",
                        help="table output format (default: csv)")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                        help=f"minimum observations before a cell may be quoted (default: {DEFAULT_MIN_N})")
    parser.add_argument("--affected-threshold", type=float, default=0.90,
                        help="median Jaccard below which a cell is called affected (default: 0.90)")
    parser.add_argument("--heatmap-precision", default=DEFAULT_HEATMAP_PRECISION,
                        help="routing precision compared against BF16 in Figure 2 "
                             f"(default: {DEFAULT_HEATMAP_PRECISION})")
    parser.add_argument("--no-head-figure", action="store_true", help="skip the optional per-head boxplot")
    parser.add_argument("--self-test", action="store_true",
                        help="generate synthetic records and run the whole pipeline end to end")
    parser.add_argument("--self-test-keep", type=Path, default=None,
                        help="with --self-test, keep artifacts in this directory instead of a temp dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test(args.self_test_keep)
    if args.raw is None:
        build_parser().error("--raw is required unless --self-test is given")
    figure_sparsities = tuple(args.sparsity) if args.sparsity else DEFAULT_FIGURE_SPARSITIES
    run_analysis(
        raw=args.raw,
        out_tables=args.out_tables,
        out_figures=args.out_figures,
        filters=parse_filters(args.filters),
        figure_sparsities=figure_sparsities,
        table_format=args.table_format,
        min_n=args.min_n,
        affected_threshold=args.affected_threshold,
        heatmap_precision=args.heatmap_precision,
        head_figure=not args.no_head_figure,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

