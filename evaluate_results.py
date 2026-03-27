#!/usr/bin/env python3
"""Evaluate and summarize all results CSV files in this project.

The project stores measurements with semicolon delimiters and decimal commas,
and some files contain repeated headers for different conditions. This script
normalizes those differences without changing metric values and exports:

- analysis/results_consistent_combined.csv
- analysis/results_summary_by_phase_group.csv
- analysis/steering_law_summary.csv
- analysis/plots/*.png
"""

from __future__ import annotations

import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "analysis"
OUTPUT_DIR = BASE_DIR / "analysis"
PLOTS_DIR = OUTPUT_DIR / "plots"
STEERING_JSON = BASE_DIR / "data" / "steering_law_measurements.json"


PHASE_ORDER = {
    "Baseline": 0,
    "Training 1": 1,
    "Training 2": 2,
    "Training 3": 3,
    "Training 4": 4,
    "Training 5": 5,
    "Evaluation": 6,
}

METRICS = [
    "completion_time_s",
    "position_rmse_mm",
    "mean_velocity_mm_s",
    "velocity_std_mm_s",
    "difficulty_1_10",
]


def canonical_group(raw_group: str) -> str:
    """Map inconsistent group labels to canonical group names."""
    g = normalize_header(raw_group)
    if "haptic" in g:
        return "Visual + Haptic"
    if "visual" in g:
        return "Visual"
    return "Unknown"


def infer_phase(session_label: str) -> str:
    """Return canonical phase name from session label."""
    s = normalize_header(session_label)
    if "baseline" in s:
        return "Baseline"
    if "evaluation" in s:
        return "Evaluation"
    match = re.search(r"training\s*(\d+)", s)
    if match:
        return f"Training {int(match.group(1))}"
    return session_label


def infer_training_round(phase: str) -> Optional[int]:
    """Return training round for Training phases, else None."""
    m = re.search(r"Training\s+(\d+)", phase)
    if m:
        return int(m.group(1))
    return None


def normalize_header(value: str) -> str:
    """Create a lowercase, punctuation-light header key for matching."""
    cleaned = value.strip().lower()
    cleaned = cleaned.replace("(", " ").replace(")", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_decimal(value: str) -> Optional[float]:
    """Parse numbers that may use decimal commas, return None if empty/invalid."""
    txt = value.strip()
    if not txt:
        return None

    txt = txt.replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


def safe_file_label(path: Path) -> str:
    """Extract label from names like results(Training 1).csv."""
    match = re.search(r"results\((.*?)\)", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return path.stem


def header_index(headers: List[str], keyword: str) -> Optional[int]:
    """Find header index containing keyword in normalized form."""
    wanted = normalize_header(keyword)
    for idx, h in enumerate(headers):
        if wanted in normalize_header(h):
            return idx
    return None


def get_cell(row: List[str], idx: Optional[int]) -> str:
    """Safely fetch a row cell by index."""
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return row[idx].strip()


def parse_results_file(path: Path) -> List[Dict[str, object]]:
    """Parse a results CSV file into normalized row records."""
    records: List[Dict[str, object]] = []
    file_label = safe_file_label(path)

    current_headers: Optional[List[str]] = None
    current_condition: str = ""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]

            if not any(row):
                continue

            participant_present = any(
                normalize_header(cell) == "participant" for cell in row
            )
            completion_present = any(
                "completion time" in normalize_header(cell) for cell in row
            )

            if participant_present and completion_present:
                current_headers = row
                first = row[0] if row else ""
                if first and normalize_header(first) != "participant":
                    current_condition = first
                continue

            if current_headers is None:
                continue

            participant_idx = header_index(current_headers, "participant")
            participant = get_cell(row, participant_idx)
            if not participant:
                continue

            training_group_idx = header_index(current_headers, "training group")
            training_group = get_cell(row, training_group_idx) or current_condition

            phase = infer_phase(file_label)
            group = canonical_group(training_group)

            record = {
                "source_file": path.name,
                "session": file_label,
                "phase": phase,
                "phase_order": PHASE_ORDER.get(phase, 999),
                "training_round": infer_training_round(phase),
                "participant": participant,
                "group_raw": training_group,
                "group": group,
                "completion_time_s": parse_decimal(
                    get_cell(row, header_index(current_headers, "completion time"))
                ),
                "position_rmse_mm": parse_decimal(
                    get_cell(row, header_index(current_headers, "position rmse"))
                ),
                "mean_velocity_mm_s": parse_decimal(
                    get_cell(row, header_index(current_headers, "mean velocity"))
                ),
                "velocity_std_mm_s": parse_decimal(
                    get_cell(row, header_index(current_headers, "velocity std"))
                ),
                "difficulty_1_10": parse_decimal(
                    get_cell(row, header_index(current_headers, "difficulty"))
                ),
            }

            records.append(record)

    return records


def summarize_numeric(records: Iterable[Dict[str, object]], field: str) -> Tuple[str, str]:
    """Return mean and std strings for a numeric field."""
    values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    values = [float(v) for v in values if not math.isnan(float(v))]

    if not values:
        return "", ""

    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{avg:.3f}", f"{sd:.3f}"


def write_long_table(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write all normalized participant rows."""
    fieldnames = [
        "source_file",
        "session",
        "phase",
        "phase_order",
        "training_round",
        "participant",
        "group_raw",
        "group",
        "completion_time_s",
        "position_rmse_mm",
        "mean_velocity_mm_s",
        "velocity_std_mm_s",
        "difficulty_1_10",
        "steering_id",
        "throughput_bits_s",
    ]

    rows_sorted = sorted(
        records,
        key=lambda r: (
            int(r.get("phase_order", 999)),
            str(r.get("group", "")),
            str(r.get("participant", "")),
            str(r.get("session", "")),
        ),
    )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)


def write_summary(
    records: List[Dict[str, object]],
    key_fields: List[str],
    out_path: Path,
) -> None:
    """Write grouped summary statistics for each metric."""
    grouped: Dict[Tuple[str, ...], List[Dict[str, object]]] = {}
    for row in records:
        key = tuple(str(row.get(field, "")).strip() or "(unknown)" for field in key_fields)
        grouped.setdefault(key, []).append(row)

    metric_fields = METRICS

    summary_rows: List[Dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        summary_row: Dict[str, object] = {
            "n_rows": len(rows),
            "n_participants": len({str(r.get("participant", "")).strip() for r in rows}),
        }
        for field_name, field_value in zip(key_fields, key):
            summary_row[field_name] = field_value

        for metric in metric_fields:
            avg, sd = summarize_numeric(rows, metric)
            summary_row[f"{metric}_mean"] = avg
            summary_row[f"{metric}_std"] = sd

        summary_rows.append(summary_row)

    if not summary_rows:
        return

    preferred_front = [*key_fields, "n_rows", "n_participants"]
    metric_cols = [k for k in summary_rows[0].keys() if k not in preferred_front]
    fieldnames = preferred_front + metric_cols

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def load_steering_ids() -> Dict[str, float]:
    """Load IDs (full width convention) from steering law JSON."""
    if not STEERING_JSON.exists():
        return {}

    with STEERING_JSON.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    cond = data.get("conditions", {})
    return {
        "Baseline": float(cond.get("baseline", {}).get("ID_full_0p04", math.nan)),
        "Training Visual": float(
            cond.get("training_visual", {}).get("ID_full_0p04", math.nan)
        ),
        "Training Visual + Haptic": float(
            cond.get("training_visual_haptic", {}).get("ID_full_0p04", math.nan)
        ),
        "Evaluation": float(cond.get("test", {}).get("ID_full_0p04", math.nan)),
    }


def steering_id_for_record(record: Dict[str, object], ids: Dict[str, float]) -> Optional[float]:
    """Return condition-specific ID for a participant record."""
    phase = str(record.get("phase", ""))
    group = str(record.get("group", ""))

    if phase == "Baseline":
        value = ids.get("Baseline")
    elif phase.startswith("Training"):
        value = ids.get("Training Visual + Haptic" if group == "Visual + Haptic" else "Training Visual")
    elif phase == "Evaluation":
        value = ids.get("Evaluation")
    else:
        value = None

    if value is None or math.isnan(value):
        return None
    return value


def add_steering_metrics(records: List[Dict[str, object]]) -> None:
    """Annotate records with Steering ID and Throughput = ID / MT."""
    ids = load_steering_ids()
    for row in records:
        row["steering_id"] = steering_id_for_record(row, ids)
        mt = row.get("completion_time_s")
        sid = row.get("steering_id")
        if isinstance(mt, (int, float)) and isinstance(sid, (int, float)) and mt > 0:
            row["throughput_bits_s"] = float(sid) / float(mt)
        else:
            row["throughput_bits_s"] = None


def write_steering_summary(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write per phase/group steering-law summary."""
    steering_rows = [
        r
        for r in records
        if isinstance(r.get("steering_id"), (int, float))
        and isinstance(r.get("completion_time_s"), (int, float))
    ]

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in steering_rows:
        key = (str(row.get("phase", "")), str(row.get("group", "")))
        grouped.setdefault(key, []).append(row)

    out_rows: List[Dict[str, object]] = []
    for (phase, group), rows in sorted(grouped.items(), key=lambda x: (PHASE_ORDER.get(x[0][0], 999), x[0][1])):
        mts = [float(r["completion_time_s"]) for r in rows if isinstance(r.get("completion_time_s"), (int, float))]
        tps = [float(r["throughput_bits_s"]) for r in rows if isinstance(r.get("throughput_bits_s"), (int, float))]
        sid = next((float(r["steering_id"]) for r in rows if isinstance(r.get("steering_id"), (int, float))), math.nan)

        out_rows.append(
            {
                "phase": phase,
                "phase_order": PHASE_ORDER.get(phase, 999),
                "group": group,
                "n": len(rows),
                "steering_id": f"{sid:.6f}" if not math.isnan(sid) else "",
                "movement_time_mean_s": f"{mean(mts):.3f}" if mts else "",
                "movement_time_std_s": f"{stdev(mts):.3f}" if len(mts) > 1 else ("0.000" if mts else ""),
                "throughput_mean_bits_s": f"{mean(tps):.4f}" if tps else "",
                "throughput_std_bits_s": f"{stdev(tps):.4f}" if len(tps) > 1 else ("0.0000" if tps else ""),
            }
        )

    if not out_rows:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)


def cleanup_stale_outputs() -> None:
    """Delete legacy analysis files that are no longer part of the pipeline."""
    stale_files = [
        OUTPUT_DIR / "results_long.csv",
        OUTPUT_DIR / "results_summary_by_file.csv",
        OUTPUT_DIR / "results_summary_by_group.csv",
        OUTPUT_DIR / "results_summary_by_session_group.csv",
    ]

    for stale in stale_files:
        if stale.exists():
            stale.unlink()


def cohen_d_unpaired(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Cohen's d for two independent samples."""
    if len(x) < 2 or len(y) < 2:
        return None
    vx = stdev(x) ** 2
    vy = stdev(y) ** 2
    dof = len(x) + len(y) - 2
    if dof <= 0:
        return None
    pooled_var = ((len(x) - 1) * vx + (len(y) - 1) * vy) / dof
    if pooled_var <= 0:
        return None
    return (mean(y) - mean(x)) / math.sqrt(pooled_var)


def cohen_d_paired(diffs: List[float]) -> Optional[float]:
    """Compute paired Cohen's d using the SD of differences."""
    if len(diffs) < 2:
        return None
    s = stdev(diffs)
    if s == 0:
        return None
    return mean(diffs) / s


def exact_permutation_pvalue_mean_diff(x: List[float], y: List[float]) -> Optional[float]:
    """Two-sided exact permutation p-value for difference in means."""
    if not x or not y:
        return None
    pooled = x + y
    n = len(pooled)
    n_x = len(x)
    obs = abs(mean(x) - mean(y))
    total = 0
    extreme = 0

    for idxs in combinations(range(n), n_x):
        pick = set(idxs)
        gx = [pooled[i] for i in range(n) if i in pick]
        gy = [pooled[i] for i in range(n) if i not in pick]
        stat = abs(mean(gx) - mean(gy))
        total += 1
        if stat >= obs - 1e-12:
            extreme += 1

    if total == 0:
        return None
    return extreme / total


def exact_sign_test_pvalue(diffs: List[float]) -> Optional[float]:
    """Two-sided exact sign-test p-value for paired differences."""
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return None
    n_pos = sum(1 for d in nonzero if d > 0)
    k = min(n_pos, n - n_pos)

    cumulative = 0.0
    for i in range(k + 1):
        cumulative += math.comb(n, i) / (2**n)
    p = min(1.0, 2.0 * cumulative)
    return p


def paired_metric_values(
    records: List[Dict[str, object]],
    phase_a: str,
    phase_b: str,
    group: str,
    metric: str,
) -> Tuple[List[float], List[float]]:
    """Get participant-aligned paired values for two phases."""
    data_a: Dict[str, float] = {}
    data_b: Dict[str, float] = {}

    for row in records:
        if str(row.get("group")) != group:
            continue
        participant = str(row.get("participant", "")).strip()
        phase = str(row.get("phase", ""))
        val = row.get(metric)
        if not participant or not isinstance(val, (int, float)):
            continue
        if phase == phase_a:
            data_a[participant] = float(val)
        elif phase == phase_b:
            data_b[participant] = float(val)

    common = sorted(set(data_a).intersection(data_b))
    return [data_a[p] for p in common], [data_b[p] for p in common]


def write_between_group_stats(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write group-comparison stats per phase and metric."""
    groups = ["Visual", "Visual + Haptic"]
    phases = sorted({str(r.get("phase", "")) for r in records}, key=lambda p: PHASE_ORDER.get(p, 999))
    rows_out: List[Dict[str, object]] = []

    for phase in phases:
        phase_rows = [r for r in records if str(r.get("phase")) == phase]
        for metric in METRICS:
            x = [float(r[metric]) for r in phase_rows if str(r.get("group")) == groups[0] and isinstance(r.get(metric), (int, float))]
            y = [float(r[metric]) for r in phase_rows if str(r.get("group")) == groups[1] and isinstance(r.get(metric), (int, float))]

            if not x or not y:
                continue

            diff = mean(y) - mean(x)
            d = cohen_d_unpaired(x, y)
            p = exact_permutation_pvalue_mean_diff(x, y)

            rows_out.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "n_visual": len(x),
                    "n_visual_haptic": len(y),
                    "mean_visual": f"{mean(x):.4f}",
                    "mean_visual_haptic": f"{mean(y):.4f}",
                    "mean_diff_haptic_minus_visual": f"{diff:.4f}",
                    "cohen_d": f"{d:.4f}" if d is not None else "",
                    "exact_permutation_p": f"{p:.6f}" if p is not None else "",
                }
            )

    if not rows_out:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def write_within_group_change(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write paired within-group changes for key phase transitions."""
    rows_out: List[Dict[str, object]] = []
    groups = ["Visual", "Visual + Haptic"]
    comparisons = [
        ("Baseline", "Evaluation"),
        ("Training 1", "Training 5"),
    ]

    for group in groups:
        for phase_a, phase_b in comparisons:
            for metric in METRICS:
                a_vals, b_vals = paired_metric_values(records, phase_a, phase_b, group, metric)
                if not a_vals or not b_vals:
                    continue

                diffs = [b - a for a, b in zip(a_vals, b_vals)]
                pct = [((b - a) / a) * 100.0 for a, b in zip(a_vals, b_vals) if abs(a) > 1e-12]
                d = cohen_d_paired(diffs)
                p = exact_sign_test_pvalue(diffs)

                rows_out.append(
                    {
                        "group": group,
                        "phase_from": phase_a,
                        "phase_to": phase_b,
                        "metric": metric,
                        "n_paired": len(diffs),
                        "mean_from": f"{mean(a_vals):.4f}",
                        "mean_to": f"{mean(b_vals):.4f}",
                        "mean_change": f"{mean(diffs):.4f}",
                        "mean_change_percent": f"{mean(pct):.2f}" if pct else "",
                        "cohen_d_paired": f"{d:.4f}" if d is not None else "",
                        "exact_sign_test_p": f"{p:.6f}" if p is not None else "",
                    }
                )

    if not rows_out:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def write_training_trend_slopes(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write linear slope across training rounds for each participant and group."""
    rows_out: List[Dict[str, object]] = []
    groups = sorted({str(r.get("group", "")) for r in records if str(r.get("group", "")) in {"Visual", "Visual + Haptic"}})

    for group in groups:
        participants = sorted({str(r.get("participant", "")).strip() for r in records if str(r.get("group")) == group})
        for participant in participants:
            subset = [
                r for r in records
                if str(r.get("group")) == group
                and str(r.get("participant", "")).strip() == participant
                and isinstance(r.get("training_round"), int)
            ]
            if not subset:
                continue

            for metric in ["completion_time_s", "position_rmse_mm", "throughput_bits_s"]:
                xs: List[float] = []
                ys: List[float] = []
                round_to_val: Dict[int, float] = {}
                for row in subset:
                    tr = row.get("training_round")
                    val = row.get(metric)
                    if isinstance(tr, int) and isinstance(val, (int, float)):
                        xs.append(float(tr))
                        ys.append(float(val))
                        round_to_val[tr] = float(val)

                if len(xs) < 2:
                    continue

                slope, intercept, r2 = linear_fit(xs, ys)
                delta_1_to_5 = ""
                pct_1_to_5 = ""
                if 1 in round_to_val and 5 in round_to_val:
                    delta = round_to_val[5] - round_to_val[1]
                    delta_1_to_5 = f"{delta:.4f}"
                    if abs(round_to_val[1]) > 1e-12:
                        pct_1_to_5 = f"{(delta / round_to_val[1]) * 100.0:.2f}"

                rows_out.append(
                    {
                        "group": group,
                        "participant": participant,
                        "metric": metric,
                        "n_points": len(xs),
                        "slope_per_round": f"{slope:.4f}" if slope is not None else "",
                        "intercept": f"{intercept:.4f}" if intercept is not None else "",
                        "r_squared": f"{r2:.4f}" if r2 is not None else "",
                        "delta_round1_to_round5": delta_1_to_5,
                        "delta_percent_round1_to_round5": pct_1_to_5,
                    }
                )

    if not rows_out:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def participant_stage_map(
    records: List[Dict[str, object]],
    group: str,
    metric: str,
) -> Dict[str, Dict[str, float]]:
    """Return participant-level values for Baseline, Training(avg), Evaluation."""
    raw: Dict[str, Dict[str, List[float]]] = {}

    for row in records:
        if str(row.get("group")) != group:
            continue

        participant = str(row.get("participant", "")).strip()
        phase = str(row.get("phase", ""))
        value = row.get(metric)

        if not participant or not isinstance(value, (int, float)):
            continue

        bucket = raw.setdefault(
            participant,
            {"Baseline": [], "Training": [], "Evaluation": []},
        )
        if phase == "Baseline":
            bucket["Baseline"].append(float(value))
        elif phase.startswith("Training"):
            bucket["Training"].append(float(value))
        elif phase == "Evaluation":
            bucket["Evaluation"].append(float(value))

    stage_map: Dict[str, Dict[str, float]] = {}
    for participant, buckets in raw.items():
        stage_values: Dict[str, float] = {}
        if buckets["Baseline"]:
            stage_values["Baseline"] = mean(buckets["Baseline"])
        if buckets["Training"]:
            stage_values["Training"] = mean(buckets["Training"])
        if buckets["Evaluation"]:
            stage_values["Evaluation"] = mean(buckets["Evaluation"])
        if stage_values:
            stage_map[participant] = stage_values

    return stage_map


def write_progression_comparison(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write explicit Baseline->Training(avg)->Evaluation comparison per group."""
    rows_out: List[Dict[str, object]] = []
    groups = ["Visual", "Visual + Haptic"]
    metrics = [
        "completion_time_s",
        "position_rmse_mm",
        "difficulty_1_10",
        "throughput_bits_s",
    ]

    for group in groups:
        for metric in metrics:
            by_participant = participant_stage_map(records, group, metric)

            baseline = [v["Baseline"] for v in by_participant.values() if "Baseline" in v]
            training = [v["Training"] for v in by_participant.values() if "Training" in v]
            evaluation = [v["Evaluation"] for v in by_participant.values() if "Evaluation" in v]

            if not baseline and not training and not evaluation:
                continue

            b_mean = mean(baseline) if baseline else math.nan
            t_mean = mean(training) if training else math.nan
            e_mean = mean(evaluation) if evaluation else math.nan

            b_std = stdev(baseline) if len(baseline) > 1 else (0.0 if baseline else math.nan)
            t_std = stdev(training) if len(training) > 1 else (0.0 if training else math.nan)
            e_std = stdev(evaluation) if len(evaluation) > 1 else (0.0 if evaluation else math.nan)

            def fmt(value: float, digits: int = 4) -> str:
                if isinstance(value, float) and math.isnan(value):
                    return ""
                return f"{value:.{digits}f}"

            bt = t_mean - b_mean if not math.isnan(b_mean) and not math.isnan(t_mean) else math.nan
            te = e_mean - t_mean if not math.isnan(t_mean) and not math.isnan(e_mean) else math.nan
            be = e_mean - b_mean if not math.isnan(b_mean) and not math.isnan(e_mean) else math.nan

            bt_pct = (bt / b_mean) * 100.0 if not math.isnan(bt) and abs(b_mean) > 1e-12 else math.nan
            te_pct = (te / t_mean) * 100.0 if not math.isnan(te) and abs(t_mean) > 1e-12 else math.nan
            be_pct = (be / b_mean) * 100.0 if not math.isnan(be) and abs(b_mean) > 1e-12 else math.nan

            rows_out.append(
                {
                    "group": group,
                    "metric": metric,
                    "n_baseline": len(baseline),
                    "n_training_avg": len(training),
                    "n_evaluation": len(evaluation),
                    "baseline_mean": fmt(b_mean),
                    "baseline_std": fmt(b_std),
                    "training_avg_mean": fmt(t_mean),
                    "training_avg_std": fmt(t_std),
                    "evaluation_mean": fmt(e_mean),
                    "evaluation_std": fmt(e_std),
                    "delta_training_minus_baseline": fmt(bt),
                    "delta_evaluation_minus_training": fmt(te),
                    "delta_evaluation_minus_baseline": fmt(be),
                    "pct_training_minus_baseline": fmt(bt_pct, 2),
                    "pct_evaluation_minus_training": fmt(te_pct, 2),
                    "pct_evaluation_minus_baseline": fmt(be_pct, 2),
                }
            )

    if not rows_out:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def write_progression_normalized(records: List[Dict[str, object]], out_path: Path) -> None:
    """Write Baseline=100 normalization for stage-wise progression by group."""
    groups = ["Visual", "Visual + Haptic"]
    metrics = [
        "completion_time_s",
        "position_rmse_mm",
        "difficulty_1_10",
        "throughput_bits_s",
    ]
    stage_keys = ["Baseline", "Training", "Evaluation"]
    rows_out: List[Dict[str, object]] = []

    for metric in metrics:
        for group in groups:
            by_participant = participant_stage_map(records, group, metric)
            for stage in stage_keys:
                normalized_values: List[float] = []
                for _, stage_values in by_participant.items():
                    baseline = stage_values.get("Baseline")
                    current = stage_values.get(stage)
                    if baseline is None or current is None or abs(baseline) <= 1e-12:
                        continue
                    normalized_values.append((current / baseline) * 100.0)

                if not normalized_values:
                    continue

                rows_out.append(
                    {
                        "metric": metric,
                        "group": group,
                        "stage": stage,
                        "n": len(normalized_values),
                        "normalized_mean_pct": f"{mean(normalized_values):.4f}",
                        "normalized_std_pct": f"{stdev(normalized_values):.4f}" if len(normalized_values) > 1 else "0.0000",
                    }
                )

    if not rows_out:
        return

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def save_plot_participant_learning(records: List[Dict[str, object]], out_path: Path) -> None:
    """Plot participant-level trajectories for training rounds."""
    rounds = [1, 2, 3, 4, 5]
    groups = ["Visual", "Visual + Haptic"]
    colors = {"Visual": "tab:blue", "Visual + Haptic": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    metric_panels = [
        ("completion_time_s", "Participant Learning: Completion Time", "Time (s)"),
        ("position_rmse_mm", "Participant Learning: RMSE", "RMSE (mm)"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metric_panels):
        for group in groups:
            participants = sorted({str(r.get("participant", "")).strip() for r in records if str(r.get("group")) == group})
            for participant in participants:
                yvals: List[float] = []
                xvals: List[int] = []
                for rnd in rounds:
                    vals = [
                        float(r[metric])
                        for r in records
                        if str(r.get("group")) == group
                        and str(r.get("participant", "")).strip() == participant
                        and r.get("training_round") == rnd
                        and isinstance(r.get(metric), (int, float))
                    ]
                    if vals:
                        xvals.append(rnd)
                        yvals.append(mean(vals))

                if len(xvals) >= 2:
                    ax.plot(xvals, yvals, color=colors[group], alpha=0.25, linewidth=1)

            # Group mean overlay
            means = []
            errs = []
            for rnd in rounds:
                vals = grouped_values(records, f"Training {rnd}", group, metric)
                means.append(mean(vals) if vals else math.nan)
                errs.append(stdev(vals) if len(vals) > 1 else 0.0)
            ax.errorbar(rounds, means, yerr=errs, color=colors[group], marker="o", linewidth=2.4, capsize=4, label=group)

        ax.set_title(title)
        ax.set_xlabel("Training Round")
        ax.set_ylabel(ylabel)
        ax.set_xticks(rounds)
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_plot_progression_comparison(records: List[Dict[str, object]], out_path: Path) -> None:
    """Plot Baseline->Training(avg)->Evaluation trajectories for each group."""
    groups = ["Visual", "Visual + Haptic"]
    colors = {"Visual": "tab:blue", "Visual + Haptic": "tab:orange"}
    stage_keys = ["Baseline", "Training", "Evaluation"]
    stage_labels = ["Baseline", "Training (avg)", "Evaluation"]
    metrics = [
        ("completion_time_s", "Completion Time (s)"),
        ("position_rmse_mm", "Position RMSE (mm)"),
        ("difficulty_1_10", "Difficulty (1-10)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharex=True)
    x = [0, 1, 2]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for group in groups:
            by_participant = participant_stage_map(records, group, metric)

            means: List[float] = []
            errs: List[float] = []
            for key in stage_keys:
                vals = [v[key] for v in by_participant.values() if key in v]
                means.append(mean(vals) if vals else math.nan)
                errs.append(stdev(vals) if len(vals) > 1 else (0.0 if vals else math.nan))

            ax.errorbar(
                x,
                means,
                yerr=errs,
                color=colors[group],
                marker="o",
                linewidth=2.2,
                capsize=4,
                label=group,
            )

        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Metric Value")
    axes[0].legend(frameon=False)
    fig.suptitle("Direct Progression Comparison: Baseline -> Training -> Evaluation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_plot_progression_normalized(records: List[Dict[str, object]], out_path: Path) -> None:
    """Plot stage progression normalized to baseline for fair cross-metric comparison."""
    groups = ["Visual", "Visual + Haptic"]
    colors = {"Visual": "tab:blue", "Visual + Haptic": "tab:orange"}
    stage_keys = ["Baseline", "Training", "Evaluation"]
    stage_labels = ["Baseline", "Training (avg)", "Evaluation"]
    metrics = [
        ("completion_time_s", "Completion Time (% of Baseline)"),
        ("position_rmse_mm", "Position RMSE (% of Baseline)"),
        ("difficulty_1_10", "Difficulty (% of Baseline)"),
        ("throughput_bits_s", "Throughput (% of Baseline)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=True)
    flat_axes = list(axes.flatten())
    x = [0, 1, 2]

    for ax, (metric, title) in zip(flat_axes, metrics):
        for group in groups:
            by_participant = participant_stage_map(records, group, metric)
            means: List[float] = []
            errs: List[float] = []

            for stage in stage_keys:
                vals: List[float] = []
                for _, stage_values in by_participant.items():
                    baseline = stage_values.get("Baseline")
                    current = stage_values.get(stage)
                    if baseline is None or current is None or abs(baseline) <= 1e-12:
                        continue
                    vals.append((current / baseline) * 100.0)

                means.append(mean(vals) if vals else math.nan)
                errs.append(stdev(vals) if len(vals) > 1 else (0.0 if vals else math.nan))

            ax.errorbar(
                x,
                means,
                yerr=errs,
                color=colors[group],
                marker="o",
                linewidth=2.2,
                capsize=4,
                label=group,
            )

        ax.axhline(100.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_ylabel("%")
        ax.grid(alpha=0.25)

    flat_axes[0].legend(frameon=False)
    fig.suptitle("Normalized Progression (Baseline = 100)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def grouped_values(records: List[Dict[str, object]], phase: str, group: str, metric: str) -> List[float]:
    """Collect numeric values for one (phase, group, metric) slice."""
    vals: List[float] = []
    for r in records:
        if str(r.get("phase")) != phase or str(r.get("group")) != group:
            continue
        v = r.get(metric)
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            vals.append(float(v))
    return vals


def save_plot_stats(records: List[Dict[str, object]], out_path: Path) -> None:
    """Save grouped bar charts for baseline/training/evaluation statistics."""
    phases = ["Baseline", "Evaluation"]
    groups = ["Visual", "Visual + Haptic"]
    metrics = [
        ("completion_time_s", "Completion Time (s)"),
        ("position_rmse_mm", "Position RMSE (mm)"),
        ("difficulty_1_10", "Difficulty (1-10)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (metric_key, title) in zip(axes, metrics):
        x_positions = list(range(len(phases)))
        width = 0.36

        for gi, group in enumerate(groups):
            means = []
            errs = []
            for ph in phases:
                vals = grouped_values(records, ph, group, metric_key)
                means.append(mean(vals) if vals else math.nan)
                errs.append(stdev(vals) if len(vals) > 1 else 0.0)

            offset = -width / 2 if gi == 0 else width / 2
            ax.bar(
                [x + offset for x in x_positions],
                means,
                width=width,
                yerr=errs,
                capsize=4,
                label=group,
                alpha=0.9,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(phases)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend()
    fig.suptitle("Shared Tests: Baseline vs Evaluation by Group", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_learning_curve(records: List[Dict[str, object]], out_path: Path) -> None:
    """Save learning curve plots across Training 1..5 for both groups."""
    rounds = [1, 2, 3, 4, 5]
    groups = ["Visual", "Visual + Haptic"]
    metrics = [
        ("completion_time_s", "Completion Time (s)"),
        ("position_rmse_mm", "Position RMSE (mm)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)

    for ax, (metric_key, y_label) in zip(axes, metrics):
        for group in groups:
            means = []
            errs = []
            for rnd in rounds:
                phase = f"Training {rnd}"
                vals = grouped_values(records, phase, group, metric_key)
                means.append(mean(vals) if vals else math.nan)
                errs.append(stdev(vals) if len(vals) > 1 else 0.0)

            ax.errorbar(
                rounds,
                means,
                yerr=errs,
                marker="o",
                linewidth=2,
                capsize=4,
                label=group,
            )

        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)

    axes[0].set_title("Learning Curve: Time")
    axes[1].set_title("Learning Curve: Accuracy")
    axes[0].set_xlabel("Training Round")
    axes[1].set_xlabel("Training Round")
    axes[0].set_xticks(rounds)
    axes[1].set_xticks(rounds)
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def linear_fit(x_vals: List[float], y_vals: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return slope, intercept, and R^2 for a simple linear fit."""
    if len(x_vals) < 2 or len(y_vals) < 2:
        return None, None, None

    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    denom = sum((x - x_mean) ** 2 for x in x_vals)
    if denom == 0:
        return None, None, None

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / denom
    intercept = y_mean - slope * x_mean
    y_hat = [slope * x + intercept for x in x_vals]

    ss_res = sum((y - yh) ** 2 for y, yh in zip(y_vals, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return slope, intercept, r2


def save_plot_steering_law(records: List[Dict[str, object]], out_path: Path) -> None:
    """Plot Movement Time vs ID for each group and include linear trend."""
    groups = ["Visual", "Visual + Haptic"]
    phase_whitelist = {"Baseline", "Training 1", "Training 2", "Training 3", "Training 4", "Training 5", "Evaluation"}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for group in groups:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for phase in sorted(phase_whitelist, key=lambda p: PHASE_ORDER.get(p, 999)):
            phase_rows = [
                r for r in records if str(r.get("group")) == group and str(r.get("phase")) == phase
            ]
            mt_vals = [float(r["completion_time_s"]) for r in phase_rows if isinstance(r.get("completion_time_s"), (int, float))]
            id_vals = [float(r["steering_id"]) for r in phase_rows if isinstance(r.get("steering_id"), (int, float))]
            if not mt_vals or not id_vals:
                continue

            xs.append(mean(id_vals))
            ys.append(mean(mt_vals))
            labels.append(phase)

        if not xs:
            continue

        ax.scatter(xs, ys, s=55, label=group)

        for x, y, lbl in zip(xs, ys, labels):
            short = lbl.replace("Training ", "T")
            ax.annotate(short, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

        slope, intercept, r2 = linear_fit(xs, ys)
        if slope is not None and intercept is not None:
            xline = [min(xs), max(xs)]
            yline = [slope * x + intercept for x in xline]
            suffix = f" (R^2={r2:.2f})" if r2 is not None else ""
            ax.plot(xline, yline, linestyle="--", linewidth=1.5, label=f"{group} fit{suffix}")

    ax.set_xlabel("Steering ID (bits)")
    ax.set_ylabel("Movement Time (s)")
    ax.set_title("Steering Law Relation Across Phases")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_publication_report(records: List[Dict[str, object]], out_path: Path) -> None:
    """Create one publication-style figure with key panels."""
    groups = ["Visual", "Visual + Haptic"]
    phases_shared = ["Baseline", "Evaluation"]
    rounds = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Panel A: shared-test completion time bars
    ax = axes[0, 0]
    width = 0.36
    x_positions = list(range(len(phases_shared)))
    for gi, group in enumerate(groups):
        means = []
        errs = []
        for ph in phases_shared:
            vals = grouped_values(records, ph, group, "completion_time_s")
            means.append(mean(vals) if vals else math.nan)
            errs.append(stdev(vals) if len(vals) > 1 else 0.0)
        offset = -width / 2 if gi == 0 else width / 2
        ax.bar(
            [x + offset for x in x_positions],
            means,
            width=width,
            yerr=errs,
            capsize=4,
            label=group,
            alpha=0.92,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases_shared)
    ax.set_title("A. Shared Tests: Completion Time")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    # Panel B: shared-test RMSE bars
    ax = axes[0, 1]
    for gi, group in enumerate(groups):
        means = []
        errs = []
        for ph in phases_shared:
            vals = grouped_values(records, ph, group, "position_rmse_mm")
            means.append(mean(vals) if vals else math.nan)
            errs.append(stdev(vals) if len(vals) > 1 else 0.0)
        offset = -width / 2 if gi == 0 else width / 2
        ax.bar(
            [x + offset for x in x_positions],
            means,
            width=width,
            yerr=errs,
            capsize=4,
            alpha=0.92,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases_shared)
    ax.set_title("B. Shared Tests: Position RMSE")
    ax.set_ylabel("RMSE (mm)")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: shared-test subjective difficulty bars
    ax = axes[0, 2]
    for gi, group in enumerate(groups):
        means = []
        errs = []
        for ph in phases_shared:
            vals = grouped_values(records, ph, group, "difficulty_1_10")
            means.append(mean(vals) if vals else math.nan)
            errs.append(stdev(vals) if len(vals) > 1 else 0.0)
        offset = -width / 2 if gi == 0 else width / 2
        ax.bar(
            [x + offset for x in x_positions],
            means,
            width=width,
            yerr=errs,
            capsize=4,
            alpha=0.92,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(phases_shared)
    ax.set_title("C. Shared Tests: Difficulty")
    ax.set_ylabel("Difficulty (1-10)")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: learning curve for completion time
    ax = axes[1, 0]
    for group in groups:
        means = []
        errs = []
        for rnd in rounds:
            vals = grouped_values(records, f"Training {rnd}", group, "completion_time_s")
            means.append(mean(vals) if vals else math.nan)
            errs.append(stdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(rounds, means, yerr=errs, marker="o", linewidth=2, capsize=4, label=group)
    ax.set_title("D. Learning Curve: Time")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.3)

    # Panel E: learning curve for RMSE
    ax = axes[1, 1]
    for group in groups:
        means = []
        errs = []
        for rnd in rounds:
            vals = grouped_values(records, f"Training {rnd}", group, "position_rmse_mm")
            means.append(mean(vals) if vals else math.nan)
            errs.append(stdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(rounds, means, yerr=errs, marker="o", linewidth=2, capsize=4)
    ax.set_title("E. Learning Curve: Accuracy")
    ax.set_xlabel("Training Round")
    ax.set_ylabel("RMSE (mm)")
    ax.set_xticks(rounds)
    ax.grid(alpha=0.3)

    # Panel F: steering-law phase means and linear fit
    ax = axes[1, 2]
    phase_whitelist = ["Baseline", "Training 1", "Training 2", "Training 3", "Training 4", "Training 5", "Evaluation"]
    for group in groups:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []
        for phase in phase_whitelist:
            phase_rows = [
                r for r in records if str(r.get("group")) == group and str(r.get("phase")) == phase
            ]
            mt_vals = [float(r["completion_time_s"]) for r in phase_rows if isinstance(r.get("completion_time_s"), (int, float))]
            id_vals = [float(r["steering_id"]) for r in phase_rows if isinstance(r.get("steering_id"), (int, float))]
            if not mt_vals or not id_vals:
                continue
            xs.append(mean(id_vals))
            ys.append(mean(mt_vals))
            labels.append(phase)

        if not xs:
            continue

        ax.scatter(xs, ys, s=45, label=group)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl.replace("Training ", "T"), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)

        slope, intercept, r2 = linear_fit(xs, ys)
        if slope is not None and intercept is not None:
            xline = [min(xs), max(xs)]
            yline = [slope * x + intercept for x in xline]
            fit_label = f"{group} fit"
            if r2 is not None:
                fit_label += f" (R^2={r2:.2f})"
            ax.plot(xline, yline, linestyle="--", linewidth=1.5, label=fit_label)

    ax.set_title("F. Steering Law")
    ax.set_xlabel("ID (bits)")
    ax.set_ylabel("Movement Time (s)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Weld Task Performance Report", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("results(*).csv"))
    if not csv_files:
        raise SystemExit("No files found matching analysis/results(*).csv")

    all_records: List[Dict[str, object]] = []
    for csv_file in csv_files:
        all_records.extend(parse_results_file(csv_file))

    if not all_records:
        raise SystemExit("No participant records could be parsed from the CSV files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    add_steering_metrics(all_records)

    consistent_path = OUTPUT_DIR / "results_consistent_combined.csv"
    phase_group_summary_path = OUTPUT_DIR / "results_summary_by_phase_group.csv"
    steering_summary_path = OUTPUT_DIR / "steering_law_summary.csv"
    between_group_stats_path = OUTPUT_DIR / "stats_between_groups.csv"
    within_group_change_path = OUTPUT_DIR / "stats_within_group_change.csv"
    training_slopes_path = OUTPUT_DIR / "stats_training_slopes.csv"
    progression_comparison_path = OUTPUT_DIR / "comparison_baseline_training_evaluation.csv"
    progression_normalized_path = OUTPUT_DIR / "comparison_baseline_training_evaluation_normalized.csv"

    stats_plot_path = PLOTS_DIR / "stats_shared_tests.png"
    learning_plot_path = PLOTS_DIR / "learning_curve.png"
    steering_plot_path = PLOTS_DIR / "steering_law_relation.png"
    publication_plot_path = PLOTS_DIR / "publication_report_overview.png"
    participant_learning_plot_path = PLOTS_DIR / "participant_learning_spaghetti.png"
    progression_plot_path = PLOTS_DIR / "baseline_training_evaluation_comparison.png"
    progression_normalized_plot_path = PLOTS_DIR / "baseline_training_evaluation_normalized.png"

    write_long_table(all_records, consistent_path)
    write_summary(all_records, ["phase", "group"], phase_group_summary_path)
    write_steering_summary(all_records, steering_summary_path)
    write_between_group_stats(all_records, between_group_stats_path)
    write_within_group_change(all_records, within_group_change_path)
    write_training_trend_slopes(all_records, training_slopes_path)
    write_progression_comparison(all_records, progression_comparison_path)
    write_progression_normalized(all_records, progression_normalized_path)

    save_plot_stats(all_records, stats_plot_path)
    save_plot_learning_curve(all_records, learning_plot_path)
    save_plot_steering_law(all_records, steering_plot_path)
    save_plot_publication_report(all_records, publication_plot_path)
    save_plot_participant_learning(all_records, participant_learning_plot_path)
    save_plot_progression_comparison(all_records, progression_plot_path)
    save_plot_progression_normalized(all_records, progression_normalized_plot_path)
    cleanup_stale_outputs()

    print(f"Parsed files: {len(csv_files)}")
    print(f"Parsed rows: {len(all_records)}")
    print(f"Wrote: {consistent_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {phase_group_summary_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {steering_summary_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {between_group_stats_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {within_group_change_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {training_slopes_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {progression_comparison_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {progression_normalized_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {stats_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {learning_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {steering_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {publication_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {participant_learning_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {progression_plot_path.relative_to(BASE_DIR)}")
    print(f"Wrote: {progression_normalized_plot_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()