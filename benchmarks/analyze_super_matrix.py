#!/usr/bin/env python3
"""
Defensive stdlib post-processor and Markdown report generator for HARMONY Super Job.
Supports dual-format ingestion:
  1. Refactored upstream HARMONY structured CSV timing reports (#run, #timeline,
     #device, #process, #balance) alongside per-iteration CSVs and stdout summaries.
  2. Historical Evidence logs ([DCL][balance-metrics]) and per-iteration CSVs.

Computes paired speedup ratios, repetition dispersion statistics, unaccounted wall
time reconciliation, A/B/A drift analysis, multi-node GPU telemetry correlation, and
a bounded diagnostic Markdown report (SUPER_JOB_REPORT.md).

Strictly Python 3 standard library: zero external dependencies, zero subprocess calls.
"""

import argparse
import csv
import datetime
import json
from pathlib import Path
import re
import statistics
import sys

HISTORICAL_PRELIMINARY = {
    ("kneighbor", 42): {"erad_s": 3.739960, "hwtopolb_s": 0.220817, "ratio": 16.94, "speedup_pct": 94.10},
    ("kneighbor", 123): {"erad_s": 3.579730, "hwtopolb_s": 3.819720, "ratio": 0.94, "speedup_pct": -6.70},
    ("kneighbor", 456): {"erad_s": 3.879730, "hwtopolb_s": 0.214246, "ratio": 18.11, "speedup_pct": 94.48},
    ("leanmd", 42): {"erad_s": 4.300190, "hwtopolb_s": 0.438039, "ratio": 9.82, "speedup_pct": 89.81},
    ("leanmd", 123): {"erad_s": 4.329970, "hwtopolb_s": 4.459910, "ratio": 0.97, "speedup_pct": -3.00},
    ("leanmd", 456): {"erad_s": 4.221200, "hwtopolb_s": 0.361940, "ratio": 11.66, "speedup_pct": 91.43},
}

# Named heuristic classification thresholds for seed-123 descriptive summary and telemetry.
# These thresholds are descriptive heuristics only, NOT causal findings or verified boundaries.
# Observations and classifications are strictly limited to the measured configuration.
HEURISTIC_SEED_123_SPEEDUP_THRESHOLD = 2.0
HEURISTIC_SEED_123_PARITY_THRESHOLD = 1.05
# Non-causal heuristic threshold (MHz) for flagging inter-node graphics clock spread in telemetry.
HEURISTIC_CLOCK_DIVERGENCE_DELTA_MHZ = 15

_LOAD_LINE_RE = re.compile(
    r"part\[(\d+)\]\s+cum=([0-9.]+)%\s+share=([0-9.]+)%\s+"
    r"device_global=(\d+)\s+rank=(\d+)\s+local_index=(-?\d+)\s+offset=(\d+)\s+count=(\d+)"
)

_REPORT_INT_FIELDS = {
    "iterations", "mpi_processes", "devices", "balance_attempts", "balance_applied",
    "bytes_moved", "rank", "device", "index", "elements", "number", "iteration",
    "applied", "warmup_probe", "seed", "contention_adjusted", "power_capped",
}
_REPORT_FLOAT_FIELDS = {
    "wall_s", "opencl_compute_s", "opencl_other_s", "halo_s", "mpi_halo_s",
    "mpi_migration_s", "mpi_balance_s", "mpi_sync_s", "migration_s", "balance_s",
    "gather_s", "unmeasured_s", "compute_s", "transfer_s", "idle_s", "launch_s",
    "device_wait_s", "iteration_s_before", "iteration_s_after", "decision_s",
    "perturbation_factor", "power_budget_watts", "throttle_factor",
}


def _convert_report_row(header_cols, values):
    """Maps a CSV data row against its #section header columns with typed values."""
    record = {}
    for col, raw in zip(header_cols, values):
        val = raw.strip()
        if col in _REPORT_INT_FIELDS:
            try:
                record[col] = int(val)
            except ValueError:
                record[col] = 0
        elif col in _REPORT_FLOAT_FIELDS:
            try:
                record[col] = float(val)
            except ValueError:
                record[col] = 0.0
        else:
            record[col] = val
    return record


def is_timing_report_csv(csv_path: Path) -> bool:
    """Returns True when csv_path starts with upstream write_timing_report()'s #run header."""
    if not csv_path.exists():
        return False
    try:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    return stripped.startswith("#run,")
    except OSError:
        return False
    return False


def parse_timing_report_csv(report_path: Path):
    """
    Parses upstream Runtime::write_timing_report() structured CSV sections:
    #run, #timeline, #device, #process, #balance.
    """
    if not report_path or not report_path.exists():
        return None

    sections = {}
    current_section = None
    current_cols = []

    result = {
        "report_path": str(report_path),
        "run": None,
        "timeline": None,
        "devices": [],
        "processes": [],
        "balance": [],
    }

    try:
        with open(report_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    continue
                first = row[0].strip()
                if first.startswith("#"):
                    current_section = first[1:]
                    current_cols = [c.strip() for c in row[1:]]
                    sections[current_section] = current_cols
                    continue
                if current_section is None:
                    continue
                if first != current_section:
                    continue
                record = _convert_report_row(current_cols, row[1:])
                if current_section == "run":
                    result["run"] = record
                elif current_section == "timeline":
                    result["timeline"] = record
                elif current_section == "device":
                    result["devices"].append(record)
                elif current_section == "process":
                    result["processes"].append(record)
                elif current_section == "balance":
                    result["balance"].append(record)
    except OSError:
        return None

    if result["run"] is None and result["timeline"] is None and not result["devices"]:
        return None
    return result


def _parse_load_entries(section_text: str):
    """Parses part[...] load lines from a Historical Evidence balance block."""
    return [
        {
            "part": int(m.group(1)),
            "cum_pct": float(m.group(2)),
            "share_pct": float(m.group(3)),
            "device_global": int(m.group(4)),
            "rank": int(m.group(5)),
            "local_index": int(m.group(6)),
            "offset": int(m.group(7)),
            "count": int(m.group(8)),
        }
        for m in _LOAD_LINE_RE.finditer(section_text)
    ]


def parse_balance_metrics(log_text: str):
    """
    Parses all Historical Evidence [DCL][balance-metrics] blocks from log text into
    structured records with iteration, timing intervals, and OLD/CURRENT loads.
    """
    metrics = []
    blocks = re.split(r"(?=\[DCL\]\[balance-metrics\]\s+iteration=)", log_text)

    for block in blocks:
        iter_match = re.search(r"\[DCL\]\[balance-metrics\]\s+iteration=(\d+)", block)
        if not iter_match:
            continue

        def extract_float(field_name, default=0.0):
            m = re.search(rf"{field_name}=([0-9.eE+-]+)", block)
            return float(m.group(1)) if m else default

        old_match = re.search(r"OLD loads:(.*?)(?:CURRENT loads:|$)", block, re.DOTALL)
        curr_match = re.search(r"CURRENT loads:(.*?)(?:\[DCL\]|$)", block, re.DOTALL)

        metrics.append({
            "iteration": int(iter_match.group(1)),
            "T_compute_interval_s": extract_float("T_compute_interval_s"),
            "T_comm_interval_s": extract_float("T_comm_interval_s"),
            "T_balance_total_s": extract_float("T_balance_total_s"),
            "T_rebalance_apply_s": extract_float("T_rebalance_apply_s"),
            "T_compute_accum_s": extract_float("T_compute_accum_s"),
            "T_comm_accum_s": extract_float("T_comm_accum_s"),
            "T_balance_accum_s": extract_float("T_balance_accum_s"),
            "T_rebalance_apply_accum_s": extract_float("T_rebalance_apply_accum_s"),
            "old_loads": _parse_load_entries(old_match.group(1)) if old_match else [],
            "current_loads": _parse_load_entries(curr_match.group(1)) if curr_match else [],
        })

    return metrics


def parse_log_file(log_path: Path):
    """
    Parses a single benchmark stdout log file.
    Extracts execution summary, device inventory, total wall time, loop time, and
    Historical Evidence balance metrics.
    """
    if not log_path or not log_path.exists():
        return None

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    summary_match = re.search(
        r"\[(\w+)\]\s+algo=(\w+)\s+mode=(\w+)\s+total_time_s=([0-9.eE+-]+)"
        r"(?:\s+loop_time_s=([0-9.eE+-]+))?\s+num_rebalances=(\d+)\s+avg_rebalance_gain=([0-9.eE+-]+)",
        content,
    )

    benchmark = algo = mode = None
    total_time_s = loop_time_s = num_rebalances = avg_rebalance_gain = None

    if summary_match:
        benchmark = summary_match.group(1).lower()
        algo = summary_match.group(2).lower()
        mode = summary_match.group(3)
        total_time_s = float(summary_match.group(4))
        if summary_match.group(5) is not None:
            loop_time_s = float(summary_match.group(5))
        num_rebalances = int(summary_match.group(6))
        avg_rebalance_gain = float(summary_match.group(7))
    else:
        for attr, pattern, cast in (
            ("total_time_s", r"total_time_s=([0-9.eE+-]+)", float),
            ("benchmark", r"\[(kneighbor|leanmd)\]", lambda s: s.lower()),
            ("algo", r"algo=(\w+)", lambda s: s.lower()),
            ("mode", r"mode=(\w+)", str),
            ("num_rebalances", r"num_rebalances=(\d+)", int),
            ("avg_rebalance_gain", r"avg_rebalance_gain=([0-9.eE+-]+)", float),
        ):
            m = re.search(pattern, content, re.IGNORECASE if attr == "benchmark" else 0)
            if m:
                if attr == "total_time_s":
                    total_time_s = cast(m.group(1))
                elif attr == "benchmark":
                    benchmark = cast(m.group(1))
                elif attr == "algo":
                    algo = cast(m.group(1))
                elif attr == "mode":
                    mode = cast(m.group(1))
                elif attr == "num_rebalances":
                    num_rebalances = cast(m.group(1))
                elif attr == "avg_rebalance_gain":
                    avg_rebalance_gain = cast(m.group(1))

    if loop_time_s is None:
        loop_time_match = re.search(r"loop_time_s[:=]\s*([0-9.eE+-]+)", content)
        if loop_time_match:
            loop_time_s = float(loop_time_match.group(1))

    devices = [
        {
            "rank": int(m.group(1)),
            "global_index": int(m.group(2)),
            "name": m.group(3).strip(),
        }
        for m in re.finditer(r"\[device\]\s+rank=(\d+)\s+global_index=(\d+)\s+name=([^\n]+)", content)
    ]

    host_match = re.search(r"Master compute host\s*=\s*(\S+)", content)

    return {
        "log_path": str(log_path),
        "benchmark": benchmark,
        "algo": algo,
        "mode": mode,
        "total_time_s": total_time_s,
        "loop_time_s": loop_time_s,
        "num_rebalances": num_rebalances,
        "avg_rebalance_gain": avg_rebalance_gain,
        "devices": devices,
        "host": host_match.group(1) if host_match else None,
        "balance_metrics": parse_balance_metrics(content),
    }


def parse_csv_file(csv_path: Path):
    """
    Parses a per-iteration CSV file (iteration,step_time_s,action,rebalanced,...).
    Returns None if missing or if the file is an upstream #run timing report CSV.
    """
    if not csv_path or not csv_path.exists() or is_timing_report_csv(csv_path):
        return None

    steps = []
    step_sum_s = 0.0
    rebalance_count = 0
    rebalance_gains = []
    execution_modes = set()

    try:
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step_time = float(row.get("step_time_s", 0.0))
                    is_rebal = int(row.get("rebalanced", 0)) == 1 or row.get("action") == "rebalance"
                    gain = float(row.get("rebalance_gain", 0.0))
                    mode = (row.get("execution_mode") or "").strip()

                    step_sum_s += step_time
                    if is_rebal:
                        rebalance_count += 1
                        rebalance_gains.append(gain)
                    if mode:
                        execution_modes.add(mode)

                    steps.append({
                        "iteration": int(row.get("iteration", len(steps))),
                        "step_time_s": step_time,
                        "action": row.get("action", "compute"),
                        "rebalanced": 1 if is_rebal else 0,
                        "rebalance_gain": gain,
                        "execution_mode": mode,
                    })
                except (ValueError, TypeError):
                    continue
    except OSError:
        return None

    return {
        "csv_path": str(csv_path),
        "steps": steps,
        "step_count": len(steps),
        "step_sum_s": step_sum_s,
        "rebalance_count": rebalance_count,
        "avg_rebalance_gain": statistics.mean(rebalance_gains) if rebalance_gains else 0.0,
        "execution_modes": execution_modes,
    }


def parse_hardware_telemetry(telemetry_path: Path):
    """
    Parses hardware_telemetry.log into structured telemetry records.
    Handles multi-node (host, rank tagged), unavailable markers, and historical formats.
    """
    if not telemetry_path.exists():
        return []

    records = []
    try:
        with open(telemetry_path, encoding="utf-8", errors="replace") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return []

    def clean_num(val_str):
        m = re.search(r"([0-9.]+)", val_str)
        return float(m.group(1)) if m else 0.0

    def clean_int(val_str):
        m = re.search(r"(\d+)", val_str)
        return int(m.group(1)) if m else 0

    for line in lines:
        if line.startswith("# [telemetry-unavailable]"):
            host_m = re.search(r"host=(\S+)", line)
            rank_m = re.search(r"rank=(\d+)", line)
            reason_m = re.search(r"reason=(.*)$", line)
            records.append({
                "host": host_m.group(1) if host_m else None,
                "rank": int(rank_m.group(1)) if rank_m else None,
                "timestamp": "",
                "name": "",
                "temperature_gpu": 0,
                "utilization_gpu_pct": 0,
                "clocks_graphics_mhz": 0,
                "clocks_memory_mhz": 0,
                "power_draw_w": 0.0,
                "throttle_reasons": "",
                "available": False,
                "unavailable_reason": reason_m.group(1).strip() if reason_m else "unknown",
            })
            continue

        if line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        if any(p.lower() in ("timestamp", "timestamp.gpu", "name", "utilization.gpu", "clocks.current.graphics") for p in parts):
            continue
        if parts[0].lower() in ("host", "timestamp"):
            continue
        if any("shared_workload" in p.lower() or "compute-apps" in p.lower() for p in parts):
            continue

        host = None
        rank = None
        if len(parts) >= 10 and re.match(r"^\d+$", parts[1]):
            host, rank = parts[0], int(parts[1])
            timestamp, name = parts[2], parts[3]
            temp, util_gpu = clean_int(parts[4]), clean_int(parts[5])
            clocks_graphics, clocks_memory = clean_int(parts[6]), clean_int(parts[7])
            power = clean_num(parts[8])
            throttle = ", ".join(parts[9:]).strip()
        elif len(parts) >= 9 and re.search(r"\d{4}[-/]", parts[1]):
            host = parts[0]
            timestamp, name = parts[1], parts[2]
            temp, util_gpu = clean_int(parts[3]), clean_int(parts[4])
            clocks_graphics, clocks_memory = clean_int(parts[5]), clean_int(parts[6])
            power = clean_num(parts[7])
            throttle = ", ".join(parts[8:]).strip()
        else:
            timestamp = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            temp = clean_int(parts[2]) if len(parts) > 2 else 0
            util_gpu = clean_int(parts[3]) if len(parts) > 3 else 0
            clocks_graphics = clean_int(parts[4]) if len(parts) > 4 else 0
            clocks_memory = clean_int(parts[5]) if len(parts) > 5 else 0
            power = clean_num(parts[6]) if len(parts) > 6 else 0.0
            throttle = ", ".join(parts[7:]).strip() if len(parts) > 7 else ""

        records.append({
            "host": host,
            "rank": rank,
            "timestamp": timestamp,
            "name": name,
            "temperature_gpu": temp,
            "utilization_gpu_pct": util_gpu,
            "clocks_graphics_mhz": clocks_graphics,
            "clocks_memory_mhz": clocks_memory,
            "power_draw_w": power,
            "throttle_reasons": throttle,
            "available": True,
        })

    return records


parse_telemetry = parse_hardware_telemetry


def identify_run_attributes(filename: str, log_data=None, report_data=None):
    """Infers benchmark, algo, seed, repetition, stage, and step from filename and parsed data."""
    lower = filename.lower()

    if "warmup" in lower:
        stage = "warmup"
    elif "aba" in lower:
        stage = "aba"
    else:
        stage = "matrix"

    if "kneighbor" in lower:
        benchmark = "kneighbor"
    elif "leanmd" in lower:
        benchmark = "leanmd"
    elif log_data and log_data.get("benchmark"):
        benchmark = log_data["benchmark"]
    else:
        benchmark = "unknown"

    if "hwtopolb" in lower:
        algo = "hwtopolb"
    elif "erad" in lower:
        algo = "erad"
    elif log_data and log_data.get("algo"):
        algo = log_data["algo"]
    elif report_data and report_data.get("balance"):
        algo = report_data["balance"][0].get("algorithm") or "unknown"
    else:
        algo = "unknown"

    seed = None
    seed_match = re.search(r"(?:seed[_-]?)(\d+)", lower)
    if seed_match:
        seed = int(seed_match.group(1))
    else:
        end_match = re.search(r"[_-](42|123|456)(?:[._-]|$)", lower)
        if end_match:
            seed = int(end_match.group(1))
        elif report_data and report_data.get("balance"):
            seed = report_data["balance"][0].get("seed")
        if seed is None:
            seed = 123 if stage == "aba" else 42

    step_match = re.search(r"(?:aba[_-]?|step[_-]?)(\d+)", lower)
    step = int(step_match.group(1)) if step_match else None

    rep_match = re.search(r"(?:rep[_-]?|iteration[_-]?)(\d+)", lower)
    if rep_match:
        repetition = int(rep_match.group(1))
    elif stage == "aba" and step is not None:
        repetition = step
    else:
        repetition = 1

    return benchmark, algo, seed, repetition, stage, step


def _derive_report_path(log_path, csv_path, metadata):
    """Locates the upstream *_report.csv companion file when not explicitly provided."""
    base_dir = None
    for candidate in (csv_path, log_path):
        if candidate is not None:
            base_dir = candidate.parent
            break

    if isinstance(metadata, dict) and metadata.get("report_csv_file") and base_dir is not None:
        explicit = base_dir / metadata["report_csv_file"]
        if explicit.exists():
            return explicit

    for candidate in (csv_path, log_path):
        if candidate is not None:
            stem = candidate.stem
            if stem.endswith("_report"):
                return candidate
            derived = candidate.with_name(f"{stem}_report.csv")
            if derived.exists():
                return derived
    return None


def _extract_report_phase_metrics(report_data):
    """Extracts T_compute_s, T_comm_s, T_balance_s, and bytes_moved from upstream report CSV."""
    tl = report_data.get("timeline") or {}
    t_compute = tl.get("opencl_compute_s", 0.0)
    if t_compute == 0.0 and report_data.get("devices"):
        t_compute = max(d.get("compute_s", 0.0) for d in report_data["devices"])

    t_comm = (
        tl.get("halo_s", 0.0)
        + tl.get("mpi_halo_s", 0.0)
        + tl.get("mpi_migration_s", 0.0)
        + tl.get("mpi_balance_s", 0.0)
        + tl.get("mpi_sync_s", 0.0)
    )
    t_bal = tl.get("balance_s", 0.0) + tl.get("migration_s", 0.0)
    if t_bal == 0.0 and report_data.get("balance"):
        t_bal = sum(b.get("decision_s", 0.0) + b.get("migration_s", 0.0) for b in report_data["balance"])

    run_sec = report_data.get("run") or {}
    bytes_moved = run_sec.get("bytes_moved", 0)
    if bytes_moved == 0 and report_data.get("balance"):
        bytes_moved = sum(b.get("bytes_moved", 0) for b in report_data["balance"] if b.get("applied"))

    return t_compute, t_comm, t_bal, bytes_moved


def _resolve_expected_iterations(default_expected, manifest_run, metadata):
    """Resolves expected iteration count from manifest/metadata overrides when present."""
    for src in (manifest_run, metadata):
        if not isinstance(src, dict):
            continue
        raw = None
        if isinstance(src.get("parameters"), dict):
            raw = src["parameters"].get("iterations")
        if raw is None:
            raw = src.get("iterations", src.get("expected_iterations"))
        if raw is not None:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return default_expected


def parse_run(log_path: Path, csv_path: Path, expected_iterations=100, metadata=None, manifest_run=None, report_csv_path: Path = None):
    """
    Combines stdout log, per-iteration CSV, and upstream structured report CSV
    into a unified run record. Respects manifest failure/timeout overrides.
    """
    if manifest_run is None and metadata is not None:
        if isinstance(metadata, dict) and isinstance(metadata.get("manifest_run"), dict):
            manifest_run = metadata["manifest_run"]
        else:
            manifest_run = metadata
    if metadata is None and manifest_run is not None:
        metadata = manifest_run

    expected_iterations = _resolve_expected_iterations(expected_iterations, manifest_run, metadata)

    log_data = parse_log_file(log_path) if log_path and log_path.exists() else None
    csv_data = parse_csv_file(csv_path) if csv_path and csv_path.exists() else None
    if report_csv_path is None:
        report_csv_path = _derive_report_path(log_path, csv_path, metadata)
    report_data = parse_timing_report_csv(report_csv_path) if report_csv_path and report_csv_path.exists() else None

    fname = (
        log_path.name if log_path
        else (csv_path.name if csv_path else (report_csv_path.name if report_csv_path else ""))
    )
    bench, algo, seed, rep, stage, step = identify_run_attributes(fname, log_data, report_data)

    if metadata:
        bench = metadata.get("benchmark", bench)
        algo = metadata.get("algo") or metadata.get("algorithm") or algo
        seed = metadata.get("seed", seed)
        rep = metadata.get("repetition", rep)
        stage = metadata.get("stage", stage)
        if metadata.get("step") is not None:
            step = metadata.get("step")

    report_wall = report_data["run"].get("wall_s") if (report_data and report_data.get("run")) else None
    wall_s = log_data["total_time_s"] if log_data and log_data.get("total_time_s") is not None else report_wall
    loop_s = log_data["loop_time_s"] if log_data and log_data.get("loop_time_s") is not None else report_wall

    if metadata:
        if metadata.get("wall_s") is not None:
            wall_s = metadata.get("wall_s")
        if metadata.get("loop_s") is not None:
            loop_s = metadata.get("loop_s")
        elif metadata.get("loop_time_s") is not None:
            loop_s = metadata.get("loop_time_s")

    primary_s = loop_s if loop_s is not None else wall_s
    step_sum_s = csv_data["step_sum_s"] if csv_data else None
    unaccounted_s = (wall_s - step_sum_s) if (wall_s is not None and step_sum_s is not None) else None

    if csv_data:
        rebalance_count = csv_data["rebalance_count"]
    elif report_data and report_data.get("run"):
        rebalance_count = report_data["run"].get("balance_applied", 0)
    elif log_data and log_data.get("num_rebalances") is not None:
        rebalance_count = log_data["num_rebalances"]
    else:
        rebalance_count = 0

    t_compute_s = t_comm_s = t_bal_s = 0.0
    boundary_shifts = 0
    bytes_moved = 0

    if log_data and log_data.get("balance_metrics"):
        bm = log_data["balance_metrics"]
        last = bm[-1]
        t_compute_s = last.get("T_compute_accum_s") or sum(b.get("T_compute_interval_s", 0.0) for b in bm)
        t_comm_s = last.get("T_comm_accum_s") or sum(b.get("T_comm_interval_s", 0.0) for b in bm)
        t_bal_s = last.get("T_balance_accum_s") or sum(b.get("T_balance_total_s", 0.0) for b in bm)
        for b in bm:
            for o, c in zip(b.get("old_loads", []), b.get("current_loads", [])):
                boundary_shifts += abs(c.get("count", 0) - o.get("count", 0)) // 2
    elif report_data:
        t_compute_s, t_comm_s, t_bal_s, bytes_moved = _extract_report_phase_metrics(report_data)

    status = "SUCCESS"
    if log_data is None and csv_data is None and report_data is None:
        status = "MISSING_ALL"
    elif log_data is None and report_data is None:
        status = "MISSING_LOG"
    elif wall_s is None and loop_s is None:
        status = "FAILED"
    elif csv_data is None and report_data is None:
        status = "MISSING_CSV"
    else:
        actual_iters = (
            csv_data["step_count"] if csv_data
            else (report_data["run"].get("iterations", 0) if report_data and report_data.get("run") else 0)
        )
        if expected_iterations and actual_iters < expected_iterations:
            status = "PARTIAL"
        elif csv_data and any(m != "gpu_opencl" for m in csv_data["execution_modes"] if m):
            status = "CPU_FALLBACK"
        elif csv_data is None and report_data and report_data.get("devices"):
            if not any(d.get("type", "").upper() == "GPU" for d in report_data["devices"]):
                status = "CPU_FALLBACK"

    if manifest_run and isinstance(manifest_run, dict):
        m_status = str(manifest_run.get("status") or "").lower()
        exit_code = manifest_run.get("exit_code")
        try:
            exit_code_num = int(exit_code) if exit_code is not None else None
        except (ValueError, TypeError):
            exit_code_num = exit_code

        if m_status in ("failed", "timeout") or (exit_code_num is not None and exit_code_num != 0):
            status = "TIMEOUT" if (m_status == "timeout" or exit_code_num == 124) else "FAILED"

    return {
        "benchmark": bench,
        "algo": algo,
        "seed": seed,
        "repetition": rep,
        "step": step,
        "stage": stage,
        "wall_s": wall_s,
        "loop_s": loop_s,
        "primary_s": primary_s,
        "step_sum_s": step_sum_s,
        "unaccounted_s": unaccounted_s,
        "rebalance_count": rebalance_count,
        "T_compute_s": t_compute_s,
        "T_comm_s": t_comm_s,
        "T_balance_s": t_bal_s,
        "boundary_shifts": boundary_shifts,
        "bytes_moved": bytes_moved,
        "status": status,
        "log_path": str(log_path) if log_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "report_csv_path": str(report_csv_path) if report_csv_path else None,
        "log_data": log_data,
        "csv_data": csv_data,
        "report_data": report_data,
    }


def discover_runs(output_dir: Path, expected_iterations=100):
    """
    Discovers all benchmark runs in output_dir using run_manifest.json if present,
    or via directory scanning. Prevents double-counting *_report.csv companions.
    Raises ValueError on malformed JSON manifests.
    """
    runs = []
    manifest_path = None
    for candidate in (
        output_dir / "run_manifest.json",
        output_dir / "manifest.json",
        output_dir / "super_matrix_manifest.json",
    ):
        if candidate.exists():
            manifest_path = candidate
            break

    processed_logs = set()
    processed_csvs = set()
    processed_reports = set()
    processed_stems = set()
    effective_expected_iterations = expected_iterations

    if manifest_path and manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse manifest {manifest_path}: {e}") from e

        if isinstance(manifest_data, dict):
            manifest_params = manifest_data.get("parameters") or {}
            if isinstance(manifest_params, dict):
                effective_expected_iterations = _resolve_expected_iterations(
                    effective_expected_iterations, {"parameters": manifest_params}, None
                )

        raw_runs = manifest_data.get("runs", manifest_data if isinstance(manifest_data, list) else [])
        for item in raw_runs:
            log_p = output_dir / item["log_file"] if item.get("log_file") else None
            csv_p = output_dir / item["csv_file"] if item.get("csv_file") else None
            rep_p = output_dir / item["report_csv_file"] if item.get("report_csv_file") else None
            run_expected_iters = _resolve_expected_iterations(effective_expected_iterations, item, None)
            run = parse_run(
                log_p, csv_p, run_expected_iters,
                metadata=item, manifest_run=item, report_csv_path=rep_p
            )
            runs.append(run)
            for path_obj, seen_set in (
                (log_p, processed_logs),
                (csv_p, processed_csvs),
                (rep_p, processed_reports),
            ):
                if path_obj and path_obj.exists():
                    seen_set.add(path_obj.resolve())
                    processed_stems.add(path_obj.stem)
            if run.get("report_csv_path"):
                processed_reports.add(Path(run["report_csv_path"]).resolve())

    for log_p in sorted(output_dir.glob("*.log")):
        if log_p.name in ("hardware_telemetry.log", "run_matrix.log", "build.log"):
            continue
        if log_p.resolve() in processed_logs:
            continue
        base = log_p.stem
        csv_p = output_dir / f"{base}.csv"
        rep_p = output_dir / f"{base}_report.csv"
        run = parse_run(
            log_p,
            csv_p if csv_p.exists() else None,
            effective_expected_iterations,
            report_csv_path=rep_p if rep_p.exists() else None,
        )
        runs.append(run)
        processed_logs.add(log_p.resolve())
        processed_stems.add(base)
        if csv_p.exists():
            processed_csvs.add(csv_p.resolve())
        if rep_p.exists():
            processed_reports.add(rep_p.resolve())

    ignored_csvs = {"super_matrix_summary.csv", "super_matrix_paired.csv", "comparison_summary.csv"}
    for csv_p in sorted(output_dir.glob("*.csv")):
        if csv_p.name in ignored_csvs:
            continue
        resolved = csv_p.resolve()
        if resolved in processed_csvs or resolved in processed_reports:
            continue

        if csv_p.stem.endswith("_report") or is_timing_report_csv(csv_p):
            base_stem = csv_p.stem[:-7] if csv_p.stem.endswith("_report") else csv_p.stem
            if base_stem in processed_stems:
                continue
            log_p = output_dir / f"{base_stem}.log"
            run = parse_run(
                log_p if log_p.exists() else None,
                None,
                effective_expected_iterations,
                report_csv_path=csv_p,
            )
            runs.append(run)
            processed_reports.add(resolved)
            processed_stems.add(base_stem)
        else:
            base = csv_p.stem
            log_p = output_dir / f"{base}.log"
            rep_p = output_dir / f"{base}_report.csv"
            run = parse_run(
                log_p if log_p.exists() else None,
                csv_p,
                effective_expected_iterations,
                report_csv_path=rep_p if rep_p.exists() else None,
            )
            runs.append(run)
            processed_csvs.add(resolved)
            processed_stems.add(base)
            if rep_p.exists():
                processed_reports.add(rep_p.resolve())

    return runs


def _run_timing_fields(r):
    """Returns (wall_s, loop_s, primary_s) for a run record."""
    wall_s = r.get("wall_s") if r.get("wall_s") is not None else r.get("total_time_s")
    loop_s = r.get("loop_s") if r.get("loop_s") is not None else r.get("loop_time_s")
    primary_s = loop_s if loop_s is not None else wall_s
    return wall_s, loop_s, primary_s


def _make_pair_entry(bench, seed, rep, stage_name, step, erad_run, hw_run):
    """Builds one paired ERAD vs HWTOPOLB comparison entry."""
    erad_wall, erad_loop, erad_t = _run_timing_fields(erad_run)
    hw_wall, hw_loop, hw_t = _run_timing_fields(hw_run)
    ratio = (erad_t / hw_t) if (hw_t is not None and hw_t > 0 and erad_t is not None) else None
    speedup_pct = (((erad_t - hw_t) / erad_t) * 100.0) if (erad_t is not None and erad_t > 0 and hw_t is not None) else None
    return {
        "benchmark": bench,
        "seed": seed,
        "repetition": rep,
        "stage": stage_name,
        "step": step,
        "erad_wall_s": erad_wall,
        "hwtopolb_wall_s": hw_wall,
        "erad_loop_s": erad_loop,
        "hwtopolb_loop_s": hw_loop,
        "ratio": ratio,
        "speedup_pct": speedup_pct,
    }


def compute_paired_analysis(runs, stage=None):
    """
    Pairs ERAD and HWTOPOLB runs sharing (benchmark, repetition, seed, stage).
    For A/B/A runs, pairs step 1 vs step 2 and step 3 vs step 2 without overwriting.
    Prioritizes loop_s over wall_s when present.
    """
    pairs = []
    grouped = {}
    aba_grouped = {}

    for r in runs:
        _, _, timing = _run_timing_fields(r)
        if r.get("status") != "SUCCESS" or timing is None:
            continue
        r_stage = r.get("stage", "matrix")
        if stage is not None and r_stage != stage:
            continue
        if r_stage == "aba":
            key = (r["benchmark"], r["seed"])
            step = r.get("step")
            if step is not None:
                aba_grouped.setdefault(key, {})[step] = r
            else:
                aba_grouped.setdefault(key, {}).setdefault(r["algo"], []).append(r)
        else:
            key = (r["benchmark"], r["seed"], r.get("repetition", 1), r_stage)
            grouped.setdefault(key, {})[r["algo"]] = r

    for (bench, seed, rep, stage_name), algos in sorted(
        grouped.items(),
        key=lambda item: (
            str(item[0][0]),
            int(item[0][1]) if item[0][1] is not None else -1,
            int(item[0][2]) if item[0][2] is not None else -1,
            str(item[0][3]),
        ),
    ):
        if "erad" in algos and "hwtopolb" in algos:
            pairs.append(_make_pair_entry(bench, seed, rep, stage_name, None, algos["erad"], algos["hwtopolb"]))

    for (bench, seed), steps in sorted(
        aba_grouped.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1]) if item[0][1] is not None else -1),
    ):
        step1, step2, step3 = steps.get(1), steps.get(2), steps.get(3)
        if step1 and step2:
            pairs.append(_make_pair_entry(bench, seed, 1, "aba", 1, step1, step2))
        if step3 and step2:
            pairs.append(_make_pair_entry(bench, seed, 3, "aba", 3, step3, step2))

    return pairs


def compute_aba_drift_analysis(runs):
    """
    Computes A/B/A thermal drift metrics comparing ERAD step 1 vs step 3
    and HWTOPOLB step 2 against the bracketed ERAD mean.
    """
    aba_runs = [
        r for r in runs
        if r.get("stage") == "aba" and r.get("status") == "SUCCESS" and _run_timing_fields(r)[2] is not None
    ]
    if not aba_runs:
        return []

    grouped = {}
    for r in aba_runs:
        step = r.get("step")
        if step is not None:
            grouped.setdefault((r["benchmark"], r["seed"]), {})[step] = r

    drift_records = []
    for (bench, seed), steps in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1]) if item[0][1] is not None else -1),
    ):
        step1, step2, step3 = steps.get(1), steps.get(2), steps.get(3)
        if not (step1 and step2 and step3):
            continue
        t1, t2, t3 = _run_timing_fields(step1)[2], _run_timing_fields(step2)[2], _run_timing_fields(step3)[2]
        if t1 is None or t2 is None or t3 is None:
            continue

        drift_delta_s = t3 - t1
        bracketed_mean = (t1 + t3) / 2.0
        drift_records.append({
            "benchmark": bench,
            "seed": seed,
            "erad_step1_s": t1,
            "hwtopolb_step2_s": t2,
            "erad_step3_s": t3,
            "drift_delta_s": drift_delta_s,
            "drift_pct": (drift_delta_s / t1) * 100.0 if t1 > 0 else 0.0,
            "drift_ratio": t3 / t1 if t1 > 0 else 1.0,
            "bracketed_erad_mean_s": bracketed_mean,
            "bracketed_ratio": bracketed_mean / t2 if t2 > 0 else None,
            "bracketed_speedup_pct": ((bracketed_mean - t2) / bracketed_mean) * 100.0 if bracketed_mean > 0 else None,
            "ratio_step1": t1 / t2 if t2 > 0 else None,
            "ratio_step3": t3 / t2 if t2 > 0 else None,
        })

    return drift_records


def compute_matrix_statistics(runs, stage="matrix"):
    """
    Computes median, mean, min, max, and stddev across repetitions per (benchmark, algo, seed).
    Strictly filters by stage (default 'matrix') with zero fallback to other stages.
    """
    grouped = {}
    for r in runs:
        _, _, timing = _run_timing_fields(r)
        if r.get("status") != "SUCCESS" or timing is None:
            continue
        if stage is not None and r.get("stage") != stage:
            continue
        grouped.setdefault((r["benchmark"], r["algo"], r["seed"]), []).append(r)

    stats = []
    for (bench, algo, seed), group in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), int(item[0][2]) if item[0][2] is not None else -1),
    ):
        primary_times = [_run_timing_fields(r)[2] for r in group]
        wall_times = [_run_timing_fields(r)[0] for r in group if _run_timing_fields(r)[0] is not None]
        loop_times = [_run_timing_fields(r)[1] for r in group if _run_timing_fields(r)[1] is not None]
        unaccounted = [r["unaccounted_s"] for r in group if r.get("unaccounted_s") is not None]
        rebals = [r["rebalance_count"] for r in group if r.get("rebalance_count") is not None]

        med_primary = statistics.median(primary_times)
        mean_primary = statistics.mean(primary_times)
        min_primary = min(primary_times)
        max_primary = max(primary_times)
        stdev_primary = statistics.stdev(primary_times) if len(primary_times) >= 2 else 0.0

        stats.append({
            "benchmark": bench,
            "algo": algo,
            "seed": seed,
            "stage": stage if stage is not None else "all",
            "repetition_count": len(primary_times),
            "median_s": med_primary,
            "mean_s": mean_primary,
            "min_s": min_primary,
            "max_s": max_primary,
            "stddev_s": stdev_primary,
            "median_wall_s": statistics.median(wall_times) if wall_times else med_primary,
            "mean_wall_s": statistics.mean(wall_times) if wall_times else mean_primary,
            "min_wall_s": min(wall_times) if wall_times else min_primary,
            "max_wall_s": max(wall_times) if wall_times else max_primary,
            "stddev_wall_s": statistics.stdev(wall_times) if len(wall_times) >= 2 else 0.0,
            "median_loop_s": statistics.median(loop_times) if loop_times else None,
            "mean_loop_s": statistics.mean(loop_times) if loop_times else None,
            "min_loop_s": min(loop_times) if loop_times else None,
            "max_loop_s": max(loop_times) if loop_times else None,
            "stddev_loop_s": statistics.stdev(loop_times) if len(loop_times) >= 2 else (0.0 if loop_times else None),
            "median_unaccounted_s": statistics.median(unaccounted) if unaccounted else 0.0,
            "mean_rebalance_count": statistics.mean(rebals) if rebals else 0.0,
            "median_compute_s": statistics.median([r.get("T_compute_s", 0.0) for r in group]),
            "median_comm_s": statistics.median([r.get("T_comm_s", 0.0) for r in group]),
            "median_balance_s": statistics.median([r.get("T_balance_s", 0.0) for r in group]),
            "total_boundary_shifts": sum(r.get("boundary_shifts", 0) for r in group),
        })

    return stats


compute_summary_statistics = compute_matrix_statistics


def compute_stage_breakdowns(runs):
    """Provides separate summary statistics breakdowns for matrix, warmup, and aba stages."""
    return {
        "matrix": compute_matrix_statistics(runs, stage="matrix"),
        "warmup": compute_matrix_statistics(runs, stage="warmup"),
        "aba": compute_matrix_statistics(runs, stage="aba"),
    }


def generate_artifacts(output_dir: Path, runs, telemetry_records=None):
    """
    Generates super_matrix_summary.csv, super_matrix_paired.csv,
    hardware_telemetry.json, and SUPER_JOB_REPORT.md in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "super_matrix_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "algo", "repetition", "seed", "stage",
            "wall_s", "loop_s", "step_sum_s", "unaccounted_s", "rebalance_count",
            "T_compute_s", "T_comm_s", "T_balance_s", "boundary_shifts", "status",
        ])
        for r in runs:
            writer.writerow([
                r["benchmark"],
                r["algo"],
                r.get("repetition", ""),
                r["seed"],
                r.get("stage", ""),
                f"{r['wall_s']:.6f}" if r.get("wall_s") is not None else "",
                f"{r['loop_s']:.6f}" if r.get("loop_s") is not None else "",
                f"{r['step_sum_s']:.6f}" if r.get("step_sum_s") is not None else "",
                f"{r['unaccounted_s']:.6f}" if r.get("unaccounted_s") is not None else "",
                r.get("rebalance_count", 0),
                f"{r.get('T_compute_s', 0.0):.6f}",
                f"{r.get('T_comm_s', 0.0):.6f}",
                f"{r.get('T_balance_s', 0.0):.6f}",
                r.get("boundary_shifts", 0),
                r["status"],
            ])

    paired = compute_paired_analysis(runs)
    paired_csv_path = output_dir / "super_matrix_paired.csv"
    with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark", "seed", "repetition", "stage", "step",
            "erad_wall_s", "hwtopolb_wall_s", "erad_loop_s", "hwtopolb_loop_s",
            "ratio", "speedup_pct",
        ])
        for p in paired:
            writer.writerow([
                p["benchmark"],
                p["seed"],
                p.get("repetition", ""),
                p["stage"],
                p.get("step") if p.get("step") is not None else "",
                f"{p['erad_wall_s']:.6f}" if p.get("erad_wall_s") is not None else "",
                f"{p['hwtopolb_wall_s']:.6f}" if p.get("hwtopolb_wall_s") is not None else "",
                f"{p['erad_loop_s']:.6f}" if p.get("erad_loop_s") is not None else "",
                f"{p['hwtopolb_loop_s']:.6f}" if p.get("hwtopolb_loop_s") is not None else "",
                f"{p['ratio']:.4f}" if p["ratio"] is not None else "",
                f"{p['speedup_pct']:.2f}" if p["speedup_pct"] is not None else "",
            ])

    telemetry_log_path = output_dir / "hardware_telemetry.log"
    if telemetry_records is None and telemetry_log_path.exists():
        telemetry_records = parse_hardware_telemetry(telemetry_log_path)

    if telemetry_records:
        with open(output_dir / "hardware_telemetry.json", "w", encoding="utf-8") as f:
            json.dump(telemetry_records, f, indent=2)

    generate_markdown_report(output_dir / "SUPER_JOB_REPORT.md", runs, paired, telemetry_records, output_dir)


def _collect_environment_metadata(runs, telemetry_records):
    """Collects unique hosts, devices, and execution modes across logs, reports, and telemetry."""
    devices = set()
    hosts = set()
    modes = set()

    for r in runs:
        ld = r.get("log_data") or {}
        if ld.get("host"):
            hosts.add(ld["host"])
        if ld.get("mode"):
            modes.add(ld["mode"])
        for dev in ld.get("devices", []):
            if dev.get("name"):
                devices.add(dev["name"])

        rd = r.get("report_data") or {}
        for proc in rd.get("processes", []):
            if proc.get("host"):
                hosts.add(proc["host"])
        for dev in rd.get("devices", []):
            if dev.get("name"):
                devices.add(dev["name"])

        cd = r.get("csv_data") or {}
        modes.update(m for m in cd.get("execution_modes", ()) if m)

    for tr in telemetry_records or ():
        if tr.get("host"):
            hosts.add(tr["host"])
        if tr.get("name") and tr["name"] != "placeholder":
            devices.add(tr["name"])

    devices_str = ", ".join(sorted(devices)) if devices else "unknown (not observed in telemetry/logs)"
    hosts_str = ", ".join(sorted(hosts)) if hosts else "PBS compute allocation"
    modes_str = ", ".join(sorted(modes)) if modes else "unknown"
    return hosts_str, devices_str, modes_str


def _render_telemetry_section(md, telemetry_records):
    """Appends Section 7 (Hardware Telemetry & Environmental Stability) to Markdown lines."""
    md.append("## 7. Hardware Telemetry & Environmental Stability\n")
    node_groups = {}
    for r in telemetry_records:
        node_groups.setdefault((r.get("host"), r.get("rank")), []).append(r)

    has_node_info = any(h is not None or rk is not None for (h, rk) in node_groups)
    if has_node_info:
        md.append("### Hardware Telemetry by Node & Rank\n")
        md.append("| Host | Rank | Device | Samples | Temp (°C) Min/Mean/Max | Power (W) Min/Mean/Max | Graphics Clock (MHz) Min/Mean/Max | Memory Clock (MHz) Min/Mean/Max | Throttle Reasons |")
        md.append("|---|---:|---|---:|---:|---:|---:|---:|---|")

        sorted_keys = sorted(
            node_groups.keys(),
            key=lambda k: (str(k[0] or ""), k[1] if isinstance(k[1], int) else -1),
        )
        for h, rk in sorted_keys:
            all_recs = node_groups[(h, rk)]
            recs = [r for r in all_recs if r.get("available", True)]
            h_str = str(h) if h is not None else "unknown"
            rk_str = str(rk) if rk is not None else "N/A"
            if not recs:
                reasons = sorted({r.get("unavailable_reason", "unknown") for r in all_recs})
                md.append(f"| {h_str} | {rk_str} | UNAVAILABLE ({'; '.join(reasons)}) | 0 | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |")
                continue
            dev_names = {r["name"] for r in recs if r.get("name") and r["name"] != "placeholder"}
            dev_str = ", ".join(sorted(dev_names)) if dev_names else "placeholder"
            t_list = [r["temperature_gpu"] for r in recs if r.get("temperature_gpu")]
            p_list = [r["power_draw_w"] for r in recs if r.get("power_draw_w")]
            g_list = [r["clocks_graphics_mhz"] for r in recs if r.get("clocks_graphics_mhz")]
            m_list = [r["clocks_memory_mhz"] for r in recs if r.get("clocks_memory_mhz")]

            t_str = f"{min(t_list)} / {statistics.mean(t_list):.1f} / {max(t_list)}" if t_list else "N/A"
            p_str = f"{min(p_list):.1f} / {statistics.mean(p_list):.1f} / {max(p_list):.1f}" if p_list else "N/A"
            g_str = f"{min(g_list)} / {statistics.mean(g_list):.0f} / {max(g_list)}" if g_list else "N/A"
            m_str = f"{min(m_list)} / {statistics.mean(m_list):.0f} / {max(m_list)}" if m_list else "N/A"
            throttle_set = {
                r["throttle_reasons"] for r in recs
                if r.get("throttle_reasons") and r["throttle_reasons"] not in ("", "[N/A]", "0x0000000000000000", "None")
            }
            th_str = ", ".join(sorted(throttle_set)) if throttle_set else "None"
            md.append(f"| {h_str} | {rk_str} | {dev_str} | {len(recs)} | {t_str} | {p_str} | {g_str} | {m_str} | {th_str} |")
        md.append("")

        if len(sorted_keys) >= 2:
            temp_means, clock_means = {}, {}
            for k in sorted_keys:
                recs = [r for r in node_groups[k] if r.get("available", True)]
                t_list = [r["temperature_gpu"] for r in recs if r.get("temperature_gpu")]
                g_list = [r["clocks_graphics_mhz"] for r in recs if r.get("clocks_graphics_mhz")]
                if t_list:
                    temp_means[k] = statistics.mean(t_list)
                if g_list:
                    clock_means[k] = statistics.mean(g_list)
            if temp_means:
                min_k, max_k = min(temp_means, key=temp_means.get), max(temp_means, key=temp_means.get)
                md.append(f"- **Inter-Node Thermal Stability:** Mean temperature spread across nodes is {temp_means[max_k] - temp_means[min_k]:.1f}°C (coolest: `{min_k[0]}` rank {min_k[1]} at {temp_means[min_k]:.1f}°C, warmest: `{max_k[0]}` rank {max_k[1]} at {temp_means[max_k]:.1f}°C).")
            if clock_means:
                min_g, max_g = min(clock_means, key=clock_means.get), max(clock_means, key=clock_means.get)
                delta_g = clock_means[max_g] - clock_means[min_g]
                if delta_g < HEURISTIC_CLOCK_DIVERGENCE_DELTA_MHZ:
                    md.append(f"- **Clock Uniformity:** GPU graphics clocks remained uniform across all nodes (spread: {delta_g:.0f} MHz).")
                else:
                    md.append(f"- **Clock Divergence:** Graphics clock discrepancy detected between nodes: {delta_g:.0f} MHz difference (`{min_g[0]}` vs `{max_g[0]}`).")
            md.append("")

    avail = [r for r in telemetry_records if r.get("available", True)]
    temps = [r["temperature_gpu"] for r in avail if r.get("temperature_gpu")]
    powers = [r["power_draw_w"] for r in avail if r.get("power_draw_w")]
    gfx_clocks = [r["clocks_graphics_mhz"] for r in avail if r.get("clocks_graphics_mhz")]
    mem_clocks = [r["clocks_memory_mhz"] for r in avail if r.get("clocks_memory_mhz")]

    if has_node_info:
        md.append("### Aggregate Cluster Metrics\n")
    md.append("| Metric | Min | Mean | Max |")
    md.append("|---|---:|---:|---:|")
    if temps:
        md.append(f"| GPU Temperature (°C) | {min(temps)} | {statistics.mean(temps):.1f} | {max(temps)} |")
    if powers:
        md.append(f"| Power Draw (W) | {min(powers):.1f} | {statistics.mean(powers):.1f} | {max(powers):.1f} |")
    if gfx_clocks:
        md.append(f"| Graphics Clock (MHz) | {min(gfx_clocks)} | {statistics.mean(gfx_clocks):.0f} | {max(gfx_clocks)} |")
    if mem_clocks:
        md.append(f"| Memory Clock (MHz) | {min(mem_clocks)} | {statistics.mean(mem_clocks):.0f} | {max(mem_clocks)} |")
    md.append("")


def generate_markdown_report(report_path: Path, runs, paired, telemetry_records, output_dir: Path):
    """Renders SUPER_JOB_REPORT.md using bounded empirical summaries and Glossary terms."""
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    stats = compute_matrix_statistics(runs, stage="matrix")
    stage_breakdowns = compute_stage_breakdowns(runs)
    drift_records = compute_aba_drift_analysis(runs)

    total_runs = len(runs)
    successful_runs = sum(1 for r in runs if r.get("status") == "SUCCESS")
    failed_runs = total_runs - successful_runs
    hosts_str, devices_str, modes_str = _collect_environment_metadata(runs, telemetry_records)

    md = [
        "# HARMONY Super Job Report: GPU Variance & Rotated Matrix Investigation\n",
        f"**Generated:** {now_str}\n",
        f"**Output Directory:** `{output_dir.resolve()}`\n",
        "## 1. Environment and Execution Summary\n",
        "| Attribute | Value |",
        "|---|---|",
        f"| Execution Nodes | {hosts_str} |",
        f"| Identified Devices | {devices_str} |",
        f"| Total Benchmark Runs | {total_runs} |",
        f"| Successful Runs | {successful_runs} |",
        f"| Partial / Failed Runs | {failed_runs} |",
        f"| Execution Mode | `{modes_str}` |\n",
    ]

    if failed_runs > 0:
        md.append("> [!WARNING]")
        md.append(f"> {failed_runs} run(s) did not complete successfully or had truncated logs/CSVs. Review `super_matrix_summary.csv` for failure flags.\n")

    md.append("## 2. Workload Timing Summary & Dispersion (5-Repetition Rotated Matrix)\n")
    md.append("Evaluates timing distribution strictly across Unprofiled Repetition matrix runs. Untimed warm-up and A/B/A Test runs are isolated into separate breakdowns below.\n")
    md.append("| Benchmark | Algorithm | Seed | Repetitions | Median Loop (s) | Median Wall (s) | Mean Loop (s) | Mean Wall (s) | StdDev (s) | Min (s) | Max (s) | Med. Unaccounted (s) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not stats:
        md.append("| *(no successful matrix samples)* | - | - | 0 | - | - | - | - | - | - | - | - |\n")
        md.append("> [!WARNING]")
        md.append("> No successful matrix samples recorded. All matrix stage runs failed or were missing.\n")
    else:
        for s in stats:
            med_loop_str = f"{s['median_loop_s']:.6f}" if s.get("median_loop_s") is not None else "N/A"
            mean_loop_str = f"{s['mean_loop_s']:.6f}" if s.get("mean_loop_s") is not None else "N/A"
            md.append(
                f"| {s['benchmark']} | {s['algo'].upper()} | {s['seed']} | {s['repetition_count']} | "
                f"{med_loop_str} | {s['median_wall_s']:.6f} | {mean_loop_str} | {s['mean_wall_s']:.6f} | {s['stddev_s']:.6f} | "
                f"{s['min_s']:.6f} | {s['max_s']:.6f} | {s['median_unaccounted_s']:.6f} |"
            )
        md.append("")

    for stage_key, heading in (("warmup", "### Untimed Warm-up Breakdown\n"), ("aba", "### Dedicated A/B/A Stage Breakdown\n")):
        s_items = stage_breakdowns.get(stage_key, [])
        if s_items:
            md.append(heading)
            md.append("| Benchmark | Algorithm | Seed | Count | Median Wall (s) | Mean Wall (s) | Med. Unaccounted (s) |")
            md.append("|---|---|---:|---:|---:|---:|---:|")
            for item in s_items:
                md.append(
                    f"| {item['benchmark']} | {item['algo'].upper()} | {item['seed']} | {item['repetition_count']} | "
                    f"{item['median_wall_s']:.6f} | {item['mean_wall_s']:.6f} | {item['median_unaccounted_s']:.6f} |"
                )
            md.append("")

    md.append("## 3. HARMONY Phase Metrics & Partition Boundary Shifts\n")
    md.append("Aggregated from upstream `write_timing_report()` CSV sections (`#timeline`, `#balance`) and Historical Evidence `[DCL][balance-metrics]` blocks.\n")
    md.append("| Benchmark | Algorithm | Seed | Stage | Med. T_compute (s) | Med. T_comm (s) | Med. T_balance (s) | Total Boundary Shifts |")
    md.append("|---|---|---:|---|---:|---:|---:|---:|")
    all_summary_stats = stats + stage_breakdowns.get("warmup", []) + stage_breakdowns.get("aba", [])
    if not all_summary_stats:
        md.append("| *(no successful samples)* | - | - | - | - | - | - | - |\n")
    else:
        for s in all_summary_stats:
            md.append(
                f"| {s['benchmark']} | {s['algo'].upper()} | {s['seed']} | {s.get('stage', 'matrix')} | "
                f"{s.get('median_compute_s', 0.0):.6f} | {s.get('median_comm_s', 0.0):.6f} | "
                f"{s.get('median_balance_s', 0.0):.6f} | {s.get('total_boundary_shifts', 0)} |"
            )
        md.append("")

    md.append("## 4. Paired Comparison and Speedup Analysis (ERAD ÷ HWTOPOLB)\n")
    md.append("Compares observed paired ratios against Historical Evidence baseline runs.\n")
    md.append("| Benchmark | Seed | Repetition | Stage | Step | ERAD Wall (s) | HWTOPOLB Wall (s) | ERAD Loop (s) | HWTOPOLB Loop (s) | Observed Ratio | Observed Speedup | Hist. Ratio | Hist. Speedup | Status |")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    seed_123_ratios = []
    if not paired:
        md.append("| *(no valid pairs)* | - | - | - | - | - | - | - | - | - | - | - | - | - |\n")
    else:
        for p in paired:
            hist = HISTORICAL_PRELIMINARY.get((p["benchmark"], p["seed"]), {})
            hist_ratio = f"{hist.get('ratio', 0.0):.2f}×" if hist else "N/A"
            hist_spd = f"{hist.get('speedup_pct', 0.0):.2f}%" if hist else "N/A"
            obs_ratio = f"{p['ratio']:.2f}×" if p["ratio"] is not None else "N/A"
            obs_spd = f"{p['speedup_pct']:.2f}%" if p["speedup_pct"] is not None else "N/A"

            if p["seed"] == 123 and p.get("stage") == "matrix" and p.get("ratio") is not None:
                seed_123_ratios.append(p["ratio"])

            step_display = str(p["step"]) if p.get("step") is not None else "-"
            rep_display = str(p["repetition"]) if p.get("repetition") is not None else "-"
            status_flag = "HWTOPOLB Faster" if (p["ratio"] is not None and p["ratio"] > 1.0) else "ERAD Faster / Parity"

            ew = f"{p['erad_wall_s']:.4f}" if p.get("erad_wall_s") is not None else "N/A"
            hw = f"{p['hwtopolb_wall_s']:.4f}" if p.get("hwtopolb_wall_s") is not None else "N/A"
            el = f"{p['erad_loop_s']:.4f}" if p.get("erad_loop_s") is not None else "N/A"
            hl = f"{p['hwtopolb_loop_s']:.4f}" if p.get("hwtopolb_loop_s") is not None else "N/A"
            md.append(
                f"| {p['benchmark']} | {p['seed']} | {rep_display} | {p['stage']} | {step_display} | "
                f"{ew} | {hw} | {el} | {hl} | {obs_ratio} | {obs_spd} | {hist_ratio} | {hist_spd} | {status_flag} |"
            )
        md.append("")

    if drift_records:
        md.append("## 5. Dedicated A/B/A Thermal & Hardware Drift Analysis\n")
        md.append("Isolates hardware thermal and clock frequency drift by comparing identical ERAD step 1 vs step 3 runs, and compares HWTOPOLB step 2 against the bracketed ERAD baseline.\n")
        md.append("| Benchmark | Seed | ERAD Step 1 (s) | HWTOPOLB Step 2 (s) | ERAD Step 3 (s) | Thermal Drift Delta (s) | Thermal Drift (%) | Bracketed ERAD (s) | Bracketed Ratio | Bracketed Speedup |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in drift_records:
            br_r = f"{d['bracketed_ratio']:.2f}×" if d["bracketed_ratio"] is not None else "N/A"
            br_s = f"{d['bracketed_speedup_pct']:.2f}%" if d["bracketed_speedup_pct"] is not None else "N/A"
            md.append(
                f"| {d['benchmark']} | {d['seed']} | {d['erad_step1_s']:.4f} | {d['hwtopolb_step2_s']:.4f} | "
                f"{d['erad_step3_s']:.4f} | {d['drift_delta_s']:+.4f}s | {d['drift_pct']:+.2f}% | "
                f"{d['bracketed_erad_mean_s']:.4f} | {br_r} | {br_s} |"
            )
        md.append("")

    md.append("## 6. Evaluation of Seed 123 Variance\n")
    md.append("### Diagnostic Findings on Seed 123 Performance")
    if not seed_123_ratios:
        md.append("- No valid paired runs available for seed 123 in the rotated matrix stage.\n")
    else:
        med_r = statistics.median(seed_123_ratios)
        avg_r = statistics.mean(seed_123_ratios)
        min_r, max_r = min(seed_123_ratios), max(seed_123_ratios)
        hw_fav = sum(1 for r in seed_123_ratios if r > 1.0)
        erad_fav = len(seed_123_ratios) - hw_fav
        total_reps = len(seed_123_ratios)

        md.append(f"- **Evaluated Seed 123 Rotated Matrix Pairs:** {total_reps}")
        md.append(f"- **Median Observed Ratio:** {med_r:.2f}× (Historical baseline was 0.94×)")
        md.append(f"- **Mean Observed Ratio:** {avg_r:.2f}×")
        md.append(f"- **Observed Ratio Range (Min / Max):** {min_r:.2f}× / {max_r:.2f}×")
        md.append(f"- **Repetitions Favoring HWTOPOLB (Ratio > 1.0):** {hw_fav} of {total_reps}")
        md.append(f"- **Repetitions Favoring ERAD or Parity (Ratio <= 1.0):** {erad_fav} of {total_reps}")
        md.append("- Note: This diagnostic verdict strictly reflects rotated matrix repetitions, while warm-up and A/B/A Test runs are reported in their dedicated sections.\n")

        s123_drift = [d for d in drift_records if d.get("seed") == 123]
        if s123_drift:
            summaries = [f"{d['benchmark']}: {d['drift_delta_s']:+.4f}s ({d['drift_pct']:+.2f}%)" for d in s123_drift]
            md.append(f"- **A/B/A Thermal Drift Context (ERAD Step 1 vs Step 3):** {'; '.join(summaries)}")
        else:
            md.append("- **A/B/A Thermal Drift Context:** No dedicated A/B/A drift samples for seed 123.")

        avail_telem = [r for r in (telemetry_records or ()) if r.get("available", True)]
        if avail_telem:
            temps = [r["temperature_gpu"] for r in avail_telem if r.get("temperature_gpu", 0) > 0]
            gclocks = [r["clocks_graphics_mhz"] for r in avail_telem if r.get("clocks_graphics_mhz", 0) > 0]
            t_ctx = f"temp min={min(temps)}°C max={max(temps)}°C spread={max(temps)-min(temps)}°C" if temps else "temp N/A"
            c_ctx = f"graphics clock min={min(gclocks)}MHz max={max(gclocks)}MHz spread={max(gclocks)-min(gclocks)}MHz" if gclocks else "clock N/A"
            md.append(f"- **Hardware Telemetry Context:** {t_ctx}, {c_ctx}")
        else:
            md.append("- **Hardware Telemetry Context:** No live telemetry available across nodes.")
        md.append("")

        md.append("> [!NOTE]")
        if med_r >= HEURISTIC_SEED_123_SPEEDUP_THRESHOLD:
            md.append(f"> **Verdict: Observed HWTOPOLB Advantage Across Rotated Matrix Repetitions (Heuristic Threshold >= {HEURISTIC_SEED_123_SPEEDUP_THRESHOLD:.2f}×).**")
            md.append(f"> Across {total_reps} rotated matrix repetitions for seed 123, HWTOPOLB was faster in {hw_fav} repetitions (ratio range {min_r:.2f}× to {max_r:.2f}×, median {med_r:.2f}×).")
        elif med_r < HEURISTIC_SEED_123_PARITY_THRESHOLD:
            md.append(f"> **Verdict: Observed Parity or ERAD Advantage Across Rotated Matrix Repetitions (Heuristic Threshold < {HEURISTIC_SEED_123_PARITY_THRESHOLD:.2f}×).**")
            md.append(f"> Across {total_reps} rotated matrix repetitions for seed 123, ERAD was faster or at parity in {erad_fav} repetitions (ratio range {min_r:.2f}× to {max_r:.2f}×, median {med_r:.2f}×).")
        else:
            md.append(f"> **Verdict: Observed Mixed Performance Across Rotated Matrix Repetitions (Heuristic Band {HEURISTIC_SEED_123_PARITY_THRESHOLD:.2f}× – {HEURISTIC_SEED_123_SPEEDUP_THRESHOLD:.2f}×).**")
            md.append(f"> Seed 123 showed variation across repetitions ({hw_fav} favored HWTOPOLB, {erad_fav} favored ERAD/parity; ratio range {min_r:.2f}× to {max_r:.2f}×, median {med_r:.2f}×).")
        md.append(">")
        md.append(f"> *Descriptive Scope:* This classification applies heuristic thresholds (HEURISTIC_SEED_123_SPEEDUP_THRESHOLD = {HEURISTIC_SEED_123_SPEEDUP_THRESHOLD}, HEURISTIC_SEED_123_PARITY_THRESHOLD = {HEURISTIC_SEED_123_PARITY_THRESHOLD}) to summarize observed matrix repetitions. It does not establish a causal conclusion and is strictly limited to this two-GPU hardware configuration.\n")

    if telemetry_records:
        _render_telemetry_section(md, telemetry_records)

    md.extend([
        "## 8. Technical Evidence Assessment\n",
        "### Observed",
        "- **Command**: Execution of `run_super_matrix.sh` and `job_harmony_super.pbs` with prebuilt binaries (`bin/kneighbor`, `bin/leanmd`, `bin/topo_probe`, `tests/bin/test_async_halo_opencl`).",
        f"- **Environment**: Host compute allocation (`{hosts_str}`), devices (`{devices_str}`), mode (`{modes_str}`).",
        f"- **Exit Status**: {successful_runs} / {total_runs} runs completed with status SUCCESS.",
        "- **Measured Output**: Wall times, pure loop durations (`loop_time_s`), rebalance counts, structured CSV timing sections (`#run`, `#timeline`, `#device`, `#process`, `#balance`), and Historical Evidence logs parsed using strictly Python 3 standard library modules.\n",
        "### Inferred",
        "- **Reasoning**: Paired speedup ratios, repetition dispersion metrics isolated strictly to rotated matrix runs, A/B/A thermal drift deltas, and unaccounted wall-time reconciliation.",
        "- **Uncertainty**: Inter-node MPI contention, OS jitter, and OpenCL driver queue dispatch latency.\n",
        "### Unverified",
        "- **Missing Checks**: Multi-node cluster topologies beyond the allocated 2-node / 2-GPU configuration.",
        "- **Required Environment**: Production multi-node cluster runs with live MPI fabrics.\n",
    ])

    report_path.write_text("\n".join(md), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Defensive stdlib post-processor and Markdown report generator for HARMONY Super Job."
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("."),
        help="Path to directory containing run CSVs, logs, and telemetry.",
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Idempotent analysis mode: parse existing files and regenerate reports without executing benchmarks.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Expected iterations per run (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()

    if not output_dir.exists():
        sys.stderr.write(f"Error: Output directory does not exist: {output_dir}\n")
        return 1

    try:
        runs = discover_runs(output_dir, expected_iterations=args.iterations)
        generate_artifacts(output_dir, runs)
    except Exception as exc:
        sys.stderr.write(f"Error during Super Job post-processing: {exc}\n")
        return 1

    if not runs:
        sys.stderr.write(f"No benchmark runs discovered in {output_dir}\n")
        return 0

    print(f"Post-processing completed successfully for {len(runs)} run(s).")
    print(f"Summary: {output_dir / 'super_matrix_summary.csv'}")
    print(f"Paired:  {output_dir / 'super_matrix_paired.csv'}")
    print(f"Report:  {output_dir / 'SUPER_JOB_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
