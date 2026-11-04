#!/usr/bin/env python3
"""Join historical GPU benchmark CSVs to runtime partition and event records."""

import argparse
import csv
from decimal import Decimal
from pathlib import Path
import re

from analyze_variability import InvalidRun, nonnegative, read_condition, write_csv

METRIC = re.compile(r"^\[DCL\]\[balance-metrics\] iteration=(\d+)$")
DEVICE = re.compile(r"^\[device\] rank=(\d+) global_index=(\d+) name=(.+)$")
EVENT = re.compile(r"^    part\[(\d+)\]=(\S+)$")
PARTITION = re.compile(r"^\s+part\[(\d+)\].*?device_global=(\d+) rank=(\d+) local_index=(-?\d+) offset=(\d+) count=(\d+)(?:\s|$)")
TIMELINE_FIELDS = ("workload", "algorithm", "seed", "iteration", "action", "step_time_s", "device_event_max_s")
EVENT_FIELDS = ("workload", "algorithm", "seed", "iteration", "part", "device_global", "rank", "device_event_s")
PARTITION_FIELDS = ("workload", "algorithm", "seed", "iteration", "phase", "part", "device_global", "rank", "local_index", "offset", "count", "device_event_s")
DEVICE_FIELDS = ("workload", "algorithm", "seed", "rank", "global_index", "name")
BOUNDARY_FIELDS = ("workload", "algorithm", "seed", "iteration", "changed_partition_count", "max_boundary_shift")


def parse_partition(line, path):
    match = PARTITION.match(line)
    if not match:
        raise InvalidRun(f"{path}: malformed partition record: {line!r}")
    part, global_index, rank, local_index, offset, count = map(int, match.groups())
    return {"part": part, "device_global": global_index, "rank": rank,
            "local_index": local_index, "offset": offset, "count": count}


def validate_map(parts, global_count, path, context):
    if not parts:
        raise InvalidRun(f"{path}: {context}: missing partition map")
    by_part = sorted(parts, key=lambda part: part["part"])
    if [part["part"] for part in by_part] != list(range(len(parts))):
        raise InvalidRun(f"{path}: {context}: duplicate or missing partition index")
    cursor = 0
    for part in sorted(parts, key=lambda part: (part["offset"], part["part"])):
        if part["offset"] != cursor:
            raise InvalidRun(f"{path}: {context}: partition gap or overlap at offset {cursor}")
        cursor += part["count"]
    if cursor != global_count:
        raise InvalidRun(f"{path}: {context}: partition map covers {cursor}, expected {global_count}")
    return by_part


def records_for_log(path, iterations, global_count):
    lines = path.read_text().splitlines()
    devices = []
    metrics = {}
    boundaries = []
    current = None
    section = None
    for line_no, line in enumerate(lines, 1):
        device = DEVICE.match(line)
        metric = METRIC.match(line)
        if device:
            if current is not None:
                raise InvalidRun(f"{path}:{line_no}: device record after runtime iteration")
            devices.append({"rank": int(device[1]), "global_index": int(device[2]), "name": device[3]})
            continue
        if metric:
            number = int(metric[1])
            if number in metrics:
                raise InvalidRun(f"{path}:{line_no}: duplicate runtime iteration {number}")
            if not 1 <= number <= iterations:
                raise InvalidRun(f"{path}:{line_no}: unjoinable runtime iteration {number}")
            current = {"iteration": number - 1, "events": {}, "old": [], "current": []}
            metrics[number] = current
            section = None
            continue
        if line == "DCL loads by partition:":
            if current is None:
                raise InvalidRun(f"{path}:{line_no}: unjoinable rebalance boundary")
            boundaries.append({"after_runtime": current["iteration"], "parts": []})
            section = "boundary"
            continue
        if line == "  device_times_s:":
            if current is None:
                raise InvalidRun(f"{path}:{line_no}: unjoinable device event section")
            section = "events"
            continue
        if line == "  OLD loads:":
            if current is None:
                raise InvalidRun(f"{path}:{line_no}: unjoinable old partition section")
            section = "old"
            continue
        if line == "  CURRENT loads:":
            if current is None:
                raise InvalidRun(f"{path}:{line_no}: unjoinable current partition section")
            section = "current"
            continue
        if line.startswith("    part[") and section == "events":
            event = EVENT.match(line)
            if not event:
                raise InvalidRun(f"{path}:{line_no}: malformed device event")
            part = int(event[1])
            if part in current["events"]:
                raise InvalidRun(f"{path}:{line_no}: duplicate device event part {part}")
            current["events"][part] = nonnegative(event[2], "device event", f"{path}:{line_no}")
            continue
        if re.match(r"^\s+part\[", line) and section in ("old", "current", "boundary"):
            target = boundaries[-1]["parts"] if section == "boundary" else current[section]
            target.append(parse_partition(line, f"{path}:{line_no}"))
            continue
        if re.match(r"^\s+part\[", line):
            raise InvalidRun(f"{path}:{line_no}: unjoinable partition or device event record")
        if line.strip().startswith("total=") or line.strip().startswith("total_share="):
            section = None
        elif line.startswith("[") and not metric:
            section = None
    if sorted(metrics) != list(range(1, iterations + 1)):
        missing = sorted(set(range(1, iterations + 1)) - metrics.keys())
        raise InvalidRun(f"{path}: missing runtime iteration records {missing}")
    if not devices or sorted(d["rank"] for d in devices) != list(range(len(devices))):
        raise InvalidRun(f"{path}: missing or duplicate device rank records")
    if len({d["global_index"] for d in devices}) != len(devices):
        raise InvalidRun(f"{path}: duplicate global device index")
    for number, record in metrics.items():
        for phase in ("old", "current"):
            record[phase] = validate_map(record[phase], global_count, path, f"runtime iteration {number} {phase}")
        expected = {part["part"] for part in record["current"]}
        if set(record["events"]) != expected:
            raise InvalidRun(f"{path}: runtime iteration {number}: missing or extra device event")
        for part in record["current"]:
            if not any(d["rank"] == part["rank"] and d["global_index"] == part["device_global"] for d in devices):
                raise InvalidRun(f"{path}: runtime iteration {number}: partition device mapping mismatch")
    for boundary in boundaries:
        boundary["parts"] = validate_map(boundary["parts"], global_count, path,
                                         f"rebalance after runtime iteration {boundary['after_runtime'] + 1}")
        for part in boundary["parts"]:
            if not any(d["rank"] == part["rank"] and d["global_index"] == part["device_global"] for d in devices):
                raise InvalidRun(f"{path}: rebalance boundary: partition device mapping mismatch")
    return devices, metrics, boundaries


def same_map(left, right):
    return [(p["part"], p["device_global"], p["rank"], p["offset"], p["count"]) for p in left] == [
        (p["part"], p["device_global"], p["rank"], p["offset"], p["count"]) for p in right]


def reconstruct(run_dir, workload, algorithm, seed, iterations, global_count):
    _, csv_rows = read_condition(run_dir, workload, algorithm, seed, iterations)
    path = run_dir / f"{workload}_{algorithm}_{seed}.log"
    devices, metrics, boundaries = records_for_log(path, iterations, global_count)
    prefix = {"workload": workload, "algorithm": algorithm, "seed": seed}
    timeline, partitions, events, boundary_rows = [], [], [], []
    boundary_by_after = {}
    for boundary in boundaries:
        index = boundary["after_runtime"] + 1
        if index in boundary_by_after:
            raise InvalidRun(f"{path}: duplicate rebalance boundary for CSV iteration {index}")
        boundary_by_after[index] = boundary["parts"]
    expected_boundaries = {i for i, row in enumerate(csv_rows) if row["action"] == "rebalance"}
    if set(boundary_by_after) != expected_boundaries:
        raise InvalidRun(f"{path}: unjoinable rebalance boundary records: expected {sorted(expected_boundaries)}, found {sorted(boundary_by_after)}")
    for index, row in enumerate(csv_rows):
        record = metrics[index + 1]
        if index:
            previous = metrics[index]
            expected_old = boundary_by_after.get(index, previous["current"])
            if not same_map(expected_old, record["old"]):
                raise InvalidRun(f"{path}: runtime iteration {index + 1}: partition transition does not match prior boundary")
        if not same_map(record["old"], record["current"]):
            raise InvalidRun(f"{path}: runtime iteration {index + 1}: unexplained partition change within metrics block")
        timeline.append({**prefix, "iteration": index, "action": row["action"],
                         "step_time_s": row["step_time_s"],
                         "device_event_max_s": str(max(record["events"].values()))})
        for part in record["current"]:
            events.append({**prefix, "iteration": index, "part": part["part"],
                           "device_global": part["device_global"], "rank": part["rank"],
                           "device_event_s": str(record["events"][part["part"]])})
        for phase in ("old", "current"):
            for part in record[phase]:
                partitions.append({**prefix, "iteration": index, "phase": f"runtime_{phase}",
                                   **part, "device_event_s": str(record["events"][part["part"]])})
        if index in boundary_by_after:
            new_parts = boundary_by_after[index]
            old_parts = previous["current"]
            if [(p["part"], p["device_global"], p["rank"]) for p in old_parts] != [
                (p["part"], p["device_global"], p["rank"]) for p in new_parts]:
                raise InvalidRun(f"{path}: CSV iteration {index}: rebalance boundary changed partition identity")
            changed = sum(a["offset"] != b["offset"] or a["count"] != b["count"]
                          for a, b in zip(old_parts, new_parts))
            shift = max(abs(a["offset"] - b["offset"]) for a, b in zip(old_parts, new_parts))
            boundary_rows.append({**prefix, "iteration": index, "changed_partition_count": changed,
                                  "max_boundary_shift": shift})
            for part in new_parts:
                partitions.append({**prefix, "iteration": index, "phase": "rebalance_result",
                                   **part, "device_event_s": ""})
    return timeline, partitions, [{**prefix, **d} for d in devices], boundary_rows, events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--global-count", type=int, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.global_count <= 0:
        parser.error("--iterations and --global-count must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    collected = [[], [], [], [], []]
    try:
        for workload in args.workloads:
            for algorithm in args.algorithms:
                for seed in args.seeds:
                    result = reconstruct(args.run_dir, workload, algorithm, seed, args.iterations, args.global_count)
                    for target, rows in zip(collected, result):
                        target.extend(rows)
    except (InvalidRun, OSError, UnicodeError) as error:
        parser.exit(1, f"error: {error}\n")
    args.output_dir.mkdir(parents=True)
    for name, fields, rows in zip(("timeline", "partitions", "devices", "boundaries", "device_events"),
                                  (TIMELINE_FIELDS, PARTITION_FIELDS, DEVICE_FIELDS, BOUNDARY_FIELDS, EVENT_FIELDS), collected):
        write_csv(args.output_dir / f"{name}.csv", fields, rows)
    print(f"Joined {len(collected[0])} CSV iterations and {len(collected[1])} partition records")


if __name__ == "__main__":
    main()
