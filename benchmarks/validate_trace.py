#!/usr/bin/env python3
"""Validate rank-local benchmark diagnostic traces before analysis."""

import argparse
import csv
import math
import json
from pathlib import Path
import sys

COLUMNS = [
    "schema_version", "rank", "seq", "iteration", "seed", "algorithm", "record_type",
    "phase", "part", "device_global", "owner_rank", "local_index", "offset", "count",
    "duration_s", "device_event_s", "boundary_shift_elements_estimate",
    "transfer_bytes_measured", "device_name", "device_kind", "device_identity",
]
STEP_PHASES = {"mpi_step_max", "mpi_device_times"}
FINAL_PHASES = {"loop_total", "gather", "reference", "mpi_validation"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rank(path, rank, iterations, elements, balance_interval):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == COLUMNS, f"{path}: unexpected schema")
        rows = list(reader)
    require(bool(rows), f"{path}: empty trace")
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"{path}: malformed CSV row")
    require([int(row["seq"]) for row in rows] == list(range(len(rows))),
            f"{path}: nonconsecutive sequence")
    require(all(row["schema_version"] == "1" and int(row["rank"]) == rank for row in rows),
            f"{path}: wrong rank or version")
    require(len({row["seed"] for row in rows}) == 1 and len({row["algorithm"] for row in rows}) == 1,
            f"{path}: inconsistent condition")
    progress = []
    maps = []
    previous_new = None
    inventory = {int(row["device_global"]): row for row in rows if row["record_type"] == "inventory"}
    for iteration in range(iterations):
        current = [row for row in rows if int(row["iteration"]) == iteration]
        require(bool(current), f"{path}: missing iteration {iteration}")
        progress.append([row["phase"] for row in current if row["record_type"] == "phase"
                         and row["phase"] in STEP_PHASES])
        require(progress[-1] == ["mpi_step_max", "mpi_device_times"],
                f"{path}: incomplete MPI progress at iteration {iteration}")
        phases = [row["phase"] for row in current if row["record_type"] == "phase"]
        require(phases.count("runtime_execute") + phases.count("cpu_simulation") == 1,
                f"{path}: missing execution phase at iteration {iteration}")
        should_rebalance = balance_interval > 0 and iteration > 0 and iteration % balance_interval == 0
        require(("rebalance_apply" in phases) == should_rebalance,
                f"{path}: unexpected rebalance progress at iteration {iteration}")
        if "rebalance_apply" in phases:
            require(phases.count("rebalance_plan") == 1,
                    f"{path}: missing rebalance plan at iteration {iteration}")
        else:
            require("rebalance_plan" not in phases,
                    f"{path}: rebalance plan without apply at iteration {iteration}")
        for state in ("old", "new"):
            parts = [row for row in current if row["record_type"] == "partition" and row["phase"] == state]
            require(bool(parts), f"{path}: missing {state} map at iteration {iteration}")
            ids = [int(row["part"]) for row in parts]
            require(ids == list(range(len(parts))), f"{path}: invalid partition IDs")
            require(len({int(row["device_global"]) for row in parts}) == len(parts),
                    f"{path}: duplicate global device")
            ordered = sorted(parts, key=lambda item: (int(item["offset"]), int(item["part"])))
            cursor = 0
            for part in ordered:
                require(int(part["offset"]) == cursor and int(part["count"]) >= 0,
                        f"{path}: partition gap or overlap at iteration {iteration}")
                cursor += int(part["count"])
                if inventory:
                    device = inventory.get(int(part["device_global"]))
                    if device is not None:
                        require(int(part["owner_rank"]) == int(device["owner_rank"]),
                                f"{path}: inventory owner mismatch")
            require(cursor == elements, f"{path}: partition map does not conserve N")
            fingerprint = [(row["device_global"], row["owner_rank"], row["offset"], row["count"])
                           for row in parts]
            maps.append((iteration, state, fingerprint))
            if state == "old":
                if previous_new is not None:
                    require(fingerprint == previous_new, f"{path}: unexplained map change")
                old_map = fingerprint
            else:
                if not any(row["phase"] == "rebalance_apply" for row in current):
                    require(fingerprint == old_map, f"{path}: map changed without rebalance")
                previous_new = fingerprint
        local_parts = [row for row in current if row["record_type"] == "partition"
                       and row["phase"] == "new" and int(row["owner_rank"]) == rank]
        events = [row for row in current if row["record_type"] == "device_event"]
        if inventory:
            require(len(events) == len(local_parts),
                    f"{path}: missing device event at iteration {iteration}")
            require({row["part"] for row in events} == {row["part"] for row in local_parts},
                    f"{path}: device event partition mismatch")
        else:
            require(not events, f"{path}: device event without GPU inventory")
        for row in current:
            for field in ("duration_s", "device_event_s"):
                if row[field]:
                    value = float(row[field])
                    require(math.isfinite(value) and value >= 0, f"{path}: invalid {field}")
            if row["record_type"] == "phase":
                require(bool(row["duration_s"]), f"{path}: missing phase duration")
            if row["record_type"] == "device_event":
                require(bool(row["device_event_s"]), f"{path}: missing device event duration")
    final = [row for row in rows if int(row["iteration"]) == iterations and row["record_type"] == "phase"]
    require({row["phase"] for row in final} == FINAL_PHASES,
            f"{path}: incomplete final phases")
    require(all(int(row["iteration"]) in {-1, *range(iterations + 1)} for row in rows),
            f"{path}: out-of-range iteration")
    host_totals = {}
    device_event_total = 0.0
    for row in rows:
        if row["record_type"] == "phase" and row["duration_s"]:
            host_totals[row["phase"]] = host_totals.get(row["phase"], 0.0) + float(row["duration_s"])
        elif row["record_type"] == "device_event" and row["device_event_s"]:
            device_event_total += float(row["device_event_s"])
    summary = {"rank": rank, "trace": str(path), "host_phase_totals_s": host_totals,
               "device_event_total_s": device_event_total,
               "partition_rows": sum(row["record_type"] == "partition" for row in rows)}
    inventory_rows = [(row["device_global"], row["owner_rank"], row["local_index"],
                       row["device_identity"]) for row in rows if row["record_type"] == "inventory"]
    initial_local = {row["device_global"]: row for row in rows
                     if row["record_type"] == "partition" and row["iteration"] == "0"
                     and row["phase"] == "old" and row["owner_rank"] == str(rank)}
    require(all(device[0] in initial_local and
                initial_local[device[0]]["local_index"] == device[2]
                for device in inventory_rows), f"{path}: local inventory mismatch")
    return (progress, rows[0]["seed"], rows[0]["algorithm"], maps), summary, inventory_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--balance-interval", type=int, default=10)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    require(args.ranks > 0 and args.iterations > 0 and args.elements > 0, "positive dimensions required")
    results = [read_rank(Path(f"{args.prefix}.rank{rank}.csv"), rank, args.iterations,
                         args.elements, args.balance_interval) for rank in range(args.ranks)]
    require(all(result[0] == results[0][0] for result in results), "rank progress or condition mismatch")
    inventory = [device for result in results for device in result[2]]
    if inventory:
        require(len({device[0] for device in inventory}) == len(inventory),
                "duplicate global device in rank inventories")
        require(all(int(device[1]) == rank for rank, result in enumerate(results)
                    for device in result[2]), "rank inventory owner mismatch")
        expected = {device[0]: device[1] for device in inventory}
        first_map = results[0][0][3][0][2]
        require({part[0] for part in first_map} == set(expected),
                "partition map and device inventory differ")
        require(all(part[1] == expected[part[0]] for part in first_map),
                "partition owner differs from inventory")
    if args.summary_json:
        summary = {"schema_version": 1, "iterations": args.iterations, "elements": args.elements,
                   "seed": results[0][0][1], "algorithm": results[0][0][2],
                   "ranks": [result[1] for result in results]}
        with args.summary_json.open("x") as handle:
            handle.write(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {args.ranks} rank traces, {args.iterations} iterations, N={args.elements}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"trace validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
