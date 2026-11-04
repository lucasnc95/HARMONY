#!/usr/bin/env python3
"""Validate GPU benchmark run files and separate recorded from unaccounted time."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sys


CSV_FIELDS = ("iteration", "step_time_s", "action", "rebalanced", "rebalance_gain", "execution_mode")
CONDITION_FIELDS = ("workload", "algorithm", "seed", "wall_time_s", "step_sum_s",
                    "ordinary_step_sum_s", "rebalance_step_sum_s", "first_step_s",
                    "last_step_s", "unaccounted_s")
ITERATION_FIELDS = ("workload", "algorithm", "seed", *CSV_FIELDS)
WALL_LINE = re.compile(r"^\[(?P<workload>[^]]+)\] algo=(?P<algorithm>\S+) mode=(?P<mode>\S+) total_time_s=(?P<wall>\S+)(?:\s|$)")


class InvalidRun(ValueError):
    """An input run cannot support a timing comparison."""


def nonnegative(value, field, context):
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError):
        raise InvalidRun(f"{context}: invalid {field}: {value!r}") from None
    if not number.is_finite() or number < 0:
        raise InvalidRun(f"{context}: invalid {field}: {value!r}")
    return number


def read_condition(run_dir, workload, algorithm, seed, iterations):
    stem = f"{workload}_{algorithm}_{seed}"
    csv_path = run_dir / f"{stem}.csv"
    log_path = run_dir / f"{stem}.log"
    for path in (csv_path, log_path):
        if not path.is_file():
            raise InvalidRun(f"missing run file: {path}")

    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise InvalidRun(f"{csv_path}: malformed CSV header")
        rows = list(reader)
    if len(rows) != iterations:
        raise InvalidRun(f"{csv_path}: expected {iterations} iteration records, found {len(rows)}")

    per_iteration = []
    ordinary = Decimal(0)
    rebalance = Decimal(0)
    for expected, row in enumerate(rows):
        context = f"{csv_path}: record {expected}"
        if None in row or any(value is None for value in row.values()):
            raise InvalidRun(f"{context}: malformed CSV record")
        try:
            actual = int(row["iteration"])
        except ValueError:
            raise InvalidRun(f"{context}: invalid iteration {row['iteration']!r}") from None
        if actual != expected:
            raise InvalidRun(f"{context}: expected iteration {expected}, found {actual}")
        if row["execution_mode"] != "gpu_opencl":
            raise InvalidRun(f"{context}: expected gpu_opencl, found {row['execution_mode']!r}")
        if row["rebalanced"] not in ("0", "1"):
            raise InvalidRun(f"{context}: invalid rebalanced flag")
        if row["action"] not in ("compute", "rebalance"):
            raise InvalidRun(f"{context}: invalid action {row['action']!r}")
        if (row["action"] == "rebalance") != (row["rebalanced"] == "1"):
            raise InvalidRun(f"{context}: inconsistent action and rebalanced flag")
        elapsed = nonnegative(row["step_time_s"], "step_time_s", context)
        nonnegative(row["rebalance_gain"], "rebalance_gain", context)
        if row["rebalanced"] == "1":
            rebalance += elapsed
        else:
            ordinary += elapsed
        per_iteration.append({"workload": workload, "algorithm": algorithm,
                              "seed": seed, **row})

    walls = []
    for line in log_path.read_text().splitlines():
        match = WALL_LINE.match(line)
        if match:
            walls.append(match)
        elif "total_time_s=" in line:
            raise InvalidRun(f"{log_path}: malformed total_time_s record")
    if len(walls) != 1:
        raise InvalidRun(f"{log_path}: expected exactly one total_time_s record, found {len(walls)}")
    match = walls[0]
    if (match["workload"], match["algorithm"], match["mode"]) != (workload, algorithm, "gpu_opencl"):
        raise InvalidRun(f"{log_path}: total_time_s record does not match {stem} in gpu_opencl mode")
    wall = nonnegative(match["wall"], "total_time_s", str(log_path))
    step_sum = ordinary + rebalance
    condition = {"workload": workload, "algorithm": algorithm, "seed": seed,
                 "wall_time_s": str(wall), "step_sum_s": str(step_sum),
                 "ordinary_step_sum_s": str(ordinary), "rebalance_step_sum_s": str(rebalance),
                 "first_step_s": rows[0]["step_time_s"], "last_step_s": rows[-1]["step_time_s"],
                 "unaccounted_s": str(wall - step_sum)}
    return condition, per_iteration


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    conditions = []
    steps = []
    try:
        for workload in args.workloads:
            for algorithm in args.algorithms:
                for seed in args.seeds:
                    condition, per_iteration = read_condition(args.run_dir, workload, algorithm, seed, args.iterations)
                    conditions.append(condition)
                    steps.extend(per_iteration)
    except (InvalidRun, OSError, UnicodeError) as error:
        parser.exit(1, f"error: {error}\n")
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_condition.csv", CONDITION_FIELDS, conditions)
    write_csv(args.output_dir / "per_iteration.csv", ITERATION_FIELDS, steps)
    print(f"Validated {len(conditions)} GPU conditions and {len(steps)} ordered iterations; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
