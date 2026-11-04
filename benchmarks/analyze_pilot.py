#!/usr/bin/env python3
"""Validate an accepted pre-rebalance pilot and report its timing pattern."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

WORKLOADS = ('kneighbor', 'leanmd')
ALGORITHMS = ('erad', 'hwtopolb')
SEEDS = (42, 123, 456)
CSV_FIELDS = ('iteration', 'step_time_s', 'action', 'rebalanced', 'rebalance_gain', 'execution_mode')
MAPPING_FIELDS = ('rank', 'host', 'gpu_count', 'local_index', 'global_index', 'name', 'platform_name',
                  'platform_version', 'device_version', 'driver_version', 'identity_kind', 'identity')
SLOW_THRESHOLD_S = 0.010  # Descriptive threshold for the historical ~20 ms path.


class InvalidPilot(ValueError):
    """The pilot evidence is incomplete or inconsistent."""


def required_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidPilot(f'{path}: cannot read JSON: {error}') from error


def finite_nonnegative(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidPilot(f'{label}: invalid number {value!r}') from None
    if not math.isfinite(number) or number < 0:
        raise InvalidPilot(f'{label}: nonfinite or negative number {value!r}')
    return number


def artifact(root, value, label):
    if not isinstance(value, str):
        raise InvalidPilot(f'{label}: missing artifact path')
    path = Path(value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise InvalidPilot(f'{label}: missing or external artifact {path}')
    return path


def log_value(text, name, context):
    matches = re.findall(rf'\b{re.escape(name)}=([^\s]+)', text)
    if len(matches) != 1:
        raise InvalidPilot(f'{context}: expected exactly one {name}, found {len(matches)}')
    return finite_nonnegative(matches[0], f'{context} {name}')


def mapping_from(inventory, context):
    if inventory.get('schema_version') != 1 or not isinstance(inventory.get('ranks'), list):
        raise InvalidPilot(f'{context}: invalid device inventory schema')
    ranks = inventory['ranks']
    if len(ranks) != 2 or any(not isinstance(r, dict) for r in ranks) or \
            {r.get('rank') for r in ranks} != {0, 1}:
        raise InvalidPilot(f'{context}: expected exactly ranks 0 and 1')
    result = []
    for rank in sorted(ranks, key=lambda item: item['rank']):
        if any(rank.get(field) is None for field in MAPPING_FIELDS):
            raise InvalidPilot(f'{context}: incomplete physical device mapping')
        if type(rank['gpu_count']) is not int or rank['gpu_count'] != 1:
            raise InvalidPilot(f'{context}: expected exactly one GPU per rank')
        if rank['identity_kind'] not in ('uuid', 'pci') or not rank['identity']:
            raise InvalidPilot(f'{context}: unverifiable physical device mapping')
        result.append({**{field: rank[field] for field in MAPPING_FIELDS},
                       'cpu_affinity': rank.get('cpu_affinity',
                                                {'value': None,
                                                 'unavailable_reason': 'not recorded'})})
    if len({(item['host'], item['identity_kind'], item['identity']) for item in result}) != 2:
        raise InvalidPilot(f'{context}: ranks resolved to the same physical GPU')
    uuids = [item['identity'] for item in result if item['identity_kind'] == 'uuid']
    if len(set(uuids)) != len(uuids):
        raise InvalidPilot(f'{context}: duplicate physical GPU UUID across hosts')
    return result


def expected_order():
    result = []
    for repetition in (1, 2):
        seeds = SEEDS[repetition - 1:] + SEEDS[:repetition - 1]
        algorithms = ALGORITHMS if repetition == 1 else ALGORITHMS[::-1]
        for workload in WORKLOADS:
            for seed in seeds:
                for algorithm in algorithms:
                    result.append((repetition, workload, seed, algorithm,
                                   f'r{repetition}_{workload}_{seed}'))
    return result


def analyze(run_dir):
    root = run_dir.resolve()
    run = required_json(root / 'run_manifest.json')
    if run.get('schema_version') != 1 or run.get('mode') != 'pilot' or run.get('status') != 'completed':
        raise InvalidPilot('Expected a completed schema-1 pilot run; full or rejected runs are excluded')
    environment = run.get('environment')
    if not isinstance(environment, dict) or not isinstance(environment.get('pbs'), dict):
        raise InvalidPilot('Missing PBS environment record')
    pbs = environment['pbs']
    if not pbs.get('job_id') or len(pbs.get('slots', [])) != 2:
        raise InvalidPilot('Missing two-rank PBS allocation record')
    if not run.get('build_manifest_sha256') or not run.get('build_manifest') or run.get('rejected_samples', 0) != 0:
        raise InvalidPilot('Missing build provenance or rejected pilot samples')
    invocations = run.get('invocations')
    order = expected_order()
    if not isinstance(invocations, list) or len(invocations) != len(order):
        raise InvalidPilot(f'Expected {len(order)} pilot invocations')
    used_paths = set()
    baseline = None
    report_rows = []
    for ordinal, (row, expected) in enumerate(zip(invocations, order), 1):
        context = f'pilot invocation {ordinal}'
        if not isinstance(row, dict) or (row.get('repetition'), row.get('workload'),
                row.get('seed'), row.get('algorithm'), row.get('pair_id')) != expected:
            raise InvalidPilot(f'{context}: missing or non-rotated pilot condition')
        if row.get('phase') != 'pilot' or row.get('ordinal') != ordinal or \
                type(row.get('iterations')) is not int or not 1 <= row['iterations'] < 10:
            raise InvalidPilot(f'{context}: invalid pre-rebalance invocation')
        if report_rows and row['iterations'] != report_rows[0]['iterations']:
            raise InvalidPilot(f'{context}: inconsistent pilot iteration count')
        paths = {key: artifact(root, row.get(key), f'{context} {key}')
                 for key in ('csv', 'log', 'inventory', 'manifest')}
        if used_paths.intersection(paths.values()):
            raise InvalidPilot(f'{context}: duplicate artifact path')
        used_paths.update(paths.values())
        sample = required_json(paths['manifest'])
        if sample.get('status') != 'accepted' or sample.get('rejection_reason') is not None or \
                sample.get('exit_code') != 0:
            raise InvalidPilot(f'{context}: sample was not accepted or launcher exit was nonzero')
        if any(sample.get(key) != row[key] for key in ('phase', 'ordinal', 'repetition',
               'workload', 'seed', 'algorithm', 'pair_id', 'iterations', 'csv', 'log',
               'inventory', 'manifest')):
            raise InvalidPilot(f'{context}: sample manifest differs from run schedule')
        if sample.get('build_manifest_sha256') != run['build_manifest_sha256']:
            raise InvalidPilot(f'{context}: build provenance mismatch')
        mapping = mapping_from(required_json(paths['inventory']), context)
        if sample.get('mapping') != mapping:
            raise InvalidPilot(f'{context}: accepted mapping differs from device inventory')
        if baseline is None:
            baseline = mapping
            if run.get('baseline_mapping') != mapping:
                raise InvalidPilot('Run baseline mapping differs from device inventory')
        elif mapping != baseline:
            raise InvalidPilot(f'{context}: physical GPU mapping changed during pilot')
        try:
            with paths['csv'].open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(CSV_FIELDS):
                    raise InvalidPilot(f'{context}: malformed CSV header')
                records = list(reader)
        except (OSError, UnicodeError) as error:
            raise InvalidPilot(f'{context}: cannot read CSV: {error}') from error
        if len(records) != row['iterations']:
            raise InvalidPilot(f'{context}: expected {row["iterations"]} iteration records')
        steps = []
        for index, record in enumerate(records):
            if None in record or any(value is None for value in record.values()):
                raise InvalidPilot(f'{context}: malformed iteration {index}')
            if record['iteration'] != str(index) or record['action'] != 'compute' or \
                    record['rebalanced'] != '0' or record['execution_mode'] != 'gpu_opencl':
                raise InvalidPilot(f'{context}: invalid, reordered, CPU, or rebalanced iteration {index}')
            steps.append(finite_nonnegative(record['step_time_s'], f'{context} step {index}'))
            finite_nonnegative(record['rebalance_gain'], f'{context} gain {index}')
        try:
            log = paths['log'].read_text(encoding='utf-8')
        except (OSError, UnicodeError) as error:
            raise InvalidPilot(f'{context}: cannot read log: {error}') from error
        if len(re.findall(r'\bvalidation=PASS\b', log)) != 1:
            raise InvalidPilot(f'{context}: numerical validation did not pass exactly once')
        if len(re.findall(r'\bmode=gpu_opencl\b', log)) != 1:
            raise InvalidPilot(f'{context}: missing or ambiguous gpu_opencl log mode')
        loop = log_value(log, 'loop_time_s', context)
        validation = log_value(log, 'validation_time_s', context)
        total = log_value(log, 'total_time_s', context)
        if total + 1e-9 < max(loop, validation):
            raise InvalidPilot(f'{context}: inconsistent loop/validation/full time')
        times = sample.get('times', {})
        if not isinstance(times, dict) or any(not math.isclose(
                finite_nonnegative(times.get(key), f'{context} manifest {key}'), value,
                rel_tol=1e-6, abs_tol=1e-9) for key, value in
                (('loop_time_s', loop), ('validation_time_s', validation))):
            raise InvalidPilot(f'{context}: sample manifest timing differs from log')
        report_rows.append(dict(ordinal=ordinal, repetition=row['repetition'],
                                position=(ordinal - 1) % 12 + 1, workload=row['workload'],
                                seed=row['seed'], algorithm=row['algorithm'],
                                iterations=row['iterations'], first_s=steps[0],
                                median_s=statistics.median(steps), loop_s=loop,
                                first_slow=next((i for i, value in enumerate(steps)
                                                 if value >= SLOW_THRESHOLD_S), None),
                                slow=steps[0] >= SLOW_THRESHOLD_S, paths=paths))
    return run, report_rows


def render(run, rows):
    lines = ['# Pre-rebalance GPU pilot analysis', '',
             f'Validated **{len(rows)} accepted** pilot invocations, each with '
             f'{rows[0]["iterations"]} ordinary iterations and no scheduled rebalance. '
             'Pilot times are **excluded from final** performance statistics.', '',
             f'PBS job: `{run["environment"]["pbs"]["job_id"]}`. '
             f'Build manifest: `{run["build_manifest"]}`; SHA-256: `{run["build_manifest_sha256"]}`.',
             'Warm-up invocations: none in this short pilot. Input arguments and all raw artifact paths are preserved in `run_manifest.json` and the per-sample manifests.', '',
             'The descriptive slow-path threshold is host step time >= 10 ms; '
             'it is not a statistical decision rule. The table preserves invocation order.', '',
             '| Order | Pass | Position | Workload | Seed | Algorithm | Iteration 0 host (ms) | First slow iteration | Ordinary median (ms) | Loop (s) | Slow at iteration 0 |',
             '| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f'| {row["ordinal"]} | {row["repetition"]} | {row["position"]} | '
                     f'{row["workload"]} | {row["seed"]} | {row["algorithm"]} | '
                     f'{row["first_s"] * 1000:.3f} | {row["first_slow"] if row["first_slow"] is not None else "none"} | {row["median_s"] * 1000:.3f} | '
                     f'{row["loop_s"]:.6f} | {"yes" if row["slow"] else "no"} |')
    lines += ['', '## Pattern checks', '',
              '| Group | Slow at iteration 0 / accepted | Median iteration 0 host (ms) |',
              '| --- | ---: | ---: |']
    groups = [(f'seed {seed}', [r for r in rows if r['seed'] == seed]) for seed in SEEDS]
    groups += [(f'algorithm {algorithm}', [r for r in rows if r['algorithm'] == algorithm])
               for algorithm in ALGORITHMS]
    groups += [(f'pass {repetition}', [r for r in rows if r['repetition'] == repetition])
               for repetition in (1, 2)]
    groups += [(f'position {position}', [r for r in rows if r['position'] == position])
               for position in range(1, 13)]
    for label, group in groups:
        lines.append(f'| {label} | {sum(r["slow"] for r in group)}/{len(group)} | '
                     f'{statistics.median(r["first_s"] for r in group) * 1000:.3f} |')
    stable_seeds = [seed for seed in SEEDS if all(r['slow'] for r in rows if r['seed'] == seed)]
    if stable_seeds:
        lines += ['', f'The observed slow path is associated with seed(s) '
                  f'{", ".join(map(str, stable_seeds))} across these two rotated passes.']
    else:
        lines += ['', 'No seed is slow in every observed invocation.']
    lines += ['', 'Causal attribution remains **inconclusive** in this short pilot. '
              'The sequence and run-level environment are recorded, but two passes cannot '
              'exclude unobserved GPU/cluster state. Inspect per-rank `runtime.execute()` '
              'and benchmark MPI reduction host durations next if the slow path persists; '
              'compare device event duration separately. No timed sample from this pilot '
              'enters the final speedup estimate.', '',
              '## Provenance and limits', '',
              f'Run-level environment record: `{json.dumps(run["environment"], sort_keys=True)}`.',
              'Physical rank-to-device mapping was stable across accepted samples by '
              'the recorded identity tokens. These tokens and the PBS record are '
              'necessary evidence, not independent proof of scheduler isolation. '
              'Clocks, power, and contention are interpreted only when measured in the run manifest.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    try:
        run, rows = analyze(args.run_dir)
        report = render(run, rows)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report)
    except (InvalidPilot, OSError, TypeError) as error:
        parser.exit(1, f'error: {error}\n')
    print(f'Validated {len(rows)} pilot invocations; report: {args.output}')


if __name__ == '__main__':
    main()
