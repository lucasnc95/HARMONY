#!/usr/bin/env python3
"""Preview or execute the controlled two-GPU HARMONY comparison."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

WORKLOADS = ('kneighbor', 'leanmd')
ALGORITHMS = ('erad', 'hwtopolb')
SEEDS = (42, 123, 456)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def schedule(run_dir, bin_dir, mode, pilot_iterations, aba_workload=None):
    rows = []
    count = 0

    def add(phase, repetition, workload, seed, algorithm, pair_id=None,
            n=1000000, diagnostics=False):
        nonlocal count
        count += 1
        stem = f'{count:03d}_{workload}_{algorithm}_{seed}'
        prefix = run_dir / phase / stem
        iterations = 100 if phase in ('full', 'warmup', 'aba', 'smoke') else pilot_iterations
        argv = [str(bin_dir / f'{workload}.out'), '--n', str(n),
                '--iterations', str(iterations), '--balance-interval', '10',
                '--perturbation', '0.10', '--seed', str(seed),
                '--balance-algo', algorithm,
                '--k', '5' if workload == 'kneighbor' else '4',
                '--output-csv', str(prefix) + '.csv',
                '--device-inventory-json', str(prefix) + '.inventory.json',
                '--require-two-rank-gpu', '--quiet-runtime-metrics']
        diagnostic_prefix = str(prefix) + '.trace' if diagnostics else None
        if diagnostics:
            argv += ['--diagnostics-prefix', diagnostic_prefix]
        rows.append(dict(phase=phase, ordinal=count, repetition=repetition,
                         workload=workload, seed=seed, algorithm=algorithm,
                         pair_id=pair_id, iterations=iterations, n=n,
                         diagnostics_prefix=diagnostic_prefix, argv=argv,
                         csv=str(prefix) + '.csv', log=str(prefix) + '.log',
                         inventory=str(prefix) + '.inventory.json',
                         manifest=str(prefix) + '.manifest.json'))

    if mode == 'smoke':
        add('smoke', 1, 'kneighbor', 123, 'hwtopolb', n=100000)
        add('smoke', 1, 'kneighbor', 123, 'hwtopolb', n=100000, diagnostics=True)
    elif mode == 'aba':
        for algorithm in ('erad', 'hwtopolb', 'erad'):
            add('aba', 1, aba_workload, 123, algorithm)
    elif mode == 'full':
        for workload in WORKLOADS:
            for algorithm in ALGORITHMS:
                add('warmup', 0, workload, SEEDS[0], algorithm)
        repetitions = range(5)
    elif mode == 'pilot':
        repetitions = range(2)
    for repetition in repetitions if mode in ('pilot', 'full') else ():
        seeds = SEEDS[repetition % len(SEEDS):] + SEEDS[:repetition % len(SEEDS)]
        algorithms = ALGORITHMS if repetition % 2 == 0 else ALGORITHMS[::-1]
        for workload in WORKLOADS:
            for seed in seeds:
                pair_id = f'r{repetition + 1}_{workload}_{seed}'
                for algorithm in algorithms:
                    add(mode, repetition + 1, workload, seed, algorithm, pair_id)
    paths = [row[key] for row in rows for key in ('csv', 'log', 'inventory', 'manifest')]
    if len(paths) != len(set(paths)):
        raise ValueError('Schedule contains duplicate artifact paths')
    return rows


def build_verified(bin_dir, manifest_path):
    with open(manifest_path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    for workload in (*WORKLOADS, 'test_async_halo_opencl'):
        binary = bin_dir / f'{workload}.out'
        recorded = manifest['binaries'][workload]
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f'Missing executable: {binary}')
        if digest(binary) != recorded['sha256']:
            raise ValueError(f'Binary SHA-256 mismatch: {binary}')
    return manifest


def pbs_environment():
    job_id = os.environ.get('PBS_JOBID')
    nodefile = os.environ.get('PBS_NODEFILE')
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise ValueError('PBS_JOBID and a readable PBS_NODEFILE are required')
    slots = Path(nodefile).read_text(encoding='utf-8').splitlines()
    if len(slots) != 2 or len(set(slots)) != 2:
        raise ValueError('Expected two PBS slots on distinct one-GPU nodes')
    launcher = shutil.which('prun')
    if not launcher:
        raise ValueError('prun launcher is required')
    return dict(job_id=job_id, nodefile=nodefile, slots=slots, launcher=launcher,
                scheduler_output=os.environ.get('HARMONY_PBS_OUTPUT_PATH'))


def observation(command):
    try:
        value = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT,
                                        timeout=10).strip()
        return dict(value=value, unavailable_reason=None)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(value=None, unavailable_reason=str(error))


def run_bounded(command, log, seconds):
    """Stop the launcher and its local process group when a sample hangs."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def environment_manifest(pbs):
    return dict(pbs=pbs, started_at=datetime.now(timezone.utc).isoformat(),
                module_list=observation(['bash', '-lc', 'module list 2>&1']),
                mpi_version=observation(['mpirun', '--version']),
                cpu_model=observation(['bash', '-lc',
                                       "awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo"]),
                gpu_clock=observation(['prun', 'nvidia-smi', '--query-gpu=clocks.current.graphics',
                                       '--format=csv,noheader']),
                gpu_power=observation(['prun', 'nvidia-smi', '--query-gpu=power.draw',
                                       '--format=csv,noheader']),
                gpu_contention=observation(['prun', 'nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader']))


def accepted_output(row):
    try:
        with open(row['csv'], newline='', encoding='utf-8') as stream:
            records = list(csv.DictReader(stream))
        if len(records) != row['iterations']:
            raise ValueError(f'Expected {row["iterations"]} iterations, found {len(records)}')
        if [int(record['iteration']) for record in records] != list(range(row['iterations'])):
            raise ValueError('Missing or duplicate iteration')
        if any(record['execution_mode'] != 'gpu_opencl' for record in records):
            raise ValueError('CPU fallback or mixed execution mode')
        if any(not math.isfinite(float(record['step_time_s'])) or
               float(record['step_time_s']) < 0 for record in records):
            raise ValueError('Nonfinite or negative step timing')
        log = Path(row['log']).read_text(encoding='utf-8')
        if re.findall(r'\bmode=([^\s]+)', log) != ['gpu_opencl']:
            raise ValueError('GPU execution mode is missing or inconsistent')
        validation = re.findall(r'\bvalidation=([^\s]+)', log)
        if validation != ['PASS']:
            raise ValueError('Numerical validation did not pass')
        times = {}
        for name in ('loop_time_s', 'validation_time_s'):
            matches = re.findall(rf'\b{name}=([^\s]+)', log)
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous {name}')
            times[name] = float(matches[0])
            if not math.isfinite(times[name]) or times[name] < 0:
                raise ValueError(f'Invalid {name}')
        inventory = json.loads(Path(row['inventory']).read_text(encoding='utf-8'))
        ranks = inventory['ranks']
        if inventory.get('schema_version') != 1 or len(ranks) != 2 or \
                {item['rank'] for item in ranks} != {0, 1}:
            raise ValueError('Expected one inventory record for each of two ranks')
        required = ('rank', 'host', 'gpu_count', 'local_index', 'global_index', 'name',
                    'platform_name', 'platform_version', 'device_version',
                    'driver_version', 'identity_kind', 'identity')
        if any(any(item.get(key) is None for key in required) for item in ranks):
            raise ValueError('Incomplete device inventory')
        if any(type(item['gpu_count']) is not int or item['gpu_count'] != 1 for item in ranks):
            raise ValueError('Expected exactly one GPU per rank')
        if any(item['identity_kind'] not in ('uuid', 'pci') or not item['identity']
               for item in ranks):
            raise ValueError('Unverifiable physical GPU identity')
        if len({item['identity_kind'] for item in ranks}) != 1:
            raise ValueError('GPU identity kinds differ across ranks')
        physical = {(item['identity_kind'],
                     item['identity'] if item['identity_kind'] == 'uuid'
                     else (item['host'], item['identity'])) for item in ranks}
        if len(physical) != 2:
            raise ValueError('The two ranks resolved to the same physical GPU')
        mapping = [{**{key: item[key] for key in required},
                    'cpu_affinity': item.get('cpu_affinity',
                                             {'value': None, 'unavailable_reason': 'not recorded'})}
                   for item in sorted(ranks, key=lambda x: x['rank'])]
        return times, mapping
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f'Incomplete sample artifacts: {error}') from error


def validate_smoke_csv(row):
    with open(row['csv'], newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['iteration', 'step_time_s', 'action', 'rebalanced',
                                 'rebalance_gain', 'execution_mode']:
            raise ValueError('Smoke CSV has an unexpected schema')
        for iteration, record in enumerate(reader):
            rebalance = iteration > 0 and iteration % 10 == 0
            if record['action'] != ('rebalance' if rebalance else 'compute') or \
                    record['rebalanced'] != ('1' if rebalance else '0'):
                raise ValueError(f'Smoke iteration {iteration} has wrong rebalance progress')


def validate_smoke_trace(row, run_dir, mapping, times):
    prefix = row['diagnostics_prefix']
    summary = run_dir / 'smoke_trace_summary.json'
    try:
        result = subprocess.run([
            sys.executable, str(Path(__file__).with_name('validate_trace.py')),
            prefix, '--ranks', '2', '--iterations', '100', '--elements', '100000',
            '--balance-interval', '10', '--summary-json', str(summary),
        ], capture_output=True, text=True, check=False, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise ValueError('Smoke rank trace validation timed out') from error
    if result.returncode != 0:
        raise ValueError(f'Smoke rank traces failed validation: {result.stderr.strip()}')
    paths = [Path(f'{prefix}.rank{rank}.csv') for rank in (0, 1)]
    if not summary.is_file() or any(not path.is_file() for path in paths):
        raise ValueError('Smoke trace summary or rank trace is missing')
    contents = json.loads(summary.read_text(encoding='utf-8'))
    if contents.get('seed') != '123' or contents.get('algorithm') != 'hwtopolb' or \
            contents.get('iterations') != 100 or contents.get('elements') != 100000:
        raise ValueError('Smoke trace summary differs from the planned condition')
    ranks = contents.get('ranks')
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise ValueError('Smoke trace summary lacks both ranks')
    for rank, path in enumerate(paths):
        with path.open(newline='', encoding='utf-8') as stream:
            inventory = [record for record in csv.DictReader(stream)
                         if record['record_type'] == 'inventory']
        expected = mapping[rank]
        if len(inventory) != 1 or inventory[0]['device_global'] != str(expected['global_index']) or \
                inventory[0]['device_identity'] != \
                f"{expected['identity_kind']}:{expected['identity']}":
            raise ValueError(f'Smoke rank {rank} trace GPU differs from device inventory')
    loops = [float(item['host_phase_totals_s']['loop_total']) for item in ranks]
    if not math.isclose(max(loops), times['loop_time_s'], rel_tol=1e-4, abs_tol=1e-5):
        raise ValueError('Smoke rank-local loop totals differ from benchmark loop time')
    return dict(summary=str(summary), summary_sha256=digest(summary),
                rank_traces=[dict(path=str(path), sha256=digest(path)) for path in paths])


def execute(args, rows):
    bin_dir = args.bin_dir.resolve()
    build_manifest = args.build_manifest.resolve()
    build = build_verified(bin_dir, build_manifest)
    pbs = pbs_environment()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run = dict(schema_version=1, mode=args.mode, status='running',
               build_manifest=str(build_manifest), build_manifest_sha256=digest(build_manifest),
               build=build, environment=environment_manifest(pbs),
               input_parameters=dict(n=100000 if args.mode == 'smoke' else 1000000,
                                     full_iterations=100,
                                     pilot_iterations=args.pilot_iterations,
                                     rebalance_interval=10, perturbation=0.10,
                                     kneighbor_k=5, leanmd_cutoff=4),
               invocations=rows, baseline_mapping=None)
    write_json(args.run_dir / 'run_manifest.json', run)
    if args.mode in ('full', 'aba', 'smoke'):
        halo = bin_dir / 'test_async_halo_opencl.out'
        halo_log = args.run_dir / 'halo_reference.log'
        check = dict(command=[pbs['launcher'], str(halo)], log=str(halo_log),
                     binary_sha256=build['binaries']['test_async_halo_opencl']['sha256'],
                     status='running', exit_code=None)
        run['halo_reference'] = check
        write_json(args.run_dir / 'run_manifest.json', run)
        with open(halo_log, 'x', encoding='utf-8') as log:
            try:
                check['exit_code'] = run_bounded(check['command'], log, 180)
            except subprocess.TimeoutExpired:
                check['status'] = 'rejected'
                check['rejection_reason'] = 'Asynchronous halo reference timed out'
        if check['status'] != 'rejected':
            output = halo_log.read_text(encoding='utf-8')
            if check['exit_code'] == 0 and output.splitlines().count(
                    'PASS: asynchronous OpenCL halo matches two-step CPU reference') == 1 \
                    and 'SKIP:' not in output:
                check['status'] = 'passed'
            else:
                check['status'] = 'rejected'
                check['rejection_reason'] = 'Two-rank asynchronous OpenCL halo reference did not pass'
        if check['status'] != 'passed':
            run['status'] = 'rejected'
            run['rejected_samples'] = 0
            run['completed_at'] = datetime.now(timezone.utc).isoformat()
            write_json(args.run_dir / 'run_manifest.json', run)
            return 1
        write_json(args.run_dir / 'run_manifest.json', run)
    baseline = None
    pairs = {}
    failures = 0
    failed_pairs = set()
    timed_out = False
    for row in rows:
        sample = dict(row, status='planned', rejection_reason=None,
                      build_manifest_sha256=run['build_manifest_sha256'],
                      started_at=None, completed_at=None, exit_code=None)
        path = Path(row['manifest'])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, sample)
        try:
            if row['pair_id'] in failed_pairs:
                raise ValueError('Earlier member of pair was rejected')
            build_verified(bin_dir, build_manifest)
            if digest(build_manifest) != run['build_manifest_sha256']:
                raise ValueError('Build manifest changed during the run')
            if any(Path(row[key]).exists() for key in ('csv', 'log', 'inventory')) or \
                    (row['diagnostics_prefix'] and any(
                        Path(f"{row['diagnostics_prefix']}.rank{rank}.csv").exists()
                        for rank in (0, 1))):
                raise ValueError('Sample artifact path already exists')
            sample['status'] = 'running'
            sample['started_at'] = datetime.now(timezone.utc).isoformat()
            write_json(path, sample)
            with open(row['log'], 'x', encoding='utf-8') as log:
                exit_code = run_bounded([pbs['launcher'], *row['argv']], log,
                                        args.sample_timeout_seconds)
            sample['exit_code'] = exit_code
            if exit_code != 0:
                raise ValueError(f'Benchmark exited {exit_code}')
            times, mapping = accepted_output(row)
            if times['loop_time_s'] <= 0:
                raise ValueError('Benchmark loop time must be positive')
            if args.mode == 'smoke':
                validate_smoke_csv(row)
                if row['diagnostics_prefix']:
                    sample['trace'] = validate_smoke_trace(row, args.run_dir, mapping, times)
            if row['pair_id']:
                previous = pairs.get(row['pair_id'])
                if previous and previous['mapping'] != mapping:
                    previous_sample = previous['sample']
                    previous_sample['status'] = 'rejected'
                    previous_sample['rejection_reason'] = 'Pair device mapping mismatch'
                    write_json(Path(previous_sample['manifest']), previous_sample)
                    raise ValueError('Pair device mapping mismatch')
                pairs[row['pair_id']] = dict(mapping=mapping, sample=sample)
            if baseline is None:
                baseline = mapping
                run['baseline_mapping'] = mapping
                write_json(args.run_dir / 'run_manifest.json', run)
            elif mapping != baseline:
                raise ValueError('Device mapping differs from run baseline')
            sample.update(status='accepted', times=times, mapping=mapping)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            if isinstance(error, subprocess.TimeoutExpired):
                timed_out = True
                reason = f'Benchmark sample timed out after {args.sample_timeout_seconds} seconds'
            else:
                reason = str(error)
            sample.update(status='rejected', rejection_reason=reason)
            if row['pair_id']:
                failed_pairs.add(row['pair_id'])
            previous = pairs.get(row['pair_id']) if row['pair_id'] else None
            if previous and previous['sample'] is not sample:
                prior = previous['sample']
                prior['status'] = 'rejected'
                prior['rejection_reason'] = f'Paired sample rejected: {error}'
                write_json(Path(prior['manifest']), prior)
            failures += 1
        sample['completed_at'] = datetime.now(timezone.utc).isoformat()
        write_json(path, sample)
        if timed_out:
            break
    run['status'] = 'completed' if failures == 0 else 'rejected'
    if args.mode == 'smoke' and failures == 0:
        unprofiled, profiled = (
            json.loads(Path(row['manifest']).read_text(encoding='utf-8')) for row in rows)
        run['smoke_comparison'] = dict(
            condition=dict(workload='kneighbor', seed=123, algorithm='hwtopolb',
                           n=100000, iterations=100),
            unprofiled_loop_s=unprofiled['times']['loop_time_s'],
            profiled_loop_s=profiled['times']['loop_time_s'],
            trace_overhead_ratio=(profiled['times']['loop_time_s'] /
                                  unprofiled['times']['loop_time_s']),
            trace=profiled['trace'])
    run['rejected_samples'] = sum(
        json.loads(Path(row['manifest']).read_text(encoding='utf-8'))['status'] == 'rejected'
        for row in rows if Path(row['manifest']).is_file())
    run['completed_at'] = datetime.now(timezone.utc).isoformat()
    write_json(args.run_dir / 'run_manifest.json', run)
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('pilot', 'smoke', 'full', 'aba'), required=True)
    parser.add_argument('--aba-workload', choices=WORKLOADS)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=Path('build/cluster'))
    parser.add_argument('--build-manifest', type=Path, default=Path('build/cluster/build_manifest.json'))
    parser.add_argument('--pilot-iterations', type=int, default=8)
    parser.add_argument('--sample-timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.pilot_iterations < 1 or args.pilot_iterations >= 10:
        parser.error('Pilot iterations must be from 1 to 9')
    if args.sample_timeout_seconds < 1 or args.sample_timeout_seconds > 86400:
        parser.error('Sample timeout must be from 1 to 86400 seconds')
    if (args.mode == 'aba') != bool(args.aba_workload):
        parser.error('--aba-workload is required only for aba mode')
    rows = schedule(args.run_dir.resolve(), args.bin_dir.resolve(), args.mode,
                    args.pilot_iterations, args.aba_workload)
    if args.dry_run:
        print(json.dumps(dict(schema_version=1, mode=args.mode,
                              run_dir=str(args.run_dir.resolve()), invocations=rows), indent=2))
        return 0
    try:
        return execute(args, rows)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f'Controlled run refused: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
