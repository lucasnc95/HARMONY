#!/usr/bin/env python3
"""Run a short, one-rank HARMONY GPU check inside one PBS allocation."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

from run_variability import build_verified, digest, run_bounded, write_json


def host_name(value):
    return value.strip().split('.')[0].split('[')[0]


def validate_allocation(expected_host):
    job_id = os.environ.get('PBS_JOBID')
    nodefile = os.environ.get('PBS_NODEFILE')
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise ValueError('PBS_JOBID and a readable PBS_NODEFILE are required')
    slots = Path(nodefile).read_text(encoding='utf-8').splitlines()
    if len(slots) != 1 or host_name(slots[0]) != expected_host:
        raise ValueError(f'Expected one MPI slot on {expected_host}, received {slots}')
    return dict(job_id=job_id, nodefile=nodefile, slots=slots)


def execute(args):
    if not re.fullmatch(r'compute-0-(?:7|8|2)', args.expected_host):
        raise ValueError('Expected host must be compute-0-7, compute-0-8, or compute-0-2')
    pbs = validate_allocation(args.expected_host)
    bin_dir = args.bin_dir.resolve()
    build_manifest = bin_dir / 'build_manifest.json'
    build = build_verified(bin_dir, build_manifest)
    for binary in ('topo_probe', 'harmony_smoke'):
        if not (bin_dir / f'{binary}.out').is_file():
            raise ValueError(f'Missing {binary}.out; run cluster_build.sh on the frontend')
    manifest_path = args.run_dir / 'run_manifest.json'
    if manifest_path.exists():
        raise ValueError(f'Existing run manifest must be preserved: {manifest_path}')
    manifest = dict(schema_version=1, status='running', expected_host=args.expected_host,
                    pbs=pbs, build_manifest_sha256=digest(build_manifest),
                    build=build, started_at=datetime.now(timezone.utc).isoformat(), checks={})
    write_json(manifest_path, manifest)

    def run(name, command, seconds):
        log = args.run_dir / f'{name}.log'
        check = dict(command=command, log=str(log), status='running')
        manifest['checks'][name] = check
        write_json(manifest_path, manifest)
        try:
            with log.open('x', encoding='utf-8') as stream:
                code = run_bounded(command, stream, seconds)
            check.update(exit_code=code, status='passed' if code == 0 else 'failed')
        except subprocess.TimeoutExpired:
            check.update(exit_code=None, status='timed_out')
        write_json(manifest_path, manifest)
        if check['status'] != 'passed':
            raise ValueError(f'{name} failed; inspect {log}')
        return log

    try:
        probe_path = args.run_dir / 'topology.json'
        run('probe', ['prun', str(bin_dir / 'topo_probe.out'),
                      '--output-json', str(probe_path)], 180)
        topology = json.loads(probe_path.read_text(encoding='utf-8'))
        if topology.get('synthetic') or len(topology.get('pcie_latency_ns', [])) != 1 or \
                len(topology.get('mpi_latency_ns', [])) != 1:
            raise ValueError('Probe did not measure one OpenCL GPU and one MPI rank')
        run('harmony', ['prun', str(bin_dir / 'harmony_smoke.out'),
                        '--x', '16', '--y', '16', '--z', '32', '--iterations', '2',
                        '--repeat', '1', '--balance-mode', 'off', '--gather-final', '1'], 180)
        for workload in ('kneighbor', 'leanmd'):
            for algorithm in ('erad', 'hwtopolb'):
                name = f'{workload}_{algorithm}'
                csv_path = args.run_dir / f'{name}.csv'
                inventory_path = args.run_dir / f'{name}.inventory.json'
                log = run(name, ['prun', str(bin_dir / f'{workload}.out'),
                                 '--n', '100000', '--iterations', '20',
                                 '--balance-interval', '10', '--seed', '123',
                                 '--balance-algo', algorithm, '--perturbation', '0.10',
                                 '--output-csv', str(csv_path),
                                 '--device-inventory-json', str(inventory_path),
                                 '--quiet-runtime-metrics'], 300)
                with csv_path.open(newline='', encoding='utf-8') as stream:
                    records = list(csv.DictReader(stream))
                inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
                ranks = inventory.get('ranks', [])
                if len(records) != 20 or any(row.get('execution_mode') != 'gpu_opencl'
                                             for row in records):
                    raise ValueError(f'{name}: missing samples or CPU fallback')
                if len(ranks) != 1 or ranks[0].get('rank') != 0 or \
                        ranks[0].get('gpu_count') != 1 or \
                        host_name(ranks[0].get('host', '')) != args.expected_host:
                    raise ValueError(f'{name}: expected exactly one GPU on {args.expected_host}')
                output = log.read_text(encoding='utf-8')
                if re.findall(r'\bvalidation=([^\s]+)', output) != ['PASS'] or \
                        re.findall(r'\bmode=([^\s]+)', output) != ['gpu_opencl']:
                    raise ValueError(f'{name}: GPU mode or numerical validation missing')
        manifest['status'] = 'completed'
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        raise
    finally:
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-host', required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
