#!/usr/bin/env python3
"""Capture one traced comparison sample on its original full-run PBS allocation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from report_variability import InvalidRun, sha256, validate
from run_variability import run_bounded, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def plan(run_dir, ordinal, output_dir, sample_timeout_seconds=1800):
    run, rows, _ = validate(run_dir)
    sample = next((row for row in rows if row['ordinal'] == ordinal and
                   row['phase'] == 'full'), None)
    if sample is None:
        raise InvalidRun('Choose an accepted full-matrix sample ordinal (5..64)')
    pbs = run['environment']['pbs']
    nodefile = os.environ.get('PBS_NODEFILE')
    if os.environ.get('PBS_JOBID') != pbs['job_id'] or not nodefile or \
            not Path(nodefile).is_file() or \
            Path(nodefile).read_text(encoding='utf-8').splitlines() != pbs['slots']:
        raise InvalidRun('Diagnostic capture requires the original two-node PBS allocation')
    launcher = Path(str(pbs.get('launcher', '')))
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise InvalidRun('Recorded PBS launcher is unavailable')
    binary = Path(sample['argv'][0])
    expected = (run['build'].get('binaries') or {}).get(sample['workload'])
    if not isinstance(expected, dict) or not binary.is_file() or \
            not os.access(binary, os.X_OK) or sha256(binary) != expected.get('sha256'):
        raise InvalidRun('Diagnostic benchmark binary differs from full-run build')
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise InvalidRun(f'Diagnostic output directory already exists: {output_dir}')
    if output_dir.is_relative_to((run_dir / 'full').resolve()) or \
            output_dir.is_relative_to((run_dir / 'warmup').resolve()):
        raise InvalidRun('Diagnostic output directory cannot overlap timed samples')
    prefix = output_dir / 'trace'
    paths = dict(csv=str(output_dir / 'diagnostic.csv'),
                 log=str(output_dir / 'diagnostic.log'),
                 inventory=str(output_dir / 'diagnostic.inventory.json'),
                 rank0=f'{prefix}.rank0.csv', rank1=f'{prefix}.rank1.csv')
    command = [str(launcher), *sample['argv']]
    for flag, path in (('--output-csv', paths['csv']),
                       ('--device-inventory-json', paths['inventory'])):
        command[command.index(flag) + 1] = path
    command += ['--diagnostics-prefix', str(prefix)]
    return dict(schema_version=1, status='planned', exit_code=None, pbs=pbs,
                sample_timeout_seconds=sample_timeout_seconds,
                build_manifest_sha256=run['build_manifest_sha256'],
                binary_sha256=expected['sha256'], workload=sample['workload'],
                seed=sample['seed'], algorithm=sample['algorithm'],
                matched_sample_manifest=sample['manifest'], command=command,
                trace_prefix=str(prefix), artifacts=paths, artifact_sha256={},
                started_at=None, completed_at=None)


def capture(output_dir, manifest):
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / 'diagnostic.manifest.json'
    manifest['status'] = 'running'
    manifest['started_at'] = now()
    write_json(path, manifest)
    try:
        with open(manifest['artifacts']['log'], 'x', encoding='utf-8') as stream:
            manifest['exit_code'] = run_bounded(
                manifest['command'], stream, manifest['sample_timeout_seconds'])
        if manifest['exit_code'] != 0 or len([
                value for value in manifest['artifacts'].values() if Path(value).is_file()]) != 5:
            raise InvalidRun('Diagnostic command failed or omitted an artifact')
        # Artifact presence only proves capture. The report revalidates GPU mode,
        # numerical result, rank traces, and phase/partition consistency.
        manifest['status'] = 'captured'
    except subprocess.TimeoutExpired:
        manifest['status'] = 'rejected'
        manifest['rejection_reason'] = (
            f"Diagnostic command timed out after {manifest['sample_timeout_seconds']} seconds")
    except (OSError, InvalidRun) as error:
        manifest['status'] = 'rejected'
        manifest['rejection_reason'] = str(error)
    try:
        manifest['artifact_sha256'] = {
            key: sha256(Path(value)) for key, value in manifest['artifacts'].items()
            if Path(value).is_file()
        }
    except OSError as error:
        manifest['status'] = 'rejected'
        manifest['rejection_reason'] = f'Could not hash diagnostic artifacts: {error}'
    manifest['completed_at'] = now()
    write_json(path, manifest)
    return manifest['status'] == 'captured'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--sample-ordinal', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.sample_timeout_seconds <= 86400:
        parser.error('Sample timeout must be from 1 to 86400 seconds')
    try:
        manifest = plan(args.run_dir, args.sample_ordinal, args.output_dir,
                        args.sample_timeout_seconds)
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            return 0
        return 0 if capture(args.output_dir.resolve(), manifest) else 1
    except (InvalidRun, OSError, ValueError, KeyError, IndexError) as error:
        print(f'Diagnostic capture refused: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
