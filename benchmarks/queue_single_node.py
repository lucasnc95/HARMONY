#!/usr/bin/env python3
"""Queue independent one-rank GPU checks on the three legacy compute nodes."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from controlled_pbs import ROOT, WorkflowError, ensure_build, now, save, scheduler_record


NODES = ('compute-0-7', 'compute-0-8', 'compute-0-2')
SELECT = 'select=1:ncpus=4:mpiprocs=1:mem=8gb:ngpus=1:host={node}'


def job_name(experiment_id, node):
    return 'hsn' + hashlib.sha256(f'{experiment_id}:{node}'.encode()).hexdigest()[:8]


def queue(args):
    if os.environ.get('PBS_JOBID'):
        raise WorkflowError('Queue jobs on the frontend, outside a PBS allocation')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.experiment_id):
        raise WorkflowError('Experiment ID must be 1-64 ASCII letters, digits, _ or -')
    root = args.results_root.resolve() / f'single-node-{args.experiment_id}'
    if args.preview:
        for node in NODES:
            print(f'{node}: qsub -l {SELECT.format(node=node)} job_harmony_single_node.pbs')
        return
    if shutil.which('qsub') is None:
        raise WorkflowError('qsub is unavailable; run on the cluster frontend')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = root / 'workflow.json'
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('schema_version') != 1 or state.get('experiment_id') != args.experiment_id or \
                    state.get('nodes') != list(NODES):
                raise WorkflowError('Existing single-node workflow has different parameters')
        else:
            state = dict(schema_version=1, experiment_id=args.experiment_id,
                         nodes=list(NODES), stages={}, build_manifest_sha256=None)
            save(state_path, state)
        ensure_build(state, root)
        save(state_path, state)
        adoption = None
        if args.adopt_job_id:
            adopted_node, separator, adopted_id = args.adopt_job_id.partition('=')
            if not separator or adopted_node not in NODES or not adopted_id:
                raise WorkflowError('--adopt-job-id must be NODE=JOBID for a requested node')
            adoption = (adopted_node, adopted_id)
        for node in NODES:
            existing = state['stages'].get(node)
            if existing:
                if existing['status'] == 'submission_pending':
                    if not adoption or adoption[0] != node:
                        raise WorkflowError(f'{node}: qsub outcome is uncertain; inspect qstat '
                                            'and rerun with --adopt-job-id NODE=JOBID')
                    _, _, record = scheduler_record(adoption[1])
                    name = re.search(r'^\s*Job_Name\s*=\s*(\S+)', record,
                                     re.MULTILINE | re.IGNORECASE)
                    if not name or name.group(1) != existing['job_name']:
                        raise WorkflowError(f'{node}: adopted PBS job name does not match')
                    existing.update(status='submitted', job_id=adoption[1])
                    save(state_path, state)
                    adoption = None
                print(f'{node}: already queued as {existing["job_id"]}')
                continue
            if adoption:
                raise WorkflowError('--adopt-job-id does not identify a pending submission')
            run_dir = root / node
            run_dir.mkdir(exist_ok=True)
            output = run_dir / 'pbs.out'
            if any(path.exists() for path in (output, run_dir / 'pbs.status',
                                              run_dir / 'run_manifest.json')):
                raise WorkflowError(f'{node}: existing artifacts need inspection: {run_dir}')
            name = job_name(args.experiment_id, node)
            stage = dict(status='submission_pending', job_id=None, job_name=name,
                         run_dir=str(run_dir), submitted_at=now())
            state['stages'][node] = stage
            save(state_path, state)
            command = ['qsub', '-N', name, '-o', str(output), '-j', 'oe', '-V',
                       '-l', SELECT.format(node=node), str(ROOT / 'job_harmony_single_node.pbs')]
            env = dict(os.environ, HARMONY_SINGLE_NODE_RUN_DIR=str(run_dir),
                       HARMONY_SINGLE_NODE_EXPECTED_HOST=node)
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=60, check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise WorkflowError(f'{node}: qsub outcome is uncertain: {error}') from error
            stage['qsub_stdout'] = result.stdout.strip()
            stage['qsub_stderr'] = result.stderr.strip()
            job_id = result.stdout.strip().split()[0] if result.stdout.strip() else ''
            if result.returncode != 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', job_id):
                save(state_path, state)
                raise WorkflowError(f'{node}: qsub outcome is uncertain; inspect the scheduler')
            stage.update(status='submitted', job_id=job_id)
            save(state_path, state)
            print(f'{node}: queued PBS job {job_id}', flush=True)
        if adoption:
            raise WorkflowError('--adopt-job-id did not match a pending submission')
        print(f'All three independent jobs queued. Results: {root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'Results')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--adopt-job-id', help='Reconcile an uncertain qsub as NODE=JOBID')
    args = parser.parse_args()
    try:
        queue(args)
    except (WorkflowError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
