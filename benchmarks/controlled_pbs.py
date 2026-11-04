#!/usr/bin/env python3
"""Queue one immutable, resumable PBS comparison with dependent stages."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from run_variability import build_verified, digest


ROOT = Path(__file__).resolve().parents[1]
MODES = ('pilot', 'smoke', 'full', 'aba-kneighbor', 'aba-leanmd')


class WorkflowError(Exception):
    """A stage cannot safely advance without new evidence or intervention."""


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_digest():
    sources = sorted(set(
        path for directory in (ROOT / 'dcl', ROOT / 'benchmarks')
        for pattern in ('*.hpp', '*.cpp', '*.cl')
        for path in directory.rglob(pattern)
    ))
    value = hashlib.sha256()
    for source in sources:
        value.update(str(source.relative_to(ROOT)).encode() + b'\0' + source.read_bytes())
    return value.hexdigest()


def git_revision():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise WorkflowError('Run from a Git checkout on the cluster frontend')
    return result.stdout.strip()


def run_checked(command, **kwargs):
    try:
        return subprocess.run(command, check=False, text=True, **kwargs)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkflowError(f'{command[0]} could not complete: {error}') from error


def ensure_build(workflow, path):
    bin_dir = ROOT / 'build' / 'cluster'
    manifest_path = bin_dir / 'build_manifest.json'
    revision = git_revision()
    current_source = source_digest()

    def inspect():
        try:
            build = build_verified(bin_dir, manifest_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise WorkflowError(f'Build manifest or binary is invalid: {error}') from error
        if build.get('source_sha256') != current_source or \
                build.get('source_revision') != revision:
            raise WorkflowError('Build source digest or revision differs from this checkout')
        return digest(manifest_path)

    try:
        manifest_sha = inspect()
    except WorkflowError:
        if workflow['stages']:
            raise
        result = run_checked(['bash', str(ROOT / 'cluster_build.sh')], cwd=ROOT)
        if result.returncode != 0:
            raise WorkflowError(f'Frontend build exited {result.returncode}')
        manifest_sha = inspect()
    if workflow['build_manifest_sha256'] not in (None, manifest_sha):
        raise WorkflowError('Frontend build changed after the first PBS submission')
    workflow['build_manifest_sha256'] = manifest_sha
    snapshot = path / 'build_manifest.json'
    copy_once(manifest_path, snapshot)


def copy_once(source, destination):
    if not source.is_file():
        raise WorkflowError(f'Missing artifact: {source}')
    if destination.exists():
        if not destination.is_file() or digest(source) != digest(destination):
            raise WorkflowError(f'Existing artifact differs: {destination}')
        return
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', dir=destination.parent)
    try:
        with source.open('rb') as reader, os.fdopen(fd, 'wb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if digest(source) != digest(destination):
                raise WorkflowError(f'Existing artifact differs: {destination}') from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def preview(mode, run_dir, workload=None):
    command = [sys.executable, str(ROOT / 'benchmarks' / 'run_variability.py'),
               '--dry-run', '--mode', 'aba' if workload else mode,
               '--run-dir', str(run_dir)]
    if workload:
        command += ['--aba-workload', workload]
    result = run_checked(command, cwd=ROOT, capture_output=True)
    if result.returncode != 0:
        raise WorkflowError(f'Matrix preview failed: {result.stderr.strip()}')
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise WorkflowError(f'Matrix preview is not JSON: {error}') from error


def stage_paths(workflow_root, mode):
    run_dir = workflow_root / mode
    return run_dir, workflow_root / f'{mode}.pbs.out', workflow_root / f'{mode}.pbs.status'


def scheduler_record(job_id, saved=None):
    if saved is not None:
        raw = saved.read_text(encoding='utf-8')
    else:
        result = run_checked(['qstat', '-xf', job_id], capture_output=True, timeout=20)
        if result.returncode != 0:
            raise WorkflowError(f'qstat cannot verify {job_id}; preserve the job and provide '
                                '--scheduler-record MODE=PATH with its qstat -xf output')
        raw = result.stdout
    identity = re.search(r'^\s*Job Id:\s*(\S+)', raw, re.MULTILINE | re.IGNORECASE)
    if not identity or identity.group(1) != job_id:
        raise WorkflowError(f'Scheduler record does not identify PBS job {job_id}')
    state = re.search(r'^\s*job_state\s*=\s*(\S+)', raw, re.MULTILINE | re.IGNORECASE)
    exit_status = re.search(r'^\s*Exit_status\s*=\s*(-?\d+)', raw,
                            re.MULTILINE | re.IGNORECASE)
    if not state:
        raise WorkflowError(f'qstat output for {job_id} has no job_state')
    return state.group(1), int(exit_status.group(1)) if exit_status else None, raw


def script_status(path, job_id):
    if not path.is_file():
        raise WorkflowError(f'PBS script did not preserve its exit status: {path}')
    values = dict(re.findall(r'^([a-z_]+)=(\S+)$', path.read_text(encoding='utf-8'),
                             re.MULTILINE))
    if values.get('pbs_job_id') != job_id or not re.fullmatch(
            r'\d+', values.get('script_exit_code', '')):
        raise WorkflowError('PBS script status is malformed or belongs to another job')
    return int(values['script_exit_code'])


def smoke_evidence_valid(run_dir, manifest):
    comparison = manifest.get('smoke_comparison')
    if not isinstance(comparison, dict) or comparison.get('condition') != dict(
            workload='kneighbor', seed=123, algorithm='hwtopolb',
            n=100000, iterations=100):
        return False
    for key in ('unprofiled_loop_s', 'profiled_loop_s', 'trace_overhead_ratio'):
        value = comparison.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            return False
    trace = comparison.get('trace')
    if not isinstance(trace, dict) or not isinstance(trace.get('rank_traces'), list) or \
            len(trace['rank_traces']) != 2:
        return False
    artifacts = [dict(path=trace.get('summary'), sha256=trace.get('summary_sha256')),
                 *trace['rank_traces']]
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str) or \
                not isinstance(item.get('sha256'), str):
            return False
        path = Path(item['path']).resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or \
                digest(path) != item['sha256']:
            return False
    return True


def produce_once(workflow, key, output, command):
    recorded = workflow['reports'].get(key)
    if recorded:
        if not output.is_file() or digest(output) != recorded:
            raise WorkflowError(f'Preserved report changed or disappeared: {output}')
        return
    fd, temporary = tempfile.mkstemp(prefix=f'.{output.stem}.', suffix='.md',
                                     dir=output.parent)
    os.close(fd)
    Path(temporary).unlink()
    try:
        result = run_checked([*command[:-1], temporary], cwd=ROOT, capture_output=True)
        if result.returncode != 0:
            raise WorkflowError(f'Report command failed: {result.stderr.strip()}')
        if not Path(temporary).is_file():
            raise WorkflowError(f'Report command produced no file: {output}')
        if output.exists():
            if digest(output) != digest(Path(temporary)):
                raise WorkflowError(f'Existing report differs from regenerated evidence: {output}')
        else:
            os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    workflow['reports'][key] = digest(output)


def submit(workflow, workflow_root, mode, dependency=None):
    if shutil.which('qsub') is None:
        raise WorkflowError('qsub is unavailable; run this script on the cluster frontend')
    run_dir, output, status = stage_paths(workflow_root, mode)
    if any(path.exists() for path in (run_dir, output, status)):
        raise WorkflowError(f'Fresh PBS artifacts already exist for {mode}; choose another experiment ID')
    workload = mode.removeprefix('aba-') if mode.startswith('aba-') else None
    plan = preview(mode, run_dir, workload)
    plan_path = workflow_root / f'{mode}.preview.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise WorkflowError(f'Existing preview changed: {plan_path}')
    else:
        save(plan_path, plan)
    code = {'pilot': 'p', 'smoke': 's', 'full': 'f',
            'aba-kneighbor': 'ak', 'aba-leanmd': 'al'}[mode]
    job_name = 'hgv' + code + hashlib.sha256(
        workflow['experiment_id'].encode()).hexdigest()[:8]
    stage = dict(status='submission_pending', job_id=None, job_name=job_name,
                 run_dir=str(run_dir), output=str(output), dependency=dependency,
                 submitted_at=now())
    workflow['stages'][mode] = stage
    save(workflow_root / 'workflow.json', workflow)
    env = dict(os.environ, HARMONY_VARIABILITY_RUN_DIR=str(run_dir),
               HARMONY_PBS_OUTPUT_PATH=str(output),
               HARMONY_VARIABILITY_MODE='aba' if workload else mode,
               HARMONY_SAMPLE_TIMEOUT_SECONDS=str(workflow['sample_timeout_seconds']))
    if workload:
        env['HARMONY_ABA_WORKLOAD'] = workload
    if mode == 'full' and workflow['diagnostic_ordinal']:
        env['HARMONY_DIAGNOSTIC_SAMPLE_ORDINAL'] = str(workflow['diagnostic_ordinal'])
    command = ['qsub', '-N', job_name, '-o', str(output), '-j', 'oe', '-V']
    if dependency:
        command += ['-W', f'depend=afterok:{dependency}']
    command += [str(ROOT / 'job_harmony_variability.pbs')]
    try:
        result = run_checked(command, cwd=ROOT, env=env, capture_output=True, timeout=60)
    except WorkflowError:
        raise WorkflowError('qsub outcome is uncertain. This experiment will not be '
                            'submitted again automatically; inspect qstat and use '
                            '--adopt-job-id if it was accepted') from None
    stage['qsub_stdout'] = result.stdout.strip()
    stage['qsub_stderr'] = result.stderr.strip()
    job_id = result.stdout.strip().split()[0] if result.stdout.strip() else ''
    if result.returncode != 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', job_id):
        save(workflow_root / 'workflow.json', workflow)
        raise WorkflowError('qsub outcome is uncertain. Inspect the scheduler; do not '
                            'resubmit this experiment ID without reconciling the job')
    stage.update(status='submitted', job_id=job_id)
    save(workflow_root / 'workflow.json', workflow)
    print(f'{mode}: submitted PBS job {job_id}', flush=True)


def collect(workflow, workflow_root, mode, adopted_job_id=None, saved_record=None):
    stage = workflow['stages'][mode]
    if adopted_job_id and stage['status'] != 'submission_pending':
        raise WorkflowError('--adopt-job-id applies only to an uncertain submission')
    if stage['status'] == 'submission_pending':
        if not adopted_job_id:
            raise WorkflowError(f'{mode}: qsub outcome is unknown. Inspect qstat and '
                                'rerun with --adopt-job-id JOBID; no duplicate submission was made')
        state, _, record = scheduler_record(adopted_job_id, saved_record)
        name = re.search(r'^\s*Job_Name\s*=\s*(\S+)', record,
                         re.MULTILINE | re.IGNORECASE)
        if not name or name.group(1) != stage['job_name']:
            raise WorkflowError('Adopted PBS job name does not match this experiment')
        stage.update(status='submitted', job_id=adopted_job_id)
        save(workflow_root / 'workflow.json', workflow)
        print(f'{mode}: adopted {adopted_job_id} (scheduler state {state})')
    if stage['status'] == 'complete':
        return True
    if stage['status'] == 'failed':
        raise WorkflowError(f'{mode}: preserved failed run; use a new experiment ID after repair')
    job_id = stage['job_id']
    terminal_record = workflow_root / f'{mode}.pbs.scheduler-status.txt'
    state, exit_status, record = scheduler_record(
        job_id, terminal_record if terminal_record.is_file() else saved_record)
    if state.upper() not in ('F', 'C'):
        if stage.get('last_observed_state') != state:
            stage['last_observed_state'] = state
            save(workflow_root / 'workflow.json', workflow)
            print(f'{mode}: PBS job {job_id} is {state}; waiting', flush=True)
        return False
    if exit_status is None:
        raise WorkflowError(f'{mode}: terminal PBS record has no Exit_status')
    if not terminal_record.exists():
        with terminal_record.open('x', encoding='utf-8') as stream:
            stream.write(record)
    if not stage.get('terminal_observed_at'):
        stage['terminal_observed_at'] = now()
        save(workflow_root / 'workflow.json', workflow)
    run_dir, output, status = stage_paths(workflow_root, mode)
    def wait_for_artifacts(missing):
        observed = datetime.fromisoformat(stage['terminal_observed_at'])
        if (datetime.now(timezone.utc) - observed).total_seconds() >= 120:
            raise WorkflowError(f'{mode}: terminal PBS job still lacks artifacts after 120 seconds: '
                                f'{missing}')
        if not stage.get('waiting_for_artifacts'):
            stage['waiting_for_artifacts'] = True
            save(workflow_root / 'workflow.json', workflow)
            print(f'{mode}: PBS job finished; waiting for output staging: {missing}', flush=True)
        return False

    missing = [str(path) for path in (run_dir, output, status) if not path.exists()]
    if missing:
        return wait_for_artifacts(missing)
    copy_once(output, run_dir / 'pbs.out')
    copy_once(status, run_dir / 'pbs.status')
    copy_once(workflow_root / 'build_manifest.json', run_dir / 'build_manifest.json')
    scheduler_path = run_dir / 'pbs.scheduler-status.txt'
    copy_once(terminal_record, scheduler_path)
    script_exit = script_status(status, job_id)
    stage.update(scheduler_exit_status=exit_status, script_exit_status=script_exit,
                 completed_at=now())
    if exit_status != 0 or script_exit != 0:
        stage['status'] = 'failed'
        save(workflow_root / 'workflow.json', workflow)
        raise WorkflowError(f'{mode}: PBS or script exited nonzero; raw artifacts were preserved')
    manifest_path = run_dir / 'run_manifest.json'
    if not manifest_path.is_file():
        return wait_for_artifacts(str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_mode = 'aba' if mode.startswith('aba-') else mode
    if manifest.get('mode') != expected_mode or manifest.get('status') != 'completed' or \
            manifest.get('rejected_samples') != 0 or \
            manifest.get('build_manifest_sha256') != workflow['build_manifest_sha256'] or \
            manifest.get('environment', {}).get('pbs', {}).get('job_id') != job_id:
        stage['status'] = 'failed'
        save(workflow_root / 'workflow.json', workflow)
        raise WorkflowError(f'{mode}: run manifest failed admission; artifacts were preserved')
    if mode == 'smoke' and not smoke_evidence_valid(run_dir, manifest):
        stage['status'] = 'failed'
        save(workflow_root / 'workflow.json', workflow)
        raise WorkflowError('smoke: rank trace or overhead evidence is incomplete')
    stage['status'] = 'complete'
    save(workflow_root / 'workflow.json', workflow)
    print(f'{mode}: PBS job {job_id} and run manifest completed successfully')
    return True


def report(workflow, workflow_root, aba_modes):
    full = workflow_root / 'full'
    diagnostic = full / f"diagnostic-{workflow['diagnostic_ordinal']}" / 'diagnostic.manifest.json'
    command = [sys.executable, str(ROOT / 'benchmarks' / 'report_variability.py'),
               '--run-dir', str(full)]
    if workflow['diagnostic_ordinal']:
        if not diagnostic.is_file():
            raise WorkflowError(f'Expected diagnostic capture is missing: {diagnostic}')
        command += ['--diagnostic-manifest', str(diagnostic)]
    if aba_modes:
        key = 'full-report-' + '-'.join(aba_modes)
        for mode in aba_modes:
            command += ['--aba-run-dir', str(workflow_root / mode)]
    else:
        key = 'full-report'
    output = workflow_root / f'{key}.md'
    produce_once(workflow, key, output, [*command, '--output', str(output)])
    save(workflow_root / 'workflow.json', workflow)
    print(f'Validated report: {output}')


def advance_once(args, finalizing=False, selected_aba_modes=None):
    if os.environ.get('PBS_JOBID') and not finalizing:
        raise WorkflowError('Run this coordinator on the frontend, outside a PBS allocation')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.experiment_id):
        raise WorkflowError('Experiment ID must be 1-64 ASCII letters, digits, _ or -')
    records = {}
    for item in args.scheduler_record:
        mode, separator, raw_path = item.partition('=')
        if not separator or mode not in MODES or not raw_path or mode in records:
            raise WorkflowError('--scheduler-record must be a unique MODE=PATH entry')
        records[mode] = Path(raw_path).resolve()
    workflow_root = args.results_root.resolve() / f'controlled-{args.experiment_id}'
    if args.preview:
        plans = {mode: preview(mode, workflow_root / mode)
                 for mode in ('pilot', 'smoke', 'full')}
        for workload in args.aba_workload:
            mode = f'aba-{workload}'
            plans[mode] = preview(mode, workflow_root / mode, workload)
        print(json.dumps(dict(experiment_id=args.experiment_id, plans=plans), indent=2))
        return True
    workflow_root.mkdir(parents=True, exist_ok=True)
    with (workflow_root / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = workflow_root / 'workflow.json'
        if state_path.is_file():
            workflow = json.loads(state_path.read_text(encoding='utf-8'))
            if workflow.get('schema_version') != 1 or \
                    workflow.get('experiment_id') != args.experiment_id or \
                    workflow.get('diagnostic_ordinal') != args.diagnostic_ordinal or \
                    workflow.get('sample_timeout_seconds') != args.sample_timeout_seconds:
                raise WorkflowError('Existing workflow parameters differ; reuse the original options')
        else:
            workflow = dict(schema_version=1, experiment_id=args.experiment_id,
                            diagnostic_ordinal=args.diagnostic_ordinal,
                            sample_timeout_seconds=args.sample_timeout_seconds,
                            build_manifest_sha256=None, stages={}, reports={},
                            aba_modes=[])
            save(state_path, workflow)
        if not finalizing and shutil.which('qsub') is None:
            raise WorkflowError('qsub is unavailable; run this script on the cluster frontend')
        ensure_build(workflow, workflow_root)
        for workload in args.aba_workload:
            mode = f'aba-{workload}'
            if mode not in workflow['aba_modes']:
                workflow['aba_modes'].append(mode)
        save(state_path, workflow)
        pending = [mode for mode, stage in workflow['stages'].items()
                   if stage['status'] == 'submission_pending']
        if args.adopt_job_id and len(pending) != 1:
            raise WorkflowError('--adopt-job-id requires exactly one uncertain submission')
        modes = ('pilot', 'smoke', 'full', *(selected_aba_modes if finalizing else workflow['aba_modes']))
        for mode in modes:
            if mode not in workflow['stages']:
                if finalizing:
                    raise WorkflowError(f'{mode}: expected queued PBS job is missing')
                if args.adopt_job_id:
                    raise WorkflowError('--adopt-job-id requires an existing uncertain submission')
                submit(workflow, workflow_root, mode)
                return False
            adopted = args.adopt_job_id if pending and mode == pending[0] else None
            if not collect(workflow, workflow_root, mode, adopted, records.get(mode)):
                return False
            if mode == 'pilot':
                run_dir = workflow_root / 'pilot'
                output = run_dir / 'pilot-analysis.md'
                produce_once(workflow, 'pilot-analysis', output,
                             [sys.executable, str(ROOT / 'benchmarks' / 'analyze_pilot.py'),
                              '--run-dir', str(run_dir), '--output', str(output)])
                save(state_path, workflow)
            if mode == 'full':
                report(workflow, workflow_root, [])
        aba_modes = selected_aba_modes if finalizing else workflow['aba_modes']
        if aba_modes:
            report(workflow, workflow_root, aba_modes)
        if finalizing:
            key = 'baseline' if not aba_modes else '-'.join(aba_modes)
            entry = workflow.get('finalizers', {}).get(key)
            if entry:
                entry.update(status='complete', completed_at=now())
                save(state_path, workflow)
        print(f'Workflow complete. Preserve {workflow_root} and provide its raw artifacts for review.')
        return True


def adopt_queued(entry, job_id):
    state, _, record = scheduler_record(job_id)
    name = re.search(r'^\s*Job_Name\s*=\s*(\S+)', record,
                     re.MULTILINE | re.IGNORECASE)
    if not name or name.group(1) != entry['job_name']:
        raise WorkflowError('Adopted PBS job name does not match this experiment')
    entry.update(status='submitted', job_id=job_id)
    print(f'Adopted PBS job {job_id} (scheduler state {state})', flush=True)


def queue_finalizer(workflow, workflow_root, key, dependency):
    output = workflow_root / f'finalize-{key}.pbs.out'
    if output.exists():
        raise WorkflowError(f'Fresh finalizer output already exists: {output}')
    job_name = 'hgvz' + hashlib.sha256(
        (workflow['experiment_id'] + key).encode()).hexdigest()[:8]
    entry = dict(status='submission_pending', job_id=None, job_name=job_name,
                 dependency=dependency, output=str(output), submitted_at=now())
    workflow['finalizers'][key] = entry
    save(workflow_root / 'workflow.json', workflow)
    env = dict(os.environ, HARMONY_VARIABILITY_EXPERIMENT_ID=workflow['experiment_id'],
               HARMONY_VARIABILITY_RESULTS_ROOT=str(workflow_root.parent),
               HARMONY_FINALIZE_ABA_MODES=','.join(workflow['aba_modes']))
    command = ['qsub', '-N', job_name, '-o', str(output), '-j', 'oe', '-V']
    if dependency:
        command += ['-W', f'depend=afterok:{dependency}']
    command += [str(ROOT / 'job_harmony_variability_finalize.pbs')]
    try:
        result = run_checked(command, cwd=ROOT, env=env, capture_output=True, timeout=60)
    except WorkflowError:
        raise WorkflowError('Finalizer qsub outcome is uncertain; inspect qstat and '
                            'resume with --adopt-job-id if accepted') from None
    entry['qsub_stdout'] = result.stdout.strip()
    entry['qsub_stderr'] = result.stderr.strip()
    job_id = result.stdout.strip().split()[0] if result.stdout.strip() else ''
    if result.returncode != 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', job_id):
        save(workflow_root / 'workflow.json', workflow)
        raise WorkflowError('Finalizer qsub outcome is uncertain; inspect the scheduler')
    entry.update(status='submitted', job_id=job_id)
    save(workflow_root / 'workflow.json', workflow)
    print(f'finalize-{key}: queued PBS job {job_id}', flush=True)


def queue_all(args):
    if os.environ.get('PBS_JOBID'):
        raise WorkflowError('Queue jobs on the frontend, outside a PBS allocation')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.experiment_id):
        raise WorkflowError('Experiment ID must be 1-64 ASCII letters, digits, _ or -')
    if shutil.which('qsub') is None:
        raise WorkflowError('qsub is unavailable; run this script on the cluster frontend')
    workflow_root = args.results_root.resolve() / f'controlled-{args.experiment_id}'
    workflow_root.mkdir(parents=True, exist_ok=True)
    with (workflow_root / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = workflow_root / 'workflow.json'
        if state_path.is_file():
            workflow = json.loads(state_path.read_text(encoding='utf-8'))
            if workflow.get('schema_version') != 1 or \
                    workflow.get('experiment_id') != args.experiment_id or \
                    workflow.get('diagnostic_ordinal') != args.diagnostic_ordinal or \
                    workflow.get('sample_timeout_seconds') != args.sample_timeout_seconds:
                raise WorkflowError('Existing workflow parameters differ; reuse the original options')
        else:
            workflow = dict(schema_version=1, experiment_id=args.experiment_id,
                            diagnostic_ordinal=args.diagnostic_ordinal,
                            sample_timeout_seconds=args.sample_timeout_seconds,
                            build_manifest_sha256=None, stages={}, reports={},
                            aba_modes=[], finalizers={})
            save(state_path, workflow)
        workflow.setdefault('finalizers', {})
        ensure_build(workflow, workflow_root)
        for workload in args.aba_workload:
            mode = f'aba-{workload}'
            if mode not in workflow['aba_modes']:
                workflow['aba_modes'].append(mode)
        save(state_path, workflow)
        key = 'baseline' if not workflow['aba_modes'] else '-'.join(workflow['aba_modes'])
        modes = ('pilot', 'smoke', 'full', *workflow['aba_modes'])
        pending = [entry for entry in (
            *(workflow['stages'][mode] for mode in modes if mode in workflow['stages']),
            *(entry for entry in workflow['finalizers'].values()))
            if entry['status'] == 'submission_pending']
        if args.adopt_job_id and len(pending) != 1:
            raise WorkflowError('--adopt-job-id requires exactly one uncertain submission')
        if pending:
            if not args.adopt_job_id or len(pending) != 1:
                raise WorkflowError('A qsub outcome is uncertain; reconcile it with --adopt-job-id')
            adopt_queued(pending[0], args.adopt_job_id)
            save(state_path, workflow)
        previous = None
        for mode in modes:
            stage = workflow['stages'].get(mode)
            if stage is None:
                submit(workflow, workflow_root, mode, previous)
                stage = workflow['stages'][mode]
            elif stage['status'] == 'submission_pending':
                raise WorkflowError(f'{mode}: qsub outcome is uncertain; use --adopt-job-id')
            if not stage.get('job_id'):
                raise WorkflowError(f'{mode}: no verified PBS job ID')
            previous = None if stage['status'] == 'complete' else stage['job_id']
        finalizer = workflow['finalizers'].get(key)
        if finalizer is None:
            queue_finalizer(workflow, workflow_root, key, previous)
        elif finalizer['status'] == 'submission_pending':
            raise WorkflowError('Finalizer qsub outcome is uncertain; use --adopt-job-id')
        print(f'All PBS jobs queued. Results: {workflow_root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'Results')
    parser.add_argument('--diagnostic-ordinal', type=int, default=8)
    parser.add_argument('--sample-timeout-seconds', type=int, default=1800)
    parser.add_argument('--poll-interval-seconds', type=int, default=30,
                        help='Seconds between scheduler checks while waiting (default: 30)')
    parser.add_argument('--once', action='store_true',
                        help='Submit or check one stage and exit without waiting')
    parser.add_argument('--watch', action='store_true',
                        help='Wait on the frontend and advance stages there')
    parser.add_argument('--finalize', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--aba-workload', choices=('kneighbor', 'leanmd'),
                        action='append', default=[])
    parser.add_argument('--adopt-job-id')
    parser.add_argument('--scheduler-record', action='append', default=[], metavar='MODE=PATH',
                        help='Saved qstat -xf output for a terminal job no longer in qstat')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if args.diagnostic_ordinal not in (0, *range(5, 65)):
        parser.error('Diagnostic ordinal must be 0 (disabled) or 5-64')
    if not 1 <= args.sample_timeout_seconds <= 86400:
        parser.error('Sample timeout must be from 1 to 86400 seconds')
    if not 1 <= args.poll_interval_seconds <= 3600:
        parser.error('Poll interval must be from 1 to 3600 seconds')
    try:
        if args.preview:
            advance_once(args)
        elif args.finalize:
            workflow_root = args.results_root.resolve() / f'controlled-{args.experiment_id}'
            workflow = json.loads((workflow_root / 'workflow.json').read_text(encoding='utf-8'))
            args.diagnostic_ordinal = workflow['diagnostic_ordinal']
            args.sample_timeout_seconds = workflow['sample_timeout_seconds']
            selected = [mode for mode in os.environ.get('HARMONY_FINALIZE_ABA_MODES', '').split(',')
                        if mode]
            if any(mode not in workflow['aba_modes'] for mode in selected):
                raise WorkflowError('Finalizer A/B/A modes differ from the queued workflow')
            deadline = time.monotonic() + 600
            while not advance_once(args, finalizing=True, selected_aba_modes=selected):
                if time.monotonic() >= deadline:
                    raise WorkflowError('Finalizer waited 600 seconds for PBS output staging')
                time.sleep(5)
        elif args.watch or args.once:
            while True:
                if advance_once(args) or args.once:
                    break
                args.adopt_job_id = None
                time.sleep(args.poll_interval_seconds)
        else:
            queue_all(args)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; rerun the same command and experiment ID to resume.\n')
    except (WorkflowError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
