"""Command-level checks for resumable PBS submission and evidence collection."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest


REPO = Path(__file__).resolve().parents[1]


class ControlledPbsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'benchmarks').mkdir()
        for name in ('controlled_pbs.py', 'run_variability.py'):
            shutil.copy2(REPO / 'benchmarks' / name, self.root / 'benchmarks' / name)
        for name in ('analyze_pilot.py', 'report_variability.py'):
            (self.root / 'benchmarks' / name).write_text(
                '#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
                "Path(sys.argv[sys.argv.index('--output') + 1]).write_text('validated\\n')\n")
        (self.root / 'cluster_build.sh').write_text('#!/bin/bash\nexit 99\n')
        for name in ('job_harmony_variability.pbs', 'job_harmony_variability_finalize.pbs'):
            shutil.copy2(REPO / name, self.root / name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', 'add', 'benchmarks', 'cluster_build.sh',
                        'job_harmony_variability.pbs',
                        'job_harmony_variability_finalize.pbs'], cwd=self.root, check=True)
        subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root,
                                           text=True).strip()
        binary_dir = self.root / 'build' / 'cluster'
        binary_dir.mkdir(parents=True)
        binaries = {}
        for name in ('kneighbor', 'leanmd', 'test_async_halo_opencl'):
            path = binary_dir / f'{name}.out'
            path.write_text('#!/bin/sh\nexit 0\n')
            path.chmod(0o755)
            binaries[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (binary_dir / 'build_manifest.json').write_text(json.dumps(dict(
            source_revision=revision, source_sha256=hashlib.sha256(b'').hexdigest(),
            binaries=binaries)))
        scripts = self.root / 'fake-bin'
        scripts.mkdir()
        qsub = scripts / 'qsub'
        qsub.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
counter = Path(os.environ['FAKE_QSUB_COUNT'])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if os.environ.get('FAKE_QSUB_LOG'):
    with Path(os.environ['FAKE_QSUB_LOG']).open('a') as log:
        log.write(json.dumps(dict(argv=sys.argv[1:], mode=os.environ.get('HARMONY_VARIABILITY_MODE'),
                                  finalize=os.environ.get('HARMONY_FINALIZE_ABA_MODES'))) + '\\n')
if os.environ.get('FAKE_QSUB_BAD') or os.environ.get('FAKE_QSUB_BAD_AT') == str(count):
    print('submission response unavailable')
    sys.exit(1)
print(f'{122 + count}.fake')
''')
        qsub.chmod(0o755)
        qstat = scripts / 'qstat'
        qstat.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
if os.environ.get('FAKE_QSTAT_MISSING'):
    sys.exit(153)
state = os.environ.get('FAKE_JOB_STATE', 'R')
if os.environ.get('FAKE_AUTO_DIR'):
    state = 'F' if (Path(os.environ['FAKE_AUTO_DIR']) / (sys.argv[-1] + '.done')).exists() else 'Q'
print(f'Job Id: {sys.argv[-1]}')
print(f'    Job_Name = {os.environ.get("FAKE_JOB_NAME", "unknown")}')
print(f'    job_state = {state}')
if state == 'F':
    print(f'    Exit_status = {os.environ.get("FAKE_JOB_EXIT", "0")}')
''')
        qstat.chmod(0o755)
        self.env = dict(os.environ, PATH=str(scripts) + os.pathsep + os.environ['PATH'],
                        FAKE_QSUB_COUNT=str(self.root / 'qsub-count'),
                        FAKE_QSUB_LOG=str(self.root / 'qsub-log.jsonl'))
        self.command = [sys.executable, str(self.root / 'benchmarks' / 'controlled_pbs.py'),
                        '--experiment-id', 'sample', '--results-root', str(self.root / 'Results')]
        self.workflow_root = self.root / 'Results' / 'controlled-sample'

    def invoke(self, *extra, env=None):
        return subprocess.run([*self.command, '--once', *extra], cwd=self.root,
                              env=env or self.env, capture_output=True, text=True)

    def workflow(self):
        return json.loads((self.workflow_root / 'workflow.json').read_text())

    def requests(self):
        return [json.loads(line) for line in (self.root / 'qsub-log.jsonl').read_text().splitlines()]

    def test_default_command_queues_every_stage_and_finalizer_once(self):
        first = subprocess.run(self.command, cwd=self.root, env=self.env,
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn('All PBS jobs queued', first.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '4')
        requests = self.requests()
        self.assertEqual([item['mode'] for item in requests],
                         ['pilot', 'smoke', 'full', None])
        self.assertEqual([item['finalize'] for item in requests],
                         [None, None, None, ''])
        for index, prior in [(1, '123.fake'), (2, '124.fake'), (3, '125.fake')]:
            self.assertIn(f'depend=afterok:{prior}', requests[index]['argv'])
        self.assertFalse(any(arg.startswith('depend=') for arg in requests[0]['argv']))
        self.assertEqual([self.workflow()['stages'][mode]['job_id']
                          for mode in ('pilot', 'smoke', 'full')],
                         ['123.fake', '124.fake', '125.fake'])
        self.assertEqual(self.workflow()['finalizers']['baseline']['job_id'], '126.fake')
        second = subprocess.run(self.command, cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '4')

    def test_queue_resumes_after_ambiguous_second_submission(self):
        bad = dict(self.env, FAKE_QSUB_BAD_AT='2')
        first = subprocess.run(self.command, cwd=self.root, env=bad,
                               capture_output=True, text=True, timeout=10)
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self.workflow()['stages']['smoke']['status'], 'submission_pending')
        repeated = subprocess.run(self.command, cwd=self.root, env=self.env,
                                  capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')
        name = self.workflow()['stages']['smoke']['job_name']
        adopted = subprocess.run([*self.command, '--adopt-job-id', '124.fake'],
                                 cwd=self.root, env=dict(self.env, FAKE_JOB_NAME=name),
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(adopted.returncode, 0, adopted.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '4')
        self.assertIn('depend=afterok:124.fake', self.requests()[2]['argv'])

    def test_optional_aba_jobs_are_queued_in_the_same_dependency_chain(self):
        command = [*self.command, '--aba-workload', 'kneighbor',
                   '--aba-workload', 'leanmd']
        result = subprocess.run(command, cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '6')
        requests = self.requests()
        self.assertEqual([item['mode'] for item in requests],
                         ['pilot', 'smoke', 'full', 'aba', 'aba', None])
        self.assertEqual(requests[-1]['finalize'], 'aba-kneighbor,aba-leanmd')
        self.assertIn('depend=afterok:125.fake', requests[3]['argv'])
        self.assertIn('depend=afterok:126.fake', requests[4]['argv'])
        self.assertIn('depend=afterok:127.fake', requests[5]['argv'])

    def test_later_aba_followup_queues_after_completed_baseline(self):
        queued = subprocess.run(self.command, cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(queued.returncode, 0, queued.stderr)
        for mode, job_id in [('pilot', '123.fake'), ('smoke', '124.fake'),
                             ('full', '125.fake')]:
            self.finish_job(mode, job_id)
        finalized = subprocess.run([*self.command, '--finalize'], cwd=self.root,
                                   env=dict(self.env, PBS_JOBID='126.fake', FAKE_JOB_STATE='F'),
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(finalized.returncode, 0, finalized.stderr)
        followup = subprocess.run([*self.command, '--aba-workload', 'kneighbor'],
                                  cwd=self.root, env=self.env,
                                  capture_output=True, text=True, timeout=10)
        self.assertEqual(followup.returncode, 0, followup.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '6')
        self.assertEqual(self.requests()[4]['mode'], 'aba')
        self.assertFalse(any(arg.startswith('depend=') for arg in self.requests()[4]['argv']))
        self.assertIn('depend=afterok:127.fake', self.requests()[5]['argv'])

    def test_finalizer_collects_queued_jobs_inside_pbs_allocation(self):
        queued = subprocess.run(self.command, cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(queued.returncode, 0, queued.stderr)
        for mode, job_id in [('pilot', '123.fake'), ('smoke', '124.fake'),
                             ('full', '125.fake')]:
            self.finish_job(mode, job_id)
        finalizer = subprocess.run([*self.command, '--finalize'], cwd=self.root,
                                   env=dict(self.env, PBS_JOBID='126.fake', FAKE_JOB_STATE='F'),
                                   capture_output=True, text=True, timeout=10)
        self.assertEqual(finalizer.returncode, 0, finalizer.stderr)
        self.assertTrue((self.workflow_root / 'full-report.md').is_file())
        self.assertEqual({mode: stage['status'] for mode, stage in self.workflow()['stages'].items()},
                         {'pilot': 'complete', 'smoke': 'complete', 'full': 'complete'})

    def test_pilot_job_fails_when_its_analysis_fails(self):
        scripts = self.root / 'fake-bin'
        (scripts / 'module').write_text('#!/bin/sh\nexit 0\n')
        (scripts / 'module').chmod(0o755)
        (scripts / 'python3').write_text(
            '#!/bin/sh\ncase "$1" in\n'
            '  benchmarks/run_variability.py) exit 0 ;;\n'
            '  benchmarks/analyze_pilot.py) exit 42 ;;\n'
            'esac\nexit 99\n')
        (scripts / 'python3').chmod(0o755)
        env = dict(self.env, PBS_O_WORKDIR=str(self.root), PBS_JOBID='123.fake',
                   HARMONY_VARIABILITY_MODE='pilot',
                   HARMONY_VARIABILITY_RUN_DIR=str(self.root / 'pilot'))
        result = subprocess.run(['bash', str(self.root / 'job_harmony_variability.pbs')],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertIn('script_exit_code=42',
                      (self.root / 'pilot.pbs.status').read_text())

    def finish_job(self, mode, job_id):
        run_dir = self.workflow_root / mode
        run_dir.mkdir()
        (self.workflow_root / f'{mode}.pbs.out').write_text(f'PBS output for {job_id}\n')
        (self.workflow_root / f'{mode}.pbs.status').write_text(
            f'pbs_job_id={job_id}\nscript_exit_code=0\n')
        (run_dir / 'run_manifest.json').write_text(json.dumps(dict(
            mode='aba' if mode.startswith('aba-') else mode, status='completed',
            rejected_samples=0,
            build_manifest_sha256=self.workflow()['build_manifest_sha256'],
            environment=dict(pbs=dict(job_id=job_id)))))
        if mode == 'pilot':
            (run_dir / 'pilot-analysis.md').write_text('validated\n')
        if mode == 'smoke':
            paths = [run_dir / 'smoke_trace_summary.json',
                     run_dir / 'smoke.rank0.csv', run_dir / 'smoke.rank1.csv']
            for path in paths:
                path.write_text('validated trace\n')
            run = json.loads((run_dir / 'run_manifest.json').read_text())
            run['smoke_comparison'] = dict(
                condition=dict(workload='kneighbor', seed=123, algorithm='hwtopolb',
                               n=100000, iterations=100),
                unprofiled_loop_s=1.0, profiled_loop_s=1.2,
                trace_overhead_ratio=1.2,
                trace=dict(summary=str(paths[0]),
                           summary_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                           rank_traces=[dict(path=str(path),
                                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                        for path in paths[1:]]))
            (run_dir / 'run_manifest.json').write_text(json.dumps(run))
        if mode == 'full':
            diag = run_dir / 'diagnostic-8'
            diag.mkdir()
            (diag / 'diagnostic.manifest.json').write_text('{}')

    def test_one_command_waits_for_pbs_and_completes_all_stages(self):
        done = self.root / 'finished'
        done.mkdir()
        errors = []

        def finish_in_order():
            try:
                for mode, job_id in [('pilot', '123.fake'), ('smoke', '124.fake'),
                                     ('full', '125.fake')]:
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline:
                        state = self.workflow() if (self.workflow_root / 'workflow.json').is_file() else {}
                        stage = state.get('stages', {}).get(mode)
                        if stage and stage['status'] == 'submitted':
                            self.finish_job(mode, job_id)
                            (done / f'{job_id}.done').touch()
                            break
                        time.sleep(0.02)
                    else:
                        raise AssertionError(f'{mode} was never submitted')
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=finish_in_order)
        worker.start()
        try:
            result = subprocess.run([*self.command, '--watch', '--poll-interval-seconds', '1'],
                                    cwd=self.root, env=dict(self.env, FAKE_AUTO_DIR=str(done)),
                                    capture_output=True, text=True, timeout=15)
        finally:
            worker.join(timeout=15)
        self.assertEqual(errors, [])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Workflow complete', result.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        self.assertEqual({mode: stage['status'] for mode, stage in
                          self.workflow()['stages'].items()},
                         {'pilot': 'complete', 'smoke': 'complete', 'full': 'complete'})
        self.assertTrue((self.workflow_root / 'full-report.md').is_file())
        repeated = subprocess.run([*self.command, '--watch', '--poll-interval-seconds', '1'],
                                  cwd=self.root, env=dict(self.env, FAKE_AUTO_DIR=str(done)),
                                  capture_output=True, text=True, timeout=5)
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')

    def test_interrupted_wait_resumes_without_resubmitting(self):
        process = subprocess.Popen([*self.command, '--watch', '--poll-interval-seconds', '1'],
                                   cwd=self.root, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                state = self.workflow() if (self.workflow_root / 'workflow.json').is_file() else {}
                if state.get('stages', {}).get('pilot', {}).get('status') == 'submitted':
                    break
                time.sleep(0.02)
            else:
                self.fail('Pilot was not submitted')
            process.send_signal(signal.SIGINT)
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130)
            self.assertIn('rerun the same command', stderr)
            resumed = self.invoke()
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_rerun_never_duplicates_jobs_and_advances_after_verified_completion(self):
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.assertEqual(self.workflow()['stages']['pilot']['job_id'], '123.fake')
        second = self.invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn('is R', second.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.finish_job('pilot', '123.fake')
        terminal = dict(self.env, FAKE_JOB_STATE='F')
        third = self.invoke(env=terminal)
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.out').is_file())
        self.assertTrue((self.workflow_root / 'pilot' / 'pilot-analysis.md').is_file())
        self.assertEqual(self.workflow()['stages']['smoke']['job_id'], '124.fake')
        self.finish_job('smoke', '124.fake')
        fourth = self.invoke(env=terminal)
        self.assertEqual(fourth.returncode, 0, fourth.stderr)
        self.assertEqual(self.workflow()['stages']['full']['job_id'], '125.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        self.finish_job('full', '125.fake')
        fifth = self.invoke(env=terminal)
        self.assertEqual(fifth.returncode, 0, fifth.stderr)
        self.assertTrue((self.workflow_root / 'full-report.md').is_file())
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')
        sixth = self.invoke(env=terminal)
        self.assertEqual(sixth.returncode, 0, sixth.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '3')

    def test_uncertain_qsub_result_cannot_submit_twice(self):
        bad = dict(self.env, FAKE_QSUB_BAD='1')
        first = self.invoke(env=bad)
        self.assertNotEqual(first.returncode, 0)
        self.assertEqual(self.workflow()['stages']['pilot']['status'], 'submission_pending')
        second = self.invoke()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('no duplicate submission', second.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        name = self.workflow()['stages']['pilot']['job_name']
        adopted = self.invoke('--adopt-job-id', '123.fake',
                              env=dict(self.env, FAKE_JOB_NAME=name))
        self.assertEqual(adopted.returncode, 0, adopted.stderr)
        self.assertEqual(self.workflow()['stages']['pilot']['job_id'], '123.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')

    def test_later_stage_uncertain_submission_can_be_adopted(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        uncertain = self.invoke(env=dict(self.env, FAKE_JOB_STATE='F', FAKE_QSUB_BAD='1'))
        self.assertNotEqual(uncertain.returncode, 0)
        self.assertEqual(self.workflow()['stages']['smoke']['status'], 'submission_pending')
        name = self.workflow()['stages']['smoke']['job_name']
        adopted = self.invoke('--adopt-job-id', '124.fake',
                              env=dict(self.env, FAKE_JOB_NAME=name))
        self.assertEqual(adopted.returncode, 0, adopted.stderr)
        self.assertEqual(self.workflow()['stages']['smoke']['job_id'], '124.fake')
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')

    def test_preview_is_read_only_and_needs_no_scheduler(self):
        env = dict(self.env, PATH=os.environ['PATH'])
        result = self.invoke('--preview', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.workflow_root.exists())

    def test_saved_terminal_scheduler_record_can_advance_after_qstat_expiry(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        saved = self.root / 'pilot-qstat.txt'
        saved.write_text('Job Id: 123.fake\n    job_state = F\n    Exit_status = 0\n')
        env = dict(self.env, FAKE_QSTAT_MISSING='1')
        blocked = self.invoke(env=env)
        self.assertNotEqual(blocked.returncode, 0)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        resumed = self.invoke('--scheduler-record', f'pilot={saved}', env=env)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')

    def test_terminal_job_waits_for_staged_output_even_if_qstat_expires(self):
        self.assertEqual(self.invoke().returncode, 0)
        terminal = self.invoke(env=dict(self.env, FAKE_JOB_STATE='F'))
        self.assertEqual(terminal.returncode, 0, terminal.stderr)
        self.assertIn('waiting for output staging', terminal.stdout)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.finish_job('pilot', '123.fake')
        resumed = self.invoke(env=dict(self.env, FAKE_QSTAT_MISSING='1'))
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual((self.root / 'qsub-count').read_text(), '2')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.scheduler-status.txt').is_file())

    def test_failed_pbs_exit_preserves_artifacts_without_submitting_next_stage(self):
        self.assertEqual(self.invoke().returncode, 0)
        self.finish_job('pilot', '123.fake')
        env = dict(self.env, FAKE_JOB_STATE='F', FAKE_JOB_EXIT='1')
        failed = self.invoke(env=env)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')
        self.assertEqual(self.workflow()['stages']['pilot']['status'], 'failed')
        self.assertTrue((self.workflow_root / 'pilot' / 'pbs.out').is_file())

    def test_missing_build_is_created_once_before_first_submission(self):
        manifest = self.root / 'build' / 'cluster' / 'build_manifest.json'
        saved = self.root / 'saved-build-manifest.json'
        saved.write_bytes(manifest.read_bytes())
        manifest.unlink()
        build_counter = self.root / 'build-count'
        (self.root / 'cluster_build.sh').write_text(
            '#!/bin/bash\nset -eu\n'
            f'printf x >> "{build_counter}"\n'
            f'cp "{saved}" "{manifest}"\n')
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.invoke()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(build_counter.read_text(), 'x')
        self.assertEqual((self.root / 'qsub-count').read_text(), '1')


if __name__ == '__main__':
    unittest.main()
