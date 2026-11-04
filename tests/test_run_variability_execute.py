"""Command-boundary execution gate tests with synthetic benchmark artifacts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'run_variability.py'
FAKE_PRUN = '''#!/usr/bin/env python3
import csv, json, os, sys, time
from pathlib import Path
if '--version' in sys.argv:
    print('fake-prun 1.0')
    sys.exit(0)
if sys.argv[1].endswith('test_async_halo_opencl.out'):
    if os.environ.get('FAKE_HALO_FAIL'):
        print('SKIP: asynchronous halo test needs two OpenCL GPU partitions')
        sys.exit(77)
    if os.environ.get('FAKE_HALO_FALSE_PASS'):
        print('NOT PASS: asynchronous OpenCL halo matches two-step CPU reference')
        sys.exit(0)
    print('PASS: asynchronous OpenCL halo matches two-step CPU reference')
    sys.exit(0)
def value(name):
    return sys.argv[sys.argv.index(name) + 1]
output = Path(value('--output-csv'))
if os.environ.get('FAKE_SAMPLE_HANG'):
    print('partial output before timeout', flush=True)
    time.sleep(30)
count = int(value('--iterations'))
mode = 'cpu_simulation' if os.environ.get('FAKE_CPU') else 'gpu_opencl'
with output.open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['iteration', 'step_time_s', 'action', 'rebalanced',
                     'rebalance_gain', 'execution_mode'])
    for step in range(count):
        rebalance = step > 0 and step % 10 == 0
        writer.writerow([step, '0.1', 'rebalance' if rebalance else 'compute',
                         int(rebalance), 0, mode])
identity = 'changed' if os.environ.get('FAKE_MAPPING') and 'hwtopolb' in output.name else 'gpu'
ranks = [dict(rank=rank, host=f'node{rank}', gpu_count=1, local_index=0, global_index=rank,
              name='fake GPU', platform_name='fake', platform_version='3.0',
              device_version='3.0', driver_version='1', identity_kind='uuid',
              identity=identity if os.environ.get('FAKE_DUPLICATE_UUID') else f'{identity}-{rank}')
         for rank in (0, 1)]
if os.environ.get('FAKE_WRONG_GPU_COUNT'):
    ranks[1]['gpu_count'] = 2
Path(value('--device-inventory-json')).write_text(json.dumps({'schema_version': 1, 'ranks': ranks}))
if os.environ.get('FAKE_WRITE_TRACE') and '--diagnostics-prefix' in sys.argv:
    columns = ['schema_version', 'rank', 'seq', 'iteration', 'seed', 'algorithm',
               'record_type', 'phase', 'part', 'device_global', 'owner_rank',
               'local_index', 'offset', 'count', 'duration_s', 'device_event_s',
               'boundary_shift_elements_estimate', 'transfer_bytes_measured',
               'device_name', 'device_kind', 'device_identity']
    for rank in (0, 1):
        path = Path(value('--diagnostics-prefix') + f'.rank{rank}.csv')
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            seq = 0
            def emit(iteration, record_type, phase='', **fields):
                global seq
                writer.writerow(dict(schema_version=1, rank=rank, seq=seq,
                                     iteration=iteration, seed=value('--seed'),
                                     algorithm=value('--balance-algo'),
                                     record_type=record_type, phase=phase, **fields))
                seq += 1
            emit(-1, 'inventory', device_global=rank, owner_rank=rank,
                 local_index=0, device_name='fake GPU', device_kind='gpu',
                 device_identity=f'uuid:gpu-{rank}')
            for iteration in range(count):
                for phase in ('old', 'new'):
                    for part in (0, 1):
                        emit(iteration, 'partition', phase, part=part,
                             device_global=part, owner_rank=part, local_index=0,
                             offset=part * 50000, count=50000)
                if iteration > 0 and iteration % 10 == 0:
                    emit(iteration, 'phase', 'rebalance_plan', duration_s=0.001)
                    emit(iteration, 'phase', 'rebalance_apply', duration_s=0.001)
                emit(iteration, 'phase', 'runtime_execute', duration_s=0.01)
                emit(iteration, 'device_event', part=rank, device_event_s=0.005)
                emit(iteration, 'phase', 'mpi_step_max', duration_s=0.01)
                emit(iteration, 'phase', 'mpi_device_times', duration_s=0.001)
            for phase in ('loop_total', 'gather', 'reference', 'mpi_validation'):
                emit(count, 'phase', phase, duration_s=1.0 if phase == 'loop_total' else 0.01)
summary_mode = 'cpu_simulation' if os.environ.get('FAKE_SUMMARY_CPU') else mode
print(f'mode={summary_mode} validation=PASS loop_time_s=1.0 '
      'validation_time_s=0.1 total_time_s=1.1')
if os.environ.get('FAKE_CONFLICTING_VALIDATION'):
    print('validation=FAIL loop_time_s=2.0')
'''


class ExecuteGates(unittest.TestCase):
    def fixture(self, root):
        binaries = root / 'bin'
        binaries.mkdir()
        for workload in ('kneighbor', 'leanmd'):
            binary = binaries / (workload + '.out')
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
        halo = binaries / 'test_async_halo_opencl.out'
        halo.write_text('#!/bin/sh\nexit 0\n')
        halo.chmod(0o755)
        manifest = binaries / 'build_manifest.json'
        manifest.write_text(json.dumps({'binaries': {
            workload: {'path': str(binaries / (workload + '.out')),
                       'sha256': hashlib.sha256((binaries / (workload + '.out')).read_bytes()).hexdigest()}
            for workload in ('kneighbor', 'leanmd', 'test_async_halo_opencl')}}))
        launcher = root / 'prun'
        launcher.write_text(FAKE_PRUN)
        launcher.chmod(0o755)
        nodefile = root / 'nodefile'
        nodefile.write_text('node0\nnode1\n')
        env = dict(os.environ, PBS_JOBID='fake.1', PBS_NODEFILE=str(nodefile),
                   PATH=str(root) + os.pathsep + os.environ['PATH'])
        command = [sys.executable, str(SCRIPT), '--execute', '--mode', 'pilot',
                   '--run-dir', str(root / 'run'), '--bin-dir', str(binaries),
                   '--build-manifest', str(manifest)]
        return command, env, binaries, nodefile

    def test_wrong_allocation_and_binary_digest_refused_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, binaries, nodefile = self.fixture(root)
            nodefile.write_text('node0\n')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('two PBS slots', process.stderr)
            self.assertFalse((root / 'run').exists())
            nodefile.write_text('node0\nnode1\n')
            (binaries / 'leanmd.out').write_text('#!/bin/sh\nexit 2\n')
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('SHA-256 mismatch', process.stderr)
            self.assertFalse((root / 'run').exists())

    def test_cpu_fallback_and_pair_mapping_drift_reject_samples(self):
        for flag in ('FAKE_CPU', 'FAKE_MAPPING'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command, env, _, _ = self.fixture(root)
                env[flag] = '1'
                process = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
                self.assertEqual(len(samples), 24)
                self.assertTrue(any(row['status'] == 'rejected' for row in samples))
                if flag == 'FAKE_MAPPING':
                    first = [row for row in samples if row['pair_id'] == 'r1_kneighbor_42']
                    self.assertEqual({row['status'] for row in first}, {'rejected'})

    def test_conflicting_validation_summary_rejects_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            env['FAKE_CONFLICTING_VALIDATION'] = '1'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
            self.assertTrue(all(row['status'] == 'rejected' for row in samples))
            self.assertTrue(any('Numerical validation did not pass' in row['rejection_reason']
                                for row in samples))

    def test_cpu_summary_conflicting_with_gpu_csv_rejects_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            env['FAKE_SUMMARY_CPU'] = '1'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
            self.assertTrue(all(row['status'] == 'rejected' for row in samples))
            self.assertTrue(any('GPU execution mode' in row['rejection_reason'] for row in samples))

    def test_duplicate_uuid_or_wrong_gpu_count_rejects_sample(self):
        for flag, message in (('FAKE_DUPLICATE_UUID', 'same physical GPU'),
                              ('FAKE_WRONG_GPU_COUNT', 'exactly one GPU per rank')):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                command, env, _, _ = self.fixture(root)
                env[flag] = '1'
                process = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
                self.assertTrue(all(row['status'] == 'rejected' for row in samples))
                self.assertTrue(any(message in row['rejection_reason'] for row in samples))

    def test_valid_synthetic_pilot_accepts_all_24_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(run['rejected_samples'], 0)
            self.assertEqual([row['rank'] for row in run['baseline_mapping']], [0, 1])
            samples = [json.loads(path.read_text()) for path in (root / 'run').rglob('*.manifest.json')]
            self.assertTrue(all(row['exit_code'] == 0 for row in samples))
            self.assertEqual(len(list((root / 'run').rglob('*.manifest.json'))), 24)
            analysis = subprocess.run(
                [sys.executable, str(SCRIPT.with_name('analyze_pilot.py')),
                 '--run-dir', str(root / 'run'), '--output', str(root / 'pilot-report.md')],
                capture_output=True, text=True)
            self.assertEqual(analysis.returncode, 0, analysis.stderr)
            self.assertIn('24 accepted', (root / 'pilot-report.md').read_text())
            second = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('File exists', second.stderr)

    def test_sample_timeout_rejects_run_and_preserves_partial_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            env['FAKE_SAMPLE_HANG'] = '1'
            command += ['--sample-timeout-seconds', '1']
            process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=12)
            self.assertNotEqual(process.returncode, 0)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['status'], 'rejected')
            samples = list((root / 'run').rglob('*.manifest.json'))
            self.assertEqual(len(samples), 1)
            sample = json.loads(samples[0].read_text())
            self.assertEqual(sample['status'], 'rejected')
            self.assertIn('timed out', sample['rejection_reason'])
            self.assertIn('partial output before timeout', Path(sample['log']).read_text())

    def test_aba_executes_three_ordered_samples_after_halo_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'aba'
            command += ['--aba-workload', 'kneighbor']
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['mode'], 'aba')
            self.assertEqual(run['halo_reference']['status'], 'passed')
            self.assertEqual([row['algorithm'] for row in run['invocations']],
                             ['erad', 'hwtopolb', 'erad'])
            self.assertEqual(len(list((root / 'run').rglob('*.manifest.json'))), 3)

    def test_full_run_stops_before_samples_when_halo_reference_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'full'
            env['FAKE_HALO_FAIL'] = '1'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['status'], 'rejected')
            self.assertEqual(run['halo_reference']['status'], 'rejected')
            self.assertIn('SKIP:', (root / 'run' / 'halo_reference.log').read_text())
            self.assertEqual(list((root / 'run').rglob('*.manifest.json')), [])

    def test_full_run_rejects_non_exact_halo_pass_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'full'
            env['FAKE_HALO_FALSE_PASS'] = '1'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['halo_reference']['status'], 'rejected')
            self.assertEqual(list((root / 'run').rglob('*.manifest.json')), [])

    def test_full_run_accepts_matrix_only_after_halo_reference_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'full'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['halo_reference']['status'], 'passed')
            self.assertEqual(run['halo_reference']['exit_code'], 0)
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(len(list((root / 'run').rglob('*.manifest.json'))), 64)
            report = subprocess.run(
                [sys.executable, str(SCRIPT.with_name('report_variability.py')),
                 '--run-dir', str(root / 'run'), '--output', str(root / 'full-report.md')],
                capture_output=True, text=True)
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertIn('Validated 30 adjacent pairs',
                          (root / 'full-report.md').read_text())

    def test_smoke_rejects_missing_rank_traces_after_gpu_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'smoke'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['mode'], 'smoke')
            self.assertEqual(run['status'], 'rejected')
            self.assertEqual(run['halo_reference']['status'], 'passed')
            samples = [json.loads(Path(row['manifest']).read_text())
                       for row in run['invocations']]
            self.assertEqual([row['status'] for row in samples], ['accepted', 'rejected'])
            self.assertIn('rank traces failed validation', samples[1]['rejection_reason'])
            self.assertNotIn('smoke_comparison', run)

    def test_smoke_accepts_two_rank_traces_and_records_overhead(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command, env, _, _ = self.fixture(root)
            command[command.index('pilot')] = 'smoke'
            env['FAKE_WRITE_TRACE'] = '1'
            process = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            run = json.loads((root / 'run' / 'run_manifest.json').read_text())
            self.assertEqual(run['status'], 'completed')
            self.assertEqual(run['rejected_samples'], 0)
            comparison = run['smoke_comparison']
            self.assertEqual(comparison['trace_overhead_ratio'], 1.0)
            self.assertEqual(len(comparison['trace']['rank_traces']), 2)
            self.assertTrue((root / 'run' / 'smoke_trace_summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
