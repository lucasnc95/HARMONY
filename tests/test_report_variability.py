"""Command-level checks for the final GPU comparison report."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'report_variability.py'
CAPTURE = Path(__file__).resolve().parents[1] / 'benchmarks' / 'capture_diagnostic.py'
WORKLOADS = ('kneighbor', 'leanmd')
SEEDS = (42, 123, 456)
ALGORITHMS = ('erad', 'hwtopolb')


class ReportCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'full'
        self.run.mkdir()
        self.output = self.root / 'report.md'
        self.mapping = [dict(rank=rank, host=f'node{rank}', gpu_count=1,
                             local_index=0, global_index=rank, name='GPU A',
                             platform_name='Vendor', platform_version='OpenCL 3.0',
                             device_version='OpenCL 3.0', driver_version='1',
                             identity_kind='uuid', identity=f'gpu-{rank}',
                             cpu_affinity=dict(value=f'{rank}, {rank + 2}',
                                               unavailable_reason=None))
                        for rank in (0, 1)]
        self.rows = []
        for workload in WORKLOADS:
            for algorithm in ALGORITHMS:
                self.add('warmup', 0, workload, 42, algorithm, None)
        for repetition in range(1, 6):
            seeds = SEEDS[(repetition - 1) % 3:] + SEEDS[:(repetition - 1) % 3]
            algorithms = ALGORITHMS if repetition % 2 else ALGORITHMS[::-1]
            for workload in WORKLOADS:
                for seed in seeds:
                    for algorithm in algorithms:
                        self.add('full', repetition, workload, seed, algorithm,
                                 f'r{repetition}_{workload}_{seed}')
        self.run_data = dict(schema_version=1, mode='full', status='completed',
                             rejected_samples=0, invocations=self.rows,
                             baseline_mapping=self.mapping,
                             build_manifest='/build/manifest.json', build_manifest_sha256='c' * 64,
                             build=dict(source_revision='deadbeef',
                                        binaries={**{name: dict(sha256=name + '-sha') for name in WORKLOADS},
                                                  'test_async_halo_opencl': dict(
                                                      path='/bin/test_async_halo_opencl.out',
                                                      sha256='a' * 64)}),
                             input_parameters=dict(n=1000000, full_iterations=100,
                                                   rebalance_interval=10, perturbation=0.10,
                                                   kneighbor_k=5, leanmd_cutoff=4),
                             environment=dict(pbs=dict(job_id='123.cluster', launcher='/bin/prun',
                                                       slots=['node0', 'node1']),
                                              gpu_clock=dict(value=None, unavailable_reason='no tool')))
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))

    def add(self, phase, repetition, workload, seed, algorithm, pair_id):
        ordinal = len(self.rows) + 1
        prefix = self.run / phase / f'{ordinal:03d}_{workload}_{algorithm}_{seed}'
        prefix.parent.mkdir(exist_ok=True)
        row = dict(phase=phase, repetition=repetition, workload=workload,
                   seed=seed, algorithm=algorithm, pair_id=pair_id, ordinal=ordinal,
                   iterations=100, csv=str(prefix) + '.csv', log=str(prefix) + '.log',
                   inventory=str(prefix) + '.inventory.json',
                   manifest=str(prefix) + '.manifest.json',
                   argv=['/bin/' + workload + '.out', '--n', '1000000',
                         '--iterations', '100', '--balance-interval', '10',
                         '--perturbation', '0.10', '--seed', str(seed),
                         '--balance-algo', algorithm,
                         '--k', '5' if workload == 'kneighbor' else '4',
                         '--require-two-rank-gpu', '--quiet-runtime-metrics',
                         '--output-csv', str(prefix) + '.csv',
                         '--device-inventory-json', str(prefix) + '.inventory.json'])
        self.rows.append(row)
        loop = 1.0 if algorithm == 'erad' else 0.8
        with open(row['csv'], 'w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['iteration', 'step_time_s', 'action', 'rebalanced',
                             'rebalance_gain', 'execution_mode'])
            for iteration in range(100):
                writer.writerow([iteration, '0.005', 'rebalance' if iteration and iteration % 10 == 0 else 'compute',
                                 int(iteration != 0 and iteration % 10 == 0), 0, 'gpu_opencl'])
        Path(row['log']).write_text(f'mode=gpu_opencl loop_time_s={loop} '
                                    f'validation_time_s=0.1 total_time_s={loop + 0.2} '
                                    'validation=PASS\n')
        Path(row['inventory']).write_text(json.dumps(dict(schema_version=1, ranks=self.mapping)))
        Path(row['manifest']).write_text(json.dumps(dict(row, status='accepted',
            rejection_reason=None, exit_code=0,
            build_manifest_sha256='c' * 64, mapping=self.mapping,
            times=dict(loop_time_s=loop, validation_time_s=0.1),
            started_at='2026-10-13T00:00:00Z', completed_at='2026-10-13T00:00:01Z')))

    def invoke(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), '--run-dir', str(self.run),
                               '--output', str(self.output), *extra], capture_output=True, text=True)

    def diagnostic_fixture(self):
        directory = self.run / 'diagnostic'
        directory.mkdir()
        prefix = directory / 'trace'
        fields = ['schema_version', 'rank', 'seq', 'iteration', 'seed', 'algorithm',
                  'record_type', 'phase', 'part', 'device_global', 'owner_rank', 'local_index',
                  'offset', 'count', 'duration_s', 'device_event_s',
                  'boundary_shift_elements_estimate', 'transfer_bytes_measured',
                  'device_name', 'device_kind', 'device_identity']
        for rank in (0, 1):
            rows = []
            def add(iteration, kind, phase, **values):
                row = dict.fromkeys(fields, '')
                row.update(schema_version='1', rank=str(rank), seq=str(len(rows)),
                           iteration=str(iteration), seed='42', algorithm='erad',
                           record_type=kind, phase=phase)
                row.update({key: str(value) for key, value in values.items()})
                rows.append(row)
            add(-1, 'inventory', 'discovered', device_global=rank, owner_rank=rank,
                local_index=0, device_name='GPU A', device_kind='gpu',
                device_identity=f'uuid:gpu-{rank}')
            for iteration in range(100):
                for state in ('old', 'new'):
                    for part in (0, 1):
                        add(iteration, 'partition', state, part=part, device_global=part,
                            owner_rank=part, local_index=0, offset=part * 500000,
                            count=500000)
                if iteration and iteration % 10 == 0:
                    add(iteration, 'phase', 'rebalance_plan', duration_s=0.01)
                    add(iteration, 'phase', 'rebalance_apply', duration_s=0.01)
                add(iteration, 'phase', 'runtime_execute', duration_s=0.004)
                add(iteration, 'device_event', 'kernel', part=rank, device_global=rank,
                    owner_rank=rank, local_index=0, offset=rank * 500000, count=500000,
                    device_event_s=0.003)
                add(iteration, 'phase', 'mpi_step_max', duration_s=0.001)
                add(iteration, 'phase', 'mpi_device_times', duration_s=0.001)
            add(100, 'phase', 'loop_total', duration_s=1.0 if rank == 0 else 0.9)
            for phase in ('gather', 'reference', 'mpi_validation'):
                add(100, 'phase', phase, duration_s=0.01)
            with Path(f'{prefix}.rank{rank}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        sample = self.rows[4]
        diag_csv = directory / 'diagnostic.csv'
        diag_log = directory / 'diagnostic.log'
        diag_inventory = directory / 'diagnostic.inventory.json'
        diag_csv.write_text(Path(sample['csv']).read_text())
        diag_log.write_text('mode=gpu_opencl loop_time_s=1.0 validation_time_s=0.1 '
                            'total_time_s=1.2 validation=PASS\n')
        diag_inventory.write_text(json.dumps(dict(schema_version=1, ranks=self.mapping)))
        command = ['/bin/prun', *sample['argv']]
        for flag, path in (('--output-csv', diag_csv),
                           ('--device-inventory-json', diag_inventory)):
            command[command.index(flag) + 1] = str(path)
        command += ['--diagnostics-prefix', str(prefix)]
        paths = dict(csv=str(diag_csv), log=str(diag_log), inventory=str(diag_inventory),
                     rank0=f'{prefix}.rank0.csv', rank1=f'{prefix}.rank1.csv')
        digest = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for key, path in paths.items()}
        manifest = directory / 'diagnostic.manifest.json'
        manifest.write_text(json.dumps(dict(schema_version=1, status='captured', exit_code=0,
            pbs=dict(job_id='123.cluster', slots=['node0', 'node1'], launcher='/bin/prun'),
            build_manifest_sha256='c' * 64, binary_sha256='kneighbor-sha',
            workload='kneighbor', seed=42, algorithm='erad',
            matched_sample_manifest=sample['manifest'], command=command,
            trace_prefix=str(prefix), artifacts=paths, artifact_sha256=digest,
            started_at='2026-10-13T00:00:01Z', completed_at='2026-10-13T00:00:02Z')))
        return manifest

    def test_complete_matrix_reports_all_adjacent_pairs_without_claiming_cluster_correctness(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('30 adjacent pairs', report)
        self.assertIn('r1_kneighbor_42', report)
        self.assertIn('1.2500', report)
        self.assertIn('1.200000', report)
        self.assertIn('0.800000', report)
        self.assertIn('unavailable: no tool', report)
        self.assertIn('CPU affinity', report)
        self.assertIn('0, 2', report)
        self.assertIn('halo', report.lower())
        self.assertIn('inconclusive', report.lower())

    def test_incomplete_cpu_or_changed_mapping_is_rejected_before_report(self):
        row = self.rows[5]
        csv_path = Path(row['csv'])
        inventory_path = Path(row['inventory'])
        original_csv = csv_path.read_text()
        original_inventory = inventory_path.read_text()
        for altered in (''.join(original_csv.splitlines(keepends=True)[:-1]),
                        original_csv.replace('gpu_opencl', 'cpu_simulation', 1)):
            with self.subTest(kind=altered[-30:]):
                csv_path.write_text(altered)
                self.assertNotEqual(self.invoke().returncode, 0)
                self.assertFalse(self.output.exists())
        csv_path.write_text(original_csv)
        inventory = json.loads(original_inventory)
        inventory['ranks'][0]['identity'] = 'different'
        inventory_path.write_text(json.dumps(inventory))
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_nonzero_exit_or_changed_run_baseline_is_rejected(self):
        sample_path = Path(self.rows[4]['manifest'])
        sample = json.loads(sample_path.read_text())
        sample['exit_code'] = 17
        sample_path.write_text(json.dumps(sample))
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.output.exists())
        sample['exit_code'] = 0
        sample_path.write_text(json.dumps(sample))
        run = dict(self.run_data, baseline_mapping=[dict(self.mapping[0], identity='other'),
                                                    self.mapping[1]])
        (self.run / 'run_manifest.json').write_text(json.dumps(run))
        self.assertNotEqual(self.invoke().returncode, 0)
        self.assertFalse(self.output.exists())

    def test_single_seed_reversal_prevents_general_gain_even_with_halo_pass(self):
        for row in self.rows:
            if row['phase'] == 'full' and row['seed'] == 123 and row['algorithm'] == 'hwtopolb':
                log_path = Path(row['log'])
                log_path.write_text(log_path.read_text().replace('loop_time_s=0.8', 'loop_time_s=1.1')
                                    .replace('total_time_s=1.0', 'total_time_s=1.3'))
                manifest_path = Path(row['manifest'])
                sample = json.loads(manifest_path.read_text())
                sample['times']['loop_time_s'] = 1.1
                manifest_path.write_text(json.dumps(sample))
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
                                        '/bin/test_async_halo_opencl.out'], log=str(halo),
                                        binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('0.9091', report)
        self.assertIn('No useful gain', report)
        self.assertIn('ERAD/HWTOPOLB/ERAD', report)

    def test_existing_report_is_never_overwritten(self):
        self.output.write_text('previous result')
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'previous result')

    def test_unverified_halo_record_cannot_unlock_useful_gain(self):
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='b' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('digest differs', report)
        self.assertIn('inconclusive', report.lower())
        self.assertNotIn('Useful gain supported', report)

    def test_verified_halo_allows_bounded_performance_signal_but_not_causal_claim(self):
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('Paired performance supports a useful gain', report)
        self.assertIn('Report status:** provisional', report)
        self.assertIn('No causal attribution', report)

    def test_diagnostic_manifest_adds_rank_local_accounting_without_changing_pairs(self):
        manifest = self.diagnostic_fixture()
        result = self.invoke('--diagnostic-manifest', str(manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('30 adjacent pairs', report)
        self.assertIn('Rank-local diagnostic accounting', report)
        self.assertIn('0.220000', report)
        self.assertIn('0.220000; 22.0%', report)
        self.assertIn('0.300000', report)
        self.assertIn('Rank 0 determines the diagnostic global loop maximum', report)
        self.assertIn('rank0.csv', report)
        self.assertIn('iteration 0', report)
        self.assertIn('iteration 9', report)
        self.assertIn('iteration 10', report)
        self.assertIn('iteration 11', report)
        self.assertIn('Step time (s)', report)
        self.assertIn('old → new', report)
        self.assertIn('trace#seq=', report)
        self.assertIn('No causal attribution', report)

    def test_diagnostic_capture_preview_requires_same_pbs_allocation_without_running(self):
        import os
        launcher = self.root / 'prun'
        launcher.write_text('#!/bin/sh\nexit 0\n')
        launcher.chmod(0o755)
        binary = self.root / 'kneighbor.out'
        binary.write_text('#!/bin/sh\nexit 0\n')
        binary.chmod(0o755)
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        self.rows[4]['argv'][0] = str(binary)
        sample_path = Path(self.rows[4]['manifest'])
        sample = json.loads(sample_path.read_text())
        sample['argv'][0] = str(binary)
        sample_path.write_text(json.dumps(sample))
        self.run_data['invocations'][4]['argv'][0] = str(binary)
        self.run_data['build']['binaries']['kneighbor'] = dict(path=str(binary), sha256=binary_hash)
        self.run_data['environment']['pbs']['launcher'] = str(launcher)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        nodefile = self.root / 'nodefile'
        nodefile.write_text('node0\nnode1\n')
        env = dict(os.environ, PBS_JOBID='123.cluster', PBS_NODEFILE=str(nodefile))
        command = [sys.executable, str(CAPTURE), '--dry-run', '--run-dir', str(self.run),
                   '--sample-ordinal', '5', '--output-dir', str(self.root / 'diagnostic'),
                   '--sample-timeout-seconds', '7']
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout)
        self.assertIn('--diagnostics-prefix', preview['command'])
        self.assertEqual(preview['sample_timeout_seconds'], 7)
        self.assertEqual(preview['pbs']['job_id'], '123.cluster')
        self.assertFalse((self.root / 'diagnostic').exists())
        env['PBS_JOBID'] = 'other.cluster'
        refused = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse((self.root / 'diagnostic').exists())

    def test_diagnostic_hash_or_condition_mismatch_fails_before_report(self):
        manifest_path = self.diagnostic_fixture()
        original = json.loads(manifest_path.read_text())
        for altered in [dict(original, seed=123),
                        dict(original, artifact_sha256=dict(original['artifact_sha256'],
                            rank0='0' * 64)),
                        dict(original, pbs=dict(original['pbs'], job_id='different.cluster'))]:
            with self.subTest(change=altered):
                manifest_path.write_text(json.dumps(altered))
                result = self.invoke('--diagnostic-manifest', str(manifest_path))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_diagnostic_trace_validator_rejects_missing_rank_without_report(self):
        manifest_path = self.diagnostic_fixture()
        manifest = json.loads(manifest_path.read_text())
        rank1 = Path(manifest['artifacts']['rank1'])
        rank1.unlink()
        result = self.invoke('--diagnostic-manifest', str(manifest_path))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_diagnostic_trace_validator_rejects_corrupt_sequence_with_matching_hash(self):
        manifest_path = self.diagnostic_fixture()
        manifest = json.loads(manifest_path.read_text())
        rank0 = Path(manifest['artifacts']['rank0'])
        with rank0.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[2]['seq'] = rows[1]['seq']
        with rank0.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        manifest['artifact_sha256']['rank0'] = hashlib.sha256(rank0.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        result = self.invoke('--diagnostic-manifest', str(manifest_path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid rank traces', result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejected_manifest_and_nonfinite_loop_are_refused(self):
        row = self.rows[4]
        path = Path(row['manifest'])
        original = json.loads(path.read_text())
        for changed in [dict(original, status='rejected'),
                        dict(original, times=dict(loop_time_s='nan', validation_time_s=0.1))]:
            with self.subTest(changed=changed['status'], times=changed.get('times')):
                path.write_text(json.dumps(changed))
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def test_global_step_maxima_need_not_sum_to_max_rank_loop(self):
        row = self.rows[4]
        csv_path = Path(row['csv'])
        csv_path.write_text(csv_path.read_text().replace('0.005', '0.020'))
        log_path = Path(row['log'])
        log_path.write_text(log_path.read_text().replace('total_time_s=1.2',
                                                       'total_time_s=1.0'))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('-1.000000', report)
        self.assertIn('may each be maxima from different ranks', report)

    def test_early_late_drift_blocks_gain_even_when_iqr_is_small(self):
        ratios = (1.2, 1.2, 1.25, 1.321, 1.321)
        for row in self.rows:
            if row['phase'] != 'full' or row['workload'] != 'kneighbor' or \
                    row['seed'] != 42 or row['algorithm'] != 'hwtopolb':
                continue
            loop = 1 / ratios[row['repetition'] - 1]
            log_path = Path(row['log'])
            log_path.write_text(log_path.read_text().replace('loop_time_s=0.8',
                                                        f'loop_time_s={loop}'))
            manifest_path = Path(row['manifest'])
            sample = json.loads(manifest_path.read_text())
            sample['times']['loop_time_s'] = loop
            manifest_path.write_text(json.dumps(sample))
        halo = self.run / 'halo_reference.log'
        halo.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        self.run_data['halo_reference'] = dict(command=['/bin/prun',
            '/bin/test_async_halo_opencl.out'], log=str(halo),
            binary_sha256='a' * 64, status='passed', exit_code=0)
        (self.run / 'run_manifest.json').write_text(json.dumps(self.run_data))
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        group = next(line for line in report.splitlines()
                     if line.startswith('| kneighbor | 42 |'))
        self.assertIn('| 1.2000 | 1.3210 |', group)
        self.assertTrue(group.endswith('| no |'))
        self.assertIn('inconclusive', report.lower())
        self.assertIn('UTC end ERAD/HWTOPOLB', report)

    def test_isolated_aba_evidence_keeps_order_and_raw_times_outside_pairs(self):
        aba_root = self.root / 'aba'
        (aba_root / 'aba').mkdir(parents=True)
        aba_rows = []
        for ordinal, algorithm in enumerate(('erad', 'hwtopolb', 'erad'), 1):
            source = next(row for row in self.rows if row['phase'] == 'full' and
                          row['workload'] == 'kneighbor' and row['seed'] == 123 and
                          row['algorithm'] == algorithm)
            stem = aba_root / 'aba' / f'{ordinal:03d}_kneighbor_{algorithm}_123'
            row = dict(source, phase='aba', repetition=1, ordinal=ordinal, pair_id=None,
                       csv=f'{stem}.csv', log=f'{stem}.log',
                       inventory=f'{stem}.inventory.json', manifest=f'{stem}.manifest.json')
            row['argv'] = list(source['argv'])
            for flag, key in (('--output-csv', 'csv'), ('--device-inventory-json', 'inventory')):
                row['argv'][row['argv'].index(flag) + 1] = row[key]
            for key in ('csv', 'log', 'inventory'):
                Path(row[key]).write_bytes(Path(source[key]).read_bytes())
            sample = json.loads(Path(source['manifest']).read_text())
            sample.update(row)
            Path(row['manifest']).write_text(json.dumps(sample))
            aba_rows.append(row)
        halo_log = aba_root / 'halo_reference.log'
        halo_log.write_text('PASS: asynchronous OpenCL halo matches two-step CPU reference\n')
        aba = dict(self.run_data, mode='aba', invocations=aba_rows,
                   environment=dict(self.run_data['environment'],
                                    pbs=dict(self.run_data['environment']['pbs'], job_id='456.cluster')),
                   halo_reference=dict(command=['/bin/prun', '/bin/test_async_halo_opencl.out'],
                                       log=str(halo_log), binary_sha256='a' * 64,
                                       status='passed', exit_code=0))
        (aba_root / 'run_manifest.json').write_text(json.dumps(aba))
        result = self.invoke('--aba-run-dir', str(aba_root))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.output.read_text()
        self.assertIn('ERAD/HWTOPOLB/ERAD', report)
        self.assertIn('456.cluster', report)
        self.assertIn('30 adjacent pairs', report)
        self.assertIn('1.000000', report)
        self.assertIn('0.800000', report)
        self.output.unlink()
        Path(aba_rows[1]['csv']).unlink()
        self.assertNotEqual(self.invoke('--aba-run-dir', str(aba_root)).returncode, 0)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
