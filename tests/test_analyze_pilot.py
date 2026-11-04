"""Command-level checks for accepted pre-rebalance pilot analysis."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

COMMAND = Path(__file__).resolve().parents[1] / 'benchmarks' / 'analyze_pilot.py'
SEEDS = (42, 123, 456)
ALGORITHMS = ('erad', 'hwtopolb')
WORKLOADS = ('kneighbor', 'leanmd')


class PilotAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'pilot'
        self.run.mkdir()
        self.rows = []
        ordinal = 0
        for repetition in (1, 2):
            seeds = SEEDS[repetition - 1:] + SEEDS[:repetition - 1]
            algorithms = ALGORITHMS if repetition == 1 else ALGORITHMS[::-1]
            for workload in WORKLOADS:
                for seed in seeds:
                    for algorithm in algorithms:
                        ordinal += 1
                        prefix = self.run / f'{ordinal:03d}_{workload}_{algorithm}_{seed}'
                        mapping = [dict(rank=rank, host=f'node{rank}', gpu_count=1, local_index=0,
                                        global_index=rank, name='GPU A', platform_name='Vendor',
                                        platform_version='OpenCL 3.0', device_version='OpenCL 3.0',
                                        driver_version='1', identity_kind='uuid', identity=f'gpu-{rank}',
                                        cpu_affinity=dict(value=str(rank), unavailable_reason=None))
                                   for rank in (0, 1)]
                        row = dict(phase='pilot', ordinal=ordinal, repetition=repetition,
                                   workload=workload, seed=seed, algorithm=algorithm,
                                   pair_id=f'r{repetition}_{workload}_{seed}', iterations=8,
                                   csv=str(prefix) + '.csv', log=str(prefix) + '.log',
                                   inventory=str(prefix) + '.inventory.json',
                                   manifest=str(prefix) + '.manifest.json', argv=[])
                        self.rows.append(row)
                        with open(row['csv'], 'w', newline='') as stream:
                            writer = csv.writer(stream)
                            writer.writerow(['iteration', 'step_time_s', 'action', 'rebalanced',
                                             'rebalance_gain', 'execution_mode'])
                            for iteration in range(8):
                                writer.writerow([iteration, '0.020' if seed == 123 else '0.0004',
                                                 'compute', 0, 0, 'gpu_opencl'])
                        Path(row['log']).write_text(f'[{workload}] algo={algorithm} mode=gpu_opencl '
                                                    'loop_time_s=0.16 validation_time_s=0.01 '
                                                    'total_time_s=0.17 validation=PASS\n')
                        Path(row['inventory']).write_text(json.dumps(dict(schema_version=1, ranks=mapping)))
                        Path(row['manifest']).write_text(json.dumps(dict(row, status='accepted',
                            rejection_reason=None, exit_code=0, build_manifest_sha256='abc',
                            times=dict(loop_time_s=0.16, validation_time_s=0.01),
                            mapping=mapping)))
        (self.run / 'run_manifest.json').write_text(json.dumps(dict(schema_version=1, mode='pilot',
            status='completed', invocations=self.rows,
            environment=dict(pbs=dict(job_id='123.cluster', slots=['node0', 'node1'])),
            build_manifest='/tmp/build_manifest.json', build_manifest_sha256='abc',
            baseline_mapping=mapping)))

    def invoke(self):
        return subprocess.run([sys.executable, str(COMMAND), '--run-dir', str(self.run),
                               '--output', str(self.root / 'report.md')], text=True, capture_output=True)

    def test_complete_rotated_pilot_reports_slow_onset_without_causal_claim(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = (self.root / 'report.md').read_text()
        self.assertIn('24 accepted', report)
        self.assertIn('iteration 0', report)
        self.assertIn('seed 123', report)
        self.assertIn('inconclusive', report.lower())
        self.assertIn('excluded from final', report.lower())

    def test_incomplete_or_cpu_sample_is_rejected_without_report(self):
        sample = self.rows[0]
        csv_path = Path(sample['csv'])
        original = csv_path.read_text()
        for altered in ['\n'.join(original.splitlines()[:-1]) + '\n',
                        original.replace('gpu_opencl', 'cpu_simulation', 1)]:
            with self.subTest(altered=altered[-30:]):
                csv_path.write_text(altered)
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'report.md').exists())
        csv_path.write_text(original)

    def test_rejected_manifest_or_changed_pair_mapping_is_rejected(self):
        sample = self.rows[0]
        path = Path(sample['manifest'])
        original = json.loads(path.read_text())
        for update in [dict(status='rejected'),
                       dict(mapping=[dict(original['mapping'][0], identity='different'), original['mapping'][1]])]:
            with self.subTest(update=update):
                path.write_text(json.dumps(dict(original, **update)))
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'report.md').exists())
        path.write_text(json.dumps(original))

    def test_rebalance_or_full_run_is_rejected(self):
        sample = self.rows[0]
        path = Path(sample['csv'])
        original = path.read_text()
        path.write_text(original.replace(',compute,0,', ',rebalance,1,', 1))
        self.assertNotEqual(self.invoke().returncode, 0)
        path.write_text(original)
        run_path = self.run / 'run_manifest.json'
        run = json.loads(run_path.read_text())
        run['mode'] = 'full'
        run_path.write_text(json.dumps(run))
        self.assertNotEqual(self.invoke().returncode, 0)

    def test_invalid_gpu_count_or_duplicate_uuid_is_rejected(self):
        sample = self.rows[0]
        inventory_path = Path(sample['inventory'])
        original = json.loads(inventory_path.read_text())
        for ranks in [
            [dict(original['ranks'][0], gpu_count=2), original['ranks'][1]],
            [original['ranks'][0], dict(original['ranks'][1], identity=original['ranks'][0]['identity'])],
        ]:
            with self.subTest(ranks=ranks):
                inventory_path.write_text(json.dumps(dict(original, ranks=ranks)))
                result = self.invoke()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'report.md').exists())

    def test_run_baseline_mapping_must_match_rank_affinity(self):
        path = self.run / 'run_manifest.json'
        run = json.loads(path.read_text())
        run['baseline_mapping'][0]['cpu_affinity']['value'] = 'other-cpu'
        path.write_text(json.dumps(run))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Run baseline mapping differs', result.stderr)
        self.assertFalse((self.root / 'report.md').exists())

    def test_accepted_sample_requires_successful_launcher_exit(self):
        path = Path(self.rows[0]['manifest'])
        sample = json.loads(path.read_text())
        sample['exit_code'] = 7
        path.write_text(json.dumps(sample))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('launcher exit', result.stderr)
        self.assertFalse((self.root / 'report.md').exists())

    def test_malformed_sample_timing_is_rejected_without_traceback(self):
        path = Path(self.rows[0]['manifest'])
        sample = json.loads(path.read_text())
        sample['times']['loop_time_s'] = 'invalid'
        path.write_text(json.dumps(sample))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('manifest loop_time_s', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
