import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'run_variability.py'


class RunnerCLITests(unittest.TestCase):
    def preview(self, mode='full'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'new-run'
            process = subprocess.run([sys.executable, str(SCRIPT), '--dry-run',
                                      '--mode', mode, '--run-dir', str(root)],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(root.exists())
            return json.loads(process.stdout)

    def test_full_plan_has_adjacent_rotated_pairs_and_unique_artifacts(self):
        plan = self.preview()
        samples = plan['invocations']
        warmups = [row for row in samples if row['phase'] == 'warmup']
        timed = [row for row in samples if row['phase'] == 'full']
        self.assertEqual(len(warmups), 4)
        self.assertEqual(len(timed), 60)
        self.assertEqual(len({row['pair_id'] for row in timed}), 30)
        for left, right in zip(timed[::2], timed[1::2]):
            self.assertEqual(left['pair_id'], right['pair_id'])
            self.assertEqual({left['algorithm'], right['algorithm']}, {'erad', 'hwtopolb'})
        self.assertEqual([row['seed'] for row in timed[:6:2]], [42, 123, 456])
        self.assertEqual([row['seed'] for row in timed[12:18:2]], [123, 456, 42])
        self.assertEqual(timed[0]['algorithm'], 'erad')
        self.assertEqual(timed[12]['algorithm'], 'hwtopolb')
        self.assertEqual(len({row['csv'] for row in samples}), 64)
        self.assertTrue(all('--require-two-rank-gpu' in row['argv'] for row in samples))
        self.assertTrue(all(row['argv'][row['argv'].index('--n') + 1] == '1000000' for row in timed))

    def test_pilot_is_short_and_separate(self):
        plan = self.preview('pilot')
        samples = plan['invocations']
        self.assertEqual(len(samples), 24)
        self.assertTrue(all(row['phase'] == 'pilot' for row in samples))
        self.assertTrue(all(row['iterations'] == 8 for row in samples))
        self.assertTrue(all('/pilot/' in row['csv'] for row in samples))
        self.assertEqual(len({(row['workload'], row['algorithm'], row['seed']) for row in samples}), 12)
        self.assertEqual(samples[0]['algorithm'], 'erad')
        self.assertEqual(samples[12]['algorithm'], 'hwtopolb')
        self.assertEqual([row['seed'] for row in samples[:6:2]], [42, 123, 456])
        self.assertEqual([row['seed'] for row in samples[12:18:2]], [123, 456, 42])

    def test_aba_preview_is_isolated_seed_123_same_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'aba'
            process = subprocess.run([sys.executable, str(SCRIPT), '--dry-run',
                                      '--mode', 'aba', '--aba-workload', 'kneighbor',
                                      '--run-dir', str(root)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(root.exists())
            samples = json.loads(process.stdout)['invocations']
            self.assertEqual([row['algorithm'] for row in samples],
                             ['erad', 'hwtopolb', 'erad'])
            self.assertEqual([row['seed'] for row in samples], [123] * 3)
            self.assertEqual([row['workload'] for row in samples], ['kneighbor'] * 3)
            self.assertTrue(all('/aba/' in row['csv'] for row in samples))

    def test_smoke_preview_pairs_one_unprofiled_and_one_traced_gpu_condition(self):
        samples = self.preview('smoke')['invocations']
        self.assertEqual(len(samples), 2)
        self.assertEqual([(row['workload'], row['seed'], row['algorithm'], row['n'])
                          for row in samples], [('kneighbor', 123, 'hwtopolb', 100000)] * 2)
        self.assertEqual([row['iterations'] for row in samples], [100, 100])
        self.assertIsNone(samples[0]['diagnostics_prefix'])
        self.assertNotIn('--diagnostics-prefix', samples[0]['argv'])
        self.assertIn('--diagnostics-prefix', samples[1]['argv'])
        self.assertEqual(len({row['csv'] for row in samples}), 2)


if __name__ == '__main__':
    unittest.main()
