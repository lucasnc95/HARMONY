"""The traced diagnostic obeys the same launcher bound as matrix samples."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from capture_diagnostic import capture  # noqa: E402


class CaptureDiagnosticTests(unittest.TestCase):
    def test_completed_diagnostic_records_all_five_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / 'diagnostic'
            paths = {key: str(output_dir / f'{key}.txt')
                     for key in ('log', 'csv', 'inventory', 'rank0', 'rank1')}
            command = [sys.executable, '-c',
                       "from pathlib import Path; import sys; "
                       "[Path(path).write_text('sample') for path in sys.argv[1:]]",
                       *(value for key, value in paths.items() if key != 'log')]
            manifest = dict(command=command, artifacts=paths, artifact_sha256={},
                            sample_timeout_seconds=1, exit_code=None)
            self.assertTrue(capture(output_dir, manifest))
            saved = json.loads((output_dir / 'diagnostic.manifest.json').read_text())
            self.assertEqual(saved['status'], 'captured')
            self.assertEqual(saved['exit_code'], 0)
            self.assertEqual(set(saved['artifact_sha256']), set(paths))

    def test_hung_diagnostic_is_rejected_and_preserves_partial_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / 'diagnostic'
            log = output_dir / 'diagnostic.log'
            manifest = dict(
                command=[sys.executable, '-c',
                         "import time; print('started', flush=True); time.sleep(30)"],
                artifacts=dict(log=str(log)), artifact_sha256={},
                sample_timeout_seconds=1, exit_code=None)
            self.assertFalse(capture(output_dir, manifest))
            saved = json.loads((output_dir / 'diagnostic.manifest.json').read_text())
            self.assertEqual(saved['status'], 'rejected')
            self.assertIn('timed out', saved['rejection_reason'])
            self.assertIn('started', log.read_text())


if __name__ == '__main__':
    unittest.main()
