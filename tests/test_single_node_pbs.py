"""Check independent PBS queueing and one-node allocation admission."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import queue_single_node  # noqa: E402
import run_single_node  # noqa: E402


class SingleNodePbsTests(unittest.TestCase):
    def test_queues_each_node_once_with_its_own_gpu_chunk(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(experiment_id='test-a', results_root=Path(temporary),
                                      preview=False, adopt_job_id=None)
            calls = []

            def fake_qsub(command, **_kwargs):
                calls.append(command)
                return mock.Mock(returncode=0, stdout=f'{100 + len(calls)}.pbs\n', stderr='')

            with mock.patch.object(queue_single_node, 'ensure_build'), \
                 mock.patch.object(queue_single_node.shutil, 'which', return_value='/bin/qsub'), \
                 mock.patch.object(queue_single_node.subprocess, 'run', side_effect=fake_qsub):
                queue_single_node.queue(args)
                queue_single_node.queue(args)

            self.assertEqual(len(calls), 3)
            for command, node in zip(calls, queue_single_node.NODES):
                self.assertEqual(command[command.index('-l') + 1],
                                 queue_single_node.SELECT.format(node=node))
            state = Path(temporary) / 'single-node-test-a' / 'workflow.json'
            self.assertEqual(len(json.loads(state.read_text())['stages']), 3)

    def test_uncertain_submission_never_requeues_same_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(experiment_id='test-b', results_root=Path(temporary),
                                      preview=False, adopt_job_id=None)
            with mock.patch.object(queue_single_node, 'ensure_build'), \
                 mock.patch.object(queue_single_node.shutil, 'which', return_value='/bin/qsub'), \
                 mock.patch.object(queue_single_node.subprocess, 'run',
                                   return_value=mock.Mock(returncode=1, stdout='', stderr='lost')) as submit:
                with self.assertRaises(queue_single_node.WorkflowError):
                    queue_single_node.queue(args)
                with self.assertRaises(queue_single_node.WorkflowError):
                    queue_single_node.queue(args)
            self.assertEqual(submit.call_count, 1)
            state = Path(temporary) / 'single-node-test-b' / 'workflow.json'
            self.assertEqual(json.loads(state.read_text())['stages']['compute-0-7']['status'],
                             'submission_pending')

    def test_rejects_wrong_or_multiple_pbs_hosts(self):
        with tempfile.TemporaryDirectory() as temporary:
            nodefile = Path(temporary) / 'nodes'
            nodefile.write_text('compute-0-7.example.org\n')
            with mock.patch.dict('os.environ', {'PBS_JOBID': '123.pbs',
                                                'PBS_NODEFILE': str(nodefile)}):
                self.assertEqual(run_single_node.validate_allocation('compute-0-7')['slots'],
                                 ['compute-0-7.example.org'])
                with self.assertRaises(ValueError):
                    run_single_node.validate_allocation('compute-0-8')
                nodefile.write_text('compute-0-7\ncompute-0-7\n')
                with self.assertRaises(ValueError):
                    run_single_node.validate_allocation('compute-0-7')


if __name__ == '__main__':
    unittest.main()
