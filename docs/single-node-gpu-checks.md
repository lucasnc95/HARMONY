# Independent GPU checks on compute-0 nodes

The LIMC cluster guide v5.0 lists `compute-0-7`, `compute-0-8`, and
`compute-0-2` as AMD Opteron 6272 machines with two Tesla M2075 GPUs each.
This describes the guide's inventory, not a verified current OpenCL setup.
The PBS job checks the actual allocation and OpenCL device at run time.

From the current Git checkout on the cluster frontend, run one command:

```sh
python3 benchmarks/queue_single_node.py --experiment-id legacy-20261014-a
```

The coordinator builds once if necessary, then queues three independent PBS
jobs. It exits after submission, so the terminal may be closed. Each job
requests one chunk with four CPUs, 8 GB RAM, one GPU, and one MPI slot on its
specific host. No job requires another to finish. The existing two-node A100
variance procedure remains independent and retains its original resource
selection and results.

For each node, the job checks that PBS assigned one slot on the requested
host, measures one real OpenCL GPU without synthetic mode, executes a short
one-rank main-program run, and runs k-neighbor and LeanMD under ERAD and
HWTOPOLB. The benchmark CSVs must contain 20 GPU OpenCL iterations and each
benchmark must report `validation=PASS`. One visible GPU per rank is required;
PBS may choose either physical GPU on the host. The main-program run checks
execution only and does not establish numerical agreement by itself.

All artifacts are under `Results/single-node-EXPERIMENT_ID/NODE/`: PBS output,
script exit status, topology JSON, executable logs, benchmark CSVs, GPU
inventory, and a run manifest. `workflow.json` records the three PBS job IDs
and pinned frontend build. A rerun with the same ID does not queue duplicates.
If `qsub` timed out or its answer was lost, inspect `qstat -xf` for the saved
job name and rerun with `--adopt-job-id NODE=JOBID`. If no job was accepted,
preserve the uncertain record and choose a new experiment ID only after
confirming that the old request is absent from PBS.

Check each job with `qstat -xf JOBID` and inspect its `pbs.status` and
`run_manifest.json`. A failed GPU probe or CPU fallback means that machine
has not passed the GPU test. The current workspace has no cluster PBS access,
so these hardware checks must be run on the cluster. Do not pool their timings
with the two-node A100 variance experiment or claim a speedup from these short
functional checks.
