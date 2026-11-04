# Two-GPU variance investigation: pending cluster validation

## Current evidence and correctness

The historical run is archived read-only under `Results/2026-10-13-historical-gpu-comparison/`; file hashes and the recovered PBS ID `183164.frontend` are in `benchmarks/historical_archive_manifest.json`. The archived output and reconstruction are summarized in `benchmarks/historical_gpu_evidence.md`. All 12 historical condition files report `gpu_opencl` and contain 100 ordered iterations. The earlier two-rank asynchronous OpenCL halo reference was reported as passing in the investigation plan, but the original run metadata lacks a verifiable per-rank physical GPU fingerprint, compiler, revision, and complete environment record. Those missing facts remain unknown.

The k-neighbor tolerance is absolute error at most `1e-5`. The LeanMD tolerance is `1e-4 + 1e-4 * abs(reference)`. Both require finite output and reference values. A fresh two-rank halo reference result must be captured with the controlled full run before drawing a performance conclusion. No local CPU fallback validates the OpenCL kernels or physical GPU mapping.

## Historical performance finding

For both workloads, every ERAD run and the HWTOPOLB seed-123 run enters a roughly 20 ms ordinary-step path at iteration 0. The first scheduled rebalance is at iteration 10, so later partition movement cannot explain the **onset**. The archived full-wall time has an unexplained remainder of roughly one second in each slow run after recorded step times are summed. The old logs do not isolate that host interval. OpenCL event times are a separate, potentially overlapping device timeline. These are single historical samples; no useful-gain or causal claim follows from them.

The evaluated HWTOPOLB variant uses throughput and seeded perturbation. It does **not** consume the measured topology JSON. Any topology-aware performance claim requires a separate implementation and experiment.

## Controlled report command

The resumable command in `docs/controlled-gpu-variance-run.md` queues the
pilot, diagnostic smoke, full matrix, and finalizer with PBS dependencies. The
finalizer writes
`Results/controlled-EXPERIMENT_ID/full-report.md` after validating the PBS
records and the full run. Repeating the command does not replace that report.
For an independent read-only revalidation from the preserved full-run directory:

```sh
python3 benchmarks/report_variability.py \
  --run-dir "$PWD/Results/controlled-EXPERIMENT_ID/full" \
  --output "$PWD/Results/controlled-EXPERIMENT_ID/independent-report.md" \
  --diagnostic-manifest "$PWD/Results/controlled-EXPERIMENT_ID/full/diagnostic-8/diagnostic.manifest.json"
```

The command reads the `run_manifest.json`, four warm-ups, 60 full samples, and the full-run `halo_reference` record. It validates each sample's successful launcher exit, CSV, log, inventory, build digest, strict GPU mapping against the run baseline, reference result, order, and unique paths before reporting the 30 adjacent pairs. It calculates each ERAD/HWTOPOLB loop-time ratio within its pair and reports median, min/max, and Tukey-hinge quartiles for each workload and seed. The raw paired table keeps the primary complete-loop time, secondary full-wall time, post-loop validation time, and descriptive `loop minus sum of step maxima` gap separate. MPI maxima may come from different ranks, so that gap is not an additive phase residual and may be negative. The command refuses to overwrite an existing output file. The pilot and traced diagnostic runs never enter the headline matrix.

A useful gain requires every workload/seed median paired ratio to reach `1/0.9`, with stable repetitions. The predeclared stability check requires `(Q3-Q1)/median <= 0.10`, `Q1 >= 1/0.9`, and no more than 10% early-versus-late paired-ratio median drift (first two versus last two pairs in run order). The report also shows the raw loop-time range for each algorithm. A near-threshold or unstable group calls for more rotated samples or an inconclusive conclusion. A seed-level reversal blocks a general gain claim. If seed 123 remains anomalous, run a separate same-allocation ERAD/HWTOPOLB/ERAD diagnostic sequence; preserve its exact order and raw artifacts outside the paired estimate.

The coordinator defaults to diagnostic ordinal 8 and passes that selection to
the PBS wrapper. The wrapper runs `capture_diagnostic.py` after the unprofiled
matrix on the **same allocation** and writes a new `diagnostic-ORDINAL/`
directory. The capture records the exact command, PBS job, build and binary
hashes, raw artifact hashes, trace paths, and matched unprofiled sample. It
marks the manifest `captured` after exit zero and all five files exist; the
report independently validates the raw GPU outputs and traces. The capture
never replaces a headline sample. Use `--diagnostic-ordinal` on the
coordinator to change the condition before the first submission.

When `--diagnostic-manifest` is supplied, the report independently validates both raw rank trace files with `validate_trace.py`, checks their GPU identity and diagnostic log against the same-allocation full run, and reports each rank's `loop_total` minus its nonoverlapping host phases. OpenCL event time remains a separate overlapping device timeline. The report cites rank, trace sequence, step time, old/new partition offsets and counts, and host/device timing for iteration 0 and iterations adjacent to every scheduled rebalance boundary. It does not infer a cause from one diagnostic invocation. A matching fast/control trace and the isolated A/B/A follow-up are still needed if seed 123 reverses. Without a diagnostic manifest, the report remains explicitly provisional. If measured rank-local host phases fail to reconcile with that rank's loop, state the missing measurement and do not attribute the remainder to a particular component.

`--aba-run-dir` is optional and repeatable. Use it when a seed-123 reversal remains after the full matrix. The report validates the isolated run's halo result, common build and fixed input, exactly three accepted strict-GPU samples on one two-node PBS allocation, stable physical GPU mapping, and raw numerical/timing artifacts before printing their ERAD/HWTOPOLB/ERAD order and loop/full-wall times. The sequence is excluded from the 30 paired ratios. Preserve the PBS output and scheduler exit status alongside it; a three-sample order check alone does not establish the cause.

## Outstanding PBS evidence

- Complete rotated pilot before rebalance and its analysis.
- One small two-rank GPU trace smoke, a matching unprofiled invocation, and a validated complete trace from both ranks.
- Full unprofiled matrix on two ranks with one verified physical GPU per rank, all 64 invocations accepted, plus a fresh two-rank halo reference PASS in the same run manifest.
- Hardware, compiler and flags, MPI, node/affinity, OpenCL and driver, clocks/power/contention, input, warm-up, repeated timings, and same-environment baseline; mark any unmeasured optional field unavailable.
- Isolated seed-123 A/B/A if the rotated matrix still shows the anomaly.

These gates remain unvalidated in this workspace because PBS and the two-GPU cluster are unavailable here. Do not replace this pending status with a conclusion until the raw cluster artifacts have been reviewed.
