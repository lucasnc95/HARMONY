# HARMONY GPU Benchmark Variance Investigation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task by task. Checkboxes track completion; none of the tasks below has been executed by creating this document.

**Goal:** Explain why HWTOPOLB is much faster than ERAD for seeds 42 and 456 but approximately equal or slower for seed 123 on the two-GPU cluster, and establish whether any repeatable performance gain remains after controlling run order and measurement conditions.

**Architecture:** First recover all information available in the existing CSVs and logs. Then add opt-in, per-rank phase and partition traces to the two benchmark drivers. Finally run a balanced, repeated experiment on the same cluster allocation and issue a correctness and performance report with separate conclusions.

**Tech Stack:** C++20, MPI, OpenCL 3.0, PBS/OpenHPC, Bash, Python 3 standard library.

**Spec:** [Benchmark comparison report](../../../benchmarks/comparison_report.md), [cluster workflow](../../../README.md), and the observed run recorded below.

## Observed run and open questions

The cluster run used two MPI ranks on two nodes, one measured OpenCL GPU per rank, `N=1,000,000`, 100 iterations, rebalance interval 10, perturbation 0.10, and seeds 42, 123, and 456. All four summary rows reported `gpu_opencl`; every run recorded nine rebalances. The asynchronous OpenCL halo test passed against its CPU reference. The PBS job exited nonzero only when the postprocessor called `git` on a compute node; commit `59b6a6c` removed that dependency. The recovered metadata has null commit, host, platform, and compiler fields, so those values must not be inferred from it.

| Benchmark | Algorithm | Seed 42 (s) | Seed 123 (s) | Seed 456 (s) |
| --- | --- | ---: | ---: | ---: |
| k-neighbor | ERAD | 3.73996 | 3.57973 | 3.87973 |
| k-neighbor | HWTOPOLB | 0.220817 | 3.81972 | 0.214246 |
| LeanMD | ERAD | 4.30019 | 4.32997 | 4.22120 |
| LeanMD | HWTOPOLB | 0.438039 | 4.45991 | 0.36194 |

The paired ERAD/HWTOPOLB ratios range from 0.94 to 18.11. The identical seed pattern across workloads suggests seed-dependent partitioning or a run-order/environment interaction; it does not identify the cause. Existing per-iteration CSVs contain `step_time_s`, `action`, `rebalanced`, and `rebalance_gain`, but no partition trajectory or per-rank phase breakdown. The comparison currently does not pass `topo_metrics_cluster.json` to `hwtopolb_loads()`. Do not attribute a gain to measured topology until that behavior exists and is tested.

## Global constraints and acceptance gates

- Keep all scripts, source changes, comments, and reports in English. Use C++20, `CL_TARGET_OPENCL_VERSION=300`, and the project build procedure. Compile on the frontend; execute GPU runs only inside PBS allocations.
- Preserve algorithm behavior while adding diagnostics. Diagnostic logging must be opt-in and must not be enabled in the final unprofiled timing samples.
- Every timed sample must use two ranks, one verified GPU per rank, the same input, compiler flags, executable digest, device assignment policy, and allocation type. Record GPU model, driver, OpenCL platform/version, CPU model, MPI version, affinity, clocks/power state if available, and PBS job ID.
- Preserve the current numerical checks: k-neighbor absolute error at most `1e-5`; LeanMD error at most `1e-4 + 1e-4 * abs(reference)`. Require finite values and a successful two-rank halo reference test before interpreting times.
- Report raw per-seed samples, per-seed paired ratios, median and spread. A pooled median across seeds cannot hide a slow seed. Do not claim a general speedup if any seed still reverses the result or if repeated samples remain unstable.
- Keep original cluster CSVs/logs immutable in a dated run archive. Write all new outputs to a separate run directory. Never overwrite the seed-42/123/456 evidence above.

## File map

| File | Responsibility |
| --- | --- |
| `benchmarks/analyze_variability.py` (new) | Validate existing run files and produce per-seed, per-iteration timing diagnostics. |
| `tests/test_analyze_variability.py` (new) | Check complete/incomplete run handling, mode checks, and timing decomposition on small fixtures. |
| `benchmarks/kneighbor.cpp`, `benchmarks/leanmd.cpp` | Add optional rank-local phase, device, and partition tracing without changing the normal timed path. |
| `dcl/runtime_impl.hpp` (only if needed) | Add a narrowly scoped, opt-in timer inside a runtime phase that benchmark-level timing cannot separate. |
| `benchmarks/run_variability.sh` (new) | Run the controlled benchmark matrix using prebuilt binaries and unique output paths. |
| `cluster_build.sh` | Record the source revision and hashes when producing the benchmark binaries. |
| `job_harmony_variability.pbs` (new) | Reserve the same two-GPU/two-rank cluster shape and run the controlled matrix. |
| `benchmarks/variance_report.md` (new) | Record correctness, environment, raw timing references, diagnosis, and bounded conclusions. |

## Review focus

1. A missing, duplicate, truncated, or CPU-fallback CSV must fail analysis rather than enter a speedup calculation; Task 1 tests this.
2. A slow `total_time_s` with ordinary `step_time_s` values must be reported as unaccounted time, not labeled kernel cost; Task 1 tests this.
3. Rank-local trace files must not race on the same path, and empty partitions must remain visible; Task 2 tests this.
4. A diagnostic run must not silently replace an unprofiled sample or reuse an existing output filename; Task 3 tests this.
5. A GPU or rank mapping mismatch between samples must invalidate a paired comparison; Task 3 tests this.

## Task 1: Reconstruct the current run without rerunning it

**Files:** Create `benchmarks/analyze_variability.py` and `tests/test_analyze_variability.py`; write a read-only evidence snapshot under a dated, ignored local results directory.

- [ ] Copy the 12 original CSVs, 12 logs, summary, topology JSON, and recovered metadata from the cluster into a dated archive. Record their SHA-256 hashes and the original PBS job ID. Do not commit bulk raw data.
- [ ] Implement a CLI that accepts a run directory and the expected seeds/iteration count, validates all 12 files and `gpu_opencl`, and emits one row per benchmark, algorithm, seed, and iteration. Keep wall time, summed step time, rebalanced-step sum, ordinary-step sum, first-iteration time, final-iteration time, and `wall - sum(step_time_s)` separately.
- [ ] Add fixtures for missing/duplicate iteration, wrong execution mode, nonfinite timing, missing `total_time_s`, and an intentionally large unaccounted-time remainder. Run only `python3 -m unittest discover -s tests -p 'test_analyze_variability.py'`.
- [ ] Apply the analyzer to the archived run. For seed 123 versus 42/456, locate the first slow iteration and determine whether the excess is concentrated at iterations 10, 20, ..., 90, at the final reference check, or outside the recorded steps. Publish a short evidence table before forming a cause hypothesis.
- [ ] Commit the analyzer and its focused tests; keep the archived results outside Git.

**Decision gate:** If the existing CSVs localize the entire difference to a specific phase, instrument that phase first in Task 2. If they do not, retain all phase probes described below. Do not infer partition movement from the `rebalanced=1` flag alone.

## Task 2: Add opt-in phase and partition traces

**Files:** Modify `benchmarks/kneighbor.cpp` and `benchmarks/leanmd.cpp`; extend the focused analyzer tests as needed.

- [ ] Add `--diagnostics-prefix PATH`. When set, each MPI rank writes a separate `PATH.rank<RANK>.csv` and a device inventory file containing rank, local/global device index, OpenCL name/type, and initial partition. Fail clearly if a trace file cannot be opened.
- [ ] At every iteration, record rank, iteration, seed, algorithm, old/new global offsets and counts for every partition, and elapsed host time for `runtime.rebalance_to()`, `runtime.execute()`, the benchmark's MPI reductions, and final gather/reference validation. Record `kernel_seconds_last` separately as device event timing; it may overlap host time and must not be added blindly to host phases.
- [ ] Record transferred bytes if the runtime exposes a measured count. Otherwise record changed partition boundaries and label any byte estimate as an estimate. Do not treat the number of rebalance attempts as migration volume.
- [ ] Include a monotonically increasing phase/iteration identifier in each rank trace. Confirm that both ranks produced 100 iterations, compatible collective progress, and no path collisions. Compile both benchmarks on the frontend and run one small two-rank GPU diagnostic case before the full matrix.
- [ ] Compare a diagnostic run with an uninstrumented run of the same condition to estimate tracing overhead; use only uninstrumented runs for headline performance results.
- [ ] Commit the opt-in instrumentation after the focused build and cluster smoke check.

**Decision gate:** Use rank traces to decide whether seed 123 is slow because of partition skew, migration/transfer, MPI waiting, kernel execution, or time outside those measured phases. If phase totals do not reconcile with wall time, add one narrowly targeted probe before explaining the variance.

## Task 3: Run a controlled two-GPU comparison

**Files:** Create `benchmarks/run_variability.sh` and `job_harmony_variability.pbs`; modify `cluster_build.sh`; add a small dry-run test for matrix generation and unique paths.

- [ ] Build precompiled binaries on the frontend; record both binary SHA-256 values and the source revision at build time. Refuse a benchmark run if binaries are missing or their recorded hashes differ.
- [ ] Request the same two-node, two-rank, one-GPU-per-rank PBS shape used by `job_hwtopolb.pbs`. At job start, log PBS ID/nodefile, module versions, OpenCL device identity per rank, GPU clocks/power mode if readable, and whether another workload shares either GPU.
- [ ] Run one untimed warm-up per benchmark/algorithm combination. Then collect at least five unprofiled repetitions for each of the three seeds and both algorithms on both benchmarks, retaining each raw log/CSV. Rotate seed order across repetitions and alternate algorithm order; do not run all ERAD samples before all HWTOPOLB samples.
- [ ] Refuse CPU fallback, incomplete iterations, numerical mismatch, missing GPU, wrong rank count, duplicate output path, or changed device mapping. Keep the same `N=1,000,000`, 100 iterations, interval 10, perturbation 0.10, k-neighbor `k=5`, and LeanMD cutoff 4 unless a second, explicitly labeled input-size experiment is added.
- [ ] If seed 123 remains anomalous, run an isolated A/B/A sequence (ERAD/HWTOPOLB/ERAD) for seed 123 on the same allocation. Run diagnostic traces separately from the unprofiled timing matrix.
- [ ] Check the matrix generator with a no-execution dry run, then submit the PBS job. Archive the job output, exit status, all raw samples, device inventory, and manifest.

**Decision gate:** A repeatable seed effect follows the seed across rotated run positions. A run-order or cluster-state effect follows position, GPU state, or node contention instead. If neither pattern is stable, increase repetitions and report the instability rather than a speedup.

## Task 4: Attribute cost and publish the result

**Files:** Create `benchmarks/variance_report.md`; optionally add compact plots under the dated results archive.

- [ ] Present correctness outcomes separately from timing outcomes. Include the numerical tolerances, halo reference result, GPU/OpenCL identity, MPI rank placement, and any unavailable environmental measurement.
- [ ] For each benchmark and seed, report all paired raw times, paired median ratio, min/max or interquartile spread, and run order. Show per-iteration timing around rebalance boundaries and old/new partition counts from both ranks.
- [ ] Reconcile wall time with host compute, rebalance/migration, MPI waiting, final validation, and unaccounted time. Treat OpenCL event times as a separate device timeline. Use profiler output only to diagnose the cause, not as a timed sample for the speedup estimate.
- [ ] State the best-supported cause of the seed-123 reversal, citing exact run/iteration/partition evidence. If evidence remains ambiguous, list the missing measurement and stop short of a causal claim.
- [ ] State explicitly that the present `hwtopolb_loads()` comparison uses throughput plus seeded perturbation and does not consume the measured topology JSON. Reserve any claim about topology-aware gains for a separate implementation and validation effort.
- [ ] Review the report against every acceptance gate above and commit the report. Do not commit large raw logs or claim a general HWTOPOLB speedup while a seed reversal persists.
