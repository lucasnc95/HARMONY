# Non-Invasive PBS Super Job Architecture for Variance Investigation

After earlier multi-job PBS dependency chains and invasive C++ diagnostic modifications triggered regressions that required reverts, we decided to implement the variance investigation plan strictly as a single-allocation PBS batch script ("Super Job") with external runner and post-processing tools, keeping the core HARMONY C++ runtime frozen and untouched. This guarantees that no cluster toolchain or MPI lifecycle bugs are re-introduced into the runtime, while leveraging runtime timing reports, external `nvidia-smi` telemetry, and prebuilt frontend binaries to isolate whether the seed-123 slowdown is an algorithmic or environmental artifact.

Minor non-invasive adjustments were made strictly to benchmark drivers (`benchmarks/kneighbor.cpp` and `benchmarks/leanmd.cpp`) to reject non-finite stencil outputs and isolate pure loop execution time (`loop_time_s`) from post-run validation, and to preflight testing (`tests/test_async_halo_opencl.cpp`) to enforce at least one verified OpenCL GPU per MPI rank.

## Status

accepted

## Considered Options

- **Invasive runtime-level instrumentation**: Modifying core runtime files (`src/runtime_impl.hpp`, `include/harmony/algorithms.hpp`) inside Layer 2 evaluation harnesses to emit custom trace CSVs or alter scheduling internals. Rejected because modifying frozen core C++ source code risks re-introducing cluster compilation bugs and MPI lifecycle defects.
- **Multi-job PBS dependency chain (`qsub -W depend=afterok`)**: Splitting execution across multiple small PBS jobs. Rejected because queue latency, job scheduling delays, and varying node allocations introduce uncontrolled hardware state divergence between runs.
- **Single-allocation non-invasive Super Job**: Retaining the frozen runtime core, executing an untimed warm-up, a rotated 5-repetition matrix, an A/B/A test for seed 123, and defensive Python stdlib post-processing inside a single PBS allocation. Accepted.

## Consequences

- The core runtime library (`include/harmony/` and `src/`) remains 100% frozen, untouched, and protected from Layer 2 regressions.
- Benchmark drivers `benchmarks/kneighbor.cpp` and `benchmarks/leanmd.cpp` isolate pure computation `loop_time_s` from verification inflation without altering runtime load-balancing behavior.
- Preflight validation `tests/test_async_halo_opencl.cpp` strictly verifies OpenCL GPUs per MPI rank before benchmark execution begins.
- Build procedures (`./build.sh all`) compile frontend binaries cleanly (`bin/` and `tests/bin/`) without recompilation on compute nodes.
- The entire evaluation completes deterministically on the exact same physical node and GPU state within a single 2-hour PBS reservation.

## Amendment (2026-10-21)

While the core runtime library remains 100% frozen and untouched, minimal modifications were authorized and applied strictly to benchmark drivers and preflight tests:
- `tests/test_async_halo_opencl.cpp`: Enforce a minimum of 1 verified OpenCL GPU per MPI rank in preflight validation.
- `benchmarks/kneighbor.cpp`: Add finite numerical checks (`!std::isfinite(val)`) to reject non-finite/NaN/Inf stencil outputs, and isolate pure loop execution time (`loop_time_s`) from verification inflation.
- `benchmarks/leanmd.cpp`: Add finite numerical checks to reject non-finite force outputs, and isolate pure loop execution time (`loop_time_s`) from verification inflation.

## Amendment (2026-11-18)

Upon migrating the fork's variance investigation and load-balancing capabilities (R1-R5, Super Job, and Pre-Cluster Validation) onto the refactored upstream `harmony` architecture (`namespace harmony`):
- The Layer 1 core freeze applies to `include/harmony/` and `src/` against baseline commit `8d7c3392c8d616997b342f778ca041a9a717aff2` (post-review Layer 1 fix update over `c5376aaa2c1a7bf6c68d92fcc3c349cb7a2fdd77`).
- Frontend compilation uses `./build.sh all` (producing `lib/libharmony.a`, `bin/topo_probe`, `bin/migration_profiler`, `bin/kneighbor`, `bin/leanmd`, and `tests/bin/test_*`).
- `job_harmony_super.pbs` integrates upstream `scripts/pbs/job_env.sh` and `scripts/pbs/rank_wrapper.sh` (per-rank GPU pinning and MIG rejection) while preserving the single-allocation 2-hour reservation, shared 60-second Preflight Verification budget, and prebuilt binary checks.
- `benchmarks/analyze_super_matrix.py` supports both upstream structured CSV timing reports (`#run`/`#timeline`/`#device`/`#process`/`#balance` via `--report-csv`) and Historical Evidence (`[DCL][balance-metrics]` logs), writing `super_matrix_summary.csv`, `super_matrix_paired.csv`, `hardware_telemetry.json`, and `SUPER_JOB_REPORT.md`.

No files under `include/harmony/` or `src/` were modified in Layer 2 (#42, #43).
