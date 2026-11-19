# Optional Google Colab Pre-Cluster Validation Testbed

To enable rapid fail-fast verification of C++17 compilation, OpenCL runtime driver compatibility, MPI halo exchanges, and numerical stability on physical GPU hardware without incurring HPC queue latency, we adopt an optional Pre-Cluster Validation testbed under `colab/`. This testbed complements the physical PBS cluster workflow (`job_harmony_super.pbs` per ADR-0001) while keeping the core HARMONY C++ runtime in `include/harmony/` and `src/` 100% frozen and untouched.

## Status

accepted

## Context

In high-performance computing development workflows, cluster compute resources are managed through batch schedulers (PBS) with shared queues, turnaround latency, and finite user quotas. Submitting jobs with unverified code risks wasting allocation time on trivial failures such as compiler incompatibilities, missing OpenCL ICD loader links, or MPI lifecycle faults.

Conversely, developer workstations and local machines frequently lack physical OpenCL GPU hardware, forcing developers to rely on CPU fallback simulation or skip device verification entirely. Developers require a fast, accessible, fail-fast mechanism to verify compilation, OpenCL platform discovery, and GPU execution paths on real hardware before submitting evaluation jobs to the cluster queue.

Google Colab provides on-demand access to real NVIDIA GPUs with OpenCL driver support. However, running validation workloads in Colab must be strictly scoped to avoid conflating single-node virtualized environments with real distributed HPC cluster evaluations.

## Decision

We adopt an optional Pre-Cluster Validation testbed located in the `colab/` directory, consisting of modular bash scripts and an interactive notebook:

1. **Modular Scripts**:
   - `colab/bootstrap.sh`: Non-invasively provisions system packages (Open MPI, OpenCL development headers, ICD loader, and `clinfo`) and verifies GPU discovery via `nvidia-smi` and `clinfo`, rejecting CPU-only fallback environments.
   - `colab/build.sh`: Compiles the `harmony` static library (`lib/libharmony.a`), tools (`bin/topo_probe`, `bin/migration_profiler`), benchmark drivers (`bin/kneighbor`, `bin/leanmd`), and C++ test suites (`tests/bin/test_*`) by delegating to `./build.sh lib tools benchmarks tests` (`-std=c++17`, OpenCL).
   - `colab/run_validation.sh`: Executes a three-phase validation workflow against the compiled `tests/bin/` and `bin/` binaries:
     - Phase 1: Unit tests covering R1-R5 (`tests/bin/test_topo_metrics`, `tests/bin/test_algorithms`, `tests/bin/test_numa_cost`, `tests/bin/test_contention_power`, `tests/bin/test_hierarchical`).
     - Phase 2: Asynchronous GPU halo communication test (`tests/bin/test_async_halo_opencl`) across 1 and 2 MPI ranks sharing the physical GPU.
     - Phase 3: Short functional runs of `bin/kneighbor` and `bin/leanmd` benchmarks under `erad` and `hwtopolb` algorithms to verify finite numerical convergence (`std::isfinite` checks) without generating heavy load.

2. **Interactive Notebook**:
   - `colab/harmony_colab.ipynb`: Provides a self-contained notebook interface with a 1-click execution flow for developers to clone, bootstrap, build, and validate changes directly from GitHub.

3. **Core Runtime Freeze**:
   - The core runtime in `include/harmony/` and `src/` remains 100% frozen and untouched (baseline `8d7c3392c8d616997b342f778ca041a9a717aff2`). All validation scripts and notebook drivers operate purely as external callers.

## Consequences

- **Fail-Fast Verification**: Developers can verify C++17 compilation, OpenCL kernel builds, and MPI communication on real GPU hardware in minutes before cluster submission.
- **Explicit Operational Boundaries**: Colab executes on a single-node virtualized GPU with MPI ranks oversubscribed on a single GPU. It validates functional correctness, driver detection, halo exchanges, and numerical finiteness, but **cannot substitute multi-node cluster benchmark measurements, interconnect latency evaluations, or speedup claims**. All authoritative performance baselines, rotated matrix evaluations, and speedup claims remain strictly governed by physical PBS cluster executions (ADR-0001).
- **Zero Impact on Core Runtime**: The core library in `include/harmony/` and `src/` is 100% frozen, maintaining architectural stability and preventing regression risks.
- **Independent Tooling**: The Pre-Cluster Validation workflow is completely optional and modular; the PBS cluster scripts (`./build.sh all`, `job_harmony_super.pbs`) remain fully autonomous.
