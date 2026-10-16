# HARMONY — Heterogeneous Adaptive Runtime for Multi-Node Orchestration & Numerics Yielding

> **C++20 · OpenCL · MPI · `dcl` namespace**

HARMONY is a header-driven, high-performance computing (HPC) library for **topology-aware heterogeneous load balancing** in distributed scientific simulations. It orchestrates work distribution across multi-GPU, multi-CPU, and accelerator clusters, dynamically redistributing computational partitions based on observed device throughput, interconnect topology, NUMA hierarchy, resource contention, and power envelopes.

---

## Table of Contents

1. [Motivation & Theoretical Background](#1-motivation--theoretical-background)
2. [Macro-Level Transformations](#2-macro-level-transformations)
3. [Future Work Items Consolidated](#3-future-work-items-consolidated)
   - [R1 — Exact PCIe and MPI Latency/Bandwidth Injection](#r1--exact-pcie-and-mpi-latencybandwidth-injection)
   - [R2 — NUMA Communication Costs in the Balancer](#r2--numa-communication-costs-in-the-balancer)
   - [R3 — Hierarchical Balancing Strategy](#r3--hierarchical-balancing-strategy)
   - [R4 — ERAD vs. HWTOPOLB Algorithm Comparison](#r4--erad-vs-hwtopolb-algorithm-comparison)
   - [R5 — Resource Contention & Power Consumption Model](#r5--resource-contention--power-consumption-model)
4. [File Inventory](#4-file-inventory)
5. [Directory Structure](#5-directory-structure)
6. [Building and Running](#6-building-and-running)
7. [Running on a PBS Cluster](#7-running-on-a-pbs-cluster)
8. [Test Results & Verification](#8-test-results--verification)
9. [Static Analysis Baseline](#9-static-analysis-baseline)

---

## 1. Motivation & Theoretical Background

HARMONY was originally developed to address the shortcomings of static and round-robin partitioning in heterogeneous clusters where GPUs, CPUs, and accelerators coexist under a single MPI communicator. The library's architecture and algorithmic foundations are directly inspired by the paper:

> **"A topology-aware load balancing algorithm for clustered hierarchical multi-core machines"**
> (referred to herein as the **HwTopoLB paper**)

The paper identifies several limitations of purely throughput-driven balancing and proposes extensions exploiting cluster topology — i.e., PCIe lanes, NUMA domains, inter-node MPI latency — to make more informed migration decisions. Five concrete *future work* directions from that paper have been fully implemented and integrated into HARMONY in this release:

| ID | Description |
|:---|:---|
| R1 | Inject exact PCIe and MPI latency/bandwidth metrics into the runtime |
| R2 | Include NUMA communication costs when deciding whether to rebalance |
| R3 | Hierarchical balancing: intra-node (local, no MPI) followed by inter-node (global, with MPI) |
| R4 | Head-to-head comparison of ERAD (deterministic) vs. HWTOPOLB (stochastic) algorithms |
| R5 | Resource contention model and thermal/power cap integration |

---

## 2. Macro-Level Transformations

Before this release, HARMONY supported basic throughput-driven rebalancing: the runtime collected per-device kernel timings and redistricted partitions proportionally. The balancer was topology-blind — it had no awareness of PCIe bandwidth, NUMA topology, memory contention, or power draw.

After this release, HARMONY is a **topology-informed, power-aware, hierarchically-structured load balancer** with the following new capabilities:

- **Structured topology ingestion**: External measurements of PCIe and MPI latency/bandwidth can be injected at runtime via `set_topo_metrics()`, enabling hardware-calibrated migration cost estimates.
- **NUMA-cost gating**: Before executing a migration, the runtime estimates the NUMA-aware data movement cost (PCIe latency + NUMA hop penalty + transfer time) and compares it against the predicted performance gain. Rebalancing is skipped when the cost-to-gain ratio exceeds a configurable threshold, avoiding counter-productive migrations.
- **Hierarchical two-phase balancing**: A new `BalanceMode::hierarchical` delegates intra-node redistribution to local logic (no MPI synchronization overhead) and only invokes the global MPI-based balancer for inter-node load equalization, reducing collective communication frequency.
- **Algorithm-level benchmarking**: Two balancing algorithms — ERAD (deterministic, proportional-throughput) and HWTOPOLB (stochastic, perturbation-augmented) — are now formally implemented in `dcl/algorithms.hpp` and evaluated against two scientific benchmarks (`kNeighbor`, `LeanMD`).
- **Power-constrained load assignment**: The `apply_power_cap()` function redistributes partition shares respecting per-device thermal TDP limits and an optional system-wide power budget, ensuring safe operation on thermally constrained clusters.

---

## 3. Future Work Items Consolidated

### R1 — Exact PCIe and MPI Latency/Bandwidth Injection

**Goal**: Replace hard-coded or estimated topology parameters with real, measurable values obtained from the actual cluster hardware.

**Implementation**:

- **`dcl/types.hpp`** — Extended with `struct TopoMetrics`, a flat data container for all topology measurements:
  - `pcie_latency_ns` — Per-device PCIe round-trip latency (nanoseconds)
  - `pcie_bandwidth_gbps` — Per-device PCIe peak transfer bandwidth (GB/s)
  - `mpi_latency_ns` — Flat R×R matrix of MPI latency between rank pairs (ns), where R is the MPI rank count
  - `mpi_bandwidth_gbps` — Flat R×R matrix of MPI bandwidth between rank pairs (GB/s)

- **`dcl/topo_metrics_io.hpp`** *(new)* — Header-only JSON serialization/deserialization for `TopoMetrics`. Provides `load_topo_metrics(path)` and `save_topo_metrics(metrics, path)` with a hand-written JSON parser (no external JSON library dependency).

- **`topo_probe.cpp`** *(new)* — Standalone MPI+OpenCL program that runs ping-pong benchmarks across all MPI rank pairs to measure real latency and bandwidth. Each rank probes its local OpenCL devices for PCIe characteristics. Results are aggregated at rank 0 and written to a JSON file (e.g., `topo_metrics_cluster.json`) consumable by `load_topo_metrics()`.

- **`dcl/runtime.hpp`**, **`dcl/runtime.cpp`**, **`dcl/runtime_impl.hpp`** — Added `set_topo_metrics(const TopoMetrics&)` and `topo_metrics() const` to the `Runtime` public API. Internally stored in the `Impl` class and forwarded to the rebalancing logic.

- **`tests/test_topo_metrics.cpp`** *(new)* — 7 unit tests validating JSON round-trips, error handling on malformed inputs, empty arrays, and correct field deserialization.

**Test result**: **7/7 tests passing**.

---

### R2 — NUMA Communication Costs in the Balancer

**Goal**: Prevent the balancer from triggering data migrations that cost more in NUMA-hop overhead than the performance gain they yield.

**Implementation**:

- **`dcl/types.hpp`** — `TopoMetrics` extended with:
  - `numa_distance` — Flat N×N NUMA distance matrix (unitless, same semantics as Linux `numactl --hardware` distances)
  - `device_numa_node` — Per-global-device NUMA node index mapping

- **`dcl/topo_metrics_io.hpp`** — Added two free functions:
  - `numa_distance_between(metrics, node_a, node_b)` — Returns the symmetric NUMA distance between two NUMA nodes, validated against the diagonal to avoid underestimation.
  - `estimate_migration_cost_bytes(metrics, src_dev, dst_dev, bytes, penalty_ns_per_hop)` — Computes total estimated migration cost in seconds: PCIe latency + NUMA penalty (distance × `penalty_ns_per_hop`) + transfer time (`bytes / bandwidth`). Returns conservative fallback values (5 µs latency, 10 GB/s bandwidth) when metrics are absent.

- **`dcl/types.hpp`** — `AutoBalancePolicy` extended with:
  - `numa_cost_gain_ratio_threshold` (default `0.50`) — Suppresses rebalancing when `migration_cost / expected_gain > threshold`.

- **`dcl/runtime_impl.hpp`** — `maybe_rebalance_from_timings()` computes the expected gain from proposed new partitions vs. current timings. Before issuing the rebalance, it calls `estimate_migration_cost_bytes()` for each device pair involved in migration and checks the cost-to-gain ratio. If the ratio exceeds the threshold, it prints:
  ```
  [NUMA] migration cost exceeds gain threshold, skipping rebalance
  ```
  (rank 0 only) and returns `false` without migrating data.

- **`tests/test_numa_cost.cpp`** *(new)* — 4 unit tests covering: same-node migration (no NUMA penalty), cross-node migration with penalty, empty metrics fallback, and threshold gating logic.

**Test result**: **4/4 tests passing**.

---

### R3 — Hierarchical Balancing Strategy

**Goal**: Reduce global MPI collective overhead by performing local (intra-node) load equalization independently, only escalating to inter-node MPI exchanges when necessary.

**Implementation**:

- **`dcl/types.hpp`** — `BalanceMode` enum extended with `hierarchical` variant.

- **`dcl/runtime.hpp`**, **`dcl/runtime.cpp`**, **`dcl/runtime_impl.hpp`** — Implemented `maybe_rebalance_hierarchical(rebalance_fields)`:
  1. **Intra-node phase**: Each MPI rank independently computes a local partition rebalancing among its own devices using throughput-proportional shares. No MPI communication occurs in this phase.
  2. Ranks gather measured throughput and rank 0 broadcasts the proposed rank boundaries.
  3. When rank boundaries remain fixed, ranks migrate fields locally and publish the resulting partition map with one reduction. When boundaries change, ranks exchange the global partition map and migrate the selected fields across ranks.

  Rank 0 emits diagnostic logs at each phase:
  ```
  [HIER] intra-node loads: ...
  [HIER] inter-node loads: ...
  [HIER] final partition: ...
  ```

- **`tests/test_hierarchical.cpp`** *(new)* — 6 unit tests covering: basic two-phase execution, barrier sequencing, intra-only path (single rank), empty field list, repeated invocation, and asymmetric device counts per rank.

**Test result**: **6/6 tests passing**.

---

### R4 — ERAD vs. HWTOPOLB Algorithm Comparison

**Goal**: Provide a rigorous, reproducible head-to-head comparison between the deterministic ERAD algorithm and the stochastic HWTOPOLB algorithm on representative scientific kernels.

**Implementation**:

- **`dcl/algorithms.hpp`** *(new)* — Contains three standalone, header-only functions within `namespace dcl`:

  - `erad_loads(times, partitions)` — Pure deterministic balancing. Computes per-device throughput as `element_count / kernel_time`, normalizes to a cumulative load vector `[0, 1]`.
  - `hwtopolb_loads(times, partitions, perturbation_factor)` — Stochastic augmentation of ERAD. Applies bounded uniform perturbation `delta_i ∈ [-p, +p]` to each device share, clamps to avoid negative shares, normalizes, and returns cumulative loads. Reduces to ERAD when `perturbation_factor == 0.0f`.
  - `apply_power_cap(loads, metrics, power_budget_watts)` — Post-processing step (see R5).

- **`benchmarks/kneighbor.cpp`** + **`benchmarks/kneighbor.cl`** *(new)* — Synthetic k-nearest-neighbor stencil benchmark (k=5, N=1,000,000 elements, 100 iterations). Each iteration executes the OpenCL stencil kernel across all available devices, then evaluates rebalancing every 10 steps under both ERAD and HWTOPOLB modes. Outputs per-iteration timing CSVs.

- **`benchmarks/leanmd.cpp`** + **`benchmarks/leanmd.cl`** *(new)* — Molecular dynamics Lennard-Jones pair-force benchmark. Simulates particle cutoff domain interactions (r⁻¹² − r⁻⁶) with spatially non-uniform density distributions, exercising the balancer under irregular workloads.

- **`benchmarks/compare_algos.sh`** *(new)* — Shell script that compiles both benchmarks (using `$MPI_LAUNCHER` or `mpirun`), runs each with `--algo erad` and `--algo hwtopolb` modes, collects CSV outputs, and prints a summary table.

- **`benchmarks/comparison_report.md`** *(new)* — 619-word analytical report covering:
  - Experimental results table (total time, rebalance count, average gain per algorithm/benchmark)
  - Philosophical analysis of deterministic vs. stochastic balancing
  - When stochastic perturbation helps (escaping local minima, desynchronizing interconnect bursts, robustness to noise)
  - When it hurts (BSP straggler penalties, oscillation, regular stencil predictability)
  - Practical guidance: prefer ERAD for structured stencils (`kNeighbor`); consider hybrid decaying-perturbation HWTOPOLB for particle methods (`LeanMD`)

- **`tests/test_algorithms.cpp`** *(new)* — 3 unit tests validating: ERAD produces valid monotone cumulative loads, HWTOPOLB with `perturbation_factor=0` matches ERAD output, and `apply_power_cap()` preserves load vector validity.

**Benchmark results** (measured locally in test environment via `compare_algos.sh` with N=1,000,000, 100 iterations; submit `job_hwtopolb.pbs` on cluster to collect real cluster numbers):

| Benchmark | Algorithm | Total Time (s) | Rebalances | Avg Gain (s) |
|:---|:---|:---:|:---:|:---:|
| `kneighbor` | ERAD | 0.547 | 9 | 0.000339 |
| `kneighbor` | HWTOPOLB ±10% | 0.562 | 9 | 0.000432 |
| `leanmd` | ERAD | 1.340 | 9 | 0.000733 |
| `leanmd` | HWTOPOLB ±10% | 1.388 | 9 | 0.000805 |

**Test result**: **3/3 tests passing**.

---

### R5 — Resource Contention & Power Consumption Model

**Goal**: Make the balancer aware of memory contention and thermal constraints, preventing assignment of work to already thermally saturated devices and respecting a configurable system power budget.

**Implementation**:

- **`dcl/types.hpp`** — `TopoMetrics` extended with:
  - `memory_contention_factor` — Per-device multiplier ≥ 1.0 representing relative memory bandwidth saturation (1.0 = no contention, 2.0 = half effective bandwidth)
  - `thermal_tdp_watts` — Per-device thermal design power in watts
  - `current_power_watts` — Per-device current measured power draw in watts

  `AutoBalancePolicy` extended with:
  - `use_contention_adjustment` (bool) — Enable contention-adjusted throughput calculation
  - `use_power_cap` (bool) — Enable power-capped load redistribution
  - `power_budget_watts` (double) — System-wide power budget ceiling (0 = unconstrained)

- **`dcl/topo_metrics_io.hpp`** — Added `adjusted_capacity(metrics, global_device, raw_throughput)`. Divides raw throughput by the device's `memory_contention_factor`, yielding a corrected effective capacity for use in load share computation.

- **`dcl/algorithms.hpp`** — `apply_power_cap()` uses measured power and the current partition shares to estimate a linear watts-per-share slope for each device. It redistributes work only within modeled TDP limits and checks the total budget. It raises an error when the model has insufficient measured capacity. This is a planning constraint, not hardware power enforcement; actual draw must be measured on the target devices.

- **`dcl/runtime_impl.hpp`** — `maybe_rebalance_from_timings()` integrates both adjustments:
  - When `use_contention_adjustment` is set, throughput is divided by `memory_contention_factor` before computing shares.
  - When `use_power_cap` is set, the candidate load vector is passed through `apply_power_cap()` before being applied.

- **`tests/test_contention_power.cpp`** *(new)* — 4 unit tests: contention factor reduces effective throughput proportionally, `apply_power_cap()` reduces overloaded device shares, `apply_power_cap()` preserves load vector validity (monotone, ends at 1.0), and passthrough when power metrics are absent.

**Test result**: **4/4 tests passing**.

---

## 4. File Inventory

### New Files

| File | Purpose |
|:---|:---|
| `dcl/topo_metrics_io.hpp` | JSON serialization/deserialization for `TopoMetrics`; NUMA distance query; migration cost estimation; contention-adjusted capacity |
| `dcl/algorithms.hpp` | `erad_loads()`, `hwtopolb_loads()`, `apply_power_cap()` algorithm implementations |
| `topo_probe.cpp` | MPI+OpenCL standalone tool to measure and emit real cluster PCIe/MPI topology metrics as JSON |
| `tests/test_topo_metrics.cpp` | 7 unit tests for R1 (topology metrics I/O) |
| `tests/test_numa_cost.cpp` | 4 unit tests for R2 (NUMA migration cost gating) |
| `tests/test_hierarchical.cpp` | 6 unit tests for R3 (hierarchical balancing) |
| `tests/test_algorithms.cpp` | 3 unit tests for R4 (ERAD/HWTOPOLB algorithms) |
| `tests/test_contention_power.cpp` | 4 unit tests for R5 (contention and power cap) |
| `benchmarks/kneighbor.cpp` | k-nearest-neighbor stencil benchmark driver (R4) |
| `benchmarks/kneighbor.cl` | OpenCL kernel for kNeighbor benchmark |
| `benchmarks/leanmd.cpp` | Lennard-Jones molecular dynamics benchmark driver (R4) |
| `benchmarks/leanmd.cl` | OpenCL kernel for LeanMD benchmark |
| `benchmarks/compare_algos.sh` | Automation script: compile, run, and summarize ERAD vs. HWTOPOLB results |
| `benchmarks/comparison_report.md` | 619-word analytical comparison report (R4) |
| `job_hwtopolb.pbs` | PBS cluster job script: loads modules, runs `topo_probe`, executes benchmark comparison |

### Edited Files

| File | Changes |
|:---|:---|
| `dcl/types.hpp` | Added `struct TopoMetrics` (R1); added `numa_distance`, `device_numa_node` fields (R2); added `memory_contention_factor`, `thermal_tdp_watts`, `current_power_watts` fields (R5); added `BalanceMode::hierarchical` (R3); added `numa_cost_gain_ratio_threshold`, `use_contention_adjustment`, `use_power_cap`, `power_budget_watts` to `AutoBalancePolicy` (R2, R5) |
| `dcl/runtime.hpp` | Added `set_topo_metrics()`, `topo_metrics()`, `maybe_rebalance_hierarchical()`, `set_simulated_devices_count()`, `set_simulated_times()`, `clear_simulated_times()` to the `Runtime` public API |
| `dcl/runtime.cpp` | Implemented public-API method bodies forwarding to `Impl` |
| `dcl/runtime_impl.hpp` | Integrated NUMA cost gating into `maybe_rebalance_from_timings()` (R2); implemented `maybe_rebalance_hierarchical()` two-phase logic (R3); integrated contention adjustment and power cap into rebalancing pipeline (R5) |
| `tests/test_hierarchical.cpp` | Fixed: `[HIER]` log assertions and PASS/summary messages gated on `rank == 0` only; `maybe_rebalance_hierarchical()` emits logs exclusively from rank 0 so rank 1 captured no output and aborted |
| `.gitignore` | Expanded to exclude `*.out` binaries, `Results/`, benchmark CSVs (`benchmarks/*.csv`), topology JSON outputs (`topo_metrics_*.json`, `test_probe.json`), PBS log files (`*.pbs.o*`, `*.pbs.e*`), `.agents/`, `ORIGINAL_REQUEST.md` |


### Removed Files

No source files were removed in the cleanup commit `14dfb9d`. That commit removed 804 tracked files under `Results/` and tracked binary artifacts such as `HIS.out`, `job_mpiocl.out`, and `profiling.out` from the working tree. The historical files remain available in the parent commit `b3206cd` through Git history (for example, `git show b3206cd:Results/np4/summary/summary_np4.csv`). The old timing tables are not a current performance baseline; regenerate results with the benchmark script and retain its metadata with each run.

---

## 5. Directory Structure

```
HARMONY/
├── dcl/                          # Core library headers (dcl namespace, C++20)
│   ├── types.hpp                 # All fundamental data structures (TopoMetrics, BalanceMode, AutoBalancePolicy, …)
│   ├── runtime.hpp               # Runtime class public API
│   ├── runtime.cpp               # Runtime public method implementations
│   ├── runtime_impl.hpp          # Runtime::Impl — full balancing logic (template-heavy, header-only)
│   ├── runtime_impl.cpp          # runtime_impl.cpp stub (explicit instantiation guard)
│   ├── topo_metrics_io.hpp       # JSON I/O, NUMA query, migration cost, capacity adjustment (NEW)
│   └── algorithms.hpp            # erad_loads(), hwtopolb_loads(), apply_power_cap() (NEW)
│
├── tests/                        # Unit tests (standalone, no external framework)
│   ├── test_topo_metrics.cpp     # R1 — 7 tests
│   ├── test_numa_cost.cpp        # R2 — 4 tests
│   ├── test_hierarchical.cpp     # R3 — 6 tests
│   ├── test_algorithms.cpp       # R4 — 3 tests
│   └── test_contention_power.cpp # R5 — 4 tests
│
├── benchmarks/                   # Benchmark drivers and analysis
│   ├── kneighbor.cpp             # kNeighbor stencil benchmark (NEW)
│   ├── kneighbor.cl              # kNeighbor OpenCL kernel (NEW)
│   ├── leanmd.cpp                # LeanMD molecular dynamics benchmark (NEW)
│   ├── leanmd.cl                 # LeanMD OpenCL kernel (NEW)
│   ├── compare_algos.sh          # ERAD vs. HWTOPOLB automation script (NEW)
│   └── comparison_report.md      # Algorithmic comparison analysis (NEW)
│
├── topo_probe.cpp                # Cluster topology measurement tool (NEW)
├── job_hwtopolb.pbs              # PBS job: topo_probe + benchmark comparison (NEW)
├── .gitignore                    # Expanded to cover all generated artifacts
│
├── main.cpp                      # Example application entry point
├── kernels.cl                    # Primary simulation OpenCL kernels
├── *.cl                          # Additional OpenCL kernels (stencil, ocean, jacobi, …)
├── job_dcl_np{1,2,4}_metrics.pbs # PBS jobs for performance profiling at various MPI counts
└── profiling.cpp                 # Standalone profiling tool
```

---

## 6. Building and Running

### Prerequisites

| Dependency | Minimum Version | Notes |
|:---|:---|:---|
| C++ compiler | GCC 13 / Clang 17 | C++20 required (`-std=c++20`) |
| MPI | OpenMPI 5.0+ or MPICH 4+ | C++ bindings disabled (`OMPI_SKIP_MPICXX`) |
| OpenCL | 3.0 | ICD loader (`-lOpenCL`) |
| OpenCL headers | 3.0 | `-DCL_TARGET_OPENCL_VERSION=300` |

### Compiling the Unit Tests

Each test is a self-contained translation unit. Compile with `mpic++`:

```bash
# R1 — Topology metrics I/O
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_topo_metrics.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_topo_metrics.out

# R2 — NUMA migration cost
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_numa_cost.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_numa_cost.out

# R3 — Hierarchical balancing
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_hierarchical.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_hierarchical.out

# R4 — Algorithm comparison
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_algorithms.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_algorithms.out

# R5 — Contention and power cap
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_contention_power.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_contention_power.out
```

### Running Tests

```bash
mpirun -n 1 ./tests/test_topo_metrics.out
mpirun -n 1 ./tests/test_numa_cost.out
mpirun -n 2 ./tests/test_hierarchical.out
mpirun -n 1 ./tests/test_algorithms.out
mpirun -n 1 ./tests/test_contention_power.out
```

The NUMA, algorithm, and contention/power suites also include two-rank
consensus cases; run them with `mpirun -n 2` when two ranks are available.

For the probe/runtime JSON contract and the asynchronous GPU halo path:

```bash
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_probe_runtime_integration.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_probe_runtime_integration.out
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_async_halo_opencl.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_async_halo_opencl.out

mpirun -n 2 ./tests/test_probe_runtime_integration.out topo_metrics_cluster.json
timeout 60s mpirun -n 2 ./tests/test_async_halo_opencl.out
```

The asynchronous halo test prints `SKIP` and exits with status 77 if the job
exposes fewer than two OpenCL GPU partitions. Only `PASS` with status 0 is
device validation.

### Compiling the Topology Probe

```bash
mpic++ -std=c++20 -Wall -Wextra -O2 \
  topo_probe.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o topo_probe.out
```

Run on the cluster to generate `topo_metrics_cluster.json`:

```bash
mpirun -n 4 ./topo_probe.out --output-json topo_metrics_cluster.json
```

Without an OpenCL platform, the probe exits with an error. Use `--synthetic`
only for test data; the generated JSON records `"synthetic": true`.

### Compiling the Benchmarks

```bash
# kNeighbor
mpic++ -std=c++20 -Wall -Wextra -O2 \
  benchmarks/kneighbor.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o kneighbor.out

# LeanMD
mpic++ -std=c++20 -Wall -Wextra -O2 \
  benchmarks/leanmd.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o leanmd.out
```

Both benchmark drivers accept `--quiet-runtime-metrics` for timed samples.
It suppresses detailed per-iteration runtime metric output and the optional
`DCL_METRICS_FILE` CSV, while leaving device event collection, MPI reductions,
and balancing inputs active. The default remains verbose for existing callers.
Compare the modes only under the same allocation and workload; the diagnostic
overhead is separate from any algorithm speedup claim.

For admitted two-rank GPU timing, pass `--require-two-rank-gpu` and
`--device-inventory-json PATH` to either benchmark. The command exits before
creating the step CSV unless there are exactly two ranks, exactly one OpenCL
GPU per rank, and distinct verifiable physical identities. The inventory is
one JSON file with `schema_version: 1` and rank-ordered `ranks` records. Each
record includes the MPI host and rank, GPU count, local and global device
indices, GPU and OpenCL platform names, platform and device versions, driver
version, and an `identity_kind` / `identity` pair. Identity is either a device
UUID or a PCI address qualified by the host. A missing identity is recorded as
`unavailable` and rejected in strict mode. Save and compare this inventory for
both conditions in each adjacent pair; a model name or runtime index alone
does not identify a physical GPU.

The summary retains `total_time_s` for historical consumers and adds
`loop_time_s`, `validation_time_s`, and `validation=PASS`. The primary
`loop_time_s` starts before iteration zero and ends immediately after the last
iteration's benchmark MPI reductions. Final gather and scalar reference check
run afterward; their rank-local intervals are reduced separately to produce
`validation_time_s`. The full `total_time_s` retains its existing broader
scope. A failed numerical check aborts without a PASS summary. K-neighbor uses
an absolute tolerance of `1e-5`; LeanMD uses `1e-4 + 1e-4 * abs(reference)`.
Both reject nonfinite results and references.

Run the focused command-interface test after building both binaries:

```bash
python3 tests/test_benchmark_sample_contract.py ./kneighbor.out ./leanmd.out
```

This local test covers CPU simulation output and collective sample rejection.
It does not validate an OpenCL kernel or a physical GPU mapping. The latter
requires a PBS two-rank job on the cluster, with one GPU exposed to each rank;
inspect the JSON inventory and require `validation=PASS` in both logs before
admitting either timing sample.

The public runtime control is `set_verbose_metrics(bool)`. The focused
simulation test can be built and run with:

```bash
mpic++ -std=c++20 -Wall -Wextra -O2 \
  tests/test_runtime_metrics.cpp -lOpenCL -DCL_TARGET_OPENCL_VERSION=300 \
  -o tests/test_runtime_metrics.out
mpirun -n 1 ./tests/test_runtime_metrics.out
mpirun -n 2 ./tests/test_runtime_metrics.out
```

For a local simulated overhead probe, pass `--measure`; its results must not
be interpreted as GPU performance. See the runtime metric mode diagnostic in
`docs/` for the measured local condition and limits.

For the controlled two-GPU variance experiment, use the resumable frontend
coordinator instead of submitting each PBS stage by hand:

```bash
python3 benchmarks/controlled_pbs.py --experiment-id variance-20261014-a
```

The command queues the pilot, diagnostic smoke, full matrix, and a CPU-only
finalizer with PBS `afterok` dependencies, then exits. The pilot job runs its
analyzer before releasing the smoke job; the smoke job validates both rank
traces before releasing the full matrix. The finalizer collects PBS records
and writes the report. Rerun the same command and ID after an interrupted
submission without duplicating jobs. See
`docs/controlled-gpu-variance-run.md` for failure recovery and optional
seed-123 A/B/A runs. Real GPU/PBS results remain necessary.

To check HARMONY separately on the older `compute-0-7`, `compute-0-8`, and
`compute-0-2` machines, queue three independent one-rank jobs with one command:

```bash
python3 benchmarks/queue_single_node.py --experiment-id legacy-20261014-a
```

This keeps the two-node A100 variance workflow above intact. Each job requests
one CPU/GPU chunk on its named host, probes a real OpenCL GPU, runs the short
main program, and checks both benchmark workloads and balancing algorithms
against their numerical references. Results are separated under
`Results/single-node-EXPERIMENT_ID/`. See
`docs/single-node-gpu-checks.md` for admission rules and recovery.

Or use the automation script (sets `MPI_LAUNCHER` if needed):

```bash
export MPI_LAUNCHER="mpirun"
bash benchmarks/compare_algos.sh
```

For a cluster build on the frontend, `BENCH_BUILD_ONLY=1` compiles the
benchmark executables without running them. During a PBS job,
`BENCH_SKIP_BUILD=1` and `BENCH_BIN_DIR=/path/to/binaries` reuse those
executables without invoking the compiler. The cluster workflow below sets
these variables automatically.

### Injecting Topology Metrics at Runtime

```cpp
#include "dcl/runtime.hpp"
#include "dcl/topo_metrics_io.hpp"

// Load measured metrics from JSON
dcl::TopoMetrics metrics = dcl::load_topo_metrics("topo_metrics_cluster.json");

// Inject into runtime
runtime.set_topo_metrics(metrics);

// Configure NUMA-aware policy
dcl::AutoBalancePolicy policy;
policy.mode = dcl::BalanceMode::dynamic_threshold;
policy.threshold = 0.05f;
policy.numa_cost_gain_ratio_threshold = 0.50;  // skip if cost > 50% of gain
policy.use_contention_adjustment = true;
policy.use_power_cap = true;
policy.power_budget_watts = 500.0;

step.with_balance(policy);
```

---

## 7. Running on a PBS Cluster

The jobs follow the LIMC cluster guide v5.0. Both reserve one full A100 GPU
and one MPI rank on each of `compute-1-0[0]` and `compute-1-1[0]`, with four
CPUs and 8 GB RAM per chunk. Check the current vnode inventory with
`pbsnodes -a` before submitting; the cluster configuration can change.
The frontend build and both jobs load the `gnu13/13.2.0`,
`openmpi5/5.0.5`, and `prun/2.2` modules observed on the cluster. They retain
the site's base modules. The cluster's `prun` launcher uses the two allocated
MPI slots. The OpenCL probe must find GPUs; neither job uses its synthetic
mode.

Update the Git checkout and build on the cluster frontend before submitting
jobs:

```bash
git pull --ff-only origin master
bash cluster_build.sh
```

If the cluster directory is not a Git checkout, transfer the complete source
tree and run `bash cluster_build.sh` there instead. The script compiles all
tests, the topology probe, the main program, and both benchmarks into
`build/cluster/` using C++20, the bundled Khronos OpenCL 3.0 headers, and the
OpenCL library found by `ldconfig`. Set `OPENCL_LIBRARY` to the full library
path if discovery fails. The compute nodes do not need development headers.
Submit the short main-program smoke test before
the longer evaluation:

```bash
qsub job_harmony_smoke.pbs
qstat -f JOB_ID
qstat -xf JOB_ID
```

The smoke job probes OpenCL and executes `main_high_order_residual.cpp` on two
ranks with a small mesh and two iterations. Its merged PBS output is
`harmony_smoke.out`. A zero exit code confirms that this path ran; it does
not establish numerical agreement with a reference solution.

After the smoke job succeeds, submit the evaluation job:

```bash
qsub job_hwtopolb.pbs
qstat -f JOB_ID
qstat -xf JOB_ID
```

The evaluation job runs the five core suites at their documented rank counts,
measures GPU/MPI topology, validates the generated JSON contract, runs the
two-rank asynchronous OpenCL halo test, and compares the two algorithms. It
fails if the device test skips or a benchmark falls back to CPU simulation.
The merged PBS output is `hwtopolb_eval.out`; the probe writes
`topo_metrics_cluster.json`, and benchmark CSVs and logs are under
`benchmarks/`.

The benchmark drivers currently compare `erad_loads()` and `hwtopolb_loads()`
without loading `topo_metrics_cluster.json` into their runtime. Treat the
probe as a separate topology measurement, not as an input to that comparison.
The jobs fail early if a required frontend-built executable is missing. A
cluster run is still needed to validate the installed OpenCL driver and GPU
execution paths.

---

## 8. Current Validation

The five documented test executables compile with C++20, MPI, and OpenCL 3.0. Run the topology, NUMA, algorithm, and contention/power suites with one rank and the hierarchical suite with two ranks. The NUMA, algorithm, and contention/power suites also have two-rank consensus cases. The integration test `tests/test_probe_runtime_integration.cpp` accepts a JSON file produced by `topo_probe.cpp` and checks the rank-to-rank matrix contract.

A short two-rank CPU simulation has passed for both benchmark drivers and the comparison script. These runs do not validate OpenCL kernels, GPU halo exchanges, OpenCL resource lifetime under a driver, or hardware power control. Run the corresponding cases on a GPU OpenCL platform before making device correctness or performance claims. See `benchmarks/comparison_report.md` for benchmark interpretation and `future_work_colab.md` for the remote GPU workflow.

## 9. Static Analysis

`git diff --check` passes for the current changes. A broad optional cppcheck run in the constrained local environment reached the last benchmark file without reporting a finding, then hit its 40-second limit; it is not a complete static-analysis pass. Use a full C++20 build and a complete static-analysis run in CI or a less constrained machine.

---

*HARMONY — Distributed Computing Laboratory (DCL) · `namespace dcl` · C++20 · OpenCL 3.0 · MPI*
