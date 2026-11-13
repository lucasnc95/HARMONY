# HARMONY

An abstraction layer over **OpenCL + MPI** that runs OpenCL kernels on a
heterogeneous cluster (CPUs, GPUs and accelerators on several nodes) with
**automatic load balancing**.

You write ordinary OpenCL kernels and describe the data and what each kernel
reads and writes; the runtime (namespace `harmony`):

- discovers the OpenCL devices of every MPI rank;
- partitions a 1D domain among all devices;
- runs several kernels per step, concurrently when they do not depend on each other;
- exchanges halos between neighbouring partitions only when needed, with every
  message in flight at once and overlapped with the computation of the interior;
- measures the time of each device and redistributes the data when it pays off;
- gathers the result on rank 0;
- measures the time of every phase and prints a readable report.

## Repository layout

```
build.sh               builds the library and the tools
bsbash/                build functions (incremental compilation, MPI/OpenCL detection)
include/harmony/       public API
  types.hpp              types: FieldSpec, KernelBinding, HaloSpec, ExecutionStep, AutoBalancePolicy, ...
  runtime.hpp            harmony::Runtime and the builders (bind / step)
src/                   library (libharmony.a)
  build.sh               list of library sources
  runtime.cpp            public API facade
  runtime_impl.hpp       declaration of Runtime::Impl (private)
  impl/                  implementation, one file per responsibility
    devices.cpp            MPI, device discovery, OpenCL queues
    resources.cpp          fields (buffers) and kernels
    partitioning.cpp       domain decomposition, loads <-> partitions
    execution.cpp          execute(): invocations, queues, full/interior/border phases
    events.cpp             dependencies between commands, kernel statistics, synchronization
    communication.cpp      chunked MPI transfers and gather()
    halo.cpp               halo exchange between neighbouring partitions
    redistribution.cpp     data migration after a repartition, rebalance_to()
    balancing.cpp          balancing policies (threshold and profiled)
    timing.cpp             per-phase timers and the timing report
  detail/                internal helpers (errors, load math, tables)
tools/migration_profiler/  measures the migration cost (profiled policies)
examples/              standalone applications, each with its own build.sh
  quickstart_stencil1d/  minimal example, checked against the CPU
  multi_kernel/          three kernels per step, checked against the CPU
  cardiac_tissue/        cardiac tissue with a ToR-ORd-like cell model, 2 kernels per step + VTK snapshots
  his/                   Human Immune System simulation (3D mesh) + VTK snapshots
  stencil2d/             2D stencils and a map kernel (ocean, lu, map): HARMONY vs. hand-written MPI+OpenCL
  run_checks.sh          builds and checks every example
```

## Requirements

- C++17 compiler with an MPI wrapper (`mpicxx`, `mpic++`; MPICH or Open MPI)
- OpenCL 1.2+ (headers and ICD loader) and at least one driver/device
- bash

## Building

```bash
./build.sh
```

Produces `lib/libharmony.a` and `bin/migration_profiler`. The build is
incremental: only what changed (sources, included headers or flags) is
recompiled.

```
Usage: ./build.sh [options] [targets]

Targets (default: lib tools):
  lib        static library lib/libharmony.a
  tools      bin/migration_profiler
  examples   builds every example with its own build.sh (needs lib)
  install    copies include/ and lib/ to the prefix given with -p
  clean      removes build/, lib/, bin/ and the examples' build output

Options:
  -r         release build (default): -O3
  -d         debug build: -O0 -g
  -j N       parallel compilations (default: number of cores)
  -p DIR     installation prefix for "install"
  -q         quiet: only errors
  -v         verbose: print every command
  -h         this help
```

Environment variables: `MPICXX` (MPI wrapper), `OPENCL_INCLUDE_DIR` (directory
with `CL/cl.h`), `OPENCL_LIBRARY_DIR` (directory with `libOpenCL.so`),
`CL_TARGET_OPENCL_VERSION` (default 300) and `EXTRA_CXXFLAGS`.

OpenCL is searched in the default paths and in common locations
(`/usr/local/cuda`, `/opt/rocm`, ...). The flags found are written to
`lib/harmony_config.sh`, which the `build.sh` of every example (or of your own
application) reads to use the same compiler and OpenCL. When the OpenCL
library is outside the default paths, its directory is stored in the
executable (rpath), so `LD_LIBRARY_PATH` is not needed.

For example, with a specific MPI wrapper and OpenCL installation:

```bash
MPICXX=/path/to/mpi/bin/mpic++ \
OPENCL_INCLUDE_DIR=/path/to/opencl/include \
OPENCL_LIBRARY_DIR=/path/to/opencl/lib \
./build.sh
```

To install elsewhere (headers + library):

```bash
./build.sh -p $HOME/opt/harmony install
```

## Using the API

`bind(...).arg(...)` binds the arguments of a kernel, and
`step(...).invoke(...).with_halo_exchange(...).with_balance(...)` builds one
iteration.

```cpp
#include <harmony/runtime.hpp>

int main(int argc, char** argv) {
    auto runtime = harmony::Runtime::create(argc, argv);       // initializes MPI
    runtime.discover_devices({harmony::DeviceKind::all, 0});   // every device of every rank
    runtime.set_partition({n, 1, sizeof(float), 1});           // n elements of one float

    auto u = runtime.create_field({"u", n, 1, sizeof(float), harmony::BufferUsage::read_write, host_u.data(),
                                   harmony::RedistributionDependency::proportional});
    // ... v, g, e in the same way
    auto gradient = runtime.create_kernel({"kernels.cl", "gradient", ""});
    auto update   = runtime.create_kernel({"kernels.cl", "update", ""});
    auto energy   = runtime.create_kernel({"kernels.cl", "energy", ""});

    auto bind_gradient = runtime.bind(gradient)
        .arg(0, u, harmony::Access::read)
        .arg(1, g, harmony::Access::write)
        .arg(2, n)
        .build();
    auto bind_update = runtime.bind(update)
        .arg(0, u, harmony::Access::read)
        .arg(1, g, harmony::Access::read)
        .arg(2, v, harmony::Access::write)
        .arg(3, c)
        .arg(4, n)
        .build();
    auto bind_energy = runtime.bind(energy)
        .arg(0, u, harmony::Access::read)
        .arg(1, e, harmony::Access::write)
        .arg(2, n)
        .build();

    harmony::ExecutionStep step = runtime.step("advance")
        .invoke(bind_gradient)
        .invoke(bind_update)
        .invoke(bind_energy)
        .with_halo_exchange(harmony::HaloSpec{1, {u, g}})      // width 1, fields u and g
        .with_halo_sync(harmony::HaloSync::overlap)
        .with_balance(harmony::AutoBalancePolicy{
            harmony::BalanceMode::dynamic_profiled,
            100,                     // interval
            0.0f,                    // threshold (threshold modes)
            iterations,              // total_iterations (profiled modes)
            "profiling_results.txt", // profiling_file (profiled modes)
            10                       // warmup: first attempt at iteration 10
        })
        .tag_field(v, harmony::StepFieldRole::rebalance_source)  // optional
        .synchronize_at_end(false)
        .build();

    runtime.start_timing();
    for (int it = 0; it < iterations; ++it) runtime.execute(step);   // usually alternates u <-> v
    runtime.stop_timing();
    runtime.print_timing_report();                                    // tables on rank 0

    runtime.gather(v, result.data(), n * sizeof(float));              // whole field on rank 0
}
```

Concepts:

- **Field**: a distributed buffer of `global_elements` elements. Every device
  keeps a full-size replica but only updates its own partition (+ halos).
  `RedistributionDependency::proportional` means the field follows the
  partition; `none` means replicated data (e.g. parameters), which can only be
  read.
- **Partition**: boundaries are always multiples of `granularity` (e.g. one XY
  plane of a 3D mesh).
- **KernelBinding** (`runtime.bind(kernel)`): a kernel and its arguments.
  - `arg(i, field, access)`: a field and how the kernel uses it:
    `Access::read`, `Access::write` (overwrites the whole partition) or
    `Access::read_write`;
  - `arg(i, field)`: without an explicit access the runtime uses `read` for
    replicated fields and for fields with a halo exchange in the step, and
    `read_write` (the most conservative) otherwise;
  - `arg(i, value)`: a scalar (int, float, trivially copyable struct, ...).
- **ExecutionStep** (`runtime.step(name)`): one iteration.
  - `invoke(binding, geometry)`: one kernel; several `invoke` calls = several
    kernels, in program order. `LaunchGeometry{global_offset, global_size,
    local_size}` is optional; `global_size` must be 0 or the number of
    elements of the partition (each device runs only its own part);
  - `with_halo_exchange(HaloSpec{width, {fields}})`: fields whose borders the
    kernels read. May be called several times for different widths;
  - `with_halo_sync(HaloSync)`: what happens right before the halo exchange;
  - `with_balance(AutoBalancePolicy{...})`: balancing policy;
  - `tag_field(field, StepFieldRole::rebalance_source)`: fields moved when the
    division changes. Without any `tag_field`, every proportional field of the
    step is moved;
  - `synchronize_at_end(bool)`: wait for the devices at the end of the step
    (default `false`, which keeps computation and communication overlapped).
- **DeviceSelection** (`discover_devices({kind, max_devices_per_rank,
  kernel_queues, cpu_host_cores})`): which devices are used,
  how many independent kernel queues each one gets (default 4) and how many
  cores of each CPU device are left free for the MPI process (default 0; the
  CPU device then becomes an OpenCL sub-device with the remaining cores).

### Several kernels per step

A step is an **ordered list of `invoke`s**, and every field argument declares
its access. The result is always the same as running the kernels one after
another, in the written order; concurrency is an optimization of the runtime.
From the accesses and the `HaloSpec`s the runtime derives:

| What | How |
|---|---|
| **Dependencies** | read-after-write, write-after-read and write-after-write, per field and per device, enforced with OpenCL events |
| **Concurrency** | each device has several kernel queues (`DeviceSelection::kernel_queues`, default 4). A kernel that depends on an earlier kernel of the step goes to that kernel's queue; an independent one goes to the next queue and runs at the same time. In the example above, `update` waits for `gradient` (it reads `g`), and `energy` runs concurrently with both |
| **When to exchange halos** | a field's halo becomes stale when the field is written and is exchanged only when a later kernel reads it (field listed in a `HaloSpec`), once. In the example, `u` is exchanged before `gradient` and `g` before `update`, with no explicit call |
| **What to migrate** | the fields tagged with `tag_field(..., rebalance_source)`; without tags, every proportional field used by the step |

Rules checked by `execute()` (error with a clear message):

- replicated fields (`none`) cannot be written;
- a field with a halo exchange cannot be `read_write` in the same kernel, nor
  read with a halo by a kernel that also writes it (the neighbours would read
  values already changed); use two buffers;
- `LaunchGeometry::global_size` must be 0 or the number of elements.

Limitations: all kernels of a step traverse the same partitioned domain (one
work-item per element), and a `write` must write every element of the
partition. Every device keeps the whole domain of each field in one OpenCL
buffer, so each field must fit in the largest buffer of the smallest device
(`CL_DEVICE_MAX_MEM_ALLOC_SIZE`; on NVIDIA usually 1/4 of the memory). The
device table printed by `print_devices()` shows it ("Largest buffer"), and
HARMONY warns when a field does not fit.

## Halo exchange

A kernel that needs an out-of-date halo runs in three phases per partition
(`run_interior_phase`, `exchange_halo_set`, `run_border_phase`): **interior**
(no neighbour data), **halo exchange** and **borders**. Without a pending
halo it runs at once. The exchange processes every message of every field and
every pair of neighbours together:

1. every `MPI_Irecv` is posted first;
2. every device read is enqueued at once; between devices of the same rank
   and the same OpenCL context the copy happens on the device
   (`clEnqueueCopyBuffer`, without the host);
3. each `MPI_Isend` starts as soon as its read finishes;
4. each received halo is written to its device as soon as it arrives
   (`MPI_Waitany`);
5. finally, the sends and the writes are waited for.

The interior never waits for the exchange (it does not read the ghost
region); the borders depend on the halo writes through OpenCL events. Halo
messages use their own MPI communicator.

The synchronization before the exchange is chosen per step (`with_halo_sync(...)`):

| `HaloSync` | Behaviour |
|---|---|
| `overlap` (default) | the exchange happens while the interior is computed |
| `local` | waits for the devices of the rank to finish the interior first |
| `global` | `local` + `MPI_Barrier`: every rank starts the exchange together |

In the examples: `--halo-sync overlap|local|global`.

## Timing and logs

Measuring adds no synchronization: CPU times are read with `MPI_Wtime` and
device times come from OpenCL profiling events, so the overlap between
computation and halo exchange is unchanged.

- `start_timing()` and `stop_timing()` are collective and delimit the measured
  interval. `stop_timing()` returns the wall time of the slowest process and
  freezes the counters: whatever comes later, such as `gather()`, is not in the
  report.
- `print_timing_report()` (collective) prints the summary on rank 0;
  `timing_report()` returns the same data in a struct (`TimingReport`).
- `write_timing_report(path)` (collective) writes the same report as CSV, for
  scripts.
- `set_verbose(false)` turns off the log of the balancing attempts (on by
  default; `--verbose 0` in the examples).

### Balancing log

Printed by rank 0 during the run. At the first attempt the policy is
described (warm-up, interval and criterion). Every attempt has three lines:

1. `Balance #N | iteration I | policy | CHANGED/kept the division`;
2. `Why:` the reason of the decision (threshold: the change of the division
   compared with the threshold; profiled: how much time the new division saves
   over the remaining iterations compared with the cost of moving the data);
3. `Time per iteration:` the time per iteration of the slowest device (it sets
   the time of the step), measured now and expected with the new division,
   assuming every device keeps its measured speed.

When the division changes, a table shows for each device its kernel time per
iteration in the measured window, its elements and share before and after,
and marks the slowest one. Example (small HIS mesh, 2 processes, warm-up 30,
interval 20):

```
[HARMONY] Load balancing: dynamic threshold
          Warm-up: 30 iterations with the initial division, then an attempt every 20 iterations.
          Each attempt measures the last 20 iterations and proposes shares proportional to the speed of each device;
          the new division is applied when it changes by at least 0.1 (L2 distance of the cumulative loads).
[HARMONY] Balance #1 | iteration 30 | dynamic threshold | CHANGED the division
          Why: division change 0.125 >= threshold 0.1
          Time per iteration: 497.3 us now (slowest: CPU dev 0) -> 380.0 us expected | measured over 20 iterations
          Dev  Rank  Type  Name                                  Kernel time/iteration  Elements before  Elements after  Change  Share before  Share after  Note
          ---  ----  ----  ------------------------------------  ---------------------  ---------------  --------------  ------  ------------  -----------  ----------
          0    0     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               497.3 us            1,280             960    -320        50.0 %       37.5 %  <- slowest
          1    1     CPU   cpu-skylake-avx512-Intel(R) Xeon(...               304.0 us            1,280           1,600    +320        50.0 %       62.5 %
          Data moved: 10.0 KiB in 194.2 us | decision took 158.1 us
[HARMONY] Balance #2 | iteration 50 | dynamic threshold | CHANGED the division
          Why: division change 0.375 >= threshold 0.1
          Time per iteration: 761.3 us now (slowest: CPU dev 1) -> 309.9 us expected | measured over 20 iterations
          ...
```

### Final summary

It starts with **one line for the critical path** (TIMELINE OF THE CRITICAL
PATH): the MPI process that drives the slowest device (the one with the most
kernel time), with the wall time split into exclusive parts that add up to
it: OpenCL compute (kernel time of that device), OpenCL overhead (the rest of
the time blocked in OpenCL: launches, copies, driver latency), halo
preparation, MPI waits for halos, migration, balancing and barriers, data
migration, balancing and unmeasured time. This line is meant for stacked-bar
plots (it is also in the CSV, row `timeline`, and in `TimingReport::timeline`).

The times of each MPI process are **exclusive**: when a phase starts inside
another one, the outer phase is paused. The columns therefore add up exactly
to the wall time, and "Not measured" is the time spent outside HARMONY
(application code between calls). For each device, Compute + Transfers + Idle
also add up to the wall time; concurrent kernels are counted once. The LOAD
BALANCING section lists the attempts (with the time per iteration before →
after) and the share of every device at the start and after each change.

```
====================================================================================================
HARMONY - TIMING SUMMARY
====================================================================================================
Wall time: 950.52 ms    Iterations: 101 (9.41 ms per iteration)
MPI processes: 2    Devices: 2
Load balancing: dynamic threshold, warm-up 30, interval 20, threshold 0.1

TIMELINE OF THE CRITICAL PATH (the parts add up to the wall time)
  ...

TIME OF EACH DEVICE (Compute + Transfers + Idle = wall time)
  Compute:    OpenCL kernels running
  Transfers:  copies between host and device memory (halos and data migration)
  Idle:       no work (waiting for other devices, MPI or the CPU)

  Dev  Rank  Type  Name                                  Elements   Share           Compute        Transfers                Idle
  ---  ----  ----  ------------------------------------  --------  ------  ----------------  ---------------  ------------------
  0    0     CPU   cpu-skylake-avx512-Intel(R) Xeon(...     2,048  80.0 %  44.59 ms (4.7 %)  69.1 us (0.0 %)  905.87 ms (95.3 %)
  1    1     CPU   cpu-skylake-avx512-Intel(R) Xeon(...       512  20.0 %  49.38 ms (5.2 %)  99.4 us (0.0 %)  901.04 ms (94.8 %)
  Busiest device: CPU dev 1, 49.38 ms of kernels = 488.9 us per iteration. Least busy: CPU dev 0, 1.11x less.
  A large Idle on the fast devices means they wait for the busiest one.

TIME OF EACH MPI PROCESS (the CPU that drives the devices; the columns add up to the wall time)
  ...

LOAD BALANCING
  4 attempts, 4 changed the division. Data moved: 128.0 KiB in 801.0 us. Decisions: 421.9 us.

  #  Iteration  Result         Time/iteration  Data moved  Moved in  Why
  -  ---------  -------  --------------------  ----------  --------  --------------------------------------
  1         30  changed  497.3 us -> 380.0 us    10.0 KiB  194.2 us  division change 0.125 >= threshold 0.1
  2         50  changed  761.3 us -> 309.9 us    30.0 KiB  187.8 us  division change 0.375 >= threshold 0.1
  ...

  Share of the domain per device: at the start and after each change (#attempt @iteration)
  Dev  Type   Start  #1 @30  #2 @50  #3 @70  #4 @90
  ---  ----  ------  ------  ------  ------  ------
  0    CPU   50.0 %  37.5 %  75.0 %  22.5 %  80.0 %
  1    CPU   50.0 %  62.5 %  25.0 %  77.5 %  20.0 %
====================================================================================================
```

## Balancing policies

`AutoBalancePolicy{mode, interval, threshold, total_iterations, profiling_file, warmup}`

| Mode | When it tries | How it decides |
|---|---|---|
| `off` | never | — |
| `static_threshold` | once, at iteration `warmup` | L2 distance between the current and the proposed cumulative loads ≥ `threshold` |
| `dynamic_threshold` | at iteration `warmup`, then every `interval` iterations | same |
| `static_profiled` | once, at iteration `warmup` | estimated time saved until the end of the run > estimated migration cost |
| `dynamic_profiled` | at iteration `warmup`, then every `interval` iterations | same |

**Warm-up** (`warmup`): iterations run with the initial division before the
first attempt (`0` = `interval`). Every attempt measures at most the last
`interval` iterations; with `warmup > interval`, the start of the run (JIT
compilation, initial transient, stimulus) is left out of the first
measurement. In the examples: `--balance-warmup N` (his, cardiac, stencil2d)
or `--warmup N` (quickstart and multi_kernel).

The proposed loads are proportional to the measured throughput of each device
(elements per second of kernel time in the window). The profiled modes use
the file written by `migration_profiler`, with linear segments
`cost = m * bytes + b`; without the file a conservative model is used. The
gain per iteration uses the actual number of iterations of the window.

A manual repartition is also possible with `Runtime::rebalance_to(cumulative_loads)`.

## Examples

The examples are **not part of the library**: each directory is a standalone
application with its own `main`, argument parser, kernels and `build.sh`. They
use only the public API (`include/harmony`) and `lib/libharmony.a`, and can be
copied out of the repository.

```bash
./build.sh                         # library
./examples/his/build.sh            # one example -> examples/his/bin/his
./build.sh examples                # or all of them
```

The `build.sh` of each example uses `HARMONY_ROOT` (default: the repository
root; the prefix of `./build.sh -p DIR install` also works) and reads
`lib/harmony_config.sh` to use the same compiler and OpenCL as the library;
`MPICXX`, `OPENCL_INCLUDE_DIR` and `OPENCL_LIBRARY_DIR` take precedence. Every
example accepts `--kernel-file` to point to another `.cl` file.

### quickstart_stencil1d

```bash
mpirun -np 2 examples/quickstart_stencil1d/bin/quickstart_stencil1d --n 100000 --iterations 200 --interval 50
```

One kernel per step. Prints `max |device - cpu|` and returns a non-zero code if
it differs from the sequential version.

### multi_kernel

```bash
mpirun -np 2 examples/multi_kernel/bin/multi_kernel --n 200000 --iterations 100
```

Three kernels per step (`gradient` → `update`, and `energy` concurrently), with
the halo of the intermediate field `g` exchanged automatically. Checked
against the CPU.

### cardiac_tissue — cardiac tissue (ToR-ORd-like model)

```bash
mpirun -np 2 examples/cardiac_tissue/bin/cardiac_tissue --nx 24 --ny 24 --nz 384 --time 100
```

Monodomain with operator splitting, **two kernels per step** (`tor_ord.cl`):

| Kernel | Accesses | What it does |
|---|---|---|
| `diffusion` | `vm`: read (halo of 1 plane), `vm_tmp`: write | 7-point Laplacian, no-flux boundaries |
| `reaction` | `vm_tmp`: read, `state`: read_write, `vm`: write | integrates the cell model over `dt` |

`reaction` waits for `diffusion` (it reads `vm_tmp`), and the halo of `vm` is
exchanged automatically before every `diffusion`, because `vm` was written by
the `reaction` of the previous step. The 3D tissue (`nx × ny × nz`) is
linearized in 1D with `z` as the slowest axis, so HARMONY cuts the tissue in
`z` planes and the halo is one plane (`nx·ny` elements).

**Cell model** (`torord_like.h`, the same file compiled in the kernel and in
the CPU check): O'Hara-Rudy family (ORd 2011, the basis of ToR-ORd), with 40
state variables + V, ~15 currents, three cell types (endo/mid/epi along `x`,
as in MonoAlg3D's `ToRORd_fkatp_mixed_endo_mid_epi`) and two ToR-ORd-style
extensions: ICaL split between subspace (80%) and cytosol (20%), and chloride
currents (ICaCl, IClb). **It is not the published ToR-ORd calibration**: it is
a benchmark model with the same structure and cost. For scientific use,
replace `torord_like.h` with the official equations. Single cell at 1 Hz:
rest −87.5 mV, peak ~39 mV, max dV/dt ~250 V/s, APD90 of 248 ms (endo),
229 ms (epi) and 341 ms (mid). In tissue the conduction velocity is ~0.5 m/s.

**Why it is a good balancing test:** every cell splits `dt` into substeps
according to the stiffness of the phase of the action potential (limits on
the change of V, `--dv-max`, and of cytosolic Ca²⁺, `--ca-tol`). With the
default values the cost per cell is ~120 substeps in the upstroke, 10–13
during Ca²⁺ release and early plateau, ~2.5 in the plateau, ~1.5 in
repolarization and 1 at rest. The wave starts at `z = 0` and travels along `z`,
the partitioned axis, so the expensive region moves from one device to the
next; then the whole tissue goes through plateau, repolarization and rest,
and the total cost changes again. Because the cost per element moves with the
wave front, dynamic balancing based on the previous window can oscillate, and
a static decision taken during the stimulus can become wrong later. To make
the kernel heavier: lower `--dv-max`/`--ca-tol`, raise `--min-substeps` or use
`--precision double`.

At the end the output shows the conduction velocity, the cell-model work per
band of `z` (where the load was) and the timing report. `--check 1` compares
with a sequential CPU run (small tissues only).

| Option | Default | Description |
|---|---|---|
| `--nx --ny --nz` | `24 24 384` | tissue size in cells; the wave travels along `z` |
| `--dx` / `--dt` | `0.01` cm / `0.01` ms | discretization (stability: D·dt/dx² ≤ 1/6) |
| `--time` | `100` | simulated ms |
| `--diffusivity` | `0.0012` | cm²/ms |
| `--bcl` / `--stim-planes` | `500` / `3` | time between stimuli and stimulated planes |
| `--dv-max` / `--ca-tol` | `0.005` / `2e-5` | substep tolerances (smaller = heavier) |
| `--min-substeps` / `--max-substeps` | `1` / `1000` | substep limits |
| `--precision` | `float` | `float` or `double` |
| `--balance-mode` / `--balance-strategy` | `dynamic` / `threshold` | as in HIS |
| `--rebalance-interval` / `--rebalance-threshold` | `200` steps / `0.01` | |
| `--balance-warmup` | `0` (= interval) | steps before the first attempt |
| `--device` | `all` | `all`, `gpu` or `cpu`: OpenCL devices used |
| `--progress` | `60` | seconds between progress lines (with an estimate of the total time) |
| `--cpu-host-cores` | `0` | cores of each CPU device left to the MPI process |
| `--report-csv` | — | timing report as CSV |
| `--halo-sync`, `--verbose`, `--timing-file`, `--check` | | as in the other examples |

### his — Human Immune System

```bash
mpirun -np 2 examples/his/bin/his --x 50 --y 50 --z 3200 --iterations 10000 \
  --balance-mode dynamic --balance-strategy threshold \
  --rebalance-interval 1000 --rebalance-threshold 0.0003125
```

| Option | Default | Description |
|---|---|---|
| `--x --y --z` | `50 50 3200` | mesh dimensions |
| `--iterations` | `10000` | time steps |
| `--precision` | `float` | `float` or `double` |
| `--balance-mode` | `dynamic` | `off`, `static` or `dynamic` |
| `--balance-strategy` | `threshold` | `threshold` or `profiled` |
| `--rebalance-interval` | `1000` | iterations between attempts |
| `--balance-warmup` | `0` (= interval) | iterations before the first attempt |
| `--rebalance-threshold` | `0.0003125` | threshold of the threshold strategy |
| `--profiling-file` | `profiling_results.txt` | profile of the profiled strategy |
| `--timing-file` | — | appends the wall time (one line per run) |
| `--gather-final` | `1` | gathers the final mesh on rank 0 and prints a checksum |
| `--print-mesh` | `0` | prints the antigen concentration of the final mesh |
| `--halo-sync` | `overlap` | `overlap`, `local` or `global` |
| `--verbose` | `1` | `0` turns off the log of the balancing attempts |
| `--device` | `all` | `all`, `gpu` or `cpu` |
| `--cpu-host-cores` | `0` | cores of each CPU device left to the MPI process |
| `--report-csv` | — | timing report as CSV |

Note on the model: in the equation of the anti-inflammatory cytokine (CA),
the diffusion term uses `Laplaciano(CELULAS_A_OFFSET, ...)`, i.e. the
diffusion of the **antigen** and not of CA itself, so CA exceeds its ceiling
`caInf = 3.6` near the antigen front. Fixing it changes the reference results
(checksum).

### stencil2d — ocean, lu and map: HARMONY vs. MPI+OpenCL

One program with three kernels (`stencil2d.cl`), selected with `--kernel`:

| Kernel | What it does |
|---|---|
| `ocean` | 5-point Jacobi average on an `nx × ny` grid (memory bound), halo of one row |
| `lu` | 9-point weighted average + a loop of floating-point operations, halo of one row |
| `map` | element-wise map with the same floating-point loop, no halo |

```bash
mpirun -np 2 examples/stencil2d/bin/stencil2d --impl both --kernel lu --nx 1024 --ny 1024 --iters 100
```

With `--impl both` the HARMONY version and a hand-written MPI+OpenCL version
run, their times are shown side by side (with the speedup) and the results
are compared (they must be bit-identical).

| Option | Default | Description |
|---|---|---|
| `--impl` | `harmony` | `harmony`, `mpiocl` or `both` |
| `--kernel` | `ocean` | `ocean`, `lu` or `map` |
| `--nx --ny` / `--n` | `1024 1024` / `16777216` | grid size (ocean, lu) / number of elements (map) |
| `--iters` | `100` | iterations |
| `--precision` | `float` | `float` or `double` (both versions) |
| `--flop-repeat` / `--chains` | `64` / `1` | the floating-point loop of `lu`/`map` does 4 × `flop-repeat` × `chains` operations per element. With 1 chain each work-item only has dependent operations, which favours GPUs; with 4 chains a CPU can fill its vector units |
| `--device` | `all` | `all`, `gpu` or `cpu` |
| `--balance-mode` / `--balance-strategy` | `off` / `profiled` | balancing (HARMONY only) |
| `--rebalance-interval` / `--balance-warmup` / `--rebalance-threshold` / `--profiling-file` | | as in the other examples |
| `--halo-sync` | `overlap` | `overlap`, `local` or `global` (HARMONY only) |
| `--timing-file` | — | appends the wall time (with `--impl both`: HARMONY and MPI+OpenCL) |
| `--report` / `--report-csv` | `1` / — | timing report on screen / as CSV |
| `--gather` | `1` | collects the result on rank 0 (`0` for benchmarks; required by `--impl both`) |
| `--cpu-host-cores` | `0` | cores of each CPU device left to the MPI process |

## Figures: VTK snapshots

HIS and the cardiac tissue have a second program, `his_snapshots` and
`cardiac_snapshots` (built by the `build.sh` of each example), which runs the
**same simulation** (same kernels and HARMONY steps, with balancing) and writes
the mesh as legacy VTK files (`STRUCTURED_POINTS`, binary) at a given
interval, **without timing**. Run it once, with a small mesh, to make figures
of the phases of the computation in ParaView (the series `name_0000.vtk`,
`name_0001.vtk`, ... opens as an animation).

```bash
mpirun -np 2 examples/cardiac_tissue/bin/cardiac_snapshots --nx 24 --ny 24 --nz 256 --time 300 --every 2 --output vtk/cardiac
mpirun -np 2 examples/his/bin/his_snapshots --x 32 --y 32 --z 256 --iterations 20000 --every 1000 --output vtk/his
```

| Program | Fields (one value per point) |
|---|---|
| `cardiac_snapshots` | `Vm` (mV), `Cai` (mM), `activation_time` (ms; −1 = not activated yet), `substeps_per_step` (cell-model substeps per time step since the previous snapshot: the **cost** of each phase), `celltype` (0 endo, 1 epi, 2 mid), `device`, `rank` |
| `his_snapshots` | the 8 populations (`A_antigen`, `MR_resting_macrophage`, `MA_activated_macrophage`, `N_neutrophil`, `CH_proinflammatory_cytokine`, `ND_apoptotic_neutrophil`, `G_protein_granules`, `CA_antiinflammatory_cytokine`), `device`, `rank` |

`device` is the device that owns each point at that moment: coloring by it
shows the division of the domain following the wave (cardiac) when balancing
is on. Options: size, `--time`/`--iterations`, `--every`, `--output`,
`--precision`, `--device` and the balancing options (`--help` lists them all).
The spacing in the files is in mm (cardiac: `dx`; HIS: 0.1, the `deltaX` of the
kernel).

## Migration profile

```bash
mpirun -np 2 bin/migration_profiler profiling_results.txt
```

Measures the time to migrate growing volumes between neighbouring devices
(OpenCL + MPI) and writes the linear segments used by the profiled policies.
It should be measured with the same nodes as the runs.

## Running with MPI

Start one MPI process per node; each process drives every OpenCL device of its
node (`discover_devices`). Many MPI launchers bind each process to a single
core by default, and an OpenCL CPU device then exposes a single compute unit.
Disable the binding when CPU devices are used, for example with Open MPI:

```bash
mpirun --bind-to none -np 2 ./my_app
```

`DeviceSelection::cpu_host_cores` (`--cpu-host-cores` in the examples) leaves
cores of the CPU device free for the MPI process, which drives the other
devices and the communication.

## Tests

```bash
./examples/run_checks.sh
```

Builds the examples and runs the checks: quickstart, multi_kernel and
cardiac_tissue against the CPU, small HIS runs with balancing (float and
double), stencil2d against the MPI+OpenCL version (float and double) and the
two VTK snapshot programs. `RANKS=4` changes the number of ranks and
`MPIRUN="mpirun --oversubscribe"` the launcher. At least one OpenCL device is
needed; without a GPU, [POCL](https://portablecl.org) provides a CPU device.

Note: POCL 5.0 has an internal bug (`pocl_release_dlhandle_cache: Assertion
'found->ref_count > 0'`) when it exposes more than one device
(`POCL_DEVICES="cpu cpu"`). To emulate several devices per rank, also set
`POCL_KERNEL_CACHE=0`.
