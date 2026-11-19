# HARMONY Heterogeneous Balancing and Cluster Evaluation

Domain model and ubiquitous language for HARMONY's dynamic load balancing, OpenCL GPU execution, and HPC PBS cluster evaluation.

## Language

**Super Job**:
A single-allocation PBS batch script that executes the complete variance evaluation matrix, hardware telemetry, and post-processing without modifying core C++ source files.
_Avoid_: Orchestrator, multi-stage pipeline, mega job, bot

**Unprofiled Repetition**:
A benchmark run executed without active profilers or invasive tracing hooks, used exclusively for representative wall-clock timing samples.
_Avoid_: Profiled run, traced sample, benchmark pass

**Rotated Matrix**:
An experimental schedule that systematically alternates algorithm execution order and permutes seed sequences across repetitions to decouple seed bias from thermal or run-order effects.
_Avoid_: Random matrix, shuffle run

**A/B/A Test**:
A sequential execution of `ERAD -> HWTOPOLB -> ERAD` under identical seed and warmed hardware conditions to isolate algorithmic perturbation from state drift.
_Avoid_: Sandwich test, back-to-back run, repeat probe

**Preflight Verification**:
A fast check of OpenCL GPU discovery and MPI halo communication executed immediately at job start to fail fast before launching benchmark workloads.
_Avoid_: Smoke test suite, sanity check

**Historical Evidence**:
The immutable raw CSVs, logs, and metadata recorded from the initial cluster run on compute nodes 1-0 and 1-1.
_Avoid_: Legacy logs, old runs

**Pre-Cluster Validation**:
An optional verification run in a single-node virtualized GPU environment (such as Google Colab) to validate C++ toolchain compilation, OpenCL runtime discovery, and MPI halo exchange prior to PBS batch submission on the HPC cluster.
_Avoid_: Cloud cluster, Colab benchmark, production run
