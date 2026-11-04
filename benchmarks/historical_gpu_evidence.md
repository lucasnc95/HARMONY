# Historical two-GPU comparison: validated evidence

The original 12 CSVs, 12 logs, comparison summary, recovered metadata, topology record, and PBS output were copied on 2026-10-13 to `Results/2026-10-13-historical-gpu-comparison/`. The 28 source files remain unchanged. SHA-256 values for every copy are in `historical_archive_manifest.json`. The original PBS job ID, `183164.frontend`, is recoverable from the `prun` hostfile path in the preserved PBS output and per-condition logs. This is a recovered identifier, not a fresh scheduler query. The `Results/` tree is ignored by Git.

Run the read-only validator from the repository root:

```sh
python3 benchmarks/analyze_variability.py \
  --run-dir Results/2026-10-13-historical-gpu-comparison \
  --output-dir Results/2026-10-13-historical-gpu-analysis \
  --workloads kneighbor leanmd --algorithms erad hwtopolb \
  --seeds 42 123 456 --iterations 100
```

It accepted exactly 12 `gpu_opencl` conditions and 1,200 ordered iterations. The output consists of `per_condition.csv` and `per_iteration.csv`. Use a fresh output directory on each invocation; existing output is never overwritten. The expected matrix and iteration count are explicit so short pilot runs can use the same command. The original files are read, never edited.

| Workload | Algorithm | Seed | Wall (s) | Ordinary steps (s) | Rebalance steps (s) | First step (ms) | Last step (ms) | Unaccounted (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-neighbor | ERAD | 42 | 3.740 | 1.830 | 0.911 | 20.141 | 20.138 | 0.999 |
| k-neighbor | ERAD | 123 | 3.580 | 1.830 | 0.751 | 20.121 | 20.123 | 0.999 |
| k-neighbor | ERAD | 456 | 3.880 | 1.829 | 1.060 | 20.171 | 20.073 | 0.990 |
| k-neighbor | HWTOPOLB | 42 | 0.221 | 0.040 | 0.122 | 1.560 | 0.412 | 0.059 |
| k-neighbor | HWTOPOLB | 123 | 3.820 | 1.830 | 0.971 | 20.121 | 20.080 | 1.019 |
| k-neighbor | HWTOPOLB | 456 | 0.214 | 0.049 | 0.120 | 0.892 | 0.730 | 0.045 |
| LeanMD | ERAD | 42 | 4.300 | 1.830 | 1.431 | 20.195 | 20.130 | 1.040 |
| LeanMD | ERAD | 123 | 4.330 | 1.829 | 1.471 | 20.146 | 20.077 | 1.030 |
| LeanMD | ERAD | 456 | 4.221 | 1.830 | 1.361 | 20.144 | 20.071 | 1.031 |
| LeanMD | HWTOPOLB | 42 | 0.438 | 0.047 | 0.308 | 1.212 | 0.417 | 0.083 |
| LeanMD | HWTOPOLB | 123 | 4.460 | 1.830 | 1.590 | 20.220 | 20.077 | 1.040 |
| LeanMD | HWTOPOLB | 456 | 0.362 | 0.045 | 0.249 | 1.691 | 0.740 | 0.067 |

The HWTOPOLB seed-123 slow path is visible at iteration 0 in both workloads, before the first scheduled rebalance at iteration 10. Ordinary-step sums are approximately 1.83 s in the slow runs and 0.04–0.05 s in the fast HWTOPOLB runs. The excess is therefore not concentrated in rebalance steps. The `unaccounted_s` column is **wall time minus the sum of recorded step times**. It is not assigned to kernel execution, MPI, validation, or any other phase because this historical run has no measurement that separates those costs. The large remainder near 1 s in each slow run remains unexplained.

These are single historical samples with no warm-up or controlled order. They establish the onset and accounting gap, not a repeatable speedup or a cause for the variance. The topology JSON was measured separately and was not consumed by the benchmark algorithms.

## Reconstructed runtime partition and device timelines

Run the read-only reconstruction against the preserved archive with a fresh output path:

```sh
python3 benchmarks/reconstruct_historical.py \
  --run-dir Results/2026-10-13-historical-gpu-comparison \
  --output-dir Results/2026-10-13-historical-partition-analysis \
  --workloads kneighbor leanmd --algorithms erad hwtopolb \
  --seeds 42 123 456 --iterations 100 --global-count 1000000
```

The command joins the one-based runtime `iteration` to the zero-based CSV iteration. A standalone `DCL loads by partition` record after runtime iteration 10 is the rebalance result for CSV iteration 10; the next runtime metrics block confirms that map. The join rejects missing, duplicate, out-of-range, or ambiguous iteration and boundary records, and checks every old, current, and rebalance-result partition map for exact coverage of the million elements. Zero-count partitions remain rows in the output. The 12 historical pairs yielded 1,200 joined step records, 2,400 per-device event records, 4,800 old/current partition records, and 216 rebalance-result partition records, with nine boundaries per pair. There were no unjoinable records in this archive.

`timeline.csv` gives host step latency and the maximum device event time in **separate columns**. `device_events.csv` holds the event duration for each partition, rank, and device index; `partitions.csv` holds old, current, and rebalance-result offsets and counts; `boundaries.csv` holds boundary displacement; `devices.csv` holds the initial runtime rank/global-index/name declarations. These device indices and identical model names cannot prove that ranks used different physical GPUs. The historical logs contain no UUID, PCI bus ID, or per-rank physical-device fingerprint. Device event durations may overlap host work and cannot be subtracted from host step latency as a phase accounting equation.

The table below uses a **10 ms descriptive threshold** to locate the first slow ordinary step. Median host times are in ms; median maximum device event times for iterations 0–9 are in microseconds. “Later ordinary” excludes the rebalance steps at CSV iterations 10, 20, ..., 90. The boundary column reports the largest offset displacement in elements among the nine recorded rebalance results; it does not measure bytes migrated.

| Workload | Algorithm | Seed | First slow ordinary iteration | Iteration 0 host (ms) | Ordinary 0–9 median (ms) | Later ordinary median (ms) | Device event 0–9 median (µs) | Maximum boundary shift (elements) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k-neighbor | ERAD | 42 | 0 | 20.141 | 20.126 | 20.126 | 11.776 | 627,374 |
| k-neighbor | ERAD | 123 | 0 | 20.121 | 20.117 | 20.119 | 20.992 | 217,670 |
| k-neighbor | ERAD | 456 | 0 | 20.171 | 20.114 | 20.086 | 24.576 | 582,238 |
| k-neighbor | HWTOPOLB | 42 | None | 1.560 | 0.415 | 0.419 | 10.752 | 576,583 |
| k-neighbor | HWTOPOLB | 123 | 0 | 20.121 | 20.120 | 20.118 | 14.336 | 452,483 |
| k-neighbor | HWTOPOLB | 456 | None | 0.892 | 0.415 | 0.548 | 10.240 | 506,581 |
| LeanMD | ERAD | 42 | 0 | 20.195 | 20.110 | 20.108 | 19.968 | 619,663 |
| LeanMD | ERAD | 123 | 0 | 20.146 | 20.120 | 20.085 | 23.552 | 698,294 |
| LeanMD | ERAD | 456 | 0 | 20.144 | 20.124 | 20.106 | 19.968 | 495,153 |
| LeanMD | HWTOPOLB | 42 | None | 1.212 | 0.421 | 0.433 | 18.432 | 505,107 |
| LeanMD | HWTOPOLB | 123 | 0 | 20.220 | 20.073 | 20.117 | 29.184 | 538,649 |
| LeanMD | HWTOPOLB | 456 | None | 1.691 | 0.421 | 0.429 | 18.432 | 637,295 |

The roughly 20 ms path is present at ordinary iteration 0 in every ERAD run and both HWTOPOLB seed-123 runs, while the first scheduled rebalance is at iteration 10. Later partition movement therefore **cannot explain the onset** of the slow path. The historical records do not isolate the host phases inside the step, so they do not establish whether the unexplained host latency comes from MPI waiting, runtime diagnostics, device synchronization, or another cause. Controlled per-rank measurement is required for attribution.
