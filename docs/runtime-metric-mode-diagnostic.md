# Runtime metric mode diagnostic

Issue #5 introduces `set_verbose_metrics(bool)` and the benchmark flag
`--quiet-runtime-metrics`. The control suppresses detailed runtime text and
the optional metrics CSV, while preserving device timing collection and the
same MPI reductions. Existing callers remain verbose by default.

## Local validation

- Build: GCC 13.3.0, `mpic++ -std=c++20 -Wall -Wextra -O2`, OpenCL 3.0 headers,
  `-lOpenCL`; Open MPI 4.1.6.
- Host: `luiz`, AMD EPYC 7763 virtual CPU, two online CPUs, no visible OpenCL
  GPU. Two ranks were oversubscribed on this host.
- The focused public API test passed with one and two MPI ranks. It compared
  the complete simulated partition trajectory under identical timing inputs,
  checked conservation at every step, and confirmed verbose output is present
  by default and absent in quiet mode. Both benchmark drivers compiled and
  accepted the quiet flag in a small CPU fallback smoke run.
- The test uses simulated device times. It does not validate OpenCL event
  timing, device kernels, or performance on a two-GPU allocation. Those checks
  belong in the cluster run.

## Output overhead probe

The focused test's `--measure` option runs 1,000 identical simulated
rebalance iterations in each mode with two MPI ranks. It captures rank-local
stdout in memory so terminal speed does not dominate. Three repeated runs
gave the following rank-0 host times in seconds:

| Run | Verbose | Quiet |
| --- | ---: | ---: |
| 1 | 0.0448872 | 0.0205137 |
| 2 | 0.0361271 | 0.0160481 |
| 3 | 0.1492920 | 0.0147056 |

The third verbose sample shows substantial local variation. These values
measure the diagnostic path on this oversubscribed CPU host; they are not an
algorithm speedup estimate or evidence about the observed 20 ms GPU steps.
Repeat the A/B measurement inside the same two-GPU allocation before making
any GPU performance claim.
