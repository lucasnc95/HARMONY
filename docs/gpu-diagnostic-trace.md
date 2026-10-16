# Rank-local GPU variance diagnostic traces

Pass `--diagnostics-prefix PREFIX` to either benchmark for a diagnostic run. Each MPI rank exclusively creates `PREFIX.rank<RANK>.csv`; an existing file causes every rank to stop before the timed loop. All ranks must use the same prefix. The trace is buffered in memory and written after validation. Keep diagnostic runs separate from the unprofiled timing matrix because recording itself can change timing.

The trace schema is version 1. Its columns are `schema_version,rank,seq,iteration,seed,algorithm,record_type,phase,part,device_global,owner_rank,local_index,offset,count,duration_s,device_event_s,boundary_shift_elements_estimate,transfer_bytes_measured,device_name,device_kind,device_identity`. `seq` starts at zero and increases for every row. Iterations are zero based; inventory rows use `-1`, and post-loop rows use the configured iteration count. Partition rows use phase `old` or `new` and include zero-count partitions. Their `boundary_shift_elements_estimate` is an absolute offset change in elements, not measured migration bytes. `transfer_bytes_measured` remains blank until the runtime exposes that measurement.

Host `phase` rows contain rank-local `duration_s`: `rebalance_plan`, `rebalance_apply`, `runtime_execute` (or `cpu_simulation`), `mpi_step_max`, `mpi_device_times`, `loop_total`, `gather`, `reference`, and `mpi_validation`. The `runtime_execute` interval may include synchronization and communication inside the runtime. Device event rows contain `device_event_s` and may overlap host intervals; they must not be added to host totals. `loop_total` overlaps the other per-iteration host phases and is used as a reconciliation reference, not an additive phase. Rank clocks are not synchronized across hosts.

Validate traces before interpreting them:

```bash
python3 benchmarks/validate_trace.py /path/to/trace \
  --ranks 2 --iterations 100 --elements 1000000 --balance-interval 10 \
  --summary-json /path/to/trace-summary.json
```

The validator checks schema, sequence, exactly one execution and compatible MPI progress at every iteration, scheduled rebalance phases, old/new partition conservation and continuity across ranks, device inventory consistency, nonnegative finite durations, and final validation phases. Summary JSON version 1 contains `iterations`, `elements`, `seed`, `algorithm`, and a `ranks` array. Each rank entry contains `rank`, source `trace` path, `host_phase_totals_s` by phase, `device_event_total_s`, and `partition_rows`. Treat the per-rank phase totals as local observations; the benchmark's `loop_time_s` is a maximum over ranks, so summing maxima from different ranks is invalid.

## Pending cluster validation

On the frontend, compile both drivers with the README C++20, MPI, and OpenCL 3.0 commands. Under PBS, run one small-input diagnostic with 100 iterations, two ranks, and one verifiable physical GPU per rank. Supply `--require-two-rank-gpu`, `--device-inventory-json`, and `--diagnostics-prefix`; retain inventory, both traces, benchmark CSV, and logs. Run the validator above with the actual input size. Check both ranks' physical identities, numerical `validation=PASS`, and complete trace progress. On the same allocation, run the identical condition without diagnostics to estimate trace overhead; record both `loop_time_s` values. A CPU simulation verifies trace formatting and MPI control flow only. The GPU and overhead checks remain unvalidated until cluster results are supplied.
