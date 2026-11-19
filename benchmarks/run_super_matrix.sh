#!/usr/bin/env bash
# HARMONY Non-Invasive PBS Super Job Matrix Runner
# Executes Preflight Verification, untimed warm-up, 5-repetition Rotated Matrix,
# seed-123 A/B/A Test, and multi-node GPU telemetry in an isolated output directory.
# Strictly enforces prebuilt frontend binaries and forbids compute-node compilation.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"

bin_dir="${BENCH_BIN_DIR:-${root_dir}/bin}"
test_bin_dir="${BENCH_TEST_BIN_DIR:-}"
output_dir="${BENCH_OUTPUT_DIR:-${root_dir}/benchmarks/super_matrix_results}"
bench_n="${BENCH_N:-1000000}"
bench_iterations="${BENCH_ITERATIONS:-100}"
bench_interval="${BENCH_BALANCE_INTERVAL:-10}"
bench_seeds="${BENCH_SEEDS:-42 123 456}"
bench_timeout="${SAMPLE_TIMEOUT_SECONDS:-180}"
bench_device="${BENCH_DEVICE:-}"
dry_run="${DRY_RUN:-0}"
skip_preflight="${BENCH_SKIP_PREFLIGHT:-0}"
mpi_launcher="${MPI_LAUNCHER:-}"

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

HARMONY Non-Invasive PBS Super Job Matrix Runner.

Options:
  --bin-dir <PATH>         Path to prebuilt cluster binaries (default: \$BENCH_BIN_DIR or bin/)
  --test-bin-dir <PATH>    Path to prebuilt test binaries (default: tests/bin/ when using default bin/)
  --output-dir <PATH>      Directory to store logs, CSVs, and telemetry (default: \$BENCH_OUTPUT_DIR)
  --n <INT>                Global problem size N (default: 1000000)
  --iterations <INT>       Number of iterations per benchmark run (default: 100)
  --balance-interval <INT> Interval between rebalance steps (default: 10)
  --seeds <LIST>           Seeds to evaluate, space or comma separated (default: "42 123 456")
  --timeout <INT>          Timeout per sample in seconds (default: 180 or \$SAMPLE_TIMEOUT_SECONDS)
  --device <MODE>          Optional device mode override (auto|gpu|cpu|all|sim)
  --launcher <CMD>         MPI launcher command or override (default: prun or \$MPI_LAUNCHER)
  --skip-preflight         Bypass Stage 1 Preflight Verification (topo probe and halo test)
  --dry-run                Validate configuration and prebuilt binaries, display plan, and exit
  -h, --help               Display this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin-dir) bin_dir="$2"; shift 2 ;;
        --test-bin-dir) test_bin_dir="$2"; shift 2 ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        --n) bench_n="$2"; shift 2 ;;
        --iterations) bench_iterations="$2"; shift 2 ;;
        --balance-interval) bench_interval="$2"; shift 2 ;;
        --seeds) bench_seeds="$2"; shift 2 ;;
        --timeout) bench_timeout="$2"; shift 2 ;;
        --device) bench_device="$2"; shift 2 ;;
        --launcher) mpi_launcher="$2"; shift 2 ;;
        --skip-preflight) skip_preflight=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) show_usage; exit 0 ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

for pair in "n:$bench_n" "iterations:$bench_iterations" "balance-interval:$bench_interval"; do
    name="${pair%%:*}"
    val="${pair#*:}"
    if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --${name} must be a positive integer, got: '$val'" >&2
        exit 1
    fi
done

if ! [[ "$bench_timeout" =~ ^[1-9][0-9]*s?$ ]]; then
    echo "Error: --timeout must be a positive integer or duration (e.g. 180 or 180s), got: '$bench_timeout'" >&2
    exit 1
fi

cleaned_seeds="${bench_seeds//,/ }"
read -r -a seeds_array <<< "$cleaned_seeds"
if [[ ${#seeds_array[@]} -eq 0 ]]; then
    echo "Error: --seeds must not be empty" >&2
    exit 1
fi
for s in "${seeds_array[@]}"; do
    if ! [[ "$s" =~ ^[0-9]+$ ]]; then
        echo "Error: Invalid seed '$s'. Seeds must be non-negative integers." >&2
        exit 1
    fi
done

mkdir -p "$output_dir" || {
    echo "Error: Failed to create output directory: $output_dir" >&2
    exit 1
}

if [[ ! -d "$bin_dir" ]]; then
    echo "Error: Missing binary directory: $bin_dir" >&2
    exit 1
fi

# Resolve test_async_halo_opencl location: explicit --test-bin-dir, else bin_dir, else tests/bin when using repo default bin/.
if [[ -n "$test_bin_dir" ]]; then
    halo_bin="${test_bin_dir}/test_async_halo_opencl"
elif [[ -x "${bin_dir}/test_async_halo_opencl" ]]; then
    halo_bin="${bin_dir}/test_async_halo_opencl"
elif [[ "$bin_dir" == "${root_dir}/bin" && -x "${root_dir}/tests/bin/test_async_halo_opencl" ]]; then
    halo_bin="${root_dir}/tests/bin/test_async_halo_opencl"
else
    halo_bin="${bin_dir}/test_async_halo_opencl"
fi

for binary_path in "${bin_dir}/kneighbor" "${bin_dir}/leanmd" "${bin_dir}/topo_probe" "$halo_bin"; do
    if [[ ! -x "$binary_path" ]]; then
        echo "Error: Missing prebuilt binary: $binary_path" >&2
        echo "Compiling on compute nodes is strictly forbidden. Build on frontend using: ./build.sh all" >&2
        exit 1
    fi
done

if [[ "$dry_run" == 1 ]]; then
    dryrun_plan="[DRY-RUN] HARMONY Non-Invasive PBS Super Job Execution Plan
=====================================================================
Binary Directory:    $bin_dir
Halo Test Binary:    $halo_bin
Output Directory:    $output_dir
Parameters:          N=$bench_n, iterations=$bench_iterations, interval=$bench_interval
Seeds:               ${seeds_array[*]}
Launcher:            ${mpi_launcher:-prun}

Stage 1: Preflight Verification
  - Topo probe: $bin_dir/topo_probe --output-json $output_dir/topo_metrics_cluster.json
  - Halo test:  $halo_bin --require-gpu

Stage 2: Untimed Warm-up Phase
  - kneighbor (erad, hwtopolb)
  - leanmd (erad, hwtopolb)

Stage 3: Rotated Matrix Execution
  - Workloads:   kneighbor, leanmd
  - Algorithms:  erad, hwtopolb
  - Seeds:       ${seeds_array[*]}
  - Repetitions: 5 (alternating ERAD/HWTOPOLB order with rotated seed sequences)

Stage 4: A/B/A Test Isolated Verification (Seed 123)
  - ERAD -> HWTOPOLB -> ERAD for seed 123

Stage 5: Post-Processing & Telemetry
  - Consolidated summary, paired ratios, and SUPER_JOB_REPORT.md
=====================================================================
[DRY-RUN] Validation completed successfully. No workloads executed."

    echo "$dryrun_plan"
    echo "$dryrun_plan" > "$output_dir/dryrun_plan.txt"
    exit 0
fi

if [[ "$mpi_launcher" == "none" ]]; then
    launcher_cmd=()
elif [[ -n "$mpi_launcher" ]]; then
    read -r -a launcher_cmd <<< "$mpi_launcher"
elif command -v prun >/dev/null 2>&1; then
    launcher_cmd=("prun")
elif command -v mpirun >/dev/null 2>&1; then
    launcher_cmd=("mpirun")
else
    launcher_cmd=()
fi

rank_wrapper="${root_dir}/scripts/pbs/rank_wrapper.sh"
mpi_exec_cmd=("${launcher_cmd[@]}")
if [[ ${#launcher_cmd[@]} -gt 0 && -n "${GPU_PIN:-}" && -x "$rank_wrapper" ]]; then
    mpi_exec_cmd+=("$rank_wrapper" --gpu-pin "$GPU_PIN" --)
fi

if [[ "$bench_timeout" =~ ^[0-9]+$ ]]; then
    sample_timeout_str="${bench_timeout}s"
else
    sample_timeout_str="$bench_timeout"
fi

query_local_nvidia_smi() {
    local target_host="$1"
    local target_rank="$2"
    local telem_file="$3"

    command -v nvidia-smi >/dev/null 2>&1 || return 1
    local smi_out=""
    smi_out=$(nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null) || return 1
    [[ -n "$smi_out" ]] || return 1

    while IFS= read -r line; do
        [[ -n "$line" ]] && echo "${target_host}, ${target_rank}, ${line}" >> "$telem_file"
    done <<< "$smi_out"

    local apps_out=""
    apps_out=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    local apps_found=0
    if [[ -n "$apps_out" ]]; then
        while IFS= read -r app_line; do
            if [[ -n "$app_line" ]] && [[ "$app_line" != *"NVIDIA"* ]]; then
                echo "# [compute-apps] host=${target_host} rank=${target_rank} ${app_line}" >> "$telem_file"
                apps_found=1
            fi
        done <<< "$apps_out"
    fi
    if [[ $apps_found -eq 0 ]]; then
        echo "# [compute-apps] host=${target_host} rank=${target_rank} shared_workload=none" >> "$telem_file"
    fi
    return 0
}

record_telemetry() {
    local block_name="$1"
    local telem_file="${output_dir}/hardware_telemetry.log"
    local now
    now="$(date -u +"%Y-%m-%d %H:%M:%S" 2>/dev/null || date +"%Y-%m-%d %H:%M:%S")"
    echo "# [telemetry] block=${block_name} timestamp=${now}" >> "$telem_file"

    local probed=0

    # 1. Query telemetry across all participating ranks/nodes via configured launcher
    if [[ ${#launcher_cmd[@]} -gt 0 ]]; then
        local probe_script="${script_dir}/telemetry_probe.sh"
        local probe_out=""
        if probe_out=$("${launcher_cmd[@]}" "$probe_script" 2>/dev/null) && [[ -n "$probe_out" ]]; then
            if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
                local all_found=1
                while IFS= read -r node || [[ -n "$node" ]]; do
                    node="$(echo "$node" | tr -d '[:space:]')"
                    [[ -z "$node" ]] && continue
                    local node_clean="${node%\[*\]}"
                    if ! echo "$probe_out" | grep -Fq "$node_clean" && ! echo "$probe_out" | grep -Fq "$node"; then
                        all_found=0
                        break
                    fi
                done < "$PBS_NODEFILE"
                if [[ $all_found -eq 1 ]]; then
                    echo "$probe_out" >> "$telem_file"
                    probed=1
                fi
            else
                echo "$probe_out" >> "$telem_file"
                probed=1
            fi
        fi
    fi

    # 2. PBS_NODEFILE fallback: query only locally-reachable hosts; mark remote ranks unavailable.
    if [[ $probed -eq 0 && -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
        local rank=0
        local local_host
        local_host="$(hostname -s 2>/dev/null || hostname)"
        while IFS= read -r node || [[ -n "$node" ]]; do
            node="$(echo "$node" | tr -d '[:space:]')"
            [[ -z "$node" ]] && continue
            local node_clean="${node%\[*\]}"
            local node_short="${node_clean%%.*}"
            local node_probed=0
            local unavailable_reason="no ssh-free path to query nvidia-smi on a non-local node (launcher telemetry probe failed or did not cover it)"
            if [[ $rank -eq 0 || "$node_short" == "$local_host" ]]; then
                if query_local_nvidia_smi "$node_clean" "$rank" "$telem_file"; then
                    node_probed=1
                elif command -v nvidia-smi >/dev/null 2>&1; then
                    unavailable_reason="nvidia-smi query failed on the local node"
                else
                    unavailable_reason="nvidia-smi not available on the local node"
                fi
            fi
            if [[ $node_probed -eq 0 ]]; then
                echo "# [telemetry-unavailable] host=${node_clean} rank=${rank} block=${block_name} reason=${unavailable_reason}" >> "$telem_file"
            fi
            probed=1
            rank=$((rank + 1))
        done < "$PBS_NODEFILE"
    fi

    # 3. Local host fallback
    if [[ $probed -eq 0 ]]; then
        local host
        host="$(hostname -s 2>/dev/null || hostname)"
        if ! query_local_nvidia_smi "$host" "0" "$telem_file"; then
            local msg="nvidia-smi not available"
            command -v nvidia-smi >/dev/null 2>&1 && msg="nvidia-smi query failed"
            echo "${host}, 0, ${now}, placeholder, N/A, N/A, N/A, N/A, N/A, [N/A] (${msg} for block ${block_name})" >> "$telem_file"
            echo "# [compute-apps] host=${host} rank=0 shared_workload=none" >> "$telem_file"
        fi
    fi
}

init_manifest() {
    python3 - "$output_dir/run_manifest.json" "$bench_n" "$bench_iterations" "$bench_interval" "${seeds_array[*]}" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
data = {
    "parameters": {
        "n": int(sys.argv[2]),
        "iterations": int(sys.argv[3]),
        "balance_interval": int(sys.argv[4]),
        "seeds": [int(s) for s in sys.argv[5].split() if s],
    },
    "runs": [],
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
}

record_sample_manifest() {
    local run_id="$1" stage="$2" rep="$3" bench="$4" algo="$5" seed="$6" step="$7"
    local cmd_str="$8" csv_rel="$9" report_csv_rel="${10}" log_rel="${11}"
    local exit_code="${12}" elapsed="${13}" status="${14}"

    python3 - "$output_dir/run_manifest.json" \
        "$run_id" "$stage" "$rep" "$bench" "$algo" "$seed" "$step" \
        "$cmd_str" "$csv_rel" "$report_csv_rel" "$log_rel" "$exit_code" "$elapsed" "$status" <<'PY'
import json
import os
import sys

manifest_path = sys.argv[1]
entry = {
    "run_id": int(sys.argv[2]),
    "stage": sys.argv[3],
    "repetition": int(sys.argv[4]) if sys.argv[4] not in ("null", "None", "") else None,
    "benchmark": sys.argv[5],
    "algorithm": sys.argv[6],
    "algo": sys.argv[6],
    "seed": int(sys.argv[7]),
    "step": int(sys.argv[8]) if sys.argv[8] not in ("null", "None", "") else None,
    "command": sys.argv[9],
    "csv_file": sys.argv[10],
    "report_csv_file": sys.argv[11],
    "log_file": sys.argv[12],
    "exit_code": int(sys.argv[13]),
    "elapsed_seconds": float(sys.argv[14]),
    "status": sys.argv[15],
}

data = {"runs": []}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
data.setdefault("runs", []).append(entry)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
PY
}

get_rep_seeds() {
    local rep="$1"
    if [[ ${#seeds_array[@]} -eq 3 ]]; then
        local s0="${seeds_array[0]}" s1="${seeds_array[1]}" s2="${seeds_array[2]}"
        case "$(( (rep - 1) % 3 ))" in
            0) echo "$s0 $s1 $s2" ;;
            1) echo "$s1 $s2 $s0" ;;
            2) echo "$s2 $s0 $s1" ;;
        esac
    else
        echo "${seeds_array[*]}"
    fi
}

get_rep_algos() {
    local rep="$1"
    if (( rep % 2 == 1 )); then
        echo "erad hwtopolb"
    else
        echo "hwtopolb erad"
    fi
}

run_counter=0
failed_samples=0
warmup_failed_samples=0

execute_sample() {
    local stage="$1" rep="$2" bench="$3" algo="$4" seed="$5" step="$6"

    run_counter=$((run_counter + 1))
    local run_id="$run_counter"
    local binary="$bin_dir/${bench}"

    local stem
    if [[ "$stage" == "warmup" ]]; then
        stem="warmup_${bench}_${algo}_${seed}"
    elif [[ "$stage" == "matrix" ]]; then
        stem="${bench}_${algo}_${seed}_rep${rep}"
    elif [[ "$stage" == "aba" ]]; then
        stem="${bench}_${algo}_${seed}_aba${step}"
    else
        stem="${bench}_${algo}_${seed}"
    fi

    local csv_basename="${stem}.csv"
    local report_csv_basename="${stem}_report.csv"
    local log_basename="${stem}.log"
    local csv_file="${output_dir}/${csv_basename}"
    local report_csv_file="${output_dir}/${report_csv_basename}"
    local log_file="${output_dir}/${log_basename}"

    local cmd=("${mpi_exec_cmd[@]}" "$binary" --n "$bench_n" --iterations "$bench_iterations"
        --balance-interval "$bench_interval" --balance-algo "$algo"
        --perturbation 0.10 --seed "$seed"
        --output-csv "$csv_file" --report-csv "$report_csv_file")
    if [[ -n "$bench_device" ]]; then
        cmd+=(--device "$bench_device")
    fi

    echo "[Sample $run_id] stage=$stage rep=$rep bench=$bench algo=$algo seed=$seed step=$step"

    local start_time end_time elapsed exit_code
    start_time=$(python3 -c 'import time; print(time.time())')

    set +e
    timeout "$sample_timeout_str" "${cmd[@]}" > "$log_file" 2>&1
    exit_code=$?
    set -e

    end_time=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(round($end_time - $start_time, 4))")

    local status="success"
    if [[ $exit_code -eq 124 ]]; then
        status="timeout"
        if [[ "$stage" == "warmup" ]]; then
            warmup_failed_samples=$((warmup_failed_samples + 1))
            echo "[WARNING] Warm-up sample $run_id timed out after $sample_timeout_str: $bench $algo seed $seed (warmup failure recorded in manifest; does not count toward runner exit code)" >&2
        else
            failed_samples=$((failed_samples + 1))
            echo "[WARNING] Sample $run_id timed out after $sample_timeout_str: $bench $algo seed $seed" >&2
        fi
    elif [[ $exit_code -ne 0 ]]; then
        status="failed"
        if [[ "$stage" == "warmup" ]]; then
            warmup_failed_samples=$((warmup_failed_samples + 1))
            echo "[WARNING] Warm-up sample $run_id failed with exit code $exit_code: $bench $algo seed $seed (warmup failure recorded in manifest; does not count toward runner exit code)" >&2
        else
            failed_samples=$((failed_samples + 1))
            echo "[WARNING] Sample $run_id failed with exit code $exit_code: $bench $algo seed $seed" >&2
        fi
    fi

    record_sample_manifest "$run_id" "$stage" "$rep" "$bench" "$algo" "$seed" "$step" \
        "${cmd[*]}" "$csv_basename" "$report_csv_basename" "$log_basename" "$exit_code" "$elapsed" "$status"
}

# Stage 1: Preflight Verification
if [[ "$skip_preflight" -eq 1 ]]; then
    echo "=== Stage 1: Preflight Verification (SKIPPED via --skip-preflight) ==="
else
    echo "=== Stage 1: Preflight Verification ==="
    preflight_budget="${PREFLIGHT_BUDGET_SECONDS:-60}"
    preflight_deadline=$((SECONDS + preflight_budget))
    preflight_remaining() {
        local remaining=$((preflight_deadline - SECONDS))
        if [[ $remaining -lt 1 ]]; then
            echo "Preflight failed: budget of ${preflight_budget}s exhausted before $1" >&2
            exit 1
        fi
        echo "$remaining"
    }

    topo_stdout="$output_dir/topo_probe.stdout"
    remaining=$(preflight_remaining "topology probe")
    echo "Running topology probe (shared preflight budget: ${remaining}s of ${preflight_budget}s left): $bin_dir/topo_probe"
    timeout "${remaining}s" "${mpi_exec_cmd[@]}" "$bin_dir/topo_probe" --output-json "$output_dir/topo_metrics_cluster.json" | tee "$topo_stdout"

    python3 - "$output_dir/topo_metrics_cluster.json" "$topo_stdout" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        metrics = json.load(f)
except Exception as err:
    sys.stderr.write(f"Preflight failed: could not parse topology JSON: {err}\n")
    sys.exit(1)

if metrics.get("synthetic", True):
    sys.stderr.write("Preflight failed: topology probe reported synthetic metrics\n")
    sys.exit(1)

if len(metrics.get("pcie_latency_ns", [])) < 2:
    sys.stderr.write("Preflight failed: expected at least two measured OpenCL GPUs\n")
    sys.exit(1)

counts = None
with open(sys.argv[2], encoding="utf-8") as stdout_log:
    for line in stdout_log:
        if line.startswith("rank_device_counts="):
            try:
                counts = [int(v) for v in line.split("=", 1)[1].strip().split(",")]
            except ValueError:
                pass
if counts is None:
    sys.stderr.write("Preflight failed: topo_probe did not report rank_device_counts; rebuild it with ./build.sh all\n")
    sys.exit(1)
if len(counts) < 2 or any(c < 1 for c in counts):
    sys.stderr.write(f"Preflight failed: per-rank OpenCL GPU counts must be >= 1 on every rank (at least 2 ranks), got {counts}\n")
    sys.exit(1)
if sum(counts) != len(metrics["pcie_latency_ns"]):
    sys.stderr.write(f"Preflight failed: per-rank GPU counts {counts} disagree with {len(metrics['pcie_latency_ns'])} measured devices\n")
    sys.exit(1)
PY

    remaining=$(preflight_remaining "asynchronous halo test")
    echo "Running asynchronous OpenCL halo test (shared preflight budget: ${remaining}s of ${preflight_budget}s left): $halo_bin"
    if ! timeout "${remaining}s" "${mpi_exec_cmd[@]}" "$halo_bin" --require-gpu --timeout-seconds "$remaining"; then
        echo "ERROR: Asynchronous halo test preflight failed (requires at least 1 OpenCL GPU per rank)" >&2
        exit 1
    fi
fi

init_manifest

# Phase 1: Untimed Warm-up Phase
echo "=== Phase 1: Untimed Warm-up Phase ==="
record_telemetry "before_warmup"
default_seed="${seeds_array[0]}"
for bench in kneighbor leanmd; do
    for algo in erad hwtopolb; do
        execute_sample "warmup" "null" "$bench" "$algo" "$default_seed" "null"
    done
done
record_telemetry "after_warmup"

# Phase 2: Rotated Matrix Execution (5 repetitions)
echo "=== Phase 2: Rotated Matrix Execution ==="
for rep in 1 2 3 4 5; do
    echo "--- Repetition $rep / 5 ---"
    record_telemetry "before_rep_${rep}"
    read -r -a rep_algos <<< "$(get_rep_algos "$rep")"
    read -r -a rep_seeds <<< "$(get_rep_seeds "$rep")"

    for bench in kneighbor leanmd; do
        for algo in "${rep_algos[@]}"; do
            for seed in "${rep_seeds[@]}"; do
                execute_sample "matrix" "$rep" "$bench" "$algo" "$seed" "null"
            done
        done
    done
    record_telemetry "after_rep_${rep}"
done

# Phase 3: Dedicated A/B/A Test (Seed 123)
echo "=== Phase 3: Dedicated A/B/A Test (Seed 123) ==="
record_telemetry "before_aba"
for bench in kneighbor leanmd; do
    execute_sample "aba" "null" "$bench" "erad" "123" "1"
    execute_sample "aba" "null" "$bench" "hwtopolb" "123" "2"
    execute_sample "aba" "null" "$bench" "erad" "123" "3"
done
record_telemetry "after_aba"

if [[ "$warmup_failed_samples" -gt 0 ]]; then
    echo "[WARNING] $warmup_failed_samples warm-up sample(s) failed or timed out (recorded in manifest; does not count toward runner exit code)" >&2
fi

if [[ "$failed_samples" -gt 0 ]]; then
    echo "ERROR: $failed_samples sample(s) failed or timed out during matrix execution" >&2
    exit 1
fi

echo "=== Super Matrix execution completed successfully ==="
exit 0
