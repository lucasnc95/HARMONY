#!/usr/bin/env bash
# Paper tests of one scenario. Sourced by paper_<scenario>.pbs after
# SCENARIO is set:
#
#   SCENARIO=n2
#   source "$HARMONY_ROOT/scripts/pbs/paper_run.sh"
#   paper_main
#
# Configurations (CONFIGS; default: gpu for gpu1, the other five otherwise):
#   gpu                GPU only, no balancing
#   off                every device, equal division, no balancing
#   static_threshold   one balancing attempt at iteration INTERVAL; applied if
#                      the division changes by at least THRESHOLD
#   static_profiled    one balancing attempt at iteration INTERVAL; applied if
#                      the time saved exceeds the migration cost
#   dynamic_threshold  balancing attempts at every multiple of INTERVAL
#   dynamic_profiled   balancing attempts at every multiple of INTERVAL
# Configurations with balancing also have a warm-up probe at iteration WARMUP.
#
# Variables (qsub -v; lists separated by ':' or spaces):
#   BENCHES      benchmarks              (his:ocean:lu:map:cardiac)
#   PRECISIONS   precisions              (double)
#   CONFIGS      configurations          (see above)
#   RUNS         runs per configuration  (10)
#   WARMUP       warm-up probe iteration (10)
#   INTERVAL     balancing interval      (10 % of the iterations)
#   THRESHOLD    threshold strategies    (0.01)
#   LENGTH       iterations, or ms for cardiac (paper_cases.sh)
#   CPU_HOST_CORES  cores of each CPU device left to the MPI process (1)
#   HALO_SYNC    overlap | local | global (overlap)
#   OUT_DIR      results directory       (results/paper)
#   MONITOR      1 = record GPU state and node load during each run (1)
#
# Results: $OUT_DIR/<bench>/<precision>/<scenario>/<config>/
#   info.txt            command, nodes and schedule
#   timings.txt         wall time of each run, one per line
#   run_<N>.log         HARMONY log of run N
#   report_run<N>.csv   timing report of run N (CSV)
#   monitor_run<N>_<host>_gpu.csv, monitor_run<N>_<host>_procs.txt
# Statistics: scripts/pbs/paper_stats.sh $OUT_DIR

: "${SCENARIO:?SCENARIO must be set (gpu1, n1, n2 or n3)}"

# shellcheck disable=SC1091
source "${PAPER_CASES_FILE:-$HARMONY_ROOT/scripts/pbs/paper_cases.sh}"

BENCHES=${BENCHES:-$PAPER_BENCHMARKS}
BENCHES=${BENCHES//:/ }
PRECISIONS=${PRECISIONS:-double}
PRECISIONS=${PRECISIONS//:/ }
if [ "$SCENARIO" = "gpu1" ]; then
    CONFIGS=${CONFIGS:-gpu}
else
    CONFIGS=${CONFIGS:-"off static_threshold static_profiled dynamic_threshold dynamic_profiled"}
fi
CONFIGS=${CONFIGS//:/ }
RUNS=${RUNS:-10}
WARMUP=${WARMUP:-10}
THRESHOLD=${THRESHOLD:-0.01}
CPU_HOST_CORES=${CPU_HOST_CORES:-1}
HALO_SYNC=${HALO_SYNC:-overlap}
OUT_DIR=${OUT_DIR:-results/paper}
MONITOR=${MONITOR:-1}
WRAPPER="$HARMONY_ROOT/scripts/pbs/rank_wrapper.sh"
PROFILE="$OUT_DIR/profiles/${SCENARIO}.txt"

# paper_config_args <config> <warmup> <interval>: device and balancing options
paper_config_args() {
    local schedule="--balance-warmup $2 --rebalance-interval $3"
    local all="--device all --cpu-host-cores $CPU_HOST_CORES"
    case "$1" in
        gpu)               echo "--device gpu --balance-mode off" ;;
        off)               echo "$all --balance-mode off" ;;
        static_threshold)  echo "$all --balance-mode static --balance-strategy threshold --rebalance-threshold $THRESHOLD $schedule" ;;
        static_profiled)   echo "$all --balance-mode static --balance-strategy profiled $schedule" ;;
        dynamic_threshold) echo "$all --balance-mode dynamic --balance-strategy threshold --rebalance-threshold $THRESHOLD $schedule" ;;
        dynamic_profiled)  echo "$all --balance-mode dynamic --balance-strategy profiled $schedule" ;;
        *) echo "unknown configuration '$1'" >&2; return 1 ;;
    esac
}

# paper_prepare: GPU check of every rank and migration profile of the scenario.
paper_prepare() {
    echo "GPU check:"
    if ! run_mpi "$WRAPPER" --check-gpus -- true < /dev/null; then
        echo "error: GPU check failed; no test was run" >&2
        exit 3
    fi

    mkdir -p "$OUT_DIR/profiles"
    if [ ! -s "$PROFILE" ]; then
        echo "Migration profile: $PROFILE"
        run_mpi "$WRAPPER" -- "$HARMONY_ROOT/bin/migration_profiler" "$PROFILE" \
            < /dev/null > "$OUT_DIR/profiles/${SCENARIO}.log" 2>&1 \
            || echo "warning: migration profiler failed (see $OUT_DIR/profiles/${SCENARIO}.log)" >&2
    fi
}

# paper_run <benchmark> <precision>: every configuration, RUNS runs each.
paper_run() {
    local bench=$1 precision=$2
    local app problem length length_option iterations interval
    app=$(paper_app "$bench")
    problem=$(paper_problem "$bench" "$precision")
    length=${LENGTH:-$(paper_length "$bench" "$precision")}
    length_option=$(paper_length_option "$bench" "$length")
    iterations=$(paper_iterations "$bench" "$length")
    interval=${INTERVAL:-$(( iterations / 10 > 0 ? iterations / 10 : 1 ))}

    echo ""
    echo "$bench ($precision), scenario $SCENARIO: $iterations iterations, warm-up $WARMUP, interval $interval"

    local config args dir run status
    for config in $CONFIGS; do
        args=$(paper_config_args "$config" "$WARMUP" "$interval")
        dir="$OUT_DIR/$bench/$precision/$SCENARIO/$config"
        mkdir -p "$dir"
        : > "$dir/timings.txt"
        {
            echo "benchmark=$bench precision=$precision scenario=$SCENARIO config=$config"
            echo "nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
            echo "iterations=$iterations warmup=$WARMUP interval=$interval threshold=$THRESHOLD runs=$RUNS"
            echo "command=$app $problem $length_option --precision $precision $args --halo-sync $HALO_SYNC"
            echo "started=$(date '+%F %T')"
        } > "$dir/info.txt"

        echo "  $config"
        for run in $(seq 1 "$RUNS"); do
            local monitor_args=()
            [ "$MONITOR" = "1" ] && monitor_args=("--monitor=$dir/monitor_run${run}")
            # < /dev/null: the launcher would otherwise read this script's input.
            # shellcheck disable=SC2086
            if run_mpi "$WRAPPER" ${monitor_args[@]+"${monitor_args[@]}"} -- \
                "$app" $problem $length_option --precision "$precision" $args --halo-sync "$HALO_SYNC" \
                --profiling-file "$PROFILE" --timing-file "$dir/timings.txt" \
                --report-csv "$dir/report_run${run}.csv" \
                < /dev/null > "$dir/run_${run}.log" 2>&1; then
                status="$(tail -n 1 "$dir/timings.txt") s"
            else
                status="FAILED (see $dir/run_${run}.log)"
            fi
            echo "    run $run: $status"
        done
        echo "finished=$(date '+%F %T')" >> "$dir/info.txt"
    done
}

# paper_main: every benchmark and precision of the scenario.
paper_main() {
    paper_prepare
    local bench precision
    for bench in $BENCHES; do
        for precision in $PRECISIONS; do
            paper_run "$bench" "$precision"
        done
    done
}
