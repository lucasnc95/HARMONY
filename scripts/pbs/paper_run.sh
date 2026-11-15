#!/usr/bin/env bash
# Runs the paper tests of one scenario (sourced by paper_gpu1.pbs,
# paper_n1.pbs, paper_n2.pbs, paper_n3.pbs and paper_job.pbs, after
# job_env.sh and paper_cases.sh).
#
#   paper_run <benchmark> <precision>     uses $SCENARIO and the variables below
#
# Every balancing configuration also has a warm-up probe at iteration
# WARMUP (adjusts the equal division early; not a balancing attempt).
# Configurations (CONFIGS, default: all of the scenario):
#   gpu                GPU only, equal shares, no balancing            (gpu1)
#   off                GPU + CPU, equal shares, no balancing           (n1 n2 n3)
#   static_threshold   one attempt at iteration INTERVAL, accepted if the
#                      division changes by >= THRESHOLD (default 0.01)
#   static_profiled    one attempt at iteration INTERVAL, accepted if the time
#                      saved > migration cost (migration profile of the nodes)
#   dynamic_threshold  attempts at INTERVAL, 2*INTERVAL, ...
#   dynamic_profiled   idem, with the cost model
#
# Variables: RUNS (10), RUNS_OFF (runs of "off"; default RUNS), CONFIGS,
# THRESHOLD (0.01), WARMUP (iteration of the warm-up probe, not counted as a
# balancing attempt; default 10), INTERVAL (balancing at its multiples;
# default 10 % of the run), LENGTH (iterations, or ms for cardiac; default from paper_cases.sh),
# CPU_HOST_CORES (1), HALO_SYNC (overlap), GPU_PIN (e.g. "compute-1-0=0"),
# OUT_DIR (results/paper).
#
# Results: $OUT_DIR/<bench>/<precision>/<scenario>/<config>/
#   timings.txt         wall time of each run (one per line)
#   run_<N>.log         full output (devices, balancing log, timing summary)
#   report_run<N>.csv   timing report as CSV (critical-path line, devices, ...)
#   info.txt            what was run
# Statistics of everything: scripts/pbs/paper_stats.sh

: "${SCENARIO:?set SCENARIO (gpu1 n1 n2 n3)}"
RUNS=${RUNS:-10}
RUNS_OFF=${RUNS_OFF:-$RUNS}
THRESHOLD=${THRESHOLD:-0.01}
CPU_HOST_CORES=${CPU_HOST_CORES:-1}
HALO_SYNC=${HALO_SYNC:-overlap}
GPU_PIN=${GPU_PIN:-}
OUT_DIR=${OUT_DIR:-results/paper}
CONFIGS=${CONFIGS:-}
CONFIGS=${CONFIGS//:/ }   # submit_paper_tests.sh sends "off:static_threshold:..."
if [ "$SCENARIO" = "gpu1" ]; then
    CONFIGS=${CONFIGS:-gpu}
else
    CONFIGS=${CONFIGS:-"off static_threshold static_profiled dynamic_threshold dynamic_profiled"}
fi
WRAPPER="$HARMONY_ROOT/scripts/pbs/rank_wrapper.sh"

# Which GPUs every rank sees; stops the job if one of them is a MIG instance
# (e.g. compute-1-0 without vnode=compute-1-0[0]).
echo "GPUs seen by each rank:"
if ! run_mpi "$WRAPPER" --gpu-pin "$GPU_PIN" --check-gpus -- true < /dev/null; then
    echo "error: the GPU check failed (see the lines above); nothing was run." >&2
    exit 3
fi

# One migration profile per scenario (same nodes), measured once.
PROFILE="$OUT_DIR/profiles/${SCENARIO}.txt"
mkdir -p "$OUT_DIR/profiles"
if [ ! -s "$PROFILE" ]; then
    echo "Measuring the migration cost of scenario $SCENARIO -> $PROFILE"
    run_mpi "$WRAPPER" --gpu-pin "$GPU_PIN" -- "$HARMONY_ROOT/bin/migration_profiler" "$PROFILE" \
        < /dev/null > "$OUT_DIR/profiles/${SCENARIO}.log" 2>&1 \
        || echo "warning: migration profiler failed (see $OUT_DIR/profiles/${SCENARIO}.log)"
fi

paper_run() {
    local bench=$1 precision=$2
    local app problem length iterations interval warmup
    app=$(paper_app "$bench")
    problem=$(paper_problem "$bench" "$precision")
    length=${LENGTH:-$(paper_length "$bench" "$precision")}
    iterations=$(paper_iterations "$bench" "$length")
    interval=${INTERVAL:-$(( iterations / 10 > 0 ? iterations / 10 : 1 ))}
    warmup=${WARMUP:-10}

    # config_args <config>: device and balancing options
    config_args() {
        local policy="--balance-warmup $warmup --rebalance-interval $interval"
        local all="--device all --cpu-host-cores $CPU_HOST_CORES"
        case "$1" in
            gpu)               echo "--device gpu --balance-mode off" ;;
            off)               echo "$all --balance-mode off" ;;
            static_threshold)  echo "$all --balance-mode static --balance-strategy threshold --rebalance-threshold $THRESHOLD $policy" ;;
            static_profiled)   echo "$all --balance-mode static --balance-strategy profiled $policy" ;;
            dynamic_threshold) echo "$all --balance-mode dynamic --balance-strategy threshold --rebalance-threshold $THRESHOLD $policy" ;;
            dynamic_profiled)  echo "$all --balance-mode dynamic --balance-strategy profiled $policy" ;;
            *) echo "unknown config '$1'" >&2; return 1 ;;
        esac
    }

    echo ""
    echo "Benchmark $bench ($precision), scenario $SCENARIO on $NODES node(s) - $(date '+%F %T')"
    echo "  problem: $problem $(paper_length_option "$bench" "$length")"
    echo "  iterations $iterations, warm-up $warmup, interval $interval, threshold $THRESHOLD"

    local config args dir timing runs run
    for config in $CONFIGS; do
        args=$(config_args "$config")
        dir="$OUT_DIR/$bench/$precision/$SCENARIO/$config"
        mkdir -p "$dir"
        timing="$dir/timings.txt"
        : > "$timing"
        runs=$RUNS
        [ "$config" = "off" ] && runs=$RUNS_OFF
        {
            echo "benchmark=$bench precision=$precision scenario=$SCENARIO config=$config"
            echo "nodes=$(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
            echo "iterations=$iterations warmup=$warmup interval=$interval threshold=$THRESHOLD runs=$runs"
            echo "command=$app $problem $(paper_length_option "$bench" "$length") --precision $precision $args"
            echo "started=$(date '+%F %T')"
        } > "$dir/info.txt"

        echo "== $config ($runs runs)"
        for run in $(seq 1 "$runs"); do
            # < /dev/null: mpiexec would otherwise read this script's stdin.
            # shellcheck disable=SC2086
            if run_mpi "$WRAPPER" --gpu-pin "$GPU_PIN" -- "$app" $problem $(paper_length_option "$bench" "$length") \
                --precision "$precision" $args --halo-sync "$HALO_SYNC" --profiling-file "$PROFILE" \
                --timing-file "$timing" --report-csv "$dir/report_run${run}.csv" \
                < /dev/null > "$dir/run_${run}.log" 2>&1; then
                echo "   run $run: $(tail -n 1 "$timing") s"
            else
                echo "   run $run: FAILED (see $dir/run_${run}.log)"
            fi
        done
        echo "finished=$(date '+%F %T')" >> "$dir/info.txt"
    done
}
