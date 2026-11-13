#!/usr/bin/env bash
# Submits the paper tests: every benchmark x precision x scenario is one job
# of scripts/pbs/paper_job.pbs (run from the repository root).
#
#   DRY_RUN=1 scripts/pbs/submit_paper_tests.sh        # only print the qsub commands
#   scripts/pbs/submit_paper_tests.sh                  # everything in double (20 jobs)
#   BENCHES="lu map" PRECISIONS="float double" SCENARIOS="gpu1 n1" scripts/pbs/submit_paper_tests.sh
#   RUNS=1 OUT_DIR=results/pilot scripts/pbs/submit_paper_tests.sh   # pilot: measure the times first
#
# Scenarios and nodes (one MPI process per node; RAM asked explicitly,
# otherwise PBS reserves only 6 GB):
#   gpu1  compute-0-2 (GPU only)
#   n1    compute-0-2 (GPU + CPU)
#   n2    compute-0-2 + compute-3-6
#   n3    compute-0-2 + compute-3-6 + compute-1-0 (128 cores, one GPU)
#
# Environment (defaults in brackets):
#   BENCHES     [his ocean lu map cardiac]
#   PRECISIONS  [double]  (only double for now; PRECISIONS="float double" runs both)
#   SCENARIOS   [gpu1 n1 n2 n3]
#   RUNS        [10]   runs per configuration
#   RUNS_OFF    [RUNS] runs of the configuration "off" (GPU + CPU with equal
#               shares: by far the slowest; see the estimate in the README)
#   PER_CONFIG  [0]    1 = one job per configuration (shorter jobs, for walltime limits)
#   WALLTIME    [12:00:00]
#   CHUNK_0_2   [1:host=compute-0-2:ncpus=32:ngpus=1:mem=28gb:mpiprocs=1]
#   CHUNK_3_6   [1:host=compute-3-6:ncpus=24:ngpus=1:mem=20gb:mpiprocs=1]
#   CHUNK_1_0   [1:host=compute-1-0:ncpus=128:ngpus=1:mem=64gb:mpiprocs=1]
#   GPU_PIN     [compute-1-0=0]   GPU used on compute-1-0 when PBS does not
#               restrict it (the GPU of vnode 0); see rank_wrapper.sh
#   QUEUE       [-]    qsub -q
#   Forwarded to the job: RUNS_OFF WARMUP INTERVAL LENGTH THRESHOLD CPU_HOST_CORES
#               HALO_SYNC OUT_DIR CONFIGS MODULES LAUNCHER
#
# compute-1-0 has several vnodes. The chunk above takes 128 cores and one GPU
# of the host; rank_wrapper.sh then shows only GPU 0 to the program unless
# the PBS GPU hook already chose one (CUDA_VISIBLE_DEVICES). To force PBS to
# give the GPU of vnode 0, use two chunks, for example
#   CHUNK_1_0="1:vnode=compute-1-0[0]:ngpus=1:ncpus=16:mem=32gb:mpiprocs=1+1:host=compute-1-0:ncpus=112:mem=32gb:mpiprocs=0"
# (check the vnodes with: pbsnodes -av | grep -A15 'compute-1-0\[').

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

BENCHES=${BENCHES:-"his ocean lu map cardiac"}
PRECISIONS=${PRECISIONS:-double}   # for now only double; "float double" for both
SCENARIOS=${SCENARIOS:-"gpu1 n1 n2 n3"}
RUNS=${RUNS:-10}
PER_CONFIG=${PER_CONFIG:-0}
WALLTIME=${WALLTIME:-12:00:00}
CHUNK_0_2=${CHUNK_0_2:-1:host=compute-0-2:ncpus=32:ngpus=1:mem=28gb:mpiprocs=1}
CHUNK_3_6=${CHUNK_3_6:-1:host=compute-3-6:ncpus=24:ngpus=1:mem=20gb:mpiprocs=1}
CHUNK_1_0=${CHUNK_1_0:-1:host=compute-1-0:ncpus=128:ngpus=1:mem=64gb:mpiprocs=1}
GPU_PIN=${GPU_PIN:-compute-1-0=0}
DRY_RUN=${DRY_RUN:-0}

select_of() {
    case "$1" in
        gpu1|n1) echo "$CHUNK_0_2" ;;
        n2)      echo "$CHUNK_0_2+$CHUNK_3_6" ;;
        n3)      echo "$CHUNK_0_2+$CHUNK_3_6+$CHUNK_1_0" ;;
        *) echo "unknown scenario '$1'" >&2; return 1 ;;
    esac
}

configs_of() {
    if [ "$1" = "gpu1" ]; then echo "gpu"
    else echo "off static_threshold static_profiled dynamic_threshold dynamic_profiled"; fi
}

submit() {
    local bench=$1 precision=$2 scenario=$3 configs=$4
    local vars="HARMONY_ROOT=$(pwd),BENCH=$bench,PRECISION=$precision,SCENARIO=$scenario,RUNS=$RUNS,GPU_PIN=$GPU_PIN"
    vars="$vars,CONFIGS=${configs// /:}"   # ":" instead of spaces inside qsub -v
    for v in RUNS_OFF WARMUP INTERVAL LENGTH THRESHOLD CPU_HOST_CORES HALO_SYNC OUT_DIR MODULES LAUNCHER; do
        if [ -n "${!v:-}" ]; then vars="$vars,$v=${!v}"; fi
    done
    local name="H${bench:0:3}${precision:0:1}_${scenario}"
    if [ "$PER_CONFIG" = "1" ]; then
        case "$configs" in
            off) name="${name}_off" ;; static_threshold) name="${name}_sth" ;;
            static_profiled) name="${name}_spr" ;; dynamic_threshold) name="${name}_dth" ;;
            dynamic_profiled) name="${name}_dpr" ;; *) name="${name}_${configs:0:3}" ;;
        esac
    fi
    local cmd=(qsub -N "${name:0:15}" -l "select=$(select_of "$scenario")" -l place=scatter
               -l "walltime=$WALLTIME" -v "$vars")
    [ -n "${QUEUE:-}" ] && cmd+=(-q "$QUEUE")
    cmd+=(scripts/pbs/paper_job.pbs)
    echo "${cmd[*]}"
    [ "$DRY_RUN" = "1" ] || "${cmd[@]}"
}

for bench in $BENCHES; do
    for precision in $PRECISIONS; do
        for scenario in $SCENARIOS; do
            configs=${CONFIGS:-$(configs_of "$scenario")}
            if [ "$PER_CONFIG" = "1" ]; then
                for config in $configs; do submit "$bench" "$precision" "$scenario" "$config"; done
            else
                submit "$bench" "$precision" "$scenario" "$configs"
            fi
        done
    done
done
