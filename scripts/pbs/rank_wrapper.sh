#!/usr/bin/env bash
# Runs a program on one MPI rank, optionally checking the GPUs it sees and
# monitoring its node.
#
#   rank_wrapper.sh [--check-gpus] [--monitor=prefix] -- program [args...]
#
# --check-gpus  lists the GPUs this rank sees (nvidia-smi -L) and fails if
#               any of them is a MIG instance (NVIDIA's OpenCL does not run
#               on MIG). Used once at the start of each job.
#
# --monitor=P   while the program runs, records on this node:
#                 P_<host>_gpu.csv    GPU clocks, temperature, power, use and
#                                     throttle reasons every 2 s (nvidia-smi)
#                 P_<host>_procs.txt  every 15 s: load, the processes using the
#                                     most CPU (any user) and the GPU processes
#               to find other jobs/processes or throttling slowing the run down.

set -euo pipefail

check=0
monitor=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check-gpus) check=1; shift ;;
        --monitor=*) monitor=${1#--monitor=}; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

host=$(hostname -s)
if [ "$check" = "1" ]; then
    echo "[rank_wrapper] $host: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}" >&2
fi

if [ "$check" = "1" ] && command -v nvidia-smi > /dev/null 2>&1; then
    gpus=$(nvidia-smi -L 2>&1 || true)
    while IFS= read -r line; do echo "[rank_wrapper] $host: $line" >&2; done <<< "$gpus"
    if grep -q 'MIG' <<< "$gpus"; then
        echo "[rank_wrapper] $host: ERROR: a MIG instance is visible; OpenCL cannot use it." >&2
        echo "[rank_wrapper] $host: ask PBS for the GPU of vnode 0 (vnode=compute-1-0[0])." >&2
        exit 3
    fi
fi

if [ -z "$monitor" ]; then
    exec "$@"
fi

monitors=()
if command -v nvidia-smi > /dev/null 2>&1; then
    select_gpu=()
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && select_gpu=(-i "$CUDA_VISIBLE_DEVICES")
    fields="timestamp,index,name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used"
    # Throttle reasons: the field name depends on the driver version.
    for reasons in clocks_event_reasons.active clocks_throttle_reasons.active; do
        if nvidia-smi ${select_gpu[@]+"${select_gpu[@]}"} --query-gpu="$fields,$reasons" --format=csv,noheader > /dev/null 2>&1; then
            fields="$fields,$reasons"
            break
        fi
    done
    nvidia-smi ${select_gpu[@]+"${select_gpu[@]}"} --query-gpu="$fields" --format=csv -l 2 > "${monitor}_${host}_gpu.csv" 2>&1 &
    monitors+=($!)
fi
(
    while true; do
        echo "=== $(date '+%F %T')  load: $(cut -d' ' -f1-3 /proc/loadavg)"
        ps -eo pid,user,pcpu,pmem,etime,comm --sort=-pcpu | head -n 12
        if command -v nvidia-smi > /dev/null 2>&1; then
            echo "--- GPU processes (all GPUs of the node)"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1
        fi
        sleep 15
    done
) > "${monitor}_${host}_procs.txt" 2>&1 &
monitors+=($!)

status=0
"$@" || status=$?
kill "${monitors[@]}" 2> /dev/null || true
wait 2> /dev/null || true
exit "$status"
