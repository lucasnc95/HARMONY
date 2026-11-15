#!/usr/bin/env bash
# Runs a program on one MPI rank after choosing which GPUs it may see.
#
#   rank_wrapper.sh [--gpu-pin host=list[,host=list...]] [--check-gpus] -- program [args...]
#
# --gpu-pin compute-1-0=0  fallback: makes the rank on compute-1-0 see only
# GPU 0 (several: compute-1-0=0:1) through CUDA_VISIBLE_DEVICES, honoured by
# NVIDIA's OpenCL. When PBS already set CUDA_VISIBLE_DEVICES (the GPU it
# reserved, e.g. the one of vnode compute-1-0[0]), that choice is kept. Environment variables of the job script do not
# reach the ranks on other nodes (mpiexec only forwards some), which is why
# this is done here, on each rank.
#
# --check-gpus  lists the GPUs this rank sees (nvidia-smi -L) and fails if
#               any of them is a MIG instance (NVIDIA's OpenCL does not run
#               on MIG). Used once at the start of each job.

set -euo pipefail

pins=""
check=0
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-pin) pins=$2; shift 2 ;;
        --check-gpus) check=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

host=$(hostname -s)
if [ -n "$pins" ]; then
    IFS=',' read -r -a entries <<< "$pins"
    for entry in "${entries[@]}"; do
        if [ "${entry%%=*}" = "$host" ]; then
            if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
                list=${entry#*=}
                export CUDA_VISIBLE_DEVICES=${list//:/,}   # several GPUs: compute-1-0=0:1
            fi
        fi
    done
fi
echo "[rank_wrapper] $host: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}" >&2

if [ "$check" = "1" ] && command -v nvidia-smi > /dev/null 2>&1; then
    gpus=$(nvidia-smi -L 2>&1 || true)
    while IFS= read -r line; do echo "[rank_wrapper] $host: $line" >&2; done <<< "$gpus"
    if grep -q 'MIG' <<< "$gpus"; then
        echo "[rank_wrapper] $host: ERROR: a MIG instance is visible; OpenCL cannot use it." >&2
        echo "[rank_wrapper] $host: ask PBS for the GPU of vnode 0 (vnode=compute-1-0[0])." >&2
        exit 3
    fi
fi

exec "$@"
