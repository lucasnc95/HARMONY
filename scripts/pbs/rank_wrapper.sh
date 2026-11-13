#!/usr/bin/env bash
# Runs a program on one MPI rank after choosing which GPUs it may see.
#
#   rank_wrapper.sh [--gpu-pin host=list[,host=list...]] -- program [args...]
#
# --gpu-pin compute-1-0=0  makes the rank on compute-1-0 see only GPU 0
# (several GPUs: compute-1-0=0:1)
# (CUDA_VISIBLE_DEVICES, honoured by NVIDIA's OpenCL), unless the batch
# system already restricted the GPUs of the job (CUDA_VISIBLE_DEVICES set):
# then that choice is kept. Environment variables of the job script do not
# reach the ranks on other nodes (mpiexec only forwards some), which is why
# this is done here, on each rank.

set -euo pipefail

pins=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-pin) pins=$2; shift 2 ;;
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

exec "$@"
