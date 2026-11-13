#!/usr/bin/env bash
# Common setup of the HARMONY PBS jobs (sourced by every *.pbs script).
#
# MPI processes are started with OpenHPC's `prun`, which launches one rank
# per slot of the PBS allocation (one per line of $PBS_NODEFILE) using the
# launcher of the loaded MPI module. To get one rank per node, ask PBS for
# one MPI process per chunk:
#
#   qsub -l select=2:ncpus=16:mpiprocs=1 ...                    any 2 nodes
#   qsub -l select=1:host=compute-3-3:ncpus=16:mpiprocs=1+1:host=compute-3-4:ncpus=16:mpiprocs=1 ...
#
# (With Torque-style "-l nodes=N:ppn=16" the node file has 16 lines per node,
# so prun would start 16 ranks per node.)
#
# Variables (override with qsub -v):
#   HARMONY_ROOT  repository (or install) directory   (default: submission directory)
#   MODULES       modules loaded before running         (default: "gnu13 openmpi5")
#   LAUNCHER      command that starts the MPI ranks     (default: prun)
#   BIND          CPU binding of the ranks              (default: none)
#                 By default Open MPI pins each rank to ONE core, and OpenCL CPU
#                 devices then expose a single compute unit. "none" lets each
#                 rank (and its CPU device) use every core PBS gave it; set
#                 BIND="" to keep the MPI default.

cd "${PBS_O_WORKDIR:-$(pwd)}"
HARMONY_ROOT=${HARMONY_ROOT:-$(pwd)}
MODULES=${MODULES-"gnu13 openmpi5"}
LAUNCHER=${LAUNCHER:-prun}
BIND=${BIND-none}

# Lmod is not always initialized in batch shells.
if ! type module > /dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /opt/ohpc/admin/lmod/lmod/init/bash; do
        if [ -f "$init" ]; then
            # shellcheck disable=SC1090
            source "$init"
            break
        fi
    done
fi

if [ -n "$MODULES" ]; then
    if type module > /dev/null 2>&1; then
        # shellcheck disable=SC2086
        module load $MODULES || echo "warning: could not load modules: $MODULES" >&2
        # prun has its own module in OpenHPC.
        command -v prun > /dev/null 2>&1 || module load prun > /dev/null 2>&1 || true
    else
        echo "warning: 'module' is not available; using the current environment" >&2
    fi
fi

if [ "$LAUNCHER" = "prun" ] && ! command -v prun > /dev/null 2>&1; then
    echo "error: prun not found (load the OpenHPC 'prun' module or set LAUNCHER)" >&2
    exit 1
fi

# OpenCL vendor runtime used by the cluster's older jobs (only if present).
if [ -d /share/apps/AMDAPPSDK-3.0/lib/x86_64 ]; then
    export LD_LIBRARY_PATH=/share/apps/AMDAPPSDK-3.0/lib/x86_64/:${LD_LIBRARY_PATH:-}
fi

if [ -n "$BIND" ]; then
    export OMPI_MCA_hwloc_base_binding_policy="$BIND"     # Open MPI <= 4
    export PRTE_MCA_hwloc_default_binding_policy="$BIND"  # Open MPI 5 (PRRTE)
fi

# Ranks started by the launcher, and distinct nodes.
NP=$(wc -l < "$PBS_NODEFILE")
NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
echo "Allocation: $NP MPI ranks on $NODES nodes"
sort "$PBS_NODEFILE" | uniq -c

# run_mpi <program> [args...]: starts the program on every rank.
run_mpi() {
    # shellcheck disable=SC2086
    $LAUNCHER "$@"
}

# Which CPUs each rank may use, and the RAM of its node (diagnostic only).
echo "Per-rank CPU affinity and memory:"
# shellcheck disable=SC2016
run_mpi bash -c 'echo "  $(hostname): CPUs $(grep Cpus_allowed_list /proc/self/status | cut -f2), node RAM $(awk "/MemTotal/ {printf \"%.1f GB\", \$2/1048576}" /proc/meminfo)"' \
    || echo "  (could not run the affinity check)"
