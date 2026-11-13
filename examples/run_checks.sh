#!/usr/bin/env bash
# Builds and runs every example with its correctness check.
#
#   ./examples/run_checks.sh            # 2 MPI ranks
#   RANKS=4 ./examples/run_checks.sh
#   MPIRUN="mpirun --oversubscribe" ./examples/run_checks.sh
#
# Needs HARMONY built (./build.sh) and at least one OpenCL device.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

RANKS=${RANKS:-2}
MPIRUN=${MPIRUN:-mpirun}
failed=0

run_check() {
    local name=$1
    shift
    local log
    log=$(mktemp)
    if $MPIRUN -np "$RANKS" "$@" > "$log" 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name  (output below)"
        tail -n 20 "$log"
        failed=1
    fi
    rm -f "$log"
}

for ex in quickstart_stencil1d multi_kernel his stencil2d cardiac_tissue; do
    (cd "$ex" && ./build.sh > /dev/null) || { echo "FAIL  build of $ex"; failed=1; }
done

run_check "quickstart_stencil1d (device vs CPU)" \
    quickstart_stencil1d/bin/quickstart_stencil1d --n 20000 --iterations 120 --interval 20
run_check "multi_kernel (3 kernels per step, device vs CPU)" \
    multi_kernel/bin/multi_kernel --n 50000 --iterations 60 --interval 15
run_check "his (small mesh, dynamic balancing)" \
    his/bin/his --x 8 --y 8 --z 40 --iterations 101 --rebalance-interval 20
run_check "his double precision" \
    his/bin/his --x 8 --y 8 --z 40 --iterations 41 --rebalance-interval 20 --precision double
run_check "stencil2d (HARMONY vs MPI+OpenCL)" \
    stencil2d/bin/stencil2d --impl both --kernel lu --nx 128 --ny 128 --iters 50 --report 0
run_check "stencil2d double precision, 4 chains, static balancing (HARMONY vs MPI+OpenCL)" \
    stencil2d/bin/stencil2d --impl both --kernel lu --precision double --chains 4 --flop-repeat 8 \
    --nx 128 --ny 128 --iters 40 --report 0 --balance-mode static --balance-warmup 10 --rebalance-interval 5

run_check "cardiac_tissue (2 kernels, ToR-ORd-like model, device vs CPU)" \
    cardiac_tissue/bin/cardiac_tissue --nx 3 --ny 3 --nz 24 --time 6 --rebalance-interval 100 --check 1

snapshots=$(mktemp -d)
run_check "his_snapshots (VTK files)" \
    his/bin/his_snapshots --x 8 --y 8 --z 40 --iterations 20 --every 10 --output "$snapshots/his"
run_check "cardiac_snapshots (VTK files)" \
    cardiac_tissue/bin/cardiac_snapshots --nx 3 --ny 3 --nz 24 --time 2 --every 1 --output "$snapshots/cardiac"
if [ "$(ls "$snapshots"/his/*.vtk "$snapshots"/cardiac/*.vtk 2>/dev/null | wc -l)" -ne 6 ]; then
    echo "FAIL  snapshots: expected 3 + 3 VTK files in $snapshots"
    failed=1
fi
rm -rf "$snapshots"

exit $failed
