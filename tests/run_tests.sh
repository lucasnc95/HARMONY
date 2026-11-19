#!/usr/bin/env bash
# Runs HARMONY's compiled C++ tests across 1 and 2 MPI ranks.
#
#   ./run_tests.sh
#   MPIRUN="mpirun --oversubscribe" ./run_tests.sh
#   RANKS_LIST="1 2" ./run_tests.sh

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

MPIRUN=${MPIRUN:-"mpirun --oversubscribe"}
RANKS_LIST=${RANKS_LIST:-"1 2"}

if [ "$(id -u)" -eq 0 ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

shopt -s nullglob
TEST_BINS=(bin/test_*)
shopt -u nullglob

if [ ${#TEST_BINS[@]} -eq 0 ]; then
    echo "error: no test binaries found in $PWD/bin (run ./build.sh tests first)" >&2
    exit 1
fi

failed=0
executed=0

for exe in "${TEST_BINS[@]}"; do
    [ -x "$exe" ] || continue
    name=$(basename "$exe")
    if [ "$name" = "test_probe_runtime_integration" ]; then
        continue
    fi
    for np in $RANKS_LIST; do
        executed=$((executed + 1))
        log=$(mktemp)
        $MPIRUN -np "$np" "./$exe" > "$log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "PASS  $name (-np $np)"
        elif [ "$rc" -eq 77 ]; then
            echo "SKIP  $name (-np $np)"
        else
            echo "FAIL  $name (-np $np, exit $rc)"
            cat "$log"
            failed=1
        fi
        rm -f "$log"
    done
done

TOPO_PROBE="../bin/topo_probe"
PROBE_TEST="bin/test_probe_runtime_integration"

if [ ! -x "$TOPO_PROBE" ]; then
    echo "FAIL  topo_probe binary not found at $TOPO_PROBE (run ./build.sh tools first)"
    failed=1
elif [ ! -x "$PROBE_TEST" ]; then
    echo "FAIL  test_probe_runtime_integration binary not found at $PROBE_TEST"
    failed=1
else
    tmp_dir=$(mktemp -d)
    nosynth_json="$tmp_dir/nosynth.json"
    nosynth_log="$tmp_dir/nosynth.log"
    executed=$((executed + 1))
    $MPIRUN -np 1 "$TOPO_PROBE" --output-json "$nosynth_json" > "$nosynth_log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "PASS  topo_probe (non-synthetic rejects missing OpenCL GPU, exit $rc)"
    elif grep -q "^mode=measured$" "$nosynth_log" && [ -f "$nosynth_json" ]; then
        echo "PASS  topo_probe (non-synthetic measured real OpenCL GPU)"
    else
        echo "FAIL  topo_probe (non-synthetic exited 0 without mode=measured)"
        cat "$nosynth_log"
        failed=1
    fi

    for np in $RANKS_LIST; do
        executed=$((executed + 1))
        probe_json="$tmp_dir/probe_np${np}.json"
        probe_log="$tmp_dir/probe_np${np}.log"
        integ_log="$tmp_dir/integ_np${np}.log"

        $MPIRUN -np "$np" "$TOPO_PROBE" --synthetic --output-json "$probe_json" > "$probe_log" 2>&1
        rc=$?
        if [ "$rc" -ne 0 ] || [ ! -f "$probe_json" ] || \
           ! grep -q "^mode=synthetic$" "$probe_log" || \
           ! grep -q "^rank_device_counts=" "$probe_log"; then
            echo "FAIL  topo_probe --synthetic (-np $np, exit $rc)"
            cat "$probe_log"
            failed=1
            continue
        fi
        echo "PASS  topo_probe --synthetic (-np $np)"

        $MPIRUN -np "$np" "./$PROBE_TEST" "$probe_json" > "$integ_log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "PASS  test_probe_runtime_integration (-np $np)"
        else
            echo "FAIL  test_probe_runtime_integration (-np $np, exit $rc)"
            cat "$integ_log"
            failed=1
        fi
    done

    rm -rf "$tmp_dir"
fi

ASYNC_HALO_TEST="bin/test_async_halo_opencl"
if [ ! -x "$ASYNC_HALO_TEST" ]; then
    echo "FAIL  test_async_halo_opencl binary not found at $ASYNC_HALO_TEST"
    failed=1
else
    executed=$((executed + 1))
    halo_log=$(mktemp)
    $MPIRUN -np 2 "./$ASYNC_HALO_TEST" --require-gpu > "$halo_log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q "^PASS:" "$halo_log"; then
        echo "PASS  test_async_halo_opencl --require-gpu (-np 2)"
    elif [ "$rc" -eq 1 ] && grep -q "Preflight failed:" "$halo_log"; then
        echo "PASS  test_async_halo_opencl --require-gpu (rejects missing OpenCL GPU, exit 1)"
    else
        echo "FAIL  test_async_halo_opencl --require-gpu (-np 2, exit $rc)"
        cat "$halo_log"
        failed=1
    fi
    rm -f "$halo_log"
fi

if [ "$executed" -eq 0 ]; then
    echo "error: no executable test binaries found in $PWD/bin" >&2
    exit 1
fi

exit "$failed"
