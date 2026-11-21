#!/usr/bin/env bash
# Benchmark cases of the paper: problem size, run length and executable.
# Sourced by paper_run.sh.
#
# Sizes. The partitioned axis (z; rows for ocean and lu) is the longest one.
# Every device keeps the whole domain of each field, and a CPU device keeps it
# in host RAM, so the sizes are limited by the node with the least free
# memory (compute-3-4, about 7 GB): about 2.5 GiB of data per device, which
# gives a peak of about 4 GiB per MPI process (data of the CPU device plus
# one field of initial data) plus the GPU driver.
#
#   benchmark  precision  domain                    data per device
#   his        float      128 x 128 x 2560 points   2.5 GiB
#   his        double     128 x 128 x 1280 points   2.5 GiB
#   ocean, lu  float      8192 x 40960 (nx x ny)    2.5 GiB
#   ocean, lu  double     8192 x 20480 (nx x ny)    2.5 GiB
#   map        float      335,544,320 elements      2.5 GiB
#   map        double     167,772,160 elements      2.5 GiB
#   cardiac    float      128 x 128 x 768 cells     2.1 GiB
#   cardiac    double     128 x 128 x 384 cells     2.0 GiB
#
# Cardiac discretization: dx = 0.02 cm, dt = 0.05 ms
# (D * dt / dx^2 = 0.0012 * 0.05 / 0.02^2 = 0.15 <= 1/6).
#
# Run length: his and ocean 10000 iterations; lu and map 5000 (float) and
# 1000 (double); cardiac 20 ms (float) and 5 ms (double). LENGTH overrides it.

PAPER_BENCHMARKS="his ocean lu map cardiac"

# paper_app <benchmark>: executable
paper_app() {
    case "$1" in
        his)          echo "$HARMONY_ROOT/examples/his/bin/his" ;;
        ocean|lu|map) echo "$HARMONY_ROOT/examples/stencil2d/bin/stencil2d" ;;
        cardiac)      echo "$HARMONY_ROOT/examples/cardiac_tissue/bin/cardiac_tissue" ;;
        *) echo "unknown benchmark '$1'" >&2; return 1 ;;
    esac
}

# paper_problem <benchmark> <precision>: problem size and fixed options
paper_problem() {
    case "$1:$2" in
        his:float)      echo "--x 128 --y 128 --z 2560 --gather-final 0" ;;
        his:double)     echo "--x 128 --y 128 --z 1280 --gather-final 0" ;;
        ocean:float)    echo "--impl harmony --gather 0 --kernel ocean --nx 8192 --ny 40960" ;;
        ocean:double)   echo "--impl harmony --gather 0 --kernel ocean --nx 8192 --ny 20480" ;;
        lu:float)       echo "--impl harmony --gather 0 --kernel lu --nx 8192 --ny 40960 --flop-repeat 16 --chains 4" ;;
        lu:double)      echo "--impl harmony --gather 0 --kernel lu --nx 8192 --ny 20480 --flop-repeat 16 --chains 4" ;;
        map:float)      echo "--impl harmony --gather 0 --kernel map --n 335544320 --flop-repeat 16 --chains 4" ;;
        map:double)     echo "--impl harmony --gather 0 --kernel map --n 167772160 --flop-repeat 16 --chains 4" ;;
        cardiac:float)  echo "--nx 128 --ny 128 --nz 768 --dx 0.02 --dt 0.05" ;;
        cardiac:double) echo "--nx 128 --ny 128 --nz 384 --dx 0.02 --dt 0.05" ;;
        *) echo "unknown case '$1:$2'" >&2; return 1 ;;
    esac
}

# paper_length <benchmark> <precision>: iterations (cardiac: simulated ms)
paper_length() {
    case "$1:$2" in
        his:*|ocean:*)        echo 10000 ;;
        lu:float|map:float)   echo 5000 ;;
        lu:double|map:double) echo 1000 ;;
        cardiac:float)        echo 20 ;;
        cardiac:double)       echo 5 ;;
        *) echo "unknown case '$1:$2'" >&2; return 1 ;;
    esac
}

# paper_length_option <benchmark> <length>: the option that sets the length
paper_length_option() {
    case "$1" in
        his)          echo "--iterations $2" ;;
        ocean|lu|map) echo "--iters $2" ;;
        cardiac)      echo "--time $2" ;;
    esac
}

# paper_iterations <benchmark> <length>: number of iterations (time steps)
paper_iterations() {
    case "$1" in
        cardiac) awk -v t="$2" 'BEGIN { printf "%d", t / 0.05 + 0.5 }' ;;
        *)       echo "$2" ;;
    esac
}
