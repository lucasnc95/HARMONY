#!/usr/bin/env bash
# Benchmarks of the paper: problem size, run length and balancing schedule
# of each benchmark in each precision. Sourced by paper_job.pbs.
#
# SIZES. Every device keeps the WHOLE domain of each field in ONE OpenCL
# buffer (HARMONY replicates the fields and updates only its partition), so
# a field must fit in the largest buffer of the smallest device. On NVIDIA
# that is 1/4 of the memory: ~2.9 GiB on the RTX A2000 12 GB. Every field
# below is <= 2.75 GiB (the device table of each run shows "Largest buffer";
# HARMONY warns if a field does not fit):
#
#   benchmark  precision  domain                      fields          per device
#   his        float      128 x 128 x 5376 points     2 x 2.63 GiB    5.3 GiB
#   his        double     128 x 128 x 2688 points     2 x 2.63 GiB    5.3 GiB
#   ocean, lu  float      26624 x 26624               2 x 2.64 GiB    5.3 GiB
#   ocean, lu  double     18944 x 18944               2 x 2.67 GiB    5.3 GiB
#   map        float      700,000,000 elements        2 x 2.61 GiB    5.2 GiB
#   map        double     350,000,000 elements        2 x 2.61 GiB    5.2 GiB
#   cardiac    float      256 x 256 x 280 cells       state 2.73 GiB  3.0 GiB
#   cardiac    double     256 x 256 x 140 cells       state 2.73 GiB  3.0 GiB
#
# The kernels also index with int (<= 2^31 values per field), which these
# sizes respect. A node with a GPU and a CPU device needs the per-device
# memory twice (GPU + host RAM) while the fields are created: <= ~11 GB,
# inside the RAM asked for every node.
#
# CARDIAC DISCRETIZATION. dx = 0.02 cm (200 um): the float tissue is
# 5.12 x 5.12 x 5.60 cm (a ventricular block), the double one 5.12 x 5.12 x
# 2.80 cm. dt = 0.05 ms: the explicit diffusion needs D*dt/dx^2 <= 1/6
# (here 0.0012 * 0.05 / 0.02^2 = 0.15); the cell model subdivides dt itself
# where it is stiff. The wave (~0.5 m/s) starts at z = 0 and travels along
# z, the partitioned axis.
#
# RUN LENGTH. Chosen so that a run on ONE GPU takes roughly 0.5-2 minutes
# (estimates from earlier HIS/cardiac runs; double precision is 20-60x
# slower on these GPUs, hence the shorter runs). Measure with a pilot
# (submit_paper_tests.sh with RUNS=1) and set LENGTH if needed.
#
# BALANCING. interval = 10 % of the run; warm-up = one interval (the first
# attempt measures iterations 1..warm-up). WARMUP and INTERVAL override them.

PAPER_BENCHMARKS="his ocean lu map cardiac"

# paper_app <benchmark>: executable
paper_app() {
    case "$1" in
        his)            echo "$HARMONY_ROOT/examples/his/bin/his" ;;
        ocean|lu|map)   echo "$HARMONY_ROOT/examples/stencil2d/bin/stencil2d" ;;
        cardiac)        echo "$HARMONY_ROOT/examples/cardiac_tissue/bin/cardiac_tissue" ;;
        *) echo "unknown benchmark '$1'" >&2; return 1 ;;
    esac
}

# paper_problem <benchmark> <precision>: size and fixed options
paper_problem() {
    case "$1:$2" in
        his:float)      echo "--x 128 --y 128 --z 5376 --gather-final 0" ;;
        his:double)     echo "--x 128 --y 128 --z 2688 --gather-final 0" ;;
        ocean:float)    echo "--impl harmony --gather 0 --kernel ocean --nx 26624 --ny 26624" ;;
        ocean:double)   echo "--impl harmony --gather 0 --kernel ocean --nx 18944 --ny 18944" ;;
        lu:float)       echo "--impl harmony --gather 0 --kernel lu --nx 26624 --ny 26624 --flop-repeat 16 --chains 4" ;;
        lu:double)      echo "--impl harmony --gather 0 --kernel lu --nx 18944 --ny 18944 --flop-repeat 16 --chains 4" ;;
        map:float)      echo "--impl harmony --gather 0 --kernel map --n 700000000 --flop-repeat 16 --chains 4" ;;
        map:double)     echo "--impl harmony --gather 0 --kernel map --n 350000000 --flop-repeat 16 --chains 4" ;;
        cardiac:float)  echo "--nx 256 --ny 256 --nz 280 --dx 0.02 --dt 0.05 --progress 300" ;;
        cardiac:double) echo "--nx 256 --ny 256 --nz 140 --dx 0.02 --dt 0.05 --progress 300" ;;
        *) echo "unknown case '$1:$2'" >&2; return 1 ;;
    esac
}

# paper_length <benchmark> <precision>: iterations (cardiac: simulated ms)
paper_length() {
    case "$1:$2" in
        his:float)      echo 2000 ;;
        his:double)     echo 1000 ;;
        ocean:*)        echo 2000 ;;
        lu:float|map:float)   echo 1000 ;;
        lu:double|map:double) echo 100 ;;
        cardiac:float)  echo 20 ;;   # 400 steps of 0.05 ms
        cardiac:double) echo 5 ;;    # 100 steps
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

# paper_iterations <benchmark> <length>: number of HARMONY iterations (steps)
paper_iterations() {
    case "$1" in
        cardiac) awk -v t="$2" 'BEGIN { printf "%d", t / 0.05 + 0.5 }' ;;
        *)       echo "$2" ;;
    esac
}
