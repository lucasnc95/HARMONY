#!/usr/bin/env python3
"""
Mock MPI Launcher for HARMONY Super Job offline integration tests.
Simulates two-rank OpenCL benchmark outputs, device discovery, topology metrics,
rank_wrapper GPU/MIG checks, and upstream structured CSV timing reports
without requiring live GPUs or PBS.
"""

from datetime import datetime
import json
import os
import shutil
import subprocess
import sys
import time


def derive_report_csv_path(output_csv):
    """Mirrors harmony_bench::derive_report_csv_path in benchmark_common.hpp."""
    if not output_csv:
        return None
    directory, filename = os.path.split(output_csv)
    stem, ext = os.path.splitext(filename)
    report_name = f"{stem}_report{ext or '.csv'}"
    return os.path.join(directory, report_name) if directory else report_name


def parse_args(argv):
    """Finds the target binary and splits launcher/wrapper flags from target arguments."""
    binary_index = -1
    for i, arg in enumerate(argv):
        base = os.path.basename(arg)
        if base.endswith(".out") or any(
            name in base
            for name in (
                "topo_probe",
                "test_async_halo_opencl",
                "kneighbor",
                "leanmd",
                "telemetry",
                "nvidia-smi",
            )
        ):
            binary_index = i
            break

    if binary_index == -1:
        if len(argv) > 0:
            return argv[0], argv[1:]
        return None, []

    binary = argv[binary_index]
    target_args = argv[binary_index + 1 :]
    return binary, target_args


def extract_named_arg(args, name, default=None):
    """Extracts a named argument like --n 1000 from args."""
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1]
    return default


def simulate_rank_wrapper_check(argv):
    """Simulates rank_wrapper.sh --check-gpus MIG/GPU validation."""
    if os.environ.get("MOCK_MIG_DETECTED") == "1":
        sys.stderr.write("[rank_wrapper] compute-1-0: GPU 0: NVIDIA A100 (UUID: GPU-0)\n")
        sys.stderr.write("[rank_wrapper] compute-1-0:   MIG 1g.10gb Device 0: (UUID: MIG-0)\n")
        sys.stderr.write("[rank_wrapper] compute-1-0: ERROR: a MIG instance is visible; OpenCL cannot use it.\n")
        sys.stderr.write("[rank_wrapper] compute-1-0: ask PBS for the GPU of vnode 0 (vnode=compute-1-0[0]).\n")
        sys.exit(3)

    sys.stderr.write("[rank_wrapper] compute-1-0: CUDA_VISIBLE_DEVICES=<all>\n")
    sys.stderr.write("[rank_wrapper] compute-1-0: GPU 0: NVIDIA A100 80GB PCIe\n")
    sys.stderr.write("[rank_wrapper] compute-1-1: CUDA_VISIBLE_DEVICES=<all>\n")
    sys.stderr.write("[rank_wrapper] compute-1-1: GPU 0: NVIDIA A100 80GB PCIe\n")
    sys.exit(0)


def simulate_topo_probe(target_args):
    """Simulates bin/topo_probe execution."""
    if os.environ.get("MOCK_FAIL_TOPO_PROBE") == "1":
        sys.stderr.write("ERROR: Simulated topo_probe failure\n")
        sys.exit(1)

    sleep_s = float(os.environ.get("MOCK_SLEEP_TOPO", "0") or 0)
    if sleep_s > 0:
        time.sleep(sleep_s)

    out_json = extract_named_arg(target_args, "--output-json")
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        topo_data = {
            "pcie_latency_ns": [42780.25, 31074.1],
            "pcie_bandwidth_gbps": [10.79253818, 10.37190217],
            "mpi_latency_ns": [0.0, 4998767.433, 4999223.567, 0.0],
            "mpi_bandwidth_gbps": [0.0, 0.02786097291, 0.03073437697, 0.0],
            "numa_distance": [],
            "device_numa_node": [],
            "memory_contention_factor": [],
            "thermal_tdp_watts": [],
            "current_power_watts": [],
            "synthetic": False,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(topo_data, f, indent=2)

    counts = "2,0" if os.environ.get("MOCK_ASYMMETRIC_TOPO") == "1" else "1,1"
    sys.stdout.write("# Topology Probe Measurements\n")
    sys.stdout.write("mode=measured\n")
    sys.stdout.write("total_devices=2\n")
    sys.stdout.write("mpi_ranks=2\n")
    sys.stdout.write(f"rank_device_counts={counts}\n")
    if out_json:
        sys.stdout.write(f"saved_json={out_json}\n")
    sys.exit(0)


def simulate_test_async_halo(target_args):
    """Simulates tests/bin/test_async_halo_opencl execution."""
    if os.environ.get("MOCK_FAIL_PREFLIGHT_HALO") == "1":
        sys.stderr.write("Preflight failed: Simulated halo communication failure\n")
        sys.exit(1)

    halo_sleep_s = float(os.environ.get("MOCK_SLEEP_HALO", "0") or 0)
    if halo_sleep_s > 0:
        time.sleep(halo_sleep_s)

    if os.environ.get("MOCK_ASYMMETRIC_GPUS") == "1":
        if "--require-gpu" in target_args:
            sys.stderr.write(
                "Preflight failed: requires >= 1 verified OpenCL GPU per MPI rank "
                "and >= 2 total GPUs (found min_per_rank=0, total=2)\n"
            )
            sys.exit(1)
        sys.stdout.write(
            "SKIP: asynchronous OpenCL halo test needs >= 1 OpenCL GPU per MPI rank "
            "and >= 2 total devices (found min=0, total=2)\n"
        )
        sys.exit(77)

    sys.stdout.write("PASS: asynchronous OpenCL halo matches two-step CPU reference (ranks=2, min_per_rank=1, total_devices=2, gpu=1)\n")
    sys.exit(0)


def write_mock_report_csv(report_csv, n, iterations, interval, algo, seed, loop_time):
    """Writes an upstream write_timing_report() multi-section CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)
    rebalances = (iterations - 1) // interval if interval > 0 else 0
    applied = 1 if rebalances > 0 else 0
    bytes_moved = 400000 if applied else 0
    compute_s = loop_time * 0.72
    halo_s = loop_time * 0.04
    mpi_halo_s = loop_time * 0.08
    mpi_mig_s = loop_time * 0.02 if applied else 0.0
    mpi_bal_s = loop_time * 0.01 if rebalances > 0 else 0.0
    mig_s = loop_time * 0.03 if applied else 0.0
    bal_s = loop_time * 0.02 if rebalances > 0 else 0.0
    other_s = loop_time * 0.05
    unmeasured_s = max(0.0, loop_time - (compute_s + other_s + halo_s + mpi_halo_s + mpi_mig_s + mpi_bal_s + mig_s + bal_s))

    with open(report_csv, "w", encoding="utf-8") as f:
        f.write("#run,wall_s,iterations,mpi_processes,devices,balance_attempts,balance_applied,bytes_moved\n")
        f.write(f"run,{loop_time:.9g},{iterations},2,2,{rebalances},{applied},{bytes_moved}\n")
        f.write(
            "#timeline,rank,device,wall_s,opencl_compute_s,opencl_other_s,halo_s,mpi_halo_s,"
            "mpi_migration_s,mpi_balance_s,mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s\n"
        )
        f.write(
            f"timeline,1,1,{loop_time:.9g},{compute_s:.9g},{other_s:.9g},{halo_s:.9g},{mpi_halo_s:.9g},"
            f"{mpi_mig_s:.9g},{mpi_bal_s:.9g},0,{mig_s:.9g},{bal_s:.9g},0,{unmeasured_s:.9g}\n"
        )
        f.write("#device,index,rank,type,name,elements,compute_s,transfer_s,idle_s\n")
        half = n // 2
        f.write(f"device,0,0,GPU,NVIDIA A100 80GB PCIe,{half},{compute_s * 0.9:.9g},{halo_s:.9g},{loop_time - compute_s * 0.9 - halo_s:.9g}\n")
        f.write(f"device,1,1,GPU,NVIDIA A100 80GB PCIe,{n - half},{compute_s:.9g},{halo_s:.9g},{loop_time - compute_s - halo_s:.9g}\n")
        f.write(
            "#process,rank,host,launch_s,device_wait_s,halo_s,mpi_halo_s,mpi_migration_s,mpi_balance_s,"
            "mpi_sync_s,migration_s,balance_s,gather_s,unmeasured_s\n"
        )
        f.write(
            f"process,0,compute-1-0,{other_s * 0.5:.9g},{compute_s * 0.9 + other_s * 0.5:.9g},{halo_s:.9g},"
            f"{mpi_halo_s:.9g},{mpi_mig_s:.9g},{mpi_bal_s:.9g},0,{mig_s:.9g},{bal_s:.9g},0,{unmeasured_s:.9g}\n"
        )
        f.write(
            f"process,1,compute-1-1,{other_s * 0.5:.9g},{compute_s + other_s * 0.5:.9g},{halo_s:.9g},"
            f"{mpi_halo_s:.9g},{mpi_mig_s:.9g},{mpi_bal_s:.9g},0,{mig_s:.9g},{bal_s:.9g},0,{unmeasured_s:.9g}\n"
        )
        f.write(
            "#balance,number,iteration,policy,applied,iteration_s_before,iteration_s_after,bytes_moved,"
            "migration_s,decision_s,warmup_probe,reason,algorithm,perturbation_factor,seed,"
            "contention_adjusted,power_capped,power_budget_watts,throttle_factor\n"
        )
        if rebalances > 0:
            iter_s = loop_time / max(1, iterations)
            f.write(
                f"balance,1,{interval},dynamic threshold,{applied},{iter_s:.9g},{iter_s * 0.9:.9g},"
                f"{bytes_moved},{mig_s:.9g},{bal_s:.9g},0,division changed,{algo},0.1,{seed},0,0,0,1\n"
            )


def simulate_benchmark(binary_name, target_args):
    """Simulates bin/kneighbor and bin/leanmd benchmark execution."""
    if os.environ.get("MOCK_FAIL_BENCHMARK") == "1":
        sys.stderr.write(f"ERROR: Simulated benchmark failure in {binary_name}\n")
        sys.exit(1)

    fail_pattern = os.environ.get("MOCK_FAIL_IF_CONTAINS")
    if fail_pattern and fail_pattern in " ".join(target_args):
        sys.stderr.write(f"ERROR: Simulated failure for {binary_name} with pattern: {fail_pattern}\n")
        sys.exit(1)

    fail_warmup_file = os.environ.get("MOCK_FAIL_WARMUP_ONCE_FILE")
    if fail_warmup_file and not os.path.exists(fail_warmup_file) and "warmup_" in " ".join(target_args):
        with open(fail_warmup_file, "w", encoding="utf-8") as f:
            f.write("failed warmup once\n")
        sys.stderr.write(f"ERROR: One-time simulated warmup failure in {binary_name}\n")
        sys.exit(1)

    fail_once_file = os.environ.get("MOCK_FAIL_ONCE_FILE")
    if fail_once_file and not os.path.exists(fail_once_file):
        if "warmup_" not in " ".join(target_args):
            with open(fail_once_file, "w", encoding="utf-8") as f:
                f.write("failed once\n")
            sys.stderr.write(f"ERROR: One-time simulated failure in {binary_name}\n")
            sys.exit(1)

    sleep_pattern = os.environ.get("MOCK_SLEEP_IF_CONTAINS")
    if sleep_pattern and sleep_pattern in " ".join(target_args):
        sleep_sec = float(os.environ.get("MOCK_SLEEP_SECONDS", "5"))
        time.sleep(sleep_sec)

    sleep_once_file = os.environ.get("MOCK_SLEEP_ONCE_FILE")
    if sleep_once_file and not os.path.exists(sleep_once_file):
        if "warmup_" not in " ".join(target_args):
            with open(sleep_once_file, "w", encoding="utf-8") as f:
                f.write("slept once\n")
            sleep_sec = float(os.environ.get("MOCK_SLEEP_SECONDS", "5"))
            time.sleep(sleep_sec)

    n = int(extract_named_arg(target_args, "--n", "1000000"))
    iterations = int(extract_named_arg(target_args, "--iterations", "100"))
    interval = int(extract_named_arg(target_args, "--balance-interval", "10"))
    algo = extract_named_arg(target_args, "--balance-algo", "erad")
    seed = extract_named_arg(target_args, "--seed", "42")
    output_csv = extract_named_arg(target_args, "--output-csv")
    report_csv = extract_named_arg(target_args, "--report-csv") or derive_report_csv_path(output_csv)

    is_kneighbor = "kneighbor" in binary_name
    bench_name = "kneighbor" if is_kneighbor else "leanmd"
    wall_time = 3.73996 if is_kneighbor else 0.438039
    loop_time = 3.50000 if is_kneighbor else 0.400000

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", encoding="utf-8") as f:
            f.write("iteration,step_time_s,action,rebalanced,rebalance_gain,execution_mode\n")
            for i in range(iterations):
                is_rebal = (interval > 0 and i > 0 and i % interval == 0)
                step_time = 0.050129352 if is_rebal else 0.020121021
                action = "rebalance" if is_rebal else "compute"
                rebalanced = 1 if is_rebal else 0
                gain = 0.000063493 if is_rebal else 0.0
                f.write(f"{i},{step_time:.9f},{action},{rebalanced},{gain:.9f},gpu_opencl\n")

    if report_csv:
        write_mock_report_csv(report_csv, n, iterations, interval, algo, seed, loop_time)

    sys.stdout.write("[device] rank=0 global_index=0 name=NVIDIA A100 80GB PCIe\n")
    sys.stdout.write("[device] rank=1 global_index=1 name=NVIDIA A100 80GB PCIe\n")

    rebalances = (iterations - 1) // interval if interval > 0 else 0
    sys.stdout.write(
        f"[{bench_name}] algo={algo} mode=gpu_opencl total_time_s={wall_time} "
        f"loop_time_s={loop_time:.6f} num_rebalances={rebalances} avg_rebalance_gain=6.34934e-05\n"
    )
    sys.exit(0)


def simulate_telemetry(target_args):
    """Simulates multi-rank hardware telemetry output across 2 ranks/hosts."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            res = subprocess.run(
                [
                    smi,
                    "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory,power.draw,clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            if lines:
                sys.stdout.write(f"compute-1-0, 0, {lines[0]}\n")
                line1 = lines[1] if len(lines) > 1 else lines[0]
                sys.stdout.write(f"compute-1-1, 1, {line1}\n")

                try:
                    apps_res = subprocess.run(
                        [
                            smi,
                            "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    apps_lines = [
                        l.strip()
                        for l in apps_res.stdout.splitlines()
                        if l.strip()
                        and "NVIDIA" not in l
                        and not any(k in l for k in ("clocks", "temperature", "power"))
                    ]
                    if apps_lines:
                        for app_l in apps_lines:
                            sys.stdout.write(f"# [compute-apps] host=compute-1-0 rank=0 {app_l}\n")
                            sys.stdout.write(f"# [compute-apps] host=compute-1-1 rank=1 {app_l}\n")
                    else:
                        sys.stdout.write("# [compute-apps] host=compute-1-0 rank=0 shared_workload=none\n")
                        sys.stdout.write("# [compute-apps] host=compute-1-1 rank=1 shared_workload=none\n")
                except Exception:
                    sys.stdout.write("# [compute-apps] host=compute-1-0 rank=0 shared_workload=none\n")
                    sys.stdout.write("# [compute-apps] host=compute-1-1 rank=1 shared_workload=none\n")

                sys.exit(0)
        except Exception:
            pass

    sys.stdout.write(f"compute-1-0, 0, {now}, NVIDIA A100-PCIE-40GB, 38, 12, 1410, 1215, 78.42, [N/A]\n")
    sys.stdout.write(f"compute-1-1, 1, {now}, NVIDIA A100-PCIE-40GB, 42, 18, 1410, 1215, 82.10, [N/A]\n")
    sys.stdout.write("# [compute-apps] host=compute-1-0 rank=0 shared_workload=none\n")
    sys.stdout.write("# [compute-apps] host=compute-1-1 rank=1 shared_workload=none\n")
    sys.exit(0)


def main():
    binary, target_args = parse_args(sys.argv[1:])
    if not binary:
        if any("telemetry" in a or "nvidia-smi" in a for a in sys.argv[1:]):
            simulate_telemetry([])
        sys.stderr.write("mock_cluster_launcher: No executable binary specified\n")
        sys.exit(1)

    binary_name = os.path.basename(binary)
    if "rank_wrapper" in binary_name and "--check-gpus" in target_args:
        simulate_rank_wrapper_check(sys.argv[1:])
    elif "topo_probe" in binary_name:
        simulate_topo_probe(target_args)
    elif "test_async_halo" in binary_name:
        simulate_test_async_halo(target_args)
    elif "kneighbor" in binary_name or "leanmd" in binary_name:
        simulate_benchmark(binary_name, target_args)
    elif "telemetry" in binary_name or "nvidia-smi" in binary_name or any("telemetry" in a for a in sys.argv[1:]):
        simulate_telemetry(target_args)
    else:
        sys.stdout.write(f"Mock execution for {binary_name} succeeded.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
