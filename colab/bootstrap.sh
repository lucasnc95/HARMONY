#!/usr/bin/env bash
# ==============================================================================
# HARMONY Pre-Cluster Validation: Colab Environment Bootstrap
# Idempotent environment setup and OpenCL GPU discovery for single-node Colab.
# ==============================================================================
set -euo pipefail

# Ensure non-interactive Debian package operations
export DEBIAN_FRONTEND=noninteractive

# Configure Open MPI flags required when executing as root in Colab containers
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

resolve_sudo_cmd() {
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        echo "sudo -n"
    fi
}

# Persist OMPI root flags in /etc/environment and profile.d if writable or non-interactive sudo available
persist_ompi_root_flags() {
    local sudo_cmd
    sudo_cmd="$(resolve_sudo_cmd)"

    local env_file="${ENV_FILE:-/etc/environment}"
    local profile_dir="${OPENMPI_PROFILE_DIR:-/etc/profile.d}"

    # 1. /etc/environment
    if [ -w "${env_file}" ] || ([ -n "${sudo_cmd}" ] && [ -f "${env_file}" ]); then
        if ! grep -q "OMPI_ALLOW_RUN_AS_ROOT=" "${env_file}" 2>/dev/null; then
            echo "OMPI_ALLOW_RUN_AS_ROOT=1" | ${sudo_cmd} tee -a "${env_file}" >/dev/null 2>&1 || true
        fi
        if ! grep -q "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=" "${env_file}" 2>/dev/null; then
            echo "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" | ${sudo_cmd} tee -a "${env_file}" >/dev/null 2>&1 || true
        fi
    fi

    # 2. /etc/profile.d/openmpi_root.sh
    if [ -d "${profile_dir}" ] && ([ -w "${profile_dir}" ] || [ -n "${sudo_cmd}" ]); then
        cat << 'EOF' | ${sudo_cmd} tee "${profile_dir}/openmpi_root.sh" >/dev/null 2>&1 || true
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
EOF
        ${sudo_cmd} chmod +x "${profile_dir}/openmpi_root.sh" >/dev/null 2>&1 || true
    fi

    # 3. User bashrc
    if [ -n "${HOME:-}" ] && [ -w "${HOME}/.bashrc" ]; then
        if ! grep -q "OMPI_ALLOW_RUN_AS_ROOT=" "${HOME}/.bashrc" 2>/dev/null; then
            echo "export OMPI_ALLOW_RUN_AS_ROOT=1" >> "${HOME}/.bashrc" 2>/dev/null || true
            echo "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> "${HOME}/.bashrc" 2>/dev/null || true
        fi
    fi
}

persist_ompi_root_flags

echo "=== [Colab Bootstrap] Installing system packages ==="
if [[ "${BOOTSTRAP_SKIP_APT:-0}" != "1" ]]; then
    sudo_cmd="$(resolve_sudo_cmd)"

    if command -v apt-get >/dev/null 2>&1; then
        ${sudo_cmd} apt-get update -qq
        ${sudo_cmd} apt-get install -y --no-install-recommends \
            build-essential \
            openmpi-bin libopenmpi-dev \
            ocl-icd-opencl-dev opencl-headers \
            clinfo
    else
        echo "WARNING: apt-get package manager not found. Skipping package installation." >&2
    fi
else
    echo "Notice: BOOTSTRAP_SKIP_APT=1 set. Skipping apt package installation."
fi

# Ensure OpenCL vendor ICD registry exists for NVIDIA runtime
ensure_nvidia_icd() {
    local sudo_cmd
    sudo_cmd="$(resolve_sudo_cmd)"

    local vendor_dir="${OPENCL_VENDOR_DIR:-/etc/OpenCL/vendors}"
    if [ -d "${vendor_dir}" ] || [ "$(id -u)" -eq 0 ] || [ -n "${sudo_cmd}" ] || [ -w "$(dirname "${vendor_dir}" 2>/dev/null || true)" ]; then
        ${sudo_cmd} mkdir -p "${vendor_dir}" 2>/dev/null || true
        if [ ! -f "${vendor_dir}/nvidia.icd" ] && [ -d "${vendor_dir}" ]; then
            echo "libnvidia-opencl.so.1" | ${sudo_cmd} tee "${vendor_dir}/nvidia.icd" >/dev/null 2>&1 || true
        fi
    fi
}

ensure_nvidia_icd

echo "=== [Colab Bootstrap] Validating GPU and OpenCL Environment ==="

# 1. Validate nvidia-smi command presence
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi command not found!" >&2
    echo "No NVIDIA GPU accelerator detected on this runtime." >&2
    echo "In Google Colab, change runtime accelerator via: Runtime -> Change runtime type -> T4 GPU." >&2
    exit 1
fi

# 2. Check communication with GPU driver
if ! gpu_info="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1)"; then
    echo "ERROR: nvidia-smi failed to communicate with NVIDIA GPU driver." >&2
    echo "Diagnostics: ${gpu_info}" >&2
    exit 1
fi
echo "NVIDIA GPU Detected: ${gpu_info}"

# 3. Validate clinfo command presence
if ! command -v clinfo >/dev/null 2>&1; then
    echo "ERROR: clinfo command not found. OpenCL diagnostic tools failed to install." >&2
    exit 1
fi

# 4. Query clinfo -l and validate platforms
echo "--- clinfo -l summary ---"
if ! clinfo_list="$(clinfo -l 2>&1)"; then
    echo "ERROR: clinfo -l command execution failed." >&2
    echo "Diagnostics: ${clinfo_list}" >&2
    exit 1
fi
echo "${clinfo_list}"

# Reject if 0 platforms or empty device output
if echo "${clinfo_list}" | grep -q "0 platforms" || [ -z "$(echo "${clinfo_list}" | tr -d '[:space:]')" ]; then
    echo "ERROR: No OpenCL platforms detected by clinfo." >&2
    echo "Ensure NVIDIA driver OpenCL ICD is registered in /etc/OpenCL/vendors/nvidia.icd." >&2
    exit 1
fi

# 5. Guard against CPU-only fallback: require at least one OpenCL GPU device
clinfo_full="$(clinfo 2>&1)" || true

has_gpu_device=0

# Check CL_DEVICE_TYPE_GPU or GPU in Device Type field
if echo "${clinfo_full}" | grep -Ei "Device Type\s*:\s*.*(GPU|CL_DEVICE_TYPE_GPU)" >/dev/null 2>&1; then
    has_gpu_device=1
fi

# Also check if device listed in clinfo -l matches detected GPU name
if [ "${has_gpu_device}" -eq 0 ]; then
    first_gpu_name="$(echo "${gpu_info}" | head -n1 | cut -d',' -f1 | xargs || true)"
    if [ -n "${first_gpu_name}" ] && echo "${clinfo_list}" | grep -qi "${first_gpu_name}"; then
        has_gpu_device=1
    fi
fi

if [ "${has_gpu_device}" -eq 0 ]; then
    echo "ERROR: No OpenCL GPU device detected! CPU-only fallback is guarded against and rejected." >&2
    echo "Google Colab Pre-Cluster Validation requires a physical OpenCL GPU device." >&2
    echo "Platforms detected:" >&2
    echo "${clinfo_list}" >&2
    exit 1
fi

echo "=== [Colab Bootstrap] Success: Valid OpenCL GPU Environment Configured ==="
exit 0
