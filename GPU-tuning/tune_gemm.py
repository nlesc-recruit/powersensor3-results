#!/usr/bin/env python3
import warnings
from collections import OrderedDict
import argparse
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.pmt import PMTObserver
from kernel_tuner.observers.powersensor import PowerSensorObserver
import numpy as np
import os

try:
    from ncu_metrics import *
except ImportError:
    pass

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Device name, used in output filename")
    parser.add_argument("-m", type=int, required=True, help="Size of M axis")
    parser.add_argument("-n", type=int, required=True, help="Size of N axis")
    parser.add_argument("-k", type=int, required=True, help="Size of K axis")
    parser.add_argument("-b", type=int, default=1, help="Size of Batch axis (default: %(default)s)")
    parser.add_argument("--nbit", type=int, choices=[1, 16, 32], required=True, help="Bits per sample")
    parser.add_argument("--kernel", choices=["basic", "opt"], default="opt",
                        help="Tune the basic or opt kernel (default: %(default)s)")
    parser.add_argument("--backend", required=True, choices=["cupy", "hip"], help="Kernel Tuner backend")
    parser.add_argument("--observer", dest="observer_type", required=False, choices=["nvml", "pmt", "powersensor"],
                        help="Kernel Tuner power observer (Default: PMT if backend is HIP, NVML otherwise)")
    parser.add_argument("--ccglib", required=True, help="Path to ccglib directory")
    parser.add_argument(
        "--ncu", action="store_true", help="Enable NCU metrics"
    )
    parser.add_argument(
        "--freq", type=lambda s: [int(item) for item in s.split(',')], help="Tune specified clock frequencies"
    )
    parser.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Overwrite any existing .json files",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    name = args.name
    ccglib_dir = args.ccglib
    kernel_name = f"wmma_complex_gemm_{args.kernel}"
    m_global = args.m
    n_global = args.n
    k_global = args.k
    batch_size = args.b
    nbit = args.nbit
    backend = args.backend

    observer_type = args.observer_type
    # observer default depends on backend
    if observer_type is None:
        if backend == "hip":
            observer_type = "pmt"
        else:
            observer_type = "nvml"

    # on AMD GPUs, the warp size can be 32 or 64 and the shared memory size is different from nvidia
    if backend == "hip":
        from pyhip import hip
        device_properties = hip.hipGetDeviceProperties(0)
        warp_size = device_properties.warpSize
        smem_size = device_properties.sharedMemPerBlock
    else:
        # assume nvidia defaults
        warp_size = 32
        smem_size = 49512

    # block size x is always warp_size, so the other block sizes can be at
    # most 1024 / warp_size
    tune_params = {
        "block_size_x": [warp_size],  # must be warp size
        "block_size_y": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        "block_size_z": [2**i for i in range(0, 6) if warp_size * 2**i <= 1024],
        "M_PER_BLOCK": [2**i for i in range(4, 9)],  # minimum m_per_wmma
        "N_PER_BLOCK": [2**i for i in range(3, 9)],  # minimum n_per_wmma
    }

    # multiple buffers is only supported on nvidia
    # assume that when HIP is used, we are running on an AMD GPU
    if backend == "hip":
        tune_params["NBUFFER"] = [1]
    else:
        tune_params["NBUFFER"] = [1, 2, 4, 8] #[1, 2, 4, 8]

    defines = {key: key for key in tune_params}

    defines["kernel_tuner"] = 1
    defines["BATCH_SIZE"] = batch_size
    defines["M_GLOBAL"] = m_global
    defines["N_GLOBAL"] = n_global
    defines["K_GLOBAL"] = k_global
    defines["N_PER_WARP"] = lambda p: int(p["N_PER_BLOCK"] // p["block_size_y"])
    defines["M_PER_WARP"] = lambda p: int(p["M_PER_BLOCK"] // p["block_size_z"])
    defines["K_PADDING"] = 0
    defines["NBIT"] = nbit
    defines["WARP_SIZE"] = warp_size
    defines["COMPLEX_MIDDLE"] = 1
    defines["A_ROW_MAJOR"] = 1
    defines["B_COL_MAJOR"] = 1
    defines["C_ROW_MAJOR"] = 1

    if nbit == 1:
        kernel_file = "gemm_kernel_int1.cu"
        scaling_factor = 8
        dtype_ab = np.uint8
        dtype_c = np.int32
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 8
        defines["K_PER_WMMA"] = 256
    elif nbit == 16:
        kernel_file = "gemm_kernel_float.cu"
        scaling_factor = 1
        dtype_ab = np.float16
        dtype_c = np.float32
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 16
        defines["K_PER_WMMA"] = 16
    elif nbit == 32:
        kernel_file = "gemm_kernel_float.cu"
        scaling_factor = 1
        dtype_ab = np.float32
        dtype_c = np.float32
        defines["M_PER_WMMA"] = 16
        defines["N_PER_WMMA"] = 16
        defines["K_PER_WMMA"] = 8
    else:
        raise ValueError(f"Invalid nbit: {nbit}")

    A = np.zeros((batch_size, 2, m_global, k_global // scaling_factor), dtype=dtype_ab)
    B = np.zeros((batch_size, 2, n_global, k_global // scaling_factor), dtype=dtype_ab)
    C = np.zeros((batch_size, 2, n_global, m_global), dtype=dtype_c)

    problem_size = (n_global, m_global, batch_size)
    arguments = (C, A, B)

    grid_div = {
        "grid_div_x": lambda p: p["N_PER_BLOCK"],
        "grid_div_y": lambda p: p["M_PER_BLOCK"],
        "grid_div_z": lambda p: 1
    }

    metrics = {
        "TFLOPS": lambda p: 8e-9 * m_global * n_global * k_global * batch_size / p["time"],
        "N_PER_WARP": lambda p: p["N_PER_BLOCK"] // p["block_size_y"],
        "M_PER_WARP": lambda p: p["M_PER_BLOCK"] // p["block_size_z"]
    }

    observers = []
    if observer_type == "pmt":
        if backend == "hip":
            sensor_name = "rocm"
        else:
            sensor_name = "nvidia"
        pmtobserver = PMTObserver({sensor_name: 0}, use_continuous_observer=True)
        observers.append(pmtobserver)
        metrics["Watt"] = lambda p: 1e3 * p[f"{sensor_name}_energy"] / p["time"]
        metrics["TFLOPS/J"] = lambda p: 8e-12 * m_global * n_global * k_global * batch_size / p[f"{sensor_name}_energy"]
    elif observer_type == "nvml":
        nvmlobserver = NVMLObserver(["nvml_energy", "temperature"])
        observers.append(nvmlobserver)
        metrics["Watt"] = lambda p: 1e3 * p["nvml_energy"] / p["time"]
        metrics["TFLOPS/J"] = lambda p: 8e-12 * m_global * n_global * k_global * batch_size / p["nvml_energy"]
    elif observer_type == "powersensor":
        powersensor_observer = PowerSensorObserver(["ps_energy", "ps_power"])
        observers.append(powersensor_observer)
        metrics["Watt"] = lambda p: p["ps_power"]
        metrics["TFLOPS/J"] = lambda p: 8e-12 * m_global * n_global * k_global * batch_size / p["ps_energy"]

    if args.ncu:
        observers.append(get_ncu_observer())
        metrics.update(get_ncu_metrics())

    if args.freq:
        nvmlobserver = NVMLObserver(["core_freq", "temperature"], nvidia_smi_fallback="/cm/shared/package/utils/bin/nvidia-smi-fallback")
        observers.append(nvmlobserver)
        tune_params["nvml_gr_clock"] = args.freq
        tune_params = OrderedDict(tune_params)
        tune_params.move_to_end("nvml_gr_clock", last=False)  # Moves key nvml_gr_clock to the front



    with open(f"{ccglib_dir}/kernels/{kernel_file}", "r") as fp:
        kernel_source = fp.read()

    compiler_options = [f"-I{ccglib_dir}/kernels", f"-I{ccglib_dir}/include", "-std=c++17", "-D__CUDA_NO_HALF_CONVERSIONS__"]

    def restrict(p):
        n_global_per_warp = int(p["N_PER_BLOCK"] // p["block_size_y"])
        m_global_per_warp = int(p["M_PER_BLOCK"] // p["block_size_z"])
        if n_global_per_warp == 0 or m_global_per_warp == 0:
            return False

        # factor 2 for complex
        ab_size = (
            2
            * (p["NBUFFER"] * (p["M_PER_BLOCK"] + p["N_PER_BLOCK"]))
            * defines["K_PER_WMMA"]
            * np.dtype(dtype_ab).itemsize
            / scaling_factor
        )
        c_size = (
            (p["M_PER_BLOCK"] / m_global_per_warp)
            * (p["N_PER_BLOCK"] / n_global_per_warp)
            * defines["M_PER_WMMA"]
            * defines["N_PER_WMMA"]
            * np.dtype(dtype_c).itemsize
        )

        m_is_padded = m_global % p["M_PER_BLOCK"] != 0
        n_is_padded = n_global % p["N_PER_BLOCK"] != 0
        if (m_is_padded or n_is_padded) and c_size > ab_size:
            return False

        smem_buffer_size = ab_size
        valid = (
            p["M_PER_BLOCK"] % m_global_per_warp == 0
            and p["N_PER_BLOCK"] % n_global_per_warp == 0
            and m_global_per_warp % defines["M_PER_WMMA"] == 0
            and n_global_per_warp % defines["N_PER_WMMA"] == 0
            and smem_buffer_size <= smem_size
            and p["block_size_x"] * p["block_size_y"] * p["block_size_z"] <= 1024
        )
        return valid

    filename_cache = (
        f"tuning/{name}_{kernel_name}_{nbit}bit_{batch_size}x{m_global}x{n_global}x{k_global}.json"
    )
    if args.overwrite and os.path.exists(filename_cache):
            os.remove(filename_cache)

    kt.tune_kernel(kernel_name, kernel_source, problem_size, arguments, tune_params,
                   restrictions=restrict,
                   compiler_options=compiler_options,
                   # strategy="dual_annealing", strategy_options=dict(max_fevals=200),
                   # strategy="random_sample",
                   cache=filename_cache,
                   metrics=metrics, observers=observers,
                   defines=defines, lang=backend, verbose=True, objective="TFLOPS/J", objective_higher_is_better=True,
                   **grid_div)
