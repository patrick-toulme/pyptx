"""Benchmark the maintained high-performance Hopper GEMM.

This mirrors fast.cu kernel12:
- A: row-major [M, K]
- B: row-major [N, K]
- C buffer: row-major [N, M] holding logical (A @ B.T).T
- Hilbert schedule and 3D TMA descriptors match the CUDA reference

Usage:
    python benchmarks/bench_hopper_gemm.py
    python benchmarks/bench_hopper_gemm.py --size 8192 --iters 8
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys

import torch
from cuda.bindings import driver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.hopper.gemm_highperf_hopper import SMEM_TOTAL, gemm_warp_specialized
from pyptx.jax_support import synthesize_tma_descriptor_3d
from pyptx.types import bf16


BM, BN, BK = 128, 256, 64
NUM_THREADS = 384
CLUSTER_M, CLUSTER_N = 2, 1
NUM_SM = 128
SPACE_LEN = 128
NUM_CONSUMERS = 2


def _cuda_ok(err, where: str) -> None:
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{where}: {err}")


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _d2xy(n: int, d: int) -> tuple[int, int]:
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def create_hilbert(m_tiles: int, n_tiles: int, cores: int) -> list[int]:
    dim = 1 << ((max(m_tiles, n_tiles) - 1).bit_length())
    space = [-1] * (NUM_SM * SPACE_LEN)
    fcores = 64
    total = 0
    pos = [[] for _ in range(cores)]
    core = 0

    for i in range(dim * dim):
        x, y = _d2xy(dim, i)
        if x < m_tiles and y < n_tiles:
            if len(pos[core]) >= SPACE_LEN:
                raise RuntimeError("Hilbert schedule exceeded SPACE_LEN")
            pos[core].append((x << 16) | y)
            total += 1
            core += 1
            if core == fcores:
                core = 0

    core = fcores
    for i in range(fcores):
        if len(pos[-1]) >= len(pos[0]) - 1:
            break
        pos[core].append(pos[i].pop())
        core += 1
        if core == cores:
            core = fcores

    for i, entries in enumerate(pos):
        base = i * SPACE_LEN
        space[base:base + len(entries)] = entries

    if total != m_tiles * n_tiles:
        raise RuntimeError(f"Hilbert schedule mismatch: {total} != {m_tiles * n_tiles}")

    return space


def create_linear_once(m_tiles: int, n_tiles: int, cores: int) -> list[int]:
    space = [-1] * (NUM_SM * SPACE_LEN)
    total = m_tiles * n_tiles
    count = min(total, cores)
    for i in range(count):
        tile_m = i // n_tiles
        tile_n = i % n_tiles
        space[i * SPACE_LEN] = (tile_m << 16) | tile_n
    return space


def create_single_cluster_linear(m_tiles: int, n_tiles: int) -> list[int]:
    total = m_tiles * n_tiles
    if total > SPACE_LEN:
        raise RuntimeError(f"single_cluster_linear needs <= {SPACE_LEN} tiles, got {total}")
    space = [-1] * (NUM_SM * SPACE_LEN)
    idx = 0
    for tile_m in range(m_tiles):
        for tile_n in range(n_tiles):
            space[idx] = (tile_m << 16) | tile_n
            idx += 1
    return space


def create_single_cluster_repeat(tile: int, count: int) -> list[int]:
    if count > SPACE_LEN:
        raise RuntimeError(f"single_cluster_repeat needs <= {SPACE_LEN} tiles, got {count}")
    space = [-1] * (NUM_SM * SPACE_LEN)
    for i in range(count):
        space[i] = tile
    return space


def make_tma_3d(
    ptr: int,
    height: int,
    width: int,
    box_major: int,
    box_minor: int,
    *,
    swizzle_128b: bool,
    padding: bool,
) -> bytes:
    u64 = driver.cuuint64_t
    u32 = driver.cuuint32_t
    swizzle = (
        driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        if swizzle_128b
        else driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    )
    err, tmap = driver.cuTensorMapEncodeTiled(
        driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        ptr,
        [u64(64), u64(int(height)), u64(int(width) // 64)],
        [u64(2 * int(width)), u64(128)],
        [u32(72 if padding else 64), u32(int(box_major)), u32(int(box_minor) // 64)],
        [u32(1), u32(1), u32(1)],
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    _cuda_ok(err, "cuTensorMapEncodeTiled")
    return ctypes.string_at(int(tmap.getPtr()), 128)


def load_kernel(kernel_obj):
    ptx = kernel_obj.ptx().encode()
    err, module = driver.cuModuleLoadData(ptx)
    _cuda_ok(err, "cuModuleLoadData")
    err, fn = driver.cuModuleGetFunction(module, b"gemm_warp_specialized")
    _cuda_ok(err, "cuModuleGetFunction")
    err, = driver.cuFuncSetAttribute(
        fn,
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SMEM_TOTAL,
    )
    _cuda_ok(err, "cuFuncSetAttribute")
    return module, fn


def make_kernel_params(
    a_ptr: int,
    b_ptr: int,
    hilbert_ptr: int,
    c_ptr: int,
    M: int,
    N: int,
    K: int,
    tma_a_devptr: int,
    tma_b_devptr: int,
    tma_c_devptr: int,
):
    a_ptr_val = ctypes.c_uint64(int(a_ptr))
    b_ptr_val = ctypes.c_uint64(int(b_ptr))
    h_ptr = ctypes.c_uint64(int(hilbert_ptr))
    c_ptr_val = ctypes.c_uint64(int(c_ptr))
    m_val = ctypes.c_uint32(M)
    n_val = ctypes.c_uint32(N)
    k_val = ctypes.c_uint32(K)
    a_desc_ptr = ctypes.c_uint64(int(tma_a_devptr))
    b_desc_ptr = ctypes.c_uint64(int(tma_b_devptr))
    c_desc_ptr = ctypes.c_uint64(int(tma_c_devptr))
    params = (ctypes.c_void_p * 10)()
    params[0] = ctypes.cast(ctypes.byref(a_ptr_val), ctypes.c_void_p).value
    params[1] = ctypes.cast(ctypes.byref(b_ptr_val), ctypes.c_void_p).value
    params[2] = ctypes.cast(ctypes.byref(h_ptr), ctypes.c_void_p).value
    params[3] = ctypes.cast(ctypes.byref(c_ptr_val), ctypes.c_void_p).value
    params[4] = ctypes.cast(ctypes.byref(m_val), ctypes.c_void_p).value
    params[5] = ctypes.cast(ctypes.byref(n_val), ctypes.c_void_p).value
    params[6] = ctypes.cast(ctypes.byref(k_val), ctypes.c_void_p).value
    params[7] = ctypes.cast(ctypes.byref(a_desc_ptr), ctypes.c_void_p).value
    params[8] = ctypes.cast(ctypes.byref(b_desc_ptr), ctypes.c_void_p).value
    params[9] = ctypes.cast(ctypes.byref(c_desc_ptr), ctypes.c_void_p).value
    return params, (
        a_ptr_val,
        b_ptr_val,
        h_ptr,
        c_ptr_val,
        m_val,
        n_val,
        k_val,
        a_desc_ptr,
        b_desc_ptr,
        c_desc_ptr,
    )


def upload_tma_descriptor(blob: bytes, stream) -> int:
    buf = ctypes.create_string_buffer(blob, 128)
    err, dev_ptr = driver.cuMemAlloc(128)
    _cuda_ok(err, "cuMemAlloc(TMA)")
    err, = driver.cuMemcpyHtoDAsync(dev_ptr, ctypes.addressof(buf), 128, stream)
    _cuda_ok(err, "cuMemcpyHtoDAsync(TMA)")
    return int(dev_ptr)


def materialize_tma_descriptor_3d(
    tensor: torch.Tensor,
    *,
    box_major: int,
    box_minor: int,
    swizzle_128b: bool,
    padding: bool,
    stream,
) -> tuple[int, object]:
    host_tmap, host_blob_ptr, device_blob_ptr = synthesize_tma_descriptor_3d(
        tensor.shape[0],
        tensor.shape[1],
        bf16,
        box_major,
        box_minor,
        swizzle_128b=swizzle_128b,
        padding=padding,
    )
    replace = ctypes.CDLL("libcuda.so.1").cuTensorMapReplaceAddress
    replace.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    replace.restype = ctypes.c_int
    rc = replace(ctypes.c_void_p(host_blob_ptr), ctypes.c_void_p(tensor.data_ptr()))
    if rc != 0:
        raise RuntimeError(f"cuTensorMapReplaceAddress failed: {rc}")
    err, = driver.cuMemcpyHtoDAsync(device_blob_ptr, host_blob_ptr, 128, stream)
    _cuda_ok(err, "cuMemcpyHtoDAsync(TMA 3D)")
    return int(device_blob_ptr), host_tmap


def launch(fn, params, stream) -> None:
    attr = driver.CUlaunchAttribute()
    attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.value.clusterDim.x = CLUSTER_M
    attr.value.clusterDim.y = CLUSTER_N
    attr.value.clusterDim.z = 1

    cfg = driver.CUlaunchConfig()
    cfg.gridDimX = NUM_SM
    cfg.gridDimY = 1
    cfg.gridDimZ = 1
    cfg.blockDimX = NUM_THREADS
    cfg.blockDimY = 1
    cfg.blockDimZ = 1
    cfg.sharedMemBytes = SMEM_TOTAL
    cfg.hStream = stream
    cfg.attrs = [attr]
    cfg.numAttrs = 1

    err, = driver.cuLaunchKernelEx(cfg, fn, ctypes.addressof(params), 0)
    _cuda_ok(err, "cuLaunchKernelEx")


def bench(fn, params, stream, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        launch(fn, params, stream)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch(fn, params, stream)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iters


def run_benchmark(
    *,
    size: int = 8192,
    warmup: int = 2,
    iters: int = 8,
    torch_ref: bool = False,
    schedule: str = "hilbert",
    limit_tiles: int = 0,
) -> dict[str, float | int]:
    driver.cuInit(0)
    _ = torch.zeros(1, device="cuda")

    M = N = K = size
    assert M % (BM * CLUSTER_M) == 0
    assert N % (BN * CLUSTER_N) == 0
    assert K % BK == 0

    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    c_out = torch.zeros((N, M), device="cuda", dtype=torch.bfloat16)

    m_tiles = math.ceil(M / (BM * CLUSTER_M))
    n_tiles = math.ceil(N / (BN * CLUSTER_N))
    cores = NUM_SM // (CLUSTER_M * CLUSTER_N)
    if schedule == "hilbert":
        schedule_host = create_hilbert(m_tiles, n_tiles, cores)
    elif schedule == "linear_once":
        schedule_host = create_linear_once(m_tiles, n_tiles, cores)
    elif schedule == "single_cluster_linear":
        schedule_host = create_single_cluster_linear(m_tiles, n_tiles)
    else:
        schedule_host = create_single_cluster_repeat(0, limit_tiles or 2)
    if limit_tiles > 0:
        seen = 0
        for i, val in enumerate(schedule_host):
            if val != -1:
                seen += 1
                if seen > limit_tiles:
                    schedule_host[i] = -1
    hilbert = torch.tensor(schedule_host, device="cuda", dtype=torch.int32)
    active_tiles = int((hilbert != -1).sum().item())

    _, fn = load_kernel(gemm_warp_specialized)
    stream = torch.cuda.current_stream().cuda_stream
    tma_a_dev, _ = materialize_tma_descriptor_3d(
        a,
        box_major=BM,
        box_minor=BK,
        swizzle_128b=True,
        padding=False,
        stream=stream,
    )
    tma_b_dev, _ = materialize_tma_descriptor_3d(
        b,
        box_major=BN,
        box_minor=BK,
        swizzle_128b=True,
        padding=False,
        stream=stream,
    )
    tma_c_dev, _ = materialize_tma_descriptor_3d(
        c_out,
        box_major=BN,
        box_minor=BM // NUM_CONSUMERS,
        swizzle_128b=False,
        padding=True,
        stream=stream,
    )
    params, _ = make_kernel_params(
        a.data_ptr(),
        b.data_ptr(),
        hilbert.data_ptr(),
        c_out.data_ptr(),
        M,
        N,
        K,
        tma_a_dev,
        tma_b_dev,
        tma_c_dev,
    )

    launch(fn, params, stream)
    torch.cuda.synchronize()
    torch_diff = None
    torch_num_diff = None
    if torch_ref:
        c_torch = torch.matmul(a.float(), b.float().T).to(torch.bfloat16).T.contiguous()
        torch_diff = float((c_out.float() - c_torch.float()).abs().max().item())
        torch_num_diff = int((c_out != c_torch).sum().item())

    flops = 2 * M * N * K
    ms = bench(fn, params, stream, warmup=warmup, iters=iters)
    tflops = flops / (ms / 1e3) / 1e12
    result: dict[str, float | int] = {
        "size": size,
        "smem": SMEM_TOTAL,
        "scheduled_tiles": active_tiles,
        "ms": ms,
        "tflops": tflops,
    }
    if torch_diff is not None:
        result["torch_max_diff"] = torch_diff
    if torch_num_diff is not None:
        result["torch_num_diff"] = torch_num_diff
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--torch-ref", action="store_true")
    parser.add_argument(
        "--schedule",
        choices=("hilbert", "linear_once", "single_cluster_linear", "single_cluster_repeat"),
        default="hilbert",
    )
    parser.add_argument("--limit-tiles", type=int, default=0)
    args = parser.parse_args()

    result = run_benchmark(
        size=args.size,
        warmup=args.warmup,
        iters=args.iters,
        torch_ref=args.torch_ref,
        schedule=args.schedule,
        limit_tiles=args.limit_tiles,
    )
    print(f"size={args.size}x{args.size}x{args.size}")
    print(f"smem={result['smem']} bytes")
    print(f"schedule={args.schedule} scheduled tiles={result['scheduled_tiles']}")
    if args.torch_ref:
        print(
            f"kernel_vs_torch max_diff={result.get('torch_max_diff', float('nan')):g} "
            f"num_diff={int(result.get('torch_num_diff', -1))}"
        )
    print(f"kernel: {result['ms']:.3f} ms  {result['tflops']:.1f} TFLOPS")


if __name__ == "__main__":
    main()
