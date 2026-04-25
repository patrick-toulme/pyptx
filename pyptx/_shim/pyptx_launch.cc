// pyptx launch shim.
//
// Provides exactly one XLA FFI custom call target ("pyptx_launch") that
// decodes the CUDA stream + input/output buffers + cubin_handle attribute
// out of an XLA_FFI_CallFrame and calls cuLaunchKernel on the stream XLA
// is sequencing on.
//
// Everything upstream of this (tracing, PTX emission, driver JIT to a
// CUfunction, cache) stays in Python. The Python side populates a
// process-local registry via pyptx_shim_register_launch() before any
// jit'd call actually executes. At launch time, the FFI handler reads
// the registry, marshals kernel argument pointers, and calls into
// libcuda.so via dlsym-resolved cuLaunchKernel. We dlsym instead of
// linking -lcuda so the shim builds cleanly on boxes that only have
// jaxlib's headers, not the full CUDA SDK.
//
// Build:
//   g++ -std=c++17 -O2 -fPIC -shared \
//       -I $JAXLIB_INCLUDE \
//       pyptx_launch.cc \
//       -o libpyptx_shim.so \
//       -ldl
//
// Link time needs: libdl. Runtime needs: libcuda.so.1 (present on any
// NVIDIA-driver box, where jax[cuda] works).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef PYPTX_NO_XLA_FFI
#include "xla/ffi/api/ffi.h"
#endif

// ---------------------------------------------------------------------------
// CUDA driver API types
// ---------------------------------------------------------------------------
// Prefer the real CUDA headers when they're available on the machine so
// extensible-launch structs like CUlaunchAttribute always match the
// driver exactly. Fall back to local forward declarations on boxes that
// only have jaxlib headers.

#if __has_include("/usr/local/cuda/include/cuda.h")
#include "/usr/local/cuda/include/cuda.h"
#define PYPTX_SHIM_USING_SYSTEM_CUDA_H 1
#elif __has_include("/usr/include/cuda.h")
#include "/usr/include/cuda.h"
#define PYPTX_SHIM_USING_SYSTEM_CUDA_H 1
#else
extern "C" {
typedef struct CUstream_st* CUstream;
typedef struct CUfunc_st* CUfunction;
typedef int CUresult;
}

typedef struct {
  unsigned int x, y, z;
} CUlaunchAttributeValue_clusterDim;

typedef union {
  char pad[64];
  CUlaunchAttributeValue_clusterDim clusterDim;
} CUlaunchAttributeValue;

typedef struct {
  unsigned int id;
  char pad[8 - sizeof(unsigned int)];
  CUlaunchAttributeValue value;
} CUlaunchAttribute;

typedef struct {
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  CUlaunchAttribute* attrs;
  unsigned int numAttrs;
} CUlaunchConfig;
#endif

using cuLaunchKernel_t = CUresult (*)(
    CUfunction f,
    unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra);

using cuLaunchKernelEx_t = CUresult (*)(
    const CUlaunchConfig* config,
    CUfunction f,
    void** kernelParams,
    void** extra);
using cuGetProcAddress_t = CUresult (*)(
    const char* symbol,
    void** pfn,
    int cudaVersion,
    uint64_t flags);

// cuTensorMapReplaceAddress(CUtensorMap *tensorMap, void *globalAddress)
//
// Modifies an existing 128-byte CUtensorMap to point at a new global
// address. Used per-launch to patch a precomputed descriptor with the
// buffer pointer XLA hands us. Hopper+.
using cuTensorMapReplaceAddress_t = CUresult (*)(void* /*CUtensorMap*/, void* globalAddress);

// cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream)
using cuMemcpyHtoDAsync_t = CUresult (*)(uint64_t /*CUdeviceptr*/, const void*, size_t, CUstream);

static std::once_flag g_cuda_once;
static cuLaunchKernel_t g_cuLaunchKernel = nullptr;
static cuLaunchKernelEx_t g_cuLaunchKernelEx = nullptr;
static cuGetProcAddress_t g_cuGetProcAddress = nullptr;
static cuTensorMapReplaceAddress_t g_cuTensorMapReplaceAddress = nullptr;
static cuMemcpyHtoDAsync_t g_cuMemcpyHtoDAsync = nullptr;
static std::string g_cuda_load_error;
#ifdef PYPTX_SHIM_USING_SYSTEM_CUDA_H
static constexpr CUlaunchAttributeID kCuLaunchAttributeClusterDimension =
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
#else
static constexpr unsigned int kCuLaunchAttributeClusterDimension = 4;
#endif

static void LoadCudaDriver() {
  std::call_once(g_cuda_once, []() {
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
      const char* err = dlerror();
      g_cuda_load_error =
          std::string("pyptx shim: failed to dlopen libcuda.so.1: ") +
          (err ? err : "unknown");
      return;
    }
    g_cuLaunchKernel =
        reinterpret_cast<cuLaunchKernel_t>(dlsym(handle, "cuLaunchKernel"));
    if (!g_cuLaunchKernel) {
      g_cuda_load_error = "pyptx shim: dlsym(cuLaunchKernel) returned null";
      return;
    }
    // Prefer the plain exported symbol first. This matches the known-good
    // cuda-python driver path more closely than the PTDS or cuGetProcAddress
    // variants, and avoids subtle ABI/default-stream mismatches.
    g_cuLaunchKernelEx =
        reinterpret_cast<cuLaunchKernelEx_t>(dlsym(handle, "cuLaunchKernelEx"));
    if (!g_cuLaunchKernelEx) {
      g_cuLaunchKernelEx = reinterpret_cast<cuLaunchKernelEx_t>(
          dlsym(handle, "cuLaunchKernelEx_ptsz"));
    }
    if (!g_cuLaunchKernelEx) {
      g_cuGetProcAddress =
          reinterpret_cast<cuGetProcAddress_t>(dlsym(handle, "cuGetProcAddress"));
      if (g_cuGetProcAddress) {
        void* proc = nullptr;
        if (g_cuGetProcAddress(
                "cuLaunchKernelEx",
                &proc,
                12000,
#ifdef CU_GET_PROC_ADDRESS_DEFAULT
                CU_GET_PROC_ADDRESS_DEFAULT
#else
                0
#endif
                ) == 0 && proc != nullptr) {
          g_cuLaunchKernelEx = reinterpret_cast<cuLaunchKernelEx_t>(proc);
        }
      }
    }
    // TMA helpers — optional; only required by kernels that register
    // TMA specs. If these aren't present (e.g. old driver), launches
    // still work for non-TMA kernels.
    g_cuTensorMapReplaceAddress =
        reinterpret_cast<cuTensorMapReplaceAddress_t>(
            dlsym(handle, "cuTensorMapReplaceAddress"));
    g_cuMemcpyHtoDAsync =
        reinterpret_cast<cuMemcpyHtoDAsync_t>(
            dlsym(handle, "cuMemcpyHtoDAsync_v2"));
    if (!g_cuMemcpyHtoDAsync) {
      g_cuMemcpyHtoDAsync =
          reinterpret_cast<cuMemcpyHtoDAsync_t>(
              dlsym(handle, "cuMemcpyHtoDAsync"));
    }
  });
}

// ---------------------------------------------------------------------------
// Launch-config registry (populated from Python)
// ---------------------------------------------------------------------------

namespace ffi = xla::ffi;

// Per-launch TMA descriptor spec.
//
// At compile time Python pre-allocates a host-side 128-byte CUtensorMap
// blob (via cuTensorMapEncodeTiled with a placeholder address) and a
// device-side 128-byte buffer. It hands both pointers + the XLA arg
// index whose buffer pointer should replace the global-address field to
// the shim.
//
// At each launch we:
//   1. Read buffer pointer = args[xla_arg_index].untyped_data()
//   2. cuTensorMapReplaceAddress(host_blob, buffer_pointer)
//   3. cuMemcpyHtoDAsync(device_blob, host_blob, 128, stream)
//   4. Push device_blob into kernel_params (8-byte .param .u64)
struct TmaSpec {
  uint32_t xla_arg_index;  // which XLA input buffer owns the data
  void* host_blob;         // pointer to 128-byte host-side CUtensorMap
  uint64_t device_blob;    // 128-byte device-side buffer (CUdeviceptr)
};

struct ScalarParam {
  uint64_t value_bits;
  uint32_t size_bytes;
};

struct LaunchConfig {
  CUfunction fn;
  uint32_t grid_x, grid_y, grid_z;
  uint32_t block_x, block_y, block_z;
  uint32_t cluster_x, cluster_y, cluster_z;
  uint32_t smem_bytes;
  std::vector<ScalarParam> scalar_params;
  // TMA specs in the order the emitted PTX's entry signature expects
  // them. Each one contributes one .param .u64 slot at the tail of the
  // kernel parameter list.
  std::vector<TmaSpec> tma_specs;
};

static std::mutex g_registry_mu;
static std::unordered_map<int64_t, LaunchConfig> g_registry;

extern "C" void pyptx_shim_register_launch(
    int64_t handle,
    void* cu_function,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint32_t cx, uint32_t cy, uint32_t cz,
    uint32_t smem) {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  g_registry[handle] = LaunchConfig{
      reinterpret_cast<CUfunction>(cu_function),
      gx, gy, gz,
      bx, by, bz,
      cx, cy, cz,
      smem,
      {},  // scalar_params
      {},  // tma_specs
  };
}

extern "C" __attribute__((visibility("default"))) void pyptx_shim_add_scalar_param(
    int64_t handle,
    uint64_t value_bits,
    uint32_t size_bytes) {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  auto it = g_registry.find(handle);
  if (it == g_registry.end()) return;
  it->second.scalar_params.push_back(ScalarParam{value_bits, size_bytes});
}

// Register a TMA spec for a handle. The host blob must live at least
// until pyptx_shim_clear_registry is called; the device blob is freed
// by Python via cuMemFree when the kernel is retired. The xla_arg_index
// is the index of the XLA input buffer whose device pointer should
// replace the descriptor's globalAddress field each launch.
//
// The order of calls defines the order of the extra .param .u64 slots
// at the tail of the emitted kernel entry signature.
extern "C" void pyptx_shim_add_tma_spec(
    int64_t handle,
    uint32_t xla_arg_index,
    void* host_blob,
    uint64_t device_blob) {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  auto it = g_registry.find(handle);
  if (it == g_registry.end()) return;
  it->second.tma_specs.push_back(TmaSpec{xla_arg_index, host_blob, device_blob});
}

extern "C" void pyptx_shim_clear_registry() {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  g_registry.clear();
}

extern "C" size_t pyptx_shim_registry_size() {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  return g_registry.size();
}

// Returns 1 if handle present, 0 otherwise. Useful for Python-side sanity.
extern "C" int pyptx_shim_has_handle(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_registry_mu);
  return g_registry.count(handle) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Shared launch core — framework agnostic
// ---------------------------------------------------------------------------
//
// Given a cubin_handle, an already-resolved list of device buffer
// pointers (inputs-then-outputs, before TMA descriptor appending),
// and a CUstream, do the TMA patch-and-upload then call
// cuLaunchKernel. This is the common core between the XLA FFI
// handler (JAX path) and the raw ctypes entry point (PyTorch path).
//
// Returns 0 on success, a negative error code on failure. Fills
// ``err_msg`` (if non-null) with a 128-byte description on failure.

static int PyptxLaunchCore(
    int64_t cubin_handle,
    CUstream stream,
    void** buffer_ptrs_in,
    size_t n_buffers,
    char* err_msg) {
  LoadCudaDriver();
  if (!g_cuLaunchKernel) {
    if (err_msg) {
      std::snprintf(err_msg, 128, "%s", g_cuda_load_error.c_str());
    }
    return -1;
  }

  LaunchConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_registry_mu);
    auto it = g_registry.find(cubin_handle);
    if (it == g_registry.end()) {
      if (err_msg) {
        std::snprintf(err_msg, 128,
                      "pyptx: no launch config registered for handle %lld",
                      static_cast<long long>(cubin_handle));
      }
      return -2;
    }
    cfg = it->second;
  }

  std::vector<std::array<unsigned char, 8>> scalar_values;
  scalar_values.reserve(cfg.scalar_params.size());
  for (const ScalarParam& spec : cfg.scalar_params) {
    std::array<unsigned char, 8> bytes{};
    std::memcpy(bytes.data(), &spec.value_bits, spec.size_bytes);
    scalar_values.push_back(bytes);
  }

  std::vector<uint64_t> tma_device_ptrs;
  tma_device_ptrs.reserve(cfg.tma_specs.size());

  // TMA descriptor patch-and-upload, one per spec.
  if (!cfg.tma_specs.empty()) {
    if (!g_cuTensorMapReplaceAddress || !g_cuMemcpyHtoDAsync) {
      if (err_msg) {
        std::snprintf(err_msg, 128,
                      "pyptx: TMA requires driver 12.0+ (missing "
                      "cuTensorMapReplaceAddress / cuMemcpyHtoDAsync)");
      }
      return -3;
    }
    for (const TmaSpec& spec : cfg.tma_specs) {
      if (spec.xla_arg_index >= n_buffers) {
        if (err_msg) {
          std::snprintf(err_msg, 128,
                        "pyptx: TMA spec xla_arg_index=%u out of range (%zu)",
                        spec.xla_arg_index, n_buffers);
        }
        return -4;
      }
      void* data_ptr = buffer_ptrs_in[spec.xla_arg_index];
      CUresult err = g_cuTensorMapReplaceAddress(spec.host_blob, data_ptr);
      if (err != 0) {
        if (err_msg) {
          std::snprintf(err_msg, 128,
                        "cuTensorMapReplaceAddress failed: %d", (int)err);
        }
        return -5;
      }
      err = g_cuMemcpyHtoDAsync(spec.device_blob, spec.host_blob, 128, stream);
      if (err != 0) {
        if (err_msg) {
          std::snprintf(err_msg, 128,
                        "cuMemcpyHtoDAsync (TMA desc) failed: %d", (int)err);
        }
        return -6;
      }
      tma_device_ptrs.push_back(spec.device_blob);
    }
  }

  // cuLaunchKernel expects kernelParams as an array of pointers to each
  // argument value. Entry-param order is:
  //   1. regular buffer pointers (inputs then outputs)
  //   2. scalar raw params
  //   3. synthesized TMA descriptor pointers
  std::vector<void*> kernel_params;
  kernel_params.reserve(n_buffers + scalar_values.size() + tma_device_ptrs.size());
  for (size_t i = 0; i < n_buffers; ++i) {
    kernel_params.push_back(static_cast<void*>(&buffer_ptrs_in[i]));
  }
  for (auto& scalar : scalar_values) {
    kernel_params.push_back(static_cast<void*>(scalar.data()));
  }
  for (auto& tma_ptr : tma_device_ptrs) {
    kernel_params.push_back(static_cast<void*>(&tma_ptr));
  }

  CUresult err = static_cast<CUresult>(0);
  if (cfg.cluster_x > 1 || cfg.cluster_y > 1 || cfg.cluster_z > 1) {
    if (!g_cuLaunchKernelEx) {
      if (err_msg) {
        std::snprintf(err_msg, 128,
                      "pyptx: cluster launch requires cuLaunchKernelEx");
      }
      return -8;
    }
    CUlaunchAttribute attr{};
    attr.id = kCuLaunchAttributeClusterDimension;
    attr.value.clusterDim.x = cfg.cluster_x;
    attr.value.clusterDim.y = cfg.cluster_y;
    attr.value.clusterDim.z = cfg.cluster_z;
    CUlaunchConfig launch_cfg{};
    launch_cfg.gridDimX = cfg.grid_x;
    launch_cfg.gridDimY = cfg.grid_y;
    launch_cfg.gridDimZ = cfg.grid_z;
    launch_cfg.blockDimX = cfg.block_x;
    launch_cfg.blockDimY = cfg.block_y;
    launch_cfg.blockDimZ = cfg.block_z;
    launch_cfg.sharedMemBytes = cfg.smem_bytes;
    launch_cfg.hStream = stream;
    launch_cfg.attrs = &attr;
    launch_cfg.numAttrs = 1;
    err = g_cuLaunchKernelEx(&launch_cfg, cfg.fn, kernel_params.data(), nullptr);
    if (err != 0) {
      if (err_msg) {
        std::snprintf(err_msg, 128, "pyptx: cuLaunchKernelEx failed with %d",
                      static_cast<int>(err));
      }
      return -7;
    }
  } else {
    err = g_cuLaunchKernel(
        cfg.fn,
        cfg.grid_x, cfg.grid_y, cfg.grid_z,
        cfg.block_x, cfg.block_y, cfg.block_z,
        cfg.smem_bytes,
        stream,
        kernel_params.data(),
        nullptr);
  }
  if (err != 0) {
    if (err_msg) {
      std::snprintf(
          err_msg, 128,
          "pyptx: cuLaunchKernel failed with %d (cluster=%u,%u,%u)",
          static_cast<int>(err),
          cfg.cluster_x, cfg.cluster_y, cfg.cluster_z);
    }
    return -7;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Raw ctypes entry point — used by the PyTorch integration
// ---------------------------------------------------------------------------
//
// Python calls this via ctypes.cdll, passing:
//   handle       — cubin handle
//   stream       — CUstream value as uintptr_t (0 = default stream)
//   buffer_ptrs  — pointer to an array of void* (device pointers)
//   n_buffers    — length of buffer_ptrs
//   err_out      — optional pointer to a 128-byte buffer for error text
// Returns 0 on success, nonzero error code on failure.
//
// Unlike the XLA FFI handler, this path does NO buffer extraction —
// the caller (pyptx/torch_support.py) is responsible for calling
// ``tensor.data_ptr()`` and assembling the pointer array.
extern "C" int pyptx_shim_launch_raw(
    int64_t handle,
    uint64_t stream_u64,
    void** buffer_ptrs,
    size_t n_buffers,
    char* err_out) {
  CUstream stream = reinterpret_cast<CUstream>(stream_u64);
  return PyptxLaunchCore(handle, stream, buffer_ptrs, n_buffers, err_out);
}

// ---------------------------------------------------------------------------
// XLA FFI handler (JAX path) — only compiled when jaxlib headers available
// ---------------------------------------------------------------------------

#ifndef PYPTX_NO_XLA_FFI

static ffi::Error PyptxLaunchImpl(
    CUstream stream,
    int64_t cubin_handle,
    ffi::RemainingArgs args,
    ffi::RemainingRets rets) {
  std::vector<void*> all_ptrs;
  all_ptrs.reserve(args.size() + rets.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto maybe = args.get<ffi::AnyBuffer>(i);
    if (maybe.has_error()) return maybe.error();
    all_ptrs.push_back(maybe.value().untyped_data());
  }
  for (size_t i = 0; i < rets.size(); ++i) {
    auto maybe = rets.get<ffi::AnyBuffer>(i);
    if (maybe.has_error()) return maybe.error();
    all_ptrs.push_back(maybe.value()->untyped_data());
  }

  char err_msg[128] = {0};
  int rc = PyptxLaunchCore(cubin_handle, stream,
                           all_ptrs.data(), all_ptrs.size(), err_msg);
  if (rc != 0) {
    ffi::ErrorCode code;
    switch (rc) {
      case -1: code = ffi::ErrorCode::kUnavailable; break;
      case -2: code = ffi::ErrorCode::kNotFound; break;
      case -3: code = ffi::ErrorCode::kUnavailable; break;
      case -4: code = ffi::ErrorCode::kInvalidArgument; break;
      default: code = ffi::ErrorCode::kInternal; break;
    }
    return ffi::Error(code, err_msg);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PyptxLaunch, PyptxLaunchImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<CUstream>>()
        .Attr<int64_t>("cubin_handle")
        .RemainingArgs()
        .RemainingRets());

#endif  // PYPTX_NO_XLA_FFI
