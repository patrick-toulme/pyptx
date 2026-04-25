// pyptx Torch C++ extension — thin bridge avoiding ctypes overhead.
// No CUDA headers needed — stream handle passed from Python.

#include <torch/extension.h>
#include <dlfcn.h>
#include <cstdint>
#include <string>
#include <vector>

using LaunchFn = int (*)(int64_t, uint64_t, void**, size_t, char*);
static LaunchFn g_launch_fn = nullptr;
static char g_err_buf[128] = {};

static void load_shim(const std::string& shim_path) {
    if (g_launch_fn) return;
    void* handle = nullptr;
    if (!shim_path.empty()) {
        handle = dlopen(shim_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    if (!handle) {
        handle = dlopen(nullptr, RTLD_NOW);
    }
    if (handle) {
        g_launch_fn = (LaunchFn)dlsym(handle, "pyptx_shim_launch_raw");
    }
    if (!g_launch_fn) {
        throw std::runtime_error(
            "pyptx: cannot find pyptx_shim_launch_raw in " + shim_path);
    }
}

std::vector<torch::Tensor> launch_kernel(
    int64_t cubin_handle,
    int64_t stream_ptr,
    std::vector<torch::Tensor> inputs,
    std::vector<std::vector<int64_t>> out_shapes,
    std::vector<int64_t> out_dtypes
) {
    if (!g_launch_fn) {
        throw std::runtime_error("pyptx: call load_shim() first");
    }

    auto device = inputs[0].device();

    std::vector<torch::Tensor> outputs;
    outputs.reserve(out_shapes.size());
    for (size_t i = 0; i < out_shapes.size(); i++) {
        auto dtype = static_cast<c10::ScalarType>(out_dtypes[i]);
        auto opts = torch::TensorOptions().dtype(dtype).device(device);
        outputs.push_back(torch::empty(out_shapes[i], opts));
    }

    size_t n = inputs.size() + outputs.size();
    std::vector<void*> ptrs(n);
    for (size_t i = 0; i < inputs.size(); i++) {
        ptrs[i] = inputs[i].data_ptr();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        ptrs[inputs.size() + i] = outputs[i].data_ptr();
    }

    int rc = g_launch_fn(
        cubin_handle,
        static_cast<uint64_t>(stream_ptr),
        ptrs.data(),
        n,
        g_err_buf
    );
    if (rc != 0) {
        throw std::runtime_error(
            std::string("pyptx: launch failed: ") + g_err_buf);
    }

    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_shim", &load_shim);
    m.def("launch_kernel", &launch_kernel);
}
