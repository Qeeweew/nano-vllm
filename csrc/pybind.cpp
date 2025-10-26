#include <torch/extension.h>
#include "moe_infer.h"  // For the MoEInfer class
#include "q8_gemm.h"    // For the functions we are binding

// ==========================================================================================
// CUDA HOST LAUNCHER (MOVED FROM Q8_GEMM.CPP)
// ==========================================================================================
#ifdef WITH_CUDA
#include <cuda_runtime.h> // For cudaLaunchHostFunc

struct MoECpuTaskArgs {
    void* hidden_states_ptr;
    const void* router_logits_ptr;
    int64_t num_tokens;
    int64_t top_k;
    bool normalize_prob;
    MoEInfer* moe_infer_ptr;
    bool keep_args;
    at::ScalarType dtype;
};

// The callback function remains simple.
void CUDART_CB host_fn_callback(void* user_data) {
    auto* args = static_cast<MoECpuTaskArgs*>(user_data);
    
    args->moe_infer_ptr->execute_on_cpu_from_pointers(
        args->hidden_states_ptr,
        args->router_logits_ptr,
        args->num_tokens,
        args->top_k,
        args->normalize_prob,
        args->dtype
    );

    if (!args->keep_args) {
        delete args;
    }
}

// The launch function is where we interact with PyTorch tensors and extract raw data.
// This function MUST be called while holding the GIL.
void launch_moe_cpu_task(
    torch::Tensor& hidden_states_pinned,
    const torch::Tensor& router_logits_pinned,
    py::capsule& moe_infer_handle,
    int64_t top_k,
    bool normalize_prob,
    uint64_t stream_ptr,
    bool keep_args
) {
    TORCH_CHECK(hidden_states_pinned.is_pinned(), "hidden_states must be a pinned tensor");
    TORCH_CHECK(router_logits_pinned.is_pinned(), "router_logits must be a pinned tensor");
    TORCH_CHECK(hidden_states_pinned.scalar_type() == router_logits_pinned.scalar_type(), "Dtype mismatch between hidden_states and router_logits");

    auto* args = new MoECpuTaskArgs{
        hidden_states_pinned.data_ptr(),
        router_logits_pinned.data_ptr(),
        hidden_states_pinned.size(0), // num_tokens
        top_k,
        normalize_prob,
        moe_infer_handle.get_pointer<MoEInfer>(),
        keep_args,
        hidden_states_pinned.scalar_type()
    };
    
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream_ptr), host_fn_callback, args);
}
#endif // WITH_CUDA

// ==========================================================================================
// PYBIND11 MODULE DEFINITION (MOVED FROM Q8_GEMM.CPP)
// ==========================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // These functions are for standard CPU execution and testing.
    m.def("quantize_weight_only", &quantize_weight_only, "Quantize weight for q8_gemm");
    m.def("q8_gemm", &q8_gemm, "q8_gemm kernel (A_fp32 @ B_q8.T)");
    m.def("moe_q8_forward", &moe_q8_forward, "Full MoE expert forward pass with int8 GEMM on CPU for float, bfloat16, and float16 inputs");
    m.def("gating_top_k_softmax", &gating_top_k_softmax, "Perform Top-K and Softmax on CPU for MoE gating, returning new tensors");
    
    // Bindings for the MoEInfer class
    m.def("create_moe_infer_handle", [](int64_t num_experts, int64_t hidden_size, int64_t intermediate_size) {
        auto* ptr = new MoEInfer(num_experts, hidden_size, intermediate_size);
        return py::capsule(ptr, [](void* p) { delete reinterpret_cast<MoEInfer*>(p); });
    });
    m.def("moe_infer_quantize_and_store", [](py::capsule& handle, int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight) {
        handle.get_pointer<MoEInfer>()->quantize_and_store_expert(expert_idx, proj_name, weight);
    });
    m.def("moe_infer_store_quantized", [](py::capsule& handle, const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d, const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {
        handle.get_pointer<MoEInfer>()->store_quantized_weights(gate_up_qs, gate_up_d, down_proj_qs, down_proj_d);
    });

    // Add the CUDA-integrated function, protected by the preprocessor guard.
#ifdef WITH_CUDA
    m.def("launch_moe_cpu_task", &launch_moe_cpu_task, "Launches the GIL-free MoE CPU task via cudaLaunchHostFunc.");
#endif
}