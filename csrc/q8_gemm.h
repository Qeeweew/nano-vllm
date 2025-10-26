#pragma once
#include <torch/extension.h>
#include <vector>

// --- Function declarations for Python bindings ---

// Quantizes a weight tensor without repacking.
std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float);

// Performs a GEMM operation with a quantized and packed weight.
torch::Tensor q8_gemm(
    torch::Tensor A_float,
    torch::Tensor B_qs_packed,
    torch::Tensor B_d_packed
);

// Performs the full MoE forward pass on CPU.
torch::Tensor moe_q8_forward(
    torch::Tensor x,
    torch::Tensor routing_weights,
    torch::Tensor selected_experts,
    torch::Tensor gate_up_qs_stacked,
    torch::Tensor gate_up_d_stacked,
    torch::Tensor down_proj_qs_stacked,
    torch::Tensor down_proj_d_stacked
);

// Performs Top-K gating and softmax.
std::vector<torch::Tensor> gating_top_k_softmax(
    const torch::Tensor& logits,
    int top_k,
    bool normalize
);


// --- Function declarations for internal C++ usage ---

// Repacks a quantized weight from row-major to the kernel's expected format.
template<typename D_TYPE>
void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed
);

// Templated implementation of the gating logic.
template <typename T>
void gating_top_k_softmax_ptr_impl(
    const T* logits_ptr, int64_t num_tokens, int64_t num_experts, int64_t top_k, bool normalize,
    float* routing_weights_out, int32_t* selected_experts_out);

// Templated implementation of the non-NUMA MoE forward pass.
template <typename T>
void moe_q8_forward_ptr_impl(
    T* hidden_states_ptr, const float* routing_weights_ptr, const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_ptr, const at::Half* gate_up_d_ptr,
    const int8_t* down_proj_qs_ptr, const at::Half* down_proj_d_ptr,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts, int64_t intermediate_size, int64_t intermediate_size_x2, int64_t top_k);

// Templated implementation of the NUMA-aware MoE forward pass.
template <typename T>
void moe_q8_forward_ptr_numa_impl(
    T* x_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const std::vector<void*>& gate_up_qs_stacked_numa,
    const std::vector<void*>& gate_up_d_stacked_numa,
    const std::vector<void*>& down_proj_qs_stacked_numa,
    const std::vector<void*>& down_proj_d_stacked_numa,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t top_k,
    int numa_nodes);

#ifdef WITH_CUDA
#include <pybind11/pybind11.h>
namespace py = pybind11;
// Declaration for the CUDA host function launcher.
void launch_moe_cpu_task(
    torch::Tensor& hidden_states_pinned,
    const torch::Tensor& router_logits_pinned,
    py::capsule& moe_infer_handle,
    int64_t top_k,
    bool normalize_prob,
    uint64_t stream_ptr,
    bool keep_args
);
#endif // WITH_CUDA