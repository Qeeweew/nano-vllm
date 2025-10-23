#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

// Forward-declare the functions from the user's kernel implementations
template <typename T>
void gating_top_k_softmax_ptr_impl(
    const T* logits_ptr, int64_t num_tokens, int64_t num_experts, int64_t top_k, bool normalize,
    float* routing_weights_out, int32_t* selected_experts_out);

template <typename T>
void moe_q8_forward_ptr_impl(
    T* hidden_states_ptr, const float* routing_weights_ptr, const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_ptr, const at::Half* gate_up_d_ptr,
    const int8_t* down_proj_qs_ptr, const at::Half* down_proj_d_ptr,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts, int64_t int64_termediate_size, int64_t int64_termediate_size_x2, int64_t top_k);

class MoEInfer {
public:
    MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size);
    ~MoEInfer();

    MoEInfer(const MoEInfer&) = delete;
    MoEInfer& operator=(const MoEInfer&) = delete;
    MoEInfer(MoEInfer&&) = delete;
    MoEInfer& operator=(MoEInfer&&) = delete;

    void quantize_and_store_expert(int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight);
    void store_quantized_weights(const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d, const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d);

    // This is the new GIL-free entry point for the CUDA callback
    void execute_on_cpu_from_pointers(
        void* hidden_states_ptr,
        const void* router_logits_ptr,
        int64_t num_tokens,
        int64_t top_k,
        bool normalize_prob,
        at::ScalarType dtype
    );

private:
    template <typename T>
    void execute_on_cpu_templated(
        T* hidden_states_ptr,
        const T* router_logits_ptr,
        int64_t num_tokens,
        int64_t top_k,
        bool normalize_prob
    );

    const int64_t num_experts_;
    const int64_t hidden_size_;
    const int64_t intermediate_size_;

    void* gate_up_qs_stacked_ = nullptr;
    void* gate_up_d_stacked_ = nullptr;
    void* down_proj_qs_stacked_ = nullptr;
    void* down_proj_d_stacked_ = nullptr;

    size_t gate_up_qs_bytes_;
    size_t gate_up_d_bytes_;
    size_t down_proj_qs_bytes_;
    size_t down_proj_d_bytes_;
};