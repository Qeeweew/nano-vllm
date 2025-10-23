#include "moe_infer.h"
#include <cstdio>
#include <stdexcept>
#include <cstring>

// Forward-declare the quantization function from q8_gemm.cpp
std::pair<torch::Tensor, torch::Tensor> quantize_repack_weight(torch::Tensor weight);

// Constructor, destructor, and weight loading methods remain the same as before...
MoEInfer::MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size)
    : num_experts_(num_experts), hidden_size_(hidden_size), intermediate_size_(intermediate_size) {
    gate_up_qs_bytes_ = num_experts_ * (intermediate_size_ * 2) * hidden_size_;
    gate_up_d_bytes_ = num_experts_ * (intermediate_size_ * 2) * (hidden_size_ / 32) * sizeof(at::Half);
    down_proj_qs_bytes_ = num_experts_ * hidden_size_ * intermediate_size_;
    down_proj_d_bytes_ = num_experts_ * hidden_size_ * (intermediate_size_ / 32) * sizeof(at::Half);
    gate_up_qs_stacked_ = std::aligned_alloc(64, gate_up_qs_bytes_);
    gate_up_d_stacked_ = std::aligned_alloc(64, gate_up_d_bytes_);
    down_proj_qs_stacked_ = std::aligned_alloc(64, down_proj_qs_bytes_);
    down_proj_d_stacked_ = std::aligned_alloc(64, down_proj_d_bytes_);
    if (!gate_up_qs_stacked_ || !gate_up_d_stacked_ || !down_proj_qs_stacked_ || !down_proj_d_stacked_) {
        std::free(gate_up_qs_stacked_); std::free(gate_up_d_stacked_); std::free(down_proj_qs_stacked_); std::free(down_proj_d_stacked_);
        throw std::runtime_error("Failed to allocate memory for MoE weights");
    }
}

MoEInfer::~MoEInfer() {
    std::free(gate_up_qs_stacked_); std::free(gate_up_d_stacked_); std::free(down_proj_qs_stacked_); std::free(down_proj_d_stacked_);
}

void MoEInfer::store_quantized_weights(
    const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
    const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {
    memcpy(gate_up_qs_stacked_, gate_up_qs.data_ptr(), gate_up_qs_bytes_);
    memcpy(gate_up_d_stacked_, gate_up_d.data_ptr(), gate_up_d_bytes_);
    memcpy(down_proj_qs_stacked_, down_proj_qs.data_ptr(), down_proj_qs_bytes_);
    memcpy(down_proj_d_stacked_, down_proj_d.data_ptr(), down_proj_d_bytes_);
}

void MoEInfer::quantize_and_store_expert(
    int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight) {
    auto [qs, d] = quantize_repack_weight(weight);
    char* qs_dest_base = static_cast<char*>(gate_up_qs_stacked_);
    char* d_dest_base = static_cast<char*>(gate_up_d_stacked_);
    size_t qs_offset = 0, d_offset = 0;
    if (proj_name == "gate_proj" || proj_name == "up_proj") {
        qs_offset = expert_idx * (intermediate_size_ * 2 * hidden_size_);
        d_offset = expert_idx * (intermediate_size_ * 2 * (hidden_size_ / 32) * sizeof(at::Half));
        if (proj_name == "up_proj") {
            qs_offset += intermediate_size_ * hidden_size_;
            d_offset += intermediate_size_ * (hidden_size_ / 32) * sizeof(at::Half);
        }
    } else if (proj_name == "down_proj") {
        qs_dest_base = static_cast<char*>(down_proj_qs_stacked_);
        d_dest_base = static_cast<char*>(down_proj_d_stacked_);
        qs_offset = expert_idx * (hidden_size_ * intermediate_size_);
        d_offset = expert_idx * (hidden_size_ * (intermediate_size_ / 32) * sizeof(at::Half));
    } else { throw std::invalid_argument("Unknown expert projection name: " + proj_name); }
    memcpy(qs_dest_base + qs_offset, qs.data_ptr(), qs.nbytes());
    memcpy(d_dest_base + d_offset, d.data_ptr(), d.nbytes());
}

// Main templated computation function (pure C++, GIL-free)
template <typename T>
void MoEInfer::execute_on_cpu_templated(
    T* hidden_states_ptr,
    const T* router_logits_ptr,
    int64_t num_tokens,
    int64_t top_k,
    bool normalize_prob
) {
    // Use std::vector for intermediate results, as requested
    std::vector<float> routing_weights(num_tokens * top_k);
    std::vector<int32_t> selected_experts(num_tokens * top_k);

    // Part A: Gating
    gating_top_k_softmax_ptr_impl<T>(
        router_logits_ptr,
        num_tokens,
        num_experts_,
        top_k,
        normalize_prob,
        routing_weights.data(),
        selected_experts.data()
    );
    
     // Part B: MoE forward
    moe_q8_forward_ptr_impl<T>(
        hidden_states_ptr,
        routing_weights.data(),
        selected_experts.data(),
        static_cast<const int8_t*>(gate_up_qs_stacked_),
        static_cast<const at::Half*>(gate_up_d_stacked_),
        static_cast<const int8_t*>(down_proj_qs_stacked_),
        static_cast<const at::Half*>(down_proj_d_stacked_),
        num_tokens,
        hidden_size_,
        num_experts_,
        intermediate_size_,
        intermediate_size_ * 2,
        top_k
    );
}

// Dispatcher function that calls the templated version
void MoEInfer::execute_on_cpu_from_pointers(
    void* hidden_states_ptr,
    const void* router_logits_ptr,
    int64_t num_tokens,
    int64_t top_k,
    bool normalize_prob,
    at::ScalarType dtype
) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        dtype, "moe_execute_dispatcher",
        [&] {
            execute_on_cpu_templated<scalar_t>(
                static_cast<scalar_t*>(hidden_states_ptr),
                static_cast<const scalar_t*>(router_logits_ptr),
                num_tokens,
                top_k,
                normalize_prob
            );
        }
    );
}