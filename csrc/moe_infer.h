#pragma once
#include <torch/extension.h>
#include <string>
#include <vector>

#ifdef WITH_NUMA
#include <numa.h>
#endif
#include <omp.h>

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

    // --- NUMA-specific members ---
    int numa_nodes_ = 1;
    int64_t intermediate_size_per_node_;

    // We replace single pointers with vectors of pointers, one for each NUMA node.
    std::vector<void*> gate_up_qs_numa_buffers_;
    std::vector<void*> gate_up_d_numa_buffers_;
    std::vector<void*> down_proj_qs_numa_buffers_;
    std::vector<void*> down_proj_d_numa_buffers_;
};