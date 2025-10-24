#include "moe_infer.h"
#include <cstdio>
#include <stdexcept>
#include <cstring>

// Forward-declare the quantization function from q8_gemm.cpp
std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float);
template<typename D_TYPE> void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed
);

// Constructor, destructor, and weight loading methods remain the same as before...
MoEInfer::MoEInfer(int64_t num_experts, int64_t hidden_size, int64_t intermediate_size)
    : num_experts_(num_experts), hidden_size_(hidden_size), intermediate_size_(intermediate_size) {

#ifdef WITH_NUMA
    if (numa_available() != -1) {
        numa_nodes_ = numa_max_node() + 1;
        // Ensure that intermediate_size is divisible by the number of NUMA nodes
        TORCH_CHECK(intermediate_size_ % numa_nodes_ == 0,
                    "Intermediate size must be divisible by the number of NUMA nodes.");
    }
#endif
    intermediate_size_per_node_ = intermediate_size_ / numa_nodes_;
    
    // Resize vectors to hold pointers for each node
    gate_up_qs_numa_buffers_.resize(numa_nodes_);
    gate_up_d_numa_buffers_.resize(numa_nodes_);
    down_proj_qs_numa_buffers_.resize(numa_nodes_);
    down_proj_d_numa_buffers_.resize(numa_nodes_);
    
    // Calculate memory size PER NODE
    const int64_t gate_up_qs_bytes_per_node = num_experts_ * (intermediate_size_per_node_ * 2) * hidden_size_;
    const int64_t gate_up_d_bytes_per_node = num_experts_ * (intermediate_size_per_node_ * 2) * (hidden_size_ / 32) * sizeof(at::Half);
    const int64_t down_proj_qs_bytes_per_node = num_experts_ * hidden_size_ * intermediate_size_per_node_;
    const int64_t down_proj_d_bytes_per_node = num_experts_ * hidden_size_ * (intermediate_size_per_node_ / 32) * sizeof(at::Half);

    for (int i = 0; i < numa_nodes_; ++i) {
#ifdef WITH_NUMA
        if (numa_nodes_ > 1) {
            gate_up_qs_numa_buffers_[i] = numa_alloc_onnode(gate_up_qs_bytes_per_node, i);
            gate_up_d_numa_buffers_[i] = numa_alloc_onnode(gate_up_d_bytes_per_node, i);
            down_proj_qs_numa_buffers_[i] = numa_alloc_onnode(down_proj_qs_bytes_per_node, i);
            down_proj_d_numa_buffers_[i] = numa_alloc_onnode(down_proj_d_bytes_per_node, i);
        } else { // Fallback for single-node or systems without NUMA
#endif
            gate_up_qs_numa_buffers_[i] = std::aligned_alloc(64, gate_up_qs_bytes_per_node);
            gate_up_d_numa_buffers_[i] = std::aligned_alloc(64, gate_up_d_bytes_per_node);
            down_proj_qs_numa_buffers_[i] = std::aligned_alloc(64, down_proj_qs_bytes_per_node);
            down_proj_d_numa_buffers_[i] = std::aligned_alloc(64, down_proj_d_bytes_per_node);
#ifdef WITH_NUMA
        }
#endif
        if (!gate_up_qs_numa_buffers_[i] || !gate_up_d_numa_buffers_[i] || !down_proj_qs_numa_buffers_[i] || !down_proj_d_numa_buffers_[i]) {
            throw std::runtime_error("Failed to allocate memory for MoE weights on NUMA node " + std::to_string(i));
        }
    }
}


MoEInfer::~MoEInfer() {
    for (int i = 0; i < numa_nodes_; ++i) {
#ifdef WITH_NUMA
        if (numa_nodes_ > 1) {
            if (gate_up_qs_numa_buffers_[i]) numa_free(gate_up_qs_numa_buffers_[i], /* size */ 0);
            if (gate_up_d_numa_buffers_[i]) numa_free(gate_up_d_numa_buffers_[i], /* size */ 0);
            if (down_proj_qs_numa_buffers_[i]) numa_free(down_proj_qs_numa_buffers_[i], /* size */ 0);
            if (down_proj_d_numa_buffers_[i]) numa_free(down_proj_d_numa_buffers_[i], /* size */ 0);
        } else {
#endif
            std::free(gate_up_qs_numa_buffers_[i]);
            std::free(gate_up_d_numa_buffers_[i]);
            std::free(down_proj_qs_numa_buffers_[i]);
            std::free(down_proj_d_numa_buffers_[i]);
#ifdef WITH_NUMA
        }
#endif
    }
}

// This function now de-interleaves the incoming monolithic tensor into per-NUMA buffers.
void MoEInfer::store_quantized_weights(const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d,
                                       const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {
    auto* src_gate_up_qs = gate_up_qs.data_ptr<int8_t>();
    auto* src_gate_up_d = gate_up_d.data_ptr<at::Half>();
    auto* src_down_proj_qs = down_proj_qs.data_ptr<int8_t>();
    auto* src_down_proj_d = down_proj_d.data_ptr<at::Half>();

#ifdef WITH_NUMA
    if (numa_nodes_ > 1) {
        // --- SCENARIO 1: Multi-node NUMA Path ---
        // Slice the source tensors and repack each slice into its corresponding NUMA buffer.
        for (int node_id = 0; node_id < numa_nodes_; ++node_id) {
            auto* dest_gate_up_qs_packed = static_cast<int8_t*>(gate_up_qs_numa_buffers_[node_id]);
            auto* dest_gate_up_d_packed = static_cast<at::Half*>(gate_up_d_numa_buffers_[node_id]);
            auto* dest_down_proj_qs_packed = static_cast<int8_t*>(down_proj_qs_numa_buffers_[node_id]);
            auto* dest_down_proj_d_packed = static_cast<at::Half*>(down_proj_d_numa_buffers_[node_id]);

            for (int64_t exp_idx = 0; exp_idx < num_experts_; ++exp_idx) {
                // Repack gate_up weights
                {
                    std::vector<int8_t> temp_slice_qs((intermediate_size_per_node_ * 2) * hidden_size_);
                    std::vector<at::Half> temp_slice_d((intermediate_size_per_node_ * 2) * (hidden_size_ / 32));
                    const int64_t src_expert_offset_qs = exp_idx * (intermediate_size_ * 2) * hidden_size_;
                    const int64_t src_expert_offset_d = exp_idx * (intermediate_size_ * 2) * (hidden_size_ / 32);
                    memcpy(temp_slice_qs.data(), src_gate_up_qs + src_expert_offset_qs + (node_id * intermediate_size_per_node_) * hidden_size_, intermediate_size_per_node_ * hidden_size_);
                    memcpy(temp_slice_qs.data() + intermediate_size_per_node_ * hidden_size_, src_gate_up_qs + src_expert_offset_qs + (intermediate_size_ + node_id * intermediate_size_per_node_) * hidden_size_, intermediate_size_per_node_ * hidden_size_);
                    memcpy(temp_slice_d.data(), src_gate_up_d + src_expert_offset_d + (node_id * intermediate_size_per_node_) * (hidden_size_ / 32), intermediate_size_per_node_ * (hidden_size_ / 32) * sizeof(at::Half));
                    memcpy(temp_slice_d.data() + intermediate_size_per_node_ * (hidden_size_ / 32), src_gate_up_d + src_expert_offset_d + (intermediate_size_ + node_id * intermediate_size_per_node_) * (hidden_size_ / 32), intermediate_size_per_node_ * (hidden_size_ / 32) * sizeof(at::Half));
                    
                    int8_t* expert_dest_qs = dest_gate_up_qs_packed + exp_idx * (intermediate_size_per_node_ * 2 * hidden_size_);
                    at::Half* expert_dest_d = dest_gate_up_d_packed + exp_idx * (intermediate_size_per_node_ * 2 * (hidden_size_ / 32));
                    repack_B_q8_0_from_ptr<at::Half>(intermediate_size_per_node_ * 2, hidden_size_, temp_slice_qs.data(), temp_slice_d.data(), expert_dest_qs, expert_dest_d);
                }
                // Repack down_proj weights
                {
                    std::vector<int8_t> temp_slice_qs(hidden_size_ * intermediate_size_per_node_);
                    std::vector<at::Half> temp_slice_d(hidden_size_ * (intermediate_size_per_node_ / 32));
                    const int64_t src_expert_offset_qs = exp_idx * hidden_size_ * intermediate_size_;
                    const int64_t src_expert_offset_d = exp_idx * hidden_size_ * (intermediate_size_ / 32);
                    for (int64_t h = 0; h < hidden_size_; ++h) {
                        const int8_t* src_qs_row = src_down_proj_qs + src_expert_offset_qs + h * intermediate_size_ + node_id * intermediate_size_per_node_;
                        const at::Half* src_d_row = src_down_proj_d + src_expert_offset_d + h * (intermediate_size_ / 32) + node_id * (intermediate_size_per_node_ / 32);
                        memcpy(temp_slice_qs.data() + h * intermediate_size_per_node_, src_qs_row, intermediate_size_per_node_);
                        memcpy(temp_slice_d.data() + h * (intermediate_size_per_node_ / 32), src_d_row, (intermediate_size_per_node_ / 32) * sizeof(at::Half));
                    }
                    int8_t* expert_dest_qs = dest_down_proj_qs_packed + exp_idx * (hidden_size_ * intermediate_size_per_node_);
                    at::Half* expert_dest_d = dest_down_proj_d_packed + exp_idx * (hidden_size_ * (intermediate_size_per_node_ / 32));
                    repack_B_q8_0_from_ptr<at::Half>(hidden_size_, intermediate_size_per_node_, temp_slice_qs.data(), temp_slice_d.data(), expert_dest_qs, expert_dest_d);
                }
            }
        }
        return;
    }
#endif
    {
        // --- SCENARIO 2: Single Node / Non-NUMA Path ---
        // Repack the full source tensors directly into the single buffer at index 0.
        auto* dest_gate_up_qs_packed = static_cast<int8_t*>(gate_up_qs_numa_buffers_[0]);
        auto* dest_gate_up_d_packed = static_cast<at::Half*>(gate_up_d_numa_buffers_[0]);
        auto* dest_down_proj_qs_packed = static_cast<int8_t*>(down_proj_qs_numa_buffers_[0]);
        auto* dest_down_proj_d_packed = static_cast<at::Half*>(down_proj_d_numa_buffers_[0]);

        for (int64_t exp_idx = 0; exp_idx < num_experts_; ++exp_idx) {
            // Repack full gate_up weight for the expert
            {
                const int8_t* expert_src_qs = src_gate_up_qs + exp_idx * (intermediate_size_ * 2 * hidden_size_);
                const at::Half* expert_src_d = src_gate_up_d + exp_idx * (intermediate_size_ * 2 * (hidden_size_ / 32));
                int8_t* expert_dest_qs = dest_gate_up_qs_packed + exp_idx * (intermediate_size_ * 2 * hidden_size_);
                at::Half* expert_dest_d = dest_gate_up_d_packed + exp_idx * (intermediate_size_ * 2 * (hidden_size_ / 32));
                
                repack_B_q8_0_from_ptr<at::Half>(intermediate_size_ * 2, hidden_size_, expert_src_qs, expert_src_d, expert_dest_qs, expert_dest_d);
            }
            // Repack full down_proj weight for the expert
            {
                const int8_t* expert_src_qs = src_down_proj_qs + exp_idx * (hidden_size_ * intermediate_size_);
                const at::Half* expert_src_d = src_down_proj_d + exp_idx * (hidden_size_ * (intermediate_size_ / 32));
                int8_t* expert_dest_qs = dest_down_proj_qs_packed + exp_idx * (hidden_size_ * intermediate_size_);
                at::Half* expert_dest_d = dest_down_proj_d_packed + exp_idx * (hidden_size_ * (intermediate_size_ / 32));

                repack_B_q8_0_from_ptr<at::Half>(hidden_size_, intermediate_size_, expert_src_qs, expert_src_d, expert_dest_qs, expert_dest_d);
            }
        }
    }
}

void MoEInfer::quantize_and_store_expert(
    int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight) {
#ifdef WITH_NUMA
    if (numa_nodes_ > 1) {
        fprintf(stderr, "ERROR: MoEInfer::quantize_and_store_expert is not supported in multi-node NUMA mode.\n");
        fprintf(stderr, "Please use the offline quantization script (quantize_moe.py) and load with store_quantized_weights.\n");
        exit(1);
    }
#endif

    // This path is for non-NUMA or single-NUMA node cases.
    
    // Step 1: Quantize the FP32 weight into a simple, row-major layout.
    // This calls the same C++ function that the Python script uses.
    std::vector<torch::Tensor> quantized_tensors = quantize_weight_only(weight);
    torch::Tensor& qs_unpacked_tensor = quantized_tensors[0];
    torch::Tensor& d_unpacked_tensor = quantized_tensors[1];

    const int8_t* qs_unpacked_ptr = qs_unpacked_tensor.data_ptr<int8_t>();
    const at::Half* d_unpacked_ptr = d_unpacked_tensor.data_ptr<at::Half>();

    // Step 2: Repack the row-major quantized data into the final buffer.
    void* dest_qs_packed_base;
    void* dest_d_packed_base;
    size_t qs_offset = 0, d_offset = 0;
    int64_t N = 0, K = 0;

    if (proj_name == "gate_proj" || proj_name == "up_proj") {
        dest_qs_packed_base = gate_up_qs_numa_buffers_[0];
        dest_d_packed_base = gate_up_d_numa_buffers_[0];
        N = intermediate_size_;
        K = hidden_size_;

        qs_offset = expert_idx * 2 * N * K;
        d_offset = expert_idx * 2 * N * (K / 32);

        // Find the specific slice for 'gate' or 'up'
        if (proj_name == "up_proj") {
            qs_offset += N * K;
            d_offset += N * (K / 32);
        }
        
    } else if (proj_name == "down_proj") {
        dest_qs_packed_base = down_proj_qs_numa_buffers_[0];
        dest_d_packed_base = down_proj_d_numa_buffers_[0];
        N = hidden_size_;
        K = intermediate_size_;
        qs_offset = expert_idx * N * K;
        d_offset = expert_idx * N * (K / 32);
    } else { 
        throw std::invalid_argument("Unknown expert projection name: " + proj_name); 
    }

    repack_B_q8_0_from_ptr<at::Half>(
        N, K, 
        qs_unpacked_ptr, d_unpacked_ptr,
        static_cast<int8_t*>(dest_qs_packed_base) + qs_offset,
        static_cast<at::Half*>(dest_d_packed_base) + d_offset
    );
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
#ifdef WITH_NUMA
    moe_q8_forward_ptr_numa_impl(
        hidden_states_ptr,
        routing_weights.data(),
        selected_experts.data(),
        gate_up_qs_numa_buffers_,
        gate_up_d_numa_buffers_,
        down_proj_qs_numa_buffers_,
        down_proj_d_numa_buffers_,
        num_tokens,
        hidden_size_,
        num_experts_,
        intermediate_size_,
        top_k,
        numa_nodes_
    );
#else
    moe_q8_forward_ptr_impl<T>(
        hidden_states_ptr,
        routing_weights.data(),
        selected_experts.data(),
        static_cast<const int8_t*>(gate_up_qs_numa_buffers_[0]),
        static_cast<const at::Half*>(gate_up_d_numa_buffers_[0]),
        static_cast<const int8_t*>(down_proj_qs_numa_buffers_[0]),
        static_cast<const at::Half*>(down_proj_d_numa_buffers_[0]),
        num_tokens,
        hidden_size_,
        num_experts_,
        intermediate_size_,
        intermediate_size_ * 2,
        top_k
    );
#endif
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