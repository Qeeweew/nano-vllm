/**
 * @file pybind11.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tuple>
#include "aclrtlaunch_store_kvcache.h"
#include "aclrtlaunch_rope_custom_fp16.h" // Include the new generated header
#include "aclrtlaunch_rope_custom_bf16.h" // Include the new generated header
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace my_ops {

// --- Existing store_kvcache function ---
void run_store_kvcache(const at::Tensor &key, const at::Tensor &value, at::Tensor &k_cache, at::Tensor &v_cache,
                       const at::Tensor &slot_mapping)
{
    TORCH_CHECK(key.dim() == 3, "key must be a 3D tensor");
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be a 1D tensor");
    TORCH_CHECK(key.dtype() == torch::kFloat16 || key.dtype() == torch::kBFloat16, "Key must be float16 or bfloat16 for Ascend kernel");


    auto acl_stream = c10_npu::getCurrentNPUStream();

    int64_t N = key.size(0);
    int64_t num_heads = key.size(1);
    int64_t head_dim = key.size(2);
    int64_t D = num_heads * head_dim;

    const uint32_t blockDim = std::min((int64_t)N, (int64_t)24);

    ACLRT_LAUNCH_KERNEL(store_kvcache)
    (blockDim, acl_stream,
     const_cast<void *>(key.data_ptr()),
     key.stride(0),
     const_cast<void *>(value.data_ptr()),
     value.stride(0),
     const_cast<void *>(k_cache.data_ptr()),
     const_cast<void *>(v_cache.data_ptr()),
     const_cast<void *>(slot_mapping.data_ptr()),
     N,
     D
    );
}

// --- New function for RoPE ---
std::tuple<at::Tensor, at::Tensor> run_rope_custom(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &positions,
    const at::Tensor &cos_sin_cache
) {
    // Input validation
    TORCH_CHECK(query.dim() == 3, "query must be a 3D tensor (num_tokens, num_heads, head_dim)");
    TORCH_CHECK(key.dim() == 3, "key must be a 3D tensor (num_tokens, num_heads, head_dim)");
    TORCH_CHECK(positions.dim() == 1, "positions must be a 1D tensor (num_tokens)");
    TORCH_CHECK(cos_sin_cache.dim() == 3, "cos_sin_cache must be a 3D tensor (max_pos, 1, head_dim)");

    TORCH_CHECK(query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16, "Query must be float16 or bfloat16");
    TORCH_CHECK(query.dtype() == key.dtype(), "Query and Key must have the same dtype");
    TORCH_CHECK(positions.dtype() == torch::kInt64, "Positions must be int64");
    // The kernel expects float cache, but we will accept half and cast if needed in Python wrapper.
    TORCH_CHECK(cos_sin_cache.dtype() == torch::kFloat32, "cos_sin_cache must be float32 for the kernel");

    TORCH_CHECK(query.is_contiguous(), "Query tensor must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "Key tensor must be contiguous");

    auto acl_stream = c10_npu::getCurrentNPUStream();

    // Extract dimensions
    const uint32_t num_tokens = query.size(0);
    const uint32_t num_heads = query.size(1);
    const uint32_t head_size = query.size(2);
    const uint32_t max_positions = cos_sin_cache.size(0);

    // The kernel needs batch and seq_len. Since we receive flattened tokens,
    // we assume a batch size of 1 for simplicity, or it should be passed from python.
    // For paged attention style inference, num_tokens is the effective batch size.

    // Create output tensors
    auto output_query = at::empty_like(query);
    auto output_key = at::empty_like(key);
    const uint32_t blockDim = std::min((uint32_t)num_tokens * num_heads, (uint32_t)24);

    // Dispatch to the correct kernel based on dtype
    if (query.dtype() == torch::kFloat16) {
        ACLRT_LAUNCH_KERNEL(rope_custom_fp16)
        (blockDim, acl_stream,
         const_cast<void *>(query.data_ptr()),
         const_cast<void *>(key.data_ptr()),
         const_cast<void *>(positions.data_ptr()),
         const_cast<void *>(cos_sin_cache.data_ptr()),
         output_query.data_ptr(),
         output_key.data_ptr(),
         num_tokens, num_heads, head_size, max_positions
        );
    } else if (query.dtype() == torch::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(rope_custom_bf16)
        (blockDim, acl_stream,
         const_cast<void *>(query.data_ptr()),
         const_cast<void *>(key.data_ptr()),
         const_cast<void *>(positions.data_ptr()),
         const_cast<void *>(cos_sin_cache.data_ptr()),
         output_query.data_ptr(),
         output_key.data_ptr(),
         num_tokens, num_heads, head_size, max_positions
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RoPE kernel");
    }

    return std::make_tuple(output_query, output_key);
}

} // namespace my_ops

PYBIND11_MODULE(nanovllm_kernels, m)
{
    m.doc() = "Custom Ascend C kernel pybind11 interfaces";
    m.def("run_store_kvcache", &my_ops::run_store_kvcache, "Store key and value tensors into KV cache");
    m.def("run_rope_custom", &my_ops::run_rope_custom, "Apply Rotary Positional Embedding (RoPE) to query and key");
}