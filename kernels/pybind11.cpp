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
#include "aclrtlaunch_rope_custom_fp16.h"
#include "aclrtlaunch_rope_custom_bf16.h"
#include "aclrtlaunch_rmsnorm_fp16.h"
#include "aclrtlaunch_rmsnorm_bf16.h"
#include "aclrtlaunch_add_rmsnorm_fp16.h"
#include "aclrtlaunch_add_rmsnorm_bf16.h"
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
    const uint32_t num_kv_heads = key.size(1);
    const uint32_t max_positions = cos_sin_cache.size(0);

    // Create output tensors
    auto output_query = at::empty_like(query);
    auto output_key = at::empty_like(key);
    const uint32_t blockDim = std::min((uint32_t)num_tokens, (uint32_t)24);

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
         num_tokens, num_heads, num_kv_heads, head_size, max_positions
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
         num_tokens, num_heads, num_kv_heads, head_size, max_positions
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RoPE kernel");
    }

    return std::make_tuple(output_query, output_key);
}

at::Tensor run_rmsnorm(
    const at::Tensor &x,
    const at::Tensor &weight,
    float eps
) {
    // Input validation
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "Input x must be a 2D or 3D tensor");
    TORCH_CHECK(weight.dim() == 1, "Weight must be a 1D tensor (hidden_size)");
    TORCH_CHECK(x.size(-1) == weight.size(0), "Last dimension of x must match weight size");
    
    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16, "Input x must be float16 or bfloat16");
    TORCH_CHECK(x.dtype() == weight.dtype(), "Input x and weight must have the same dtype");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    auto acl_stream = c10_npu::getCurrentNPUStream();
    
    // Create a contiguous output tensor with the same shape as input x.
    auto output = at::empty(x.sizes(), x.options());

    // Extract dimensions and strides, treating 2D as a special case of 3D
    const uint32_t hidden_size = x.size(-1);
    uint32_t num_tokens, num_heads;
    uint32_t x_stride0, x_stride1;

    if (x.dim() == 2) {
        // Shape: (num_tokens, hidden_size)
        num_tokens = x.size(0);
        num_heads = 1;
        x_stride0 = x.stride(0);
        x_stride1 = x.stride(0);
    } else { // x.dim() == 3
        // Shape: (num_tokens, num_heads, hidden_size)
        num_tokens = x.size(0);
        num_heads = x.size(1);
        x_stride0 = x.stride(0);
        x_stride1 = x.stride(1);
    }
    
    // Total number of rows to process (each row is a vector of size hidden_size)
    const uint32_t num_rows = num_tokens * num_heads;

    // Parallelize across the combined (tokens * heads) dimension
    const uint32_t blockDim = std::min((uint32_t)num_rows, (uint32_t)24);

    if (blockDim == 0) {
        return output; // Handle empty input
    }
    
    // Dispatch to the correct kernel based on dtype
    if (x.dtype() == torch::kFloat16) {
        ACLRT_LAUNCH_KERNEL(rmsnorm_fp16)
        (blockDim, acl_stream,
         const_cast<void *>(x.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         output.data_ptr(),
         num_rows, num_heads, hidden_size,
         x_stride0, x_stride1,
         eps, 1.0f / hidden_size
        );
    } else if (x.dtype() == torch::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(rmsnorm_bf16)
        (blockDim, acl_stream,
         const_cast<void *>(x.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         output.data_ptr(),
         num_rows, num_heads, hidden_size,
         x_stride0, x_stride1,
         eps, 1.0f / hidden_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm kernel");
    }

    return output;
}

std::tuple<at::Tensor, at::Tensor> run_add_rmsnorm(
    const at::Tensor &x,
    const at::Tensor &residual,
    const at::Tensor &weight,
    float eps
) {
    // Input validation
    TORCH_CHECK(x.dim() == 2, "Input x must be a 2D tensor for add_rmsnorm");
    TORCH_CHECK(residual.dim() == 2, "Input residual must be a 2D tensor for add_rmsnorm");
    TORCH_CHECK(weight.dim() == 1, "Weight must be a 1D tensor");

    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual must have the same shape");
    TORCH_CHECK(x.size(1) == weight.size(0), "Last dimension of x must match weight size");
    
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "Input residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");

    TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16, "Inputs must be float16 or bfloat16");
    TORCH_CHECK(x.dtype() == residual.dtype(), "x and residual must have the same dtype");
    TORCH_CHECK(x.dtype() == weight.dtype(), "x and weight must have the same dtype");

    auto acl_stream = c10_npu::getCurrentNPUStream();
    
    // Create output tensors
    auto output = at::empty_like(x);
    auto residual_out = at::empty_like(x);

    // Extract dimensions
    const uint32_t num_rows = x.size(0);
    const uint32_t hidden_size = x.size(1);

    if (num_rows == 0) {
        return std::make_tuple(output, residual_out); // Handle empty input
    }

    // Parallelize across the rows
    const uint32_t blockDim = std::min((uint32_t)num_rows, (uint32_t)24);
    
    // Dispatch to the correct kernel based on dtype
    if (x.dtype() == torch::kFloat16) {
        ACLRT_LAUNCH_KERNEL(add_rmsnorm_fp16)
        (blockDim, acl_stream,
            const_cast<void *>(x.data_ptr()),
            const_cast<void *>(residual.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            output.data_ptr(),
            residual_out.data_ptr(),
            num_rows, hidden_size,
            eps, 1.0f / hidden_size
        );
    } else if (x.dtype() == torch::kBFloat16) {
        ACLRT_LAUNCH_KERNEL(add_rmsnorm_bf16)
        (blockDim, acl_stream,
            const_cast<void *>(x.data_ptr()),
            const_cast<void *>(residual.data_ptr()),
            const_cast<void *>(weight.data_ptr()),
            output.data_ptr(),
            residual_out.data_ptr(),
            num_rows, hidden_size,
            eps, 1.0f / hidden_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for Add RMSNorm kernel");
    }

    return std::make_tuple(output, residual_out);
}

std::tuple<at::Tensor, at::Tensor> run_norm_rope(
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight,
    float q_norm_eps,
    float k_norm_eps,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    bool norm_before_reshape
) {
    // --- Input Validation ---
    TORCH_CHECK(q_in.dim() == 2, "Input q must be a 2D tensor (num_tokens, q_size)");
    TORCH_CHECK(k_in.dim() == 2, "Input k must be a 2D tensor (num_tokens, kv_size)");
    TORCH_CHECK(q_norm_weight.dim() == 1, "q_norm_weight must be 1D");
    TORCH_CHECK(k_norm_weight.dim() == 1, "k_norm_weight must be 1D");
    TORCH_CHECK(q_in.dtype() == k_in.dtype(), "q and k must have the same dtype");
    TORCH_CHECK(q_in.dtype() == q_norm_weight.dtype(), "q and q_norm_weight must have the same dtype");
    TORCH_CHECK(k_in.dtype() == k_norm_weight.dtype(), "k and k_norm_weight must have the same dtype");

    // Tensors that will be passed to run_rope_custom
    at::Tensor q_for_rope;
    at::Tensor k_for_rope;

    if (norm_before_reshape) {
        // --- Olmoe Style ---
        // 1. Apply RMSNorm on the 2D input tensors.
        auto q_norm_out = run_rmsnorm(q_in, q_norm_weight, q_norm_eps);
        auto k_norm_out = run_rmsnorm(k_in, k_norm_weight, k_norm_eps);

        // 2. Reshape the normalized tensors to 3D for RoPE.
        q_for_rope = q_norm_out.view({-1, num_q_heads, head_dim});
        k_for_rope = k_norm_out.view({-1, num_kv_heads, head_dim});
    } else {
        // --- Qwen3 Style ---
        // 1. Reshape the input tensors to 3D first.
        auto q_reshaped = q_in.view({-1, num_q_heads, head_dim});
        auto k_reshaped = k_in.view({-1, num_kv_heads, head_dim});

        // 2. Apply RMSNorm on the 3D tensors. The run_rmsnorm function
        // internally handles reshaping to 2D for the kernel.
        q_for_rope = run_rmsnorm(q_reshaped, q_norm_weight, q_norm_eps);
        k_for_rope = run_rmsnorm(k_reshaped, k_norm_weight, k_norm_eps);
    }
    
    // 3. Apply RoPE on the prepared 3D tensors.
    // The run_rope_custom function will return the final q and k.
    return run_rope_custom(q_for_rope, k_for_rope, positions, cos_sin_cache);
}


} // namespace my_ops

PYBIND11_MODULE(nanovllm_kernels, m)
{
    m.doc() = "Custom Ascend C kernel pybind11 interfaces";

    m.def("run_store_kvcache",
          &my_ops::run_store_kvcache,
          "Store key and value tensors into KV cache",
          py::arg("key"),
          py::arg("value"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("slot_mapping"));

    m.def("run_rope_custom",
          &my_ops::run_rope_custom,
          "Apply Rotary Positional Embedding (RoPE) to query and key (out-of-place)",
          py::arg("query"),
          py::arg("key"),
          py::arg("positions"),
          py::arg("cos_sin_cache"));

    m.def("run_rmsnorm",
          &my_ops::run_rmsnorm,
          "Apply Root Mean Square Normalization (RMSNorm)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("eps"));

    m.def("run_add_rmsnorm",
          &my_ops::run_add_rmsnorm,
          "Fused Add + Root Mean Square Normalization (RMSNorm)",
          py::arg("x"),
          py::arg("residual"),
          py::arg("weight"),
          py::arg("eps"));

    m.def("run_norm_rope",
          &my_ops::run_norm_rope,
          "Fused operation for RMSNorm and Rotary Positional Embedding (RoPE)",
          py::arg("q_in"),
          py::arg("k_in"),
          py::arg("q_norm_weight"),
          py::arg("k_norm_weight"),
          py::arg("q_norm_eps"),
          py::arg("k_norm_eps"),
          py::arg("positions"),
          py::arg("cos_sin_cache"),
          py::arg("num_q_heads"),
          py::arg("num_kv_heads"),
          py::arg("head_dim"),
          py::arg("norm_before_reshape")
    );
}
