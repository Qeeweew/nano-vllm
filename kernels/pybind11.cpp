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

#include "aclrtlaunch_store_kvcache.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace my_kvcache {
// The function is void because k_cache and v_cache are modified in-place
void run_store_kvcache(const at::Tensor &key, const at::Tensor &value, at::Tensor &k_cache, at::Tensor &v_cache,
                       const at::Tensor &slot_mapping)
{
    TORCH_CHECK(key.dim() == 3, "key must be a 3D tensor");
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be a 1D tensor");
    TORCH_CHECK(key.dtype() == torch::kFloat16 || key.dtype() == torch::kBFloat16, "Key must be float16 for Ascend kernel");


    auto acl_stream = c10_npu::getCurrentNPUStream();

    int64_t N = key.size(0);
    int64_t num_heads = key.size(1);
    int64_t head_dim = key.size(2);
    int64_t D = num_heads * head_dim;

    // As requested, blockDim is set to N. Each core handles one sequence.
    const uint32_t blockDim = std::min(N, (int64_t)32);

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
} // namespace my_kvcache

PYBIND11_MODULE(nanovllm_kernels, m)
{
    m.doc() = "store_kvcache pybind11 interfaces"; // optional module docstring
    m.def("run_store_kvcache", &my_kvcache::run_store_kvcache, "Store key and value tensors into KV cache");
}