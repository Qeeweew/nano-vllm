#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import sys

# --- Prerequisites ---
# Attempt to import all necessary libraries. Fail gracefully if any are missing.
try:
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    # Explicitly enable the NPU Triton backend
    torch.npu.set_compile_mode(jit_compile=True)
except (ImportError, AttributeError) as e:
    print(f"Error: torch_npu or a required feature is not available: {e}")
    print("This test requires a version of torch_npu with Triton backend support.")
    sys.exit(1)

try:
    import triton
    import triton.language as tl
except ImportError:
    print("Error: triton is not installed. This test requires it to generate the golden reference.")
    sys.exit(1)

try:
    import nanovllm_kernels
except ImportError:
    print("Error: Could not import the compiled 'store_kvcache' module.")
    print("Please ensure you have built the operator correctly (e.g., via 'pip install .').")
    sys.exit(1)


# --- Original Triton Kernel (Golden Reference) ---
# This Python code is hardware-agnostic. The torch_npu backend will
# compile it for the Ascend NPU.

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def run_store_kvcache_triton(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    # assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# --- Main Test Class ---
class TestStoreKVCacheNPU(TestCase):

    def test_ascendc_vs_triton_on_npu(self):
        """
        Compares the custom Ascend C operator against the Triton kernel,
        with BOTH running on the Ascend NPU.
        """
        # 1. Check for required hardware
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            self.skipTest("Ascend NPU not available.")

        print("\nAscend NPU detected. Comparing custom Ascend C kernel vs. Triton-on-NPU kernel.")

        # 2. Define test parameters
        N = 1
        num_heads = 16
        head_dim = 128
        max_slots = 8192
        dtype = torch.bfloat16
        D = num_heads * head_dim

        # 3. Generate input tensors on CPU, then move to NPU
        print("Generating input tensors and moving to NPU...")
        key_cpu = torch.randn(N, num_heads, head_dim, device='cpu', dtype=dtype)
        value_cpu = torch.randn(N, num_heads, head_dim, device='cpu', dtype=dtype)
        k_cache_initial_cpu = torch.zeros(max_slots, D, device='cpu', dtype=dtype)
        v_cache_initial_cpu = torch.zeros(max_slots, D, device='cpu', dtype=dtype)
        slot_mapping_cpu = torch.randint(-1, max_slots, (N,), device='cpu', dtype=torch.int32)
        
        # Move all tensors to the NPU device
        key_npu = key_cpu.npu()
        value_npu = value_cpu.npu()
        k_cache_initial_npu = k_cache_initial_cpu.npu()
        v_cache_initial_npu = v_cache_initial_cpu.npu()
        slot_mapping_npu = slot_mapping_cpu.npu()
        
        # --- 4. Run Triton reference on NPU ---
        print("Running Triton reference on NPU to get expected result...")
        # Use a fresh clone of the cache for the Triton run
        k_cache_triton_result = k_cache_initial_npu.clone()
        v_cache_triton_result = v_cache_initial_npu.clone()

        run_store_kvcache_triton(
            key_npu, value_npu, k_cache_triton_result, v_cache_triton_result, slot_mapping_npu
        )
        torch.npu.synchronize()
        print("Finished Triton-on-NPU execution.")
        
        # --- 5. Run Custom Ascend C Operator on NPU ---
        print("Running custom Ascend C operator on NPU...")
        # Use another fresh clone for the custom operator run
        k_cache_custom_result = k_cache_initial_npu.clone()
        v_cache_custom_result = v_cache_initial_npu.clone()

        nanovllm_kernels.run_store_kvcache(
            key_npu, value_npu, k_cache_custom_result, v_cache_custom_result, slot_mapping_npu
        )
        torch.npu.synchronize()
        print("Finished custom Ascend C execution.")

        # --- 6. Compare the results directly on the NPU ---
        print("Comparing results of both kernels on NPU...")
        
        # assertRtolEqual is a powerful tool from torch_npu.testing that handles device tensors
        self.assertRtolEqual(k_cache_custom_result, k_cache_triton_result)
        self.assertRtolEqual(v_cache_custom_result, v_cache_triton_result)

        print("Test Passed: Custom Ascend C operator results match Triton-on-NPU reference results!")


if __name__ == "__main__":
    run_tests()