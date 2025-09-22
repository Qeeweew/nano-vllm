/**
 * @file store_kvcache.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

constexpr int32_t BUFFER_NUM = 1;

class KernelStoreKVCache {
public:
    __aicore__ inline KernelStoreKVCache() {}

    __aicore__ inline void Init(GM_ADDR key, int64_t key_stride, GM_ADDR value, int64_t value_stride, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR slot_mapping,
                               int64_t N, int64_t D)
    {
        this->N = N;
        this->D = D;
        this->key_stride = key_stride;
        this->value_stride = value_stride;

        // Each block first reads its corresponding slot index from GM
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slot_mapping);

        // Set up Global Memory Tensors with the correct offsets for this block.
        // Source tensors are indexed by blockIdx.
        keyGm.SetGlobalBuffer((__gm__ half *)key);
        valueGm.SetGlobalBuffer((__gm__ half *)value);
        // Destination tensors are indexed by the dynamically read slot.
        kCacheGm.SetGlobalBuffer((__gm__ half *)k_cache);
        vCacheGm.SetGlobalBuffer((__gm__ half *)v_cache);

        // Initialize the pipe and queues. The buffer size is D elements.
        pipe.InitBuffer(inQueueK, BUFFER_NUM, D * sizeof(half));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, D * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int64_t len = N / AscendC::GetBlockNum();
        int64_t start = AscendC::GetBlockIdx() * len + MIN(AscendC::GetBlockIdx(), N % AscendC::GetBlockNum());
        int64_t end = start + len + (AscendC::GetBlockIdx() < N % AscendC::GetBlockNum() ? 1 : 0);

        for (int64_t i = start; i < end; ++i) {
            int32_t slot = slotMappingGm.GetValue(i);
            if (slot == -1) {
                continue; // Skip if no work for this sequence
            }
            CopyIn(i);
            CopyOut(slot);
        }
    }

private:
    /**
     * @brief Copies key and value vectors from Global Memory to Local Memory.
     */
    __aicore__ inline void CopyIn(int64_t index)
    {
        // Allocate a buffer from the queue for the key vector
        AscendC::LocalTensor<half> localK = inQueueK.AllocTensor<half>();
        // Copy data from GM to the allocated local buffer
        AscendC::DataCopy(localK, keyGm[index * key_stride], D);
        // Enqueue the buffer, signaling that the CopyIn stage is complete for this data
        inQueueK.EnQue(localK);

        // Repeat for the value vector
        AscendC::LocalTensor<half> localV = inQueueV.AllocTensor<half>();
        AscendC::DataCopy(localV, valueGm[index * value_stride], D);
        inQueueV.EnQue(localV);
    }

    /**
     * @brief Copies key and value vectors from Local Memory to their destination in Global Memory (KV Cache).
     */
    __aicore__ inline void CopyOut(int32_t slot)
    {
        // Dequeue the buffer for the key vector, waiting until its CopyIn is complete
        AscendC::LocalTensor<half> localK = inQueueK.DeQue<half>();
        // Copy data from the local buffer to the correct slot in K-Cache (GM)
        AscendC::DataCopy(kCacheGm[slot * D], localK, D);

        // Free the local buffer, making it available for reuse
        inQueueK.FreeTensor(localK);

        // Repeat for the value vector
        AscendC::LocalTensor<half> localV = inQueueV.DeQue<half>();
        AscendC::DataCopy(vCacheGm[slot * D], localV, D);
        inQueueV.FreeTensor(localV);
    }

private:
    AscendC::TPipe pipe;
    // Queues for Key and Value vectors, managed by the pipe
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> inQueueK;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> inQueueV;

    // GlobalTensor objects to represent memory regions in GM
    AscendC::GlobalTensor<half> keyGm;
    AscendC::GlobalTensor<half> valueGm;
    AscendC::GlobalTensor<half> kCacheGm;
    AscendC::GlobalTensor<half> vCacheGm;
    AscendC::GlobalTensor<int32_t> slotMappingGm;

    int64_t D;
    int64_t N;
    int64_t key_stride;
    int64_t value_stride;
};

extern "C" __global__ __aicore__ void store_kvcache(GM_ADDR key, int64_t key_stride, GM_ADDR value, int64_t value_stride, GM_ADDR k_cache, GM_ADDR v_cache,
                                                     GM_ADDR slot_mapping, int64_t D, int64_t N)
{
    KernelStoreKVCache op;
    op.Init(key, key_stride, value, value_stride, k_cache, v_cache, slot_mapping, D, N);
    op.Process();
}