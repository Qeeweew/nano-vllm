/**
 * @file rope_custom.cpp
 *
 * Fusion operator for Rotary Positional Embedding (RoPE) on Ascend.
 * Performs: index lookup -> split cos/sin -> apply rotation to query/key.
 * This version supports Grouped-Query Attention (GQA) with cleanly separated logic.
 *
 * Optimized version: Copies all heads for a token in a single DMA transfer
 * to reduce I/O overhead.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

template<typename T>
class KernelRoPE {
public:
    __aicore__ inline KernelRoPE() {}
    
    __aicore__ inline void Init(
        GM_ADDR query_gm,
        GM_ADDR key_gm,
        GM_ADDR positions_gm,
        GM_ADDR cos_sin_gm,
        GM_ADDR output_query_gm,
        GM_ADDR output_key_gm,
        uint32_t num_tokens,
        uint32_t num_q_heads,
        uint32_t num_kv_heads,
        uint32_t head_size,
        uint32_t max_positions
    ) {
        this->num_tokens = num_tokens;
        this->num_q_heads = num_q_heads;
        this->num_kv_heads = num_kv_heads;
        this->head_size = head_size;
        this->half_head_size = head_size / 2;
        this->group_size = num_q_heads / num_kv_heads;

        // Parallelize over tokens. Each core gets a slice of tokens.
        uint32_t start_token = GetBlockIdx() * (num_tokens / GetBlockNum()) + MIN(GetBlockIdx(), num_tokens % GetBlockNum());
        this->tile_length = (num_tokens / GetBlockNum()) + (GetBlockIdx() < num_tokens % GetBlockNum() ? 1 : 0);
        
        // Set up global memory pointers using the correct head counts for query and key
        this->query.SetGlobalBuffer((__gm__ T*)query_gm + start_token * num_q_heads * head_size);
        this->key.SetGlobalBuffer((__gm__ T*)key_gm + start_token * num_kv_heads * head_size);
        this->positions.SetGlobalBuffer((__gm__ int64_t*)positions_gm + start_token);
        this->cos_sin.SetGlobalBuffer((__gm__ float*)cos_sin_gm, max_positions * head_size);

        this->output_query.SetGlobalBuffer((__gm__ T*)output_query_gm + start_token * num_q_heads * head_size);
        this->output_key.SetGlobalBuffer((__gm__ T*)output_key_gm + start_token * num_kv_heads * head_size);

        // Initialize queues. Buffers are now sized to hold all heads for one token.
        pipe.InitBuffer(inQueueQuery, BUFFER_NUM, num_q_heads * head_size * sizeof(T));
        pipe.InitBuffer(inQueueKey, BUFFER_NUM, num_kv_heads * head_size * sizeof(T));
        pipe.InitBuffer(inQueueCosSin, BUFFER_NUM, 2 * head_size * sizeof(float));
        pipe.InitBuffer(outQueueQuery, BUFFER_NUM, num_q_heads * head_size * sizeof(T));
        pipe.InitBuffer(outQueueKey, BUFFER_NUM, num_kv_heads * head_size * sizeof(T));
        pipe.InitBuffer(calcBuf, 3 * head_size * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t token_idx = 0; token_idx < this->tile_length; ++token_idx) {
            // 1. Fetch position and corresponding cos/sin values for the current token
            uint32_t pos = (uint32_t)positions.GetValue(token_idx);
            CopyInCosSin(pos);
            auto cos_sin_local = inQueueCosSin.DeQue<float>();

            // 2. Duplicate cos/sin values to prepare for rotation
            DataCopy(cos_sin_local[head_size], cos_sin_local[half_head_size], half_head_size);
            DataCopy(cos_sin_local[head_size + half_head_size], cos_sin_local[half_head_size], half_head_size);
            DataCopy(cos_sin_local[half_head_size], cos_sin_local, half_head_size);

            auto cos_local = cos_sin_local;
            auto sin_local = cos_sin_local[head_size];

            // 3. Process all query heads for this token
            CopyInQueryAllHeads(token_idx);
            ComputeAllQueries(cos_local, sin_local);
            CopyOutQueryAllHeads(token_idx);

            // 4. Process all key heads for this token
            CopyInKeyAllHeads(token_idx);
            ComputeAllKeys(cos_local, sin_local);
            CopyOutKeyAllHeads(token_idx);

            inQueueCosSin.FreeTensor(cos_sin_local);
        }
    }

private:
    __aicore__ inline void CopyInCosSin(int32_t pos) {
        LocalTensor<float> cos_sin_local = inQueueCosSin.AllocTensor<float>();
        DataCopy(cos_sin_local, cos_sin[pos * head_size], head_size);
        inQueueCosSin.EnQue(cos_sin_local);
    }

    // === Query-specific Functions (Optimized for all heads) ===
    __aicore__ inline void CopyInQueryAllHeads(uint32_t token_idx) {
        LocalTensor<T> q_in_all = inQueueQuery.AllocTensor<T>();
        const uint32_t copy_size = num_q_heads * head_size;
        DataCopy(q_in_all, query[token_idx * copy_size], copy_size);
        inQueueQuery.EnQue(q_in_all);
    }

    __aicore__ inline void ComputeAllQueries(const LocalTensor<float>& cos, const LocalTensor<float>& sin) {
        LocalTensor<T> q_in_all= inQueueQuery.DeQue<T>();
        LocalTensor<T> q_out_all = outQueueQuery.AllocTensor<T>();

        // Loop over heads in local memory
        for (uint32_t h_q = 0; h_q < num_q_heads; ++h_q) {
            uint32_t offset = h_q * head_size;
            // Get sub-tensors for the current head
            auto q_in_head = q_in_all[offset];
            auto q_out_head = q_out_all[offset];
            ApplyRotation(q_in_head, cos, sin, q_out_head);
        }
        
        outQueueQuery.EnQue(q_out_all);
        inQueueQuery.FreeTensor(q_in_all);
    }

    __aicore__ inline void CopyOutQueryAllHeads(uint32_t token_idx) {
        LocalTensor<T> q_out_all = outQueueQuery.DeQue<T>();
        const uint32_t copy_size = num_q_heads * head_size;
        DataCopy(output_query[token_idx * copy_size], q_out_all, copy_size);
        outQueueQuery.FreeTensor(q_out_all);
    }

    // === Key-specific Functions (Optimized for all heads) ===
    __aicore__ inline void CopyInKeyAllHeads(uint32_t token_idx) {
        LocalTensor<T> k_in_all = inQueueKey.AllocTensor<T>();
        const uint32_t copy_size = num_kv_heads * head_size;
        DataCopy(k_in_all, key[token_idx * copy_size], copy_size);
        inQueueKey.EnQue(k_in_all);
    }

    __aicore__ inline void ComputeAllKeys(const LocalTensor<float>& cos, const LocalTensor<float>& sin) {
        LocalTensor<T> k_in_all = inQueueKey.DeQue<T>();
        LocalTensor<T> k_out_all = outQueueKey.AllocTensor<T>();
        
        // Loop over heads in local memory
        for (uint32_t h_k = 0; h_k < num_kv_heads; ++h_k) {
            uint32_t offset = h_k * head_size;
            // Get sub-tensors for the current head
            auto k_in_head = k_in_all[offset];
            auto k_out_head = k_out_all[offset];
            ApplyRotation(k_in_head, cos, sin, k_out_head);
        }
        
        outQueueKey.EnQue(k_out_all);
        inQueueKey.FreeTensor(k_in_all);
    }

    __aicore__ inline void CopyOutKeyAllHeads(uint32_t token_idx) {
        LocalTensor<T> k_out_all = outQueueKey.DeQue<T>();
        const uint32_t copy_size = num_kv_heads * head_size;
        DataCopy(output_key[token_idx * copy_size], k_out_all, copy_size);
        outQueueKey.FreeTensor(k_out_all);
    }
    
    // === Shared Rotation Logic (Unchanged) ===
    __aicore__ inline void ApplyRotation(const LocalTensor<T>& x, const LocalTensor<float>& cos, const LocalTensor<float>& sin, const LocalTensor<T>& x_out) {
        // This function works on a single head, its logic remains the same.
        // It uses a temporary buffer `calcBuf` for float conversions and calculations.
        LocalTensor<float> tmp1 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), 0);
        LocalTensor<float> tmp2 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), head_size * sizeof(float));
        LocalTensor<float> tmp3 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), 2 * head_size * sizeof(float));

        AscendC::Cast(tmp1, x, RoundMode::CAST_NONE, head_size); // half/bf16 to float

        AscendC::Mul(tmp2, tmp1, cos, head_size); // x * cos
        AscendC::Mul(tmp3, tmp1, sin, head_size); // x * sin

        AscendC::Sub(tmp1, tmp2, tmp3[half_head_size], half_head_size); // x1 * cos - x2 * sin
        AscendC::Add(tmp1[half_head_size], tmp2[half_head_size], tmp3, half_head_size); // x2 * cos + x1 * sin

        AscendC::Cast(x_out, tmp1, RoundMode::CAST_ROUND, head_size); // float to half/bf16
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueQuery, inQueueKey, inQueueCosSin;
    TBuf<TPosition::VECCALC> calcBuf;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueQuery, outQueueKey;

    GlobalTensor<T> query, key, output_query, output_key;
    GlobalTensor<float> cos_sin;
    GlobalTensor<int64_t> positions;

    uint32_t num_tokens, num_q_heads, num_kv_heads, group_size, head_size, half_head_size, tile_length;
};

// Extern "C" functions remain unchanged
extern "C" __global__ __aicore__ void rope_custom_fp16(
    GM_ADDR query_gm,
    GM_ADDR key_gm,
    GM_ADDR positions_gm,
    GM_ADDR cos_sin_gm,
    GM_ADDR output_query_gm,
    GM_ADDR output_key_gm,
    uint32_t num_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t max_positions
) {
    KernelRoPE<half> op;
    op.Init(
        query_gm, key_gm, positions_gm, cos_sin_gm,
        output_query_gm, output_key_gm,
        num_tokens, num_q_heads, num_kv_heads, head_size, max_positions
    );
    op.Process();
}

extern "C" __global__ __aicore__ void rope_custom_bf16(
    GM_ADDR query_gm,
    GM_ADDR key_gm,
    GM_ADDR positions_gm,
    GM_ADDR cos_sin_gm,
    GM_ADDR output_query_gm,
    GM_ADDR output_key_gm,
    uint32_t num_tokens,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    uint32_t head_size,
    uint32_t max_positions
) {
    KernelRoPE<bfloat16_t> op;
    op.Init(
        query_gm, key_gm, positions_gm, cos_sin_gm,
        output_query_gm, output_key_gm,
        num_tokens, num_q_heads, num_kv_heads, head_size, max_positions
    );
    op.Process();
}