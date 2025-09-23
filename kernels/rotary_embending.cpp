/**
 * @file rope_custom.cpp
 *
 * Fusion operator for Rotary Positional Embedding (RoPE) on Ascend.
 * Performs: index lookup -> split cos/sin -> apply rotation to query/key.
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
        uint32_t num_heads,
        uint32_t head_size,
        uint32_t max_positions
    ) {
        this->num_tokens = num_tokens;
        this->num_heads = num_heads;
        this->head_size = head_size;
        this->half_head_size = head_size / 2;

        uint32_t start_token;
        if (num_tokens < GetBlockNum()) {
            start_token = 0;
            split_heads = true;
        } else {
            split_heads = false;
            start_token = GetBlockIdx() * (num_tokens / GetBlockNum()) + MIN(GetBlockIdx(), num_tokens % GetBlockNum());
            this->tile_length = (num_tokens / GetBlockNum()) + (GetBlockIdx() < num_tokens % GetBlockNum() ? 1 : 0);
        }

        // Set up global memory pointers
        this->query.SetGlobalBuffer((__gm__ T*)query_gm + start_token * num_heads * head_size);
        this->key.SetGlobalBuffer((__gm__ T*)key_gm + start_token * num_heads * head_size);
        this->positions.SetGlobalBuffer((__gm__ int64_t*)positions_gm + start_token);
        this->cos_sin.SetGlobalBuffer((__gm__ float*)cos_sin_gm, max_positions * head_size);

        this->output_query.SetGlobalBuffer((__gm__ T*)output_query_gm + start_token * num_heads * head_size);
        this->output_key.SetGlobalBuffer((__gm__ T*)output_key_gm + start_token * num_heads * head_size);

        // Initialize queues
        pipe.InitBuffer(inQueueQuery, BUFFER_NUM, head_size * sizeof(T));
        pipe.InitBuffer(inQueueKey, BUFFER_NUM, head_size * sizeof(T));

        pipe.InitBuffer(inQueueCosSin, BUFFER_NUM, head_size * sizeof(float));

        pipe.InitBuffer(outQueueQuery, BUFFER_NUM, head_size * sizeof(T));
        pipe.InitBuffer(outQueueKey, BUFFER_NUM, head_size * sizeof(T));

        pipe.InitBuffer(calcBuf, 3 * head_size * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (split_heads) {
            for (uint32_t task_idx = 0; task_idx < this->num_tokens * this->num_heads; ++task_idx) {
                uint32_t token_idx = task_idx / this->num_heads;
                uint32_t head = task_idx % this->num_heads;
                uint32_t pos = (uint32_t)positions.GetValue(token_idx); // Load position index
                CopyInCosSin(pos);                  // Load cos[pos], sin[pos]
                auto cos_sin_local = inQueueCosSin.DeQue<float>();
                auto cos_local = cos_sin_local[0];
                auto sin_local = cos_sin_local[half_head_size];
                CopyIn(token_idx, head);
                Compute(token_idx, head, cos_local, sin_local);
                CopyOut(token_idx, head);
                inQueueCosSin.FreeTensor(cos_sin_local);
            }
        } else {
            for (uint32_t token_idx = 0; token_idx < this->tile_length; ++token_idx) {
                uint32_t pos = (uint32_t)positions.GetValue(token_idx); // Load position index
                // AscendC::printf("Processing token_idx=%u, position=%u\n", token_idx, pos); // Debug print
                CopyInCosSin(pos);                  // Load cos[pos], sin[pos]
                auto cos_sin_local = inQueueCosSin.DeQue<float>();
                auto cos_local = cos_sin_local[0];
                auto sin_local = cos_sin_local[half_head_size];

                for (uint32_t h = 0; h < num_heads; ++h) {
                    CopyIn(token_idx, h);
                    Compute(token_idx, h, cos_local, sin_local);
                    CopyOut(token_idx, h);
                }
                inQueueCosSin.FreeTensor(cos_sin_local);
            }
        }
    }

private:
    __aicore__ inline void CopyInCosSin(int32_t pos) {
        LocalTensor<float> cos_sin_local = inQueueCosSin.AllocTensor<float>();

        DataCopy(cos_sin_local, cos_sin[pos * head_size], head_size);

        inQueueCosSin.EnQue(cos_sin_local);
    }

    __aicore__ inline void CopyIn(uint32_t token_idx, uint32_t head) {
        LocalTensor<T> qLocal = inQueueQuery.AllocTensor<T>();
        LocalTensor<T> kLocal = inQueueKey.AllocTensor<T>();

        DataCopy(qLocal, query[token_idx * num_heads * head_size + head * head_size], head_size);
        DataCopy(kLocal, key[token_idx * num_heads * head_size + head * head_size], head_size);

        inQueueQuery.EnQue(qLocal);
        inQueueKey.EnQue(kLocal);
    }

    __aicore__ inline void ApplyRotation(const LocalTensor<T>& x, const LocalTensor<float>& cos, const LocalTensor<float>& sin, const LocalTensor<T>& x_out) {

        LocalTensor<float> tmp1 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), 0);
        LocalTensor<float> tmp2 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), head_size * sizeof(float));
        LocalTensor<float> tmp3 = calcBuf.GetWithOffset<float>(head_size * sizeof(float), 2 * head_size * sizeof(float));

        AscendC::Cast(tmp1, x, RoundMode::CAST_NONE,head_size); // half to float

        auto x1 = tmp1;
        auto x2 = tmp1[half_head_size];

        AscendC::Mul(tmp2, x1, cos, half_head_size); // x1 * cos
        AscendC::Mul(tmp3, x2, sin, half_head_size); // x2 * sin

        AscendC::Mul(tmp2[half_head_size], x2, cos, half_head_size); // x2 * cos
        AscendC::Mul(tmp3[half_head_size], x1, sin, half_head_size); // x1 * sin

        AscendC::Sub(tmp1, tmp2, tmp3, half_head_size); // x1 * cos - x2 * sin
        AscendC::Add(tmp1[half_head_size], tmp2[half_head_size], tmp3[half_head_size], half_head_size); // x2 * cos + x1 * sin

        AscendC::Cast(x_out, tmp1, RoundMode::CAST_ROUND, head_size); // float to half

    }

    __aicore__ inline void Compute(uint32_t token_idx, uint32_t head, const LocalTensor<float>& cos, const LocalTensor<float>& sin) {
        LocalTensor<T> q = inQueueQuery.DeQue<T>();
        LocalTensor<T> k = inQueueKey.DeQue<T>();

        LocalTensor<T> q_out = outQueueQuery.AllocTensor<T>();
        LocalTensor<T> k_out = outQueueKey.AllocTensor<T>();

        ApplyRotation(q, cos, sin, q_out);
        ApplyRotation(k, cos, sin, k_out);

        outQueueQuery.EnQue(q_out);
        outQueueKey.EnQue(k_out);

        inQueueQuery.FreeTensor(q);
        inQueueKey.FreeTensor(k);
    }

    __aicore__ inline void CopyOut(uint32_t token_idx, uint32_t head) {
        uint32_t offset = token_idx * num_heads * head_size + head * head_size;

        LocalTensor<T> q_out = outQueueQuery.DeQue<T>();
        LocalTensor<T> k_out = outQueueKey.DeQue<T>();

        // AscendC::DumpTensor(q_out, 0, 32);
        // AscendC::DumpTensor(k_out, 1, 32);

        DataCopy(output_query[offset], q_out, head_size);
        DataCopy(output_key[offset], k_out, head_size);

        outQueueQuery.FreeTensor(q_out);
        outQueueKey.FreeTensor(k_out);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueQuery, inQueueKey, inQueueCosSin;
    TBuf<TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueQuery, outQueueKey;

    GlobalTensor<T> query, key, output_query, output_key;
    GlobalTensor<float> cos_sin;
    GlobalTensor<int64_t> positions;

    uint32_t num_tokens, num_heads, head_size, half_head_size, tile_length;
    bool split_heads;
};

extern "C" __global__ __aicore__ void rope_custom_fp16(
    GM_ADDR query_gm,
    GM_ADDR key_gm,
    GM_ADDR positions_gm,
    GM_ADDR cos_sin_gm,
    GM_ADDR output_query_gm,
    GM_ADDR output_key_gm,
    uint32_t num_tokens,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t max_positions
) {
    KernelRoPE<half> op;
    op.Init(
        query_gm, key_gm, positions_gm, cos_sin_gm,
        output_query_gm, output_key_gm,
        num_tokens, num_heads, head_size, max_positions
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
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t max_positions
) {
    KernelRoPE<bfloat16_t> op;
    op.Init(
        query_gm, key_gm, positions_gm, cos_sin_gm,
        output_query_gm, output_key_gm,
        num_tokens, num_heads, head_size, max_positions
    );
    op.Process();
}
