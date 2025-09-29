/**
 * @file rmsnorm.cpp
 *
 * Kernel for Root Mean Square Normalization (RMSNorm) on Ascend.
 * Parallelized over the batch/token dimension.
 *
 * Calculation steps for each row:
 * 1. Cast input 'x' and 'weight' from T (half/bfloat16) to float32.
 * 2. Square the float32 'x' vector.
 * 3. ReduceSum to get the sum of squares.
 * 4. Calculate variance, add epsilon, and compute rsqrt.
 * 5. Normalize the float32 'x' vector.
 * 6. Multiply the normalized vector by the float32 'weight' vector.
 * 7. Cast the final float32 result back to T.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

template<typename T>
class KernelRMSNorm {
public:
    __aicore__ inline KernelRMSNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x_gm,
        GM_ADDR weight_gm,
        GM_ADDR y_gm,
        uint32_t num_tokens,
        uint32_t hidden_size,
        float eps,
        float r_hidden_size
    ) {
        this->num_tokens = num_tokens;
        this->hidden_size = hidden_size;
        this->r_hidden_size = r_hidden_size;
        this->eps = eps;

        // Parallelize over tokens. Each core gets a slice of tokens.
        uint32_t start_token = GetBlockIdx() * (num_tokens / GetBlockNum()) + MIN(GetBlockIdx(), num_tokens % GetBlockNum());
        this->tile_length = (num_tokens / GetBlockNum()) + (GetBlockIdx() < num_tokens % GetBlockNum() ? 1 : 0);
        
        // Set up global memory pointers
        this->x_gm.SetGlobalBuffer((__gm__ T*)x_gm + start_token * hidden_size);
        this->weight_gm.SetGlobalBuffer((__gm__ T*)weight_gm);
        this->y_gm.SetGlobalBuffer((__gm__ T*)y_gm + start_token * hidden_size);
        
        // Allocate local memory buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, hidden_size * sizeof(T));

        // Calculation buffer for float32 intermediates.
        // Needs space for:
        // - x_fp32 (hidden_size)
        // - weight_fp32 (hidden_size)
        // - result_fp32 (hidden_size)
        // - work buffer for ReduceSum (small, e.g., 64 floats)
        pipe.InitBuffer(calcBuf, 3 * hidden_size * sizeof(float) + 64 * sizeof(float));
    }

    __aicore__ inline void Process() {
        // Load the weight vector once from GM into local memory
        CopyInWeight();
        // Dequeue the weight tensor once before the loop begins.
        LocalTensor<T> weight_local = inQueueWeight.DeQue<T>();
        
        for (uint32_t i = 0; i < this->tile_length; ++i) {
            // Load one token (row) from global memory
            CopyInX(i);

            // Perform the RMSNorm calculation, using the already dequeued weight tensor
            Compute(weight_local);

            // Write the result for this token back to global memory
            CopyOutY(i);
        }

        // Free the weight tensor resource after the loop is finished.
        inQueueWeight.FreeTensor(weight_local);
    }

private:
    __aicore__ inline void CopyInWeight() {
        LocalTensor<T> weight_local = inQueueWeight.AllocTensor<T>();
        DataCopy(weight_local, weight_gm, hidden_size);
        inQueueWeight.EnQue(weight_local);
    }

    __aicore__ inline void CopyInX(uint32_t token_idx) {
        LocalTensor<T> x_local = inQueueX.AllocTensor<T>();
        DataCopy(x_local, x_gm[token_idx * hidden_size], hidden_size);
        inQueueX.EnQue(x_local);
    }
    
    __aicore__ inline void Compute(const LocalTensor<T>& weight_local) {
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<T> y_local = outQueueY.AllocTensor<T>();

        // Get buffers for float32 calculations
        LocalTensor<float> x_fp32 = calcBuf.Get<float>(hidden_size);
        LocalTensor<float> weight_fp32 = calcBuf.GetWithOffset<float>(hidden_size, hidden_size * sizeof(float));
        LocalTensor<float> result_fp32 = calcBuf.GetWithOffset<float>(hidden_size, 2 * hidden_size * sizeof(float));
        LocalTensor<float> reduce_work = calcBuf.GetWithOffset<float>(64, 3 * hidden_size * sizeof(float));
        LocalTensor<float> sum_val = reduce_work;

        // 1. Cast input x and weight to float32
        Cast(x_fp32, x_local, RoundMode::CAST_NONE, hidden_size);
        Cast(weight_fp32, weight_local, RoundMode::CAST_NONE, hidden_size);

        // 2. Square the float32 vector (use result_fp32 as temp buffer)
        Mul(result_fp32, x_fp32, x_fp32, hidden_size);

        // 3. Sum the squares
        ReduceSum(sum_val, result_fp32, reduce_work, hidden_size);
        
        // 4. Calculate normalization factor (1.0f / sqrt(var + eps)) on scalar unit
        Muls(sum_val, sum_val, r_hidden_size, 1);
        Adds(sum_val, sum_val, eps, 1);
        Rsqrt(sum_val, sum_val, 1);
        float rsqrt_val = sum_val.GetValue(0);

        // 5. Normalize original float32 vector
        Muls(result_fp32, x_fp32, rsqrt_val, hidden_size);

        // 6. Multiply by the weights in float32 precision
        Mul(result_fp32, result_fp32, weight_fp32, hidden_size);
        
        // 7. Cast final float32 result back to original dtype T
        Cast(y_local, result_fp32, RoundMode::CAST_ROUND, hidden_size);
        
        outQueueY.EnQue(y_local);
        inQueueX.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOutY(uint32_t token_idx) {
        LocalTensor<T> y_local = outQueueY.DeQue<T>();
        DataCopy(y_gm[token_idx * hidden_size], y_local, hidden_size);
        outQueueY.FreeTensor(y_local);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    TBuf<TPosition::VECCALC> calcBuf;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> x_gm, weight_gm, y_gm;
    
    uint32_t num_tokens;
    uint32_t hidden_size;
    uint32_t tile_length;
    float eps, r_hidden_size;
};


// Extern "C" entry points for FP16 and BF16
extern "C" __global__ __aicore__ void rmsnorm_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y,
    uint32_t num_tokens, uint32_t hidden_size, float eps, float r_hidden_size
) {
    KernelRMSNorm<half> op;
    op.Init(x, weight, y, num_tokens, hidden_size, eps, r_hidden_size);
    op.Process();
}

extern "C" __global__ __aicore__ void rmsnorm_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y,
    uint32_t num_tokens, uint32_t hidden_size, float eps, float r_hidden_size
) {
    KernelRMSNorm<bfloat16_t> op;
    op.Init(x, weight, y, num_tokens, hidden_size, eps, r_hidden_size);
    op.Process();
}