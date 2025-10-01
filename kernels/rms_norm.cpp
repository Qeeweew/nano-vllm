/**
 * @file rmsnorm.cpp
 *
 * Kernel for Root Mean Square Normalization (RMSNorm) on Ascend.
 * Parallelized over the combined batch*heads dimension.
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
        uint32_t num_rows,
        uint32_t num_heads,
        uint32_t hidden_size,
        uint32_t x_stride0,
        uint32_t x_stride1,
        float eps,
        float r_hidden_size
    ) {
        this->num_rows = num_rows;
        this->num_heads = num_heads;
        this->hidden_size = hidden_size;
        this->x_stride0 = x_stride0;
        this->x_stride1 = x_stride1;
        this->eps = eps;
        this->r_hidden_size = r_hidden_size;

        // Parallelize over rows (tokens * heads). Each core gets a slice of rows.
        this->start_row = GetBlockIdx() * (num_rows / GetBlockNum()) + MIN(GetBlockIdx(), num_rows % GetBlockNum());
        this->tile_length = (num_rows / GetBlockNum()) + (GetBlockIdx() < num_rows % GetBlockNum() ? 1 : 0);
        
        // Set up global memory base pointers
        this->x_gm.SetGlobalBuffer((__gm__ T*)x_gm);
        this->weight_gm.SetGlobalBuffer((__gm__ T*)weight_gm);
        this->y_gm.SetGlobalBuffer((__gm__ T*)y_gm);
        
        // Allocate local memory buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, hidden_size * sizeof(T));

        // Calculation buffer for float32 intermediates.
        pipe.InitBuffer(calcBuf, 3 * hidden_size * sizeof(float) + 64 * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (tile_length == 0) {
            return;
        }
        
        // Load the weight vector once from GM.
        CopyInWeight();
        LocalTensor<T> weight_T_local = inQueueWeight.DeQue<T>();
        
        // Cast weight to float32 once, before the loop.
        LocalTensor<float> weight_fp32 = calcBuf.Get<float>(hidden_size);
        Cast(weight_fp32, weight_T_local, RoundMode::CAST_NONE, hidden_size);

        // Process each row (token-head pair) assigned to this core.
        for (uint32_t i = 0; i < this->tile_length; ++i) {
            CopyInX(i);
            Compute(weight_fp32);
            CopyOutY(i);
        }

        inQueueWeight.FreeTensor(weight_T_local);
    }

private:
    __aicore__ inline void CopyInWeight() {
        LocalTensor<T> weight_local = inQueueWeight.AllocTensor<T>();
        DataCopy(weight_local, weight_gm, hidden_size);
        inQueueWeight.EnQue(weight_local);
    }

    __aicore__ inline void CopyInX(uint32_t tile_idx) {
        const uint32_t current_row_idx = this->start_row + tile_idx;
        const uint32_t batch_idx = current_row_idx / this->num_heads;
        const uint32_t head_idx = current_row_idx % this->num_heads;
        const uint32_t offset = batch_idx * this->x_stride0 + head_idx * this->x_stride1;
        
        LocalTensor<T> x_local = inQueueX.AllocTensor<T>();
        DataCopy(x_local, x_gm[offset], hidden_size);
        inQueueX.EnQue(x_local);
    }
    
    __aicore__ inline void Compute(const LocalTensor<float>& weight_fp32) {
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<T> y_local = outQueueY.AllocTensor<T>();

        const uint32_t offset_base = hidden_size * sizeof(float);
        LocalTensor<float> x_fp32 = calcBuf.GetWithOffset<float>(hidden_size, offset_base);
        LocalTensor<float> result_fp32 = calcBuf.GetWithOffset<float>(hidden_size, offset_base + hidden_size * sizeof(float));
        LocalTensor<float> reduce_work = calcBuf.GetWithOffset<float>(64, offset_base + 2 * hidden_size * sizeof(float));
        LocalTensor<float> sum_val = reduce_work;

        Cast(x_fp32, x_local, RoundMode::CAST_NONE, hidden_size);
        Mul(result_fp32, x_fp32, x_fp32, hidden_size);
        ReduceSum(sum_val, result_fp32, reduce_work, hidden_size);
        
        Muls(sum_val, sum_val, r_hidden_size, 1);
        Adds(sum_val, sum_val, eps, 1);
        Rsqrt(sum_val, sum_val, 1);
        float rsqrt_val = sum_val.GetValue(0);

        Muls(result_fp32, x_fp32, rsqrt_val, hidden_size);
        Mul(result_fp32, result_fp32, weight_fp32, hidden_size);
        Cast(y_local, result_fp32, RoundMode::CAST_ROUND, hidden_size);
        
        outQueueY.EnQue(y_local);
        inQueueX.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOutY(uint32_t tile_idx) {
        // Calculate the offset for a contiguous output tensor.
        const uint32_t current_row_idx = this->start_row + tile_idx;
        const uint32_t offset = current_row_idx * this->hidden_size;

        LocalTensor<T> y_local = outQueueY.DeQue<T>();
        DataCopy(y_gm[offset], y_local, hidden_size);
        outQueueY.FreeTensor(y_local);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    TBuf<TPosition::VECCALC> calcBuf;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<T> x_gm, weight_gm, y_gm;
    
    uint32_t num_rows, num_heads, hidden_size;
    uint32_t start_row, tile_length;
    uint32_t x_stride0, x_stride1;
    float eps, r_hidden_size;
};


// Extern "C" entry points for FP16 and BF16 with updated signatures
extern "C" __global__ __aicore__ void rmsnorm_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y,
    uint32_t num_rows, uint32_t num_heads, uint32_t hidden_size,
    uint32_t x_stride0, uint32_t x_stride1,
    float eps, float r_hidden_size
) {
    KernelRMSNorm<half> op;
    op.Init(x, weight, y, num_rows, num_heads, hidden_size, x_stride0, x_stride1, eps, r_hidden_size);
    op.Process();
}

extern "C" __global__ __aicore__ void rmsnorm_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y,
    uint32_t num_rows, uint32_t num_heads, uint32_t hidden_size,
    uint32_t x_stride0, uint32_t x_stride1,
    float eps, float r_hidden_size
) {
    KernelRMSNorm<bfloat16_t> op;
    op.Init(x, weight, y, num_rows, num_heads, hidden_size, x_stride0, x_stride1, eps, r_hidden_size);
    op.Process();
}

// =================================================================================================
// Kernel for Fused Add + RMSNorm (Corrected and Verified Implementation)
// =================================================================================================
template<typename T>
class KernelAddRMSNorm {
public:
    __aicore__ inline KernelAddRMSNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x_gm,
        GM_ADDR residual_gm,
        GM_ADDR weight_gm,
        GM_ADDR y_gm,
        GM_ADDR residual_out_gm,
        uint32_t num_rows,
        uint32_t hidden_size,
        float eps,
        float r_hidden_size
    ) {
        this->num_rows = num_rows;
        this->hidden_size = hidden_size;
        this->eps = eps;
        this->r_hidden_size = r_hidden_size;

        // Parallelize over rows. Each core gets a slice of rows.
        this->start_row = GetBlockIdx() * (num_rows / GetBlockNum()) + MIN(GetBlockIdx(), num_rows % GetBlockNum());
        this->tile_length = (num_rows / GetBlockNum()) + (GetBlockIdx() < num_rows % GetBlockNum() ? 1 : 0);
        
        // Set up global memory base pointers
        this->x_gm.SetGlobalBuffer((__gm__ T*)x_gm);
        this->residual_gm.SetGlobalBuffer((__gm__ T*)residual_gm);
        this->weight_gm.SetGlobalBuffer((__gm__ T*)weight_gm);
        this->y_gm.SetGlobalBuffer((__gm__ T*)y_gm);
        this->residual_out_gm.SetGlobalBuffer((__gm__ T*)residual_out_gm);
        
        // Allocate local memory buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(inQueueResidual, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, hidden_size * sizeof(T));
        pipe.InitBuffer(outQueueResidual, BUFFER_NUM, hidden_size * sizeof(T));
        
        // Calculation buffer for float32 intermediates. Needs space for 3 vectors + reduce workspace
        pipe.InitBuffer(calcBuf, 3 * hidden_size * sizeof(float) + 64 * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (tile_length == 0) {
            return;
        }
        
        // Load weight once and cast to float32
        CopyInWeight();
        LocalTensor<T> weight_T_local = inQueueWeight.DeQue<T>();
        LocalTensor<float> weight_fp32 = calcBuf.Get<float>(hidden_size);
        Cast(weight_fp32, weight_T_local, RoundMode::CAST_NONE, hidden_size);

        // Main loop to process rows assigned to this core
        for (uint32_t i = 0; i < this->tile_length; ++i) {
            CopyIn(i);
            Compute(weight_fp32);
            CopyOut(i);
        }

        inQueueWeight.FreeTensor(weight_T_local);
    }

private:
    __aicore__ inline void CopyInWeight() {
        LocalTensor<T> weight_local = inQueueWeight.AllocTensor<T>();
        DataCopy(weight_local, weight_gm, hidden_size);
        inQueueWeight.EnQue(weight_local);
    }

    __aicore__ inline void CopyIn(uint32_t tile_idx) {
        const uint32_t current_row_idx = this->start_row + tile_idx;
        const uint32_t offset = current_row_idx * this->hidden_size;
        
        LocalTensor<T> x_local = inQueueX.AllocTensor<T>();
        DataCopy(x_local, x_gm[offset], hidden_size);
        inQueueX.EnQue(x_local);

        LocalTensor<T> residual_local = inQueueResidual.AllocTensor<T>();
        DataCopy(residual_local, residual_gm[offset], hidden_size);
        inQueueResidual.EnQue(residual_local);
    }
    
    __aicore__ inline void Compute(const LocalTensor<float>& weight_fp32) {
        // STEP 0: Dequeue inputs and allocate local tensors for outputs
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<T> residual_local = inQueueResidual.DeQue<T>();

        LocalTensor<T> y_local = outQueueY.AllocTensor<T>();
        LocalTensor<T> residual_out_local = outQueueResidual.AllocTensor<T>();

        // Allocate float32 buffers for calculation
        const uint32_t offset_base = hidden_size * sizeof(float);
        LocalTensor<float> x_fp32 = calcBuf.GetWithOffset<float>(hidden_size, offset_base);
        LocalTensor<float> add_result_fp32 = calcBuf.GetWithOffset<float>(hidden_size, offset_base + hidden_size * sizeof(float));
        LocalTensor<float> reduce_work = calcBuf.GetWithOffset<float>(64, offset_base + 2 * hidden_size * sizeof(float));
        LocalTensor<float> sum_val = reduce_work;

        // STEP 1: Perform Add: `add_result_fp32 = x + residual`
        Cast(x_fp32, x_local, RoundMode::CAST_NONE, hidden_size);
        Cast(add_result_fp32, residual_local, RoundMode::CAST_NONE, hidden_size);
        Add(add_result_fp32, x_fp32, add_result_fp32, hidden_size);

        // STEP 2: Enqueue the sum (which is the new residual) for CopyOut
        // THIS IS THE CRITICAL FIX: The result must be enqueued.
        Cast(residual_out_local, add_result_fp32, RoundMode::CAST_ROUND, hidden_size);
        outQueueResidual.EnQue(residual_out_local);

        // STEP 3: Perform RMSNorm on the sum (`add_result_fp32`)
        // Reuse x_fp32 buffer for squared result
        Mul(x_fp32, add_result_fp32, add_result_fp32, hidden_size); 
        ReduceSum(sum_val, x_fp32, reduce_work, hidden_size);
        
        Muls(sum_val, sum_val, r_hidden_size, 1);
        Adds(sum_val, sum_val, eps, 1);
        Rsqrt(sum_val, sum_val, 1);
        float rsqrt_val = sum_val.GetValue(0);

        // Reuse x_fp32 for normalized result
        Muls(x_fp32, add_result_fp32, rsqrt_val, hidden_size); 
        // Reuse add_result_fp32 for final weighted result
        Mul(add_result_fp32, x_fp32, weight_fp32, hidden_size); 
        
        // STEP 4: Enqueue the final normalized result for CopyOut
        Cast(y_local, add_result_fp32, RoundMode::CAST_ROUND, hidden_size);
        outQueueY.EnQue(y_local);
        
        // STEP 5: Free input tensors
        inQueueX.FreeTensor(x_local);
        inQueueResidual.FreeTensor(residual_local);
    }
    
    __aicore__ inline void CopyOut(uint32_t tile_idx) {
        const uint32_t current_row_idx = this->start_row + tile_idx;
        const uint32_t offset = current_row_idx * this->hidden_size;

        // Dequeue and copy out the final normalized result
        LocalTensor<T> y_local = outQueueY.DeQue<T>();
        DataCopy(y_gm[offset], y_local, hidden_size);
        outQueueY.FreeTensor(y_local);
        
        // Dequeue and copy out the intermediate sum (new residual)
        LocalTensor<T> residual_out_local = outQueueResidual.DeQue<T>();
        DataCopy(residual_out_gm[offset], residual_out_local, hidden_size);
        outQueueResidual.FreeTensor(residual_out_local);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueResidual, inQueueWeight;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY, outQueueResidual;
    TBuf<TPosition::VECCALC> calcBuf;

    GlobalTensor<T> x_gm, residual_gm, weight_gm, y_gm, residual_out_gm;
    
    uint32_t num_rows, hidden_size;
    uint32_t start_row, tile_length;
    float eps, r_hidden_size;
};

// Extern "C" entry points for Fused Add + RMSNorm
extern "C" __global__ __aicore__ void add_rmsnorm_fp16(
    GM_ADDR x, GM_ADDR residual, GM_ADDR weight, GM_ADDR y, GM_ADDR residual_out,
    uint32_t num_rows, uint32_t hidden_size,
    float eps, float r_hidden_size
) {
    KernelAddRMSNorm<half> op;
    op.Init(x, residual, weight, y, residual_out, num_rows, hidden_size, eps, r_hidden_size);
    op.Process();
}

extern "C" __global__ __aicore__ void add_rmsnorm_bf16(
    GM_ADDR x, GM_ADDR residual, GM_ADDR weight, GM_ADDR y, GM_ADDR residual_out,
    uint32_t num_rows, uint32_t hidden_size,
    float eps, float r_hidden_size
) {
    KernelAddRMSNorm<bfloat16_t> op;
    op.Init(x, residual, weight, y, residual_out, num_rows, hidden_size, eps, r_hidden_size);
    op.Process();
}