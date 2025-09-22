// 根据架构选择向量类型和函数
#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>  // for float32x4_t, etc.

inline static float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;

    at::parallel_for(0, num_tokens, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            float* row = C + i * intermediate_size;
            float* gate_part = row;
            float* up_part = row + size_div_2;

            int j = 0;

            // 主循环：每次处理 4 个 float（NEON 支持 128-bit，4×float）
            for (; j + 4 <= size_div_2; j += 4) {
                // 加载 gate values: g[0..3]
                float32x4_t g = vld1q_f32(gate_part + j);

                // 计算 sigmoid(g) = g / (1 + exp(-g))
                float32x4_t neg_g = vnegq_f32(g);
                float32x4_t eg = ggml_v_expf(neg_g);
                float32x4_t one = vdupq_n_f32(1.0f);
                float32x4_t denom = vaddq_f32(one, eg);
                float32x4_t sig = vdivq_f32(g, denom);

                // 加载 up_proj 值
                float32x4_t up = vld1q_f32(up_part + j);

                // SiLU: up * activation
                float32x4_t result = vmulq_f32(up, sig);

                // 存储结果
                vst1q_f32(row + j, result);
            }

            // 尾部清理（剩余元素）
            for (; j < size_div_2; ++j) {
                float gate_val = gate_part[j];
                float activation = gate_val / (1.0f + expf(-gate_val));
                row[j] = up_part[j] * activation;
            }
        }
    });
}

#else  // 非 NEON 回退到标量或 x86-SSE

// 可在此处添加 x86-SSE 实现或纯标量版本
// 示例为标量 fallback
static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;

    at::parallel_for(0, num_tokens, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            float* row = C + i * intermediate_size;
            float* gate_part = row;
            float* up_part = row + size_div_2;

            #pragma omp simd
            for (int j = 0; j < size_div_2; ++j) {
                float gate_val = gate_part[j];
                float activation = gate_val / (1.0f + expf(-gate_val));
                row[j] = up_part[j] * activation;
            }
        }
    });
}

#endif
