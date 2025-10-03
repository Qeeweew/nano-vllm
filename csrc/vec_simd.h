#include "utils.h"
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

template <ExecutionPolicy Policy>
static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;

    dispatch_for<Policy>(0, num_tokens, [&](int64_t i) {
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
    });
}

#elif defined(__AVX2__) || defined(__AVX__)

#include <immintrin.h>  // for AVX2 intrinsics

inline static __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

template <ExecutionPolicy Policy>
static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;

    dispatch_for<Policy>(0, num_tokens, [&](int64_t i) {
        float* row = C + i * intermediate_size;
        float* gate_part = row;
        float* up_part = row + size_div_2;

        int j = 0;

        // 主循环：每次处理 8 个 float（AVX2 支持 256-bit，8×float）
        for (; j + 8 <= size_div_2; j += 8) {
            // 加载 gate values: g[0..7]
            __m256 g = _mm256_load_ps(gate_part + j);

            // 计算 sigmoid(g) = g / (1 + exp(-g))
            __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);  // -g
            __m256 eg = ggml_v_expf(neg_g);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 denom = _mm256_add_ps(one, eg);
            __m256 sig = _mm256_div_ps(g, denom);

            // 加载 up_proj 值
            __m256 up = _mm256_load_ps(up_part + j);

            // SiLU: up * activation
            __m256 result = _mm256_mul_ps(up, sig);

            // 存储结果
            _mm256_store_ps(row + j, result);
        }

        // 尾部清理（剩余元素）
        for (; j < size_div_2; ++j) {
            float gate_val = gate_part[j];
            float activation = gate_val / (1.0f + expf(-gate_val));
            row[j] = up_part[j] * activation;
        }
    });
}

#else  // 非 NEON 和非 AVX2 回退到标量或 x86-SSE

// 可在此处添加 x86-SSE 实现或纯标量版本
// 示例为标量 fallback
template <ExecutionPolicy Policy>
static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;

    dispatch_for<Policy>(0, num_tokens, [&](int64_t i) {
        float* row = C + i * intermediate_size;
        float* gate_part = row;
        float* up_part = row + size_div_2;

        for (int j = 0; j < size_div_2; ++j) {
            float gate_val = gate_part[j];
            float activation = gate_val / (1.0f + expf(-gate_val));
            row[j] = up_part[j] * activation;
        }
    });
}

#endif