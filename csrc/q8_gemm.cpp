#include <torch/extension.h>
#include <type_traits>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <ATen/Parallel.h>
#include "utils.h"
#include "vec_simd.h"
#include <omp.h>
#include "moe_infer.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK8_0 32

using ggml_half = at::Half;

#if defined(__ARM_NEON)
#define MR 8
#define NR 4
#elif defined(__AVX2__)
#define MR 8
#define NR 8
#if defined (__AVXVNNI__)
#define _mm256_dpbusd_epi32 _mm256_dpbusd_avx_epi32
#endif
#endif

template<typename D_TYPE>
static inline void quantize_block_q8_0(const float *x, D_TYPE* y_d, int8_t* y_qs) {
    float d;
#if defined(__ARM_NEON)
    float32x4_t srcv [8];
    float32x4_t asrcv[8];
    float32x4_t amaxv[4];

    for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + 4*j);
    for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);
    for (int j = 0; j < 4; j++) amaxv[j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
    for (int j = 0; j < 2; j++) amaxv[j] = vmaxq_f32(amaxv[2*j], amaxv[2*j+1]);
    const float amax = vmaxvq_f32(vmaxq_f32(amaxv[0], amaxv[1]));

    d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    for (int j = 0; j < 8; j+=2) {
        const float32x4_t v0  = vmulq_n_f32(srcv[j], id);
        const float32x4_t v1  = vmulq_n_f32(srcv[j + 1], id);
        const int32x4_t   v0_i32 = vcvtnq_s32_f32(v0);
        const int32x4_t   v1_i32 = vcvtnq_s32_f32(v1);
        const int16x4_t   v0_i16 = vqmovn_s32(v0_i32);
        const int16x4_t   v1_i16 = vqmovn_s32(v1_i32);
        const int8x8_t    vi8  = vqmovn_s16(vcombine_s16(v0_i16, v1_i16));
        vst1_s8(y_qs + 4*j, vi8);
    }
#elif defined(__AVX2__)
    // Load elements into 4 AVX vectors
    __m256 v0 = _mm256_loadu_ps( x );
    __m256 v1 = _mm256_loadu_ps( x + 8 );
    __m256 v2 = _mm256_loadu_ps( x + 16 );
    __m256 v3 = _mm256_loadu_ps( x + 24 );

    // Compute max(abs(e)) for the block
    const __m256 signBit = _mm256_set1_ps( -0.0f );
    __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
    maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

    __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
    max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
    const float maxScalar = _mm_cvtss_f32( max4 );

    // Quantize these floats
    d = maxScalar / 127.f;
    const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
    const __m256 mul = _mm256_set1_ps( id );

    // Apply the multiplier
    v0 = _mm256_mul_ps( v0, mul );
    v1 = _mm256_mul_ps( v1, mul );
    v2 = _mm256_mul_ps( v2, mul );
    v3 = _mm256_mul_ps( v3, mul );

    // Round to nearest integer
    v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
    v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
    v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
    v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

    // Convert floats to integers
    __m256i i0 = _mm256_cvtps_epi32( v0 );
    __m256i i1 = _mm256_cvtps_epi32( v1 );
    __m256i i2 = _mm256_cvtps_epi32( v2 );
    __m256i i3 = _mm256_cvtps_epi32( v3 );

    // Convert int32 to int16
    i0 = _mm256_packs_epi32( i0, i1 );
    i2 = _mm256_packs_epi32( i2, i3 );
    i0 = _mm256_packs_epi16( i0, i2 );
    const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
    i0 = _mm256_permutevar8x32_epi32( i0, perm );
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y_qs), i0);
#endif
    *y_d = d;
}

// Template specialized microkernel implementation
template <int MR_T, typename B_SCALE_TYPE>
static void gemm_q8_0_microkernel_specialized(
    int kc_size,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed,
    float* C, int ldc,
    bool accumulate)
{
    // Static assert ensures MR_T is valid at compile time
    static_assert(MR_T > 0 && MR_T <= MR, "MR_T must be within (0, MR]");

#if defined(__ARM_NEON)
    const int KC_BLOCKS = kc_size / QK8_0;

    // Arrays are declared with MR elements to simplify indexing and pointer arithmetic
    // while loops are adjusted to MR_T to avoid redundant work.
    float32x4_t c_v[MR_T];
    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vld1q_f32(C + i * ldc);
    } else {
        for (int i = 0; i < MR_T; ++i) c_v[i] = vdupq_n_f32(0.0f);
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        // prefetch
        if (k_block < KC_BLOCKS - 1) {
            __builtin_prefetch(a_ptr + QK8_0 * MR, 0, 0);
            __builtin_prefetch(b_ptr + QK8_0 * NR, 0, 0);
        }

        int32x4_t sum_v[MR];
        for (int i = 0; i < MR_T; ++i) sum_v[i] = vdupq_n_s32(0);

        for (int k4_step = 0; k4_step < QK8_0 / 4; ++k4_step) {
            int8x16_t a_vec_0 = vld1q_s8(a_ptr);
            a_ptr += 16;
            int8x16_t a_vec_1;
            if constexpr (MR_T > 4) {
                a_vec_1 = vld1q_s8(a_ptr);
            }
            a_ptr += 16; // a_ptr always advances by 32 bytes (2 * 16 bytes) per k4_step

            int8x16_t b_vec = vld1q_s8(b_ptr);
            b_ptr += 16;

            // Use if constexpr to conditionally compile dot product operations
            if constexpr (MR_T > 0) sum_v[0] = vdotq_laneq_s32(sum_v[0], b_vec, a_vec_0, 0);
            if constexpr (MR_T > 1) sum_v[1] = vdotq_laneq_s32(sum_v[1], b_vec, a_vec_0, 1);
            if constexpr (MR_T > 2) sum_v[2] = vdotq_laneq_s32(sum_v[2], b_vec, a_vec_0, 2);
            if constexpr (MR_T > 3) sum_v[3] = vdotq_laneq_s32(sum_v[3], b_vec, a_vec_0, 3);
            if constexpr (MR_T > 4) sum_v[4] = vdotq_laneq_s32(sum_v[4], b_vec, a_vec_1, 0);
            if constexpr (MR_T > 5) sum_v[5] = vdotq_laneq_s32(sum_v[5], b_vec, a_vec_1, 1);
            if constexpr (MR_T > 6) sum_v[6] = vdotq_laneq_s32(sum_v[6], b_vec, a_vec_1, 2);
            if constexpr (MR_T > 7) sum_v[7] = vdotq_laneq_s32(sum_v[7], b_vec, a_vec_1, 3);
        }

        float32x4_t d_b_v;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            d_b_v = vld1q_f32(bd_ptr);
        } else {
            d_b_v = vcvt_f32_f16(vld1_f16((const __fp16 *) bd_ptr));
        }
        bd_ptr += NR;

        float32x4_t d_a_v0, d_a_v1;
        d_a_v0 = vld1q_f32(ad_ptr);
        if constexpr (MR_T > 4) d_a_v1 = vld1q_f32(ad_ptr + 4);
        ad_ptr += MR;

        if constexpr (MR_T > 0) c_v[0] = vmlaq_laneq_f32(c_v[0], vmulq_f32(vcvtq_f32_s32(sum_v[0]), d_b_v), d_a_v0, 0);
        if constexpr (MR_T > 1) c_v[1] = vmlaq_laneq_f32(c_v[1], vmulq_f32(vcvtq_f32_s32(sum_v[1]), d_b_v), d_a_v0, 1);
        if constexpr (MR_T > 2) c_v[2] = vmlaq_laneq_f32(c_v[2], vmulq_f32(vcvtq_f32_s32(sum_v[2]), d_b_v), d_a_v0, 2);
        if constexpr (MR_T > 3) c_v[3] = vmlaq_laneq_f32(c_v[3], vmulq_f32(vcvtq_f32_s32(sum_v[3]), d_b_v), d_a_v0, 3);

        if constexpr (MR_T > 4) c_v[4] = vmlaq_laneq_f32(c_v[4], vmulq_f32(vcvtq_f32_s32(sum_v[4]), d_b_v), d_a_v1, 0);
        if constexpr (MR_T > 5) c_v[5] = vmlaq_laneq_f32(c_v[5], vmulq_f32(vcvtq_f32_s32(sum_v[5]), d_b_v), d_a_v1, 1);
        if constexpr (MR_T > 6) c_v[6] = vmlaq_laneq_f32(c_v[6], vmulq_f32(vcvtq_f32_s32(sum_v[6]), d_b_v), d_a_v1, 2);
        if constexpr (MR_T > 7) c_v[7] = vmlaq_laneq_f32(c_v[7], vmulq_f32(vcvtq_f32_s32(sum_v[7]), d_b_v), d_a_v1, 3);
    }

    for (int i = 0; i < MR_T; ++i) {
        vst1q_f32(C + i * ldc, c_v[i]);
    }
#elif defined(__AVX2__)
    const int KC_BLOCKS = kc_size / QK8_0;

    __m256 c_v[MR_T];
    if (accumulate) {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_loadu_ps(C + i * ldc);
        }
    } else {
        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_setzero_ps();
        }
    }

    const int8_t* a_ptr = A_qs_packed;
    const int8_t* b_ptr = B_qs_packed;
    const float* ad_ptr = A_d_packed;
    const B_SCALE_TYPE* bd_ptr = B_d_packed;

    for (int k_block = 0; k_block < KC_BLOCKS; ++k_block) {
        __m256i sum[MR];
        for (int i = 0; i < MR_T; ++i) sum[i] = _mm256_setzero_si256();

#if defined(__AVXVNNI__) || defined(__AVX512VNNI__)
        __m256i sum_a_vec = _mm256_setzero_si256();

        for (int k = 0; k < QK8_0; k += 4) {
            __m256i b_vec = _mm256_load_si256((__m256i const*)b_ptr);
            b_vec = _mm256_sub_epi8(b_vec, _mm256_set1_epi8(-128)); // b_vec + 128
            b_ptr += NR * 4;
            __m256i a_vec = _mm256_load_si256((__m256i const*)a_ptr);
            sum_a_vec = _mm256_dpbusd_epi32(sum_a_vec, _mm256_set1_epi8(1), a_vec);

            for (int i = 0; i < MR_T; ++i) {
                __m256i a_vec = _mm256_set1_epi32(*((int*)(a_ptr + i * 4)));
                sum[i] = _mm256_dpbusd_epi32(sum[i], b_vec, a_vec);
            }
            a_ptr += MR * 4; // Move to next set of A data
        }

        __m256 bd_vec;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            bd_vec = _mm256_loadu_ps(bd_ptr);
        } else {
            bd_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*) bd_ptr));
        }
        bd_ptr += NR;

        union { float f[8]; __m256 v; } sum_a_vec_f;
        sum_a_vec_f.v = _mm256_cvtepi32_ps(sum_a_vec);
        for (int i = 0; i < MR_T; ++i) {
            __m256 sum_f = _mm256_fmadd_ps(_mm256_set1_ps(sum_a_vec_f.f[i]), _mm256_set1_ps(-128.0f), _mm256_cvtepi32_ps(sum[i]));
            c_v[i] = _mm256_fmadd_ps(_mm256_mul_ps(sum_f, _mm256_broadcast_ss(&ad_ptr[i])) , bd_vec, c_v[i]);
        }
#else
        for (int k = 0; k < QK8_0; k += 4) {
            __m256i b_vec = _mm256_load_si256((__m256i const*)b_ptr);
            b_ptr += NR * 4;

            __m256i b_vec_abs = _mm256_sign_epi8(b_vec, b_vec); // abs
            // Process each row of A
            for (int i = 0; i < MR_T; ++i) {
                __m256i a_vec = _mm256_set1_epi32(*((const int*)(a_ptr + i * 4)));
                __m256i a_vec_sign_b = _mm256_sign_epi8(a_vec, b_vec);
                sum[i] += _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(b_vec_abs, a_vec_sign_b));
            }
            a_ptr += MR * 4; // Move to next set of A data
        }

        __m256 bd_vec;
        if constexpr (std::is_same_v<B_SCALE_TYPE, float>) {
            bd_vec = _mm256_loadu_ps(bd_ptr);
        } else {
            bd_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*) bd_ptr));
        }
        bd_ptr += NR;

        for (int i = 0; i < MR_T; ++i) {
            c_v[i] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sum[i]), _mm256_broadcast_ss(&ad_ptr[i])) , bd_vec, c_v[i]);
        }
#endif

        ad_ptr += MR;
    }

    for (int i = 0; i < MR_T; ++i) { // Loop only up to MR_T
        _mm256_storeu_ps(C + i * ldc, c_v[i]);
    }
#else
    // Error case if neither NEON nor AVX2 is defined
    static_assert(false, "no implementation");
#endif
}

template<typename B_SCALE_TYPE>
void gemm_q8_0_microkernel(int kc_size, int mr, const int8_t* A_qs_packed, const float* A_d_packed, const int8_t* B_qs_packed, const B_SCALE_TYPE* B_d_packed, float* C, int ldc, bool accumulate) {
    assert(mr > 0 && mr <= MR);
    switch (mr) {
        case 1: gemm_q8_0_microkernel_specialized<1>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 2: gemm_q8_0_microkernel_specialized<2>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 3: gemm_q8_0_microkernel_specialized<3>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 4: gemm_q8_0_microkernel_specialized<4>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 5: gemm_q8_0_microkernel_specialized<5>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 6: gemm_q8_0_microkernel_specialized<6>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 7: gemm_q8_0_microkernel_specialized<7>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
        case 8: gemm_q8_0_microkernel_specialized<8>(kc_size, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed, C, ldc, accumulate); break;
    }
}

template <ExecutionPolicy Policy, typename T>
static void quantize_pack_A_q8_0(
    int M, int K, const T* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;
    
    dispatch_for<Policy>(0, num_a_packs, [&](int64_t i_pack) {
        int8_t a_qs_buf[MR * QK8_0];
        int i = i_pack * MR;
        for (int j = 0; j < K_BLOCKS; ++j) {
            int M_rem = std::min(MR, M - i);
            float* current_A_d_ptr = A_d_packed + i * K_BLOCKS + j * MR;
            for (int row = 0; row < M_rem; ++row) {
                if constexpr(std::is_same_v<T, float>) {
                    quantize_block_q8_0(A + (i + row) * lda + j * QK8_0, &current_A_d_ptr[row], &a_qs_buf[row * QK8_0]);
                } else {
                    float a_float_buf[QK8_0];
                    for (int k = 0; k < QK8_0; ++k) {
                        a_float_buf[k] = static_cast<float>(A[(i + row) * lda + j * QK8_0 + k]);
                    }
                    quantize_block_q8_0(a_float_buf, &current_A_d_ptr[row], &a_qs_buf[row * QK8_0]);
                }
            }
            int8_t* current_qs_ptr = A_qs_packed + i * K + j * QK8_0 * MR;
            for (int k = 0; k < QK8_0; k += 4) {
                for (int row = 0; row < MR; ++row) {
                    memcpy(current_qs_ptr, &a_qs_buf[row * QK8_0 + k], 4);
                    current_qs_ptr += 4;
                }
            }
        }
    });
}

template <ExecutionPolicy Policy>
static void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16,
    float* C, int ldc)
{
    const int K_BLOCKS = K / QK8_0;
    constexpr int MC = 32; constexpr int KC = 1024; constexpr int NC = 32;
    const int num_nc_blocks = (N + NC - 1) / NC;

    if (M <= MR) {
        dispatch_for<Policy>(0, num_nc_blocks, [&](int64_t jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int jr = 0; jr < nc; jr += NR) {
                gemm_q8_0_microkernel(
                    K, M,
                    A_qs_packed,
                    A_d_packed,
                    B_qs_packed + (jc + jr) * K,
                    B_d_packed_f16 + (jc + jr) * (K / QK8_0),
                    C + (jc + jr),
                    ldc,
                    false);
            }
        });
    } else {
        dispatch_for<Policy>(0, num_nc_blocks, [&](int64_t jc_idx) {
            int jc = jc_idx * NC;
            const int nc = std::min(NC, N - jc);
            for (int kc = 0; kc < K; kc += KC) {
                const int kc_size = std::min(KC, K - kc);
                const int k_block_offset = kc / QK8_0;
                for (int ic = 0; ic < M; ic += MC) {
                    const int mc = std::min(MC, M - ic);
                    for (int jr = 0; jr < nc; jr += NR) {
                        for (int ir = 0; ir < mc; ir += MR) {
                            gemm_q8_0_microkernel(
                                kc_size, std::min(MR, mc - ir),
                                A_qs_packed + (ic + ir) * K + kc * MR,
                                A_d_packed + (ic + ir) * K_BLOCKS + k_block_offset * MR,
                                B_qs_packed + (jc + jr) * K + kc * NR,
                                B_d_packed_f16 + (jc + jr) * K_BLOCKS + k_block_offset * NR,
                                C + (ic + ir) * ldc + (jc + jr), ldc, kc != 0);
                        }
                    }
                }
            }
        });
    }
}

void gemm_q8_0_aten_parallel_packed(int M, int N, int K, const float* A, int lda, const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16, float* C) {
    assert(K % QK8_0 == 0);
    const int K_BLOCKS = K / QK8_0;
    const int M_CEIL = (M + MR - 1) / MR * MR;
    
    int8_t* A_qs_packed = static_cast<int8_t*>(std::aligned_alloc(64, M_CEIL * K * sizeof(int8_t)));
    float* A_d_packed = static_cast<float*>(std::aligned_alloc(64, M_CEIL * K_BLOCKS * sizeof(float)));

    // Phase 1: Quantize and Pack A
    quantize_pack_A_q8_0<ExecutionPolicy::Parallel,float>(M, K, A, lda, A_qs_packed, A_d_packed);

    // Phase 2: GEMM Computation
    gemm_q8_0_compute_packed<ExecutionPolicy::Parallel>(M, N, K, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed_f16, C, N);

    std::free(A_qs_packed);
    std::free(A_d_packed);
}

// ==========================================================================================
// PYTORCH BINDINGS
// ==========================================================================================

template<typename D_TYPE>
static void quantize_row_q8_0_no_repack(
    const float * src,
    int8_t* dest_qs,
    D_TYPE* dest_d,
    int64_t K
) {
    const int64_t k_blocks = K / QK8_0;
    for (int i = 0; i < k_blocks; ++i) {
        quantize_block_q8_0(src + i * QK8_0, &dest_d[i], dest_qs + i * QK8_0);
    }
}

// New, CORRECT implementation of the repack function that mimics pack_B_q8_0.
template<typename D_TYPE>
void repack_B_q8_0_from_ptr(
    int64_t N, int64_t K,
    const int8_t* src_qs, const D_TYPE* src_d,
    int8_t* dest_qs_packed, D_TYPE* dest_d_packed
) {
    const int K_BLOCKS = K / QK8_0;

    for (int j = 0; j < N; j += NR) {
        // First, pack all the scales for the next NR rows
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    *dest_d_packed = src_d[(j + col) * K_BLOCKS + k_block];
                }
                dest_d_packed++;
            }
        }

        // Then, pack all the quants for the next NR rows with interleaving
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            // Interleave in chunks of 4 bytes, which is a common SIMD optimization
            for (int k_rem = 0; k_rem < QK8_0; k_rem += 4) {
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        const int8_t* src_ptr = src_qs + (j + col) * K + k_block * QK8_0 + k_rem;
                        memcpy(dest_qs_packed, src_ptr, 4);
                    }
                    dest_qs_packed += 4;
                }
            }
        }
    }
}

template void repack_B_q8_0_from_ptr<at::Half>(
    int64_t N, int64_t K,
    const int8_t* src_qs, const at::Half* src_d,
    int8_t* dest_qs_packed, at::Half* dest_d_packed
);


std::vector<torch::Tensor> quantize_weight_only(torch::Tensor B_float) {
    TORCH_CHECK(B_float.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(B_float.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(B_float.device().is_cpu(), "Weight must be on CPU");
    
    const auto N = B_float.size(0);
    const auto K = B_float.size(1);
    
    TORCH_CHECK(K % QK8_0 == 0, "Weight's K dimension must be a multiple of ", QK8_0);

    const int K_BLOCKS = K / QK8_0;
    auto B_qs_tensor = torch::empty({N, K}, torch::kInt8);
    auto B_d_tensor = torch::empty({N, K_BLOCKS}, torch::kHalf);

    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; ++j) {
            quantize_row_q8_0_no_repack(
                B_float.data_ptr<float>() + j * K,
                B_qs_tensor.data_ptr<int8_t>() + j * K,
                B_d_tensor.data_ptr<at::Half>() + j * K_BLOCKS,
                K
            );
        }
    });

    return {B_qs_tensor, B_d_tensor};
}

torch::Tensor q8_gemm(
    torch::Tensor A_float,
    torch::Tensor B_qs_packed,
    torch::Tensor B_d_packed
) {
    TORCH_CHECK(A_float.dim() == 2, "A must be 2D");
    TORCH_CHECK(A_float.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(A_float.device().is_cpu(), "A must be on CPU");

    TORCH_CHECK(B_qs_packed.is_contiguous(), "B_qs must be contiguous");
    TORCH_CHECK(B_qs_packed.device().is_cpu(), "B_qs must be on CPU");

    TORCH_CHECK(B_d_packed.is_contiguous(), "B_d must be contiguous");
    TORCH_CHECK(B_d_packed.device().is_cpu(), "B_d must be on CPU");

    const auto M = A_float.size(0);
    const auto K = A_float.size(1);
    const auto N = B_qs_packed.size(0);

    TORCH_CHECK(K == B_qs_packed.size(1), "A and B shapes are not compatible for matmul");
    TORCH_CHECK(K / QK8_0 == B_d_packed.size(1), "B_d shape is incorrect");
    
    auto C_tensor = torch::empty({M, N}, torch::kFloat);

    gemm_q8_0_aten_parallel_packed(
        M, N, K,
        A_float.data_ptr<float>(), K,
        B_qs_packed.data_ptr<int8_t>(),
        reinterpret_cast<const ggml_half*>(B_d_packed.data_ptr<at::Half>()),
        C_tensor.data_ptr<float>()
    );

    return C_tensor;
}

struct MoETokenInfo {
    int32_t token_id;
    int32_t expert_idx_in_tok; // The k-th expert for this token (0 to top_k-1)
};

// Preprocesses routing information to create maps for efficient gather and scatter.
static void preprocess_moe_routing(
    int num_experts, int top_k, int num_tokens,
    const int32_t* selected_experts,
    std::vector<int>& expert_counts,
    std::vector<int>& expert_starts,
    std::vector<MoETokenInfo>& token_map,   // For gather (expert-centric)
    std::vector<int32_t>& scatter_map      // For scatter (token-centric)
) {
    // Count tokens per expert
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int expert_id = selected_experts[t * top_k + k];
            if (expert_id >= 0 && expert_id < num_experts) {
                expert_counts[expert_id]++;
            }
        }
    }

    // Calculate start indices for each expert's token batch
    expert_starts[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        expert_starts[i + 1] = expert_starts[i] + expert_counts[i];
    }
    
    const int total_expert_tokens = expert_starts[num_experts];
    token_map.resize(total_expert_tokens);
    scatter_map.resize(num_tokens * top_k);
    
    // Create the gather map (token_map) and the scatter map
    // `current_expert_counts` tracks the current position within each expert's batch.
    std::vector<int> current_expert_counts(num_experts, 0);
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int expert_id = selected_experts[t * top_k + k];
            if (expert_id >= 0 && expert_id < num_experts) {
                int pos_in_gathered_tensor = expert_starts[expert_id] + current_expert_counts[expert_id]++;
                // gather map: stores original token info at the gathered position
                token_map[pos_in_gathered_tensor] = {t, k};
                // scatter map: stores the gathered position for the original token
                scatter_map[t * top_k + k] = pos_in_gathered_tensor;
            } else {
                scatter_map[t * top_k + k] = -1; // Mark invalid experts
            }
        }
    }
}

template <ExecutionPolicy Policy>
static void pack_A_q8_0_from_quantized_indirect(
    int M, int K,
    const int8_t* x_qs_base,
    const float* x_d_base,
    const MoETokenInfo* token_map,
    int token_offset,
    int8_t* A_qs_packed,
    float* A_d_packed
) {
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    dispatch_for<Policy>(0, num_a_packs, [&](int64_t i_pack) {
        int row_in_expert = i_pack * MR;
        for (int j = 0; j < K_BLOCKS; ++j) {
            int M_rem = std::min(MR, M - row_in_expert);
            float* current_A_d_ptr = A_d_packed + row_in_expert * K_BLOCKS + j * MR;
            for (int local_row = 0; local_row < M_rem; ++local_row) {
                int global_token_id = token_map[token_offset + row_in_expert + local_row].token_id;
                current_A_d_ptr[local_row] = x_d_base[global_token_id * K_BLOCKS + j];
            }
            int8_t* current_qs_ptr = A_qs_packed + row_in_expert * K + j * QK8_0 * MR;
            for (int k = 0; k < QK8_0; k += 4) {
                for (int local_row = 0; local_row < MR; ++local_row) {
                    if (local_row < M_rem) {
                        int global_token_id = token_map[token_offset + row_in_expert + local_row].token_id;
                        memcpy(current_qs_ptr,
                               x_qs_base + global_token_id * K + j * QK8_0 + k,
                               4);
                    }
                    current_qs_ptr += 4;
                }
            }
        }
    });
}

// Helper to align memory addresses to 64 bytes
static inline size_t align_to_64(size_t n) {
    return (n + 63) & ~63;
}

// A thread-local workspace that manages its own memory.
// It allocates or resizes its buffer only when necessary.
struct ThreadWorkspace {
    char* memory_pool = nullptr;
    size_t current_size = 0;

    // Pointers that will point into the memory_pool
    int8_t* A_qs_packed1;
    float* A_d_packed1;
    float* expert_intermediate1;
    int8_t* A_qs_packed2;
    float* A_d_packed2;
    float* temp_row_buffer; // Added for row-wise type conversion

    ThreadWorkspace() = default;

    // Ensures the workspace is large enough for the current task.
    // Re-allocates only if the required size is larger than the current allocation.
    void ensure_size(int m_ceil, int k_hidden, int k_inter, int k_inter_x2) {
        const int k_hidden_k_blocks = k_hidden / QK8_0;
        const int k_inter_k_blocks = k_inter / QK8_0;

        // Calculate sizes and offsets for all temporary buffers
        const size_t size_A_qs1 = m_ceil * k_hidden * sizeof(int8_t);
        const size_t size_A_d1 = m_ceil * k_hidden_k_blocks * sizeof(float);
        const size_t size_intermediate1 = m_ceil * k_inter_x2 * sizeof(float);
        const size_t size_A_qs2 = m_ceil * k_inter * sizeof(int8_t);
        const size_t size_A_d2 = m_ceil * k_inter_k_blocks * sizeof(float);
        const size_t size_temp_row = k_hidden * sizeof(float); // Size for the new buffer

        const size_t offset_A_qs1 = 0;
        const size_t offset_A_d1 = align_to_64(offset_A_qs1 + size_A_qs1);
        const size_t offset_intermediate1 = align_to_64(offset_A_d1 + size_A_d1);
        const size_t offset_A_qs2 = align_to_64(offset_intermediate1 + size_intermediate1);
        const size_t offset_A_d2 = align_to_64(offset_A_qs2 + size_A_qs2);
        const size_t offset_temp_row = align_to_64(offset_A_d2 + size_A_d2); // Offset for the new buffer
        
        const size_t required_size = align_to_64(offset_temp_row + size_temp_row); // Updated required size

        if (required_size > current_size) {
            if (memory_pool) {
                std::free(memory_pool);
            }
            memory_pool = static_cast<char*>(std::aligned_alloc(64, required_size));
            current_size = required_size;
        }

        // Set up the pointers into the (potentially newly allocated) memory pool.
        // This must be done on every call in case dimensions change, even if reallocation
        // doesn't happen.
        A_qs_packed1 = reinterpret_cast<int8_t*>(memory_pool + offset_A_qs1);
        A_d_packed1 = reinterpret_cast<float*>(memory_pool + offset_A_d1);
        expert_intermediate1 = reinterpret_cast<float*>(memory_pool + offset_intermediate1);
        A_qs_packed2 = reinterpret_cast<int8_t*>(memory_pool + offset_A_qs2);
        A_d_packed2 = reinterpret_cast<float*>(memory_pool + offset_A_d2);
        temp_row_buffer = reinterpret_cast<float*>(memory_pool + offset_temp_row); // Set the new pointer
    }

    // Destructor automatically cleans up memory when the thread terminates.
    ~ThreadWorkspace() {
        if (memory_pool) {
            std::free(memory_pool);
        }
    }
};

// Declare a single thread_local instance. Each thread will get its own copy.
static thread_local ThreadWorkspace ws;

// =================================================================================================
// STRATEGY 1: N-AXIS PARALLELISM (for few active experts)
// =================================================================================================
static void moe_q8_forward_n_axis_parallel(
    int num_experts, const std::vector<int>& expert_counts, const std::vector<int>& expert_starts, const std::vector<int>& active_expert_ids,
    const MoETokenInfo* token_map, int total_expert_tokens,
    int hidden_dim, int intermediate_size, int intermediate_size_x2,
    const int8_t* x_qs, const float* x_d,
    const int8_t* gate_up_qs_stacked_ptr, const at::Half* gate_up_d_stacked_ptr,
    const int8_t* down_proj_qs_stacked_ptr, const at::Half* down_proj_d_stacked_ptr,
    float* expert_intermediate2 // Output buffer
) {
    int num_threads = omp_get_max_threads();
    const int hidden_dim_k_blocks = hidden_dim / QK8_0;
    const int intermediate_dim_k_blocks = intermediate_size / QK8_0;
    int active_experts_count = static_cast<int>(active_expert_ids.size());

    // --- Pre-computation: Map threads to experts and their tasks ---
    struct ThreadTask {
        int expert_id, active_expert_id, num_tokens, token_offset;
        int thread_idx_in_expert, total_threads_for_expert;
    };
    std::vector<ThreadTask> thread_map(num_threads);
    int threads_per_expert = num_threads / active_experts_count;
    int remainder_threads = num_threads % active_experts_count;
    int current_thread = 0;
    for (int i = 0; i < active_experts_count; ++i) {
        int count = threads_per_expert + (i < remainder_threads ? 1 : 0);
        int expert_id = active_expert_ids[i];
        for (int j = 0; j < count; ++j) {
            if (current_thread < num_threads) {
                thread_map[current_thread++] = {
                    expert_id, i, expert_counts[expert_id], expert_starts[expert_id], j, count
                };
            }
        }
    }

    // --- Allocate ALL temporary buffers for this strategy ---
    std::vector<int> m_ceil_offset(active_experts_count);
    int m_ceil_cumulative = 0;

    for (int i = 0; i < active_experts_count; ++i) {
        int exp_id = active_expert_ids[i];
        const int M = expert_counts[exp_id];
        const int M_CEIL = (M + MR - 1) / MR * MR;
        m_ceil_offset[i] = m_ceil_cumulative;
        m_ceil_cumulative += M_CEIL;
    }

    // Buffer for GEMM 1 packed A
    size_t total_packed_A1_qs_size = m_ceil_cumulative * hidden_dim;
    size_t total_packed_A1_d_size = m_ceil_cumulative * hidden_dim_k_blocks;
    size_t total_packed_A2_qs_size = m_ceil_cumulative * intermediate_size;
    size_t total_packed_A2_d_size = m_ceil_cumulative * intermediate_dim_k_blocks;

    int8_t* packed_A1_qs_all = static_cast<int8_t*>(std::aligned_alloc(64, total_packed_A1_qs_size * sizeof(int8_t)));
    float* packed_A1_d_all = static_cast<float*>(std::aligned_alloc(64, total_packed_A1_d_size * sizeof(float)));
    float* expert_intermediate1 = static_cast<float*>(std::aligned_alloc(64, total_expert_tokens * intermediate_size_x2 * sizeof(float)));
    int8_t* packed_A2_qs_all = static_cast<int8_t*>(std::aligned_alloc(64, total_packed_A2_qs_size * sizeof(int8_t)));
    float* packed_A2_d_all = static_cast<float*>(std::aligned_alloc(64, total_packed_A2_d_size * sizeof(float)));

    #pragma omp parallel num_threads(num_threads)
    {
        const int thread_id = omp_get_thread_num();
        const auto& task = thread_map[thread_id];
        const int exp_id = task.expert_id;
        const int active_exp_id = task.active_expert_id;
        const int M = task.num_tokens;
        const int T_exp = task.total_threads_for_expert;
        const int t_idx = task.thread_idx_in_expert;

        int8_t* my_A1_qs_ptr = packed_A1_qs_all + m_ceil_offset[active_exp_id] * hidden_dim;
        float* my_A1_d_ptr = packed_A1_d_all + m_ceil_offset[active_exp_id] * hidden_dim_k_blocks;
        int8_t* my_A2_qs_ptr = packed_A2_qs_all + m_ceil_offset[active_exp_id] * intermediate_size;
        float* my_A2_d_ptr = packed_A2_d_all + m_ceil_offset[active_exp_id] * intermediate_dim_k_blocks;
        const int8_t* gate_up_qs_ptr = gate_up_qs_stacked_ptr + exp_id * intermediate_size_x2 * hidden_dim;
        const at::Half* gate_up_d_ptr = gate_up_d_stacked_ptr + exp_id * intermediate_size_x2 * hidden_dim_k_blocks;
        const int8_t* down_proj_qs_ptr = down_proj_qs_stacked_ptr + exp_id * hidden_dim * intermediate_size;
        const at::Half* down_proj_d_ptr = down_proj_d_stacked_ptr + exp_id * hidden_dim * intermediate_dim_k_blocks;

        if (t_idx == 0) {
            pack_A_q8_0_from_quantized_indirect<ExecutionPolicy::Sequential>(
                M, hidden_dim, x_qs, x_d, token_map, task.token_offset,
                my_A1_qs_ptr, my_A1_d_ptr
            );
        }

        #pragma omp barrier

        // --- FFN Up-Projection (GEMM 1), split by N ---
        constexpr int NC = 32;
        int n_chunk1 = (intermediate_size_x2 + T_exp - 1) / T_exp;
        n_chunk1 = ((n_chunk1 + NC - 1) / NC) * NC;
        int n_start1 = t_idx * n_chunk1;
        int n_size1 = std::min(n_chunk1, intermediate_size_x2 - n_start1);
        if (n_size1 > 0) {
            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                M, n_size1, hidden_dim, my_A1_qs_ptr, my_A1_d_ptr,
                gate_up_qs_ptr + n_start1 * hidden_dim,
                reinterpret_cast<const ggml_half*>(gate_up_d_ptr) + n_start1 * hidden_dim_k_blocks,
                expert_intermediate1 + task.token_offset * intermediate_size_x2 + n_start1, intermediate_size_x2
            );
        }

        #pragma omp barrier

        // --- SILU Activation & Quantize for GEMM 2 ---
        // Only the first thread for each expert performs these steps for the whole expert's data.
        if (t_idx == 0) {
            // 1. SILU activation
            silu_and_mul<ExecutionPolicy::Sequential>(
                expert_intermediate1 + task.token_offset * intermediate_size_x2,
                M, intermediate_size_x2
            );
            // 2. Quantize and pack the result for the next GEMM
            quantize_pack_A_q8_0<ExecutionPolicy::Sequential, float>(
                M, intermediate_size, expert_intermediate1 + task.token_offset * intermediate_size_x2, intermediate_size_x2,
                my_A2_qs_ptr, my_A2_d_ptr
            );
        }

        #pragma omp barrier

        // --- FFN Down-Projection (GEMM 2), split by N ---

        int n_chunk2 = (hidden_dim + T_exp - 1) / T_exp;
        n_chunk2 = ((n_chunk2 + NC - 1) / NC) * NC;
        int n_start2 = t_idx * n_chunk2;
        int n_size2 = std::min(n_chunk2, hidden_dim - n_start2);
        if (n_size2 > 0) {
            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                M, n_size2, intermediate_size, my_A2_qs_ptr, my_A2_d_ptr,
                down_proj_qs_ptr + n_start2 * intermediate_size,
                reinterpret_cast<const ggml_half*>(down_proj_d_ptr) + n_start2 * intermediate_dim_k_blocks,
                expert_intermediate2 + task.token_offset * hidden_dim + n_start2, hidden_dim
            );
        }
    } // end parallel region

    // --- Cleanup ---
    std::free(packed_A1_qs_all); std::free(packed_A1_d_all);
    std::free(expert_intermediate1);
    std::free(packed_A2_qs_all); std::free(packed_A2_d_all);
}

// =================================================================================================
// STRATEGY 2: TASK PARALLELISM (for many active experts)
// =================================================================================================
static void moe_q8_forward_task_parallel(
    int num_experts, const std::vector<int>& expert_counts, const std::vector<int>& expert_starts,
    const MoETokenInfo* token_map, int total_expert_tokens,
    int hidden_dim, int intermediate_size, int intermediate_size_x2,
    const int8_t* x_qs, const float* x_d,
    const int8_t* gate_up_qs_stacked_ptr, const at::Half* gate_up_d_stacked_ptr,
    const int8_t* down_proj_qs_stacked_ptr, const at::Half* down_proj_d_stacked_ptr,
    float* expert_intermediate2 // Output buffer
) { 
    constexpr int M_BLOCK = 32;
    struct MoeTask { int expert_id, num_tokens, global_token_start_pos; };
    std::vector<MoeTask> tasks;
    tasks.reserve(total_expert_tokens / M_BLOCK + num_experts);
    for (int exp_id = 0; exp_id < num_experts; ++exp_id) {
        const int count = expert_counts[exp_id];
        if (count == 0) continue;
        const int start_pos = expert_starts[exp_id];
        for (int offset = 0; offset < count; offset += M_BLOCK) {
            tasks.push_back({exp_id, std::min(M_BLOCK, count - offset), start_pos + offset});
        }
    }
    int task_count = static_cast<int>(tasks.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < task_count; ++i) {
        const auto& task = tasks[i];
        const int exp_id = task.expert_id, count = task.num_tokens, global_start_pos = task.global_token_start_pos;

        const int8_t* gate_up_qs_ptr = gate_up_qs_stacked_ptr + exp_id * intermediate_size_x2 * hidden_dim;
        const at::Half* gate_up_d_ptr = gate_up_d_stacked_ptr + exp_id * intermediate_size_x2 * (hidden_dim / QK8_0);
        const int8_t* down_proj_qs_ptr = down_proj_qs_stacked_ptr + exp_id * hidden_dim * intermediate_size;
        const at::Half* down_proj_d_ptr = down_proj_d_stacked_ptr + exp_id * hidden_dim * (intermediate_size / QK8_0);

        pack_A_q8_0_from_quantized_indirect<ExecutionPolicy::Sequential>(
            count, hidden_dim, x_qs, x_d, token_map, global_start_pos,
            ws.A_qs_packed1, ws.A_d_packed1
        );
        gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
            count, intermediate_size_x2, hidden_dim, ws.A_qs_packed1, ws.A_d_packed1,
            gate_up_qs_ptr, reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
            ws.expert_intermediate1, intermediate_size_x2
        );

        silu_and_mul<ExecutionPolicy::Sequential>(ws.expert_intermediate1, count, intermediate_size_x2);

        quantize_pack_A_q8_0<ExecutionPolicy::Sequential, float>(
            count, intermediate_size, ws.expert_intermediate1, intermediate_size_x2,
            ws.A_qs_packed2, ws.A_d_packed2
        );
        gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
            count, hidden_dim, intermediate_size, ws.A_qs_packed2, ws.A_d_packed2,
            down_proj_qs_ptr, reinterpret_cast<const ggml_half*>(down_proj_d_ptr),
            expert_intermediate2 + global_start_pos * hidden_dim, hidden_dim
        );
    }
}

template <typename T>
void moe_q8_forward_ptr_impl(
    T* x_ptr, // Input/Output
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_stacked_ptr,
    const at::Half* gate_up_d_stacked_ptr,
    const int8_t* down_proj_qs_stacked_ptr,
    const at::Half* down_proj_d_stacked_ptr,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size, int64_t intermediate_size_x2,
    int64_t top_k
) {
    // =======================================================================
    // 1. PREPROCESSING & GATHER/SCATTER MAP CREATION
    // =======================================================================
    const auto hidden_dim_k_blocks = hidden_dim / QK8_0;

    std::vector<int> expert_counts(num_experts, 0);
    std::vector<int> expert_starts(num_experts + 1, 0);
    std::vector<MoETokenInfo> token_map;
    std::vector<int32_t> scatter_map;

    preprocess_moe_routing(
        num_experts, top_k, num_tokens, selected_experts_ptr, // Use raw pointer
        expert_counts, expert_starts, token_map, scatter_map
    );

    const int total_expert_tokens = expert_starts[num_experts];
    if (total_expert_tokens == 0) {
        memset(x_ptr, 0, num_tokens * hidden_dim * sizeof(T)); // Zero out the buffer
        return;
    }


    // =======================================================================
    // 2. ALLOCATE BUFFERS
    // =======================================================================
    int8_t* x_qs = static_cast<int8_t*>(std::aligned_alloc(64, num_tokens * hidden_dim * sizeof(int8_t)));
    float* x_d = static_cast<float*>(std::aligned_alloc(64, num_tokens * hidden_dim_k_blocks * sizeof(float)));
    float* expert_intermediate2 = static_cast<float*>(std::aligned_alloc(64, total_expert_tokens * hidden_dim * sizeof(float)));
    constexpr int M_BLOCK = 32;
    static_assert(M_BLOCK % MR == 0, "M_BLOCK must be a multiple of MR");

    // =======================================================================
    // 3. QUANTIZE ACTIVATIONS (COMMON STEP)
    // =======================================================================
    #pragma omp parallel
    {
        // For task_parallel path, ws needs to be sized for M_BLOCK
        ws.ensure_size(M_BLOCK, hidden_dim, intermediate_size, intermediate_size_x2);
        #pragma omp for
        for (int64_t i = 0; i < num_tokens; ++i) {
            const T* src_row = x_ptr + i * hidden_dim; // Use raw pointer
            for (int j = 0; j < hidden_dim; ++j) {
                ws.temp_row_buffer[j] = static_cast<float>(src_row[j]);
            }
            for (int k_block = 0; k_block < hidden_dim_k_blocks; ++k_block) {
                quantize_block_q8_0(
                    ws.temp_row_buffer + k_block * QK8_0,
                    x_d + i * hidden_dim_k_blocks + k_block,
                    x_qs + i * hidden_dim + k_block * QK8_0
                );
            }
        }
    }

    // =======================================================================
    // 4. EXPERT COMPUTATION (DISPATCH TO PARALLELISM STRATEGY)
    // =======================================================================

    const int num_threads = omp_get_max_threads();

    std::vector<int> active_expert_ids;
    for(int i = 0; i < num_experts; ++i) {
        if (expert_counts[i] > 0) active_expert_ids.push_back(i);
    }
    const int active_experts_count = active_expert_ids.size();
    if (active_experts_count > 0 && active_experts_count * 2 <= num_threads) {
        moe_q8_forward_n_axis_parallel(
            num_experts, expert_counts, expert_starts, active_expert_ids,
            token_map.data(), total_expert_tokens, hidden_dim, intermediate_size, intermediate_size_x2,
            x_qs, x_d, 
            gate_up_qs_stacked_ptr, reinterpret_cast<const at::Half*>(gate_up_d_stacked_ptr), // Pass raw pointers
            down_proj_qs_stacked_ptr, reinterpret_cast<const at::Half*>(down_proj_d_stacked_ptr),
            expert_intermediate2
        );
    } else if (active_experts_count > 0) {
        moe_q8_forward_task_parallel(
            num_experts, expert_counts, expert_starts,
            token_map.data(), total_expert_tokens, hidden_dim, intermediate_size, intermediate_size_x2,
            x_qs, x_d, 
            gate_up_qs_stacked_ptr, reinterpret_cast<const at::Half*>(gate_up_d_stacked_ptr), // Pass raw pointers
            down_proj_qs_stacked_ptr, reinterpret_cast<const at::Half*>(down_proj_d_stacked_ptr),
            expert_intermediate2
        );
    }
    // =======================================================================
    // 5. SCATTER AND WEIGHTING (COMMON STEP)
    // =======================================================================
    T* final_output_ptr = x_ptr; // Use raw pointer

    #pragma omp parallel for
    for (int64_t t = 0; t < num_tokens; ++t) {
        std::vector<float> acc_buffer(hidden_dim, 0.0f);
        for (int k = 0; k < top_k; ++k) {
            const int scatter_idx = t * top_k + k;
            const int src_row_idx = scatter_map[scatter_idx];
            if (src_row_idx == -1) continue;
            const float weight = routing_weights_ptr[scatter_idx];
            const float* src_row = expert_intermediate2 + src_row_idx * hidden_dim;
            for (int j = 0; j < hidden_dim; ++j) {
                acc_buffer[j] += weight * src_row[j];
            }
        }
        T* dst_row = final_output_ptr + t * hidden_dim;
        for (int j = 0; j < hidden_dim; ++j) {
            dst_row[j] = static_cast<T>(acc_buffer[j]);
        }
    }
        // --- Cleanup ---
    std::free(x_qs);
    std::free(x_d);
    std::free(expert_intermediate2);
}

// =================================================================================================
// NUMA-AWARE TENSOR PARALLEL IMPLEMENTATION for MoE
// =================================================================================================
template <typename T>
void moe_q8_forward_ptr_numa_impl(
    T* x_ptr, // Input/Output, must be float for this implementation
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const std::vector<void*>& gate_up_qs_stacked_numa,
    const std::vector<void*>& gate_up_d_stacked_numa,
    const std::vector<void*>& down_proj_qs_stacked_numa,
    const std::vector<void*>& down_proj_d_stacked_numa,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size, // Full intermediate size
    int64_t top_k,
    int numa_nodes // Number of NUMA nodes used for partitioning
) {
    // =======================================================================
    // 1. PREPROCESSING & GATHER/SCATTER MAP CREATION
    // =======================================================================
    const int64_t hidden_dim_k_blocks = hidden_dim / QK8_0;
    const int64_t intermediate_size_per_node = intermediate_size / numa_nodes;
    const int64_t intermediate_size_x2_per_node = intermediate_size_per_node * 2;
    
    TORCH_CHECK(intermediate_size % numa_nodes == 0, "Intermediate size must be divisible by the number of NUMA nodes.");

    std::vector<int> expert_counts(num_experts, 0);
    std::vector<int> expert_starts(num_experts + 1, 0);
    std::vector<MoETokenInfo> token_map;
    std::vector<int32_t> scatter_map;

    preprocess_moe_routing(
        num_experts, top_k, num_tokens, selected_experts_ptr,
        expert_counts, expert_starts, token_map, scatter_map
    );

    const int total_expert_tokens = expert_starts[num_experts];
    if (total_expert_tokens == 0) {
        memset(x_ptr, 0, num_tokens * hidden_dim * sizeof(T));
        return;
    }

    // Create task list (same as before)
    constexpr int M_BLOCK = 32;
    struct MoeTask { int expert_id, num_tokens, global_token_start_pos; };
    std::vector<MoeTask> tasks;
    tasks.reserve(total_expert_tokens / M_BLOCK + num_experts);
    for (int exp_id = 0; exp_id < num_experts; ++exp_id) {
        const int count = expert_counts[exp_id];
        if (count == 0) continue;
        const int start_pos = expert_starts[exp_id];
        for (int offset = 0; offset < count; offset += M_BLOCK) {
            tasks.push_back({exp_id, std::min(M_BLOCK, count - offset), start_pos + offset});
        }
    }
    const int task_count = static_cast<int>(tasks.size());

    // =======================================================================
    // 2. ALLOCATE BUFFERS
    // =======================================================================
    int8_t* x_qs = static_cast<int8_t*>(std::aligned_alloc(64, num_tokens * hidden_dim * sizeof(int8_t)));
    float* x_d = static_cast<float*>(std::aligned_alloc(64, num_tokens * hidden_dim_k_blocks * sizeof(float)));

    // Allocate per-NUMA output buffers for partial results
    std::vector<float*> expert_intermediate2_numa(numa_nodes);
    for (int i = 0; i < numa_nodes; ++i) {
        size_t buffer_size = total_expert_tokens * hidden_dim * sizeof(float);
#ifdef WITH_NUMA
        if (numa_nodes > 1) {
            expert_intermediate2_numa[i] = static_cast<float*>(numa_alloc_onnode(buffer_size, i));
        } else {
            expert_intermediate2_numa[i] = static_cast<float*>(std::aligned_alloc(64, buffer_size));
        }
#else
        expert_intermediate2_numa[i] = static_cast<float*>(std::aligned_alloc(64, buffer_size));
#endif
        TORCH_CHECK(expert_intermediate2_numa[i] != nullptr, "Failed to allocate intermediate buffer on NUMA node ", i);
    }
    float* final_expert_intermediate = static_cast<float*>(std::aligned_alloc(64, total_expert_tokens * hidden_dim * sizeof(float)));
    
    const int omp_max_threads = omp_get_max_threads();

    TORCH_CHECK(omp_max_threads % numa_nodes == 0, "Number of OpenMP threads must be divisible by the number of NUMA nodes.");

    // =======================================================================
    // 3. MAIN PARALLEL REGION (Quantize -> Compute -> AllReduce -> Scatter)
    // =======================================================================
    #pragma omp parallel num_threads(omp_max_threads)
    {
        // Get thread-local workspace
        ws.ensure_size(M_BLOCK, hidden_dim, intermediate_size_per_node, intermediate_size_x2_per_node);

        // --- A. THREAD BINDING ---
        const int thread_id = omp_get_thread_num();
        const int node_id = thread_id % numa_nodes;
#ifdef WITH_NUMA
        if (numa_nodes > 1) {
            numa_run_on_node(node_id);
            numa_set_preferred(node_id);
        }
#endif

        // --- B. QUANTIZE ACTIVATIONS (Parallel) ---
        #pragma omp for
        for (int64_t i = 0; i < num_tokens; ++i) {
            const T* src_row = x_ptr + i * hidden_dim; // Use raw pointer
            for (int j = 0; j < hidden_dim; ++j) {
                ws.temp_row_buffer[j] = static_cast<float>(src_row[j]);
            }
            for (int k_block = 0; k_block < hidden_dim_k_blocks; ++k_block) {
                quantize_block_q8_0(
                    ws.temp_row_buffer + k_block * QK8_0,
                    x_d + i * hidden_dim_k_blocks + k_block,
                    x_qs + i * hidden_dim + k_block * QK8_0
                );
            }
        }

        // --- C. EXPERT COMPUTATION (Tensor Parallel over tasks) ---
        // Each thread processes tasks and writes its partial result to its local NUMA buffer.
        for (int i = thread_id / numa_nodes; i < task_count; i += omp_max_threads / numa_nodes) {
            const auto& task = tasks[i];
            const int exp_id = task.expert_id;
            const int count = task.num_tokens;
            const int global_start_pos = task.global_token_start_pos;

            // Pointers to the start of the correct expert's weights within the local NUMA buffer
            const auto* local_gate_up_qs = static_cast<const int8_t*>(gate_up_qs_stacked_numa[node_id]);
            const auto* local_gate_up_d = static_cast<const at::Half*>(gate_up_d_stacked_numa[node_id]);
            const auto* local_down_proj_qs = static_cast<const int8_t*>(down_proj_qs_stacked_numa[node_id]);
            const auto* local_down_proj_d = static_cast<const at::Half*>(down_proj_d_stacked_numa[node_id]);

            const int8_t* gate_up_qs_ptr = local_gate_up_qs + exp_id * intermediate_size_x2_per_node * hidden_dim;
            const at::Half* gate_up_d_ptr = local_gate_up_d + exp_id * intermediate_size_x2_per_node * (hidden_dim / QK8_0);
            const int8_t* down_proj_qs_ptr = local_down_proj_qs + exp_id * hidden_dim * intermediate_size_per_node;
            const at::Half* down_proj_d_ptr = local_down_proj_d + exp_id * hidden_dim * (intermediate_size_per_node / QK8_0);

            pack_A_q8_0_from_quantized_indirect<ExecutionPolicy::Sequential>(
                count, hidden_dim, x_qs, x_d, token_map.data(), global_start_pos,
                ws.A_qs_packed1, ws.A_d_packed1
            );

            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                count, intermediate_size_x2_per_node, hidden_dim, ws.A_qs_packed1, ws.A_d_packed1,
                gate_up_qs_ptr, reinterpret_cast<const ggml_half*>(gate_up_d_ptr),
                ws.expert_intermediate1, intermediate_size_x2_per_node
            );

            silu_and_mul<ExecutionPolicy::Sequential>(ws.expert_intermediate1, count, intermediate_size_x2_per_node);

            quantize_pack_A_q8_0<ExecutionPolicy::Sequential, float>(
                count, intermediate_size_per_node, ws.expert_intermediate1, intermediate_size_x2_per_node,
                ws.A_qs_packed2, ws.A_d_packed2
            );
            
            // Write partial result to the local NUMA output buffer
            gemm_q8_0_compute_packed<ExecutionPolicy::Sequential>(
                count, hidden_dim, intermediate_size_per_node, ws.A_qs_packed2, ws.A_d_packed2,
                down_proj_qs_ptr, reinterpret_cast<const ggml_half*>(down_proj_d_ptr),
                expert_intermediate2_numa[node_id] + global_start_pos * hidden_dim, hidden_dim
            );
        }

        // --- D. ALL-REDUCE STEP ---
        // Synchronize to ensure all expert computations are complete before reducing.
        #pragma omp barrier

        // Parallel reduction of per-NUMA results into the final buffer
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < total_expert_tokens * hidden_dim; ++i) {
            float sum = 0.0f;
            for (int n = 0; n < numa_nodes; ++n) {
                sum += expert_intermediate2_numa[n][i];
            }
            final_expert_intermediate[i] = sum;
        }

        // --- E. SCATTER AND WEIGHTING (Parallel) ---
        T* final_output_ptr = x_ptr; // Use raw pointer
        #pragma omp for
        for (int64_t t = 0; t < num_tokens; ++t) {
            std::vector<float> acc_buffer(hidden_dim, 0.0f);
            for (int k = 0; k < top_k; ++k) {
                const int scatter_idx = t * top_k + k;
                const int src_row_idx = scatter_map[scatter_idx];
                if (src_row_idx == -1) continue;
                const float weight = routing_weights_ptr[scatter_idx];
                const float* src_row = final_expert_intermediate + src_row_idx * hidden_dim;
                for (int j = 0; j < hidden_dim; ++j) {
                    acc_buffer[j] += weight * src_row[j];
                }
            }
            T* dst_row = final_output_ptr + t * hidden_dim;
            for (int j = 0; j < hidden_dim; ++j) {
                dst_row[j] = static_cast<T>(acc_buffer[j]);
            }
        }

    } // End of parallel region

    // =======================================================================
    // 5. CLEANUP
    // =======================================================================
    std::free(x_qs);
    std::free(x_d);
    std::free(final_expert_intermediate);
    for (int i = 0; i < numa_nodes; ++i) {
#ifdef WITH_NUMA
        if (numa_nodes > 1 && expert_intermediate2_numa[i]) {
            numa_free(expert_intermediate2_numa[i], total_expert_tokens * hidden_dim * sizeof(float));
        } else {
            std::free(expert_intermediate2_numa[i]);
        }
#else
        std::free(expert_intermediate2_numa[i]);
#endif
    }
}

template <typename T>
torch::Tensor moe_q8_forward_impl(
    torch::Tensor x,
    torch::Tensor routing_weights,
    torch::Tensor selected_experts,
    torch::Tensor gate_up_qs_stacked,
    torch::Tensor gate_up_d_stacked,
    torch::Tensor down_proj_qs_stacked,
    torch::Tensor down_proj_d_stacked
) {
    moe_q8_forward_ptr_impl<T>(
        x.data_ptr<T>(),
        routing_weights.data_ptr<float>(),
        selected_experts.data_ptr<int32_t>(),
        gate_up_qs_stacked.data_ptr<int8_t>(),
        reinterpret_cast<at::Half*>(gate_up_d_stacked.data_ptr()),
        down_proj_qs_stacked.data_ptr<int8_t>(),
        reinterpret_cast<at::Half*>(down_proj_d_stacked.data_ptr()),
        x.size(0),
        x.size(1),
        gate_up_qs_stacked.size(0),
        down_proj_qs_stacked.size(2),
        gate_up_qs_stacked.size(1),
        selected_experts.size(1)
    );
    return x;
}

// 调度器函数，Pybind将绑定到此函数
torch::Tensor moe_q8_forward(
    torch::Tensor x,
    torch::Tensor routing_weights,
    torch::Tensor selected_experts,
    torch::Tensor gate_up_qs_stacked,
    torch::Tensor gate_up_d_stacked,
    torch::Tensor down_proj_qs_stacked,
    torch::Tensor down_proj_d_stacked
) {
    // 检查输入和权重是否具有相同的dtype
    TORCH_CHECK(routing_weights.scalar_type() == torch::kFloat, 
                "routing_weights must be float32");
    torch::Tensor result;

    // 根据输入类型调用相应的模板实例
    if (x.scalar_type() == torch::kFloat) {
        result = moe_q8_forward_impl<float>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else if (x.scalar_type() == torch::kBFloat16) {
        result = moe_q8_forward_impl<at::BFloat16>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else if (x.scalar_type() == torch::kHalf) {
        result = moe_q8_forward_impl<at::Half>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else {
        TORCH_CHECK(false, "Unsupported input dtype for moe_q8_forward. Supported dtypes are float32, bfloat16, and float16.");
    }
    return result;
}

inline float fast_exp(float x)
{
    const float LOG2E = 1.4426950409f; 
    const float LN2 = 0.6931471806f;
    x = std::max(-87.3f, x);
    float k_float = floorf(x * LOG2E + 0.5f);
    int k = (int)k_float;
    float r = x - k_float * LN2;
    const float c0 = 0.99995040f;
    const float c1 = 1.00015572f;
    const float c2 = 0.50426726f;
    const float c3 = 0.16522747f;
    float exp_r = c0 + r * (c1 + r * (c2 + r * c3));
    union {uint32_t i; float f;} v;
    v.i = (uint32_t)(k + 127) << 23;
    return v.f * exp_r;
}

template <typename T>
void gating_top_k_softmax_ptr_impl(
    const T* logits_ptr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t top_k,
    bool normalize,
    float* routing_weights_out_ptr, // Output pointer
    int32_t* selected_experts_out_ptr  // Output pointer
) {
    #pragma omp parallel
    {
        // --- Thread-local buffers ---
        std::vector<int32_t> indices(num_experts);
        constexpr int MAX_TOP_K = 8;
        float exp_vals[MAX_TOP_K];

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_tokens; ++i) {
            const T* current_logits = logits_ptr + i * num_experts;
            
            // 1. Initialize and partially sort indices to find top_k experts
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(
                indices.begin(),
                indices.begin() + top_k,
                indices.end(),
                [&](int32_t a, int32_t b) {
                    return current_logits[a] > current_logits[b];
                }
            );

            // 2. Store selected expert indices
            int32_t* current_experts = selected_experts_out_ptr + i * top_k;
            memcpy(current_experts, indices.data(), top_k * sizeof(int32_t));

            // 3. Compute Softmax weights (always in float32)
            float* current_weights = routing_weights_out_ptr + i * top_k;

            if (normalize) {
                // --- Path A: Normalize over TOP-K logits only ---
                float max_logit = static_cast<float>(current_logits[indices[0]]);
                float sum_exp = 0.0f;
                for (int k = 0; k < top_k; ++k) {
                    exp_vals[k] = fast_exp(static_cast<float>(current_logits[indices[k]]) - max_logit);
                    sum_exp += exp_vals[k];
                }
                const float inv_sum_exp = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int k = 0; k < top_k; ++k) {
                    current_weights[k] = exp_vals[k] * inv_sum_exp;
                }
            } else {
                // --- Path B: Standard Softmax over all logits ---
                float max_logit = static_cast<float>(current_logits[indices[0]]);
                float sum_exp = 0.0f;
                for (int j = 0; j < num_experts; ++j) {
                    sum_exp += fast_exp(static_cast<float>(current_logits[j]) - max_logit);
                }
                const float inv_sum_exp = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                for (int k = 0; k < top_k; ++k) {
                    float val = fast_exp(static_cast<float>(current_logits[indices[k]]) - max_logit);
                    current_weights[k] = val * inv_sum_exp;
                }
            }
        }
    } // End parallel region
}

template <typename T>
void gating_top_k_softmax_impl(
    const torch::Tensor& logits,
    int64_t top_k,
    bool normalize,
    torch::Tensor& routing_weights_out,
    torch::Tensor& selected_experts_out
) {
    gating_top_k_softmax_ptr_impl<T>(
        logits.data_ptr<T>(),
        logits.size(0),
        logits.size(1),
        top_k,
        normalize,
        routing_weights_out.data_ptr<float>(),
        selected_experts_out.data_ptr<int32_t>()
    );
}

std::vector<torch::Tensor> gating_top_k_softmax(
    const torch::Tensor& logits,
    int top_k,
    bool normalize
) {
    TORCH_CHECK(logits.dim() == 2, "Logits must be 2D");
    TORCH_CHECK(logits.is_contiguous(), "Logits must be contiguous");
    TORCH_CHECK(logits.device().is_cpu(), "Logits must be on CPU");
    TORCH_CHECK(top_k > 0 && top_k <= logits.size(1) && top_k <= 8, "top_k is out of range");

    auto routing_weights_out = torch::empty({logits.size(0), top_k}, logits.options().dtype(torch::kFloat));
    auto selected_experts_out = torch::empty({logits.size(0), top_k}, torch::kInt32);

    if (logits.scalar_type() == torch::kHalf) {
        gating_top_k_softmax_impl<at::Half>(logits, top_k, normalize, routing_weights_out, selected_experts_out);
    } else if (logits.scalar_type() == torch::kBFloat16) {
        gating_top_k_softmax_impl<at::BFloat16>(logits, top_k, normalize, routing_weights_out, selected_experts_out);
    } else if (logits.scalar_type() == torch::kFloat) {
        gating_top_k_softmax_impl<float>(logits, top_k, normalize, routing_weights_out, selected_experts_out);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for gating_top_k_softmax. Supported: float16, bfloat16, float32.");
    }

    return {routing_weights_out, selected_experts_out};
}

template void gating_top_k_softmax_ptr_impl<at::Half>(
    const at::Half* logits_ptr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t top_k,
    bool normalize,
    float* routing_weights_out_ptr,
    int32_t* selected_experts_out_ptr
);

template void gating_top_k_softmax_ptr_impl<at::BFloat16>(
    const at::BFloat16* logits_ptr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t top_k,
    bool normalize,
    float* routing_weights_out_ptr,
    int32_t* selected_experts_out_ptr
);

template void moe_q8_forward_ptr_impl<at::Half>(
    at::Half* x_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_stacked_ptr,
    const at::Half* gate_up_d_stacked_ptr,
    const int8_t* down_proj_qs_stacked_ptr,
    const at::Half* down_proj_d_stacked_ptr,
    int64_t num_tokens,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t intermediate_size,
    int64_t intermediate_size_x2,
    int64_t top_k
);

template void moe_q8_forward_ptr_impl<at::BFloat16>(
    at::BFloat16* x_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const int8_t* gate_up_qs_stacked_ptr,
    const at::Half* gate_up_d_stacked_ptr,
    const int8_t* down_proj_qs_stacked_ptr,
    const at::Half* down_proj_d_stacked_ptr,
    int64_t num_tokens,
    int64_t hidden_dim,
    int64_t num_experts,
    int64_t intermediate_size,
    int64_t intermediate_size_x2,
    int64_t top_k
);

template void moe_q8_forward_ptr_numa_impl<at::Half>(
    at::Half* x_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const std::vector<void*>& gate_up_qs_stacked_numa,
    const std::vector<void*>& gate_up_d_stacked_numa,
    const std::vector<void*>& down_proj_qs_stacked_numa,
    const std::vector<void*>& down_proj_d_stacked_numa,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t top_k,
    int numa_nodes);


template void moe_q8_forward_ptr_numa_impl<at::BFloat16>(
    at::BFloat16* x_ptr,
    const float* routing_weights_ptr,
    const int32_t* selected_experts_ptr,
    const std::vector<void*>& gate_up_qs_stacked_numa,
    const std::vector<void*>& gate_up_d_stacked_numa,
    const std::vector<void*>& down_proj_qs_stacked_numa,
    const std::vector<void*>& down_proj_d_stacked_numa,
    int64_t num_tokens, int64_t hidden_dim, int64_t num_experts,
    int64_t intermediate_size,
    int64_t top_k,
    int numa_nodes);

#ifdef WITH_CUDA
#include <cuda_runtime.h> // For cudaLaunchHostFunc

struct MoECpuTaskArgs {
    void* hidden_states_ptr;
    const void* router_logits_ptr;
    int64_t num_tokens;
    int64_t top_k;
    bool normalize_prob;
    MoEInfer* moe_infer_ptr;
    bool keep_args;
    at::ScalarType dtype;
};

// The callback function remains simple.
void CUDART_CB host_fn_callback(void* user_data) {
    auto* args = static_cast<MoECpuTaskArgs*>(user_data);
    
    args->moe_infer_ptr->execute_on_cpu_from_pointers(
        args->hidden_states_ptr,
        args->router_logits_ptr,
        args->num_tokens,
        args->top_k,
        args->normalize_prob,
        args->dtype
    );

    if (!args->keep_args) {
        delete args;
    }
}

// The launch function is where we interact with PyTorch tensors and extract raw data.
// This function MUST be called while holding the GIL.
void launch_moe_cpu_task(
    torch::Tensor& hidden_states_pinned,
    const torch::Tensor& router_logits_pinned,
    py::capsule& moe_infer_handle,
    int64_t top_k,
    bool normalize_prob,
    uint64_t stream_ptr,
    bool keep_args
) {
    TORCH_CHECK(hidden_states_pinned.is_pinned(), "hidden_states must be a pinned tensor");
    TORCH_CHECK(router_logits_pinned.is_pinned(), "router_logits must be a pinned tensor");
    TORCH_CHECK(hidden_states_pinned.scalar_type() == router_logits_pinned.scalar_type(), "Dtype mismatch between hidden_states and router_logits");

    auto* args = new MoECpuTaskArgs{
        hidden_states_pinned.data_ptr(),
        router_logits_pinned.data_ptr(),
        hidden_states_pinned.size(0), // num_tokens
        top_k,
        normalize_prob,
        moe_infer_handle.get_pointer<MoEInfer>(),
        keep_args,
        hidden_states_pinned.scalar_type()
    };
    
    cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream_ptr), host_fn_callback, args);
}
#endif // WITH_CUDA

// ==========================================================================================
// PYBIND11 MODULE DEFINITION - NOW CORRECT
// ==========================================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // These functions are for standard CPU execution and testing. They are unchanged.
    m.def("quantize_weight_only", &quantize_weight_only, "Quantize weight for q8_gemm");
    m.def("q8_gemm", &q8_gemm, "q8_gemm kernel (A_fp32 @ B_q8.T)");
    m.def("moe_q8_forward", &moe_q8_forward, "Full MoE expert forward pass with int8 GEMM on CPU for float, bfloat16, and float16 inputs");
    m.def("gating_top_k_softmax", &gating_top_k_softmax, "Perform Top-K and Softmax on CPU for MoE gating, returning new tensors");
    // Add the new CUDA-integrated function, protected by the preprocessor guard.
    m.def("create_moe_infer_handle", [](int64_t num_experts, int64_t hidden_size, int64_t intermediate_size) {
        auto* ptr = new MoEInfer(num_experts, hidden_size, intermediate_size);
        return py::capsule(ptr, [](void* p) { delete reinterpret_cast<MoEInfer*>(p); });
    });
    m.def("moe_infer_quantize_and_store", [](py::capsule& handle, int64_t expert_idx, const std::string& proj_name, const torch::Tensor& weight) {
        handle.get_pointer<MoEInfer>()->quantize_and_store_expert(expert_idx, proj_name, weight);
    });
    m.def("moe_infer_store_quantized", [](py::capsule& handle, const torch::Tensor& gate_up_qs, const torch::Tensor& gate_up_d, const torch::Tensor& down_proj_qs, const torch::Tensor& down_proj_d) {
        handle.get_pointer<MoEInfer>()->store_quantized_weights(gate_up_qs, gate_up_d, down_proj_qs, down_proj_d);
    });
#ifdef WITH_CUDA
    m.def("launch_moe_cpu_task", &launch_moe_cpu_task, "Launches the GIL-free MoE CPU task via cudaLaunchHostFunc.");
#endif
}
