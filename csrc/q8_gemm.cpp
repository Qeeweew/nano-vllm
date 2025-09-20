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
#include <ATen/cpu/vec/vec.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#define QK8_0 32

// FP16 related definitions from gemm_q8_0.h
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    using ggml_half = __fp16;
    #define GGML_CPU_FP32_TO_FP16(x) ((__fp16)(x))
    #define GGML_CPU_FP16_TO_FP32(x) ((float)(x))
#elif defined(__F16C__)
    using ggml_half = uint16_t;
    #define GGML_CPU_FP32_TO_FP16(x) _cvtss_sh(x, 0)
    #define GGML_CPU_FP16_TO_FP32(x) _cvtsh_ss(x)
#endif

typedef struct {
    ggml_half d;
    int8_t  qs[QK8_0];
} block_q8_0;

#if defined(__ARM_NEON)
#define MR 8
#define NR 4
#elif defined(__AVX2__)
#define MR 8
#define NR 8
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
    if constexpr(std::is_same_v<D_TYPE, float>) {
        *y_d = d;
    } else {
        *y_d = GGML_CPU_FP32_TO_FP16(d);
    }
}

static void quantize_row_q8_0(const float * x, void * vy, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;
    block_q8_0 * y = (block_q8_0 *)vy;
    for (int i = 0; i < nb; i++) {
        quantize_block_q8_0(x + i * QK8_0, &y[i].d, y[i].qs);
    }
}

static void pack_B_q8_0(
    int N, int K, const block_q8_0* B_q, int ldb_q,
    int8_t* B_qs_packed, ggml_half* B_d_packed)
{
   const int K_BLOCKS = K / QK8_0;
    for (int j = 0; j < N; j += NR) {
        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int col = 0; col < NR; ++col) {
                if (j + col < N) {
                    *B_d_packed++ = (B_q + (j + col) * ldb_q + k_block)->d;
                }
            }
        }

        for(int k_block = 0; k_block < K_BLOCKS; ++k_block) {
            for (int k_rem = 0; k_rem < QK8_0; k_rem += 4) {
                for (int col = 0; col < NR; ++col) {
                    if (j + col < N) {
                        memcpy(B_qs_packed, (B_q + (j + col) * ldb_q + k_block)->qs + k_rem, 4);
                    }
                    B_qs_packed += 4;
                }
            }
        }
    }
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

#if defined(__AVXVNNI__)
        __m256i sum_a_vec = _mm256_setzero_si256();

        for (int k = 0; k < QK8_0; k += 4) {
            __m256i b_vec = _mm256_load_si256((__m256i const*)b_ptr);
            b_vec = _mm256_sub_epi8(b_vec, _mm256_set1_epi8(-128)); // b_vec + 128
            b_ptr += NR * 4;
            __m256i a_vec = _mm256_load_si256((__m256i const*)a_ptr);
            sum_a_vec = _mm256_dpbusd_avx_epi32(sum_a_vec, _mm256_set1_epi8(1), a_vec);

            for (int i = 0; i < MR_T; ++i) {
                __m256i a_vec = _mm256_set1_epi32(*((int*)(a_ptr + i * 4)));
                sum[i] = _mm256_dpbusd_avx_epi32(sum[i], b_vec, a_vec);
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

template <typename T>
static void quantize_pack_A_parallel_q8_0(
    int M, int K, const T* A, int lda,
    int8_t* A_qs_packed, float* A_d_packed)
{
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;
    
    at::parallel_for(0, num_a_packs, 0, [&](int64_t start, int64_t end) {
        float a_float_buf[MR * QK8_0];
        int8_t a_qs_buf[MR * QK8_0];

        for (int64_t i_pack = start; i_pack < end; ++i_pack) {
            int i = i_pack * MR;
            for (int j = 0; j < K_BLOCKS; ++j) {
                int M_rem = std::min(MR, M - i);

                // Convert input type to float before quantization
                for (int row = 0; row < M_rem; ++row) {
                    const T* src_row = A + (i + row) * lda + j * QK8_0;
                    float* dst_row = a_float_buf + row * QK8_0;
                    for (int k = 0; k < QK8_0; ++k) {
                        dst_row[k] = static_cast<float>(src_row[k]);
                    }
                }

                float* current_A_d_ptr = A_d_packed + i * K_BLOCKS + j * MR;
                for (int row = 0; row < M_rem; ++row) {
                    quantize_block_q8_0(a_float_buf + row * QK8_0, &current_A_d_ptr[row], &a_qs_buf[row * QK8_0]);
                }

                int8_t* current_qs_ptr = A_qs_packed + i * K + j * QK8_0 * MR;
                for (int k = 0; k < QK8_0; k += 4) {
                    for (int row = 0; row < MR; ++row) {
                         if (row < M_rem) {
                            memcpy(current_qs_ptr, &a_qs_buf[row * QK8_0 + k], 4);
                         } else {
                            // Padding for simplicity in kernel
                            memset(current_qs_ptr, 0, 4);
                         }
                        current_qs_ptr += 4;
                    }
                }
            }
        }
    });
}

// ==========================================================================================
// NEW: Function to perform GEMM computation with pre-packed matrices
// ==========================================================================================
static void gemm_q8_0_compute_packed(
    int M, int N, int K,
    const int8_t* A_qs_packed, const float* A_d_packed,
    const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16,
    float* C, int ldc)
{
    const int K_BLOCKS = K / QK8_0;
    constexpr int MC = 32; constexpr int KC = 1024; constexpr int NC = 32;
    const int num_nc_blocks = (N + NC - 1) / NC;
    
    at::parallel_for(0, num_nc_blocks, 0, [&](int64_t start, int64_t end) {
        for (int64_t jc_idx = start; jc_idx < end; ++jc_idx) {
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
        }
    });
}

void gemm_q8_0_aten_parallel_packed(int M, int N, int K, const float* A, int lda, const int8_t* B_qs_packed, const ggml_half* B_d_packed_f16, float* C) {
    assert(K % QK8_0 == 0);
    const int K_BLOCKS = K / QK8_0;
    const int M_CEIL = (M + MR - 1) / MR * MR;
    
    int8_t* A_qs_packed = static_cast<int8_t*>(std::aligned_alloc(64, M_CEIL * K * sizeof(int8_t)));
    float* A_d_packed = static_cast<float*>(std::aligned_alloc(64, M_CEIL * K_BLOCKS * sizeof(float)));

    // Phase 1: Quantize and Pack A
    quantize_pack_A_parallel_q8_0<float>(M, K, A, lda, A_qs_packed, A_d_packed);

    // Phase 2: GEMM Computation
    gemm_q8_0_compute_packed(M, N, K, A_qs_packed, A_d_packed, B_qs_packed, B_d_packed_f16, C, N);

    std::free(A_qs_packed);
    std::free(A_d_packed);
}


template <typename T>
static void quantize_rows_parallel_q8_0(
    int M, int K, const T* A, int lda,
    int8_t* A_qs, float* A_d)
{
    const int K_BLOCKS = K / QK8_0;
    at::parallel_for(0, M, 0, [&](int64_t start, int64_t end) {
        std::vector<float> temp_row(K);
        for(int64_t i = start; i < end; ++i) {
            const T* src_row = A + i * lda;
            // Convert input row to float for quantization
            for(int j = 0; j < K; ++j) {
                temp_row[j] = static_cast<float>(src_row[j]);
            }
            // Quantize each block in the row
            for (int k_block = 0; k_block < K_BLOCKS; ++k_block) {
                quantize_block_q8_0(
                    temp_row.data() + k_block * QK8_0,
                    A_d + i * K_BLOCKS + k_block,
                    A_qs + i * K + k_block * QK8_0
                );
            }
        }
    });
}

// ==========================================================================================
// PYTORCH BINDINGS
// ==========================================================================================

std::vector<torch::Tensor> quantize_repack_weight(torch::Tensor B_float) {
    TORCH_CHECK(B_float.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(B_float.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(B_float.device().is_cpu(), "Weight must be on CPU");
    
    const auto N = B_float.size(0);
    const auto K = B_float.size(1);
    // printf("N = %d K = %d\n", N, K);
    
    TORCH_CHECK(K % QK8_0 == 0, "Weight's K dimension must be a multiple of ", QK8_0);

    const int K_BLOCKS = K / QK8_0;
    std::vector<block_q8_0> B_q(N * K_BLOCKS);
    
    at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
        for (int64_t j = start; j < end; ++j) {
            quantize_row_q8_0(B_float.data_ptr<float>() + j * K, B_q.data() + j * K_BLOCKS, K);
        }
    });

    auto B_qs_packed_tensor = torch::empty({N, K}, torch::kInt8);
    auto B_d_packed_tensor = torch::empty({N, K_BLOCKS}, torch::kHalf);
    
    int8_t* B_qs_packed_ptr = B_qs_packed_tensor.data_ptr<int8_t>();
    ggml_half* B_d_packed_ptr = reinterpret_cast<ggml_half*>(B_d_packed_tensor.data_ptr<at::Half>());

    pack_B_q8_0(N, K, B_q.data(), K_BLOCKS, B_qs_packed_ptr, B_d_packed_ptr);

    return {B_qs_packed_tensor, B_d_packed_tensor};
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


// A specialized SiLU activation function for MoE.
// x = gate_proj(x), y = up_proj(x)
// return silu(x) * y
using Vec = at::vec::Vectorized<float>;

static void silu_and_mul(float* C, int num_tokens, int intermediate_size) {
    const int size_div_2 = intermediate_size / 2;
    at::parallel_for(0, num_tokens, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            float* row = C + i * intermediate_size;
            float* gate_part = row;
            float* up_part = row + size_div_2;
            
            int j = 0;
            // 1. Vectorized 主循环
            // 每次处理 Vec::size() 个 float 元素
            for (; j + Vec::size() <= size_div_2; j += Vec::size()) {
                // 从内存加载数据到向量寄存器
                Vec gate_vec = Vec::loadu(gate_part + j);
                Vec up_vec = Vec::loadu(up_part + j);

                const Vec one_vec(1.0f);
                Vec activation_vec = gate_vec / (one_vec + gate_vec.neg().exp());
                Vec result_vec = up_vec * activation_vec;

                result_vec.store(row + j);
            }

            // 2. Scalar 收尾循环
            // 处理剩余不足一个向量长度的元素
            for (; j < size_div_2; ++j) {
                float gate_val = gate_part[j];
                // SiLU (Swish) 激活函数: x * sigmoid(x)
                float activation = gate_val / (1.0f + expf(-gate_val));
                row[j] = up_part[j] * activation;
            }
        }
    });
}

static void pack_A_q8_0_from_quantized_indirect(
    int M, int K,
    const int8_t* x_qs_base,           // 原始量化数据（未 gather）
    const float* x_d_base,             // 原始 scale 数据
    const MoETokenInfo* token_map,     // 间接索引映射表
    int token_offset,                  // 当前专家的起始 token 在 token_map 中的偏移
    int8_t* A_qs_packed,
    float* A_d_packed
) {
    const int K_BLOCKS = K / QK8_0;
    const int num_a_packs = (M + MR - 1) / MR;

    at::parallel_for(0, num_a_packs, 0, [&](int64_t start, int64_t end) {
        for (int64_t i_pack = start; i_pack < end; ++i_pack) {
            int row_in_expert = i_pack * MR;  // 当前 pack 在专家局部行号
            for (int j = 0; j < K_BLOCKS; ++j) {
                int M_rem = std::min(MR, M - row_in_expert);
                float* current_A_d_ptr = A_d_packed + row_in_expert * K_BLOCKS + j * MR;

                // 逐行复制 d 值：从原始 x_d_base + token_map[token_offset + row].token_id
                for (int local_row = 0; local_row < M_rem; ++local_row) {
                    int global_token_id = token_map[token_offset + row_in_expert + local_row].token_id;
                    current_A_d_ptr[local_row] = x_d_base[global_token_id * K_BLOCKS + j];
                }

                // Pack qs 值
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
        }
    });
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
    const auto num_tokens = x.size(0);
    const auto hidden_dim = x.size(1);
    const auto num_experts = gate_up_qs_stacked.size(0);
    const auto intermediate_size_x2 = gate_up_qs_stacked.size(1);
    const auto intermediate_size = down_proj_qs_stacked.size(2);
    const auto top_k = selected_experts.size(1);
    const auto hidden_dim_k_blocks = hidden_dim / QK8_0;
    const auto intermediate_dim_k_blocks = intermediate_size / QK8_0;

    std::vector<int> expert_counts(num_experts, 0);
    std::vector<int> expert_starts(num_experts + 1, 0);
    std::vector<MoETokenInfo> token_map;
    std::vector<int32_t> scatter_map;
   
    torch::cuda::synchronize();

    preprocess_moe_routing(
        num_experts, top_k, num_tokens, selected_experts.data_ptr<int32_t>(),
        expert_counts, expert_starts, token_map, scatter_map
    );

    const int total_expert_tokens = expert_starts[num_experts];
    
    std::vector<int8_t> x_qs(num_tokens * hidden_dim);
    std::vector<float> x_d(num_tokens * hidden_dim_k_blocks);
    quantize_rows_parallel_q8_0<T>(
        num_tokens, hidden_dim, x.data_ptr<T>(), hidden_dim,
        x_qs.data(), x_d.data()
    );

    const int max_expert_tokens = total_expert_tokens > 0 ? *std::max_element(expert_counts.begin(), expert_counts.end()) : 0;
    const int M_CEIL_GEMM = (max_expert_tokens + MR - 1) / MR * MR;

    int8_t* A_qs_packed1 = static_cast<int8_t*>(std::aligned_alloc(64, M_CEIL_GEMM * hidden_dim * sizeof(int8_t)));
    float* A_d_packed1 = static_cast<float*>(std::aligned_alloc(64, M_CEIL_GEMM * hidden_dim_k_blocks * sizeof(float)));
    int8_t* A_qs_packed2 = static_cast<int8_t*>(std::aligned_alloc(64, M_CEIL_GEMM * intermediate_size * sizeof(int8_t)));
    float* A_d_packed2 = static_cast<float*>(std::aligned_alloc(64, M_CEIL_GEMM * intermediate_dim_k_blocks * sizeof(float)));

    float* expert_intermediate1 = static_cast<float*>(std::aligned_alloc(64, max_expert_tokens * intermediate_size_x2 * sizeof(float)));
    float* expert_intermediate2 = static_cast<float*>(std::aligned_alloc(64, total_expert_tokens * hidden_dim * sizeof(float)));

    for (int64_t exp_id = 0; exp_id < num_experts; ++exp_id) {
        int count = expert_counts[exp_id];
        if (count == 0) continue;
        int start_pos = expert_starts[exp_id];

        pack_A_q8_0_from_quantized_indirect(
            count, hidden_dim,
            x_qs.data(),           // 原始 qs
            x_d.data(),            // 原始 d
            token_map.data(),      // token 映射表
            start_pos,             // 当前专家在 token_map 中的起始偏移
            A_qs_packed1,
            A_d_packed1
        );

        // --- GEMM 1: gate_up_proj ---
        gemm_q8_0_compute_packed(
            count, intermediate_size_x2, hidden_dim,
            A_qs_packed1, A_d_packed1,
            gate_up_qs_stacked[exp_id].data_ptr<int8_t>(),
            reinterpret_cast<const ggml_half*>(gate_up_d_stacked[exp_id].data_ptr<at::Half>()),
            expert_intermediate1,
            intermediate_size_x2
        );

        // Activation
        silu_and_mul(expert_intermediate1, count, intermediate_size_x2);

        // --- GEMM 2: down_proj ---
        quantize_pack_A_parallel_q8_0<float>(
            count, intermediate_size,
            expert_intermediate1, intermediate_size_x2,
            A_qs_packed2, A_d_packed2
        );
        gemm_q8_0_compute_packed(
            count, hidden_dim, intermediate_size,
            A_qs_packed2, A_d_packed2,
            down_proj_qs_stacked[exp_id].data_ptr<int8_t>(),
            reinterpret_cast<const ggml_half*>(down_proj_d_stacked[exp_id].data_ptr<at::Half>()),
            expert_intermediate2 + start_pos * hidden_dim,
            hidden_dim
        );
    }

    T* final_output_ptr = x.data_ptr<T>();
    const T* routing_weights_ptr = routing_weights.data_ptr<T>();

    // Token-centric Parallel Aggregation (Scatter)
    at::parallel_for(0, num_tokens, 0, [&](int64_t start, int64_t end) {
        std::vector<float> acc_buffer(hidden_dim, 0.0f);
        for (int64_t t = start; t < end; ++t) {
            std::fill(acc_buffer.begin(), acc_buffer.end(), 0.0f);
            for (int k = 0; k < top_k; ++k) {
                const int scatter_idx = t * top_k + k;
                const int src_row_idx = scatter_map[scatter_idx];
                if (src_row_idx == -1) continue;

                const float weight = static_cast<float>(routing_weights_ptr[scatter_idx]);
                const float* src_row = expert_intermediate2 + src_row_idx * hidden_dim;

                for (int j = 0; j < hidden_dim; ++j) {
                    acc_buffer[j] += weight * src_row[j];
                }
            }
            T* dst_row = final_output_ptr + t * hidden_dim;
            const float* src_row_acc = acc_buffer.data();
            for (int j = 0; j < hidden_dim; ++j) {
                dst_row[j] = static_cast<T>(src_row_acc[j]);
            }
        }
    });

    std::free(A_qs_packed1);
    std::free(A_d_packed1);
    std::free(A_qs_packed2);
    std::free(A_d_packed2);
    std::free(expert_intermediate1);
    std::free(expert_intermediate2);
    
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
    TORCH_CHECK(x.scalar_type() == routing_weights.scalar_type(), 
                "x and routing_weights must have the same dtype");
    
    // 根据输入类型调用相应的模板实例
    if (x.scalar_type() == torch::kFloat) {
        return moe_q8_forward_impl<float>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else if (x.scalar_type() == torch::kBFloat16) {
        return moe_q8_forward_impl<at::BFloat16>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else if (x.scalar_type() == torch::kHalf) {
        return moe_q8_forward_impl<at::Half>(
            x, routing_weights, selected_experts, gate_up_qs_stacked,
            gate_up_d_stacked, down_proj_qs_stacked, down_proj_d_stacked);
    } else {
        TORCH_CHECK(false, "Unsupported input dtype for moe_q8_forward. Supported dtypes are float32, bfloat16, and float16.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_repack_weight", &quantize_repack_weight, "Quantize and repack weight for q8_gemm");
    m.def("q8_gemm", &q8_gemm, "q8_gemm kernel (A_fp32 @ B_q8.T)");
    m.def("moe_q8_forward", &moe_q8_forward, "Full MoE expert forward pass with int8 GEMM on CPU for float, bfloat16, and float16 inputs");
}
