// ---------------------------------------------------------------------------
// PTOAS compatibility layer: minimal FP8/FP4 + __VEC_SCOPE__ fallbacks so the
// pto-isa headers compile across AICore arch/toolchain combinations.
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ AICORE void attn_swpipe(__gm__ int64_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, __gm__ bfloat16_t* v4, __gm__ int32_t* v5, __gm__ float* v6, __gm__ int32_t* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ float* v12, __gm__ float* v13, __gm__ float* v14, __gm__ int32_t* v15, __gm__ float* v16, __gm__ bfloat16_t* v17, __gm__ float* v18, __gm__ bfloat16_t* v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int32_t v28, int32_t v29);

void LaunchAttn_swpipe(int64_t *v1, uint16_t *v2, uint16_t *v3, uint16_t *v4, int32_t *v5, float *v6, int32_t *v7, float *v8, float *v9, float *v10, float *v11, float *v12, float *v13, float *v14, int32_t *v15, float *v16, uint16_t *v17, float *v18, uint16_t *v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int32_t v28, int32_t v29, void *stream) {
    attn_swpipe<<<1, nullptr, stream>>>((__gm__ int64_t*)v1, (__gm__ bfloat16_t*)v2, (__gm__ bfloat16_t*)v3, (__gm__ bfloat16_t*)v4, (__gm__ int32_t*)v5, (__gm__ float*)v6, (__gm__ int32_t*)v7, (__gm__ float*)v8, (__gm__ float*)v9, (__gm__ float*)v10, (__gm__ float*)v11, (__gm__ float*)v12, (__gm__ float*)v13, (__gm__ float*)v14, (__gm__ int32_t*)v15, (__gm__ float*)v16, (__gm__ bfloat16_t*)v17, (__gm__ float*)v18, (__gm__ bfloat16_t*)v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29);
}
