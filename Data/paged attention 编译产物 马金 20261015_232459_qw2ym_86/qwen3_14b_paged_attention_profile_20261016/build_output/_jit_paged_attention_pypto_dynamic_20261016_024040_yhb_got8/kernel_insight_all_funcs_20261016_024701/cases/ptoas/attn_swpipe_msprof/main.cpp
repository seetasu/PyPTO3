#include "test_common.h"
#include "acl/acl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
    do {                                                                                         \
        const aclError _ret = (expr);                                                            \
        if (_ret != ACL_SUCCESS) {                                                               \
            std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); \
            const char *_recent = aclGetRecentErrMsg();                                          \
            if (_recent != nullptr && _recent[0] != '\0') {                                      \
                std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                     \
            }                                                                                    \
            rc = 1;                                                                              \
            goto cleanup;                                                                        \
        }                                                                                        \
    } while (0)

void LaunchAttn_swpipe(int64_t *v1, uint16_t *v2, uint16_t *v3, uint16_t *v4, int32_t *v5, float *v6, int32_t *v7, float *v8, float *v9, float *v10, float *v11, float *v12, float *v13, float *v14, int32_t *v15, float *v16, uint16_t *v17, float *v18, uint16_t *v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int32_t v28, int32_t v29, void *stream);

int main() {
    size_t elemCount_v1 = 1048576;
    size_t fileSize_v1 = elemCount_v1 * sizeof(int64_t);
    int64_t *v1Host = nullptr;
    int64_t *v1Device = nullptr;
    size_t elemCount_v2 = 262144;
    size_t fileSize_v2 = elemCount_v2 * sizeof(uint16_t);
    uint16_t *v2Host = nullptr;
    uint16_t *v2Device = nullptr;
    size_t elemCount_v3 = 32768;
    size_t fileSize_v3 = elemCount_v3 * sizeof(uint16_t);
    uint16_t *v3Host = nullptr;
    uint16_t *v3Device = nullptr;
    size_t elemCount_v4 = 262144;
    size_t fileSize_v4 = elemCount_v4 * sizeof(uint16_t);
    uint16_t *v4Host = nullptr;
    uint16_t *v4Device = nullptr;
    size_t elemCount_v5 = 2;
    size_t fileSize_v5 = elemCount_v5 * sizeof(int32_t);
    int32_t *v5Host = nullptr;
    int32_t *v5Device = nullptr;
    size_t elemCount_v6 = 2;
    size_t fileSize_v6 = elemCount_v6 * sizeof(float);
    float *v6Host = nullptr;
    float *v6Device = nullptr;
    size_t elemCount_v7 = 2;
    size_t fileSize_v7 = elemCount_v7 * sizeof(int32_t);
    int32_t *v7Host = nullptr;
    int32_t *v7Device = nullptr;
    size_t elemCount_v8 = 524288;
    size_t fileSize_v8 = elemCount_v8 * sizeof(float);
    float *v8Host = nullptr;
    float *v8Device = nullptr;
    size_t elemCount_v9 = 524288;
    size_t fileSize_v9 = elemCount_v9 * sizeof(float);
    float *v9Host = nullptr;
    float *v9Device = nullptr;
    size_t elemCount_v10 = 2048;
    size_t fileSize_v10 = elemCount_v10 * sizeof(float);
    float *v10Host = nullptr;
    float *v10Device = nullptr;
    size_t elemCount_v11 = 128;
    size_t fileSize_v11 = elemCount_v11 * sizeof(float);
    float *v11Host = nullptr;
    float *v11Device = nullptr;
    size_t elemCount_v12 = 2048;
    size_t fileSize_v12 = elemCount_v12 * sizeof(float);
    float *v12Host = nullptr;
    float *v12Device = nullptr;
    size_t elemCount_v13 = 10240;
    size_t fileSize_v13 = elemCount_v13 * sizeof(float);
    float *v13Host = nullptr;
    float *v13Device = nullptr;
    size_t elemCount_v14 = 128;
    size_t fileSize_v14 = elemCount_v14 * sizeof(float);
    float *v14Host = nullptr;
    float *v14Device = nullptr;
    size_t elemCount_v15 = 65536;
    size_t fileSize_v15 = elemCount_v15 * sizeof(int32_t);
    int32_t *v15Host = nullptr;
    int32_t *v15Device = nullptr;
    size_t elemCount_v16 = 589824;
    size_t fileSize_v16 = elemCount_v16 * sizeof(float);
    float *v16Host = nullptr;
    float *v16Device = nullptr;
    size_t elemCount_v17 = 589824;
    size_t fileSize_v17 = elemCount_v17 * sizeof(uint16_t);
    uint16_t *v17Host = nullptr;
    uint16_t *v17Device = nullptr;
    size_t elemCount_v18 = 147456;
    size_t fileSize_v18 = elemCount_v18 * sizeof(float);
    float *v18Host = nullptr;
    float *v18Device = nullptr;
    size_t elemCount_v19 = 32768;
    size_t fileSize_v19 = elemCount_v19 * sizeof(uint16_t);
    uint16_t *v19Host = nullptr;
    uint16_t *v19Device = nullptr;
    int64_t v20 = 2;
    int64_t v21 = 0;
    int64_t v22 = 16;
    int64_t v23 = 0;
    int64_t v24 = 256;
    int64_t v25 = 80;
    int64_t v26 = 2;
    int64_t v27 = 32;
    int32_t v28 = 0;
    int32_t v29 = 1;

    int rc = 0;
    bool aclInited = false;
    bool deviceSet = false;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    aclInited = true;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    ACL_CHECK(aclrtSetDevice(deviceId));
    deviceSet = true;
    ACL_CHECK(aclrtCreateStream(&stream));

    ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
    ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
    ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v3));
    ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v3, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize_v4));
    ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize_v4, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v5Host), fileSize_v5));
    ACL_CHECK(aclrtMalloc((void **)&v5Device, fileSize_v5, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v6Host), fileSize_v6));
    ACL_CHECK(aclrtMalloc((void **)&v6Device, fileSize_v6, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v7Host), fileSize_v7));
    ACL_CHECK(aclrtMalloc((void **)&v7Device, fileSize_v7, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v8Host), fileSize_v8));
    ACL_CHECK(aclrtMalloc((void **)&v8Device, fileSize_v8, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v9Host), fileSize_v9));
    ACL_CHECK(aclrtMalloc((void **)&v9Device, fileSize_v9, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v10Host), fileSize_v10));
    ACL_CHECK(aclrtMalloc((void **)&v10Device, fileSize_v10, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v11Host), fileSize_v11));
    ACL_CHECK(aclrtMalloc((void **)&v11Device, fileSize_v11, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v12Host), fileSize_v12));
    ACL_CHECK(aclrtMalloc((void **)&v12Device, fileSize_v12, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v13Host), fileSize_v13));
    ACL_CHECK(aclrtMalloc((void **)&v13Device, fileSize_v13, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v14Host), fileSize_v14));
    ACL_CHECK(aclrtMalloc((void **)&v14Device, fileSize_v14, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v15Host), fileSize_v15));
    ACL_CHECK(aclrtMalloc((void **)&v15Device, fileSize_v15, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v16Host), fileSize_v16));
    ACL_CHECK(aclrtMalloc((void **)&v16Device, fileSize_v16, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v17Host), fileSize_v17));
    ACL_CHECK(aclrtMalloc((void **)&v17Device, fileSize_v17, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v18Host), fileSize_v18));
    ACL_CHECK(aclrtMalloc((void **)&v18Device, fileSize_v18, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&v19Host), fileSize_v19));
    ACL_CHECK(aclrtMalloc((void **)&v19Device, fileSize_v19, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
    ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
    ReadFile("./v3.bin", fileSize_v3, v3Host, fileSize_v3);
    ReadFile("./v4.bin", fileSize_v4, v4Host, fileSize_v4);
    ReadFile("./v5.bin", fileSize_v5, v5Host, fileSize_v5);
    ReadFile("./v6.bin", fileSize_v6, v6Host, fileSize_v6);
    ReadFile("./v7.bin", fileSize_v7, v7Host, fileSize_v7);
    ReadFile("./v8.bin", fileSize_v8, v8Host, fileSize_v8);
    ReadFile("./v9.bin", fileSize_v9, v9Host, fileSize_v9);
    ReadFile("./v10.bin", fileSize_v10, v10Host, fileSize_v10);
    ReadFile("./v11.bin", fileSize_v11, v11Host, fileSize_v11);
    ReadFile("./v12.bin", fileSize_v12, v12Host, fileSize_v12);
    ReadFile("./v13.bin", fileSize_v13, v13Host, fileSize_v13);
    ReadFile("./v14.bin", fileSize_v14, v14Host, fileSize_v14);
    ReadFile("./v15.bin", fileSize_v15, v15Host, fileSize_v15);
    ReadFile("./v16.bin", fileSize_v16, v16Host, fileSize_v16);
    ReadFile("./v17.bin", fileSize_v17, v17Host, fileSize_v17);
    ReadFile("./v18.bin", fileSize_v18, v18Host, fileSize_v18);
    ReadFile("./v19.bin", fileSize_v19, v19Host, fileSize_v19);
    ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v3, v3Host, fileSize_v3, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v4Device, fileSize_v4, v4Host, fileSize_v4, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v5Device, fileSize_v5, v5Host, fileSize_v5, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v6Device, fileSize_v6, v6Host, fileSize_v6, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v7Device, fileSize_v7, v7Host, fileSize_v7, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v8Device, fileSize_v8, v8Host, fileSize_v8, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v9Device, fileSize_v9, v9Host, fileSize_v9, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v10Device, fileSize_v10, v10Host, fileSize_v10, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v11Device, fileSize_v11, v11Host, fileSize_v11, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v12Device, fileSize_v12, v12Host, fileSize_v12, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v13Device, fileSize_v13, v13Host, fileSize_v13, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v14Device, fileSize_v14, v14Host, fileSize_v14, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v15Device, fileSize_v15, v15Host, fileSize_v15, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v16Device, fileSize_v16, v16Host, fileSize_v16, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v17Device, fileSize_v17, v17Host, fileSize_v17, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v18Device, fileSize_v18, v18Host, fileSize_v18, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(v19Device, fileSize_v19, v19Host, fileSize_v19, ACL_MEMCPY_HOST_TO_DEVICE));
    if (static_cast<long double>(v24) < 0.0L ||
        static_cast<long double>(v24) > 256.0L) {
        std::fprintf(stderr, "[ERROR] dynamic scalar v24 exceeds the generated allocation bound [0, 256]. Regenerate with a larger --dynamic-dim instead of patching only main.cpp.\n");
        rc = 2;
        goto cleanup;
    }
    if (static_cast<long double>(v25) < 0.0L ||
        static_cast<long double>(v25) > 256.0L) {
        std::fprintf(stderr, "[ERROR] dynamic scalar v25 exceeds the generated allocation bound [0, 256]. Regenerate with a larger --dynamic-dim instead of patching only main.cpp.\n");
        rc = 2;
        goto cleanup;
    }
    if (static_cast<long double>(v26) < 0.0L ||
        static_cast<long double>(v26) > 256.0L) {
        std::fprintf(stderr, "[ERROR] dynamic scalar v26 exceeds the generated allocation bound [0, 256]. Regenerate with a larger --dynamic-dim instead of patching only main.cpp.\n");
        rc = 2;
        goto cleanup;
    }
    if (static_cast<long double>(v27) < 0.0L ||
        static_cast<long double>(v27) > 256.0L) {
        std::fprintf(stderr, "[ERROR] dynamic scalar v27 exceeds the generated allocation bound [0, 256]. Regenerate with a larger --dynamic-dim instead of patching only main.cpp.\n");
        rc = 2;
        goto cleanup;
    }
    LaunchAttn_swpipe(v1Device, v2Device, v3Device, v4Device, v5Device, v6Device, v7Device, v8Device, v9Device, v10Device, v11Device, v12Device, v13Device, v14Device, v15Device, v16Device, v17Device, v18Device, v19Device, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, stream);

    ACL_CHECK(aclrtSynchronizeStream(stream));

cleanup:
    if (v1Device) aclrtFree(v1Device);
    if (v1Host) aclrtFreeHost(v1Host);
    if (v2Device) aclrtFree(v2Device);
    if (v2Host) aclrtFreeHost(v2Host);
    if (v3Device) aclrtFree(v3Device);
    if (v3Host) aclrtFreeHost(v3Host);
    if (v4Device) aclrtFree(v4Device);
    if (v4Host) aclrtFreeHost(v4Host);
    if (v5Device) aclrtFree(v5Device);
    if (v5Host) aclrtFreeHost(v5Host);
    if (v6Device) aclrtFree(v6Device);
    if (v6Host) aclrtFreeHost(v6Host);
    if (v7Device) aclrtFree(v7Device);
    if (v7Host) aclrtFreeHost(v7Host);
    if (v8Device) aclrtFree(v8Device);
    if (v8Host) aclrtFreeHost(v8Host);
    if (v9Device) aclrtFree(v9Device);
    if (v9Host) aclrtFreeHost(v9Host);
    if (v10Device) aclrtFree(v10Device);
    if (v10Host) aclrtFreeHost(v10Host);
    if (v11Device) aclrtFree(v11Device);
    if (v11Host) aclrtFreeHost(v11Host);
    if (v12Device) aclrtFree(v12Device);
    if (v12Host) aclrtFreeHost(v12Host);
    if (v13Device) aclrtFree(v13Device);
    if (v13Host) aclrtFreeHost(v13Host);
    if (v14Device) aclrtFree(v14Device);
    if (v14Host) aclrtFreeHost(v14Host);
    if (v15Device) aclrtFree(v15Device);
    if (v15Host) aclrtFreeHost(v15Host);
    if (v16Device) aclrtFree(v16Device);
    if (v16Host) aclrtFreeHost(v16Host);
    if (v17Device) aclrtFree(v17Device);
    if (v17Host) aclrtFreeHost(v17Host);
    if (v18Device) aclrtFree(v18Device);
    if (v18Host) aclrtFreeHost(v18Host);
    if (v19Device) aclrtFree(v19Device);
    if (v19Host) aclrtFreeHost(v19Host);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
        stream = nullptr;
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    if (aclInited) {
        aclFinalize();
    }
    return rc;
}
