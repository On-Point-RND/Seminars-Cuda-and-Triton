#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}


torch::Tensor mutmul_cublass_sgemm(const torch::Tensor &A, const torch::Tensor &B){
    auto m = A.size(0);
    auto k = A.size(1);
    auto n = B.size(1);

    auto C = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(A.device()));


    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(
        handle, 
        CUBLAS_OP_N, /* no transposition life side CUBLAS_OP_T*/ 
        CUBLAS_OP_N, /* no transposition right side*/
        n, m, k, 
        &alpha, 
        B.data_ptr<float>(), n, 
        A.data_ptr<float>(), k, 
        &beta, 
        C.data_ptr<float>(), n));

    CHECK_CUBLAS(cublasDestroy(handle));

    return C;
}


torch::Tensor mutmul_cublass_hgemm(const torch::Tensor &A, const torch::Tensor &B){
    auto m = A.size(0);
    auto k = A.size(1);
    auto n = B.size(1);

    auto C = torch::empty({m, n}, torch::dtype(torch::kFloat16).device(A.device()));


    // CUDA setup
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    CHECK_CUBLAS(cublasHgemm(
        handle, 
        CUBLAS_OP_N, /* no transposition life side CUBLAS_OP_T*/ 
        CUBLAS_OP_N, /* no transposition right side*/
        n, m, k, 
        &alpha_h, 
        reinterpret_cast<const __half*>(B.data_ptr<torch::Half>()), n, 
        reinterpret_cast<const __half*>(A.data_ptr<torch::Half>()), k, 
        &beta_h, 
        reinterpret_cast<__half*>(C.data_ptr<torch::Half>()), n));

    CHECK_CUBLAS(cublasDestroy(handle));

    return C;
}