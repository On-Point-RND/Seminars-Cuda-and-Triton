#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matmul2d_gpu_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum = 0.0f;
        for (int l = 0; l < k; l++){
            sum += A[row * k + l] * B[n * l + col];
        };
        C[row * n + col] = sum;
    }
}

torch::Tensor matmul2d_gpu(const torch::Tensor &A, const torch::Tensor &B){
    auto m = A.size(0);
    auto k = A.size(1);
    auto n = B.size(1);

    auto result = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(A.device()));


    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 number_of_blocks((n + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);

    matmul2d_gpu_kernel<<<number_of_blocks, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), result.data_ptr<float>(), m, k, n);

    return result;
}