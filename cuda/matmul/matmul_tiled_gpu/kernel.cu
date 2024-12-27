#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void mutmul_tile_gpu_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < m && tile * TILE_SIZE + tx < k)
            sharedA[ty][tx] = A[row * k + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;
        
        if (col < n && tile * TILE_SIZE + ty < k)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * n + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for (int l = 0; l < TILE_SIZE; ++ l)
            sum += sharedA[ty][l] * sharedB[l][tx];
        
         __syncthreads();

    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

torch::Tensor mutmul_tile_gpu(const torch::Tensor &A, const torch::Tensor &B){
    auto m = A.size(0);
    auto k = A.size(1);
    auto n = B.size(1);

    auto result = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(A.device()));


    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dim3 number_of_blocks((n + threads_per_block.x - 1) / threads_per_block.x, (m + threads_per_block.y - 1) / threads_per_block.y);

    mutmul_tile_gpu_kernel<<<number_of_blocks, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), result.data_ptr<float>(), m, k, n);

    return result;
}