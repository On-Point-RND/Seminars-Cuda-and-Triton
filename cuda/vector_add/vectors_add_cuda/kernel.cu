#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void add_vector_kernel(const float* a, const float* b, float* result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        result[index] = a[index] + b[index];
    }
}

torch::Tensor VecAddCUDA(const torch::Tensor &a, const torch::Tensor &b){
    const auto width = a.size(0);

    auto result = torch::empty_like(a);

    dim3 threads_per_block(BLOCK_SIZE);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x);

    add_vector_kernel<<<number_of_blocks, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), width);

    return result;
}