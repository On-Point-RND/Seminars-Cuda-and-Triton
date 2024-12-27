#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_vector_blockidx_kernel(const float* a, const float* b, float* result) {
    result[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

torch::Tensor VecAddBlockIdxCUDA(const torch::Tensor &a, const torch::Tensor &b){
    const auto width = a.size(0);

    auto result = torch::empty_like(a);

    dim3 threads_per_block(1);
    dim3 number_of_blocks(width);

    add_vector_blockidx_kernel<<<number_of_blocks, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>());

    return result;
}