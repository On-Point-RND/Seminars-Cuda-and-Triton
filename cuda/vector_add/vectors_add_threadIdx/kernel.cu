#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void add_vector_threadidx_kernel(const float* a, const float* b, float* result) {
    result[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

torch::Tensor VecAddThreadIdxCUDA(const torch::Tensor &a, const torch::Tensor &b){
    const auto width = a.size(0);

    auto result = torch::empty_like(a);

    dim3 threads_per_block(width);
    dim3 number_of_blocks(1);

    add_vector_threadidx_kernel<<<number_of_blocks, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>());

    return result;
}