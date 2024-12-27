#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 4

__global__ void add_vector_kernel(const float* a, const float* b, float* result, int nx, int ny, int nz) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < nx && j < ny && k <nz){
        int index = i + j * nx + k * nx * ny;
        if (index < (nx * ny * nz)) {
            result[index] = a[index] + b[index];
        }
    }
}

torch::Tensor VecAddCUDA(const torch::Tensor &a, const torch::Tensor &b){
    const auto width = a.size(0); 

    auto result = torch::empty_like(a);
    int nx = 4, ny = 4, nz = 4;

    dim3 threads_per_block(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 number_of_blocks(
        (nx + threads_per_block.x - 1) / threads_per_block.x,
        (ny + threads_per_block.y - 1) / threads_per_block.y,
        (nz + threads_per_block.z - 1) / threads_per_block.z
    );


    add_vector_kernel<<<number_of_blocks, threads_per_block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), nx, ny, nz);

    return result;
}