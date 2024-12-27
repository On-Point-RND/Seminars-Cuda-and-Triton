#include <torch/extension.h>
torch::Tensor mutmul_tile_gpu(const torch::Tensor &A, const torch::Tensor &B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", torch::wrap_pybind_function(mutmul_tile_gpu), "matrix multiplication (CUDA)");
}