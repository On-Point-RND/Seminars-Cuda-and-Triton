#include <torch/extension.h>
torch::Tensor matmul2d_gpu(const torch::Tensor &A, const torch::Tensor &B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", torch::wrap_pybind_function(matmul2d_gpu), "matrix multiplication (CUDA)");
}