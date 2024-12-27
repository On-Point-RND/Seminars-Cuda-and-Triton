#include <torch/extension.h>
torch::Tensor mutmul_cublass_sgemm(const torch::Tensor &A, const torch::Tensor &B);
torch::Tensor mutmul_cublass_hgemm(const torch::Tensor &A, const torch::Tensor &B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_sgemm", torch::wrap_pybind_function(mutmul_cublass_sgemm), "matrix multiplication in float32 (CUDA)");
  m.def("matmul_hgemm", torch::wrap_pybind_function(mutmul_cublass_hgemm), "matrix multiplication in float16 (CUDA)");
}