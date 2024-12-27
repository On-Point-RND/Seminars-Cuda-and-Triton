#include <torch/extension.h>
torch::Tensor mutmul_fp32_cutlass(const torch::Tensor &A, const torch::Tensor &B);
torch::Tensor mutmul_fp16_cutlass(const torch::Tensor &A, const torch::Tensor &B);
torch::Tensor mutmul_bf16_cutlass(torch::Tensor &A, torch::Tensor &B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", torch::wrap_pybind_function(mutmul_fp32_cutlass), "matrix multiplication in float32 (CUDA)");
  m.def("matmul_fp16", torch::wrap_pybind_function(mutmul_fp16_cutlass), "matrix multiplication in float32 (CUDA)");
  m.def("matmul_bf16", torch::wrap_pybind_function(mutmul_bf16_cutlass), "matrix multiplication in float32 (CUDA)");
}