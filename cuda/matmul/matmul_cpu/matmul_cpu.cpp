#include <torch/extension.h>
torch::Tensor matmul_cpu(const torch::Tensor &a, const torch::Tensor &b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", torch::wrap_pybind_function(matmul_cpu), "Matrix multiplication (CPU)");
}