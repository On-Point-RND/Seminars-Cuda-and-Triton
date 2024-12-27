#include <torch/extension.h>
torch::Tensor VecAddBlockIdxCUDA(const torch::Tensor &a, const torch::Tensor &b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", torch::wrap_pybind_function(VecAddBlockIdxCUDA), "add vectors (CUDA)");
}