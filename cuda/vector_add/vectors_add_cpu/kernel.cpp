#include <torch/extension.h>
torch::Tensor VecAddCPU(const torch::Tensor &a, const torch::Tensor &b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", torch::wrap_pybind_function(VecAddCPU), "add vectors (CPU)");
}