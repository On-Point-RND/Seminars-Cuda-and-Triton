#include <torch/types.h>

void add_vector_cpu(const float* a, const float* b, float* result, int n) {
    for (int index = 0; index < n; index++){
        result[index] = a[index] + b[index];
    }
}

torch::Tensor VecAddCPU(const torch::Tensor &a, const torch::Tensor &b){
    const auto width = a.size(0);

    auto result = torch::empty_like(a);

    add_vector_cpu(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), width);

    return result;
}