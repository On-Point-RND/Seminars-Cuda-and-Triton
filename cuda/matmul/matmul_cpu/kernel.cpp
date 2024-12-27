#include <torch/types.h>

// Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
// a11 a12 
// a21 a22
// a31 a32

// b11 b12 b13 b14
// b21 b22 b23 b24

// a11 a12 a21 a22 a31 a32
// b11 b12 b13 b14 b21 b22 b23 b24

// a11 * b11 + a12 * b21 a11 * b12 + a12 * b22 a11 * b13 + a12 * b23 a11 * b14 + a12 * b24 (i = 0, 0 < j < n)
// a21 * b11 + a22 * b21 a21 * b12 + a22 * b22 a21 * b13 + a22 * b23 a21 * b14 + a22 * b24 (i = 1, 0 < j < n)
// a31 * b11 + a32 * b21 a31 * b12 + a32 * b22 a31 * b13 + a32 * b23 a31 * b14 + a32 * b24 (i = 2, 0 < j < n)

void matmul_cpu_kernel(const float* A, const float* B, float* C, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++){
      float sum = 0.0f;
      for (int l = 0; l < k; l++){
        sum += A[i * k + l] * B[l * n + j];  
      }
      C[i * n + j] = sum;
    }
  }
}

torch::Tensor matmul_cpu(const torch::Tensor &A, const torch::Tensor &B){
    auto m = A.size(0);
    auto k = A.size(1);
    auto n = B.size(1);

    auto result = torch::empty({m, n}, torch::dtype(torch::kFloat32).device(A.device()));

    matmul_cpu_kernel(A.data_ptr<float>(), B.data_ptr<float>(), result.data_ptr<float>(), m, k, n);

    return result;
}