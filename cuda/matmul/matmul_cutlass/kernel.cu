#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
// #include <cutlass/half.h> // Include the CUTLASS header for half_t

torch::Tensor mutmul_fp32_cutlass(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("mutmul_fp32_cutlass", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0); 
  auto N = B.size(1);
  auto K = A.size(1);  // = B.size(0)
  auto C = torch::empty({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      float,                          // ElementA
      cutlass::layout::RowMajor,      // LayoutA
      float,                          // ElementB
      cutlass::layout::RowMajor,      // LayoutB
      float,                          // ElementOutput
      cutlass::layout::RowMajor,      // LayoutOutput
      float,                          // ElementAccumulator
      cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
      cutlass::arch::Sm75
      >;

  Gemm gemmOp;
 
  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<float>(), K},
      {B.data_ptr<float>(), N},
      {C.data_ptr<float>(), N},
      {C.data_ptr<float>(), N},
      {1.0f, 0.0f}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
};



// Function to demonstrate conversion
cutlass::half_t convertFloatToHalf(float value) {
    return cutlass::half_t(value); // Use the constructor to convert float to half_t
}

torch::Tensor mutmul_fp16_cutlass(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllSameGPU("mutmul_fp16_cutlass", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0); 
  auto N = B.size(1);
  auto K = A.size(1);  // = B.size(0)
  auto C = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,                          // ElementA
      cutlass::layout::RowMajor,      // LayoutA
      cutlass::half_t,                          // ElementB
      cutlass::layout::RowMajor,      // LayoutB
      cutlass::half_t,                          // ElementOutput
      cutlass::layout::RowMajor,      // LayoutOutput
      float,                          // ElementAccumulator
      cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
      cutlass::arch::Sm75
      >;

  Gemm gemmOp;
 
  using GemmCoord = cutlass::gemm::GemmCoord;

  cutlass::half_t alpha_h = convertFloatToHalf(1.0f), beta_h = convertFloatToHalf(0.0f);
  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::half_t *)A.data_ptr<torch::Half>(), K},
      {(cutlass::half_t *)B.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)C.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)C.data_ptr<torch::Half>(), N},
      {alpha_h, beta_h}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
};

// Function to demonstrate conversion
cutlass::bfloat16_t convertFloatToBf16(float value) {
    return cutlass::bfloat16_t(value); // Use the constructor to convert float to bfloat16
}


torch::Tensor mutmul_bf16_cutlass(torch::Tensor &A, torch::Tensor &B) {
  torch::checkAllSameGPU("mutmul_bf16_cutlass", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0); 
  auto N = B.size(1);
  auto K = A.size(1);  // = B.size(0)
  auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::bfloat16_t,                          // ElementA
      cutlass::layout::RowMajor,      // LayoutA
      cutlass::bfloat16_t,                          // ElementB
      cutlass::layout::RowMajor,      // LayoutB
      cutlass::bfloat16_t,                          // ElementOutput
      cutlass::layout::RowMajor,      // LayoutOutput
      float,                          // ElementAccumulator
      cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
      cutlass::arch::Sm75
      >;

  Gemm gemmOp;
 
  using GemmCoord = cutlass::gemm::GemmCoord;

  cutlass::bfloat16_t alpha_b = convertFloatToBf16(1.0f), beta_b = convertFloatToBf16(0.0f);
  // typename Gemm::Arguments arguments{
  //     {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
  //      static_cast<GemmCoord::Index>(K)},
  //     {reinterpret_cast<cutlass::bfloat16_t*>(const_cast<torch::Tensor&>(A).data_ptr<at::BFloat16>()), K},
  //     {reinterpret_cast<cutlass::bfloat16_t*>(const_cast<torch::Tensor&>(B).data_ptr<at::BFloat16>()), N},
  //     {reinterpret_cast<cutlass::bfloat16_t*>(const_cast<torch::Tensor&>(C).data_ptr<at::BFloat16>()), N},
  //     {reinterpret_cast<cutlass::bfloat16_t*>(const_cast<torch::Tensor&>(C).data_ptr<at::BFloat16>()), N},
  //     {alpha_b, beta_b}};

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {reinterpret_cast<cutlass::bfloat16_t*>(A.data_ptr<at::BFloat16>()), K},
      {reinterpret_cast<cutlass::bfloat16_t*>(B.data_ptr<at::BFloat16>()), N},
      {reinterpret_cast<cutlass::bfloat16_t*>(C.data_ptr<at::BFloat16>()), N},
      {reinterpret_cast<cutlass::bfloat16_t*>(C.data_ptr<at::BFloat16>()), N},
      {alpha_b, beta_b}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
};
