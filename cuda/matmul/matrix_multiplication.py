import time
import torch
from torch.utils import cpp_extension
import os 
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

def time_pytorch_cpu_function(func, input1, input2, repeat):
    
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input1, input2)

    start = time.time()
    for _ in range(repeat):
        func(input1, input2)
    end = time.time()
    ms = (end - start) / repeat * 1000
    return ms

def time_pytorch_cuda_function(func, input1, input2, repeat):

    # Warmup
    for _ in range(5):
        func(input1, input2)

    start = time.time()
    for _ in range(repeat):
        func(input1, input2)
        torch.cuda.synchronize()
    end = time.time()
    torch.cuda.synchronize()
    ms = (end - start) / repeat * 1000
    return ms

# def time_pytorch_cuda_function(func, input1, input2, repeat):
#     # CUDA IS ASYNC so can't use python time module
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)

#     # Warmup
#     for _ in range(5):
#         func(input1, input2)

#     start.record()
#     for _ in range(repeat):
#         func(input1, input2)
#         torch.cuda.synchronize()
#     end.record()
#     torch.cuda.synchronize()
    
#     return start.elapsed_time(end)

matmul_cpu = cpp_extension.load(
    name='matmul_cpu',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cpu/matmul_cpu.cpp', 
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cpu/kernel.cpp'
    ]
)

matmul2d_gpu = cpp_extension.load(
    name='matmul2d_gpu',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul2d_gpu/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul2d_gpu/kernel.cu'
    ]
)

matmultile_gpu = cpp_extension.load(
    name='matmultile_gpu',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_tiled_gpu/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_tiled_gpu/kernel.cu'
    ]
)

matmul_cublass = cpp_extension.load(
    name='matmul_cublass',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cublass/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cublass/kernel.cu'
    ]
)

matmul_cutlass = cpp_extension.load(
    name='matmul_cutlass',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cutlass/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_2_matmul/matmul_cutlass/kernel.cu'
    ],
    extra_include_paths=[r"/home/RecSys/cutlass/include"]
)


# A = torch.ones((3, 4), device=torch.device('cpu'))
# B = torch.ones((4, 2), device=torch.device('cpu'))

A = torch.ones((256, 1024), device=torch.device('cpu'))
B = torch.ones((1024, 256), device=torch.device('cpu'))

# A = torch.ones((128 * 200, 256), device=torch.device('cpu'))
# B = torch.ones((256, 1024), device=torch.device('cpu'))


# CPU matmul
print(f"matrix a: {A}")
print(f"matrix b: {B}")

print("")
print(f"CPU")

t = time_pytorch_cpu_function(matmul_cpu.matmul, A, B, repeat=100)
print(f"cpu kernel matmul: {t}")

t = time_pytorch_cpu_function(torch.matmul, A, B, repeat=100)
print(f"torch cpu kernel matmul: {t}")

# GPU matmul 
A = A.to(torch.device("cuda:0"))
B = B.to(torch.device("cuda:0"))

print("")
print(f"GPU fp32")

t = time_pytorch_cuda_function(matmul2d_gpu.matmul, A, B, repeat=100)
print(f"gpu kernel matmul2d: {t}")

t = time_pytorch_cuda_function(matmultile_gpu.matmul, A, B, repeat=100)
print(f"gpu kernel tiled matmul: {t}")

t = time_pytorch_cuda_function(matmul_cublass.matmul_sgemm, A, B, repeat=100)
print(f"gpu kernel cublass matmul fp32: {t}")

t = time_pytorch_cuda_function(matmul_cutlass.matmul, A, B, repeat=100)
print(f"gpu kernel cutlass matmul fp32: {t}")

t = time_pytorch_cuda_function(torch.matmul, A, B, repeat=100)
print(f"torch gpu kernel matmul fp32: {t}")

print("")
print(f"GPU fp16")

A = A.half()
B = B.half()
t = time_pytorch_cuda_function(matmul_cublass.matmul_hgemm, A, B, repeat=100)
print(f"gpu kernel cublass matmul fp16: {t}")
t = time_pytorch_cuda_function(matmul_cutlass.matmul_fp16, A, B, repeat=100)
print(f"gpu kernel cutlass matmul fp16: {t}")
t = time_pytorch_cuda_function(torch.matmul, A, B, repeat=100)
print(f"torch gpu kernel matmul fp16: {t}")

print("")
print(f"GPU bf16")

A = A.to(torch.bfloat16)
B = B.to(torch.bfloat16)
t = time_pytorch_cuda_function(matmul_cutlass.matmul_bf16, A, B, repeat=100)
print(f"gpu kernel cutlass matmul bf16: {t}")
t = time_pytorch_cuda_function(torch.matmul, A, B, repeat=100)
print(f"torch gpu kernel matmul bf16: {t}")
