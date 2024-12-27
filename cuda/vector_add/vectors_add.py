import time
import torch
from torch.utils import cpp_extension
from vectors_add_triton import triton_kernel

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


vectors_add_cpu = cpp_extension.load(
    name='vectors_add_cpu',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cpu/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cpu/kernel_1.cpp'
    ]
)

vectors_add_blockIdx_kernel = cpp_extension.load(
    name='vectors_add_blockIdx_kernel',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_blockIdx/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_blockIdx/kernel.cu'
    ]
)

vectors_add_threadIdx_kernel = cpp_extension.load(
    name='vectors_add_threadIdx_kernel',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_threadIdx/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_threadIdx/kernel.cu'
    ]
)

vectors_add_kernel = cpp_extension.load(
    name='vectors_add_kernel',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cuda/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cuda/kernel.cu'
    ]
)

vectors_add_kernel_3d = cpp_extension.load(
    name='vectors_add_kernel',
    sources=[
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cuda_3d/kernel.cpp', 
        '/home/Quantization/gpu_mode/Lecture_1_Profiling_and_Integrating_CUDA_kernels_in_PyTorch/vectors_add_cuda_3d/kernel.cu'
    ]
)



a = torch.ones(1024, device=torch.device('cpu'))
b = 2 * torch.ones(1024, device=torch.device('cpu'))

# CPU vector addition 
print(f"vec a: {a}")
print(f"vec b: {b}")

# cpp extension
t = time_pytorch_cpu_function(vectors_add_cpu.add, a, b, repeat=100)
print(f"cpu time: {t}")

#torch  cpu
t = time_pytorch_cuda_function(torch.add, a, b, repeat=100)
print(f"cpu torch time: {t}")

# GPU vector addition
a = a.to(device=torch.device('cuda:0'))
b = b.to(device=torch.device('cuda:0'))

# each block corresponds to vector element. Only one thread per block
t = time_pytorch_cuda_function(vectors_add_blockIdx_kernel.add, a, b, repeat=100)
print(f"gpu blockIdx_kernel time: {t}")

# each thread corresponds to vector element. Only one block with threads
t = time_pytorch_cuda_function(vectors_add_threadIdx_kernel.add, a, b, repeat=100)
print(f"gpu threadIdx_kernel time: {t}")

# 1d block with threads
time_pytorch_cuda_function(vectors_add_kernel.add, a, b, repeat=100)
print(f"gpu 1d kernel time: {t}")

# 3d block
t = time_pytorch_cuda_function(vectors_add_kernel_3d.add, a, b, repeat=100)
print(f"gpu 3d kernel time: {t}")

# triton
t = time_pytorch_cuda_function(triton_kernel.add, a, b, repeat=100)
print(f"triton kernel time: {t}")

# torch gpu
t = time_pytorch_cuda_function(torch.add, a, b, repeat=100)
print(f"gpu torch time: {t}")
