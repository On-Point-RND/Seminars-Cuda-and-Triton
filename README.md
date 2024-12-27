# Seminars-Cuda-and-Triton
To run cuda kernels and use the Triton notebook, use the following image:
```
docker pull dmitryredkosk/bitsandbytes_transformer
```

To install `cutlass`, execute the following commands:
```
# Clone the Cutlass repository
git clone https://github.com/NVIDIA/cutlass.git

# Set the CUDA compiler path
export CUDACXX=/usr/local/cuda/bin/nvcc

# Navigate to the Cutlass directory and set up the build environment
cd cutlass
mkdir build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS=75
```