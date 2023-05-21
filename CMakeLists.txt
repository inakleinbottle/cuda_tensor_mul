cmake_minimum_required(VERSION 3.24)
project(cuda_tensor_mul CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_tensor_mul
        main.cu
        ftm_tiled.cuh
        common.cuh
        ftm_reference.cuh
        ftm_cuda_simple.cuh
        examples.cuh
        example1.cu
        ft_mul_kernels.h
        )

set_target_properties(cuda_tensor_mul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


find_package(Thrust REQUIRED)

target_link_libraries(cuda_tensor_mul PRIVATE Thrust::Thrust)
