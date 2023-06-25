//
// Created by sam on 08/05/23.
//
#define REPORT_NEW_MAX_ERROR

#include "examples.cuh"

#include "common.cuh"
#include "ftm_cuda_simple.cuh"
#include "ftm_reference.cuh"
#include "ft_mul_kernels.h"
#include "ftm_tiled.cuh"

#include <iostream>

void example1_ft_multiply_and_add() {

    std::cout << "Example 1\n"
              << "Compute product of lhs and rhs and add the result into the (blank) tensor out"
              << "\n\n";


    auto data = get_example_data<float>(32, 2);

    ComputeInfo info {
        data.width,
        data.depth,
        1,
        nullptr,
        nullptr
    };
    const int tile_letters = info.tile_letters;
    const int tile_width = data.width * data.width;
    const int tile_size = tile_width * tile_width;

    for (int i=0; i<data.depth; ++i) {
        std::cout << i << ' ' << compute_offset(data.level_sizes.data(), i)  << ' ' << data.level_sizes[i] << '\n';
    }

    const thrust::device_vector<float> lhs(data.lhs_data);
    const thrust::device_vector<float> rhs(data.rhs_data);
    const thrust::device_vector<int32_t> levels(data.level_sizes);

    thrust::device_vector<float> out(data.tensor_size);


    dim3 threads_per_block(32, 32);
    dim3 blocks {
        round_up_div(static_cast<uint32_t>(data.tensor_size), threads_per_block.x),
        round_up_div(static_cast<uint32_t>(data.tensor_size), threads_per_block.y)
    };

    info.levels = thrust::raw_pointer_cast(&levels[0]);

    auto start = std::chrono::high_resolution_clock::now();
//    ft_mul_kernel<<<blocks, threads_per_block>>>(
//        thrust::raw_pointer_cast(&out[0]),
//        thrust::raw_pointer_cast(&lhs[0]),
//        thrust::raw_pointer_cast(&rhs[0]),
//        data.depth,
//        thrust::raw_pointer_cast(&levels[0])
//    );

    if (info.depth >= 2*info.tile_letters) {
        ft_tiled_mul<float><<<blocks.x, threads_per_block>>>(
            {thrust::raw_pointer_cast(&out[0]), nullptr},
            {thrust::raw_pointer_cast(&lhs[0]), nullptr},
            {thrust::raw_pointer_cast(&rhs[0]), nullptr},
            info
            );
    }

    ft_mul_multiple_kernels(thrust::raw_pointer_cast(&out[0]),
                            thrust::raw_pointer_cast(&lhs[0]),
                            thrust::raw_pointer_cast(&rhs[0]),
                            std::min(data.depth, 2*info.tile_letters-1),
                            thrust::raw_pointer_cast(&levels[0]),
                            &data.level_sizes[0],
                            threads_per_block
                            );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by kernel " << time.count() << "us\n";

    thrust::host_vector<float> result(out);

    thrust::host_vector<float> expected(data.tensor_size);
    ft_mul_host(expected.data(), data.lhs_data.data(), data.rhs_data.data(), data.depth, data.level_sizes.data());


    auto err = get_error(result, expected);
    std::cout << "Max error: " << err << '\n';
}
