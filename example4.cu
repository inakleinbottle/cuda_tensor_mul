// Copyright (c) 2023 Datasig Group. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//
// Created by user on 13/06/23.
//


#include "examples.cuh"
#include "common.cuh"

#include "simple_antipode.cuh"
#include "ft_antipode_tiled.cuh"

void example4_ft_antipode() {
    std::cout << "Example 4\n"
              << "Compute the antipode a free tensor"
              << "\n\n";

    auto data = get_example_data<float>(5, 11);

    const thrust::device_vector<float> in_data(data.lhs_data);
    thrust::device_vector<float> out(data.tensor_size);
    const thrust::device_vector<int32_t> levels(data.level_sizes);

    auto threads_per_block = 128;
    auto blocks = round_up_div(data.tensor_size, threads_per_block);

    auto start = std::chrono::high_resolution_clock::now();
    ft_antipode_simple<<<blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(&out[0]),
        thrust::raw_pointer_cast(&in_data[0]),
        data.tensor_size,
        data.width
        );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by kernel " << time.count() << "us\n";




}
