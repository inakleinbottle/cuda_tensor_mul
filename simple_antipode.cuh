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

#ifndef CUDA_TENSOR_MUL_SIMPLE_ANTIPODE_CUH
#define CUDA_TENSOR_MUL_SIMPLE_ANTIPODE_CUH

#include "common.cuh"

template <typename S>
__global__ void ft_antipode_simple(
        rp_t<S> pd_out,
        crp_t<S> pd_in,
        int32_t size,
        int32_t width
    ) {
    const auto ix = static_cast<int32_t>(blockIdx.x*blockDim.x + threadIdx.x);
    int32_t degree = 0;

    if (ix < size) {
        const auto rix = reverse_idx_to(ix, width, &degree);
        if ((degree & 1) == 0) {
            pd_out[rix] = pd_in[ix];
        } else {
            pd_out[rix] = -pd_in[ix];
        }
    }

}


template <typename S>
void ft_antipode_cpu(
    rp_t<S> ph_out,
    crp_t<S> ph_in,
    int32_t degree,
    const int32_t* levels
    ) {
    const auto& width = levels[1];

    for (int32_t deg=0; deg<=degree; ++deg) {
        for (int32_t i=0; i<levels[deg]; ++i) {
            auto ridx = reverse_idx(i, width, deg);
            if ((deg & 1) == 0) {
                ph_out[ridx] = ph_in[i];
            } else {
                ph_out[ridx] = -ph_in[i];
            }
        }

    }

}


#endif //CUDA_TENSOR_MUL_SIMPLE_ANTIPODE_CUH
