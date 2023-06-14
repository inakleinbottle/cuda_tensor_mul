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

#ifndef CUDA_TENSOR_MUL_FT_ANTIPODE_TILED_CUH
#define CUDA_TENSOR_MUL_FT_ANTIPODE_TILED_CUH

#include "common.cuh"



struct AntipodeInfo {
    int32_t width;
    int32_t depth;
    int32_t tile_letters;
    int32_t untiled_size;
    const int32_t* levels;
};


template <typename S>
__global__ void ft_antipode_kernel(
    rp_t<S> pd_out,
    crp_t<S> pd_in,
    AntipodeInfo info
    ) {
    const auto& width = info.width;
    const auto& ix = threadIdx.x;
    const auto& iy = threadIdx.y;
    const auto untiled_idx = ix*blockDim.y + iy;
    const auto& tile_width = info.levels[info.tile_letters];

    const auto tot_tile_let = 2 * info.tile_letters;

    if (untiled_idx < info.untiled_size) {
        int32_t degree = 0;
        const auto reverse_idx = reverse_idx_to(untiled_idx, width, &degree);
        if ((degree & 1) == 0) {
            pd_out[reverse_idx] = pd_in[untiled_idx];
        } else {
            pd_out[reverse_idx] = -pd_in[untiled_idx];
        }
    }

    const auto rix = reverse_idx(ix, width, info.tile_letters);
    const auto riy = reverse_idx(iy, width, info.tile_letters);

    pd_out += info.untiled_size;
    pd_in += info.untiled_size;

    for (int32_t out_deg=tot_tile_let; out_deg <= info.depth; ++out_deg) {
        const auto mid_deg = out_deg - tot_tile_let;
        const auto& mid_stride = info.levels[mid_deg];

        for (int32_t idx=blockIdx.x; idx < info.levels[mid_deg]; idx += blockDim.x) {
            const auto ridx = reverse_idx(idx, width, mid_deg);
            if (ix < tile_width && iy < tile_width) {
                const auto read = (ix*mid_stride + idx)*tile_width + iy;
                const auto write = (riy*mid_stride + ridx)*tile_width + rix;


                if ((mid_deg & 1) == 0) {
                    pd_out[write] = pd_in[read];
                } else {
                    pd_out[write] = -pd_in[read];
                }
            }
        }

        pd_out += info.levels[out_deg];
        pd_in += info.levels[out_deg];
    }




}



#endif //CUDA_TENSOR_MUL_FT_ANTIPODE_TILED_CUH
