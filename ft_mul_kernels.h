//
// Created by sam on 21/05/23.
//

#ifndef CUDA_TENSOR_MUL_FT_MUL_KERNELS_H
#define CUDA_TENSOR_MUL_FT_MUL_KERNELS_H

#include "common.cuh"


template <typename S>
__global__ void ft_single_level(
    rp_t<S> pd_out,
    crp_t<S> pd_lhs,
    crp_t<S> pd_rhs,
    int32_t out_deg,
    const int32_t* levels
    ) {

    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    const auto iy = blockIdx.y * blockDim.y + threadIdx.y;


    for (int32_t lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
        const auto rhs_deg = out_deg - lhs_deg;
        const auto* lhs_p = pd_lhs + compute_offset(levels, lhs_deg);
        const auto* rhs_p = pd_rhs + compute_offset(levels, rhs_deg);

        const auto& lhs_n = levels[lhs_deg];
        const auto& rhs_n = levels[rhs_deg];

        if (ix < lhs_n && iy < rhs_n) {
//            pd_out[ix*rhs_n + iy] += lhs_p[ix]*rhs_p[iy];
            atomicAdd(pd_out + ix*rhs_n + iy, lhs_p[ix]*rhs_p[iy]);
        }


    }

}



template <typename S>
void ft_mul_multiple_kernels(rp_t<S> pd_out,
                             crp_t<S> pd_lhs,
                             crp_t<S> pd_rhs,
                             int32_t max_depth,
                             const int32_t* d_levels,
                             const int32_t* h_levels,
                             dim3 threads
                             ) {


    for (int32_t out_deg=max_depth; out_deg >= 0; --out_deg) {
        auto *out_p = pd_out + compute_offset(h_levels, out_deg);

        dim3 blocks {
            round_up_div(static_cast<uint32_t>(h_levels[out_deg]), threads.x),
            round_up_div(static_cast<uint32_t>(h_levels[out_deg]), threads.y)
        };

        ft_single_level<<<blocks, threads>>>(out_p, pd_lhs, pd_rhs, out_deg, d_levels);
        cudaDeviceSynchronize();
    }



}




#endif //CUDA_TENSOR_MUL_FT_MUL_KERNELS_H
