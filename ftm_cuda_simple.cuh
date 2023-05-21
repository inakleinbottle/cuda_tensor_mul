//
// Created by sam on 08/05/23.
//

#ifndef CUDA_TENSOR_MUL_FTM_CUDA_SIMPLE_CUH
#define CUDA_TENSOR_MUL_FTM_CUDA_SIMPLE_CUH

#include "common.cuh"


template <typename S>
__global__ void
ft_mul_kernel(rp_t<S> pd_out,
              crp_t<S> pd_lhs,
              crp_t<S> pd_rhs,
              int32_t max_depth,
              const int32_t *levels) {

    auto x_offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y_offset = blockIdx.y * blockDim.y + threadIdx.y;
    const auto step = gridDim.x * blockDim.x;
    const auto grid_step = gridDim.y * blockDim.y;

    for (int32_t out_deg = max_depth; out_deg >= 0; --out_deg) {
        auto *out_p = pd_out + compute_offset(levels, out_deg);

        auto y_set = grid_step;

        for (int32_t lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
            const auto rhs_deg = out_deg - lhs_deg;
            const auto *lhs_p = pd_lhs + compute_offset(levels, lhs_deg);
            const auto *rhs_p = pd_rhs + compute_offset(levels, rhs_deg);

            __syncthreads();

            const auto lhs_n = levels[lhs_deg];
            const auto rhs_n = levels[rhs_deg];

            auto ix = x_offset;
            auto iy = y_offset;


//            __syncthreads();

            if (ix < lhs_n && iy < rhs_n) {
                out_p[ix*rhs_n + iy] += lhs_p[ix]*rhs_p[iy];
            }
//            atomicAdd(out_p + ix*rhs_n + iy, lhs_p[ix] * rhs_p[iy]);
//                __syncthreads();

//            for (auto ix = x_offset; ix < lhs_n; ix += step) {
//                auto *optr = out_p + ix * rhs_n;
//                const auto lhs_val = lhs_p[ix];
//                for (auto jx = y_offset; jx < rhs_n; jx += grid_step) {
//                    optr[jx] += lhs_val * rhs_p[jx];
//                }
//            }
//            __syncthreads();
        }
    }
}

#endif //CUDA_TENSOR_MUL_FTM_CUDA_SIMPLE_CUH
