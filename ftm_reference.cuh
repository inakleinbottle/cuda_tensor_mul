//
// Created by sam on 08/05/23.
//

#ifndef CUDA_TENSOR_MUL_FTM_REFERENCE_H
#define CUDA_TENSOR_MUL_FTM_REFERENCE_H

#include "common.cuh"


template <typename S>
void ft_mul_host(rp_t<S> pd_out,
                 crp_t<S> pd_lhs,
                 crp_t<S> pd_rhs,
                 int32_t max_depth,
                 const int32_t *levels) {

    for (int32_t out_deg = max_depth; out_deg >= 0; --out_deg) {
        auto *out_p = pd_out + compute_offset(levels, out_deg);

        for (int32_t lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
            const auto rhs_deg = out_deg - lhs_deg;
            const auto *lhs_p = pd_lhs + compute_offset(levels, lhs_deg);
            const auto *rhs_p = pd_rhs + compute_offset(levels, rhs_deg);

            const auto lhs_n = levels[lhs_deg];
            const auto rhs_n = levels[rhs_deg];

            for (auto ix = 0; ix < lhs_n; ++ix) {
                for (auto jx = 0; jx < rhs_n; ++jx) {
                    out_p[ix * rhs_n + jx] += lhs_p[ix] * rhs_p[jx];
                }
            }
        }
    }
}


#endif //CUDA_TENSOR_MUL_FTM_REFERENCE_H
