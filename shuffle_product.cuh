//
// Created by sam on 08/09/23.
//

#ifndef CUDA_TENSOR_MUL_SHUFFLE_PRODUCT_CUH
#define CUDA_TENSOR_MUL_SHUFFLE_PRODUCT_CUH

#include "common.cuh"

__host__ __device__ inline
std::array<int32_t, 2>
unshuffle_word(int32_t width, int32_t degree, int32_t mask, int32_t word)
{
    std::array<int32_t, 2> result {0, 0};

    for (int32_t i=0; i<degree; ++i, mask>>=1) {
        result[0] *= width;
        result[1]*= width;

        auto tmp = word;
        word /= width;
        result[mask & 1] += tmp - (word * width);
    }

    return result;
}

__host__ __device__ inline int32_t right_degree(int32_t mask) {
    return __builtin_popcount(mask);
}



#endif //CUDA_TENSOR_MUL_SHUFFLE_PRODUCT_CUH
