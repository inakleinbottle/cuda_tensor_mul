//
// Created by sam on 08/05/23.
//

#ifndef CUDA_TENSOR_MUL_COMMON_CUH
#define CUDA_TENSOR_MUL_COMMON_CUH

#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <random>
#include <ostream>

template <typename T>
using rp_t = T *__restrict;
template <typename T>
using crp_t = const T *__restrict;


template <typename I, typename J>
constexpr __host__ __device__ I round_up_div(I num, J div) {
    return (num + static_cast<I>(div) - 1) / static_cast<I>(div);
}


struct DivRem {
    int div;
    int rem;
};

__device__ __host__
inline DivRem divide(int idx, int divisor) {

    DivRem result;
    result.div = idx / divisor;
    result.rem = (idx - result.div * divisor);

    return result;
}

inline __host__ __device__ int32_t reverse_idx(int32_t idx, int32_t width, int32_t depth) {
    auto out = 0;
    for (int32_t i = 0; i < depth; ++i) {
        const auto tmp = idx;
        idx /= tmp;
        const auto rem = tmp - idx * width;
        out *= width;
        out += rem;
    }
    return out;
}

inline __host__ __device__ int32_t compute_offset(const int32_t *levels, int32_t level) {
    int32_t result = 0;
    for (int32_t i = 0; i < level; ++i) {
        result += levels[i];
    }
    return result;
}


template <typename S>
S get_error(const thrust::host_vector<S>& result, const thrust::host_vector<S>& expected) {
    S err = 0.0;
    for (size_t i=0; i<result.size(); ++i) {
        auto newerr = abs(expected[i] - result[i]);
        if (newerr > err) {
#ifdef REPORT_NEW_MAX_ERROR
            std::cout << i << ' ' << newerr << '\n';
#endif
            err = newerr;
        }
    }
    return err;
}



template <typename S>
struct ExampleData {
    int32_t width;
    int32_t depth;
    int32_t tensor_size;
    thrust::host_vector<int32_t> level_sizes;
    thrust::host_vector<int32_t> level_offsets;
    thrust::host_vector<S> lhs_data;
    thrust::host_vector<S> rhs_data;
};


template <typename S>
ExampleData<S> get_example_data(int32_t width, int32_t depth, bool both_vectors=true) {

    ExampleData<S> result;
    result.width = width;
    result.depth = depth;
    result.tensor_size = 1;


    auto& sizes = result.level_sizes;
    sizes.reserve(1 + depth);
    sizes.push_back(1);

    auto& offsets = result.level_offsets;
    offsets.reserve(1 + depth);
    offsets.push_back(0);

    for (int32_t i=1; i<=depth; ++i) {
        offsets.push_back(offsets.back() + sizes.back());
        sizes.push_back(sizes.back()*width);
        result.tensor_size = result.tensor_size * width + 1;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<S> dist(-1.0, 1.0);

    auto& lhs = result.lhs_data;
    auto& rhs = result.rhs_data;

    lhs.reserve(result.tensor_size);
    if (both_vectors) {
        rhs.reserve(result.tensor_size);
    }

    for (int32_t i=0; i<result.tensor_size; ++i) {
        lhs.push_back(dist(rng));
        if (both_vectors) {
            rhs.push_back(dist(rng));
        }
    }

    return result;
}



#endif //CUDA_TENSOR_MUL_COMMON_CUH
