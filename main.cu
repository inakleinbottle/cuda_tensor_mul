#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cub/cub.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>

#include <random>

#include "stdio.h"


constexpr int32_t WIDTH = 4;
constexpr int32_t DEPTH = 8;


__host__ __device__ int32_t reverse_idx(int32_t idx, int32_t width, int32_t depth) {
    auto out = 0;
    for (int32_t i=0; i<depth; ++i) {
        const auto tmp = idx;
        idx /= tmp;
        const auto rem = tmp - idx*width;
        out *= width;
        out += rem;
    }
    return out;
}


__host__ __device__ inline  int32_t compute_offset(const uint32_t* levels, int32_t level) {
    int32_t result = 0;
    for (int32_t i=0; i<level; ++i) {
        result += levels[i];
    }
    return result;
}

template <typename I, typename J>
constexpr __host__ __device__ I round_up_div(I num, J div) {
    return (num + static_cast<I>(div) - 1) / static_cast<I>(div);
}

void ft_mul_host(float* __restrict__ pd_out,
              const float* __restrict__ pd_lhs,
              const float* __restrict__ pd_rhs,
              int32_t max_depth,
              const uint32_t* levels) {

    for (int32_t out_deg = max_depth; out_deg >= 0; --out_deg) {
        auto* out_p = pd_out + compute_offset(levels, out_deg);

        for (int32_t lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
            const auto rhs_deg =  out_deg - lhs_deg;
            const auto* lhs_p = pd_lhs + compute_offset(levels, lhs_deg);
            const auto* rhs_p = pd_rhs + compute_offset(levels, rhs_deg);

            const auto lhs_n = levels[lhs_deg];
            const auto rhs_n = levels[rhs_deg];

            for (auto ix = 0; ix < lhs_n; ++ix) {
                for (auto jx = 0; jx < rhs_n; ++jx) {
                    out_p[ix*rhs_n + jx] += lhs_p[ix] * rhs_p[jx];
                }
            }
        }
    }
}

__global__ void
ft_mul_kernel(float* __restrict__ pd_out,
              const float* __restrict__ pd_lhs,
              const float* __restrict__ pd_rhs,
              int32_t max_depth,
              const uint32_t* levels) {

    auto x_offset = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y_offset = blockIdx.y*blockDim.y + threadIdx.y;
    const auto step = gridDim.x * blockDim.x;
    const auto grid_step = gridDim.y*blockDim.y;


    for (int32_t out_deg = max_depth; out_deg >= 0; --out_deg) {
        auto* out_p = pd_out + compute_offset(levels, out_deg);

        auto y_set = grid_step;

        for (int32_t lhs_deg = out_deg; lhs_deg >= 0; --lhs_deg) {
            const auto rhs_deg =  out_deg - lhs_deg;
            const auto* lhs_p = pd_lhs + compute_offset(levels, lhs_deg);
            const auto* rhs_p = pd_rhs + compute_offset(levels, rhs_deg);

            __syncthreads();

            const auto lhs_n = levels[lhs_deg];
            const auto rhs_n = levels[rhs_deg];

            __syncthreads();
            auto ix = x_offset;
            auto iy = y_offset;
            if (ix < lhs_n && iy < rhs_n) {
                out_p[ix * rhs_n + iy] += lhs_p[ix]*rhs_p[iy];
            }
//            for (auto ix = x_offset; ix < lhs_n; ix += step) {
//                auto *optr = out_p + ix * rhs_n;
//                const auto lhs_val = lhs_p[ix];
//                for (auto jx = y_offset; jx < rhs_n; jx += grid_step) {
//                    optr[jx] += lhs_val * rhs_p[jx];
//                }
//            }
            __syncthreads();
        }
    }
}


int main() {

    thrust::host_vector<uint32_t> powers;
    powers.reserve(1 + DEPTH);
    powers.push_back(1);
    thrust::host_vector<int32_t> offsets;
    offsets.reserve(1 + DEPTH);
    offsets.push_back(0);

    int32_t tensor_size = 1;
    for (int32_t i=1; i<=DEPTH; ++i) {
        offsets.push_back(offsets.back() + powers.back());
        powers.push_back(powers.back()*WIDTH);
        tensor_size = tensor_size * WIDTH + 1;
    }

    for (auto&& val : offsets) {
        std::cout << val << '\n';
    }

    thrust::device_vector<uint32_t> device_powers(powers);
//    thrust::device_vector<int32_t> device_offsets(offsets);

    thrust::host_vector<float> in_left;
    thrust::host_vector<float> in_right;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    in_left.reserve(tensor_size);
    in_right.reserve(tensor_size);
    for (int32_t i=0; i<tensor_size; ++i) {
        in_left.push_back(dist(rng));
        in_right.push_back(dist(rng));
    }

    thrust::device_vector<float> din_left(in_left);
    thrust::device_vector<float> din_right(in_right);

    thrust::device_vector<float> dout(tensor_size);

    const uint32_t* levels = thrust::raw_pointer_cast(&device_powers[0]);
    float* pd_out = thrust::raw_pointer_cast(&dout[0]);
    const float* pd_lhs = thrust::raw_pointer_cast(&din_left[0]);
    const float* pd_rhs = thrust::raw_pointer_cast(&din_right[0]);

    dim3 threads_per_block(32, 32);
    dim3 blocks (round_up_div(powers.back(), threads_per_block.x), round_up_div(powers.back(), threads_per_block.y));

    std::cout << "Blocks: " << blocks.x << ' ' << blocks.y << '\n';

    std::chrono::high_resolution_clock clk;
    auto start = clk.now();
    ft_mul_kernel<<<blocks, threads_per_block>>>(pd_out, pd_lhs, pd_rhs, DEPTH, levels);
    cudaDeviceSynchronize();
    auto end = clk.now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time: " << time.count() << '\n';

    thrust::host_vector<float> result(dout);

    thrust::host_vector<float> expected(tensor_size);
    ft_mul_host(expected.data(), in_left.data(), in_right.data(), DEPTH, powers.data());

    float err = 0.0f;
    float newerr;
    for (int32_t i=0; i<tensor_size; ++i) {
        if ((newerr = abs(expected[i]  - result[i])) > err) {
            std::cout << i << ' ' << expected[i] << ' ' << result[i] << ' ' << newerr << '\n';
            err = newerr;
        }
    }
    std::cout << "Max error: " << err << '\n';

    return 0;
}
