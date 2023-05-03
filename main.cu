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


__host__ __device__ inline  int32_t compute_offset(const int32_t* levels, int32_t level) {
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
              const int32_t* levels) {

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
              const int32_t* levels) {

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

//            __syncthreads();
            auto ix = x_offset;
            auto iy = y_offset;
            if (ix < lhs_n && iy < rhs_n) {
                out_p[ix * rhs_n + iy] += lhs_p[ix]*rhs_p[iy];
//                __syncthreads();
            }
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

template <typename T>
using rp_t = T* __restrict;
template <typename T>
using crp_t = const T* __restrict;

template <typename T>
struct WriteTensorData {
    rp_t<T> fwd_data;
    rp_t<T> rev_data;
};

template <typename T>
struct ReadTensorData {
    crp_t<T> fwd_read;
    crp_t<T> rev_read;
};

struct ComputeInfo {
    int32_t width;
    int32_t depth;
    int32_t tile_letters;
    const int32_t* levels;
    const int32_t* reverse_letters;
};


struct DivRem {
    int div;
    int rem;
};


__device__ __host__ DivRem divide(int idx, int divisor) {

    DivRem result;
    result.div = idx / divisor;
    result.rem = (idx - result.div * divisor);

    return result;
}


template <typename T>
__global__ void ft_tiled_mul(WriteTensorData<T> out,
                             ReadTensorData<T> lhs,
                             ReadTensorData<T> rhs,
                             ComputeInfo info) {
    const auto& xi = threadIdx.x;
    const auto& yi = threadIdx.y;
    const auto grid_x = gridDim.x * blockDim.x;
    const auto grid_y = gridDim.y * blockDim.y;
    const auto& tile_width = info.levels[info.tile_letters];

    const auto tile_idx = xi*blockDim.x + yi;

    auto get_offset = [&info] (int32_t level, int32_t offset) -> int32_t {
        auto level_offset = compute_offset(info.levels, level);
        return level_offset + offset;
    };


    extern __shared__ T tile[];   // size blockDim.x * blockDim.y
    const auto tile_size = tile_width * tile_width;

    T lhs_val = 0;
    T rhs_val = 0;

    for (int32_t out_deg=info.depth; out_deg >= 2*info.tile_letters; --out_deg) {
        const auto mid_deg = out_deg - 2 * info.tile_letters;
        const auto& mid_stride = info.levels[mid_deg];


        for (int32_t mid_idx=0; mid_idx < info.levels[mid_deg]; ++mid_idx) {
            const auto mid_ridx = reverse_idx(mid_idx, info.width, mid_deg);

            tile[tile_idx] = 0;
            __syncthreads();

            for (int32_t lhs_deg = 1; lhs_deg < info.tile_letters; ++lhs_deg) {
                auto rhs_deg = out_deg - lhs_deg;

                lhs_val = 0;
                rhs_val = 0;

                const auto& remainder_bound = info.levels[info.tile_letters + rhs_deg];

                auto split = divide(xi, remainder_bound);
                if (xi < tile_width && yi < tile_width) {
                    lhs_val = lhs.fwd_read[get_offset(lhs_deg, 0)];
                    rhs_val = rhs.fwd_read[get_offset(rhs_deg, (split.rem*mid_stride + mid_idx)*tile_width) + yi];
                }

                tile[tile_idx] += lhs_val*rhs_val;
            }

            for (int32_t lhs_mid_deg = 0; lhs_mid_deg <= mid_deg; ++lhs_mid_deg) {
                auto rhs_mid_deg = mid_deg - lhs_mid_deg;

                lhs_val = 0;
                rhs_val = 0;

                auto split = divide(mid_idx, info.levels[rhs_mid_deg]);
                if (xi < tile_width) {
                    lhs_val = lhs.fwd_read[get_offset(lhs_mid_deg + info.tile_letters,
                                           xi*info.levels[lhs_mid_deg] + split.div)];
                }
                if (yi < tile_width) {
                    rhs_val = rhs.fwd_read[get_offset(rhs_mid_deg + info.tile_letters,
                                           split.rem*tile_width + yi)];
                }

                tile[tile_idx] += lhs_val * rhs_val;
            }

            for (int32_t rhs_deg = 1; rhs_deg < info.tile_letters; ++rhs_deg) {
                auto lhs_deg = out_deg - rhs_deg;

                const auto small_bound = info.levels[lhs_deg];
                const auto &remainder_bound = info.levels[rhs_deg];
                lhs_val = 0;
                rhs_val = 0;

                auto split = divide(xi, remainder_bound);
                if (xi < tile_width && yi < tile_width) {
                    rhs_val = rhs.fwd_read[get_offset(rhs_deg,
                                                      (split.rem * mid_stride + mid_idx) * tile_width + yi)];
                    lhs_val = lhs.fwd_read[get_offset(lhs_deg, split.div)];
                }

                tile[tile_idx] += lhs_val * rhs_val;
            }

            if (xi < tile_width && yi < tile_width) {
                out.fwd_data[get_offset(out_deg, (xi*mid_stride + mid_idx)*tile_width + yi)]
                    += tile[tile_idx];
            }
        }





    }


}


int main() {

    thrust::host_vector<int32_t> powers;
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

    thrust::device_vector<int32_t> device_powers(powers);
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

    const int32_t* levels = thrust::raw_pointer_cast(&device_powers[0]);
    float* pd_out = thrust::raw_pointer_cast(&dout[0]);
    const float* pd_lhs = thrust::raw_pointer_cast(&din_left[0]);
    const float* pd_rhs = thrust::raw_pointer_cast(&din_right[0]);

    dim3 threads_per_block(32, 32);
    dim3 blocks { 1 };
    auto shared_size = threads_per_block.x * threads_per_block.y * sizeof(float);


    std::cout << "Blocks: " << blocks.x << ' ' << blocks.y << '\n';



    std::chrono::high_resolution_clock clk;
    auto start = clk.now();
    ft_mul_kernel<<<blocks, threads_per_block>>>(pd_out, pd_lhs, pd_rhs, DEPTH, levels);
//    ft_tiled_mul<float><<<blocks, threads_per_block, shared_size>>>(
        {pd_out, nullptr}, {pd_lhs, nullptr}, {pd_rhs, nullptr},
        {WIDTH, DEPTH, 2, levels, nullptr}
    );
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
