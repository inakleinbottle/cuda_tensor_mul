//
// Created by sam on 08/05/23.
//

#ifndef CUDA_TENSOR_MUL_FTM_TILED_H
#define CUDA_TENSOR_MUL_FTM_TILED_H

#include "common.cuh"
#include "ftm_cuda_simple.cuh"


template <typename T>
struct WriteTensorData {
    rp_t <T> fwd_data;
    rp_t <T> rev_data;
};

template <typename T>
struct ReadTensorData {
    crp_t <T> fwd_read;
    crp_t <T> rev_read;
};

struct ComputeInfo {
    int32_t width;
    int32_t depth;
    int32_t tile_letters;
    const int32_t *levels;
    const int32_t *reverse_letters;
};



template <typename T>
__global__ void ft_tiled_mul(WriteTensorData<T> out,
                             ReadTensorData<T> lhs,
                             ReadTensorData<T> rhs,
                             ComputeInfo info) {
    const auto &xi = threadIdx.x;
    const auto &yi = threadIdx.y;
    const auto grid_x = gridDim.x * blockDim.x;
    const auto grid_y = gridDim.y * blockDim.y;
    const auto &tile_width = info.levels[info.tile_letters];

    const auto tile_idx = xi * blockDim.x + yi;

    auto get_offset = [&info](int32_t level, int32_t offset) -> int32_t {
        auto level_offset = compute_offset(info.levels, level);
        return level_offset + offset;
    };

    extern __shared__ T
    tile[];   // size blockDim.x * blockDim.y
    const auto tile_size = tile_width * tile_width;

    T lhs_val = 0;
    T rhs_val = 0;

    for (int32_t out_deg = info.depth; out_deg >= 2 * info.tile_letters; --out_deg) {
        const auto mid_deg = out_deg - 2 * info.tile_letters;
        const auto &mid_stride = info.levels[mid_deg];

        for (int32_t mid_idx = 0; mid_idx < info.levels[mid_deg]; ++mid_idx) {
            const auto mid_ridx = reverse_idx(mid_idx, info.width, mid_deg);

            tile[tile_idx] = 0;
            __syncthreads();

            for (int32_t lhs_deg = 1; lhs_deg < info.tile_letters; ++lhs_deg) {
                auto rhs_deg = out_deg - lhs_deg;

                lhs_val = 0;
                rhs_val = 0;

                const auto &remainder_bound = info.levels[info.tile_letters + rhs_deg];

                auto split = divide(xi, remainder_bound);
                if (xi < tile_width && yi < tile_width) {
                    lhs_val = lhs.fwd_read[get_offset(lhs_deg, 0)];
                    rhs_val = rhs.fwd_read[get_offset(rhs_deg, (split.rem * mid_stride + mid_idx) * tile_width) + yi];
                }

                tile[tile_idx] += lhs_val * rhs_val;
            }

            for (int32_t lhs_mid_deg = 0; lhs_mid_deg <= mid_deg; ++lhs_mid_deg) {
                auto rhs_mid_deg = mid_deg - lhs_mid_deg;

                lhs_val = 0;
                rhs_val = 0;

                auto split = divide(mid_idx, info.levels[rhs_mid_deg]);
                if (xi < tile_width) {
                    lhs_val = lhs.fwd_read[get_offset(lhs_mid_deg + info.tile_letters,
                                                      xi * info.levels[lhs_mid_deg] + split.div)];
                }
                if (yi < tile_width) {
                    rhs_val = rhs.fwd_read[get_offset(rhs_mid_deg + info.tile_letters,
                                                      split.rem * tile_width + yi)];
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
                out.fwd_data[get_offset(out_deg, (xi * mid_stride + mid_idx) * tile_width + yi)]
                    += tile[tile_idx];
            }
        }

    }


    // TODO: Handle lower degrees
    // ftm_mul_kernel_impl<T>(out.fwd_read, lhs.fwd_read, rhs.fwd_read, 2*info.tile_letters, info.levels);

}


#endif //CUDA_TENSOR_MUL_FTM_TILED_H