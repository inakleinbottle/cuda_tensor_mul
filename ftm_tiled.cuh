//
// Created by sam on 08/05/23.
//

#ifndef CUDA_TENSOR_MUL_FTM_TILED_H
#define CUDA_TENSOR_MUL_FTM_TILED_H

#include "common.cuh"
#include "ftm_cuda_simple.cuh"


template <typename T>
struct WriteTensorData {
    T* fwd_data;
    T* rev_data;
};

template <typename T>
struct ReadTensorData {
    const T* fwd_read;
    const T* rev_read;
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


    auto get_offset = [&info](int32_t level, int32_t offset) -> int32_t {
        auto level_offset = compute_offset(info.levels, level);
        return level_offset + offset;
    };
    const auto tile_size = tile_width * tile_width;

    T lhs_val = 0;
    T rhs_val = 0;
    T out_val = 0;

    for (int32_t out_deg = info.depth; out_deg >= 2 * info.tile_letters; --out_deg) {
        const auto mid_deg = out_deg - 2 * info.tile_letters;
        const auto &mid_stride = info.levels[mid_deg];
        __syncthreads();
        for (uint32_t mid_idx = blockIdx.x; mid_idx < info.levels[mid_deg]; mid_idx += blockDim.x) {
//            const auto mid_ridx = reverse_idx(mid_idx, info.width, mid_deg);


            out_val = 0;
            if (xi < tile_width && yi < tile_width) {
                const auto& lhs_unit = lhs.fwd_read[0];
                const auto& rhs_unit = rhs.fwd_read[0];
                auto offset = get_offset(out_deg, (xi * mid_stride + mid_idx) * tile_width + yi);
                lhs_val = lhs.fwd_read[offset];
                rhs_val = rhs.fwd_read[offset];
                out_val += lhs_unit*rhs_val + lhs_val*rhs_unit;
            }

            for (int32_t lhs_deg = 1; lhs_deg < info.tile_letters; ++lhs_deg) {
                auto rhs_deg = out_deg - lhs_deg;

                lhs_val = 0;
                rhs_val = 0;

                const auto small_bound = info.levels[lhs_deg];
                const auto &remainder_bound = info.levels[info.tile_letters + rhs_deg];

                auto split = divide(xi, small_bound);
                if (xi < tile_width && yi < tile_width) {
                    lhs_val = lhs.fwd_read[get_offset(lhs_deg, 0)];
                    rhs_val = rhs.fwd_read[get_offset(rhs_deg, (split.rem * mid_stride + mid_idx) * tile_width) + yi];
                }

                out_val += lhs_val * rhs_val;
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

                out_val += lhs_val * rhs_val;
            }

            for (int32_t rhs_deg = 1; rhs_deg < info.tile_letters; ++rhs_deg) {
                auto lhs_deg = out_deg - rhs_deg;

                const auto &remainder_bound = info.levels[rhs_deg];
                lhs_val = 0;
                rhs_val = 0;

                auto split = divide(xi, remainder_bound);
                if (xi < tile_width && yi < tile_width) {
                    rhs_val = rhs.fwd_read[get_offset(rhs_deg,
                                                      (split.rem * mid_stride + mid_idx) * tile_width + yi)];
                    lhs_val = lhs.fwd_read[get_offset(lhs_deg, split.div)];
                }

                out_val += lhs_val * rhs_val;
            }

            __syncthreads();
            if (xi < tile_width && yi < tile_width) {
                out.fwd_data[get_offset(out_deg, (xi * mid_stride + mid_idx) * tile_width + yi)]
                    += out_val;
            }
        }

    }


    // TODO: Handle lower degrees
    // ftm_mul_kernel_impl<T>(out.fwd_read, lhs.fwd_read, rhs.fwd_read, 2*info.tile_letters, info.levels);

}


#endif //CUDA_TENSOR_MUL_FTM_TILED_H
