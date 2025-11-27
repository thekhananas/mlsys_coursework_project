#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
#define BLOCK_SIZE 256

// Note: 1. External libraries are not allowed 
//       2. The solve function signature must keep unchanged 
//       3. Store the final result in the array y

// This method is for warp reduction
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// This method is for block reduction
__inline__ __device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid = threadIdx.x / THREADS_PER_WARP;

    val = warp_reduce_sum(val);

    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / THREADS_PER_WARP)) ? shared_mem[lane] : 0.0f;

    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps)
{
    // TODO: Your Kernel Implementation
    
    int c = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sdata[32];
    __shared__ float s_mean;
    __shared__ float s_var;

    int plane_size = H * W;
    int stride = C * plane_size; // Distance between batches in NCHW
    
    // Base pointers for this channel
    const float* x_in = x + c * plane_size;
    float* y_out = y + c * plane_size;

    // Step 1: Accumulate Sum and Sum-Squares
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    // Optimization: Check if we can use float4 (requires size divisible by 4)
    // Most standard benchmarks (like ResNet) have H*W divisible by 4
    if (plane_size % 4 == 0) {
        int plane_vec = plane_size / 4;
        
        for (int n = 0; n < N; ++n) {
            // Reinterpret pointer as float4 for vectorized load
            // Note: offset must be calculated in bytes/floats first
            const float4* x_vec = (const float4*)(x_in + n * stride);
            
            for (int i = tid; i < plane_vec; i += blockDim.x) {
                float4 v = x_vec[i];
                
                local_sum += v.x + v.y + v.z + v.w;
                local_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            }
        }
    } else {
        // Fallback for odd shapes
        for (int n = 0; n < N; ++n) {
            int batch_offset = n * stride;
            for (int i = tid; i < plane_size; i += blockDim.x) {
                float val = x_in[batch_offset + i];
                local_sum += val;
                local_sq += val * val;
            }
        }
    }

    float reduce_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = reduce_sum;
    __syncthreads();

    float reduce_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) s_var = reduce_sq;
    __syncthreads();

    // Step 2: Calculate Constants
    if (tid == 0) {
        // Use fast inverse approximate for division
        float inv_n = 1.0f / (float)(N * plane_size);
        float mean = s_mean * inv_n;
        float mean_sq = s_var * inv_n;
        float var = mean_sq - (mean * mean);

        float inv_std = rsqrtf(var + eps);
        
        // y = scale * x + shift
        s_mean = gamma[c] * inv_std;                 // scale
        s_var = beta[c] - (gamma[c] * mean * inv_std); // shift
    }
    __syncthreads();

    float scale = s_mean;
    float shift = s_var;

    // Step 3: Normalize and Write (Vectorized)
    if (plane_size % 4 == 0) {
        int plane_vec = plane_size / 4;
        for (int n = 0; n < N; ++n) {
            // Pointers for current batch
            const float4* x_vec = (const float4*)(x_in + n * stride);
            float4* y_vec = (float4*)(y_out + n * stride);

            for (int i = tid; i < plane_vec; i += blockDim.x) {
                float4 v = x_vec[i];
                float4 res;
                
                res.x = scale * v.x + shift;
                res.y = scale * v.y + shift;
                res.z = scale * v.z + shift;
                res.w = scale * v.w + shift;
                
                y_vec[i] = res;
            }
        }
    } else {
        // Fallback
        for (int n = 0; n < N; ++n) {
            int batch_offset = n * stride;
            for (int i = tid; i < plane_size; i += blockDim.x) {
                int idx = batch_offset + i;
                y_out[idx] = scale * x_in[idx] + shift;
            }
        }
    }
}

void solve(const float* input, 
           const float* gamma, 
           const float* beta, 
           float* output, 
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    
    // TODO: Kernel Configurabtions (dim3 block, dim3 grid)
    
    // One block per channel is sufficient
    dim3 grid(C, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
}