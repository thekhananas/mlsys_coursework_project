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

    // Pre-calculate plane size and channel offsets
    int plane = H * W;
    int channel_offset = c * plane;
    
    // Step 1: Accumulate Sum and Sum-Squares
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    // Use float4 to load 4 elements at once for better bandwidth
    // We treat the pointer as a float4 array
    int vec_limit = plane / 4; 

    for (int n = 0; n < N; ++n) {
        int batch_offset = n * C * plane + channel_offset;
        const float* src = x + batch_offset;
        
        // Reinterpret cast to read 4 floats at a time
        const float4* src4 = (const float4*)src;

        // Vectorized loop
        for (int i = tid; i < vec_limit; i += blockDim.x) {
            float4 v = src4[i];
            
            local_sum += v.x + v.y + v.z + v.w;
            local_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }

        // Cleanup loop for remaining pixels if H*W is not divisible by 4
        int remainder_start = vec_limit * 4;
        for (int i = remainder_start + tid; i < plane; i += blockDim.x) {
            float val = src[i];
            local_sum += val;
            local_sq += val * val;
        }
    }

    // Reduce
    float reduce_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = reduce_sum;
    __syncthreads();

    float reduce_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) s_var = reduce_sq;
    __syncthreads();

    // Step 2: Normalize
    // Thread 0 calculates the fused scale/shift
    if (tid == 0) {
        float mean = s_mean / (N * plane);
        float mean_sq = s_var / (N * plane);
        float var = mean_sq - (mean * mean);
        
        float inv_std = rsqrtf(var + eps);
        
        s_mean = gamma[c] * inv_std;                   // scale
        s_var = beta[c] - (gamma[c] * mean * inv_std); // shift
    }
    __syncthreads();

    float scale = s_mean;
    float shift = s_var;

    // Apply normalization
    for (int n = 0; n < N; ++n) {
        int batch_offset = n * C * plane + channel_offset;
        const float* src = x + batch_offset;
        float* dst = y + batch_offset;
        
        const float4* src4 = (const float4*)src;
        float4* dst4 = (float4*)dst;

        // Vectorized write
        for (int i = tid; i < vec_limit; i += blockDim.x) {
            float4 v = src4[i];
            
            // Normalize all 4 components
            v.x = scale * v.x + shift;
            v.y = scale * v.y + shift;
            v.z = scale * v.z + shift;
            v.w = scale * v.w + shift;
            
            dst4[i] = v;
        }

        // Cleanup write
        int remainder_start = vec_limit * 4;
        for (int i = remainder_start + tid; i < plane; i += blockDim.x) {
            dst[i] = scale * src[i] + shift;
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