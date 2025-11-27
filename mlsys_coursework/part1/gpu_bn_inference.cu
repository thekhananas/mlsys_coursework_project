#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
#define BLOCK_SIZE 256

// Note: 1. External libraries are not allowed 
//       2. The solve function signature must keep unchanged 
//       3. Store the final result in the array y

// Warp-level reduction using shuffle instructions (optimized for float)
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Block-level reduction using Shared Memory + Warp Shuffle
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

    // Shared memory for reduction (float is faster than double)
    __shared__ float sdata[32];
    __shared__ float s_mean;
    __shared__ float s_var;

    // Pointers for this channel
    // x and y are flattened, so we offset by c * (H * W) but stride by C * H * W
    int plane_size = H * W;
    int stride = C * plane_size;
    const float* x_in = x + c * plane_size;
    float* y_out = y + c * plane_size;

    // Step 1: Accumulate Sum and Sum-Squares in ONE pass
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int n = 0; n < N; ++n) {
        // Offset for the current batch
        int batch_offset = n * stride;

        for (int i = tid; i < plane_size; i += blockDim.x) {
            float val = x_in[batch_offset + i];
            local_sum += val;
            local_sq += val * val;
        }
    }

    // Reduce both statistics
    // We can reuse sdata since we syncthreads in between
    float reduce_sum = block_reduce_sum(local_sum, sdata);
    if (tid == 0) s_mean = reduce_sum;
    __syncthreads();

    float reduce_sq = block_reduce_sum(local_sq, sdata);
    if (tid == 0) s_var = reduce_sq;
    __syncthreads();

    // Step 2: Calculate Mean/Var and Normalize
    // Only thread 0 does the math to avoid redundant work
    // We store the 'scale' and 'shift' in shared memory to broadcast
    if (tid == 0) {
        float mean = s_mean / (N * plane_size);
        float mean_sq = s_var / (N * plane_size);
        float var = mean_sq - (mean * mean);

        float inv_std = rsqrtf(var + eps); // rsqrtf is a hardware intrinsic, faster than 1.0/sqrt
        
        // y = gamma * (x - mean) * inv_std + beta
        // y = (gamma * inv_std) * x + (beta - gamma * mean * inv_std)
        s_mean = gamma[c] * inv_std;                            // Reuse s_mean as 'scale'
        s_var = beta[c] - (gamma[c] * mean * inv_std);          // Reuse s_var as 'shift'
    }
    __syncthreads();

    float scale = s_mean;
    float shift = s_var;

    // Final pass to write output
    for (int n = 0; n < N; ++n) {
        int batch_offset = n * stride;
        
        // Simple unroll hint for the compiler
        #pragma unroll
        for (int i = tid; i < plane_size; i += blockDim.x) {
            int idx = batch_offset + i;
            y_out[idx] = scale * x_in[idx] + shift;
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
    
    dim3 grid(C, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
}