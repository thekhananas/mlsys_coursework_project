#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

// Note: 1. External libraries are not allowed 
//       2. The solve function signature must keep unchanged 
//       3. Store the final result in the array y

// Warp-level reduction to sum values within a warp
__inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Block-level reduction to sum values across the entire block
// Requires shared memory buffer of size [32] (max warps per block)
__inline__ __device__ double block_reduce_sum(double val, double* shared_mem) {
    int lane = threadIdx.x % THREADS_PER_WARP;
    int wid = threadIdx.x / THREADS_PER_WARP;

    // 1. Reduce within warp
    val = warp_reduce_sum(val);

    // 2. Write warp results to shared memory
    if (lane == 0) shared_mem[wid] = val;
    __syncthreads();

    // 3. Read back warp results and perform final reduce in the first warp
    // Assuming blockDim.x <= 1024 (max 32 warps)
    val = (threadIdx.x < blockDim.x / THREADS_PER_WARP) ? shared_mem[lane] : 0.0;
    
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// Single kernel implementation that performs Mean, Var, and Normalization passes
// Strategy: One Block per Channel.
__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ gamma,  // [C]
    const float* __restrict__ beta,   // [C]
    float* __restrict__ y,            // [N,C,H,W]
    int N, int C, int H, int W,
    float eps)
{
    // TODO: Your Kernel Implementation
    // Identify the channel this block is responsible for
    int c = blockIdx.x;
    if (c >= C) return;

    // Shared memory for block reduction (needs 32 doubles for max 1024 threads)
    __shared__ double s_reduce[32];
    // Shared memory to broadcast computed stats to all threads in block
    __shared__ double s_mean;
    __shared__ double s_var;

    int tid = threadIdx.x;
    int plane_size = H * W;
    size_t batch_stride = (size_t)C * plane_size; // Stride to get to next N

    // Pointers to the specific channel data
    const float* x_c = x + (size_t)c * plane_size;
    float* y_c = y + (size_t)c * plane_size;

    // -------------------------------------------------------------------------
    // Pass 1: Calculate Mean
    // -------------------------------------------------------------------------
    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
        // Coalesced access over H*W plane
        for (int hw = tid; hw < plane_size; hw += blockDim.x) {
            size_t idx = (size_t)n * batch_stride + hw;
            sum += (double)x_c[idx];
        }
    }

    sum = block_reduce_sum(sum, s_reduce);

    if (tid == 0) {
        s_mean = sum / (double)(N * plane_size);
    }
    __syncthreads(); // Wait for s_mean to be ready

    // -------------------------------------------------------------------------
    // Pass 2: Calculate Variance
    // -------------------------------------------------------------------------
    double mean = s_mean;
    double sum_sq = 0.0;

    for (int n = 0; n < N; ++n) {
        for (int hw = tid; hw < plane_size; hw += blockDim.x) {
            size_t idx = (size_t)n * batch_stride + hw;
            double diff = (double)x_c[idx] - mean;
            sum_sq += diff * diff;
        }
    }

    sum_sq = block_reduce_sum(sum_sq, s_reduce);

    if (tid == 0) {
        s_var = sum_sq / (double)(N * plane_size);
    }
    __syncthreads(); // Wait for s_var to be ready

    // -------------------------------------------------------------------------
    // Pass 3: Normalize and Write Output
    // -------------------------------------------------------------------------
    double var = s_var;
    
    // Precompute scale and shift parameters
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // y = [gamma * inv_std] * x + [beta - gamma * mean * inv_std]
    
    // Note: Using 1.0/sqrt matches the math. rsqrt is typically float precision.
    // We use double precision for the inverse std dev calculation to match ref.
    double inv_std = 1.0 / sqrt(var + (double)eps); 
    
    double d_gamma = (double)gamma[c];
    double d_beta = (double)beta[c];

    // Compute fused scale and shift for efficiency
    float scale = (float)(d_gamma * inv_std);
    float shift = (float)(d_beta - d_gamma * mean * inv_std);

    for (int n = 0; n < N; ++n) {
        for (int hw = tid; hw < plane_size; hw += blockDim.x) {
            size_t idx = (size_t)n * batch_stride + hw;
            y_c[idx] = scale * x_c[idx] + shift;
        }
    }
}

void solve(const float* input, // [N,C,H,W] 
           const float* gamma, // [C] 
           const float* beta, // [C]
           float* output, // [N,C,H,W]
           int N, int C, int H, int W, float eps)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;

    // TODO: Kernel Configurabtions (dim3 block, dim3 grid)
    
    // Configuration: One block per channel
    // This allows us to keep mean/var in shared memory without global allocation
    dim3 grid(C);
    dim3 block(256);

    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
}