#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

#define BLOCK_SIZE 256

// Note: 1. External libraries are not allowed 
//       2. The solve function signature must keep unchanged 
//       3. Store the final result in the array y

__global__ void gpu_bn_inference_nchw(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps)
{
    // TODO: Your Kernel Implementation
    // Strategy: Each block works on one Channel (c)
    int c = blockIdx.x;
    
    // Shared memory to store partial sums from threads
    __shared__ double sdata[BLOCK_SIZE];
    
    // Shared variables so all threads can read the final mean/var
    __shared__ double s_mean;
    __shared__ double s_var;

    int tid = threadIdx.x;
    
    // Total pixels this channel has across all batches
    int num_elements = N * H * W;
    
    
    // Step 1: Calculate Sum to find Mean
    
    double sum = 0.0;

    // Iterate linearly over all pixels for this channel
    // Using a simple stride loop
    for (int i = tid; i < num_elements; i += blockDim.x) {
        // Retrieve N, H, W indices from the linear index 'i'
        // This makes the logic very clear, even if division is a bit slow
        int n = i / (H * W);
        int rem = i % (H * W);
        
        // Calculate global index: Batch_Offset + Channel_Offset + Pixel_Offset
        int global_idx = n * (C * H * W) + c * (H * W) + rem;
        
        sum += (double)x[global_idx];
    }
    
    // Load partial sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Naive Parallel Reduction (The "Modulo" approach)
    // Concept: sum sdata[tid] and sdata[tid + s] if tid is a multiple of 2*s.
    // This is the most intuitive way to learn reduction, though it has thread divergence.
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the final mean
    if (tid == 0) {
        s_mean = sdata[0] / (double)num_elements;
    }
    __syncthreads(); // Everyone wait for mean

    
    // Step 2: Calculate Variance
    
    double sum_sq = 0.0;

    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int global_idx = n * (C * H * W) + c * (H * W) + rem;
        
        double diff = (double)x[global_idx] - s_mean;
        
        // Using pow() is very readable for "squared"
        sum_sq += pow(diff, 2.0);
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduction again for variance
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_var = sdata[0] / (double)num_elements;
    }
    __syncthreads();

    
    // Step 3: Normalize and Write Output
    
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int global_idx = n * (C * H * W) + c * (H * W) + rem;

        double val = (double)x[global_idx];
        
        // Standard formula implementation
        // Using sqrt from math.h explicitly
        double std_dev = sqrt(s_var + (double)eps);
        
        double normalized = (val - s_mean) / std_dev;
        
        double result = normalized * (double)gamma[c] + (double)beta[c];
        
        y[global_idx] = (float)result;
    }
}

void solve(const float* input, 
           const float* gamma, 
           const float* beta, 
           float* output, 
           int N, int C, int H, int W, float eps)
{
    // Simple configuration: 1 block per channel
    dim3 grid(C, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);

    gpu_bn_inference_nchw<<<grid, block>>>(input, gamma, beta, output, N, C, H, W, eps);
}