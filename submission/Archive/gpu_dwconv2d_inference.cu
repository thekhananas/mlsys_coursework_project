#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;
#define BLOCK_SIZE 256

// Note: 1. External libraries are not allowed
//       2. The solve function signature must keep unchanged
//       3. Store the final result in the array y

__global__ void gpu_dwconv2d(
    const float* __restrict__ x,      // [N,C,H,W]
    const float* __restrict__ k,      // [C,1,KH,KW] flattened as [C*KH*KW]
    float* __restrict__ y,            // [N,C,OH,OW]
    int N, int C, int H, int W,
    int KH, int KW,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int OH, int OW)
{
    // TODO: Your Kernel Implementation
    
    
    int n = blockIdx.z;
    int c = blockIdx.y;

    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We only need to guard the linear spatial index
    if (tid < OH * OW) {
        // Recover output height and width indices from linear index
        int oh = tid / OW;
        int ow = tid % OW;

        float sum = 0.0f;

        // Depthwise convolution: Iterate over kernel window. 
        // Note: For dwconv, there is only 1 channel in the kernel per input channel.
        for(int kh = 0; kh < KH; ++kh) {
            for(int kw = 0; kw < KW; ++kw) {
                // Calculate input indices based on stride and padding
                int ih = oh * stride_h - pad_h + kh;
                int iw = ow * stride_w - pad_w + kw;

                // Boundary check for input image (padding is effectively zero-val)
                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // Calculate 1D flat indices
                    // Input x: [n, c, ih, iw]
                    // Size of one batch: C*H*W, Size of one channel: H*W
                    int x_idx = n * (C * H * W) + c * (H * W) + ih * W + iw;

                    // Kernel k: [c, 0, kh, kw] (effectively [C, KH, KW])
                    // Size of one kernel filter: KH*KW
                    int k_idx = c * (KH * KW) + kh * KW + kw;

                    sum += x[x_idx] * k[k_idx];
                }
            }
        }

        // Output y: [n, c, oh, ow]
        int y_idx = n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow;
        y[y_idx] = sum;
    }
}

void solve(const float* input,      // [N,C,H,W] 
           const float* kernel,     // [C,1,KH,KW] flattened as [C*KH*KW]
           float* output,           // [N,C,OH,OW]
           int N, int C, int H, int W, // input shape: batch, channel, height, weight
           int KH, int KW, // kernel height and weight
           int pad_h, int pad_w,    // kernel padding
           int stride_h, int stride_w)  // kernel stride
{
    if (N<=0 || C<=0 || H<=0 || W<=0 || KH<=0 || KW<=0) return;

    int OH = (H + 2*pad_h - KH) / stride_h + 1;
    int OW = (W + 2*pad_w - KW) / stride_w + 1;
    if (OH <= 0 || OW <= 0) return;

   
    // TODO: Kernel Configurabtions (dim3 block, dim3 grid)
    
    
    dim3 blockSize(BLOCK_SIZE);

    // Calculate total spatial pixels to process per channel per batch
    int total_spatial = OH * OW;
    
    
    // x: Spatial pixels (enough blocks to cover OH*OW)
    // y: Channels (C)
    // z: Batch size (N)
    dim3 gridSize((total_spatial + blockSize.x - 1) / blockSize.x, C, N);

    // Kernel Launch
    gpu_dwconv2d<<<gridSize, blockSize>>>(
        input, kernel, output,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        OH, OW);
}