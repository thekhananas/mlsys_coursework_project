#include "solve.h"
#include <cuda_runtime.h>

#define FULL_MASK 0xffffffffu
constexpr int THREADS_PER_WARP = 32;

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

    // TODO: Kernel Launch
    /*
    gpu_dwconv2d<<<kernel_config1, kernel_config2>>>(
        input, kernel, output,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        OH, OW);
   */
}

