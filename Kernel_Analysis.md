# CUDA Kernel Performance & Bottleneck Analysis

## The Core Problem: Memory Bandwidth

Both Batch Normalisation and Depthwise 2D Convolution suffer from low arithmetic intensity during forward inference: they perform few floating-point operations per byte moved between DRAM and the SMs.

On modern GPUs, runtime is governed by global-memory traffic, coalescing into 128-byte cache-line transactions, occupancy, and minimising synchronisation overhead.

---

## 1. Batch Normalisation Inference

### The Bottleneck
Per the COMP70114 specification, Part 1 computes per-channel batch mean and variance on the fly across $(N, H, W)$ and applies per-channel affine scaling $(\gamma, \beta)$. Separating reduction and normalisation into multiple kernel launches forces partial statistics and unnormalised tensors to round-trip through global memory.

### How the Kernel Solves It
In [`submission/gpu_bn_inference.cu`](submission/gpu_bn_inference.cu), the entire operation runs inside a **single fused kernel launch** performing two passes over `x`:

* **One Block Per Channel (`BLOCK_SIZE = 1024`):** Grid dimensions `(C, 1, 1)` map each block to one channel, letting the block process all $(N, H, W)$ elements for that channel independently with zero global atomics.
* **L1/L2 Cache Reuse Across Passes:** Pass 1 reads `x` to accumulate per-thread `sum` and `sum_sq`, synchronises via `__syncthreads()`, and Pass 2 re-reads `x` to normalise and write `y`. For channels whose per-batch/channel footprint fits in cache (e.g., 64 KB across batch for LLaMA 70B `H=1, W=4096`), Pass 2 benefits from L1/L2 residency while avoiding extra temporary allocations.
* **128-Bit Vectorized Loads & Stores (`float4`):** Reinterpreting row pointers as `const float4*` / `float4*` moves 16 bytes per thread per instruction for coalesced bus transactions (assuming 16-byte alignment when `H * W % 4 == 0`, satisfied by all 4 benchmark shapes, with a scalar cleanup loop for general spatial sizes).
* **Register Accumulation & Warp Shuffles:** Each thread accumulates `local_sum` and `local_sq` in registers and reduces across warps using `__shfl_down_sync`. Only 32 floats per reduction touch shared memory, eliminating bank conflicts.
* **Fused Scale-Shift Math:** Thread 0 computes $\text{scale} = \gamma \cdot \text{rsqrtf}(\sigma^2 + \varepsilon)$ and $\text{shift} = \beta - \text{scale} \cdot \mu$, turning Pass 2 into a single FMA per element ($y = \text{scale} \cdot x + \text{shift}$). *(Trade-off: computing variance via $E[X^2] - E[X]^2$ in FP32 uses one reduction pair per element, whereas Welford's online algorithm would offer better numerical stability on high-mean, low-variance inputs.)*

---

## 2. Depthwise 2D Convolution

### The Bottleneck
Unlike standard 2D convolution, depthwise convolution removes cross-channel reuse: each channel $c$ is convolved independently with its own $K_H \times K_W$ filter. Low data reuse makes coalesced global reads and low launch/memory-allocation overhead critical.

### How the Kernel Solves It
[`submission/gpu_dwconv2d_inference.cu`](submission/gpu_dwconv2d_inference.cu) implements a straightforward **one-thread-per-output-pixel** direct baseline (`BLOCK_SIZE = 256`):

* **3D Grid Mapping:** Launches `(ceil(OH*OW / 256), C, N)` blocks so every thread writes a unique `y[n, c, oh, ow]` element with no atomics.
* **Coalesced Spatial Memory Access:** Adjacent threads along `threadIdx.x` map to consecutive `ow` indices, producing contiguous global reads along input width when `stride_w = 1`.
* **Implicit Zero-Padding:** Inline coordinate checks (`ih >= 0 && ih < H && iw >= 0 && iw < W`) treat out-of-bounds reads as zero in registers, eliminating padded buffer allocation and copy overhead.
* **Register Accumulation:** Accumulates the $K_H \times K_W$ dot product in a register `sum` before writing once to `y`. *(No shared-memory tiling or `float4` vectorisation is used in this baseline.)*

---

## 3. Timing Methodology, Hardware & Verified Benchmarks

### Evaluation Environment & Rules
* **Hardware:** Coursework development ran on Imperial College lab workstations (NVIDIA GeForce RTX 4080, `70114_1_spec.pdf` §4.1). Final grading and profiling ran on the COMP70114 online judge equipped with an **NVIDIA GeForce RTX 5090** (1,792 GB/s theoretical peak memory bandwidth, CUDA 13.0, Driver 580.82.07 / 580.95.05, confirmed in `70114_1_spec.pdf` §4.2 and `.nsys-rep` traces).
* **Timer Scope:** Both harnesses (`test_bn.cu`, `test_dwconv2d.cu`) generate FP32 inputs directly on GPU (`curandGenerateUniform`, seed `42`) and time only `solve(...)` using `cudaEventRecord` / `cudaEventElapsedTime` divided by `iters`. Host-to-device and device-to-host transfers are excluded.
* **What OJ "Time Cost" Measures:** Scored `./main` runs `warmup = 5, iters = 10` per test shape and sums the per-shape averages into one **Time Cost**. Nsight Systems (`./main_ncu`) runs `warmup = 0, iters = 1`.
* **Baselines:** External libraries (cuDNN, PyTorch) were prohibited by the specification. The CPU references (`cpu_bn_inference_nchw_ref`, `cpu_dwconv2d_ref`) validate correctness only (within `1e-5` tolerance) and are **never timed**. The only GPU performance baseline in the repository is the iterative progression of the BatchNorm CUDA kernel across commits on the same RTX 5090 judge.

### Batch Normalisation: Iterative Progression (RTX 5090 OJ, 4/4 Passed)

| Commit / Submission | Kernel Configuration | OJ Time Cost | Speedup vs Initial CUDA |
| :--- | :--- | :---: | :---: |
| `b5fb3d8` | `BLOCK_SIZE = 256`, branched `H*W % 4 == 0` | **1.316 ms** | 1.00× (baseline) |
| `b02c4db` | `BLOCK_SIZE = 256`, unified `float4` + remainder loop | **1.299 ms** | 1.01× |
| `f45a2ac` (Sub `#3739`) | `BLOCK_SIZE = 1024`, unified `float4` + remainder loop | **0.983 ms** | 1.34× |
| Final (Sub `#3869`) | Identical executable code to `#3739` (comments only) | **0.981 ms** | **1.34× (25.5% reduction)** |

### Batch Normalisation: Per-Shape Nsight Systems Trace (`#3739`, `Warmup = 0, Iters = 1`)

Extracted from `assets/submission_3739_nsys_profile.nsys-rep` on RTX 5090 (harness formula counts 1 read of `x` + `gamma`/`beta` + 1 write of `y`):

| Shape | Elements | Latency (1 iter) | Effective Bandwidth | % of RTX 5090 Peak (1,792 GB/s) |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-50 stem `(16, 64, 112, 112)` | 12,845,056 | 0.103 ms | 994.51 GB/s | 55.5% (cold first case) |
| MobileNetV2 early `(32, 32, 112, 112)` | 12,845,056 | 0.067 ms | 1,543.14 GB/s | 86.1% |
| LLaMA/Mistral 7B `(4, 4096, 1, 2048)` | 33,554,432 | 0.168 ms | 1,597.72 GB/s | 89.2% |
| LLaMA 70B width `(4, 8192, 1, 4096)` | 134,217,728 | 0.705 ms | 1,522.60 GB/s | 85.0% |
| **Sum (Un-warmed `Iters = 1`)** | **193,462,272** | **1.043 ms** | — | **Scored `./main` (`warmup=5, iters=10`): 0.981 ms** |

### Depthwise Conv2D: Per-Shape Nsight Systems Trace (`#3747`, `Warmup = 0, Iters = 1`)

Extracted from the `sub_3747` Nsight trace (`assets/Unconfirmed 441338.crdownload`) on RTX 5090 (5/5 passed within `1e-5` of `cpu_dwconv2d_ref`):

| Shape | Kernel / Pad / Stride | Latency (1 iter) | Throughput | Effective Bandwidth |
| :--- | :--- | :---: | :---: | :---: |
| `(32, 32, 112, 112)` → `56×56` | `3×3`, pad `(1,1)`, stride `(2,2)` | 0.077 ms | 0.75 TFLOPS/s | 830.40 GB/s |
| `(32, 64, 56, 56)` → `56×56` | `3×3`, pad `(1,1)`, stride `(1,1)` | 0.045 ms | 2.55 TFLOPS/s | 1,133.97 GB/s (63.3% peak) |
| `(8, 1024, 14, 14)` → `14×14` | `3×3`, pad `(1,1)`, stride `(1,1)` | 0.016 ms | 1.79 TFLOPS/s | 797.15 GB/s |
| `(16, 128, 28, 28)` → `28×28` | `5×5`, pad `(2,2)`, stride `(1,1)` | 0.022 ms | 3.66 TFLOPS/s | 585.73 GB/s |
| `(16, 128, 28, 28)` → `14×14` | `5×5`, pad `(2,2)`, stride `(2,2)` | 0.009 ms | 2.13 TFLOPS/s | 854.69 GB/s |
| **Sum (Un-warmed `Iters = 1`)** | — | **0.170 ms** | — | **Scored `./main` (`warmup=5, iters=10`): 0.098 ms** |
