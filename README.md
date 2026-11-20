# CUDA Inference Kernels: Batch Normalisation & Depthwise Conv2D

High-performance, dependency-free CUDA C++ kernels developed for **COMP70114 (Machine Learning Systems & Hardware)** at Imperial College London.

> **Live Showcase & Kernel Summary:**  
> [**https://thekhananas.github.io/mlsys_coursework_project/**](https://thekhananas.github.io/mlsys_coursework_project/)  
> *(Includes animated execution-flow diagrams and verified RTX 5090 online-judge results)*

---

## Performance Snapshot

All kernels were implemented in pure CUDA C++ with zero external libraries (no cuDNN or PyTorch), checked for correctness against untimed CPU reference implementations within `1e-5` tolerance, and evaluated on the COMP70114 **NVIDIA GeForce RTX 5090** online judge (1,792 GB/s theoretical memory bandwidth; CUDA 13.0).

Reported **Time Cost** values are summed `cudaEvent` GPU kernel latencies across all test shapes (`warmup = 5, iters = 10` for scored `./main`, excluding host–device transfers):

| Operator | OJ Time Cost | Test Suite | Effective BW (Nsight 1-iter trace) | Baseline & Key Techniques |
| :--- | :---: | :---: | :---: | :--- |
| **Batch Normalisation** (Submission `#3869`) | **0.981 ms** | 4 / 4 passed (ResNet-50, MobileNetV2, LLaMA 7B/70B) | 995–1,598 GB/s (85.0–89.2% RTX 5090 peak on warmed shapes) | 1.34× speedup vs initial `1.316 ms` (`BLOCK_SIZE = 256`) CUDA commit; 1 block/channel (1,024 thr), `float4` I/O, warp shuffle, fused FMA |
| **Depthwise Conv2D** (Submission `#3747`) | **0.098 ms** | 5 / 5 passed (3×3 & 5×5 CNN shapes) | 586–1,134 GB/s (up to 63.3% RTX 5090 peak) | Direct 1-thread/pixel 3D-grid baseline (`BLOCK_SIZE = 256`), coalesced spatial reads, implicit zero-padding |

For per-shape breakdowns, timing methodology, and design trade-offs, read [**Kernel_Analysis.md**](Kernel_Analysis.md).

---

## Core Optimizations

### Batch Normalisation ([`submission/gpu_bn_inference.cu`](submission/gpu_bn_inference.cu))
* **One-Block-Per-Channel Grid (`BLOCK_SIZE = 1024`):** Eliminates global inter-block atomics by assigning each channel to an independent 1,024-thread block, improving latency by 25.5% (`1.316 ms` → `0.981 ms`) over the initial 256-thread configuration.
* **Single Fused Launch with L1/L2 Cache Reuse:** Executes two loops over `x` inside one kernel launch (Pass 1 accumulates sum and sum-of-squares; `__syncthreads()`; Pass 2 normalises and writes `y`), avoiding global intermediate buffers and benefiting from per-channel cache residency.
* **128-Bit Vectorized I/O:** Loads and stores memory using `float4` pointers (`H * W` divisible by 4 across all benchmark shapes, with a scalar remainder loop) to maximise memory coalescing.
* **Two-Stage Register Reduction:** Collapses per-thread running sums via `__shfl_down_sync` warp shuffles, using only 32 floats of shared memory and avoiding bank conflicts.
* **Fused Scale-Shift Transformation:** Computes normalisation coefficients upfront with `rsqrtf`, turning output generation into a single fused multiply-add ($y = \text{scale} \cdot x + \text{shift}$).

### Depthwise Conv2D ([`submission/gpu_dwconv2d_inference.cu`](submission/gpu_dwconv2d_inference.cu))
* **One-Thread-Per-Output-Pixel (`BLOCK_SIZE = 256`):** Maps threads across a 3D grid `(spatial_blocks, C, N)` to guarantee race-free, atomic-free writes.
* **Coalesced Spatial Indexing:** Indexes threads along the output width ($OW$) so adjacent threads read consecutive memory addresses when `stride = 1`.
* **Zero-Copy Implicit Padding:** Replaces padded buffer allocations and copies with inline boundary checks directly in registers.

---

## Repository Structure

```
├── docs/                     # GitHub Pages showcase
│   ├── index.html            # Editorial summary & execution-flow SVGs
│   ├── style.css             # Typography & layout styles
│   └── assets/               # Evaluation screenshots and technical report PDF
├── submission/               # Submitted CUDA kernels and academic report
│   ├── gpu_bn_inference.cu   # Batch Normalisation kernel
│   └── gpu_dwconv2d_inference.cu # Depthwise Conv2D kernel
├── mlsys_coursework/         # Development harnesses and test suites
│   ├── part1/                # BN tests (cpu reference, test harness, script)
│   └── part2/                # DWConv tests (cpu reference, test harness, script)
├── report/                   # LaTeX source and evaluation graphics
├── Kernel_Analysis.md        # Technical bottleneck, per-shape trace, and baseline breakdown
└── README.md
```

---

## Build & Run

Ensure CUDA Toolkit 12+ and `nvcc` are installed. Note: `-arch=sm_89` targets the Imperial lab development workstations (NVIDIA RTX 4080); adjust to `sm_120` (or the target GPU compute capability) for NVIDIA RTX 5090.

### Batch Normalisation
```bash
cd mlsys_coursework/part1
nvcc -O2 -std=c++14 -lcurand -arch=sm_89 \
  gpu_bn_inference.cu cpu_bn_inference_ref.cu test_bn.cu -o test_bn
./test_bn
```

### Depthwise Conv2D
```bash
cd mlsys_coursework/part2
nvcc -O2 -std=c++14 -lcurand -arch=sm_89 \
  gpu_dwconv2d_inference.cu cpu_dwconv2d_inference_ref.cu test_dwconv2d.cu -o test_dwconv2d
./test_dwconv2d
```

---

## Hosting on GitHub Pages

This repository is pre-configured with the `/docs` directory for GitHub Pages:

1. In your GitHub repository, navigate to **Settings** > **Pages** (under the "Code and automation" sidebar).
2. Under **Build and deployment** > **Source**, select **Deploy from a branch**.
3. Under **Branch**, select `main` and choose the `/docs` folder from the dropdown.
4. Click **Save**. Within 1–2 minutes, GitHub will publish the site at:
   `https://thekhananas.github.io/mlsys_coursework_project/`
