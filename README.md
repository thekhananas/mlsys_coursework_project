# CUDA Inference Kernels: Batch Normalisation & Depthwise Conv2D

High-performance, dependency-free CUDA C++ kernels developed for **COMP70114 (Machine Learning Systems & Hardware)** at Imperial College London.

> **Live Interactive Showcase & Hardware Simulator:**  
> [**https://thekhananas.github.io/mlsys_coursework_project/**](https://thekhananas.github.io/mlsys_coursework_project/)  
> *(Features interactive thread-execution visualizers and live memory bandwidth profiler)*

---

## Performance Snapshot

All kernels were evaluated against CPU reference implementations across vision and language model tensor shapes (ResNet-50, MobileNetV2, and LLaMA dimensions) with strict zero-external-library constraints.

| Operator | Latency | Test Suite | Effective Bandwidth | Key Techniques |
| :--- | :---: | :---: | :---: | :--- |
| **Batch Normalisation** | **0.981 ms** | 4 / 4 passed (100%) | 742 GB/s (84.3% RTX 4080 peak) | 1 block/channel, `float4` loads, warp shuffle reduction, fused FMA |
| **Depthwise Conv2D** | **0.098 ms** | 5 / 5 passed (100%) | 620 GB/s (70.5% RTX 4080 peak) | 1 thread/pixel, coalesced spatial access, implicit zero-padding |

For an in-depth breakdown of bottlenecks and design decisions, read [**KERNEL_ANALYSIS.md**](KERNEL_ANALYSIS.md).

---

## Core Optimizations

### Batch Normalisation ([`submission/gpu_bn_inference.cu`](submission/gpu_bn_inference.cu))
* **One-Block-Per-Channel Grid:** Eliminates global inter-block atomics by assigning each channel to an independent 1,024-thread block.
* **128-Bit Vectorized I/O:** Loads and stores memory using `float4` pointers to saturate bus lines and maximize memory coalescing.
* **Two-Stage Register Reduction:** Collapses per-thread running sums via `__shfl_down_sync` warp shuffles, using only 32 floats of shared memory and avoiding bank conflicts.
* **Fused Scale-Shift Transformation:** Computes normalization coefficients upfront with `rsqrtf`, turning output generation into a single fused multiply-add ($y = ax + b$).

### Depthwise Conv2D ([`submission/gpu_dwconv2d_inference.cu`](submission/gpu_dwconv2d_inference.cu))
* **One-Thread-Per-Output-Pixel:** Maps threads across a 3D grid `(spatial_blocks, C, N)` to guarantee race-free, atomic-free writes.
* **Coalesced Spatial Indexing:** Indexes threads along the output width ($OW$) so adjacent threads read consecutive memory addresses when `stride = 1`.
* **Zero-Copy Implicit Padding:** Replaces padded buffer allocations and copies with inline boundary checks directly in registers.

---

## Repository Structure

```
├── docs/                     # Industry-standard GitHub Pages showcase
│   ├── index.html            # Web showcase (Steve Krug / Apple-inspired UI)
│   ├── styles.css            # Dark command-center theme & typography
│   ├── app.js                # Canvas visualizer & interactive tensor profiler
│   └── assets/               # Evaluation screenshots and technical report PDF
├── submission/               # Submitted CUDA kernels and academic report
│   ├── gpu_bn_inference.cu   # Batch Normalisation kernel
│   └── gpu_dwconv2d_inference.cu # Depthwise Conv2D kernel
├── mlsys_coursework/         # Development harnesses and test suites
│   ├── part1/                # BN tests (cpu reference, test harness, script)
│   └── part2/                # DWConv tests (cpu reference, test harness, script)
├── report/                   # LaTeX source and evaluation graphics
├── KERNEL_ANALYSIS.md        # Technical bottleneck and speedup breakdown
└── README.md
```

---

## Build & Run

Ensure CUDA Toolkit 12+ and `nvcc` are installed.

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

This repository is pre-configured with the industry-standard `/docs` directory for GitHub Pages:

1. In your GitHub repository, navigate to **Settings** > **Pages** (under the "Code and automation" sidebar).
2. Under **Build and deployment** > **Source**, select **Deploy from a branch**.
3. Under **Branch**, select `main` and choose the `/docs` folder from the dropdown.
4. Click **Save**. Within 1–2 minutes, GitHub will publish the site at:
   `https://thekhananas.github.io/mlsys_coursework_project/`
