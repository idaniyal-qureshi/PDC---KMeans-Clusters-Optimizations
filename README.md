# KMeans Multi-Version Benchmark

This project benchmarks four different implementations of the K-Means clustering algorithm: Sequential, Parallel (OpenMP), Optimized (SoA), and GPU-Accelerated (OpenCL).

## How to Run on Google Colab

To get the most accurate results for the OpenCL version, follow these steps to enable the GPU:

### 1. Enable T4 GPU
1.  Open your Colab notebook.
2.  Go to the top menu: **Edit** -> **Notebook settings** (or **Runtime** -> **Change runtime type**).
3.  Under **Hardware accelerator**, select **T4 GPU**.
4.  Click **Save**.

### 2. Execution Commands
Upload the project files to your Colab session storage and run the following cells:

```bash
# 1. Give execution permission to the script
!chmod +x run_benchmark.sh

# 2. Run the full benchmark (this may take a few minutes)
!./run_benchmark.sh

# 3. View the final performance report
!cat benchmark_report.txt
```

## 📊 Benchmark Versions
- **Seq**: Basic single-threaded C++ implementation.
- **Par**: Multi-threaded version using OpenMP.
- **Opt**: Performance-optimized version using Structure of Arrays (SoA) and local accumulation.
- **OCL**: High-performance GPU version using OpenCL (optimized for zero-copy iterations).

## 🛠 Project Structure
- `main_sequential.cpp`: Sequential implementation.
- `main_parallel.cpp`: OpenMP parallel implementation.
- `main_optimized.cpp`: Memory-efficient SoA implementation.
- `main_opencl.cpp`: OpenCL host code (handles GPU orchestration).
- `kmeans_gpu.cl`: OpenCL kernels (Distance calculation & Centroid updates).
- `run_benchmark.sh`: Automated test suite and report generator.
