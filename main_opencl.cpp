#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#endif

using namespace std;

// These will be overridden by -D flags in the compiler
#ifndef NUM_POINTS
#define NUM_POINTS 100000
#endif
#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 10
#endif
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS 20
#endif
#ifndef KERNEL_FILE
#define KERNEL_FILE "kmeans_gpu.cl"
#endif

void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

string loadKernel(const char* filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "FATAL: Could not find kernel file at: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main() {
    int n_p = NUM_POINTS;
    int n_c = NUM_CLUSTERS;
    int n_i = MAX_ITERATIONS;

    cl_platform_id platform;
    checkErr(clGetPlatformIDs(1, &platform, NULL), "Get Platform");
    cl_device_id device;
    checkErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL), "Get Device");
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    string kernelSource = loadKernel(KERNEL_FILE);
    const char* source = kernelSource.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    
    if(clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        char buffer[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        fprintf(stderr, "CL Build Error:\n%s\n", buffer);
        exit(1);
    }
    cl_kernel k_assign = clCreateKernel(program, "compute_distances_kernel", NULL);
    cl_kernel k_clear = clCreateKernel(program, "clear_centroids_kernel", NULL);
    cl_kernel k_accum = clCreateKernel(program, "accumulate_centroids_kernel", NULL);
    cl_kernel k_final = clCreateKernel(program, "finalize_centroids_kernel", NULL);

    vector<double> h_px(n_p), h_py(n_p), h_cx(n_c), h_cy(n_c);
    for(int i=0; i<n_p; i++) { h_px[i]=rand()%100; h_py[i]=rand()%100; }
    for(int i=0; i<n_c; i++) { h_cx[i]=rand()%100; h_cy[i]=rand()%100; }

    cl_mem d_px = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*n_p, h_px.data(), NULL);
    cl_mem d_py = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double)*n_p, h_py.data(), NULL);
    cl_mem d_cx = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*n_c, h_cx.data(), NULL);
    cl_mem d_cy = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double)*n_c, h_cy.data(), NULL);
    cl_mem d_ids = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n_p, NULL, NULL);
    
    // Additional buffers for GPU-side reduction
    cl_mem d_sum_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(long)*n_c, NULL, NULL);
    cl_mem d_sum_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(long)*n_c, NULL, NULL);
    cl_mem d_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*n_c, NULL, NULL);

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < n_i; iter++) {
        // 1. Clear accumulators
        clSetKernelArg(k_clear, 0, sizeof(cl_mem), &d_sum_x);
        clSetKernelArg(k_clear, 1, sizeof(cl_mem), &d_sum_y);
        clSetKernelArg(k_clear, 2, sizeof(cl_mem), &d_counts);
        clSetKernelArg(k_clear, 3, sizeof(int), &n_c);
        size_t global_c = n_c;
        clEnqueueNDRangeKernel(queue, k_clear, 1, NULL, &global_c, NULL, 0, NULL, NULL);

        // 2. Assign Clusters
        clSetKernelArg(k_assign, 0, sizeof(cl_mem), &d_px);
        clSetKernelArg(k_assign, 1, sizeof(cl_mem), &d_py);
        clSetKernelArg(k_assign, 2, sizeof(cl_mem), &d_cx);
        clSetKernelArg(k_assign, 3, sizeof(cl_mem), &d_cy);
        clSetKernelArg(k_assign, 4, sizeof(cl_mem), &d_ids);
        clSetKernelArg(k_assign, 5, sizeof(int), &n_p);
        clSetKernelArg(k_assign, 6, sizeof(int), &n_c);
        size_t global_p = n_p;
        clEnqueueNDRangeKernel(queue, k_assign, 1, NULL, &global_p, NULL, 0, NULL, NULL);

        // 3. Accumulate Coordinates
        clSetKernelArg(k_accum, 0, sizeof(cl_mem), &d_px);
        clSetKernelArg(k_accum, 1, sizeof(cl_mem), &d_py);
        clSetKernelArg(k_accum, 2, sizeof(cl_mem), &d_ids);
        clSetKernelArg(k_accum, 3, sizeof(cl_mem), &d_sum_x);
        clSetKernelArg(k_accum, 4, sizeof(cl_mem), &d_sum_y);
        clSetKernelArg(k_accum, 5, sizeof(cl_mem), &d_counts);
        clSetKernelArg(k_accum, 6, sizeof(int), &n_p);
        clEnqueueNDRangeKernel(queue, k_accum, 1, NULL, &global_p, NULL, 0, NULL, NULL);

        // 4. Finalize Centroids (Average)
        clSetKernelArg(k_final, 0, sizeof(cl_mem), &d_cx);
        clSetKernelArg(k_final, 1, sizeof(cl_mem), &d_cy);
        clSetKernelArg(k_final, 2, sizeof(cl_mem), &d_sum_x);
        clSetKernelArg(k_final, 3, sizeof(cl_mem), &d_sum_y);
        clSetKernelArg(k_final, 4, sizeof(cl_mem), &d_counts);
        clSetKernelArg(k_final, 5, sizeof(int), &n_c);
        clEnqueueNDRangeKernel(queue, k_final, 1, NULL, &global_c, NULL, 0, NULL, NULL);

        clFinish(queue);
    }

    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double>(end - start).count();
    printf("total time: %f seconds\n", total_time);

    clReleaseKernel(k_assign); clReleaseKernel(k_clear);
    clReleaseKernel(k_accum); clReleaseKernel(k_final);
    clReleaseProgram(program);
    clReleaseMemObject(d_px); clReleaseMemObject(d_py);
    clReleaseMemObject(d_cx); clReleaseMemObject(d_cy);
    clReleaseMemObject(d_ids);
    clReleaseMemObject(d_sum_x); clReleaseMemObject(d_sum_y);
    clReleaseMemObject(d_counts);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return 0;
}