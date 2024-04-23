
#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to execute on the GPU
__global__ void helloFromGPU() {
    printf("Hello, World from GPU!\n");
}

int main() {
    std::cout << "Hello, World from CPU!\n";

    // Launch the GPU Kernel
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize(); // Ensures that the GPU finishes executing before proceeding

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "helloFromGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}

