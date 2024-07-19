
#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to execute on the GPU
__global__ void helloFromGPU() {
    printf("Hello, World from GPU!\n");
}

int main() {
    std::cout << "Hello, World from CPU!\n";

    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "helloFromGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    return 0;
}

