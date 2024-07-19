
#include <iostream>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int size) {

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	c[i] = a[i] + b[i]; 

}

using namespace std;

int main() {

	int elements { 10 };
	size_t size = elements*sizeof(float);

	float* h_a { (float *) malloc(size) };
	float* h_b { (float *) malloc(size) };
	float* h_c { (float *) malloc(size) };

	for (int i=0; i<elements; i++) {
		h_a[i] = 2;
		h_b[i] = 3;
	}

	float* d_a = nullptr;
	float* d_b = nullptr;
	float* d_c = nullptr;

	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int threads { 512 };
	int blocks { elements + threads - 1 / threads };
	vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, elements);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}


