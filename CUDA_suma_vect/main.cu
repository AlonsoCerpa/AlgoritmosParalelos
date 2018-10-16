#include <iostream>
#include <math.h>
#include <chrono>

__global__
void vecAddKernel(float* d_A, float* d_B, float* d_C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        d_C[i] = d_A[i] + d_B[i];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
}

int main(void)
{
    int n = 100;
    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    for (int i = 0; i < n; ++i)
    {
        h_A[i] = i;
        h_B[i] = i;
    }

    vecAdd(h_A, h_B, h_C, n);

    for (int i = 0; i < n; ++i)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
	
	return 0;
}