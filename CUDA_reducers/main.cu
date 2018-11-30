#include <iostream>

#define n 64
#define blockSize 16
#define size_partial_sum blockSize * 2

__global__
void sum_reducer1(int *d_data)
{
    __shared__ int partialSum[size_partial_sum];
    partialSum[threadIdx.x] = d_data[threadIdx.x + blockIdx.x * blockDim.x * 2];
    partialSum[threadIdx.x + blockDim.x] = d_data[blockDim.x + threadIdx.x + blockIdx.x * blockDim.x * 2];
    int t = threadIdx.x;
    for (unsigned int stride = 1; stride < blockDim.x * 2; stride *= 2)
    {
        __syncthreads();
        if (t % stride == 0)
            partialSum[t*2] += partialSum[t*2 + stride];
    }

    if (threadIdx.x == 0)
        printf("blockIdx.x = %d, partialSum = %i\n", blockIdx.x, partialSum[0]);
}

__global__
void sum_reducer2(int *d_data)
{
    __shared__ int partialSum[size_partial_sum];
    partialSum[threadIdx.x] = d_data[threadIdx.x + blockIdx.x * blockDim.x * 2];
    partialSum[threadIdx.x + blockDim.x] = d_data[blockDim.x + threadIdx.x + blockIdx.x * blockDim.x * 2];
    int t = threadIdx.x;
    for (int stride = blockDim.x; stride >= 1; stride = stride >> 1)
    {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t+stride];
    }

    if (threadIdx.x == 0)
        printf("blockIdx.x = %d, partialSum = %i\n", blockIdx.x, partialSum[0]);
}

int main()
{
    int h_data[n];
    for (int i = 0; i < n; ++i)
    {
        h_data[i] = i;
    }
    int *d_data;
    cudaMalloc((void **)&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n/(blockSize * 2.0)), 1, 1);
    dim3 dimBlock(blockSize, 1, 1);
    //sum_reducer1<<<dimGrid, dimBlock>>>(d_data);
    sum_reducer2<<<dimGrid, dimBlock>>>(d_data);

    cudaFree(d_data);

    return 0;
}