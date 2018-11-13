#include <iostream>

__global__
void sum_mat_kernel(float* d_A, float* d_B, float* d_C, int n)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < n && row < n)
    {
        int idx = row*n + col;
        d_C[idx] = d_A[idx] + d_B[idx]; 
    }
}

__global__
void sum_mat_row_kernel(float* d_A, float* d_B, float* d_C, int n)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < n)
    {
        int idx = row*n;
        for (int i = 0; i < n; ++i)
        {
            d_C[idx + i] = d_A[idx + i] + d_B[idx + i];
        }
    }
}

__global__
void sum_mat_col_kernel(float* d_A, float* d_B, float* d_C, int n)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (col < n)
    {
        int idx = col;
        for (int i = 0; i < n; ++i)
        {
            d_C[idx] = d_A[idx] + d_B[idx];
            idx += n;
        }
    }
}

void sum_mat_row(float* h_A, float* h_B, float* h_C, int n)
{
    float *d_A, *d_B, *d_C;

    int size_mat = n * n * sizeof(float);
    cudaMalloc((void **) &d_A, size_mat);
    cudaMemcpy(d_A, h_A, size_mat, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_B, size_mat);
    cudaMemcpy(d_B, h_B, size_mat, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size_mat);

    dim3 dimGrid(1, ceil(n/16.0), 1);
    dim3 dimBlock(1, 16, 1);
    sum_mat_row_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size_mat, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
}

void sum_mat_col(float* h_A, float* h_B, float* h_C, int n)
{
    float *d_A, *d_B, *d_C;

    int size_mat = n * n * sizeof(float);
    cudaMalloc((void **) &d_A, size_mat);
    cudaMemcpy(d_A, h_A, size_mat, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_B, size_mat);
    cudaMemcpy(d_B, h_B, size_mat, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size_mat);

    dim3 dimGrid(ceil(n/16.0), 1, 1);
    dim3 dimBlock(16, 1, 1);
    sum_mat_col_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size_mat, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
}

void sum_mat(float* h_A, float* h_B, float* h_C, int n)
{
    float *d_A, *d_B, *d_C;

    int size_mat = n * n * sizeof(float);
    cudaMalloc((void **) &d_A, size_mat);
    cudaMemcpy(d_A, h_A, size_mat, cudaMemcpyHostToDevice);
    
    cudaMalloc((void **) &d_B, size_mat);
    cudaMemcpy(d_B, h_B, size_mat, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size_mat);

    dim3 dimGrid(ceil(n/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    sum_mat_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size_mat, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
}

void print_mat(float* m, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << m[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    int n = 4;
    float *h_A = new float[n*n];
    float *h_B = new float[n*n];
    float *h_C = new float[n*n];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_A[i*n + j] = i;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_B[i*n + j] = j;
        }
    }

    //sum_mat(h_A, h_B, h_C, n);
    sum_mat_row(h_A, h_B, h_C, n);
    //sum_mat_col(h_A, h_B, h_C, n);

    print_mat(h_A, n);
    std::cout << "\n";
    print_mat(h_B, n);
    std::cout << "\n";
    print_mat(h_C, n);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}