#include <iostream>

__global__
void mult_mat_vec_kernel(float* d_A, float* d_B, float* d_C, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < n)
    {
        int idx1 = idx * n;
        float sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sum += d_A[idx1 + i] * d_B[i];
        }
        d_C[idx] = sum;
    }
}

void mult_mat_vec(float* h_A, float* h_B, float* h_C, int n)
{
    float *d_A, *d_B, *d_C;

    int size_mat = n * n * sizeof(float);
    cudaMalloc((void **) &d_A, size_mat);
    cudaMemcpy(d_A, h_A, size_mat, cudaMemcpyHostToDevice);
    
    int size_vec = n * sizeof(float);
    cudaMalloc((void **) &d_B, size_vec);
    cudaMemcpy(d_B, h_B, size_vec, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size_vec);

    dim3 dimGrid(ceil(n/16.0), 1, 1);
    dim3 dimBlock(16, 1, 1);
    mult_mat_vec_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size_vec, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B);
}

void print_vec(float* m, int n)
{
    for (int j = 0; j < n; ++j)
    {
        std::cout << m[j] << " ";
    }
    std::cout << "\n";
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
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_A[i*n + j] = i;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        h_B[i] = i;
    }

    mult_mat_vec(h_A, h_B, h_C, n);

    print_mat(h_A, n);
    std::cout << "\n";
    print_vec(h_B, n);
    std::cout << "\n";
    print_vec(h_C, n);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}