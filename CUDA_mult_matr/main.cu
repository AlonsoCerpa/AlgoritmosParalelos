#include <iostream>

__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((Row < Width) && (Col < Width)) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row*Width+k]*N[k*Width+Col];
        }
        P[Row*Width+Col] = Pvalue;
    }
}

int main()
{
    const int w = 2;
    float M[w][w] = {{0, 1}, {2, 3}};
    float N[w][w] = {{0, 1}, {2, 3}};
    float P[w][w];

    int sizeMat = w * w * sizeof(float);
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **) &d_M, w * w * sizeof(float));
    cudaMalloc((void **) &d_N, w * w * sizeof(float));
    cudaMalloc((void **) &d_P, w * w * sizeof(float));

    cudaMemcpy(d_M, M, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, sizeMat, cudaMemcpyHostToDevice);

    dim3 dimBlock(w, w, 1);
    MatrixMulKernel<<<1, dimBlock>>>(d_M, d_N, d_P, w);

    cudaMemcpy(P, d_P, sizeMat, cudaMemcpyDeviceToHost);

    for (int i = 0; i < w; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            std::cout << P[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}