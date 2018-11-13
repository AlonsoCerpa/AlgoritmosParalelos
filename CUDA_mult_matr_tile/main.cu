#include <iostream>

#define TILE_WIDTH 2

__global__ void MatrixMulTileKernel(float* d_M, float* d_N, float* d_P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
     // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    // Loop over the d_M and d_N tiles required to compute d_P element
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph)
    {
     // Collaborative loading of d_M and d_N tiles into shared memory
        Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
        __syncthreads();
    
        for (int k = 0; k < TILE_WIDTH; ++k) 
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    
    }
    d_P[Row*Width + Col] = Pvalue;
}



int main()
{
    const int w = 4;
    float M[w][w] = {{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}};
    float N[w][w] = {{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}};
    float P[w][w];

    int sizeMat = w * w * sizeof(float);
    float *d_M, *d_N, *d_P;

    cudaMalloc((void **) &d_M, w * w * sizeof(float));
    cudaMalloc((void **) &d_N, w * w * sizeof(float));
    cudaMalloc((void **) &d_P, w * w * sizeof(float));

    cudaMemcpy(d_M, M, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, P, sizeMat, cudaMemcpyHostToDevice);

    dim3 dimGrid(2, 2, 1);
    dim3 dimBlock(2, 2, 1);
    MatrixMulTileKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, w);

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