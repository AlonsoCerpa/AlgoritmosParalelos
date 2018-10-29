#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

//compilar con:
//nvcc main.cu `pkg-config --cflags --libs opencv`

__global__
void blurKernel(unsigned char *Pout, unsigned char *Pin, int width, int height, int sizePatch)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height)
    {
        int pixels = 0;
        int pixVal = 0;

        for (int i = -sizePatch; i < sizePatch + 1; ++i)
        {
            for (int j = -sizePatch; j < sizePatch + 1; ++j)
            {
                int currRow = Row + i;
                int currCol = Col + j;
                if (currCol > -1 && currCol < width && currRow > -1 && currRow < height)
                {
                    pixVal += Pin[currRow * width + currCol];
                    ++pixels;
                }
            }
        }

        Pout[Row * width + Col] = (unsigned char) pixVal / pixels;
    }
}

void blur(unsigned char* h_BGR, unsigned char* h_blur, int width, int height, int size, int sizePatch)
{
    unsigned char *d_BGR, *d_blur;

    cudaMalloc((void **) &d_BGR, size);
    cudaMemcpy(d_BGR, h_BGR, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_blur, size);


    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blurKernel<<<dimGrid, dimBlock>>>(d_blur, d_BGR, width, height, sizePatch);

    cudaMemcpy(h_blur, d_blur, size, cudaMemcpyDeviceToHost);

    cudaFree(d_BGR);
    cudaFree(d_blur);
}

int main()
{
    cv::String imageName("tiger.jpg"); // by default
    cv::Mat imageBGR = cv::imread(imageName, cv::IMREAD_GRAYSCALE); // Read the file
    if(imageBGR.empty())                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    cv::Mat imageBlur = imageBGR.clone();

    int sizePatch = 1;
    blur((unsigned char *)imageBGR.data, (unsigned char *)imageBlur.data, imageBGR.cols, imageBGR.rows, imageBGR.total(), sizePatch);

    cv::namedWindow("GrayScale display", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("GrayScale display", imageBGR);                // Show our image inside it.
    
    cv::namedWindow("Blur display", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Blur display", imageBlur);
    
    cv::waitKey(0); // Wait for a keystroke in the window

    return 0;
}