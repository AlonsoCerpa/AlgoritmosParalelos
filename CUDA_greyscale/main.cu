#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#define CHANNELS 3

//compilar con:
//nvcc main.cu `pkg-config --cflags --libs opencv`

__global__
void colorToGreyscaleConversionKernel(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height)
    {
        int greyoffset = Row*width + Col;
        int rgbOffset = greyoffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyoffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

void colorToGreyscaleConversion(unsigned char* h_BGR, unsigned char* h_greyScale, int width, int height, int size)
{
    unsigned char *d_BGR, *d_greyScale;

    cudaMalloc((void **) &d_BGR, size * CHANNELS);
    cudaMemcpy(d_BGR, h_BGR, size * CHANNELS, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_greyScale, size);

    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGreyscaleConversionKernel<<<dimGrid, dimBlock>>>(d_greyScale, d_BGR, width, height);

    cudaMemcpy(h_greyScale, d_greyScale, size, cudaMemcpyDeviceToHost);

    cudaFree(d_BGR);
    cudaFree(d_greyScale);
}

int main()
{
    cv::String imageName("img.jpg"); // by default
    cv::Mat imageBGR = cv::imread(imageName, cv::IMREAD_COLOR); // Read the file
    if(imageBGR.empty())                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    cv::Mat imageGrey(imageBGR.rows, imageBGR.cols, CV_8UC1);

    colorToGreyscaleConversion((unsigned char *)imageBGR.data, (unsigned char *)imageGrey.data, imageBGR.cols, imageBGR.rows, imageBGR.total());

    cv::namedWindow("BGR display", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("BGR display", imageBGR);                // Show our image inside it.
    
    cv::namedWindow("Grey display", cv::WINDOW_AUTOSIZE); // Create a window for display.
    cv::imshow("Grey display", imageGrey);
    
    cv::waitKey(0); // Wait for a keystroke in the window

    return 0;
}