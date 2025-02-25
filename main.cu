#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 32  // square block dimension; should be a multiple of warp size
#define BLOCK_ROWS 32  // we use a square block, so BLOCK_ROWS = TILE_DIM

// CUDA kernel: perform 1D horizontal erosion then write result in transposed order.
// Input is a width×height image; output is a height×width image (i.e. fully transposed).
__global__ void erosion1D_transpose_kernel_coalesced(const unsigned char* __restrict__ input,
                                                     unsigned char* __restrict__ output,
                                                     int width, int height,
                                                     int window_radius)
{
    // Using a square block of TILE_DIM x TILE_DIM threads.
    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // column index in input
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // row index in input

    // Each thread computes the erosion result for its pixel, if in range.
    unsigned char result = 255;  // neutral element for min (erosion)
    if (x < width && y < height)
    {
        unsigned char min_val = 255;
        #pragma unroll
        for (int dx = -window_radius; dx <= window_radius; dx++) {
            int curX = x + dx;
            // Clamp to valid column range:
            curX = (curX < 0) ? 0 : ((curX >= width) ? width - 1 : curX);
            unsigned char pix = input[y * width + curX];
            min_val = min(min_val, pix);
        }
        result = min_val;
    }

    // Declare shared memory tile with extra column to avoid bank conflicts.
    __shared__ unsigned char tile[TILE_DIM][TILE_DIM+1];

    // Each thread writes its computed result into shared memory.
    tile[threadIdx.y][threadIdx.x] = result;
    __syncthreads();

    // Compute transposed coordinates.
    // In the transposed output, the row index becomes the original column index and vice versa.
    int transposedRow = blockIdx.x * TILE_DIM + threadIdx.y;  // becomes new row index
    int transposedCol = blockIdx.y * TILE_DIM + threadIdx.x;    // becomes new column index

    // Write the tile out in transposed order:
    // Note that the output image has dimensions (height, width)
    if (transposedRow < width && transposedCol < height)
    {
        // Read from shared memory with swapped indices.
        output[transposedRow * height + transposedCol] = tile[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    // Load the image from "../imgs/lena.jpg" in grayscale.
    cv::Mat inputImage = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not open or find the image at ../imgs/lena.jpg" << std::endl;
        return -1;
    }

    int width  = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // The output image will be the transposed result of the erosion:
    // Its dimensions will be (height x width)
    cv::Mat outputImage(height, width, CV_8UC1);
    // Note: Since the output is transposed relative to the input, when viewing
    // it you'll see the rows and columns swapped.

    // Allocate device memory.
    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    // Copy the input image to device.
    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // Set up kernel launch parameters.
    int window_radius = 1; // For a 3-pixel horizontal window.
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel.
    erosion1D_transpose_kernel_coalesced<<<gridDim, blockDim>>>(d_input, d_output, width, height, window_radius);
    cudaDeviceSynchronize();

    // Copy the result back to host memory.
    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_output);

    // Display the images.
    cv::imshow("Original Image", inputImage);
    cv::imshow("Eroded Image (Transposed)", outputImage);
    cv::waitKey(0);

    return 0;
}
