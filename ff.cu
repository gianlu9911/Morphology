#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include "SequentialMorphology.h"

// CUDA error check macro.
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA error in " << __FILE__                \
                      << " at line " << __LINE__ << ": "              \
                      << cudaGetErrorString(err) << std::endl;        \
            exit(err);                                                \
        }                                                             \
    } while (0)

// CUDA kernel for vertical erosion on a grayscale image.
// Each thread processes one pixel.
__global__ void verticalErosionKernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width, int height,
                                        int radius)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx >= totalPixels) return;

    // Map 1D index to 2D coordinates (row, col)
    int col = idx % width;
    int row = idx / width;

    // Clamp vertical boundaries to the image
    int rowStart = max(0, row - radius);
    int rowEnd   = min(height - 1, row + radius);

    unsigned char minVal = 255;
    for (int r = rowStart; r <= rowEnd; ++r)
    {
        unsigned char pixel = input[r * width + col];
        minVal = min(minVal, pixel);
    }
    output[idx] = minVal;
}

// Optimized CUDA kernel for horizontal erosion on a grayscale image using shared memory.
// Each block processes a contiguous segment of a single row.
// The kernel loads a tile from global memory, including halo pixels for the neighborhood,
// into shared memory for efficient, coalesced access.
__global__ void horizontalErosionKernelShared(const unsigned char* input,
                                              unsigned char* output,
                                              int width, int height,
                                              int radius)
{
    // Each block processes one row.
    int row = blockIdx.y;
    // The x coordinate of this thread in the image.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Allocate shared memory: each block holds its tile plus halo pixels on both sides.
    extern __shared__ unsigned char s_data[];

    // Offset in shared memory where this block's actual data begins.
    // The left halo occupies the first 'radius' positions.
    int s_index = threadIdx.x + radius;

    // Load the central data element if within bounds.
    if (col < width)
        s_data[s_index] = input[row * width + col];
    else
        s_data[s_index] = 255; // Set to max for erosion if out-of-bound.

    // Load left halo pixels.
    if (threadIdx.x < radius) {
        int halo_col = blockIdx.x * blockDim.x + threadIdx.x - radius;
        s_data[threadIdx.x] = (halo_col >= 0) ? input[row * width + halo_col] : 255;
    }
    
    // Load right halo pixels.
    int rightHaloIndex = threadIdx.x + blockDim.x + radius;
    int halo_col = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
    if (threadIdx.x < radius) {
        s_data[rightHaloIndex] = (halo_col < width) ? input[row * width + halo_col] : 255;
    }
    
    __syncthreads();

    // Now, if the current pixel is within the image, perform erosion over its horizontal window.
    if (col < width)
    {
        unsigned char minVal = 255;
        // The shared memory window for this pixel spans from s_index - radius to s_index + radius.
        for (int offset = -radius; offset <= radius; ++offset)
        {
            minVal = min(minVal, s_data[s_index + offset]);
        }
        output[row * width + col] = minVal;
    }
}

int main()
{
    // Run the sequential test.
    sequentialTest("../imgs/lena_4k.jpg");

    // Read the image in grayscale.
    cv::Mat image = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()){
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;
    size_t imgSize = totalPixels * sizeof(unsigned char);

    // Allocate device memory for input, vertical and horizontal outputs.
    unsigned char *d_input = nullptr, *d_output_vertical = nullptr, *d_output_horizontal = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output_vertical, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output_horizontal, imgSize));

    // Copy the input image from host to device.
    CUDA_CHECK(cudaMemcpy(d_input, image.ptr(), imgSize, cudaMemcpyHostToDevice));

    // Set erosion radius.
    const int RADIUS = 3;
    int radius = RADIUS;

    // --- Vertical Erosion Kernel Execution Timing ---
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    
    cudaEvent_t startVert, stopVert;
    CUDA_CHECK(cudaEventCreate(&startVert));
    CUDA_CHECK(cudaEventCreate(&stopVert));
    
    CUDA_CHECK(cudaEventRecord(startVert));
    verticalErosionKernel<<<numBlocks, blockSize>>>(d_input, d_output_vertical, width, height, radius);
    CUDA_CHECK(cudaEventRecord(stopVert));
    CUDA_CHECK(cudaEventSynchronize(stopVert));
    
    float msVertical = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msVertical, startVert, stopVert));
    std::cout << "Vertical kernel execution time: " << msVertical << " ms" << std::endl;
    
    // Destroy vertical kernel events.
    CUDA_CHECK(cudaEventDestroy(startVert));
    CUDA_CHECK(cudaEventDestroy(stopVert));

    // --- Horizontal Erosion Kernel Execution Timing ---
    int tileWidth = 256;
    dim3 blockDim(tileWidth, 1, 1);
    dim3 gridDim((width + tileWidth - 1) / tileWidth, height, 1);
    size_t sharedMemSize = (tileWidth + 2 * radius) * sizeof(unsigned char);

    cudaEvent_t startHoriz, stopHoriz;
    CUDA_CHECK(cudaEventCreate(&startHoriz));
    CUDA_CHECK(cudaEventCreate(&stopHoriz));
    
    CUDA_CHECK(cudaEventRecord(startHoriz));
    horizontalErosionKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output_horizontal, width, height, radius);
    CUDA_CHECK(cudaEventRecord(stopHoriz));
    CUDA_CHECK(cudaEventSynchronize(stopHoriz));
    
    float msHorizontal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msHorizontal, startHoriz, stopHoriz));
    std::cout << "Horizontal kernel execution time: " << msHorizontal << " ms" << std::endl;
    
    // Destroy horizontal kernel events.
    CUDA_CHECK(cudaEventDestroy(startHoriz));
    CUDA_CHECK(cudaEventDestroy(stopHoriz));

    // Compute the total kernel execution time (vertical + horizontal) in seconds.
    float totalMs = msVertical + msHorizontal;
    float totalSec = totalMs / 1000.0f;
    std::cout << "Total erosion execution time: " << totalSec << " seconds" << std::endl;

    // Copy the processed images back from device to host.
    cv::Mat outputVertical(height, width, CV_8UC1);
    cv::Mat outputHorizontal(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(outputVertical.ptr(), d_output_vertical, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputHorizontal.ptr(), d_output_horizontal, imgSize, cudaMemcpyDeviceToHost));

    // Display the images.
    cv::imshow("Original Image", image);
    cv::imshow("Vertical Eroded Image", outputVertical);
    cv::imshow("Horizontal Eroded Image (Optimized)", outputHorizontal);
    cv::waitKey(0);

    // Cleanup device memory.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_vertical));
    CUDA_CHECK(cudaFree(d_output_horizontal));

    return 0;
}