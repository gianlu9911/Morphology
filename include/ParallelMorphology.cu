#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define TILE_SIZE 16 // Work area inside each block
#define KERNEL_RADIUS 1 // For a 3x3 kernel

__global__ void erosionKernel(unsigned char *input, unsigned char *output, int width, int height) {
    // Shared memory with border (halo)
    __shared__ unsigned char sharedMem[TILE_SIZE + 2 * KERNEL_RADIUS][TILE_SIZE + 2 * KERNEL_RADIUS];

    // Compute global and shared memory indices
    int x = blockIdx.x * TILE_SIZE + threadIdx.x - KERNEL_RADIUS;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y - KERNEL_RADIUS;

    int sharedX = threadIdx.x;
    int sharedY = threadIdx.y;

    // Load the tile with halo into shared memory (coalesced access)
    if (x >= 0 && x < width && y >= 0 && y < height) {
        sharedMem[sharedY][sharedX] = input[y * width + x];
    } else {
        sharedMem[sharedY][sharedX] = 255; // Assume background for padding
    }

    __syncthreads();

    // Compute erosion only for valid pixels
    if (threadIdx.x >= KERNEL_RADIUS && threadIdx.x < TILE_SIZE + KERNEL_RADIUS &&
        threadIdx.y >= KERNEL_RADIUS && threadIdx.y < TILE_SIZE + KERNEL_RADIUS) {
        
        unsigned char minVal = 255;

        // Apply 3x3 erosion (optimized memory access)
        for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
            for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
                minVal = min(minVal, sharedMem[sharedY + dy][sharedX + dx]);
            }
        }

        // Compute output indices
        int outX = blockIdx.x * TILE_SIZE + threadIdx.x - KERNEL_RADIUS;
        int outY = blockIdx.y * TILE_SIZE + threadIdx.y - KERNEL_RADIUS;

        if (outX >= 0 && outX < width && outY >= 0 && outY < height) {
            output[outY * width + outX] = minVal;
        }
    }
}

void erosionCUDA(const cv::Mat &inputImage, cv::Mat &outputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    unsigned char *h_input = nullptr, *h_output = nullptr;

    // Allocate pinned memory on the host for input and output images
    cudaMallocHost(&h_input, imageSize);
    cudaMallocHost(&h_output, imageSize);

    // Allocate memory on the device
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Copy data from host to pinned memory
    memcpy(h_input, inputImage.data, imageSize);

    // Copy input data from pinned host memory to device memory
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(TILE_SIZE + 2 * KERNEL_RADIUS, TILE_SIZE + 2 * KERNEL_RADIUS);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA Events to measure the time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    erosionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Erosion kernel execution time: " << milliseconds << " ms" << std::endl;

    // Copy result back to pinned memory
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Copy the result from pinned memory to OpenCV Mat
    memcpy(outputImage.data, h_output, imageSize);

    // Free allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
