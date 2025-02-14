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

    // Load the tile with halo into shared memory
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
        
        // Apply 3x3 erosion
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
    
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE + 2 * KERNEL_RADIUS, TILE_SIZE + 2 * KERNEL_RADIUS);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    erosionKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
