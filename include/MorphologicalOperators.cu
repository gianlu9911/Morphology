#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// Define the structuring element dimensions (3x3) and its origin.
#define SE_WIDTH     3
#define SE_HEIGHT    3
#define SE_ORIGIN_X  1  // Center of 3x3 kernel
#define SE_ORIGIN_Y  1

// Tiled kernel for binary erosion.
// Each thread computes one output pixel using a shared memory tile that
// covers the block plus a halo required by the structuring element.
__global__ void binaryErosionTiledKernel(const unsigned char* input, 
                                           unsigned char* output,
                                           int width, int height)
{
    // Compute output pixel coordinates.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Determine the dimensions of the shared memory tile.
    // The tile covers the block plus the extra halo needed on each side.
    const int tileWidth  = blockDim.x + SE_WIDTH - 1;
    const int tileHeight = blockDim.y + SE_HEIGHT - 1;

    // Allocate shared memory dynamically.
    extern __shared__ unsigned char sTile[];

    // Top-left coordinates in global memory that correspond to the shared tile.
    int sharedStartX = blockIdx.x * blockDim.x - SE_ORIGIN_X;
    int sharedStartY = blockIdx.y * blockDim.y - SE_ORIGIN_Y;

    // Each thread loads one or more pixels into shared memory.
    // We use a strided loop to cover the entire tile.
    for (int j = threadIdx.y; j < tileHeight; j += blockDim.y) {
        for (int i = threadIdx.x; i < tileWidth; i += blockDim.x) {
            int globalX = sharedStartX + i;
            int globalY = sharedStartY + j;
            // If within image bounds, load the pixel; otherwise assume background.
            if (globalX >= 0 && globalX < width && globalY >= 0 && globalY < height) {
                sTile[j * tileWidth + i] = input[globalY * width + globalX];
            } else {
                // For binary erosion, out-of-bound pixels are considered background (0).
                sTile[j * tileWidth + i] = 0;
            }
        }
    }
    __syncthreads();

    // Only process output pixels that lie within image bounds.
    if (x < width && y < height) {
        unsigned char minVal = 255;
        // The corresponding top-left corner of the 3x3 neighborhood in the shared tile.
        int localX = threadIdx.x;
        int localY = threadIdx.y;

        // Loop over the structuring element.
        for (int j = 0; j < SE_HEIGHT; j++) {
            for (int i = 0; i < SE_WIDTH; i++) {
                unsigned char candidate = sTile[(localY + j) * tileWidth + (localX + i)];
                // For erosion, any background (0) forces the output to 0.
                if (candidate < minVal) {
                    minVal = candidate;
                    // Early exit if we hit background.
                    if (minVal == 0)
                        goto write_result;
                }
            }
        }
    write_result:
        output[y * width + x] = minVal;
    }
}

int main()
{
    // Load image in grayscale.
    cv::Mat img = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not open image ../imgs/j.jpg" << std::endl;
        return -1;
    }

    // Convert image to binary (0 and 255).
    cv::Mat binaryImg;
    cv::threshold(img, binaryImg, 128, 255, cv::THRESH_BINARY);

    int width  = binaryImg.cols;
    int height = binaryImg.rows;
    size_t imgSize = width * height * sizeof(unsigned char);

    // Allocate device memory.
    unsigned char *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    // Copy input image data to device.
    cudaMemcpy(d_input, binaryImg.data, imgSize, cudaMemcpyHostToDevice);

    // Define block dimensions. Adjust these for your GPU.
    dim3 block(32,32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Calculate shared memory size:
    // Each block needs a tile of size (blockDim.x + SE_WIDTH - 1) * (blockDim.y + SE_HEIGHT - 1)
    int tileWidth  = block.x + SE_WIDTH - 1;
    int tileHeight = block.y + SE_HEIGHT - 1;
    size_t sharedMemSize = tileWidth * tileHeight * sizeof(unsigned char);

    // Launch the tiled erosion kernel.
    binaryErosionTiledKernel<<<grid, block, sharedMemSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to host.
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy(result.data, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Save the result image.
    cv::imwrite("../erosion_tiled_result.jpg", result);
    std::cout << "Erosion result saved as erosion_tiled_result.jpg" << std::endl;

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}