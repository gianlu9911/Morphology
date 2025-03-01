#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

#define WARP_SIZE 32

// CUDA error checking macro.
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel: Each warp computes the erosion for one output pixel.
// The erosion operation is defined over a window of size (2*p - 1) centered at the pixel.
// It performs two warp-level reductions: one over the left section (from the window's left bound to the pixel)
// and one over the right section (from the pixel to the window's right bound). The final output is the minimum
// of these two values.
__global__ void erosion1dFullKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int p) // p must be odd; full window length = 2*p - 1
{
    // Total number of output pixels.
    int totalPixels = width * height;
    
    // Each warp handles one output pixel.
    // Compute global warp ID:
    int warpsPerBlock = blockDim.x / WARP_SIZE;
    int globalWarpId = blockIdx.x * warpsPerBlock + (threadIdx.x / WARP_SIZE);
    
    if (globalWarpId >= totalPixels) return;
    
    // Determine the output pixel coordinates.
    int row = globalWarpId / width;
    int col = globalWarpId % width;
    
    // Define the full window (centered at col):
    // Window extends from col - (p - 1) to col + (p - 1)
    int leftBound = col - (p - 1);
    int rightBound = col + (p - 1);
    
    // Clamp window boundaries to the image row.
    if (leftBound < 0) leftBound = 0;
    if (rightBound >= width) rightBound = width - 1;
    
    // Left section: from leftBound to col (inclusive).
    int leftCount = col - leftBound + 1;
    // Right section: from col to rightBound (inclusive).
    int rightCount = rightBound - col + 1;
    
    int lane = threadIdx.x % WARP_SIZE;
    
    // --- Left Section Reduction ---
    unsigned char leftLocalMin = 255;
    // Each thread in the warp loads part of the left section.
    for (int i = lane; i < leftCount; i += WARP_SIZE) {
        int x = leftBound + i;
        unsigned char val = d_input[row * width + x];
        leftLocalMin = min(leftLocalMin, val);
    }
    // Warp-level reduction over the left section.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        unsigned char other = __shfl_down_sync(0xffffffff, leftLocalMin, offset);
        leftLocalMin = min(leftLocalMin, other);
    }
    
    // --- Right Section Reduction ---
    unsigned char rightLocalMin = 255;
    for (int i = lane; i < rightCount; i += WARP_SIZE) {
        int x = col + i; // right section starts at col.
        unsigned char val = d_input[row * width + x];
        rightLocalMin = min(rightLocalMin, val);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        unsigned char other = __shfl_down_sync(0xffffffff, rightLocalMin, offset);
        rightLocalMin = min(rightLocalMin, other);
    }
    
    // The final erosion value is the minimum of the two reductions.
    unsigned char finalMin = min(leftLocalMin, rightLocalMin);
    
    // Lane 0 of each warp writes the output.
    if (lane == 0) {
        d_output[row * width + col] = finalMin;
    }
}

int main_not_now()
{
    // Load a grayscale image using OpenCV.
    cv::Mat img = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    int width = img.cols;
    int height = img.rows;
    
    size_t imageSize = width * height * sizeof(unsigned char);
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    // Copy input image to device.
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imageSize, cudaMemcpyHostToDevice));
    
    // Launch configuration: one warp per output pixel.
    int totalPixels = width * height;
    // Each warp has WARP_SIZE threads.
    int warpsNeeded = totalPixels;
    int threadsPerBlock = 256; // e.g., 256 threads per block.
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    int blocks = (warpsNeeded + warpsPerBlock - 1) / warpsPerBlock;
    
    erosion1dFullKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, width, height, 7);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy the result back to host.
    cv::Mat output(img.size(), img.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Display the input and the eroded output.
    cv::imshow("Input", img);
    cv::imshow("Eroded Output", output);
    cv::waitKey(0);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return 0;
}


// Kernel: Vertical erosion using tiling for improved coalesced global memory accesses.
// For each block, we load a tile of the input image into shared memory. The tile dimensions
// are TILE_WIDTH x (TILE_HEIGHT + 2*(p-1)) to include the apron rows needed for a vertical window
// of size (2*p - 1). Then, each thread computes the vertical erosion for its pixel using the data
// in shared memory.
__global__ void verticalErosionTiledKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int p) // p must be odd; vertical window height = 2*p - 1
{
    // Define apron size (p-1) above and below.
    const int apron = p - 1;
    // Shared memory tile height includes the output tile plus the top and bottom apron.
    const int tileHeightShared = TILE_HEIGHT + 2 * apron;
    const int tileWidth = TILE_WIDTH;

    // Calculate the starting coordinates of the tile in global memory.
    int tileStartX = blockIdx.x * tileWidth;
    int tileStartY = blockIdx.y * TILE_HEIGHT;

    // Allocate shared memory tile.
    // The shared memory buffer holds tileWidth * tileHeightShared elements.
    extern __shared__ unsigned char s_tile[];

    // Each thread loads one or more elements into shared memory in a coalesced manner.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Total number of threads in the block.
    int numThreads = blockDim.x * blockDim.y;
    int threadId = ty * blockDim.x + tx;
    int totalElements = tileWidth * tileHeightShared;
    for (int i = threadId; i < totalElements; i += numThreads)
    {
        int x = i % tileWidth;
        int y = i / tileWidth;
        // Compute the corresponding global row, with an offset to include the top apron.
        int globalY = tileStartY + y - apron;
        // Clamp to image boundaries.
        if (globalY < 0) globalY = 0;
        if (globalY >= height) globalY = height - 1;
        int globalX = tileStartX + x;
        unsigned char val = 0;
        if (globalX < width)
            val = d_input[globalY * width + globalX];
        s_tile[y * tileWidth + x] = val;
    }
    __syncthreads();

    // Now, each thread computes vertical erosion for one output pixel in the tile.
    // The output region in shared memory starts at row 'apron' and spans TILE_HEIGHT rows.
    int outX = tileStartX + tx;
    int outY = tileStartY + ty;
    if (tx < tileWidth && ty < TILE_HEIGHT && outX < width && outY < height)
    {
        // The corresponding location in shared memory.
        int sharedY = apron + ty;
        unsigned char minVal = 255;
        // The vertical window in shared memory spans from (sharedY - (p-1)) to (sharedY + (p-1)).
        int windowStart = sharedY - (p - 1);
        int windowEnd = sharedY + (p - 1);
        for (int r = windowStart; r <= windowEnd; r++)
        {
            unsigned char val = s_tile[r * tileWidth + tx];
            minVal = min(minVal, val);
        }
        // Write the computed erosion value to global memory.
        d_output[outY * width + outX] = minVal;
    }
}

int main()
{
    // Load the grayscale image using OpenCV.
    cv::Mat img = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    int width = img.cols;
    int height = img.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imageSize, cudaMemcpyHostToDevice));

    int p = 7;  // Must be odd; vertical window height will be 2*p - 1.
    // Define grid dimensions based on tile size.
    dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    // Shared memory per block: TILE_WIDTH * (TILE_HEIGHT + 2*(p-1)) bytes.
    size_t sharedMemSize = TILE_WIDTH * (TILE_HEIGHT + 2 * (p - 1)) * sizeof(unsigned char);

    verticalErosionTiledKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, width, height, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cv::Mat output(img.size(), img.type());
    CUDA_CHECK(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    cv::imshow("Input", img);
    cv::imshow("Vertical Eroded Output (Tiled)", output);
    cv::waitKey(0);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}