#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

// Define tile sizes for the vertical (tiled) kernel.
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

//---------------------------------------------------------------------
// Helper function for vectorized 32-bit loads from global memory.
// Assumes d_input is 8-bit data that is 4-byte aligned and index is in [0, numPixels).
// Loads 32 bits (4 bytes) at a time and extracts the byte corresponding to 'index'.
__device__ inline unsigned char loadPixel(const unsigned char* d_input, int index) {
    int intIndex = index / 4;           // which 32-bit word
    int offset   = index % 4;           // which byte inside that word
    int data = ((const int*)d_input)[intIndex]; // 32-bit load
    return (data >> (8 * offset)) & 0xFF;
}
//---------------------------------------------------------------------

/***********************************
 * Horizontal Erosion Kernel (8-bit)
 ***********************************/
// For an output pixel at (row, col), the full horizontal window (of length 2*p - 1)
// extends from col - (p-1) to col + (p-1). We split this window into left and right
// sections and use warp-level reduction (via __shfl_down_sync) to compute the minimum.
/***********************************
 * Horizontal Erosion Tiled Kernel (8-bit)
 ***********************************/
// For an output pixel at (row, col), the full horizontal window (of length 2*p - 1)
// extends from col - (p-1) to col + (p-1). Each block processes a TILE_HEIGHT x TILE_WIDTH
// tile of output pixels and loads a corresponding tile from global memory with extra columns
// (apron) on the left and right.
__global__ void horizontalErosionTiledKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int p) // p must be odd; horizontal window width = 2*p - 1
{
    const int apron = p - 1;
    // Shared tile width includes the extra columns on left and right.
    const int tileWidthShared = TILE_WIDTH + 2 * apron;
    const int tileHeight = TILE_HEIGHT;
    
    // Compute the starting coordinates for this tile in the image.
    int tileStartX = blockIdx.x * TILE_WIDTH;
    int tileStartY = blockIdx.y * tileHeight;
    
    // Declare shared memory.
    // Shared memory size should be allocated as: TILE_HEIGHT * (TILE_WIDTH + 2*(p-1)) bytes.
    extern __shared__ unsigned char s_tile[];
    
    // Load the shared memory tile.
    // Each thread loads one or more elements in a loop.
    for (int y = threadIdx.y; y < tileHeight; y += blockDim.y) {
        int globalY = tileStartY + y;
        // Clamp the row index if needed.
        globalY = (globalY < 0) ? 0 : (globalY >= height ? height - 1 : globalY);
        for (int x = threadIdx.x; x < tileWidthShared; x += blockDim.x) {
            // Compute global x. The shared memory tile starts at global x = tileStartX - apron.
            int globalX = tileStartX + x - apron;
            // Clamp the column index to image bounds.
            globalX = (globalX < 0) ? 0 : (globalX >= width ? width - 1 : globalX);
            int index = globalY * width + globalX;
            s_tile[y * tileWidthShared + x] = d_input[index];
        }
    }
    __syncthreads();
    
    // Compute the output pixel location.
    int outX = tileStartX + threadIdx.x;
    int outY = tileStartY + threadIdx.y;
    
    // Ensure we only process valid output pixels.
    if (threadIdx.x < TILE_WIDTH && threadIdx.y < tileHeight && outX < width && outY < height) {
        // In shared memory, the current pixel is at column (threadIdx.x + apron).
        int sharedRow = threadIdx.y;
        int sharedCol = threadIdx.x + apron;
        unsigned char minVal = 255;
        // Compute horizontal erosion over the window [sharedCol - (p-1), sharedCol + (p-1)].
        int windowStart = sharedCol - (p - 1);
        int windowEnd   = sharedCol + (p - 1);
#pragma unroll
        for (int i = windowStart; i <= windowEnd; i++) {
            unsigned char val = s_tile[sharedRow * tileWidthShared + i];
            minVal = min(minVal, val);
        }
        d_output[outY * width + outX] = minVal;
    }
}


/***********************************
 * Vertical Erosion Tiled Kernel (8-bit)
 ***********************************/
// This kernel loads a tile (with extra rows as an apron) into shared memory
// using nested loops that allow coalesced 32-bit loads (via loadPixel).
// Then, each thread computes the vertical erosion (minimum over a vertical window)
// for its assigned pixel.
__global__ void verticalErosionTiledKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    int p) // p must be odd; vertical window height = 2*p - 1
{
    const int apron = p - 1;
    const int tileHeightShared = TILE_HEIGHT + 2 * apron;
    const int tileWidth = TILE_WIDTH;
    
    int tileStartX = blockIdx.x * tileWidth;
    int tileStartY = blockIdx.y * TILE_HEIGHT;
    
    extern __shared__ unsigned char s_tile[];
    
    // Load the shared memory tile using nested loops.
    for (int y = threadIdx.y; y < tileHeightShared; y += blockDim.y) {
        int globalY = tileStartY + y - apron;
        globalY = (globalY < 0) ? 0 : (globalY >= height ? height - 1 : globalY);
        for (int x = threadIdx.x; x < tileWidth; x += blockDim.x) {
            int globalX = tileStartX + x;
            int index = globalY * width + globalX;
            unsigned char val = 0;
            if (globalX < width)
                val = loadPixel(d_input, index);
            s_tile[y * tileWidth + x] = val;
        }
    }
    __syncthreads();
    
    int outX = tileStartX + threadIdx.x;
    int outY = tileStartY + threadIdx.y;
    if (threadIdx.x < tileWidth && threadIdx.y < TILE_HEIGHT && outX < width && outY < height) {
        int sharedY = apron + threadIdx.y;
        unsigned char minVal = 255;
        int windowStart = sharedY - (p - 1);
        int windowEnd   = sharedY + (p - 1);
#pragma unroll
        for (int r = windowStart; r <= windowEnd; r++) {
            unsigned char val = s_tile[r * tileWidth + threadIdx.x];
            minVal = min(minVal, val);
        }
        d_output[outY * width + outX] = minVal;
    }
}

/***********************************
 * Combined main() Function
 ***********************************/
int main()
{
    // Load a grayscale image using OpenCV.
    cv::Mat img = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    
    int width = img.cols;
    int height = img.rows;
    size_t numPixels = width * height;
    size_t imageSizeBytes = numPixels * sizeof(unsigned char);
    
    // We assume the image data are 8-bit (grayscale) and 4-byte aligned.
    // (If not, you may want to copy/align the data on the host before uploading.)
    
    // Allocate device memory.
    unsigned char *d_input = nullptr;
    unsigned char *d_outputHoriz = nullptr;
    unsigned char *d_outputVert = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imageSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_outputHoriz, imageSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_outputVert, imageSizeBytes));
    
    CUDA_CHECK(cudaMemcpy(d_input, img.data, imageSizeBytes, cudaMemcpyHostToDevice));
    
    int p = 7;  // Must be odd; full window length = 2*p - 1.
    
    /********** Horizontal Erosion **********/
    {
        int totalPixels = width * height;
        int threadsPerBlock = 256; // e.g., 256 threads per block.
        int warpsPerBlock = threadsPerBlock / WARP_SIZE;
        int blocks = (totalPixels + warpsPerBlock - 1) / warpsPerBlock;
        
        dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
        dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                    (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        size_t sharedMemSize = TILE_HEIGHT * (TILE_WIDTH + 2 * (p - 1)) * sizeof(unsigned char);

        horizontalErosionTiledKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_outputHoriz, width, height, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    /********** Vertical Erosion (Tiled) **********/
    {
        dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
        dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                     (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
        size_t sharedMemSize = TILE_WIDTH * (TILE_HEIGHT + 2 * (p - 1)) * sizeof(unsigned char);
        
        verticalErosionTiledKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_outputVert, width, height, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back to host.
    cv::Mat outputHoriz(img.size(), img.type());
    cv::Mat outputVert(img.size(), img.type());
    CUDA_CHECK(cudaMemcpy(outputHoriz.data, d_outputHoriz, imageSizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputVert.data, d_outputVert, imageSizeBytes, cudaMemcpyDeviceToHost));
    
    // Display the input and both eroded outputs.
    cv::imshow("Input", img);
    cv::imshow("Horizontal Eroded Output (8-bit, 32-bit loads)", outputHoriz);
    cv::imshow("Vertical Eroded Output (Tiled, 8-bit, 32-bit loads)", outputVert);
    cv::waitKey(0);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_outputHoriz));
    CUDA_CHECK(cudaFree(d_outputVert));



    
    return 0;
}
