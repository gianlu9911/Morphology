#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for 1xM (horizontal) morphological erosion using shared memory with multiple rows per block
__global__ void erosionKernel1xM_shared_multiRow(
    const unsigned char* d_input, 
    unsigned char* d_output, 
    int rows, 
    int cols, 
    size_t pitch, 
    int kernelSize)
{
    extern __shared__ unsigned char s_data[];
    int radius = kernelSize / 2;
    
    // Calculate global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Compute shared memory row offset for the current thread row within the block
    int sharedRowWidth = blockDim.x + 2 * radius;
    int rowOffset = threadIdx.y * sharedRowWidth;
    int sharedIndex = rowOffset + threadIdx.x + radius;
    
    // Only process valid rows
    if (y >= rows) return;
    
    // Get pointer to the start of the row in global memory
    const unsigned char* rowPtr = d_input + y * pitch;
    
    // Load main data into shared memory for this row
    if (x < cols)
        s_data[sharedIndex] = rowPtr[x];
    else
        s_data[sharedIndex] = 255;
    
    // Load left halo for this row
    if (threadIdx.x < radius) {
        int haloX = x - radius;
        s_data[rowOffset + threadIdx.x] = (haloX >= 0) ? rowPtr[haloX] : 255;
    }
    
    // Load right halo for this row
    if (threadIdx.x >= blockDim.x - radius) {
        int haloX = x + radius;
        int s_idx = rowOffset + threadIdx.x + 2 * radius; // correct offset in shared memory for right halo
        s_data[s_idx] = (haloX < cols) ? rowPtr[haloX] : 255;
    }
    
    __syncthreads();
    
    // Only process valid columns
    if (x >= cols)
        return;
    
    // Compute the minimum over the kernel window using shared memory
    unsigned char minVal = 255;
    for (int i = -radius; i <= radius; i++) {
        unsigned char val = s_data[sharedIndex + i];
        if (val < minVal)
            minVal = val;
    }
    
    // Write the result to output using pitched addressing
    unsigned char* outRow = d_output + y * pitch;
    outRow[x] = minVal;
}

int main()
{
    // Load the image in grayscale
    cv::Mat img = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    
    int rows = img.rows;
    int cols = img.cols;
    size_t imageSize = rows * cols * sizeof(unsigned char);

    // Allocate pinned host memory for input and output
    unsigned char* h_inputPinned = nullptr;
    unsigned char* h_outputPinned = nullptr;
    cudaHostAlloc(&h_inputPinned, imageSize, cudaHostAllocDefault);
    cudaHostAlloc(&h_outputPinned, imageSize, cudaHostAllocDefault);
    memcpy(h_inputPinned, img.data, imageSize);

    // Device memory pointers and pitch
    unsigned char *d_input = nullptr, *d_output = nullptr;
    size_t d_pitch = 0;

    // Allocate device pitched memory
    cudaMallocPitch(&d_input, &d_pitch, cols * sizeof(unsigned char), rows);
    cudaMallocPitch(&d_output, &d_pitch, cols * sizeof(unsigned char), rows);

    // Copy from pinned host memory to device pitched memory
    cudaMemcpy2D(d_input, d_pitch, h_inputPinned, cols * sizeof(unsigned char),
                 cols * sizeof(unsigned char), rows, cudaMemcpyHostToDevice);

    int kernelSize = 7;  // must be odd
    int radius = kernelSize / 2;
    
    // Use a 2D block: for example, 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    // Shared memory size: blockDim.y rows * (blockDim.x + 2*radius) per row
    size_t sharedMemSize = block.y * (block.x + 2 * radius) * sizeof(unsigned char);

    // Launch kernel
    erosionKernel1xM_shared_multiRow<<<grid, block, sharedMemSize>>>(d_input, d_output, rows, cols, d_pitch, kernelSize);
    cudaDeviceSynchronize();

    // Copy result back to pinned host memory
    cudaMemcpy2D(h_outputPinned, cols * sizeof(unsigned char), d_output, d_pitch,
                 cols * sizeof(unsigned char), rows, cudaMemcpyDeviceToHost);

    // Create cv::Mat headers and display
    cv::Mat pinnedInput(rows, cols, CV_8UC1, h_inputPinned);
    cv::Mat pinnedOutput(rows, cols, CV_8UC1, h_outputPinned);
    cv::imshow("Original Image (Pinned Input)", pinnedInput);
    cv::imshow("Eroded Image (Multi-row per Block)", pinnedOutput);
    cv::waitKey(0);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_inputPinned);
    cudaFreeHost(h_outputPinned);

    return 0;
}
