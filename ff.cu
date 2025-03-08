#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

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

int main2()
{
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
    int radius = 3;

    // Launch vertical erosion kernel.
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    verticalErosionKernel<<<numBlocks, blockSize>>>(d_input, d_output_vertical, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch the optimized horizontal erosion kernel.
    // We use a 2D grid: each block processes a segment of one row.
    // Define the block width (tile width) for the horizontal kernel.
    int tileWidth = 256;
    dim3 blockDim(tileWidth, 1, 1);
    // Grid covers the full width (in x) and every row (in y).
    dim3 gridDim((width + tileWidth - 1) / tileWidth, height, 1);
    // Shared memory size: tile plus halo on both sides.
    size_t sharedMemSize = (tileWidth + 2 * radius) * sizeof(unsigned char);

    horizontalErosionKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_output_vertical, d_output_horizontal, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

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



// CUDA kernel for vertical dilation
__global__ void verticalDilationKernel(const unsigned char* input,
                                       unsigned char* output,
                                       int width, int height,
                                       int radius)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx >= totalPixels) return;

    int col = idx % width;
    int row = idx / width;

    int rowStart = max(0, row - radius);
    int rowEnd = min(height - 1, row + radius);

    unsigned char maxVal = 0;
    for (int r = rowStart; r <= rowEnd; ++r)
    {
        unsigned char pixel = input[r * width + col];
        maxVal = max(maxVal, pixel);
    }
    output[idx] = maxVal;
}

// CUDA kernel for horizontal dilation with shared memory
__global__ void horizontalDilationKernelShared(const unsigned char* input,
                                               unsigned char* output,
                                               int width, int height,
                                               int radius)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned char s_data[];

    int s_index = threadIdx.x + radius;

    if (col < width)
        s_data[s_index] = input[row * width + col];
    else
        s_data[s_index] = 0;

    if (threadIdx.x < radius) {
        int halo_col = blockIdx.x * blockDim.x + threadIdx.x - radius;
        s_data[threadIdx.x] = (halo_col >= 0) ? input[row * width + halo_col] : 0;
    }
    
    int rightHaloIndex = threadIdx.x + blockDim.x + radius;
    int halo_col = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
    if (threadIdx.x < radius) {
        s_data[rightHaloIndex] = (halo_col < width) ? input[row * width + halo_col] : 0;
    }
    
    __syncthreads();

    if (col < width)
    {
        unsigned char maxVal = 0;
        for (int offset = -radius; offset <= radius; ++offset)
        {
            maxVal = max(maxVal, s_data[s_index + offset]);
        }
        output[row * width + col] = maxVal;
    }
}

__global__ void subtractionKernel(unsigned char *input, unsigned char *output, unsigned char *reference, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        output[idx] = max(0, input[idx] - reference[idx]);
    }
}
int main3()
{
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

    // Set dilation radius.
    int radius = 15;

    // --- Vertical Dilation ---
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    verticalDilationKernel<<<numBlocks, blockSize>>>(d_input, d_output_vertical, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch the optimized horizontal dilation kernel.
    int tileWidth = 256;
    dim3 blockDim(tileWidth, 1, 1);
    dim3 gridDim((width + tileWidth - 1) / tileWidth, height, 1);
    size_t sharedMemSize = (tileWidth + 2 * radius) * sizeof(unsigned char);

    horizontalDilationKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_output_vertical, d_output_horizontal, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Vertical Erosion ---
    unsigned char *d_output_vertical_erosion = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_vertical_erosion, imgSize));
    
    verticalErosionKernel<<<numBlocks, blockSize>>>(d_input, d_output_vertical_erosion, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch the optimized horizontal erosion kernel.
    unsigned char *d_output_horizontal_erosion = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_horizontal_erosion, imgSize));
    
    horizontalErosionKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_output_vertical_erosion, d_output_horizontal_erosion, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Top-hat and Black-hat ---
    unsigned char *d_top_hat = nullptr, *d_black_hat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_top_hat, imgSize));
    CUDA_CHECK(cudaMalloc(&d_black_hat, imgSize));

    // Top-hat: Original Image - Opened Image (Erosion followed by Dilation)
    unsigned char *d_opened_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_opened_image, imgSize));

    // Perform Opening (Erosion followed by Dilation)
    verticalErosionKernel<<<numBlocks, blockSize>>>(d_input, d_opened_image, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    horizontalErosionKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_opened_image, d_opened_image, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Subtract Opened Image from Original Image to get Top-hat
    subtractionKernel<<<numBlocks, blockSize>>>(d_input, d_top_hat, d_opened_image, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Black-hat: Closed Image - Original Image (Dilation followed by Erosion)
    unsigned char *d_closed_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_closed_image, imgSize));

    // Perform Closing (Dilation followed by Erosion)
    verticalDilationKernel<<<numBlocks, blockSize>>>(d_input, d_closed_image, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());
    horizontalDilationKernelShared<<<gridDim, blockDim, sharedMemSize>>>(d_closed_image, d_closed_image, width, height, radius);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Subtract Original Image from Closed Image to get Black-hat
    subtractionKernel<<<numBlocks, blockSize>>>(d_closed_image, d_black_hat, d_input, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the processed images back from device to host.
    cv::Mat outputVertical(height, width, CV_8UC1);
    cv::Mat outputHorizontal(height, width, CV_8UC1);
    cv::Mat outputVerticalErosion(height, width, CV_8UC1);
    cv::Mat outputHorizontalErosion(height, width, CV_8UC1);
    cv::Mat topHatImage(height, width, CV_8UC1);
    cv::Mat blackHatImage(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(outputVertical.ptr(), d_output_vertical, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputHorizontal.ptr(), d_output_horizontal, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputVerticalErosion.ptr(), d_output_vertical_erosion, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outputHorizontalErosion.ptr(), d_output_horizontal_erosion, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(topHatImage.ptr(), d_top_hat, imgSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blackHatImage.ptr(), d_black_hat, imgSize, cudaMemcpyDeviceToHost));

    // Resize images for visualization.
    cv::Mat image_resized, outputVertical_resized, outputHorizontal_resized;
    cv::Mat outputVerticalErosion_resized, outputHorizontalErosion_resized;
    cv::Mat topHatImage_resized, blackHatImage_resized;
    cv::resize(image, image_resized, cv::Size(), 0.05, 0.05);
    cv::resize(outputVertical, outputVertical_resized, cv::Size(), 0.05, 0.05);
    cv::resize(outputHorizontal, outputHorizontal_resized, cv::Size(), 0.05, 0.05);
    cv::resize(outputVerticalErosion, outputVerticalErosion_resized, cv::Size(), 0.05, 0.05);
    cv::resize(outputHorizontalErosion, outputHorizontalErosion_resized, cv::Size(), 0.05, 0.05);
    cv::resize(topHatImage, topHatImage_resized, cv::Size(), 0.05, 0.05);
    cv::resize(blackHatImage, blackHatImage_resized, cv::Size(), 0.05, 0.05);

    // Add labels to the images.
    cv::putText(outputVertical_resized, "Vertical Dilation", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(outputHorizontal_resized, "Horizontal Dilation", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(outputVerticalErosion_resized, "Vertical Erosion", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(outputHorizontalErosion_resized, "Horizontal Erosion", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(topHatImage_resized, "Top-hat", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);
    cv::putText(blackHatImage_resized, "Black-hat", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, 8);

    // Concatenate images horizontally for display.
    cv::Mat combined1, combined2, combined3, combined4, finalDisplay;
    cv::hconcat(image_resized, outputVertical_resized, combined1);
    cv::hconcat(outputVertical_resized, outputHorizontal_resized, combined2);
    cv::hconcat(outputVerticalErosion_resized, outputHorizontalErosion_resized, combined3);
    cv::hconcat(topHatImage_resized, blackHatImage_resized, combined4);
    cv::vconcat(combined1, combined2, finalDisplay);
    cv::vconcat(finalDisplay, combined3, finalDisplay);
    cv::vconcat(finalDisplay, combined4, finalDisplay);

    cv::imshow("Morphological Operations", finalDisplay);
    cv::waitKey(0);

    // Free memory.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_vertical));
    CUDA_CHECK(cudaFree(d_output_horizontal));
    CUDA_CHECK(cudaFree(d_output_vertical_erosion));
    CUDA_CHECK(cudaFree(d_output_horizontal_erosion));
    CUDA_CHECK(cudaFree(d_top_hat));
    CUDA_CHECK(cudaFree(d_black_hat));
    CUDA_CHECK(cudaFree(d_opened_image));
    CUDA_CHECK(cudaFree(d_closed_image));

    return 0;
}


__global__ void horizontalErosionKernelWithTiling(unsigned char *d_input, unsigned char *d_output, int width, int height, int radius, unsigned char paddingValue) {
    // Calculate the global thread index in 1D grid
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process the pixels within the image bounds
    if (idx < width * height) {
        int row = idx / width;
        int col = idx % width;

        // Tiling: Use the halo region to avoid out-of-bound accesses
        int startCol = max(0, col - radius);
        int endCol = min(width - 1, col + radius);

        unsigned char minValue = 255; // Max value for erosion

        // Loop over the horizontal tile (row) within the range defined by the radius
        for (int i = startCol; i <= endCol; i++) {
            // Handle boundary by using padding value when outside image bounds
            unsigned char pixelValue = (i >= 0 && i < width) ? d_input[row * width + i] : paddingValue;
            minValue = min(minValue, pixelValue);
        }

        // Write the result to the output image
        d_output[idx] = minValue;
    }
}

void erosionWithTilingPadding(cv::Mat &image, int radius, unsigned char paddingValue) {
    int width = image.cols;
    int height = image.rows;
    int totalPixels = width * height;
    size_t imgSize = totalPixels * sizeof(unsigned char);

    // Allocate device memory for input and output
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, imgSize));
    CUDA_CHECK(cudaMalloc(&d_output, imgSize));

    // Copy the input image from host to device
    CUDA_CHECK(cudaMemcpy(d_input, image.ptr(), imgSize, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    int blockSize = 256;
    int numBlocks = (totalPixels + blockSize - 1) / blockSize;
    
    // Launch the kernel for erosion with tiling and padding
    horizontalErosionKernelWithTiling<<<numBlocks, blockSize>>>(d_input, d_output, width, height, radius, paddingValue);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host
    cv::Mat output(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(output.ptr(), d_output, imgSize, cudaMemcpyDeviceToHost));

    // Display the result (for visualization purposes)
    cv::imshow("Erosion with Tiling and Padding", output);
    cv::waitKey(0);

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Load the image
    cv::Mat image = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    int radius = 3;  // Example radius for erosion
    unsigned char paddingValue = 0;  // Use 0 (black) for padding

    // Call the erosion function with tiling and padding
    erosionWithTilingPadding(image, radius, paddingValue);

    return 0;
}
