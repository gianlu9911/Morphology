#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to divide the image into tiles with overlap
__global__ void divideIntoTilesWithOverlap(uchar* image, int* tiles, int imageWidth, int imageHeight, int N, int p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int p2_minus_1 = 2 * p - 1;  // Total window size (p*2 - 1)
    int overlap = (p - 1) / 2;    // Apron size for overlap

    if (idx < imageHeight) {
        for (int col = overlap; col < imageWidth - overlap; ++col) {
            // Create a window of size p*2 - 1 (dilation window)
            for (int i = 0; i < p2_minus_1; ++i) {
                // Copy data into the tiles array with overlap
                int tileIdx = (idx * imageWidth + col) * p2_minus_1 + i;
                tiles[tileIdx] = image[(idx * imageWidth) + (col + i - overlap)];
            }
        }
    }
}

// Kernel to compute the max arrays (suffix and prefix max)
__global__ void computeMaxArrays(int* tiles, int* s, int* r, int imageWidth, int imageHeight, int N, int p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int p2_minus_1 = 2 * p - 1;  // Window size (p * 2 - 1)

    if (idx < imageHeight * imageWidth) {
        int row = idx / imageWidth;
        int col = idx % imageWidth;

        if (col >= (p - 1) && col < (imageWidth - (p - 1))) {
            // Window indices: [col - (p-1), col + (p-1)]
            // We need to fill s and r arrays for the window

            // Suffix max array (s) for pixels 0 to p-1 in window
            for (int i = 0; i < p; i++) {
                s[idx * p + i] = tiles[(row * imageWidth + col) * p + i];
            }
            for (int i = p - 2; i >= 0; --i) {
                s[idx * p + i] = max(s[idx * p + i], s[idx * p + i + 1]);
            }

            // Prefix max array (r) for pixels p-1 to p*2-1 in window
            for (int i = 0; i < p; i++) {
                r[idx * p + i] = tiles[(row * imageWidth + col) * p + (p-1+i)];
            }
            for (int i = 1; i < p; i++) {
                r[idx * p + i] = max(r[idx * p + i], r[idx * p + i - 1]);
            }
        }
    }
}

// Kernel to compute the dilation result
__global__ void computeDilationResult(int* s, int* r, uchar* result, int imageWidth, int imageHeight, int p) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < imageHeight * imageWidth) {
        int row = idx / imageWidth;
        int col = idx % imageWidth;

        if (col >= (p - 1) / 2 && col < (imageWidth - (p - 1) / 2)) {
            // We are within the valid region for dilation calculation

            // Calculate the index for the suffix and prefix max arrays
            int sIndex = idx * p + (col - (p - 1) / 2);
            int rIndex = idx * p + (col + (p - 1) / 2);

            // Perform dilation computation: result[j] = max(s[j-(p-1)/2], r[j+(p-1)/2])
            result[idx] = max(s[sIndex], r[rIndex]);
        }
    }
}

int main() {
    // Read the image using OpenCV
    std::string imagePath = "../imgs/lena.jpg";
    cv::Mat h_image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE); // Read as grayscale for simplicity

    if (h_image.empty()) {
        std::cerr << "Failed to load image from path: " << imagePath << std::endl;
        return -1;
    }

    // Get image dimensions
    int imageWidth = h_image.cols;
    int imageHeight = h_image.rows;

    std::cout << "Image dimensions: " << imageWidth << "x" << imageHeight << std::endl;

    // Parameters for the structuring element (example, N=3 means p=7)
    int N = 3;
    int p = 2 * N + 1;

    // Allocate device memory for image and tiles
    uchar* d_image;
    int* d_tiles;
    int* d_s;
    int* d_r;
    uchar* d_result;
    uchar* h_image_data = new uchar[imageWidth * imageHeight];  // Host memory for image
    int* h_tiles = new int[imageHeight * imageWidth * (2 * p - 1)];  // Host memory for tiles (int)
    int* h_s = new int[imageHeight * imageWidth * p]; // Host memory for suffix max (int)
    int* h_r = new int[imageHeight * imageWidth * p]; // Host memory for prefix max (int)
    uchar* h_result = new uchar[imageHeight * imageWidth]; // Host memory for final dilation result

    // Convert the image to a 1D array and copy to device
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            h_image_data[i * imageWidth + j] = h_image.at<uchar>(i, j);
        }
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_image, imageWidth * imageHeight * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&d_tiles, imageHeight * imageWidth * (2 * p - 1) * sizeof(int)));  // Changed to int
    CUDA_CHECK(cudaMalloc(&d_s, imageHeight * imageWidth * p * sizeof(int)));  // Changed to int
    CUDA_CHECK(cudaMalloc(&d_r, imageHeight * imageWidth * p * sizeof(int)));  // Changed to int
    CUDA_CHECK(cudaMalloc(&d_result, imageHeight * imageWidth * sizeof(uchar)));

    // Copy image to device
    CUDA_CHECK(cudaMemcpy(d_image, h_image_data, imageWidth * imageHeight * sizeof(uchar), cudaMemcpyHostToDevice));

    // Launch kernel to divide into tiles with overlap
    int blockSize = 256;
    int numBlocks = (imageHeight + blockSize - 1) / blockSize;
    divideIntoTilesWithOverlap<<<numBlocks, blockSize>>>(d_image, d_tiles, imageWidth, imageHeight, N, p);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel errors

    // Launch kernel to compute the max arrays
    computeMaxArrays<<<numBlocks, blockSize>>>(d_tiles, d_s, d_r, imageWidth, imageHeight, N, p);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel errors

    // Launch kernel to compute the final dilation result
    computeDilationResult<<<numBlocks, blockSize>>>(d_s, d_r, d_result, imageWidth, imageHeight, p);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel errors

    // Copy the results back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, imageHeight * imageWidth * sizeof(uchar), cudaMemcpyDeviceToHost));

    // Save the result to a new image
    cv::Mat result_image(imageHeight, imageWidth, CV_8UC1, h_result);
    cv::imwrite("../imgs/lena_dilated.jpg", result_image);

    // Free memory
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_tiles));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_result));
    delete[] h_image_data;
    delete[] h_tiles;
    delete[] h_s;
    delete[] h_r;
    delete[] h_result;

    return 0;
}
