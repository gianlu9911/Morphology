#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256  


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define BINS 256       // For grayscale images (0-255)
#define TILE_W 32      // Tile width (and blockDim.x)
#define TILE_H 32      // Tile height (and blockDim.y)

// Compute the Cumulative Distribution Function (CDF) using an inclusive scan (Hillis-Steele algorithm)
__global__ void computeCDF(int *d_hist, int *d_cdf, int total_pixels) {
    __shared__ int temp[HISTOGRAM_SIZE];
    
    int tid = threadIdx.x;
    if (tid < HISTOGRAM_SIZE) {
        temp[tid] = d_hist[tid];
    }
    __syncthreads();

    // Inclusive scan
    for (int offset = 1; offset < HISTOGRAM_SIZE; offset *= 2) {
        int val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        if (tid >= offset) temp[tid] += val;
        __syncthreads();
    }

    if (tid < HISTOGRAM_SIZE) {
        d_cdf[tid] = temp[tid];
    }
}

// Apply histogram equalization. Note that d_image uses pitched memory.
__global__ void equalizeHistogram(unsigned char *d_image, size_t pitch, const int *d_cdf, int width, int height, int cdf_min, int total_pixels) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    while (idx < width * height) {
        int row = idx / width;
        int col = idx % width;
        unsigned char pixel = *((unsigned char*)((char*)d_image + row * pitch) + col);
        // Compute new pixel value using the histogram equalization formula
        unsigned char newVal = (unsigned char)(((d_cdf[pixel] - cdf_min) * 255) / (total_pixels - cdf_min));
        *((unsigned char*)((char*)d_image + row * pitch) + col) = newVal;
        idx += stride;
    }
}

__global__ void image_histogram_tiled(const unsigned char *image, int *globalHist, int width, int height) {
    // 2D indices within the block (tile)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Global image coordinates for this thread
    int col = blockIdx.x * TILE_W + tx;
    int row = blockIdx.y * TILE_H + ty;
    
    // Allocate shared memory for the tile
    __shared__ unsigned char tile[TILE_H][TILE_W];
    // Allocate shared memory for the block's histogram (local histogram)
    __shared__ int localHist[BINS];
    
    // Initialize the local histogram; use a 1D index for initialization.
    int tid = ty * blockDim.x + tx;
    // Only the first BINS threads initialize the histogram
    if (tid < BINS) {
        localHist[tid] = 0;
    }
    __syncthreads();
    
    // Load the tile from global memory into shared memory (coalesced access)
    if (row < height && col < width) {
        tile[ty][tx] = image[row * width + col];
    } else {
        // For threads outside image bounds, set a default value (won't affect histogram)
        tile[ty][tx] = 0;
    }
    __syncthreads();
    
    // Each thread processes its pixel in the tile (if valid) and updates the local histogram.
    if (row < height && col < width) {
        unsigned char pixel = tile[ty][tx];
        // Atomic update on shared local histogram.
        atomicAdd(&localHist[pixel], 1);
    }
    __syncthreads();
    
    // Merge the block-local histogram into the global histogram.
    // Use the first BINS threads in the block.
    if (tid < BINS) {
        atomicAdd(&globalHist[tid], localHist[tid]);
    }
}


int main() {
    // Load the grayscale image using OpenCV.
    cv::Mat img = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int imgSize = width * height;
    
    // Allocate host memory for histogram and CDF.
    std::vector<int> h_histogram(BINS, 0);
    std::vector<int> h_cdf(BINS, 0);
    
    // Allocate device memory.
    unsigned char *d_image;
    int *d_histogram, *d_cdf;
    size_t pitch;
    cudaMallocPitch(&d_image, &pitch, width * sizeof(unsigned char), height);
    cudaMalloc(&d_histogram, BINS * sizeof(int));
    cudaMalloc(&d_cdf, BINS * sizeof(int));
    cudaMemset(d_histogram, 0, BINS * sizeof(int));
    
    // Copy image data to GPU.
    cudaMemcpy2D(d_image, pitch, img.data, width * sizeof(unsigned char), width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions.
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);
    
    // Launch the tiled histogram kernel.
    image_histogram_tiled<<<gridDim, blockDim>>>(d_image, d_histogram, width, height);
    cudaDeviceSynchronize();
    
    // Copy the histogram back to the host.
    cudaMemcpy(h_histogram.data(), d_histogram, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute CDF on the GPU.
    computeCDF<<<1, BINS>>>(d_histogram, d_cdf, imgSize);
    cudaMemcpy(h_cdf.data(), d_cdf, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Find minimum nonzero CDF value.
    int cdf_min = 0;
    for (int i = 0; i < BINS; ++i) {
        if (h_cdf[i] > 0) {
            cdf_min = h_cdf[i];
            break;
        }
    }
    
    // Perform histogram equalization.
    int threadsPerBlock = 256;
    int numBlocks = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    equalizeHistogram<<<numBlocks, threadsPerBlock>>>(d_image, pitch, d_cdf, width, height, cdf_min, imgSize);
    cudaDeviceSynchronize();
    
    // Copy the equalized image back to the host.
    cv::Mat equalized_img(height, width, CV_8UC1);
    cudaMemcpy2D(equalized_img.data, width * sizeof(unsigned char), d_image, pitch, width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);
    
    // Save and display the result.
    // cv::imwrite("equalized_image.jpg", equalized_img);
    cv::imshow("Equalized Image", equalized_img);
    cv::waitKey(0);
    
    // Cleanup.
    cudaFree(d_image);
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    
    return 0;
}

