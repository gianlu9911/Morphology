#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define HISTOGRAM_SIZE 256
#define BLOCK_SIZE 256  

// Compute histogram using pitched memory. The kernel uses a striding pattern to process all pixels.
__global__ void computeHistogram(const unsigned char *d_input, size_t pitch, int *d_hist, int width, int height) {
    __shared__ int hist_shared[HISTOGRAM_SIZE];

    // Initialize shared histogram
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        hist_shared[i] = 0;
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process each pixel in a strided loop
    while (idx < width * height) {
        int row = idx / width;
        int col = idx % width;
        // Compute address using the pitch
        unsigned char pixel = *((unsigned char*)((char*)d_input + row * pitch) + col);
        atomicAdd(&hist_shared[pixel], 1);
        idx += stride;
    }
    __syncthreads();

    // Merge shared histogram into global histogram
    for (int i = threadIdx.x; i < HISTOGRAM_SIZE; i += blockDim.x) {
        atomicAdd(&d_hist[i], hist_shared[i]);
    }
}

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

void histogramEqualizationCUDA(cv::Mat &inputImage) {
    int width = inputImage.cols;
    int height = inputImage.rows;
    int total_pixels = width * height;
    
    unsigned char *d_input;
    int *d_hist, *d_cdf;
    size_t pitch;
    
    // Allocate pitched memory for the image on device
    cudaMallocPitch(&d_input, &pitch, width * sizeof(unsigned char), height);
    cudaMalloc(&d_hist, HISTOGRAM_SIZE * sizeof(int));
    cudaMalloc(&d_cdf, HISTOGRAM_SIZE * sizeof(int));

    // Copy image from host to device with 2D copy (using the pitch)
    cudaMemcpy2D(d_input, pitch, inputImage.data, width * sizeof(unsigned char),
                 width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, HISTOGRAM_SIZE * sizeof(int));

    int numBlocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeHistogram<<<numBlocks, BLOCK_SIZE>>>(d_input, pitch, d_hist, width, height);
    computeCDF<<<1, HISTOGRAM_SIZE>>>(d_hist, d_cdf, total_pixels);

    // Retrieve CDF to compute cdf_min
    int h_cdf[HISTOGRAM_SIZE];
    cudaMemcpy(h_cdf, d_cdf, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    int cdf_min = 0;
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        if (h_cdf[i] > 0) {
            cdf_min = h_cdf[i];
            break;
        }
    }

    // Apply histogram equalization on device
    equalizeHistogram<<<numBlocks, BLOCK_SIZE>>>(d_input, pitch, d_cdf, width, height, cdf_min, total_pixels);

    // Copy the result back to host using cudaMemcpy2D
    cudaMemcpy2D(inputImage.data, width * sizeof(unsigned char), d_input, pitch,
                 width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    // Free allocated device memory
    cudaFree(d_input);
    cudaFree(d_hist);
    cudaFree(d_cdf);
}

int main3() {
    cv::Mat inputImage = cv::imread("../imgs/lena_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    histogramEqualizationCUDA(inputImage);

    cv::imshow("Equalized Image", inputImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define BINS 256       // For grayscale images (0-255)
#define TILE_W 32      // Tile width (and blockDim.x)
#define TILE_H 32      // Tile height (and blockDim.y)

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
    
    // Allocate host memory for histogram and initialize it to zero.
    std::vector<int> h_histogram(BINS, 0);
    
    // Allocate device memory.
    unsigned char *d_image;
    int *d_histogram;
    cudaMalloc(&d_image, imgSize);
    cudaMalloc(&d_histogram, BINS * sizeof(int));
    cudaMemset(d_histogram, 0, BINS * sizeof(int));
    
    // Copy image data to GPU.
    cudaMemcpy(d_image, img.data, imgSize, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions.
    dim3 blockDim(TILE_W, TILE_H);
    // Calculate grid dimensions, ensuring we cover the entire image.
    dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);
    
    // Launch the tiled histogram kernel.
    image_histogram_tiled<<<gridDim, blockDim>>>(d_image, d_histogram, width, height);
    cudaDeviceSynchronize();
    
    // Copy the histogram back to the host.
    cudaMemcpy(h_histogram.data(), d_histogram, BINS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print out the histogram bins.
    for (int i = 0; i < BINS; i++) {
        std::cout << "Bin " << i << ": " << h_histogram[i] << std::endl;
    }
    
    // Cleanup.
    cudaFree(d_image);
    cudaFree(d_histogram);
    
    return 0;
}
