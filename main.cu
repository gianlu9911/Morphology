#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Step 1: Extract window from a tile of a row from the image.
// Each tile is taken from the row starting at tileStart, and extended with
// an "apron" on each side so that the window size becomes 2*p - 1.
// Out-of-bound indices are clamped.
__global__ void extractWindowKernel(const int* image, int* windows,
                                      int width, int p, int numTiles) {
    // Use a 2D grid: blockIdx.x = tile index, blockIdx.y = row index.
    int tileIdx = blockIdx.x;
    int row = blockIdx.y;
    int apron = (p - 1) / 2;
    int windowSize = 2 * p - 1;
    // Compute the starting column for this tile.
    int tileStart = tileIdx * p;

    int tid = threadIdx.x;
    if (tid < windowSize) {
        // For this window element, compute the corresponding column in the image.
        int col = tileStart + tid - apron;
        // Clamp to image boundaries.
        if (col < 0) col = 0;
        if (col >= width) col = width - 1;
        // Each tile’s window is stored consecutively.
        // The global index: for row r, tile tileIdx, element tid.
        int tileOffset = (row * numTiles + tileIdx) * windowSize;
        windows[tileOffset + tid] = image[row * width + col];
    }
}

// ---------------------------------------------------------------------
// Step 2: For a given tile window, compute the suffix and prefix max arrays.
// The window is of size (2*p - 1). For the left half (indices 0 .. p-1)
// we compute a suffix max array s; for the right half (indices p-1 .. 2*p-2)
// we compute a prefix max array r.
// Two threads per tile are used. (Thread 0 processes the left side,
// thread 1 processes the right side.)
__global__ void scanKernel(const int* windows, int* d_s, int* d_r,
                             int p, int numTiles) {
    // 2D grid: blockIdx.x = tile index, blockIdx.y = row index.
    int tileIdx = blockIdx.x;
    int row = blockIdx.y;
    int windowSize = 2 * p - 1;
    int tileOffset = (row * numTiles + tileIdx) * windowSize;
    const int* w = windows + tileOffset;

    // Use dynamic shared memory: two arrays of size p.
    extern __shared__ int sharedMem[];
    int* left = sharedMem;       // for left half processing
    int* right = sharedMem + p;  // for right half processing

    int tid = threadIdx.x;
    // Global offset for s and r for this tile:
    int outOffset = (row * numTiles + tileIdx) * p;

    if (tid == 0) {
        // Process left half: load indices 0 .. p-1 in reverse order.
        for (int i = 0; i < p; i++) {
            left[i] = w[p - 1 - i];
        }
        // Serial prefix (max) scan on the reversed data.
        for (int i = 1; i < p; i++) {
            left[i] = max(left[i], left[i - 1]);
        }
        // Reverse the scanned result to get the suffix max array.
        for (int i = 0; i < p; i++) {
            d_s[outOffset + i] = left[p - 1 - i];
        }
    } else if (tid == 1) {
        // Process right half: load indices p-1 .. 2*p-2.
        for (int i = 0; i < p; i++) {
            right[i] = w[p - 1 + i];
        }
        // Serial prefix (max) scan.
        for (int i = 1; i < p; i++) {
            right[i] = max(right[i], right[i - 1]);
        }
        // Write out the prefix max array.
        for (int i = 0; i < p; i++) {
            d_r[outOffset + i] = right[i];
        }
    }
}

// ---------------------------------------------------------------------
// Step 3: Combine the prefix and suffix arrays to compute the dilation.
// For each pixel in the tile (of size p), the dilation result is computed as:
//    result[i] = max(s[i], r[i])
// This kernel is launched with one block per tile and p threads per block.
__global__ void dilationKernel(const int* d_s, const int* d_r, int* d_out,
                               int p, int numTiles) {
    // 2D grid: blockIdx.x = tile index, blockIdx.y = row index.
    int tileIdx = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (tid < p) {
        int outOffset = (row * numTiles + tileIdx) * p;
        int s_val = d_s[outOffset + tid];
        int r_val = d_r[outOffset + tid];
        d_out[outOffset + tid] = max(s_val, r_val);
    }
}

// ---------------------------------------------------------------------
// Main: Load an image, run the three steps (for horizontal dilation),
// and reassemble and save the output.
int main() {
    // 1. Load the input image (grayscale) using OpenCV.
    cv::Mat inImage = cv::imread("../imgs/lena.jpg", cv::IMREAD_GRAYSCALE);
    if (inImage.empty()) {
        std::cerr << "Error: cannot load image ../imgs/lena.jpg" << std::endl;
        return -1;
    }
    int width = inImage.cols;
    int height = inImage.rows;
    std::cout << "Loaded image: " << width << " x " << height << std::endl;

    // 2. Convert the image to 32-bit int (our kernels operate on int).
    cv::Mat inImageInt;
    inImage.convertTo(inImageInt, CV_32S);

    // 3. Set the structural element size.
    int p = 3;  // Must be odd; you can adjust this (e.g., 5, 7, etc.)
    int apron = (p - 1) / 2;
    int windowSize = 2 * p - 1;
    // We'll process each row in tiles of width p.
    int numTiles = (width + p - 1) / p; // ceiling division
    // The horizontally dilated image (per row) will be stored tile by tile.
    // Its effective width is: numTiles * p (it may be slightly wider than original).
    int outWidth = numTiles * p;

    // 4. Allocate device memory.
    size_t imageSize = width * height * sizeof(int);
    size_t outTileSize = height * numTiles * p * sizeof(int);      // for dilation result per tile
    size_t windowBufferSize = height * numTiles * windowSize * sizeof(int);
    size_t scanBufferSize = height * numTiles * p * sizeof(int);     // for each of s and r

    int *d_in = nullptr, *d_windows = nullptr;
    int *d_s = nullptr, *d_r = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, imageSize);
    cudaMalloc(&d_windows, windowBufferSize);
    cudaMalloc(&d_s, scanBufferSize);
    cudaMalloc(&d_r, scanBufferSize);
    cudaMalloc(&d_out, outTileSize);

    // 5. Copy input image data to device.
    cudaMemcpy(d_in, inImageInt.ptr<int>(), imageSize, cudaMemcpyHostToDevice);

    // 6. Launch Step 1: Extract windows for each tile of each row.
    //    Grid dimensions: (numTiles, height)
    dim3 gridExtract(numTiles, height);
    int blockExtract = windowSize; // one thread per window element
    extractWindowKernel<<<gridExtract, blockExtract>>>(d_in, d_windows, width, p, numTiles);
    cudaDeviceSynchronize();

    // 7. Launch Step 2: Compute prefix and suffix max arrays for each tile.
    //    Grid dimensions: (numTiles, height), 2 threads per block.
    dim3 gridScan(numTiles, height);
    int blockScan = 2;
    size_t sharedMemSize = 2 * p * sizeof(int);
    scanKernel<<<gridScan, blockScan, sharedMemSize>>>(d_windows, d_s, d_r, p, numTiles);
    cudaDeviceSynchronize();

    // 8. Launch Step 3: Compute dilation per tile.
    //    Grid dimensions: (numTiles, height), p threads per block.
    dim3 gridDilation(numTiles, height);
    int blockDilation = p;
    dilationKernel<<<gridDilation, blockDilation>>>(d_s, d_r, d_out, p, numTiles);
    cudaDeviceSynchronize();

    // 9. Copy the dilation (horizontal pass) result back to host.
    //     The result is stored tile-by-tile in a buffer of size: (height * numTiles * p)
    int* h_outTiles = new int[height * numTiles * p];
    cudaMemcpy(h_outTiles, d_out, outTileSize, cudaMemcpyDeviceToHost);

    // 10. Reassemble the output row from the tile results.
    //     Our intermediate output has width = numTiles * p.
    cv::Mat outImageInt(height, outWidth, CV_32S);
    for (int r = 0; r < height; r++) {
        for (int t = 0; t < numTiles; t++) {
            for (int i = 0; i < p; i++) {
                int col = t * p + i;
                // If the reassembled column exceeds the original width, clamp it.
                if(col < width)
                    outImageInt.at<int>(r, col) = h_outTiles[(r * numTiles + t) * p + i];
            }
        }
    }

    // 11. Convert the result to 8-bit and save.
    cv::Mat outImage;
    outImageInt.convertTo(outImage, CV_8U);
    if (!cv::imwrite("../output.jpg", outImage)) {
        std::cerr << "Error: cannot save output image to ../output.jpg" << std::endl;
        return -1;
    }
    std::cout << "Dilation completed. Output saved to ../output.jpg" << std::endl;

    // 12. Cleanup.
    delete[] h_outTiles;
    cudaFree(d_in);
    cudaFree(d_windows);
    cudaFree(d_s);
    cudaFree(d_r);
    cudaFree(d_out);

    return 0;
}
