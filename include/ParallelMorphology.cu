#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// Struct to encapsulate device image information.
struct DeviceImage {
    unsigned char* data;
    int width;
    int height;
    size_t pitch; // pitch (in bytes) for each row
};

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

//--------------------------------------------------
// Horizontal erosion kernel using shared memory
// Each thread block loads a tile of BLOCK_HEIGHT x (BLOCK_WIDTH + 2)
// (extra two columns for left/right halo)
//--------------------------------------------------
__global__ void erosionHorizontalKernelShared(DeviceImage d_img, DeviceImage d_out)
{
    // Shared memory tile: each row has BLOCK_WIDTH + 2 elements.
    __shared__ unsigned char s_tile[BLOCK_HEIGHT][BLOCK_WIDTH + 2];

    int tx = threadIdx.x;  // [0, BLOCK_WIDTH-1]
    int ty = threadIdx.y;  // [0, BLOCK_HEIGHT-1]
    int x = blockIdx.x * BLOCK_WIDTH + tx;
    int y = blockIdx.y * BLOCK_HEIGHT + ty;

    // Load the center pixel into shared memory (offset by 1 to leave room for left halo).
    if (y < d_img.height && x < d_img.width) {
        const unsigned char* row = d_img.data + y * d_img.pitch;
        s_tile[ty][tx + 1] = row[x];
    } else {
        s_tile[ty][tx + 1] = 0;
    }

    // Load left halo: only thread with tx==0 loads the left pixel.
    if (tx == 0) {
        int x_left = x - 1;
        if (y < d_img.height && x_left >= 0) {
            const unsigned char* row = d_img.data + y * d_img.pitch;
            s_tile[ty][0] = row[x_left];
        } else {
            s_tile[ty][0] = 0;
        }
    }

    // Load right halo: thread with tx == BLOCK_WIDTH-1 loads the right pixel.
    if (tx == BLOCK_WIDTH - 1) {
        int x_right = x + 1;
        if (y < d_img.height && x_right < d_img.width) {
            const unsigned char* row = d_img.data + y * d_img.pitch;
            s_tile[ty][tx + 2] = row[x_right];
        } else {
            s_tile[ty][tx + 2] = 0;
        }
    }

    __syncthreads();

    // Now perform the horizontal erosion: minimum of left, center, right.
    if (y < d_img.height && x < d_img.width) {
        unsigned char left   = s_tile[ty][tx];      // (tx+1-1)
        unsigned char center = s_tile[ty][tx + 1];
        unsigned char right  = s_tile[ty][tx + 2];
        unsigned char minVal = min(min(left, center), right);

        // Write result to the output image.
        unsigned char* outRow = d_out.data + y * d_out.pitch;
        outRow[x] = minVal;
    }
}

//--------------------------------------------------
// Vertical erosion kernel using shared memory
// Each block loads a tile of (BLOCK_HEIGHT + 2) x BLOCK_WIDTH
// (extra two rows for top/bottom halo)
//--------------------------------------------------
__global__ void erosionVerticalKernelShared(DeviceImage d_img, DeviceImage d_out)
{
    // Shared memory tile: BLOCK_HEIGHT+2 rows, BLOCK_WIDTH columns.
    __shared__ unsigned char s_tile[BLOCK_HEIGHT + 2][BLOCK_WIDTH];

    int tx = threadIdx.x;  // [0, BLOCK_WIDTH-1]
    int ty = threadIdx.y;  // [0, BLOCK_HEIGHT-1]
    int x = blockIdx.x * BLOCK_WIDTH + tx;
    int y = blockIdx.y * BLOCK_HEIGHT + ty;

    // Load the center pixel into shared memory at s_tile[ty+1][tx].
    if (y < d_img.height && x < d_img.width) {
        const unsigned char* row = d_img.data + y * d_img.pitch;
        s_tile[ty + 1][tx] = row[x];
    } else {
        s_tile[ty + 1][tx] = 0;
    }

    // Load top halo: if thread row is 0, load pixel from y-1.
    if (ty == 0) {
        int y_top = y - 1;
        if (y_top >= 0 && x < d_img.width) {
            const unsigned char* row = d_img.data + y_top * d_img.pitch;
            s_tile[0][tx] = row[x];
        } else {
            s_tile[0][tx] = 0;
        }
    }

    // Load bottom halo: if thread is the last row in the block, load pixel from y+1.
    if (ty == BLOCK_HEIGHT - 1) {
        int y_bot = y + 1;
        if (y_bot < d_img.height && x < d_img.width) {
            const unsigned char* row = d_img.data + y_bot * d_img.pitch;
            s_tile[BLOCK_HEIGHT + 1][tx] = row[x];
        } else {
            s_tile[BLOCK_HEIGHT + 1][tx] = 0;
        }
    }

    __syncthreads();

    // Now perform the vertical erosion: minimum of top, center, and bottom.
    if (y < d_img.height && x < d_img.width) {
        unsigned char top    = s_tile[ty][tx];
        unsigned char center = s_tile[ty + 1][tx];
        unsigned char bottom = s_tile[ty + 2][tx];
        unsigned char minVal = min(min(top, center), bottom);

        unsigned char* outRow = d_out.data + y * d_out.pitch;
        outRow[x] = minVal;
    }
}

int main()
{
    // Load the image in grayscale.
    cv::Mat binaryImg = cv::imread("../imgs/lena_binary_4k.jpg", cv::IMREAD_GRAYSCALE);
    if (binaryImg.empty()) {
        std::cerr << "Error: Could not open image ../imgs/lena_4k.jpg" << std::endl;
        return -1;
    }



    int width  = binaryImg.cols;
    int height = binaryImg.rows;
    size_t rowBytes = width * sizeof(unsigned char);

    // Allocate pitched device memory for input, intermediate, and output images.
    DeviceImage d_input, d_intermediate, d_output;
    cudaMallocPitch(&d_input.data, &d_input.pitch, rowBytes, height);
    cudaMallocPitch(&d_intermediate.data, &d_intermediate.pitch, rowBytes, height);
    cudaMallocPitch(&d_output.data, &d_output.pitch, rowBytes, height);

    d_input.width  = width; d_input.height  = height;
    d_intermediate.width = width; d_intermediate.height = height;
    d_output.width = width; d_output.height = height;

    // Copy host image data to device using cudaMemcpy2D.
    cudaMemcpy2D(d_input.data, d_input.pitch,
                 binaryImg.data, rowBytes,
                 rowBytes, height, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
              (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    // Launch the horizontal erosion kernel with shared memory.
    erosionHorizontalKernelShared<<<grid, block>>>(d_input, d_intermediate);
    cudaDeviceSynchronize();

    // Launch the vertical erosion kernel with shared memory.
    erosionVerticalKernelShared<<<grid, block>>>(d_intermediate, d_output);
    cudaDeviceSynchronize();

    // Copy the final result back to the host.
    cv::Mat result(height, width, CV_8UC1);
    cudaMemcpy2D(result.data, rowBytes,
                 d_output.data, d_output.pitch,
                 rowBytes, height, cudaMemcpyDeviceToHost);

    // Save the result.
    cv::imwrite("../erosion_shared_result.jpg", result);
    std::cout << "Erosion result saved as erosion_shared_result.jpg" << std::endl;

    // Free device memory.
    cudaFree(d_input.data);
    cudaFree(d_intermediate.data);
    cudaFree(d_output.data);

    return 0;
}
