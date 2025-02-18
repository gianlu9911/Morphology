#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16
#define SE_SIZE 3  // Structuring Element size (change to 3, 5, 7, etc.)
#define SE_RADIUS (SE_SIZE / 2)  // Radius of structuring element


// Struct to encapsulate device image information.
struct DeviceImage {
    unsigned char* data;
    int width;
    int height;
    size_t pitch;  // in bytes
};

// Morphological operators
struct MinOp {
    __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return (a < b) ? a : b;
    }
};

struct MaxOp {
    __device__ unsigned char operator()(unsigned char a, unsigned char b) const {
        return (a > b) ? a : b;
    }
};

template <typename Op>
__global__ void morphOperationKernel(DeviceImage d_img, DeviceImage d_out, bool horizontal, Op op) {
    __shared__ unsigned char tile[BLOCK_HEIGHT + SE_SIZE - 1][BLOCK_WIDTH + SE_SIZE - 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x + SE_RADIUS;
    int ty = threadIdx.y + SE_RADIUS;

    // Load central data into shared memory (with boundary check)
    if (x < d_img.width && y < d_img.height) {
        tile[ty][tx] = d_img.data[y * d_img.pitch + x];
    } else {
        tile[ty][tx] = 255;  // Neutral value for erosion; for dilation you may need 0.
    }

    // Load halo pixels along x-direction
    if (threadIdx.x < SE_RADIUS) {
        int nx = x - SE_RADIUS;
        tile[ty][threadIdx.x] = (nx >= 0) ? d_img.data[y * d_img.pitch + nx] : 255;
    }
    if (threadIdx.x >= BLOCK_WIDTH - SE_RADIUS) {
        int nx = x + SE_RADIUS;
        tile[ty][threadIdx.x + 2 * SE_RADIUS] = (nx < d_img.width) ? d_img.data[y * d_img.pitch + nx] : 255;
    }
    // Load halo pixels along y-direction
    if (threadIdx.y < SE_RADIUS) {
        int ny = y - SE_RADIUS;
        tile[threadIdx.y][tx] = (ny >= 0) ? d_img.data[ny * d_img.pitch + x] : 255;
    }
    if (threadIdx.y >= BLOCK_HEIGHT - SE_RADIUS) {
        int ny = y + SE_RADIUS;
        tile[threadIdx.y + 2 * SE_RADIUS][tx] = (ny < d_img.height) ? d_img.data[ny * d_img.pitch + x] : 255;
    }

    __syncthreads();

    // Apply the morphological operation
    if (x < d_img.width && y < d_img.height) {
        unsigned char result = tile[ty][tx];
        if (horizontal) {
            // Process a horizontal window
            for (int dx = -SE_RADIUS; dx <= SE_RADIUS; dx++) {
                result = op(result, tile[ty][tx + dx]);
            }
        } else {
            // Process a vertical window
            for (int dy = -SE_RADIUS; dy <= SE_RADIUS; dy++) {
                result = op(result, tile[ty + dy][tx]);
            }
        }
        d_out.data[y * d_out.pitch + x] = result;
    }
}

//-----------------------------------------------------------------------
// Simple kernel to compute pixel-wise subtraction (with clamping to 0)
// out = a - b  (if negative, set to 0)
__global__ void subtractKernel(DeviceImage a, DeviceImage b, DeviceImage out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < a.width && y < a.height) {
        int idx = y * a.pitch + x;
        int diff = int(a.data[idx]) - int(b.data[idx]);
        out.data[idx] = (unsigned char)(diff < 0 ? 0 : diff);
    }
}

//-----------------------------------------------------------------------
// Helper function to allocate a pitched device image
void allocateDeviceImage(DeviceImage &img, int width, int height) {
    img.width = width;
    img.height = height;
    size_t rowBytes = width * sizeof(unsigned char);
    cudaMallocPitch(&img.data, &img.pitch, rowBytes, height);
}
