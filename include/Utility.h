#include <iostream>
#include <cuda_runtime.h>

__global__ void my_kernel(){
    printf("hi from kernel\n");
}
