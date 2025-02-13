#include <iostream>
#include <cuda_runtime.h>
#include <fstream>

void saveExecutionTimeCSV(const std::string& filename, const std::string& resolution, 
                          double execution_time, const std::string& operation, 
                          const std::string& version, int kernel_size) {
    std::ofstream file;
    bool file_exists = std::ifstream(filename).good();
    
    file.open(filename, std::ios::app);
    
    // If file does not exist, write header
    if (!file_exists) {
        file << "Resolution,Execution Time (s),Operation,Version,Kernel Size\n";
    }
    
    // Write data row
    file << resolution << "," << execution_time << "," << operation << "," << version << kernel_size << "\n";
    file.close();
}

__global__ void my_kernel(){
    printf("hi from kernel\n");
}
