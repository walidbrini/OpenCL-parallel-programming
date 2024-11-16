#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace std;

#include "clutils.h"

int main(int argc, char **argv) {
    const char *clu_File = SRC_PATH "merge.cl";  // Path to OpenCL kernel code
    vector<int> A = {1, 4, 7, 13, 20};
    vector<int> B = {2, 15, 21, 40, 63};
    int N = A.size();

    // Combined size of the merged array
    int C_size = 2 * N;
    vector<int> C(C_size);

    // Initialize OpenCL
    cluInit();

    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "parallel_merge");

    // Allocate memory on the compute device
    cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_ONLY, N * sizeof(int));
    cl::Buffer b_buffer(*clu_Context, CL_MEM_READ_ONLY, N * sizeof(int));
    cl::Buffer c_buffer(*clu_Context, CL_MEM_WRITE_ONLY, C_size * sizeof(int));

    // Write input data to device buffers
    clu_Queue->enqueueWriteBuffer(a_buffer, true, 0, N * sizeof(int), A.data());
    clu_Queue->enqueueWriteBuffer(b_buffer, true, 0, N * sizeof(int), B.data());

    // Set kernel arguments
    kernel->setArg(0, a_buffer);
    kernel->setArg(1, b_buffer);
    kernel->setArg(2, c_buffer);
    kernel->setArg(3, N);

    // Execute kernel
    clu_Queue->enqueueNDRangeKernel(
        *kernel,
        cl::NullRange,
        cl::NDRange(C_size),  // One work item per element in C
        cl::NullRange
    );

    // Wait for the kernel to finish
    clu_Queue->finish();

    // Read merged data
    clu_Queue->enqueueReadBuffer(c_buffer, true, 0, C_size * sizeof(int), C.data());

    // Print the merged array
    cout << "Merged array: ";
    for (int i = 0; i < C_size; i++) {
        cout << C[i] << " ";
    }
    cout << endl;

    return 0;
}
