#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------

#include "clutils.h"

// ----------------------------------------------------------

// Helper function to print the array
void print_tab(int T[], int N){
    for (int i = 0; i < N; i++) { 
        cout << T[i] << " , " ;     
    } 
    cout << endl; 
}

int main(int argc, char **argv) {
	const char *clu_File = SRC_PATH "bitonic_compare_swap.cl";  // path to file containing OpenCL kernel(s) code
    int T[] = {1, 2, 4, 5, 3, 10, 32, 43};

    // Ensure size of T is a power of 2
    int N = sizeof(T) / sizeof(T[0]);
    assert((N > 0) && ((N & (N - 1)) == 0));

    // Print initial array
    cout << "Original Array: ";
    print_tab(T, N);

    // Initialize OpenCL
    cluInit();

    // Load Program
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "bitonic_compare_and_swap"); 

    // Allocate memory on the compute device 
    cl::Buffer T_buffer(*clu_Context, CL_MEM_READ_WRITE, N * sizeof(int)); 

    // Write data to buffer
    clu_Queue->enqueueWriteBuffer(T_buffer, true, 0, N * sizeof(int), T);

    // Bitonic sort in parallel using OpenCL
    for (int seq_size = 2; seq_size <= N; seq_size *= 2) {
        for (int step = seq_size / 2; step > 0; step /= 2) {
            int direction = (seq_size == N) ? 1 : 0;  // 1 for ascending, 0 for descending

            // Set kernel arguments
            kernel->setArg(0, T_buffer); 
            kernel->setArg(1, step); 
            kernel->setArg(2, seq_size); 
            kernel->setArg(3, direction);

            // Launch kernel
            clu_Queue->enqueueNDRangeKernel(
                *kernel,
                cl::NullRange,
                cl::NDRange(N / 2), // Global work size
                cl::NullRange);     // Let OpenCL determine optimal local size

            // Synchronize to ensure completion of the kernel execution
            clu_Queue->finish();
        }
    }

    // Read sorted data back to host
    clu_Queue->enqueueReadBuffer(T_buffer, true, 0, N * sizeof(int), T);

    // Print sorted array
    cout << "Sorted Array: ";
    print_tab(T, N);

    return 0;
}
