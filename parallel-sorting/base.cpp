#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>

using namespace std;

#include "clutils.h"

int main(int argc, char **argv) {
    const char *clu_File = SRC_PATH "base.cl";  // Path to file containing OpenCL kernel(s) code
    int tab[] = {2, 4, 9, 5, 10, 6, 7, 8};
    int N = sizeof(tab) / sizeof(tab[0]);

    // Initialize OpenCL
    cluInit();

    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "sort");

    // Allocate memory on the compute device
    cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_WRITE, N * sizeof(int));

    clu_Queue->enqueueWriteBuffer(a_buffer, true, 0, N * sizeof(int), tab);

    for (int i = 0; i < N; i++) {
        kernel->setArg(0, a_buffer); 
        kernel->setArg(1, i);        // Parity (odd or even phase)
        kernel->setArg(2, N);        // Size of the array

        
        clu_Queue->enqueueNDRangeKernel(
            *kernel,
            cl::NullRange,
            cl::NDRange(N / 2) // half the number of work items
        ); 

        clu_Queue->finish();
    }

    // Read sorted data
    int* sorted_array = new int[N];
    clu_Queue->enqueueReadBuffer(a_buffer, true, 0, N * sizeof(int), sorted_array);

    // Print the sorted array
    cout << "Sorted array: ";
    for (int i = 0; i < N; i++) {
        cout << sorted_array[i] << " ";
    }
    cout << endl;

    delete[] sorted_array;
    return 0;
}
