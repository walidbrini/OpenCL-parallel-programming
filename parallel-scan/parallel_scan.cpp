// ----------------------------------------------------------
// Necessary includes for OpenCL and C++ standard libraries
// ----------------------------------------------------------

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace std;

// ----------------------------------------------------------
// We provide a small library so we can easily set up OpenCL
#include "clutils.h"

// ----------------------------------------------------------
// Function to perform prefix sum on the CPU 
// ----------------------------------------------------------

void cpu_sum(int arr[], int size) {
    // Calculate the prefix sum in-place
    for (int i = 1; i < size; i++) {
        arr[i] = arr[i] + arr[i - 1];
    }

    // Display the result
    for (int i = 0; i < size; i++) {
        cout << arr[i] << ", ";
    }
    cout << endl;
}

// ----------------------------------------------------------
// Main function
// ----------------------------------------------------------

int main(int argc, char **argv) {
    // Initialize array with sample values
    int tab[] = {1, 2, 3, 4, 5, 6, 7, 8};

	// N stores the size of tab 
	int N = sizeof(tab) / sizeof(tab[0]);

    const int group_size = N;


    const char *clu_File = SRC_PATH "p_scan.cl";  // Path to OpenCL kernel file

    // Initialize OpenCL
    cluInit();

    // Load Program
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "parallel_scan");

    // Allocate memory on the compute device
    cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_WRITE, N * sizeof(int));
    cl::Buffer b_buffer(*clu_Context, CL_MEM_READ_WRITE, N * sizeof(int));

    // Transfer data from host to device (to the compute buffer)
    clu_Queue->enqueueWriteBuffer(a_buffer, true, 0, N * sizeof(int), tab);

    // Execute parallel scan in log steps
    for (int i = 1; i < N; i *= 2) {
        kernel->setArg(0, a_buffer);
        kernel->setArg(1, b_buffer);
        kernel->setArg(2, i);

        // Launch kernel
        clu_Queue->enqueueNDRangeKernel(
            *kernel,
            cl::NullRange,
            cl::NDRange(group_size));  // Global work size

        // Wait for this iteration to finish
        clu_Queue->finish();

        // Swap the buffers for the next iteration
        swap(b_buffer, a_buffer);
    }

    // Read results back from the device
    int *result = new int[N];
    clu_Queue->enqueueReadBuffer(a_buffer, true, 0, N * sizeof(int), result);


	// Perform CPU-based prefix sum
    cout << "CPU Prefix Sum Result: ";
    cpu_sum(tab, group_size);

    // Display results
    cout << "Parallel Prefix Sum Result: ";
    for (int i = 0; i < N; i++) {
        cout << result[i] << "; ";
    }
    cout << endl;
    
	
	// Compare CPU and GPU results

	bool flag = true; 
	
	for (int i = 0 ; i < N ; i++){ 
		if (tab[i]!=result[i]){ 
			flag = false;  
		}
	}	

	if (flag){
		cout << "Both CPU and GPU Calculation output the same Result"; 
	}
	else { 
		cout << "CPU and GPU results are different";
	}
	cout << endl ; 

    // Clean 
    delete[] result;

    return 0;
}
