#include <iostream>
#include "clutils.h" // Include your utility functions here

using namespace std;

// Sequential CPU version of the scan algorithm (prefix sum)
void cpu_scan(const int* input, int* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// Function to print an array
void printArray(const int* arr, int size, const string& label) {
    cout << label << ": ";
    for (int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

int main(int argc, char **argv) {
    const int N = 8; // Size of input array
    int input[N] = {1, 2, 3, 4, 5, 6, 7, 8}; // Sample input
    int output[N] = {0}; // Output array for OpenCL results
    int cpu_output[N] = {0}; // Output array for CPU sequential results

    // Step 1: Print the input array
    printArray(input, N, "Input Array");

    // Step 2: Perform the scan on the CPU and print the result
    cpu_scan(input, cpu_output, N);
    printArray(cpu_output, N, "CPU Scan Result");

    // Step 3: Initialize OpenCL
    cluInit();

    // Step 4: Load and compile the OpenCL kernel
    const char *clu_File = SRC_PATH "base.cl"; // Path to the kernel file
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "scan_kernel");

    // Step 5: Create OpenCL buffers for input and output
    cl::Buffer inputBuffer(*clu_Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, input);
    cl::Buffer outputBuffer(*clu_Context, CL_MEM_READ_WRITE, sizeof(int) * N);

    // Step 6: Set kernel arguments for each stage of the scan
    for (int step = 1; step < N; step *= 2) {
        kernel->setArg(0, inputBuffer);
        kernel->setArg(1, outputBuffer);
        kernel->setArg(2, step);
        kernel->setArg(3, N);  // Size of the array

        // Execute the kernel
        cl::NDRange globalSize(N);  // Number of work items 
        clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

        // Copy output back to inputBuffer for the next iteration
        clu_Queue->enqueueCopyBuffer(outputBuffer, inputBuffer, 0, 0, sizeof(int) * N);
    }

    // Step 7: Read the result back from the device to host memory
    clu_Queue->enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(int) * N, output);

    // Step 8: Print the OpenCL parallel scan result
    printArray(output, N, "Parallel Scan Result");

    return 0;
}
