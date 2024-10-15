// ----------------------------------------------------------

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

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"


int main(int argc, char **argv)
{
    const char *clu_File = SRC_PATH "base.cl";  // path to file containing OpenCL kernel(s) code

    // Initialize OpenCL
    cluInit();

    // Load Program
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "conway_game_of_life"); 

    // Allocate memory on the compute device 
    const int height = 4; 
    const int width = 5; 

    // Local group size
    const int group_size = 2; // Change to a suitable size

    // Initialize grid
    int current_grid[height * width] = { 
        0, 0, 0, 0, 0, // Example initial configuration
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0
    };
    
    cout << "Iteration 0" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << (current_grid[i * width + j] ? 'O' : 'X') << " "; 
        }
        cout << endl; 
    }
    cout << endl;

    // Create OpenCL buffers
    cl::Buffer current_buffer(*clu_Context, CL_MEM_READ_WRITE, height * width * sizeof(int)); 
    cl::Buffer next_buffer(*clu_Context, CL_MEM_READ_WRITE, height * width * sizeof(int)); 

    // Write initial grid to device
    clu_Queue->enqueueWriteBuffer(current_buffer, CL_TRUE, 0, height * width * sizeof(int), current_grid);

    // Number of iterations
    const int iterations = 3;

    for (int iter = 0; iter < iterations; ++iter) {
        // Set the kernel arguments
        kernel->setArg(0, current_buffer);  // Current buffer
        kernel->setArg(1, next_buffer);     // Next buffer
        kernel->setArg(2, height);          // Grid height
        kernel->setArg(3, width);           // Grid width

        // Launch the kernel
        clu_Queue->enqueueNDRangeKernel(
            *kernel,
            cl::NullRange,                   // Offset
            cl::NDRange(height, width)     // Global work size 
        );

        // Finish kernel execution
        clu_Queue->finish();

        // Read the next state into current_grid
        clu_Queue->enqueueReadBuffer(next_buffer, CL_TRUE, 0, height * width * sizeof(int), current_grid);

        // Swap buffers
        std::swap(current_buffer, next_buffer);

        // Visualize the grid 
        cout << "Iteration " << iter + 1 << ":" << endl;
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				cout << (current_grid[i * width + j] ? 'O' : 'X') << " "; 
			}
			cout << endl; 
		}
		cout << endl;
    }

    return 0;
}
