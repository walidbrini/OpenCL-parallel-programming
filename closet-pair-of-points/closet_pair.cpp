// ----------------------------------------------------------
// Necessary includes for OpenCL and C++ standard libraries
// ----------------------------------------------------------

#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------
// We provide a small library so we can easily set up OpenCL
// ----------------------------------------------------------

#include "clutils.h"

// a 2D point Struct using integers
struct Coordinate {
    int x, y;  // Changed to integers
};

// Fixed number of coordinates
const int TOTAL_COORDS = 9; // Define the array size

// ----------------------------------------------------------
// Sequential CPU calculation of the closest pair of points
// ----------------------------------------------------------

float euclidean_distance(const Coordinate& a, const Coordinate& b) {
    return sqrt(static_cast<float>((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));  
}

float find_closest_pair_cpu(const Coordinate* coords, int num_coords) {
    float closest_distance = MAXFLOAT;

    // Find the closest pair
    for (int i = 0; i < num_coords; ++i) {
        for (int j = i + 1; j < num_coords; ++j) {
            float distance = euclidean_distance(coords[i], coords[j]);
            if (distance < closest_distance) {
                closest_distance = distance;
            }
        }
    }

    return closest_distance;
}

// ----------------------------------------------------------
// Main function
// ----------------------------------------------------------

int main() {
    // Initialize coordinates with integer values
    Coordinate coordinates[TOTAL_COORDS] = {
        {0, 0}, {2, 0}, {0, 2},
        {2, 2}, {4, 0}, {4, 2},
        {0, 4}, {2, 4}, {4, 4}
    };
    // Should return 2 for this exemple 

    // CPU calculation
    float cpu_closest_distance = find_closest_pair_cpu(coordinates, TOTAL_COORDS);
    cout << "CPU Closest Pair Distance" << ": " << cpu_closest_distance << endl;

    // Initialize OpenCL
    cluInit();
    const char* clu_File = SRC_PATH "compute_distances.cl"; // Path to the kernel file
    cl::Program* program = cluLoadProgram(clu_File);
    cl::Kernel* kernel = cluLoadKernel(program, "compute_distances");

    // Prepare OpenCL buffers
    float distance_results[TOTAL_COORDS * TOTAL_COORDS]; 
  
    
    cl::Buffer coordsBuffer(*clu_Context, CL_MEM_READ_ONLY, sizeof(Coordinate) * TOTAL_COORDS);
    cl::Buffer distanceBuffer(*clu_Context, CL_MEM_WRITE_ONLY, sizeof(float) * TOTAL_COORDS * TOTAL_COORDS);

    // Write buffer 
    clu_Queue->enqueueWriteBuffer(coordsBuffer, CL_TRUE, 0, sizeof(Coordinate) * TOTAL_COORDS, coordinates);

    // Set kernel arguments
    kernel->setArg(0, coordsBuffer);
    kernel->setArg(1, distanceBuffer);
    kernel->setArg(2, TOTAL_COORDS);

    // Execute the kernel with a TOTAL_COORDS * TOTAL_COORDS work items

    cl::NDRange globalSize(TOTAL_COORDS * TOTAL_COORDS); // Taille N^2
    clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, globalSize, cl::NullRange);

    // Read back the result from the device
    clu_Queue->enqueueReadBuffer(distanceBuffer, CL_TRUE, 0, sizeof(float) * TOTAL_COORDS * TOTAL_COORDS, distance_results);

    // Find the minimum 
    float gpu_closest_distance = MAXFLOAT; // Initialize to maximum float value

    // find the minimum valid distance
    for (int i = 0; i < TOTAL_COORDS * TOTAL_COORDS; ++i) {
        if (distance_results[i] < gpu_closest_distance) {
            gpu_closest_distance = distance_results[i];
        }
    }    
    cout << "GPU Closest Pair Distance" << ": " << gpu_closest_distance << endl;

    // Compare CPU and GPU results with a tolerance 
    // Floating-point for CPU and GPU they handle precision and rounding differntly
    if (abs(cpu_closest_distance - gpu_closest_distance) < 1e-6) {
        cout << "Results are consistent!" << endl;
    } else {
        cout << "Results are not consistent!" << endl;
    }

    return 0;
}
