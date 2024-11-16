#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <climits>
#include <algorithm>
using namespace std;


// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// Define Pair type for sorting network
typedef pair<int, int> Pair;

// Function to parse a single pair like (0,1)
Pair parsePair(const string& pairStr) {
    int first, second;
    sscanf(pairStr.c_str(), "(%d,%d)", &first, &second);
    return {first, second};
}

// Function to parse a group of pairs like [(0,1),(2,3)]
vector<Pair> parseGroup(const string& groupStr) {
    vector<Pair> pairs;
    string pairStr;
    size_t start = 1; // Skip the first '['
    
    while (start < groupStr.length() - 1) { // -1 to skip the last ']'
        size_t end = groupStr.find(')', start);
        if (end == string::npos) break;
        
        pairStr = groupStr.substr(start, end - start + 1);
        pairs.push_back(parsePair(pairStr));
        start = end + 2; // Skip the ')' and ',' to get to the next pair
    }
    
    return pairs;
}

// Function to parse a line containing multiple groups
vector<vector<Pair>> parseLine(const string& line) {
    vector<vector<Pair>> groups;
    string groupStr;
    istringstream iss(line);
    
    while (getline(iss, groupStr, ' ')) {
        if (!groupStr.empty()) {
            groups.push_back(parseGroup(groupStr));
        }
    }
    
    return groups;
}

// Main function to read and parse the file
vector<vector<vector<Pair>>> parseFile(const string& filename) {
    vector<vector<vector<Pair>>> result;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        if (!line.empty()) {
            result.push_back(parseLine(line));
        }
    }
    
    return result;
}

// Printing Function
void printParsedData(const vector<vector<vector<Pair>>>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        cout << "Line " << i + 1 << ":\n";
        for (const auto& group : data[i]) {
            cout << "  Group: ";
            for (const auto& pair : group) {
                cout << "(" << pair.first << "," << pair.second << ") ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
}


// Function to compare two arrays (CPU and GPU results)
bool compareArrays(const int* arr1, const int* arr2, int N) {
    for (int i = 0; i < N; i++) {
        if (arr1[i] != arr2[i]) {
            return false; // Arrays are not the same
        }
    }
    return true; // Arrays are identical
}

int main(int argc, char **argv) {
    // Path to sorting network file
    const string patternFile = SRC_PATH "steps.txt";

    // Fixed input array
    int tab[] = {1, 4, 99, 8, 101, 2, 3};
    int N = sizeof(tab) / sizeof(tab[0]);

    // Create a copy of the input array for CPU sorting
    vector<int> tab_cpu(tab, tab + N);  // Using vector for better memory management

    // Parse sorting network from file
    vector<vector<vector<Pair>>> sortingNetworks = parseFile(patternFile);

    // Validate network exists for this size
    if (sortingNetworks.size() < N - 1) {
        cerr << "No sorting network defined for size " << N << endl;
        return -1;
    }

    // Get the sorting network for this size
    const vector<vector<Pair>>& network = sortingNetworks[N - 2];

    // Initialize OpenCL
    const char *clu_File = SRC_PATH "sort.cl";
    cluInit();
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel *kernel = cluLoadKernel(program, "sort");

    // Create input buffer with initial data
    cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       N * sizeof(int), tab);

    // Process each layer of the sorting network
    for (const auto& layer : network) {
        // Create buffer for all pairs in this layer
        int num_pairs = layer.size();
        cl::Buffer pairs_buffer(*clu_Context, CL_MEM_READ_ONLY, num_pairs * sizeof(Pair));
        
        // Copy layer pairs to device
        clu_Queue->enqueueWriteBuffer(pairs_buffer, CL_TRUE, 0, 
                                    num_pairs * sizeof(Pair), layer.data());
        
        // Set kernel arguments
        kernel->setArg(0, a_buffer);
        kernel->setArg(1, pairs_buffer);
        kernel->setArg(2, num_pairs);

        // Execute kernel for all pairs in this layer
        clu_Queue->enqueueNDRangeKernel(
            *kernel,
            cl::NullRange,
            cl::NDRange(num_pairs)
        );
        
        clu_Queue->finish();
    }

    int tab_output[N];

    // Read back the result
    clu_Queue->enqueueReadBuffer(a_buffer, CL_TRUE, 0, N * sizeof(int), tab_output);

    // Print GPU results
    cout << "\nSorted Array using GPU: ";
    for (int i = 0; i < N; i++) cout << tab_output[i] << " ";
    cout << endl;

    // Perform CPU sorting
    sort(tab_cpu.begin(), tab_cpu.end()) ; 

    // Print CPU Result
    cout << "\nSorted Array using CPU: ";
    for (int i = 0; i < N; i++) cout << tab_cpu[i] << " ";
    cout << endl;

    // Compare CPU and GPU results
    if (compareArrays(tab_output, tab_cpu.data(), N)) {
        cout << "The results are identical!" << endl;
    } else {
        cout << "The results are different!" << endl;
    }

    return 0;
}