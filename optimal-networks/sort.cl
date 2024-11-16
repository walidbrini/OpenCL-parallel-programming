// Define Pair structure for OpenCL
typedef struct {
    int first;
    int second;
} Pair;

__kernel void sort(__global int* array, 
                  __global const Pair* pairs,
                  const int num_pairs) {
    // Get the global ID (which pair this work-item is processing)
    int pair_id = get_global_id(0);
    
    // Check if this work-item should process a pair
    if (pair_id < num_pairs) {
        // Get the indices for this pair
        int i = pairs[pair_id].first;
        int j = pairs[pair_id].second;
        
        // Compare and swap if necessary
        int a = array[i];
        int b = array[j];
        
        if (a > b) {
            array[i] = b;
            array[j] = a;
        }
    }
}