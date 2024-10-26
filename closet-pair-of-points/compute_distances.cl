__kernel void compute_distances(__global const int2* coords, __global float* distances, const int num_coords) {
    int global_id = get_global_id(0); // Get the unique global ID for each work item

    // Calculate the row and column index
    int i = global_id / num_coords;      
    int j = global_id % num_coords;       

    // Calculate only for unique pairs only (check if i<j)
    if (i < j) {
        // Calculate the distance
        float dx = (float)(coords[i].x - coords[j].x);
        float dy = (float)(coords[i].y - coords[j].y);
        distances[global_id] = sqrt(dx * dx + dy * dy);  // Store the distance
    } else {
        distances[global_id] = FLT_MAX;  // Set distance to maximum for pairs that are the same 
    }
}
