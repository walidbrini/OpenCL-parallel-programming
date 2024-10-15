__kernel void conway_game_of_life(__global int* current_buffer, __global int* next_buffer, int height, int width, int is_periodic) {
    int globalID_x = get_global_id(0);
    int globalID_y = get_global_id(1);

    // Ensure the global IDs are within the grid bounds
    if (globalID_x >= height || globalID_y >= width) return;

    int index = globalID_x * width + globalID_y; 
    int live_neighbors = 0;

    // Check all 8 possible neighbors
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // cell itself
            if (i == 0 && j == 0) continue;

            int neighbor_row = globalID_x + i;
            int neighbor_col = globalID_y + j;

            // Wrap around if periodic
            if (is_periodic) {
                neighbor_row = (neighbor_row + height) % height;
                neighbor_col = (neighbor_col + width) % width;
            } else {
                // Check  bounds
                if (neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 || neighbor_col >= width) {
                    continue; // Skip 
                }
            }

            live_neighbors += current_buffer[neighbor_row * width + neighbor_col];
        }
    }

    // Apply Conway's rules
    if (current_buffer[index] == 1) { // Cell is alive
        next_buffer[index] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
    } else { // Cell is dead
        next_buffer[index] = (live_neighbors == 3) ? 1 : 0;
    }
}
