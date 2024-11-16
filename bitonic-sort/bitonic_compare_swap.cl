
__kernel void bitonic_compare_and_swap(
    __global int* T,
    int step,
    int seq_size,
    int direction) 
{
    int i = get_global_id(0);
    int pair_index = i ^ step;

    if (pair_index > i) {
        // Ascending or descending order based on `direction`
        bool compare = (T[i] > T[pair_index]) == direction;
        if (compare) {
            // Swap elements if they are out of order
            int temp = T[i];
            T[i] = T[pair_index];
            T[pair_index] = temp;
        }
    }
}
