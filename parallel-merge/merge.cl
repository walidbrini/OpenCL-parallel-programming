// Function to compute low(x, array) in O(log N) using binary search
int binary_search_low(const __global int *array, int N, int x) {
    int low = 0, high = N - 1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (array[mid] < x) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return low;  // Number of elements less than x
}

__kernel void parallel_merge(
    __global const int *A,       // Input array A
    __global const int *B,       // Input array B
    __global int *C,             // Output array C
    const int N) {               // Size of A and B
    int idx = get_global_id(0);

    // Determine whether we process A or B
    if (idx < N) {
        // Handle elements from A
        int low_B = binary_search_low(B, N, A[idx]);  // Elements in B less than A[idx]
        C[idx + low_B] = A[idx];
    } else {
        // Handle elements from B
        int b_idx = idx - N;  // Adjust index for array B
        int low_A = binary_search_low(A, N, B[b_idx]);  // Elements in A less than B[b_idx]
        C[b_idx + low_A] = B[b_idx];
    }
}
