__kernel void sort(__global int* a, int parity, int N) { 
    int i = get_global_id(0) * 2;

    if (parity % 2 != 0) { 
        i += 1;
    }

    if (i + 1 < N) {
        if (a[i] > a[i + 1]) { 
            // Swap a[i] and a[i + 1]
            int temp = a[i];
            a[i] = a[i + 1];
            a[i + 1] = temp;
        }
    }
}
