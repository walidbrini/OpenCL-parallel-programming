__kernel void parallel_scan(__global int* current,__global int* next, int shift ){

    int i = get_global_id(0);


    // Shift determines the step we have to advance for every kernel

    if (i >= shift ){ 
        next[i] = current[i-shift] + current[i];  
    }
    else { 
        next[i] = current[i]; 
    }

}