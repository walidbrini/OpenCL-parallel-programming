__kernel void summation(__global int* a, __global int* b, __global int* c, __local int* shared ){ 
    int i = get_global_id(0); 

    //c [i] = a[i] + b[i]; 
    //prefetch 
    shared[get_local_id(0)] = b[i]; 
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0)==0){
        int local_sum = 0 ; 
        // I am the first, i know that shared contains the corresponding piece of global memory
        for (int j=0; j <  get_local_size(0); j++){ 
            local_sum += shared[j]; 
        }
        c[i] = local_sum ;
        // atomic_add(&c[1], c[i]); 
        // c[1] += c[i]  // error 
    }

    // the interest is i can do operation on shared very fast ! 
    // but careful because i do not know it shared is coherent

    //c[i] = a [i] + b[i]; 
}