// ----------------------------------------------------------

// ----------------------------------------------------------


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <chrono>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------


void compare(int T[], int L, int R, int d){ 
	int k = (R - L + 1) / 2; 
	for (int i = 0; i < k; i++) {
		if (d == 1) {
			if (T[L + i] < T[L + i + k]) 
				swap(T[L + i], T[L + i + k]); 
		} else {
			if (T[L + i] > T[L + i + k]) 
				swap(T[L + i], T[L + i + k]); 
		}
	}
}

void merge(int T[], int L, int R, int d){ 
	if (R - L + 1 > 1){ 
		compare(T, L, R, d);
		merge(T, L, (L + R) / 2, d); 
		merge(T, (L + R) / 2 + 1, R, d);
	}
}

void sort(int T[], int L, int R, int d){ 
	if (R - L + 1 > 1){ 
		sort(T, L, (R + L) / 2, 1); // Ascending
		sort(T, (R + L) / 2 + 1, R, 0); // Descending
		merge(T, L, R, d);
	}
}


void print_tab(int T[], int N){
	for (int i = 0; i<N; i++){ 
		cout << T[i] << " , " ; 	
	} 
	cout << endl ; 

}


int main(int argc, char **argv)
{
	const char *clu_File = SRC_PATH "base.cl";  // path to file containing OpenCL kernel(s) code
    int T[] = {1, 2, 4, 5, 3, 10, 32, 43};

	// Make sure that size of T is a power of 2 
	int N = sizeof(T)/sizeof(T[0]);
    assert((N > 0) && ((N & (N - 1)) == 0));

	// CPU method to run bitonic sort 
	sort(T,0,N-1,1); // ASC
	print_tab(T,N); 


	// Initialize OpenCL
	cluInit();

	// After this call you have access to
	// clu_Context;      <= OpenCL context (pointer)
	// clu_Devices;      <= OpenCL device list (vector)
	// clu_Queue;        <= OpenCL queue (pointer)

	// Load Program
	cl::Program *program = cluLoadProgram(clu_File);
	cl::Kernel *kernel = cluLoadKernel(program, "summation"); 

	// allocate memory on the compute device 

	const int size = 32; 
	// group size < Max work group size 
	const int group_size = 8;
	assert(size % group_size == 0 ); 

	cl::Buffer a_buffer(*clu_Context, CL_MEM_READ_ONLY, size * sizeof(int)); 
	cl::Buffer b_buffer(*clu_Context, CL_MEM_READ_ONLY, size * sizeof(int)); 
	cl::Buffer c_buffer(*clu_Context, CL_MEM_WRITE_ONLY, size * sizeof(int)); 

	// Write on the buffers 
	int* a = new int[size]; 
	int* b = new int [size]; 
	for (int i = 0 ; i < size; i++){ 
		a[i] = 0 ; 
		b[i] = i ; 
	}

	// Transofrom the , a from the host a_buufer is in the compute device 

	clu_Queue->enqueueWriteBuffer(a_buffer, true, 0, size * sizeof(int), a); 
	clu_Queue->enqueueWriteBuffer(b_buffer, true, 0, size * sizeof(int), b); 
	delete[] a ; 
	delete[] b ;

	kernel->setArg(0, a_buffer); 
	kernel->setArg(1, b_buffer); 
	kernel->setArg(2, c_buffer);
	kernel->setArg(3, cl::Local(group_size * sizeof(int)));

	// launch kernel 
	clu_Queue->enqueueNDRangeKernel(
		*kernel,
		cl::NullRange,
		cl::NDRange(size), // global work size 
		cl::NDRange(group_size)) ; // local work size

	// i'm surei'm done 
	clu_Queue->finish();

	int* c = new int[size]; 
	clu_Queue ->enqueueReadBuffer(c_buffer, true, 0, size * sizeof(int), c);
	for (int i = 0 ; i<size ; i++){ 
		std::cout << c[i] << "; "; 
	}

}
