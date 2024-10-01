
# Data Parallelism

- **SIMD**: Single Instruction, Multiple Data

# Programmable Graphics Pipeline

Modern GPUs can have up to 20,000 cores, enabling massive parallel processing capabilities.

- **CUDA / OpenCL**: These frameworks focus on general-purpose computation on GPUs and do not directly support graphics operations.
- **DirectX / OpenGL**: These are designed for graphics operations, such as rendering meshes and rectangles.

# Working with OpenCL

OpenCL works with a **host** (usually a CPU) and a **compute device** (usually a GPU). OpenCL uses a subset of C++ for its kernels.

## Code for Compute Device (GPU)

In OpenCL, a kernel represents a function that runs on a specific compute unit.

Example kernel:

```c
__kernel void mainKernel() {
    // Kernel code here
}
```

# Execution Model

To understand the execution model in OpenCL, let’s consider an example:

- `N = 8`: Total number of work-items
- `G = 4`: Number of compute units (kernel executions)

Each kernel runs on a separate work-item:
```
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
```

- `get_global_size(0)` returns `N`, the total number of work-items.
- `get_global_id(0)` returns the global ID for each work-item (for the first dimension, which is typically the x-dimension).

### Explanation of Dimensions

The argument `0` refers to the first dimension (x-dimension). OpenCL organizes work-items in up to three dimensions:
- **0**: x-axis
- **1**: y-axis
- **2**: z-axis

### Kernel Functions

A kernel has access to several functions:
- `get_global_size(0)`: Returns the total number of work-items in the grid (for 1D, the size is `N`).
- `get_local_id(0)`: Returns the local ID of the work-item within its group.

# Command Queue

OpenCL command queues execute tasks in order, but it's essential to check if tasks have completed using synchronization functions.

