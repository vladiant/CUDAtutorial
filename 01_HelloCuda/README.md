# Introduction to CUDA

## NVidia Cuda Compiler (nvcc)
* nvcc is used to compile CUDA-specific code
  * Not a compiler!
  * Uses host C or C++ compiler (MSVC, GCC)
  * Some aspects are C++ specific

* Splits code into GPU and non-GPU parts
  * Host code passed to native compiler

* Accepts project-defined GPU-specific settings
  * E.g., compute capability

* Translates code written in CUDA C into PTX
  * Graphics driver turns PTX into binary code

## Parallel Thread Execution (PTX)
* PTX is the ‘assembly language’ of CUDA
  * Similar to .NET IL or Java bytecode
  * Low-level GPU instructions

* Can be generated from a project

* Typically useful to compiler writers
  * E.g., GPU Ocelot <https://code.google.com/p/gpuocelot/>

* Inline PTX (asm)

## Location Qualifiers
* \_\_global__
  * Defines a kernel.
  * Runs on the GPU, called from the CPU.
  * Executed with <\<\<\dim3>>> arguments.

* \_\_device__
  * Runs on the GPU, called from the GPU.
  * Can be used for variables too

* \_\_host__
  * Runs on the CPU, called from the CPU.

* Qualifiers can be mixed
  * E.g. __host__ __device__ foo()
  * Code compiled for both CPU and GPU
  * Useful for testing

## Execution Model
* Thread blocks are scheduled to run on available SMs
* Each SM executes one block at a time
* Thread block is divided into warps
* Number of threads per warp depends on compute capability
* All warps are handled in parallel
* CUDA Warp Watch

## Error Handling

* CUDA does not throw
  * Silent failure

* Core functions return cudaError_t
  * Can check against cudaSuccess
  * Get description with cudaGetErrorString()

* Libraries may have different error types
  * E.g. cuRAND has curandStatus_t
