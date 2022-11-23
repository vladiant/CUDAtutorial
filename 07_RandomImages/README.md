# Generating Random Images

## Inline PTX

### PTX is the ‘assembly language’ of CUDA

### You can output PTX code from your kernel
* `nvcc –ptx`
* Project setting

### You can also load a PTX kernel in with Driver API

### Embedding PTX into kernel also possible
* `asm("mov.u32 %0, %%laneid;" : "=r"(laneid));`
* Splices the PTX right into your kernel
* Allows referencing variables

## Generating Random Images
* Determine image dimensions
* Define `x` and `y` in the 0 to 1 range
* Create and call super-complicated function `z = f(x,y)`
* Write data back to image `r = a * z * z + b * z + c`
* Needs a brand new CUDA kernel on each call 
* How to recreate, compile, load and execute a kernel? Answer: Driver API

## Driver API

### CUDA APIs
* Runtime API (what we’ve been using)
* Driver API
   * `cuda.h`, `cuda.lib`

### Driver API
* Allows low-level control of CUDA
* No ‘syntactic sugar’ (`<<<>>>`, dim3, etc.)
* Can be mixed with runtime API
* Not useful to CUDA users in most cases

## Pinned Memory

### `cudaHostAlloc(pHost, size, flags)`

### `flags parameter can be`

* cudaHostAllocMapped
   * Maps memory directly into GPU address space
   * Lets you access host memory directly from GPU
   * A.k.a. “zero-copy memory”
   * Use cudaHostGetDevicePointer() to get device address

* cudaHostAllocPortable
   * Ordinary pinned memory visible to one host thread
   * Portable pinned memory is allowed to migrate between host threads

* cudaHostAllocWriteCombined
   * Write-combined memory transfers faster across the PCI bus
   * Cannot be read efficiently by CPUs

### Can use any combination of the flags above

## Summary
* CUDA C kernels get turned into PTX
   * Can inject PTX inline
* Driver API provides low-level access to CUDA infrastructure
   * Lets you load kernels from PTX or cubin at runtime
* Pinned memory can be mapped, portable and write-combined
