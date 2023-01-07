# Events

##  How to measure performance?

### Use OS timers
* Too much noise

### Use profiler
* Times only kernel duration + other invocations

### CUDA Events
* Event = timestamp
* Timestamp recorded on the GPU
* Invoked from the CPU side

## Event API

### [cudaEvent_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gea2f543a9fc0e52fe4ae712920fd1247)
* The event handle

### [cudaEventCreate(&e)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g7c581e3613a2110ba4d4e7fd5c7da418)
* Creates the event

### [cudaEventRecord(e, 0)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446)
* Records the event, i.e. timestamp
* Second param is the stream to which to record

### [cudaEventSynchronize(e)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g949aa42b30ae9e622f6ba0787129ff22)
* CPU and GPU are async, can be doing things in parallel
* `cudaEventSynchronize()` blocks all instruction processing until the GPU has reached the event

### [cudaEventElapsedTime(&f, start, stop)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6)
* Computes elapsed time (msec) between start and stop, stored as float

## Summary
* CUDA events let you time your code on the GPU
