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

### `cudaEvent_t`
* The event handle

### `cudaEventCreate(&e)`
* Creates the event

### `cudaEventRecord(e, 0)`
* Records the event, i.e. timestamp
* Second param is the stream to which to record

### `cudaEventSynchronize(e)`
* CPU and GPU are async, can be doing things in parallel
* `cudaEventSynchronize()` blocks all instruction processing until the GPU has reached the event

### `cudaEventElapsedTime(&f, start, stop)`
* Computes elapsed time (msec) between start and stop, stored as float

## Summary
* CUDA events let you time your code on the GPU
