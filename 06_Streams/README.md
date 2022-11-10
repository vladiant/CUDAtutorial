# Streams

## Remember cudaEventRecord(event, stream)?

## A CUDA stream is a queue of GPU operations
* Kernel launch
* Memory copy

## Streams allow a form of task-based parallelism

## Performance improvement

## To leverage streams you need device overlap support
* `GPU_OVERLAP`

## Stream API
* `cudaStream_t`
* `cudaStreamCreate(&stream)`
* `kernel<<<blocks,threads,shared,stream>>>`
* `cudaMemcpyAsync()`
   * Must use pinned memory!
* stream parameter
* `cudaStreamSynchronize(stream)`

## Summary
* CUDA streams allow you to queue up operations asynchronously
   * Lets you do different things in parallel on the GPU
   * Use of pinned memory is required
