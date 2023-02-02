# Pinned Memory

## CPU memory is pageable
* Can be swapped to disk

## Pinned (page-locked) stays in place

## Performance advantage when copying to/from GPU

## Use [cudaHostAlloc()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902) instead of `malloc()` or new

## Use [cudaFreeHost()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969) to deallocate

## Cannot be swapped out
* Must have enough
* Proactively deallocate

## Summary
* Pinned memory speeds up data transfers to/from device
