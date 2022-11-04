# Pinned Memory

## CPU memory is pageable
* Can be swapped to disk

## Pinned (page-locked) stays in place

## Performance advantage when copying to/from GPU

## Use `cudaHostAlloc()` instead of `malloc()` or new

## Use `cudaFreeHost()` to deallocate

## Cannot be swapped out
* Must have enough
* Proactively deallocate

## Summary
* Pinned memory speeds up data transfers to/from device
