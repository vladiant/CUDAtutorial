# Thrust Library

## STL-like library for accelerated computation

## Included with CUDA

## `host_vector` and `device_vector`
* Assign, resize, etc. (but each `d[n] = z;` causes a `cudaMemcpy`)
* Copy with = operator
* Can interop with STL containers and CUDA raw memory

## Predefined algorithms
* Search, sort, copy, reduce

## Functor syntax
* `thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), thrust::multiplies<float>());`

## Summary
* Thrust library makes using CUDA a lot easier
