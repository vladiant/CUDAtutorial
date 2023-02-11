# Multi-GPU Programming

## Execute parts on separate devices
* Split the work
* Execute kernels on separate threads
* Combine the results

## Use [cudaSetDevice(id)](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb) to select the device to run on

## Portable zero-copy memory useful for multi-threading

## Summary
* Running on multiple devices is not difficult
