cmake_minimum_required(VERSION 3.10)

project(05_AtomicSum LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    kernel.cu
)

