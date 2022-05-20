cmake_minimum_required(VERSION 3.10)

project(02_Scan LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    kernel.cu
)

