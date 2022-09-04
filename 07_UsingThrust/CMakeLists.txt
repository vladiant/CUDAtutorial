cmake_minimum_required(VERSION 3.10)

project(07_UsingThrust LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    kernel.cu
)

