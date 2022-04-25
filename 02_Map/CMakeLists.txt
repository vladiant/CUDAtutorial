cmake_minimum_required(VERSION 3.10)

project(02_Map LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    kernel.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        curand
)

