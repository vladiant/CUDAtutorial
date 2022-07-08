cmake_minimum_required(VERSION 3.10)

project(05_MonteCarloPi LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    kernel.cu
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        curand
)

