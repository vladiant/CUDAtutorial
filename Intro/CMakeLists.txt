cmake_minimum_required(VERSION 3.10)

project(Intro LANGUAGES CUDA)

add_executable(
    ${PROJECT_NAME}
    vectorAdd.cu
)

