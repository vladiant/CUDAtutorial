cmake_minimum_required(VERSION 3.10)

project(IntroLib LANGUAGES CUDA CXX)

add_library(
    VectorAddLib
    vectorAdd.cu
    vectorAdd.hpp
)

target_include_directories(
    VectorAddLib
    PUBLIC
        .
)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        VectorAddLib
)
