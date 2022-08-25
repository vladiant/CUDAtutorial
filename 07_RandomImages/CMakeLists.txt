cmake_minimum_required(VERSION 3.10)

project(07_RandomImages LANGUAGES CUDA CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

add_library(
    GenerateBitmap
    generate_bitmap.cu
    generate_bitmap.h
)

target_link_libraries(
    GenerateBitmap
    PRIVATE
        cuda
        curand
        nvrtc
)

add_executable(
    ${PROJECT_NAME}
    main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE
        ${GTK3_DEFINITIONS}
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        ${GTK3_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        GenerateBitmap
        ${GTK3_LIBRARIES}
)

