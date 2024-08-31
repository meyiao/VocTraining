cmake_minimum_required(VERSION 2.8)
project(VocTraining)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/DBoW2
)

add_subdirectory(DBoW2)

set(LIBS
        ${OpenCV_LIBS}
        DBoW2
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

file(GLOB_RECURSE SRCs
        src/*.cc
        src/*.h
        src/*.cpp
        src/*.hpp
)

add_subdirectory(SAMPLES)



