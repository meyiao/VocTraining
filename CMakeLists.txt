cmake_minimum_required(VERSION 2.8)
project(VocTraining)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
)

set(LIBS
        ${OpenCV_LIBS}
        ${PROJECT_SOURCE_DIR}/DBoW2/lib/libDBoW2.so
    )

add_executable(VocTraining
        src/main.cpp
        src/ORBVocabulary.h
        src/utils.h src/utils.cpp
        src/ORBextractor.h src/ORBextractor.cc
        )

target_link_libraries(VocTraining ${LIBS})




