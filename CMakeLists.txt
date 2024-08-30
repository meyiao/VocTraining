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

set(SRCs src/ORBVocabulary.h
        src/utils.h src/utils.cpp
        src/ORBextractor.h src/ORBextractor.cc)

add_executable(VocTraining
        src/main.cpp
        ${SRCs}
        )
target_link_libraries(VocTraining ${LIBS})

add_executable(LoopDetectDemo
        src/LoopDetectDemo.cpp
        ${SRCs}
)
target_link_libraries(LoopDetectDemo ${LIBS})



