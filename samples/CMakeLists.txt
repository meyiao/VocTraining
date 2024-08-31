add_executable(VocTrainingDemo
        VocTrainingDemo.cpp
        ${SRCs}
)
target_link_libraries(VocTrainingDemo ${LIBS})


add_executable(LoopDetectDemo
        LoopDetectDemo.cpp
        ${SRCs}
)
target_link_libraries(LoopDetectDemo ${LIBS})