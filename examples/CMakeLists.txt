set(TARGET_NAME Examples)
project(${TARGET_NAME})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/$<CONFIG>)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB EXAM_FILES ${SOURCE_DIR}/*.cpp)
add_executable(${TARGET_NAME} ${EXAM_FILES})

file(GLOB VTK_LIB_FILES ${VTK_DIR}/lib/*.lib)
list(APPEND LIBS ${VTK_LIB_FILES})

target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_SOURCE_DIR}/src/include
    ${VTK_DIR}/include
)
target_link_libraries(${TARGET_NAME} ND_LNLib)
add_dependencies(${TARGET_NAME} ND_LNLib)