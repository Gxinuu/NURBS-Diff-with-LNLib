set(TARGET_NAME Examples)
project(${TARGET_NAME})
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(${TARGET_NAME})
target_compile_definitions(${TARGET_NAME}  PRIVATE ${TARGET_NAME}_HOME)

target_include_directories(${TARGET_NAME} PUBLIC 
    "${SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/include"
    "${VTK_DIR}/include"
    "${LNLib_DIR}/include"
)

file(GLOB VTK_LIB_FILES ${VTK_DIR}/lib/*.lib)
list(APPEND LIBS ${VTK_LIB_FILES})
target_link_libraries(${TARGET_NAME} ${VTK_LIB_FILES})
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${TARGET_NAME} ND_LNLib)
add_dependencies(${TARGET_NAME} ND_LNLib)

file(GLOB rootfiles *.cpp *.h)
source_group("" FILES ${rootfiles})
target_sources(${TARGET_NAME} PRIVATE ${rootfiles})
SUBDIRLIST(SUBDIRS ${SOURCE_DIR})
foreach(subdir ${SUBDIRS})
    file(GLOB subdirFiles ${subdir}/*.cpp ${subdir}/*.h)
    string(REPLACE "/" "\\" subdir ${subdir})
    source_group(${subdir} FILES ${subdirFiles})
    target_sources(${TARGET_NAME} PRIVATE ${subdirFiles})
endforeach()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Z7")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EX_LINKER_FLAGS_RELEASE} /DEBUG")

if(MSVC)
    file(GLOB LNLib_DLLS ${LNLib_DIR}/bin/Release/*.dll)
    foreach(Current IN LISTS LNLib_DLLS)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Current} ${CMAKE_BINARY_DIR}/$<CONFIG>)
    endforeach()

    file(GLOB VTK_DLLS ${VTK_DIR}/bin/*.dll)
    foreach(Current IN LISTS VTK_DLLS)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Current} ${CMAKE_BINARY_DIR}/$<CONFIG>)
    endforeach()

    set(Torch_LIB_DIR "D:/Libtorch/libtorch/lib")
    file(GLOB Torch_DLLS ${Torch_LIB_DIR}/*.dll)
    foreach(Current IN LISTS Torch_DLLS)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${Current} ${CMAKE_BINARY_DIR}/$<CONFIG>)
    endforeach()
    set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${LNLib_DIR}/bin/$<CONFIG>;${VTK_DIR}/bin;${Torch_LIB_DIR};")
endif()