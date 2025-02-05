set(TARGET_NAME ND_LNLib)
project(${TARGET_NAME} LANGUAGES CXX)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_library(${TARGET_NAME} SHARED "")
target_compile_definitions(${TARGET_NAME}  PRIVATE ${TARGET_NAME}_HOME)

target_include_directories(${TARGET_NAME} PUBLIC
	"${SOURCE_DIR}/include"
	"${LNLib_DIR}/include"
)

set(Torch_DIR "C:/Code/CodeReference/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories("C:/Code/CodeReference/libtorch/include","C:/Code/CodeReference/libtorch/include/torch/csrc/api/include")
target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")

target_link_libraries(${TARGET_NAME} ${LIBS} ${LNLib_DIR}/lib/$<CONFIG>/LNLib.lib)

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

if(MSVC)
    set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${LNLib_DIR}/bin/$<CONFIG>;")
endif()



