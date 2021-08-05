cmake_minimum_required(VERSION 3.14)
project(GridGraph)

set(CMAKE_CXX_STANDARD 17)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "current platform: Linux")
    set(CMAKE_C_COMPILER /usr/bin/gcc-10)
    set(CMAKE_CXX_COMPILER /usr/bin/g++-10)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "current platform: Windows")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
add_compile_options(-mcmodel=large)
link_libraries(stdc++fs)

#CXXFLAGS?= -O3 -Wall -g
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()


add_executable(preprocessing tools/preprocess.cpp)
set_target_properties(preprocessing PROPERTIES LINKER_LANGUAGE CXX)

