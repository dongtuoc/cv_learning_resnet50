cmake_minimum_required(VERSION 3.10)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
endif()

project(resnet)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mavx -Ofast")

add_executable(resnet main.cc resnet_no_malloc.cc)

file(GLOB OPS_SOURCES "ops/*.cc")
target_sources(resnet PRIVATE ${OPS_SOURCES})

include_directories(/usr/include/opencv4)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(resnet PRIVATE ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
