cmake_minimum_required(VERSION 3.5.1)
project(ftrl)

#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS -DUSEOMP)

set(SOURCE_FILES library.cpp library.h)
add_library(ftrl SHARED ${SOURCE_FILES})

add_executable(ftrl-exec ${SOURCE_FILES})