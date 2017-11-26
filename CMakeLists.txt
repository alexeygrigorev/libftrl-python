cmake_minimum_required(VERSION 3.5.1)
project(ftrl)

set(CMAKE_CXX_FLAGS "-fopenmp -DUSEOMP")

set(SOURCE_FILES "src/ftrl.cpp" "src/ftrl.h")
add_library(ftrl SHARED ${SOURCE_FILES})