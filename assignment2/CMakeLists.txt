cmake_minimum_required(VERSION 2.8)
project(OpenCV)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(sift_matcher src/sift_matcher.cpp)
add_executable(panaroma_taj_mahal src/panaroma_taj_mahal.cpp)

target_link_libraries(sift_matcher ${OpenCV_LIBS})
target_link_libraries(panaroma_taj_mahal ${OpenCV_LIBS})