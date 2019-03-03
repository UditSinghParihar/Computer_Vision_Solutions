cmake_minimum_required(VERSION 2.8)
project(OpenCV)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(sift_matcher sift_matcher.cpp)
add_executable(window_matching window_matching.cpp)
add_executable(rectify rectify.cpp)

target_link_libraries(sift_matcher ${OpenCV_LIBS})
target_link_libraries(window_matching ${OpenCV_LIBS})
target_link_libraries(rectify ${OpenCV_LIBS})