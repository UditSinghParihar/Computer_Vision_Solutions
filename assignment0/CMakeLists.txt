cmake_minimum_required (VERSION 3.5.1)
project (assignment0)
set (CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(include)

add_executable(video_to_images src/video_to_images.cpp)
add_executable(images_to_video src/images_to_video.cpp)
add_executable(chroma_key src/chroma_key.cpp)

target_link_libraries(video_to_images ${OpenCV_LIBS})
target_link_libraries(images_to_video ${OpenCV_LIBS})
target_link_libraries(chroma_key ${OpenCV_LIBS})