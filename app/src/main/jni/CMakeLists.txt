project(yolov11ncnn)

cmake_minimum_required(VERSION 3.10)

# opencv4 requires c++11
set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov11ncnn SHARED yolov11ncnn.cpp ndkcamera.cpp yolov11.cpp) 

target_link_libraries(yolov11ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
