cmake_minimum_required(VERSION 3.16)
project(YOLOv11_Async_Detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(/utf-8)

# vcpkg paths
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Administrator/source/repos/AR/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(CMAKE_PREFIX_PATH "C:/Users/Administrator/source/repos/AR/vcpkg/installed/x64-windows")

# OpenVINO path
set(OpenVINO_DIR "C:/Users/Administrator/openvino_toolkit_windows_2026.1.0.21367.63e31528c62_x86_64/runtime/cmake")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenVINO
find_package(OpenVINO REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/core/main.cpp
    src/core/FrameQueue.cpp
    src/vision/Detector.cpp
    src/vision/Tracker.cpp
    src/io/SerialPort.cpp
    src/vision/AutoLabeler.cpp
    src/vision/CameraManager.cpp
    src/vision/ContourProcessor.cpp
    src/core/CoordinateCalculator.cpp
    src/io/DataTransmitter.cpp
)

# Add executable
add_executable(yolov11_async ${SOURCES})

# Link OpenCV and OpenVINO
target_link_libraries(yolov11_async ${OpenCV_LIBS} openvino::runtime)

# On Windows, link serial port library
if(WIN32)
    target_link_libraries(yolov11_async setupapi)
endif()

# Set output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Inference test program
add_executable(test_inference test_inference.cpp
    src/vision/Detector.cpp
    src/vision/AutoLabeler.cpp
)
target_link_libraries(test_inference ${OpenCV_LIBS} openvino::runtime)