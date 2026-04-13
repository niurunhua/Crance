cmake_minimum_required(VERSION 3.16)
project(YOLOv11_Async_Detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(/utf-8)

# Find OpenCV
# 使用vcpkg安装的OpenCV
set(OpenCV_DIR "C:/Users/Administrator/source/repos/AR/vcpkg/installed/x64-windows/share/opencv4")
find_package(OpenCV REQUIRED)

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

# Link OpenCV
target_link_libraries(yolov11_async ${OpenCV_LIBS})

# On Windows, link serial port library
if(WIN32)
    target_link_libraries(yolov11_async setupapi)
endif()

# Set output directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Test program
add_executable(test_model test_model.cpp)
target_link_libraries(test_model ${OpenCV_LIBS})

# Speed test program
add_executable(test_speed test_speed.cpp)
target_link_libraries(test_speed ${OpenCV_LIBS})

# Detection test
add_executable(test_det test_speed.cpp)
target_link_libraries(test_det ${OpenCV_LIBS})

# Config test
add_executable(test_config test_config.cpp)