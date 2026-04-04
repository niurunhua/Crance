cmake_minimum_required(VERSION 3.16)
project(YOLOv11_Async_Detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(/utf-8)

# Find OpenCV
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