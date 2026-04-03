cmake_minimum_required(VERSION 3.16)
project(YOLOv11_Async_Detection CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/main.cpp
    src/FrameQueue.cpp
    src/Detector.cpp
    src/Tracker.cpp
    src/SerialPort.cpp
    src/AutoLabeler.cpp
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