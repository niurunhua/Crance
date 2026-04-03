#!/bin/bash
# Build script for Linux

set -e

echo "Creating build directory..."
mkdir -p build
cd build

echo "Running CMake..."
cmake ..

echo "Building..."
cmake --build . --config Release

echo "Build complete. Executable is in bin/"

# Copy model files (if any) to bin directory
if [ -d "../models" ]; then
    cp -r ../models bin/
fi

if [ -d "../dataset" ]; then
    cp -r ../dataset bin/
fi