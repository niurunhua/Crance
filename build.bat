@echo off
REM Build script for Windows

echo Creating build directory...
if not exist build mkdir build
cd build

echo Running CMake...
cmake ..

echo Building...
cmake --build . --config Release

echo Build complete. Executable is in bin\

REM Copy model files (if any) to bin directory
if exist ..\models xcopy /E ..\models bin\models\
if exist ..\dataset xcopy /E ..\dataset bin\dataset\