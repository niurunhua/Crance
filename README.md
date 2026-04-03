# YOLOv11 Async Detection System

C++ real-time object detection and tracking system using OpenCV DNN module with YOLOv11 model. Designed for digit (1-5) and bean (3 types) detection with position tracking.

## Features

- **Multi-threaded Producer-Consumer Architecture**: Separates image capture and inference to maximize frame rate and eliminate latency.
- **Asynchronous Inference**: Uses OpenCV DNN module for YOLOv11 inference on CPU.
- **Target Tracking & Offset Calculation**: Computes X/Y pixel offsets relative to screen center (640x480).
- **Low-pass Filter**: Smooths coordinate output to reduce jitter.
- **Target Lost Buffer**: Maintains output for several frames when detection fails to prevent sudden jumps.
- **Serial Communication**: Implements 11-byte hard-verification protocol to send signed offsets, class ID, and heartbeat to microcontroller.
- **Auto-Labeling**: Automatically saves images and generates YOLO-format label files when detection confidence exceeds threshold (e.g., 0.85).

## Requirements

- OpenCV 4.x (with DNN module)
- C++17 compiler
- CMake 3.16+
- On Windows: Serial port library (included in Windows SDK)
- On Linux: termios

## Build Instructions

1. Clone or extract the source code.
2. Install OpenCV and ensure it's discoverable by CMake.
3. Create a build directory and run CMake:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

4. The executable will be placed in `bin/` directory.

## Model Preparation

1. Export your YOLOv11 model to ONNX format (assuming input size 640x640).
2. Place the ONNX model file at `models/yolov11.onnx`.
3. Create a class names file `models/classes.txt` with one class per line (total 8 classes: digits 1-5 and three bean types). Example:

```
1
2
3
4
5
bean_type1
bean_type2
bean_type3
```

## Configuration

Edit `src/Config.h` to adjust parameters:

- `MODEL_PATH`: Path to ONNX model.
- `CLASSES_FILE`: Path to class names file.
- `NETWORK_WIDTH`, `NETWORK_HEIGHT`: Model input dimensions (should match your model).
- `INPUT_WIDTH`, `INPUT_HEIGHT`: Camera frame size (640x480 by default).
- `CONFIDENCE_THRESHOLD`, `NMS_THRESHOLD`: Detection thresholds.
- `AUTO_LABEL_THRESH`: Confidence threshold for auto-labeling.
- `SERIAL_PORT`: Serial port name (COM3 on Windows, /dev/ttyUSB0 on Linux).
- `SERIAL_BAUD`: Baud rate.
- `FILTER_ALPHA`: Low-pass filter coefficient (smaller = smoother but more lag).
- `LOST_BUFFER_FRAMES`: Number of frames to maintain output after detection lost.

## Usage

Run the executable:

```bash
./bin/yolov11_async
```

- Press ESC to exit.
- The system will capture from camera index 0 (default). To change camera index, modify `main.cpp`.
- Detected objects are drawn on screen with bounding boxes and center points.
- X/Y offsets are displayed on screen and sent via serial port.
- High-confidence detections trigger auto-labeling; images and labels are saved in `dataset/images/` and `dataset/labels/`.

## Serial Protocol

The system sends 11-byte packets with the following format:

| Byte | Content         | Description                           |
|------|-----------------|---------------------------------------|
| 0    | 0xAA            | Start marker                          |
| 1-2  | int16_t (LE)    | X offset (signed, pixels)             |
| 3-4  | int16_t (LE)    | Y offset (signed, pixels)             |
| 5    | uint8_t         | Class ID (0-255)                      |
| 6    | uint8_t         | Heartbeat counter (increments each frame) |
| 7-9  | 0x00            | Reserved (zero)                       |
| 10   | uint8_t         | Checksum (sum of bytes 0-9 modulo 256) |

## Auto-Labeling

When a detection confidence exceeds `AUTO_LABEL_THRESH`, the current frame and its detections are saved. Images are stored as JPEG, labels as YOLO-format text files (normalized coordinates). The dataset grows incrementally during operation.

## Notes

- Ensure the camera delivers frames at 640x480 resolution; the program will resize if needed.
- If serial port cannot be opened, the program continues without sending data.
- The tracker selects the detection closest to screen center as the primary target.
- The low-pass filter is a first-order exponential smoothing filter: `filtered = alpha * new + (1-alpha) * filtered`.

## Troubleshooting

- **Model fails to load**: Check that the ONNX file path is correct and the model is compatible with OpenCV DNN.
- **No detections**: Adjust confidence threshold, verify model outputs match expected format.
- **High CPU usage**: Reduce frame queue size, consider lowering camera FPS.
- **Serial port not working**: Verify port name and baud rate; check permissions on Linux.

## License

This project is provided as-is for educational and research purposes.