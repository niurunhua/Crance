#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

namespace Config {
    // Model paths
    const std::string MODEL_PATH = "C:/Users/Administrator/runs/detect/train9/weights/best.onnx"; // change to your model path
    const std::string CLASSES_FILE = "E:/yolov11/datasets/labels/train/classes.txt"; // optional

    // Image dimensions
    const int NETWORK_WIDTH = 640; // YOLO common size (32*13)
    const int NETWORK_HEIGHT = 640; // YOLO common size (32*13)
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;

    // Detection threshold
    const float CONFIDENCE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.4f;

    // Number of classes
    const int NUM_CLASSES = 5; // digits 1-5 + 3 bean types

    // Auto-labeling threshold
    const float AUTO_LABEL_THRESH = 0.85f;

    // Serial port
    const std::string SERIAL_PORT = "COM3"; // adjust for your system
    const int SERIAL_BAUD = 115200;

    // Filter parameters (low-pass filter)
    const float FILTER_ALPHA = 0.3f; // smaller = smoother but more lag

    // Target lost buffer (frames to keep output after detection lost)
    const int LOST_BUFFER_FRAMES = 5;

    // Center of screen (for offset calculation)
    const int SCREEN_CENTER_X = INPUT_WIDTH / 2;
    const int SCREEN_CENTER_Y = INPUT_HEIGHT / 2;
}

#endif // CONFIG_H