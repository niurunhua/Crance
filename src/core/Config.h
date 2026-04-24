#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace Config {
    // 相机配置
    inline constexpr int DIGIT_CAMERA_ID = 1;      // 数字相机索引
    inline constexpr int BEAN_CAMERA_ID = 0;       // 豆子相机索引
    inline constexpr double DIGIT_EXPOSURE = 1.0;  // 数字相机曝光
    inline constexpr double BEAN_EXPOSURE = -4.0;  // 豆子相机曝光
    inline constexpr int DIGIT_SKIP_FRAMES = 0;
    inline constexpr int BEAN_SKIP_FRAMES = 0;
    inline constexpr int DIGIT_SOURCE_ID = 0x01;   // 数据源标识
    inline constexpr int BEAN_SOURCE_ID = 0x02;

    // 模型配置
    inline const std::string MODEL_PATH = "E:/qzj2/weights/best.onnx";
    inline const std::string CLASSES_FILE = "E:/yolov11/datasets/labels/train/classes.txt";
    inline constexpr int NETWORK_WIDTH = 640;
    inline constexpr int NETWORK_HEIGHT = 640;
    inline constexpr float CONFIDENCE_THRESHOLD = 0.9f;  // 置信度阈值
    inline constexpr float NMS_THRESHOLD = 0.4f;         // NMS阈值
    inline constexpr int NUM_CLASSES = 8;

    // 自动标注
    inline constexpr bool AUTO_LABEL_ENABLED = false;
    inline const std::string AUTO_LABEL_OUTPUT_DIR = "E:/yolov11/auto_labels";
    inline constexpr float AUTO_LABEL_CONFIDENCE = 0.80f;

    // 图像处理
    inline constexpr int THRESHOLD = 200;           // 二值化阈值
    inline constexpr int MIN_AREA = 200;            // 最小轮廓面积
    inline constexpr int LOST_BUFFER_FRAMES = 10;   // 目标丢失缓冲
    inline constexpr int INFERENCE_INTERVAL = 20;   // 推理间隔帧数
    inline constexpr float FILTER_ALPHA = 0.5f;     // 滤波系数

    // 输入输出
    inline constexpr int INPUT_WIDTH = 640;
    inline constexpr int INPUT_HEIGHT = 640;
    inline constexpr int SCREEN_CENTER_X = INPUT_WIDTH / 2;
    inline constexpr int SCREEN_CENTER_Y = INPUT_HEIGHT / 2;
    inline const std::string SERIAL_PORT = "COM3";
    inline constexpr int SERIAL_BAUD = 115200;
}

#endif
