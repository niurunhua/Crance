#include "src/vision/Detector.h"
#include "src/vision/AutoLabeler.h"
#include "src/core/Config.h"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    system("chcp 65001");
    std::cout << "=== 推理和自动标注测试 ===" << std::endl;

    // 初始化检测器
    Detector detector(
        Config::MODEL_PATH,
        Config::CLASSES_FILE,
        Config::NETWORK_WIDTH,
        Config::NETWORK_HEIGHT,
        Config::CONFIDENCE_THRESHOLD,
        Config::NMS_THRESHOLD,
        false
    );

    if (!detector.init()) {
        std::cerr << "检测器初始化失败" << std::endl;
        return -1;
    }
    std::cout << "检测器初始化成功" << std::endl;

    // 初始化自动标注器
    AutoLabeler autoLabeler(Config::AUTO_LABEL_OUTPUT_DIR, Config::AUTO_LABEL_CONFIDENCE);
    std::cout << "自动标注器初始化成功，输出目录: " << Config::AUTO_LABEL_OUTPUT_DIR << std::endl;

    // 查找测试图像
    std::vector<std::string> imageFiles;

    // 从命令行参数获取图像路径
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            imageFiles.push_back(argv[i]);
        }
    }

    // 尝试多个可能的目录
    std::vector<std::string> searchDirs = {
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images",
        "."
    };

    for (const auto& dir : searchDirs) {
        if (imageFiles.empty()) {
            try {
                cv::glob(dir + "/*.jpg", imageFiles, false);
                cv::glob(dir + "/*.png", imageFiles, false);
            } catch (...) {
                // 目录不存在，跳过
            }
        }
    }

    // 创建测试图像用于验证推理
    if (imageFiles.empty()) {
        std::cout << "未找到图像文件，创建测试图像..." << std::endl;
        cv::Mat testImg(640, 640, CV_8UC3, cv::Scalar(200, 200, 200));
        // 添加一些随机噪声模拟真实图像
        cv::randn(testImg, cv::Scalar(128, 128, 128), cv::Scalar(50, 50, 50));
        cv::imwrite("test_inference.jpg", testImg);
        imageFiles.push_back("test_inference.jpg");
    }

    std::cout << "找到 " << imageFiles.size() << " 张测试图像" << std::endl;

    int totalDetections = 0;
    int totalLabeled = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (const auto& imgPath : imageFiles) {
        cv::Mat frame = cv::imread(imgPath);
        if (frame.empty()) {
            std::cout << "无法读取: " << imgPath << std::endl;
            continue;
        }

        std::cout << "处理: " << imgPath << " (" << frame.cols << "x" << frame.rows << ")" << std::endl;

        std::vector<Detection> detections;
        auto t1 = std::chrono::high_resolution_clock::now();
        detector.detect(frame, detections);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto inferTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

        std::cout << "  检测到 " << detections.size() << " 个目标, 耗时: " << inferTime.count() << "ms" << std::endl;
        for (const auto& det : detections) {
            std::cout << "    [类别:" << det.classId << " 置信度:" << det.confidence
                      << " 位置:(" << det.box.x << "," << det.box.y << ")]" << std::endl;
        }

        if (!detections.empty()) {
            totalDetections += detections.size();

            // 绘制检测结果
            detector.drawDetections(frame, detections);
            std::string resultPath = "result_" + imgPath.substr(imgPath.find_last_of("/\\") + 1);
            cv::imwrite(resultPath, frame);
            std::cout << "  结果保存: " << resultPath << std::endl;

            // 自动标注
            int labeled = autoLabeler.process(frame, detections);
            if (labeled > 0) {
                totalLabeled += labeled;
                std::cout << "  自动标注: " << labeled << " 个目标" << std::endl;
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << std::endl;
    std::cout << "=== 测试结果 ===" << std::endl;
    std::cout << "处理图像: " << imageFiles.size() << " 张" << std::endl;
    std::cout << "总检测数: " << totalDetections << std::endl;
    std::cout << "自动标注: " << totalLabeled << " 个目标" << std::endl;
    std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
    std::cout << "自动标注输出目录: " << Config::AUTO_LABEL_OUTPUT_DIR << std::endl;

    return 0;
}
