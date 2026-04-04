/**
 * @file main.cpp
 * @brief 双相机并行视觉系统主程序
 *
 * 工业级视觉系统，支持双相机并行处理、模块化架构、强容错能力。
 * 包含数字相机（识别数字）和豆子相机（识别豆子）双路并行处理。
 *
 * 核心特性：
 * 1. 双相机并行与多线程管理
 * 2. 动态调参UI与真实FPS计算
 * 3. 轮廓防爆与ROI局部推理
 * 4. 坐标计算与防抖状态机
 * 5. 工业级串口协议与重定向
 */

#include "../vision/CameraManager.h"
#include "Config.h"
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdlib>

// 全局运行标志
std::atomic<bool> g_running{true};

// 信号处理函数
void signalHandler(int signal) {
    std::cout << "接收到信号 " << signal << "，正在关闭程序..." << std::endl;
    g_running = false;
}

/**
 * @brief 主函数
 *
 * 程序流程：
 * 1. 初始化相机管理器（配置双相机）
 * 2. 启动相机处理线程
 * 3. 主循环：显示结果、处理用户输入
 * 4. 清理资源
 */
int main(int argc, char** argv) {
    // 设置控制台编码为UTF-8（修复中文乱码）
    system("chcp 65001");

    // 设置信号处理
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 设置OpenCV线程数
    cv::setNumThreads(std::thread::hardware_concurrency());

    std::cout << "========================================" << std::endl;
    std::cout << "   双相机并行视觉系统 - 工业级架构" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "作者: Crane视觉团队" << std::endl;
    std::cout << "版本: 2.0 (双相机重构版)" << std::endl;
    std::cout << "编译日期: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // 配置数字相机
    CameraConfig digitConfig;
    digitConfig.type = CAMERA_DIGIT;
    digitConfig.cameraIndex = Config::DIGIT_CAMERA_ID;      // 数字相机索引（从全局配置读取）
    digitConfig.modelPath = Config::MODEL_PATH;
    digitConfig.classesFile = Config::CLASSES_FILE;
    digitConfig.enabled = true;
    digitConfig.skipFrames = Config::DIGIT_SKIP_FRAMES;     // 跳帧数（从全局配置读取）
    digitConfig.sourceId = Config::DIGIT_SOURCE_ID;         // 数据源ID（从全局配置读取）
    digitConfig.exposure = Config::DIGIT_EXPOSURE;          // 曝光值（从全局配置读取）

    // 配置豆子相机
    CameraConfig beanConfig;
    beanConfig.type = CAMERA_BEAN;
    beanConfig.cameraIndex = Config::BEAN_CAMERA_ID;        // 豆子相机索引（从全局配置读取）
    beanConfig.modelPath = Config::MODEL_PATH;              // 可以使用不同的模型
    beanConfig.classesFile = Config::CLASSES_FILE;
    beanConfig.enabled = true;
    beanConfig.skipFrames = Config::BEAN_SKIP_FRAMES;       // 跳帧数（从全局配置读取）
    beanConfig.sourceId = Config::BEAN_SOURCE_ID;           // 数据源ID（从全局配置读取）
    beanConfig.exposure = Config::BEAN_EXPOSURE;            // 曝光值（从全局配置读取）

    // 创建相机管理器
    CameraManager cameraManager;

    // 初始化相机管理器
    std::cout << "正在初始化相机管理器..." << std::endl;
    if (!cameraManager.init(digitConfig, beanConfig)) {
        std::cerr << "错误：相机管理器初始化失败，程序退出" << std::endl;
        return -1;
    }

    // 启动相机线程
    std::cout << "正在启动相机处理线程..." << std::endl;
    cameraManager.start();

    std::cout << std::endl;
    std::cout << "           系统运行中..." << std::endl;
    std::cout << std::endl;

    // 创建独立显示窗口
    const std::string digitWindow = "Digit Camera";
    const std::string beanWindow = "Bean Camera";
    cv::namedWindow(digitWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(beanWindow, cv::WINDOW_NORMAL);
    cv::resizeWindow(digitWindow, Config::INPUT_WIDTH, Config::INPUT_HEIGHT);
    cv::resizeWindow(beanWindow, Config::INPUT_WIDTH, Config::INPUT_HEIGHT);

    // 调试信息显示控制
    bool showDebugInfo = true;

    // 主显示循环
    while (g_running) {
        // 获取相机结果
        CameraResult digitResult, beanResult;
        bool hasDigit = cameraManager.getResult(CAMERA_DIGIT, digitResult);
        bool hasBean = cameraManager.getResult(CAMERA_BEAN, beanResult);

        // 准备显示图像
        cv::Mat digitDisplay, beanDisplay;

        // 数字相机显示
        bool digitHealthy = cameraManager.isCameraHealthy(CAMERA_DIGIT);
        if (digitHealthy && hasDigit) {
            // 相机健康且抓到了帧，无论检测是否成功都显示图像
            if (!digitResult.frame.empty()) {
                digitDisplay = digitResult.frame.clone();
            } else {
                // 保底：创建黑底图像
                digitDisplay = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
                cv::putText(digitDisplay, "Digit Camera: No Frame",
                           cv::Point(50, Config::INPUT_HEIGHT / 2),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }

            // 绘制调试信息（如果启用）
            if (showDebugInfo && !digitDisplay.empty()) {
                // 显示FPS信息
                if (digitResult.fps > 0) {
                    std::string fpsText = "Digit FPS: " + std::to_string(static_cast<int>(digitResult.fps));
                    cv::putText(digitDisplay, fpsText,
                               cv::Point(10, digitDisplay.rows - 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                }

                // 显示检测状态
                std::string statusText = digitResult.success ? "Detection: OK" : "Detection: No Target";
                cv::putText(digitDisplay, statusText,
                           cv::Point(10, digitDisplay.rows - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        } else {
            // 相机不健康或无结果，显示离线消息
            digitDisplay = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
            cv::putText(digitDisplay, digitHealthy ? "Digit Camera: Initializing..." : "Digit Camera: Offline",
                       cv::Point(50, Config::INPUT_HEIGHT / 2),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        // 豆子相机显示
        bool beanHealthy = cameraManager.isCameraHealthy(CAMERA_BEAN);
        if (beanHealthy && hasBean) {
            // 相机健康且抓到了帧，无论检测是否成功都显示图像
            if (!beanResult.frame.empty()) {
                beanDisplay = beanResult.frame.clone();
            } else {
                // 保底：创建黑底图像
                beanDisplay = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
                cv::putText(beanDisplay, "Bean Camera: No Frame",
                           cv::Point(50, Config::INPUT_HEIGHT / 2),
                           cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }

            // 绘制调试信息（如果启用）
            if (showDebugInfo && !beanDisplay.empty()) {
                // 显示FPS信息
                if (beanResult.fps > 0) {
                    std::string fpsText = "Bean FPS: " + std::to_string(static_cast<int>(beanResult.fps));
                    cv::putText(beanDisplay, fpsText,
                               cv::Point(10, beanDisplay.rows - 30),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                }

                // 显示检测状态
                std::string statusText = beanResult.success ? "Detection: OK" : "Detection: No Target";
                cv::putText(beanDisplay, statusText,
                           cv::Point(10, beanDisplay.rows - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
        } else {
            // 相机不健康或无结果，显示离线消息
            beanDisplay = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
            cv::putText(beanDisplay, beanHealthy ? "Bean Camera: Initializing..." : "Bean Camera: Offline",
                       cv::Point(50, Config::INPUT_HEIGHT / 2),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        // 独立显示到各自窗口
        if (!digitDisplay.empty()) {
            cv::imshow(digitWindow, digitDisplay);
        }
        if (!beanDisplay.empty()) {
            cv::imshow(beanWindow, beanDisplay);
        }

        // 处理键盘输入
        int key = cv::waitKey(1);
        if (key == 27) { // ESC键
            g_running = false;
            break;
        } else if (key == '1') { // 启用/禁用数字相机
            bool enabled = cameraManager.isCameraHealthy(CAMERA_DIGIT);
            cameraManager.setDebugMode(!enabled, cameraManager.isCameraHealthy(CAMERA_BEAN));
            std::cout << (enabled ? "禁用" : "启用") << "数字相机" << std::endl;
        } else if (key == '2') { // 启用/禁用豆子相机
            bool enabled = cameraManager.isCameraHealthy(CAMERA_BEAN);
            cameraManager.setDebugMode(cameraManager.isCameraHealthy(CAMERA_DIGIT), !enabled);
            std::cout << (enabled ? "禁用" : "启用") << "豆子相机" << std::endl;
        } else if (key == 'd') { // 切换调试信息
            showDebugInfo = !showDebugInfo;
            std::cout << (showDebugInfo ? "显示" : "隐藏") << "调试信息" << std::endl;
        } else if (key == 't') { // 截取当前帧
            static int captureCount = 0;
            std::string filename = "capture_" + std::to_string(captureCount++) + ".png";
            if (hasDigit) {
                cv::imwrite("digit_" + filename, digitResult.frame);
            }
            if (hasBean) {
                cv::imwrite("bean_" + filename, beanResult.frame);
            }
            std::cout << "截取当前帧: " << filename << std::endl;
        }

        // 降低CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 停止相机管理器
    std::cout << "正在停止相机管理器..." << std::endl;
    cameraManager.stop();

    // 关闭所有窗口
    cv::destroyAllWindows();

    std::cout << "========================================" << std::endl;
    std::cout << "     程序正常退出，感谢使用！" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}