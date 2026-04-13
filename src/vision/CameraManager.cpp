#include "CameraManager.h"
#include "Detector.h"
#include "../io/DataTransmitter.h"
#include "ContourProcessor.h"
#include "AutoLabeler.h"
#include "../core/CoordinateCalculator.h"
#include "../core/Config.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <future>

// 构造函数
CameraManager::CameraManager()
    : m_running(false)
    , m_digitHealthy(false)
    , m_beanHealthy(false) {
}

// 析构函数
CameraManager::~CameraManager() {
    stop();
}

// 初始化相机管理器
bool CameraManager::init(const CameraConfig& digitConfig, const CameraConfig& beanConfig) {
    m_digitConfig = digitConfig;
    m_beanConfig = beanConfig;

    std::cout << "初始化相机管理器..." << std::endl;

    // 初始化数字相机
    bool digitOk = initCamera(m_digitConfig);
    if (digitOk) {
        std::cout << "数字相机初始化成功" << std::endl;
        // 注意：健康状态将由相机线程函数设置
        m_digitStatus = "检测器就绪";
    } else {
        std::cout << "数字相机初始化失败，将跳过此分支" << std::endl;
        m_digitHealthy = false;
        m_digitStatus = "故障";
        m_digitConfig.enabled = false;
    }

    // 初始化豆子相机
    bool beanOk = initCamera(m_beanConfig);
    if (beanOk) {
        std::cout << "豆子相机初始化成功" << std::endl;
        // 注意：健康状态将由相机线程函数设置
        m_beanStatus = "检测器就绪";
    } else {
        std::cout << "豆子相机初始化失败，将跳过此分支" << std::endl;
        m_beanHealthy = false;
        m_beanStatus = "故障";
        m_beanConfig.enabled = false;
    }

    // 初始化数据发射器
    m_transmitter = std::make_shared<DataTransmitter>();
    if (!m_transmitter->initSerial(Config::SERIAL_PORT, Config::SERIAL_BAUD)) {
        std::cout << "串口初始化失败，将使用终端输出" << std::endl;
    }

    // 初始化自动标注器
    std::cout << "=== 检查自动标注配置 ===" << std::endl;
    std::cout << "AUTO_LABEL_ENABLED = " << Config::AUTO_LABEL_ENABLED << std::endl;
    if (Config::AUTO_LABEL_ENABLED) {
        m_autoLabeler = std::make_unique<AutoLabeler>(
            Config::AUTO_LABEL_OUTPUT_DIR,
            Config::AUTO_LABEL_CONFIDENCE
        );
        std::cout << ">>> 自动标注已启用，输出目录: " << Config::AUTO_LABEL_OUTPUT_DIR << std::endl;
    } else {
        std::cout << ">>> 自动标注未启用" << std::endl;
    }


    // 检查是否有至少一个相机可用
    if (!m_digitConfig.enabled && !m_beanConfig.enabled) {
        std::cerr << "错误：所有相机都不可用，程序无法运行" << std::endl;
        return false;
    }

    std::cout << "相机管理器初始化完成" << std::endl;
    std::cout << "  数字相机: " << (m_digitConfig.enabled ? "启用" : "禁用")
              << " (" << m_digitStatus << ")" << std::endl;
    std::cout << "  豆子相机: " << (m_beanConfig.enabled ? "启用" : "禁用")
              << " (" << m_beanStatus << ")" << std::endl;

    return true;
}

// 初始化单个相机
bool CameraManager::initCamera(CameraConfig& config) {
    // 注意：不再在此处打开相机！相机的打开和轮询逻辑已移至 cameraThreadFunc 中
    // 这里只做配置检查和检测器初始化

    std::cout << "初始化相机配置（类型: " << (config.type == CAMERA_DIGIT ? "数字" : "豆子") << ")" << std::endl;

    // 初始化检测器
    std::unique_ptr<Detector> detector;
    if (!config.modelPath.empty()) {
        detector = std::make_unique<Detector>(
            config.modelPath,
            config.classesFile,
            Config::NETWORK_WIDTH,
            Config::NETWORK_HEIGHT,
            Config::CONFIDENCE_THRESHOLD,
            Config::NMS_THRESHOLD,
            false  // 所有相机都不使用白纸检测，直接全图推理
        );

        if (!detector->init()) {
            std::cerr << "检测器初始化失败: " << config.modelPath << std::endl;
            return false;
        }

        std::cout << "检测器初始化成功" << std::endl;
    } else {
        std::cout << "警告：模型路径为空，将跳过检测步骤" << std::endl;
    }

    // 存储检测器
    if (config.type == CAMERA_DIGIT) {
        m_digitDetector = std::move(detector);
    } else {
        m_beanDetector = std::move(detector);
    }

    return true;
}

// 启动所有相机处理线程
void CameraManager::start() {
    if (m_running) {
        std::cout << "相机管理器已经在运行" << std::endl;
        return;
    }

    m_running = true;

    // 启动数字相机线程
    if (m_digitConfig.enabled) {
        m_digitThread = std::thread(&CameraManager::cameraThreadFunc, this, m_digitConfig);
        std::cout << "数字相机线程已启动" << std::endl;
    }

    // 启动豆子相机线程
    if (m_beanConfig.enabled) {
        m_beanThread = std::thread(&CameraManager::cameraThreadFunc, this, m_beanConfig);
        std::cout << "豆子相机线程已启动" << std::endl;
    }

    std::cout << "所有相机线程已启动" << std::endl;
}

// 停止所有相机处理线程
void CameraManager::stop() {
    if (!m_running) {
        return;
    }

    std::cout << "正在停止相机管理器..." << std::endl;
    m_running = false;

    // 等待线程结束
    if (m_digitThread.joinable()) {
        m_digitThread.join();
        std::cout << "数字相机线程已停止" << std::endl;
    }

    if (m_beanThread.joinable()) {
        m_beanThread.join();
        std::cout << "豆子相机线程已停止" << std::endl;
    }

    // 关闭串口
    if (m_transmitter) {
        m_transmitter->closeSerial();
    }

    std::cout << "相机管理器已停止" << std::endl;
}

// 相机处理线程函数
void CameraManager::cameraThreadFunc(CameraConfig config) {
    // 【错峰启动】机制：豆子相机休眠2秒，让数字相机先完成初始化
    if (config.type == CAMERA_BEAN) {
        std::cout << "豆子相机线程：正在休眠2秒，避免硬件并发抢占..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    cv::VideoCapture cap;
    bool cameraFound = false;

    std::cout << "线程内尝试打开相机索引: " << config.cameraIndex << std::endl;

    try {
        // 直接尝试打开配置中指定的专属索引，不要轮询！
        cap.open(config.cameraIndex);
        if (!cap.isOpened()) {
            std::cout << "  相机打开失败，索引: " << config.cameraIndex << std::endl;
            if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
            if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
            return;
        }

        std::cout << "  相机已打开，进行暖机初始化..." << std::endl;

        // 不强制设置分辨率，使用相机原生分辨率
        // cap.set(cv::CAP_PROP_FRAME_WIDTH, Config::INPUT_WIDTH);
        // cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::INPUT_HEIGHT);

        // 关闭自动曝光，设置短曝光时间（根除运动模糊）
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // 手动曝光模式
        cap.set(cv::CAP_PROP_EXPOSURE, config.exposure); // 使用配置的曝光值

        // 限制底层缓冲区大小（减少延迟）
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        // 保留30帧暖机逻辑：循环cap.read() 30次等待画面稳定
        cv::Mat testFrame;
        bool foundValidFrame = false;

        for (int attempt = 0; attempt < 30; ++attempt) {
            if (cap.read(testFrame) && !testFrame.empty()) {
                std::cout << "  找到有效相机 " << config.cameraIndex << "，帧尺寸: "
                          << testFrame.cols << "x" << testFrame.rows << std::endl;
                foundValidFrame = true;
                break;
            }

            // 等待相机初始化完成
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        if (!foundValidFrame) {
            // 暖机后全为空帧，判定失败
            std::cout << "  相机 " << config.cameraIndex << " 暖机后仍无有效帧，判定失败" << std::endl;
            cap.release();
            if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
            if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
            return;
        }


        // 相机成功打开并暖机完成，设置健康标志
        if (config.type == CAMERA_DIGIT) m_digitHealthy.store(true);
        if (config.type == CAMERA_BEAN) m_beanHealthy.store(true);

        std::cout << "线程内使用相机索引: " << config.cameraIndex << " (成功)" << std::endl;

    } catch (...) {
        // 捕获任何异常，防止崩溃影响其他线程
        std::cout << "  相机 " << config.cameraIndex << " 初始化过程中发生异常" << std::endl;
        if (cap.isOpened()) {
            cap.release();
        }
        if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
        if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
        return;
    }

    // 获取对应的检测器
    Detector* detector = nullptr;
    if (config.type == CAMERA_DIGIT) {
        detector = m_digitDetector.get();
    } else {
        detector = m_beanDetector.get();
    }

    // 初始化轮廓处理器（仅数字相机需要）
    std::unique_ptr<ContourProcessor> contourProcessor;
    std::unique_ptr<CoordinateCalculator> coordCalculator;

    if (config.type == CAMERA_DIGIT) {
        contourProcessor = std::make_unique<ContourProcessor>();
        coordCalculator = std::make_unique<CoordinateCalculator>();
        coordCalculator->setImageSize(Config::INPUT_WIDTH, Config::INPUT_HEIGHT);
    }

    // 帧计数器（用于降帧处理）
    int frameCounter = 0;
    cv::Mat frame;
    cv::Mat displayFrame;  // 用于显示的帧

    // 上一帧的检测结果（用于显示稳定性）
    CameraResult lastResult;

    std::cout << "相机线程 " << (config.type == CAMERA_DIGIT ? "数字" : "豆子")
              << " 开始运行" << std::endl;

    // FPS计算变量
    auto lastFpsTime = std::chrono::steady_clock::now();
    int fpsFrameCount = 0;
    double currentFps = 0.0;

    // 保存上一次检测结果
    std::vector<Detection> lastDetections;
    int noDetectionFrames = 0;  // 连续未检测到的帧数
    const int maxNoDetectionFrames = 5;  // 5次推理(约1秒)后清除旧框

    while (m_running) {
        // 抓取帧
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        displayFrame = frame;

        // 计算FPS
        fpsFrameCount++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFpsTime);
        if (elapsed.count() >= 1000) {
            currentFps = fpsFrameCount * 1000.0 / elapsed.count();
            fpsFrameCount = 0;
            lastFpsTime = currentTime;
        }

        // 每6帧推理1次（两个相机推理不同任务，无需互斥锁）
        if (frameCounter % 6 == 0 && detector) {
            std::vector<Detection> detections;
            detector->detect(displayFrame, detections);
            if (!detections.empty()) {
                lastDetections = detections;
                noDetectionFrames = 0;  // 检测成功，重置计数器

                // 自动标注（仅豆子相机）
                if (config.type == CAMERA_BEAN && m_autoLabeler) {
                    int saved = m_autoLabeler->process(displayFrame, detections);
                    if (saved > 0) {
                        std::cout << "[自动标注] 保存了 " << saved << " 个检测" << std::endl;
                    }
                }
            } else {
                noDetectionFrames++;
                // 连续多帧未检测到，清除旧框
                if (noDetectionFrames >= maxNoDetectionFrames / 10) {
                    lastDetections.clear();
                }
            }
        }

        // 绘制上一次检测结果
        if (!lastDetections.empty()) {
            detector->drawDetections(displayFrame, lastDetections);
            const auto& det = lastDetections[0];
            if (m_transmitter) {
                DataPacket packet;
                packet.sourceId = config.sourceId;
                packet.classId = det.classId;
                packet.dx = static_cast<int16_t>(det.center.x - Config::SCREEN_CENTER_X);
                packet.dy = static_cast<int16_t>(det.center.y - Config::SCREEN_CENTER_Y);
                m_transmitter->sendPacket(packet);
            }
        }

        // 更新显示
        if (config.type == CAMERA_DIGIT) {
            std::lock_guard<std::mutex> lock(m_digitMutex);
            m_digitResult.frame = displayFrame;
            m_digitResult.fps = currentFps;
        } else {
            std::lock_guard<std::mutex> lock(m_beanMutex);
            m_beanResult.frame = displayFrame;
            m_beanResult.fps = currentFps;
        }

        frameCounter++;
    }

    cap.release();
    std::cout << "相机线程 " << (config.type == CAMERA_DIGIT ? "数字" : "豆子")
              << " 已退出" << std::endl;
}

// 处理单帧图像
bool CameraManager::processFrame(const CameraConfig& config, const cv::Mat& frame, CameraResult& result) {
    result.success = false;

    // 源头保底：确保UI端一定能拿到这帧画面
    // 如果调用者已经设置了result.frame，则保留；否则使用传入的frame
    if (result.frame.empty()) {
        if (!frame.empty()) {
            result.frame = frame.clone();
        } else {
            // 如果frame为空，创建保底黑底图像
            result.frame = cv::Mat::zeros(Config::INPUT_HEIGHT, Config::INPUT_WIDTH, CV_8UC3);
            cv::putText(result.frame, "Camera: No Frame",
                       cv::Point(50, Config::INPUT_HEIGHT / 2),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }
    }


    // 获取检测器
    Detector* detector = nullptr;
    if (config.type == CAMERA_DIGIT) {
        detector = m_digitDetector.get();
    } else {
        detector = m_beanDetector.get();
    }

    // 处理逻辑
    if (config.type == CAMERA_DIGIT && detector) {
        // 数字相机：使用轮廓处理器
        // 这里需要实现轮廓处理逻辑
        // 临时占位：直接使用检测器
        std::vector<Detection> detections;
        detector->detect(result.frame, detections);

        if (!detections.empty()) {
            // 绘制检测框
            detector->drawDetections(result.frame, detections);

            // 使用第一个检测结果
            const auto& det = detections[0];
            result.classIds.push_back(det.classId);
            result.confidences.push_back(det.confidence);
            result.centers.push_back(det.center);

            // 计算偏差
            result.dx = det.center.x - Config::SCREEN_CENTER_X;
            result.dy = det.center.y - Config::SCREEN_CENTER_Y;

            result.success = true;
        }
    } else if (config.type == CAMERA_BEAN && detector) {
        // 豆子相机：直接检测
        std::vector<Detection> detections;
        detector->detect(result.frame, detections);

        if (!detections.empty()) {
            // 绘制检测框
            detector->drawDetections(result.frame, detections);

            // 使用第一个检测结果
            const auto& det = detections[0];
            result.classIds.push_back(det.classId);
            result.confidences.push_back(det.confidence);
            result.centers.push_back(det.center);

            // 计算偏差
            result.dx = det.center.x - Config::SCREEN_CENTER_X;
            result.dy = det.center.y - Config::SCREEN_CENTER_Y;

            result.success = true;
        }
    } else {
        // 无检测器，跳过处理
        result.success = false;
        // 不提前返回，继续绘制调试信息
    }

    // 发送数据
    if (result.success && m_transmitter) {
        DataPacket packet;
        packet.sourceId = config.sourceId;
        packet.classId = result.classIds.empty() ? 0 : result.classIds[0];
        packet.dx = static_cast<int16_t>(result.dx);
        packet.dy = static_cast<int16_t>(result.dy);

        m_transmitter->sendPacket(packet);
    }


    return result.success;
}

// 获取相机处理结果
bool CameraManager::getResult(CameraType type, CameraResult& result) {
    if (type == CAMERA_DIGIT) {
        std::lock_guard<std::mutex> lock(m_digitMutex);
        if (!m_digitConfig.enabled) {
            return false;
        }
        result = m_digitResult;
        return true;
    } else {
        std::lock_guard<std::mutex> lock(m_beanMutex);
        if (!m_beanConfig.enabled) {
            return false;
        }
        result = m_beanResult;
        return true;
    }
}

// 设置调试模式
void CameraManager::setDebugMode(bool digitEnabled, bool beanEnabled) {
    m_digitConfig.enabled = digitEnabled;
    m_beanConfig.enabled = beanEnabled;

    std::cout << "调试模式设置：" << std::endl;
    std::cout << "  数字相机: " << (digitEnabled ? "启用" : "禁用") << std::endl;
    std::cout << "  豆子相机: " << (beanEnabled ? "启用" : "禁用") << std::endl;
}

// 检查相机是否运行正常
bool CameraManager::isCameraHealthy(CameraType type) const {
    return type == CAMERA_DIGIT ? m_digitHealthy.load() : m_beanHealthy.load();
}

// 获取相机状态信息
std::string CameraManager::getCameraStatus(CameraType type) const {
    return type == CAMERA_DIGIT ? m_digitStatus : m_beanStatus;
}