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

CameraManager::CameraManager()
    : m_running(false), m_digitHealthy(false), m_beanHealthy(false) {}

CameraManager::~CameraManager() {
    stop();
}

bool CameraManager::init(const CameraConfig& digitConfig, const CameraConfig& beanConfig) {
    m_digitConfig = digitConfig;
    m_beanConfig = beanConfig;

    std::cout << "初始化相机管理器..." << std::endl;

    if (initCamera(m_digitConfig)) {
        std::cout << "数字相机初始化成功" << std::endl;
        m_digitStatus = "检测器就绪";
    } else {
        std::cout << "数字相机初始化失败" << std::endl;
        m_digitHealthy = false;
        m_digitStatus = "故障";
        m_digitConfig.enabled = false;
    }

    if (initCamera(m_beanConfig)) {
        std::cout << "豆子相机初始化成功" << std::endl;
        m_beanStatus = "检测器就绪";
    } else {
        std::cout << "豆子相机初始化失败" << std::endl;
        m_beanHealthy = false;
        m_beanStatus = "故障";
        m_beanConfig.enabled = false;
    }

    m_transmitter = std::make_shared<DataTransmitter>();
    if (!m_transmitter->initSerial(Config::SERIAL_PORT, Config::SERIAL_BAUD)) {
        std::cout << "串口初始化失败，将使用终端输出" << std::endl;
    }

    if (Config::AUTO_LABEL_ENABLED) {
        m_autoLabeler = std::make_unique<AutoLabeler>(
            Config::AUTO_LABEL_OUTPUT_DIR, Config::AUTO_LABEL_CONFIDENCE);
        std::cout << "自动标注已启用: " << Config::AUTO_LABEL_OUTPUT_DIR << std::endl;
    }

    if (!m_digitConfig.enabled && !m_beanConfig.enabled) {
        std::cerr << "错误：所有相机都不可用" << std::endl;
        return false;
    }

    std::cout << "相机管理器初始化完成" << std::endl;
    return true;
}

bool CameraManager::initCamera(CameraConfig& config) {
    std::cout << "初始化相机: " << (config.type == CAMERA_DIGIT ? "数字" : "豆子") << std::endl;

    std::unique_ptr<Detector> detector;
    if (!config.modelPath.empty()) {
        detector = std::make_unique<Detector>(
            config.modelPath, config.classesFile,
            Config::NETWORK_WIDTH, Config::NETWORK_HEIGHT,
            Config::CONFIDENCE_THRESHOLD, Config::NMS_THRESHOLD, false);

        if (!detector->init()) {
            std::cerr << "检测器初始化失败: " << config.modelPath << std::endl;
            return false;
        }
        std::cout << "检测器初始化成功" << std::endl;
    }

    if (config.type == CAMERA_DIGIT) {
        m_digitDetector = std::move(detector);
    } else {
        m_beanDetector = std::move(detector);
    }

    return true;
}

void CameraManager::start() {
    if (m_running) {
        std::cout << "相机管理器已经在运行" << std::endl;
        return;
    }

    m_running = true;

    if (m_digitConfig.enabled) {
        m_digitThread = std::thread(&CameraManager::cameraThreadFunc, this, m_digitConfig);
        std::cout << "数字相机线程已启动" << std::endl;
    }

    if (m_beanConfig.enabled) {
        m_beanThread = std::thread(&CameraManager::cameraThreadFunc, this, m_beanConfig);
        std::cout << "豆子相机线程已启动" << std::endl;
    }

    std::cout << "所有相机线程已启动" << std::endl;
}

void CameraManager::stop() {
    if (!m_running) return;

    std::cout << "正在停止相机管理器..." << std::endl;
    m_running = false;

    if (m_digitThread.joinable()) {
        m_digitThread.join();
        std::cout << "数字相机线程已停止" << std::endl;
    }

    if (m_beanThread.joinable()) {
        m_beanThread.join();
        std::cout << "豆子相机线程已停止" << std::endl;
    }

    if (m_transmitter) {
        m_transmitter->closeSerial();
    }

    std::cout << "相机管理器已停止" << std::endl;
}

void CameraManager::cameraThreadFunc(CameraConfig config) {
    // 错峰启动：豆子相机延迟2秒
    if (config.type == CAMERA_BEAN) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    cv::VideoCapture cap;

    std::cout << "打开相机: " << config.cameraIndex << std::endl;

    try {
        cap.open(config.cameraIndex);
        if (!cap.isOpened()) {
            std::cout << "相机打开失败: " << config.cameraIndex << std::endl;
            if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
            if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, Config::INPUT_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::INPUT_HEIGHT);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        // 暖机
        cv::Mat testFrame;
        bool foundValidFrame = false;
        for (int attempt = 0; attempt < 30; ++attempt) {
            if (cap.read(testFrame) && !testFrame.empty()) {
                std::cout << "相机 " << config.cameraIndex << " 就绪: "
                          << testFrame.cols << "x" << testFrame.rows << std::endl;
                foundValidFrame = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        if (!foundValidFrame) {
            std::cout << "相机 " << config.cameraIndex << " 无有效帧" << std::endl;
            cap.release();
            if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
            if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
            return;
        }

        if (config.type == CAMERA_DIGIT) m_digitHealthy.store(true);
        if (config.type == CAMERA_BEAN) m_beanHealthy.store(true);

    } catch (...) {
        std::cout << "相机 " << config.cameraIndex << " 初始化异常" << std::endl;
        if (cap.isOpened()) cap.release();
        if (config.type == CAMERA_DIGIT) m_digitHealthy.store(false);
        if (config.type == CAMERA_BEAN) m_beanHealthy.store(false);
        return;
    }

    Detector* detector = (config.type == CAMERA_DIGIT) ? m_digitDetector.get() : m_beanDetector.get();

    std::unique_ptr<ContourProcessor> contourProcessor;
    std::unique_ptr<CoordinateCalculator> coordCalculator;
    if (config.type == CAMERA_DIGIT) {
        contourProcessor = std::make_unique<ContourProcessor>();
        coordCalculator = std::make_unique<CoordinateCalculator>();
        coordCalculator->setImageSize(Config::INPUT_WIDTH, Config::INPUT_HEIGHT);
    }

    int frameCounter = 0;
    cv::Mat frame, displayFrame;
    std::vector<Detection> lastDetections;
    int noDetectionFrames = 0;
    const int maxNoDetectionFrames = 5;

    auto lastFpsTime = std::chrono::steady_clock::now();
    int fpsFrameCount = 0;
    double currentFps = 0.0;

    std::cout << "相机线程 " << (config.type == CAMERA_DIGIT ? "数字" : "豆子") << " 开始运行" << std::endl;

    while (m_running) {
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        displayFrame = frame;

        // FPS计算
        fpsFrameCount++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastFpsTime);
        if (elapsed.count() >= 1000) {
            currentFps = fpsFrameCount * 1000.0 / elapsed.count();
            fpsFrameCount = 0;
            lastFpsTime = currentTime;
        }

        // 推理
        if (frameCounter % Config::INFERENCE_INTERVAL == 0 && detector) {
            std::vector<Detection> detections;
            detector->detect(displayFrame, detections);
            if (!detections.empty()) {
                lastDetections = detections;
                noDetectionFrames = 0;

                if (config.type == CAMERA_BEAN && m_autoLabeler) {
                    int saved = m_autoLabeler->process(displayFrame, detections);
                    if (saved > 0) {
                        std::cout << "[自动标注] 保存 " << saved << " 个检测" << std::endl;
                    }
                }
            } else {
                noDetectionFrames++;
                if (noDetectionFrames >= maxNoDetectionFrames / 10) {
                    lastDetections.clear();
                }
            }
        }

        // 绘制检测结果
        if (!lastDetections.empty()) {
            detector->drawDetections(displayFrame, lastDetections);
            const auto& det = lastDetections[0];
            if (m_transmitter) {
                DataPacket packet;
                packet.sourceId = config.sourceId;
                packet.classId = det.classId;
                packet.dx = (int16_t)(det.center.x - Config::SCREEN_CENTER_X);
                packet.dy = (int16_t)(det.center.y - Config::SCREEN_CENTER_Y);
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
    std::cout << "相机线程 " << (config.type == CAMERA_DIGIT ? "数字" : "豆子") << " 已退出" << std::endl;
}

bool CameraManager::getResult(CameraType type, CameraResult& result) {
    if (type == CAMERA_DIGIT) {
        std::lock_guard<std::mutex> lock(m_digitMutex);
        if (!m_digitConfig.enabled) return false;
        result = m_digitResult;
        return true;
    } else {
        std::lock_guard<std::mutex> lock(m_beanMutex);
        if (!m_beanConfig.enabled) return false;
        result = m_beanResult;
        return true;
    }
}

void CameraManager::setDebugMode(bool digitEnabled, bool beanEnabled) {
    m_digitConfig.enabled = digitEnabled;
    m_beanConfig.enabled = beanEnabled;
    std::cout << "调试模式 - 数字: " << (digitEnabled ? "启用" : "禁用")
              << ", 豆子: " << (beanEnabled ? "启用" : "禁用") << std::endl;
}

bool CameraManager::isCameraHealthy(CameraType type) const {
    return type == CAMERA_DIGIT ? m_digitHealthy.load() : m_beanHealthy.load();
}

std::string CameraManager::getCameraStatus(CameraType type) const {
    return type == CAMERA_DIGIT ? m_digitStatus : m_beanStatus;
}
