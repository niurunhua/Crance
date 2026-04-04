#ifndef CAMERA_MANAGER_H
#define CAMERA_MANAGER_H

#include <limits>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

// 前向声明
class Detector;
class DataTransmitter;

/**
 * @brief 相机类型枚举
 */
enum CameraType {
    CAMERA_DIGIT = 0,  // 数字相机（识别数字）
    CAMERA_BEAN = 1    // 豆子相机（识别豆子）
};

/**
 * @brief 相机配置结构体
 */
struct CameraConfig {
    CameraType type;                    // 相机类型
    int cameraIndex;                    // 相机索引（0,1,...）
    std::string modelPath;              // 模型路径
    std::string classesFile;            // 类别文件路径
    bool enabled;                       // 是否启用
    int skipFrames;                     // 跳帧数（用于降帧处理，0表示不跳帧）
    int sourceId;                       // 数据源ID（0x01:数字相机, 0x02:豆子相机）
    double exposure = -4.0;             // 曝光值（负值表示快门时间更短）

    CameraConfig() : type(CAMERA_DIGIT), cameraIndex(0), enabled(true), skipFrames(0), sourceId(0x01), exposure(-4.0) {}
};

/**
 * @brief 相机处理结果结构体
 */
struct CameraResult {
    bool success;                       // 处理是否成功
    cv::Mat frame;                      // 处理后的帧（用于显示）
    std::vector<cv::Rect> contours;     // 检测到的轮廓
    std::vector<int> classIds;          // 识别到的类别ID
    std::vector<float> confidences;     // 置信度
    std::vector<cv::Point2f> centers;   // 目标中心点
    int stableClassId;                  // 稳定的类别ID（防抖后）
    int dx;                             // X方向偏差
    int dy;                             // Y方向偏差
    double fps;                         // 该相机的处理FPS
};

/**
 * @brief 双相机并行管理器
 *
 * 负责管理两个相机的并行采集、处理和显示。
 * 支持强容错：如果某个相机打开失败或模型加载失败，程序不会崩溃，只会跳过损坏分支。
 * 支持多线程：每个相机独立线程处理，抓图、推理、显示互不阻塞。
 * 支持调试模式：可以灵活切换“仅运行数字相机”、“仅运行豆子相机”或“双路并行”。
 */
class CameraManager {
public:
    CameraManager();
    ~CameraManager();

    /**
     * @brief 初始化相机管理器
     * @param digitConfig 数字相机配置
     * @param beanConfig 豆子相机配置
     * @return true 初始化成功，false 初始化失败（但可能部分相机可用）
     */
    bool init(const CameraConfig& digitConfig, const CameraConfig& beanConfig);

    /**
     * @brief 启动所有相机处理线程
     */
    void start();

    /**
     * @brief 停止所有相机处理线程
     */
    void stop();

    /**
     * @brief 获取相机处理结果（线程安全）
     * @param type 相机类型
     * @param result 输出结果
     * @return true 获取成功，false 相机未启用或未运行
     */
    bool getResult(CameraType type, CameraResult& result);

    /**
     * @brief 设置调试模式
     * @param digitEnabled 启用数字相机
     * @param beanEnabled 启用豆子相机
     */
    void setDebugMode(bool digitEnabled, bool beanEnabled);

    /**
     * @brief 检查相机是否运行正常
     * @param type 相机类型
     * @return true 相机正常运行，false 相机故障或未启用
     */
    bool isCameraHealthy(CameraType type) const;

    /**
     * @brief 获取相机状态信息
     * @param type 相机类型
     * @return 状态描述字符串
     */
    std::string getCameraStatus(CameraType type) const;

private:
    /**
     * @brief 相机处理线程函数
     * @param config 相机配置
     */
    void cameraThreadFunc(CameraConfig config);

    /**
     * @brief 处理单帧图像
     * @param config 相机配置
     * @param frame 输入帧
     * @param result 输出结果
     * @return true 处理成功，false 处理失败（跳过此帧）
     */
    bool processFrame(const CameraConfig& config, const cv::Mat& frame, CameraResult& result);

    /**
     * @brief 初始化单个相机
     * @param config 相机配置
     * @return true 初始化成功，false 初始化失败
     */
    bool initCamera(CameraConfig& config);

private:
    // 相机配置
    CameraConfig m_digitConfig;
    CameraConfig m_beanConfig;

    // 线程和同步
    std::thread m_digitThread;
    std::thread m_beanThread;
    std::atomic<bool> m_running;

    // 处理结果（使用互斥锁保护）
    CameraResult m_digitResult;
    CameraResult m_beanResult;
    mutable std::mutex m_digitMutex;
    mutable std::mutex m_beanMutex;

    // 最新帧缓存（零延迟丢帧机制）
    cv::Mat m_latestFrameDigit;
    cv::Mat m_latestFrameBean;
    mutable std::mutex m_digitFrameMutex;
    mutable std::mutex m_beanFrameMutex;

    // 组件指针
    std::unique_ptr<Detector> m_digitDetector;
    std::unique_ptr<Detector> m_beanDetector;
    std::shared_ptr<DataTransmitter> m_transmitter;

    // 状态标志
    std::atomic<bool> m_digitHealthy;
    std::atomic<bool> m_beanHealthy;
    std::string m_digitStatus;
    std::string m_beanStatus;
};

#endif // CAMERA_MANAGER_H