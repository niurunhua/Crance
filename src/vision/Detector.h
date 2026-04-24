#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <functional>
#include <openvino/openvino.hpp>

// 检测结果结构体
struct Detection {
    int classId;           // 类别ID
    float confidence;      // 置信度
    cv::Rect box;          // 检测框
    cv::Point2f center;    // 中心点
};

// 目标检测器 (基于OpenVINO，支持异步推理)
class Detector {
public:
    Detector(const std::string& modelPath,
             const std::string& classesFile,
             int netWidth, int netHeight,
             float confThreshold, float nmsThreshold,
             bool useWhiteRegionDetection = true);
    ~Detector();

    bool init();

    // 同步检测
    void detect(cv::Mat& frame, std::vector<Detection>& detections);
    void detectROI(const cv::Mat& frame, const cv::Rect& roi, std::vector<Detection>& detections);

    // 异步检测
    void detectAsync(const cv::Mat& frame,
                     const cv::Rect& roi,
                     std::function<void(const std::vector<Detection>&)> callback);

    // 获取最新的检测结果
    bool getLatestDetections(std::vector<Detection>& detections);

    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

    void setThreshold(int threshold) { m_threshold = threshold; }
    void setMinArea(int minArea) { m_minArea = minArea; }
    std::vector<std::string> getClassNames() const { return m_classNames; }
    double getLastInferenceTime() const { return m_lastInferenceTime; }

private:
    void postprocess(const cv::Mat& frame, ov::Tensor& output_tensor, std::vector<Detection>& detections);
    std::vector<cv::Rect> findWhiteRegions(const cv::Mat& frame);

    // 异步推理线程函数
    void inferenceThreadFunc();

    // 模型参数
    std::string m_modelPath, m_classesFile;
    int m_netWidth, m_netHeight, m_threshold, m_minArea;
    float m_confThreshold, m_nmsThreshold;
    bool m_useWhiteRegionDetection;

    std::vector<std::string> m_classNames;
    double m_lastInferenceTime = 0;

    // OpenVINO对象
    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_infer_request;
    std::string m_input_name, m_output_name;

    // 异步推理相关
    std::atomic<bool> m_running{false};
    std::thread m_inference_thread;
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;

    // 请求队列
    struct InferenceRequest {
        cv::Mat frame;
        cv::Rect roi;
        std::function<void(const std::vector<Detection>&)> callback;
    };
    std::queue<InferenceRequest> m_request_queue;

    // 最新检测结果
    std::mutex m_result_mutex;
    std::vector<Detection> m_latest_detections;
    bool m_has_new_result{false};
};

#endif
