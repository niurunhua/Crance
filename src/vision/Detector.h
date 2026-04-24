#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <openvino/openvino.hpp>

// 检测结果结构体
struct Detection {
    int classId;           // 类别ID
    float confidence;      // 置信度
    cv::Rect box;          // 检测框
    cv::Point2f center;    // 中心点
};

// 目标检测器 (基于OpenVINO)
class Detector {
public:
    Detector(const std::string& modelPath,
             const std::string& classesFile,
             int netWidth, int netHeight,
             float confThreshold, float nmsThreshold,
             bool useWhiteRegionDetection = true);
    ~Detector() = default;

    bool init();
    void detect(cv::Mat& frame, std::vector<Detection>& detections);
    void detectROI(const cv::Mat& frame, const cv::Rect& roi, std::vector<Detection>& detections);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

    void setThreshold(int threshold) { m_threshold = threshold; }
    void setMinArea(int minArea) { m_minArea = minArea; }
    std::vector<std::string> getClassNames() const { return m_classNames; }
    double getLastInferenceTime() const { return m_lastInferenceTime; }

private:
    void postprocess(const cv::Mat& frame, ov::Tensor& output_tensor, std::vector<Detection>& detections);
    std::vector<cv::Rect> findWhiteRegions(const cv::Mat& frame);  // 查找白色纸张区域

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
};

#endif
