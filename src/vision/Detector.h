#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>

// OpenVINO headers
#include <openvino/openvino.hpp>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Point2f center;
};

class Detector {
public:
    Detector(const std::string& modelPath,
             const std::string& classesFile,
             int netWidth,
             int netHeight,
             float confThreshold,
             float nmsThreshold,
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
    std::vector<cv::Rect> findWhiteRegions(const cv::Mat& frame);

    std::string m_modelPath;
    std::string m_classesFile;
    int m_netWidth;
    int m_netHeight;
    float m_confThreshold;
    float m_nmsThreshold;
    int m_threshold;
    int m_minArea;
    bool m_useWhiteRegionDetection;

    std::vector<std::string> m_classNames;
    double m_lastInferenceTime = 0;

    // OpenVINO objects
    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_infer_request;
    std::string m_input_name;
    std::string m_output_name;
};

#endif
