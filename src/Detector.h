#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Point2f center; // center of box
};

class Detector {
public:
    Detector(const std::string& modelPath,
             const std::string& classesFile = "",
             int netWidth = 640,
             int netHeight = 640,
             float confThreshold = 0.5f,
             float nmsThreshold = 0.4f);
    ~Detector() = default;

    /**
     * @brief Initialize the detector (load model, classes, etc.)
     * @return true if successful.
     */
    bool init();

    /**
     * @brief Perform detection on a single frame.
     * @param frame Input frame (will be resized to network dimensions).
     * @param detections Output vector of detections.
     */
    void detect(cv::Mat& frame, std::vector<Detection>& detections);

    /**
     * @brief Draw detection results on the frame.
     */
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

    /**
     * @brief Get class names.
     */
    std::vector<std::string> getClassNames() const { return m_classNames; }

private:
    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    void postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, std::vector<Detection>& detections);
    std::vector<cv::Rect> findWhiteRegions(const cv::Mat& frame);

    std::string m_modelPath;
    std::string m_classesFile;
    int m_netWidth;
    int m_netHeight;
    float m_confThreshold;
    float m_nmsThreshold;
    cv::dnn::Net m_net;
    std::vector<std::string> m_classNames;
    std::vector<std::string> m_outputNames;
};

#endif // DETECTOR_H