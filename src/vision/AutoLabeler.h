#ifndef AUTO_LABELER_H
#define AUTO_LABELER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "Detector.h"

class AutoLabeler { 
public:
    AutoLabeler(const std::string& outputDir = "dataset",
                float confidenceThreshold = 0.85f);
    ~AutoLabeler() = default;

    /**
     * @brief Set output directory (will create if not exists).
     */
    void setOutputDir(const std::string& outputDir);

    /**
     * @brief Process a frame and save image+label if high confidence detections exist.
     * @param frame Input frame.
     * @param detections Detections from detector.
     * @return Number of saved annotations (0 if none).
     */
    int process(const cv::Mat& frame, const std::vector<Detection>& detections);

    /**
     * @brief Get total number of saved samples.
     */
    size_t getSavedCount() const { return m_savedCount; }

private:
    std::string m_outputDir;
    float m_confidenceThreshold;
    size_t m_savedCount;

    void ensureDirectoryExists(const std::string& path);
    void saveImageAndLabel(const cv::Mat& frame, const std::vector<Detection>& detections, const std::string& baseName);
};

#endif // AUTO_LABELER_H