#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "Detector.h"

struct TrackedObject {
    int classId;
    cv::Point2f filteredCenter; // 低通滤波后的中心点
    cv::Point2f rawCenter;
    float confidence;
    int lostCounter; // 自上次检测以来的帧数
};

class Tracker {
public:
    Tracker(float filterAlpha = 0.3f, int lostBufferFrames = 5);
    ~Tracker() = default;

    /**
     * @brief Update tracker with new detections.
     * @param detections List of detections from current frame.
     * @param frameSize Size of the frame (used for offset calculation). 
     * @return true if a target is being tracked (including buffered).
     */
    bool update(const std::vector<Detection>& detections, const cv::Size& frameSize);

    /**
     * @brief Get the currently tracked object (may be buffered).
     */
    TrackedObject getTrackedObject() const { return m_tracked; }

    /**
     * @brief Get offset from screen center.
     * @param dx Output X offset (positive if target is right of center).
     * @param dy Output Y offset (positive if target is below center).
     */
    void getOffset(int& dx, int& dy) const;

    /**
     * @brief Check if target is currently detected (not just buffered).
     */
    bool isDetected() const { return m_lostCounter == 0; }

    /**
     * @brief Reset tracker state.
     */
    void reset();

private:
    TrackedObject m_tracked;
    cv::Size m_frameSize;
    float m_filterAlpha;
    int m_lostBufferFrames;
    int m_lostCounter; // 自上次检测以来的帧数

    // 从检测中选择主要目标（离中心最近）
    bool selectPrimaryTarget(const std::vector<Detection>& detections, Detection& primary);
};

#endif // TRACKER_H