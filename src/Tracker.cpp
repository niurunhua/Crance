#include "Tracker.h"
#include "Config.h"
#include <cmath>
#include <limits>

Tracker::Tracker(float filterAlpha, int lostBufferFrames)
    : m_filterAlpha(filterAlpha),
      m_lostBufferFrames(lostBufferFrames),
      m_lostCounter(0) {
    reset();
}

void Tracker::reset() {
    m_tracked.classId = -1;
    m_tracked.filteredCenter = cv::Point2f(0, 0);
    m_tracked.rawCenter = cv::Point2f(0, 0);
    m_tracked.confidence = 0.0f;
    m_tracked.lostCounter = 0;
    m_lostCounter = 0;
    m_frameSize = cv::Size(0, 0);
}

bool Tracker::selectPrimaryTarget(const std::vector<Detection>& detections, Detection& primary) {
    if (detections.empty()) {
        return false;
    }
    // Choose detection closest to screen center
    float minDist = std::numeric_limits<float>::max();
    int bestIdx = -1;
    const cv::Point2f screenCenter(m_frameSize.width / 2.0f, m_frameSize.height / 2.0f);
    for (size_t i = 0; i < detections.size(); ++i) {
        float dist = cv::norm(detections[i].center - screenCenter);
        if (dist < minDist) {
            minDist = dist;
            bestIdx = i;
        }
    }
    if (bestIdx >= 0) {
        primary = detections[bestIdx];
        return true;
    }
    return false;
}

bool Tracker::update(const std::vector<Detection>& detections, const cv::Size& frameSize) {
    m_frameSize = frameSize;

    Detection primary;
    bool detected = selectPrimaryTarget(detections, primary);

    if (detected) {
        // Reset lost counter
        m_lostCounter = 0;
        m_tracked.lostCounter = 0;
        m_tracked.classId = primary.classId;
        m_tracked.confidence = primary.confidence;
        m_tracked.rawCenter = primary.center;

        // Apply low-pass filter to center
        if (m_tracked.filteredCenter.x == 0 && m_tracked.filteredCenter.y == 0) {
            // First detection, initialize filter
            m_tracked.filteredCenter = primary.center;
        } else {
            m_tracked.filteredCenter.x = m_filterAlpha * primary.center.x + (1 - m_filterAlpha) * m_tracked.filteredCenter.x;
            m_tracked.filteredCenter.y = m_filterAlpha * primary.center.y + (1 - m_filterAlpha) * m_tracked.filteredCenter.y;
        }
        return true;
    } else {
        // No detection, increment lost counter
        m_lostCounter++;
        m_tracked.lostCounter = m_lostCounter;
        // If within buffer frames, still consider as tracked (using last filtered center)
        if (m_lostCounter <= m_lostBufferFrames) {
            return true;
        } else {
            // Lost beyond buffer, reset
            reset();
            return false;
        }
    }
}

void Tracker::getOffset(int& dx, int& dy) const {
    if (m_tracked.classId < 0 || m_lostCounter > m_lostBufferFrames) {
        dx = 0;
        dy = 0;
        return;
    }
    const cv::Point2f screenCenter(m_frameSize.width / 2.0f, m_frameSize.height / 2.0f);
    dx = static_cast<int>(m_tracked.filteredCenter.x - screenCenter.x);
    dy = static_cast<int>(m_tracked.filteredCenter.y - screenCenter.y);
}