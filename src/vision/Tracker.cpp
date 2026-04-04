#include "Tracker.h"
#include "../core/Config.h"
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
    // 选择离屏幕中心最近的检测目标
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
        // 重置丢失计数器
        m_lostCounter = 0;
        m_tracked.lostCounter = 0;
        m_tracked.classId = primary.classId;
        m_tracked.confidence = primary.confidence;
        m_tracked.rawCenter = primary.center;

        // 对中心点应用低通滤波器
        if (m_tracked.filteredCenter.x == 0 && m_tracked.filteredCenter.y == 0) {
            // 首次检测，初始化滤波器
            m_tracked.filteredCenter = primary.center;
        } else {
            m_tracked.filteredCenter.x = m_filterAlpha * primary.center.x + (1 - m_filterAlpha) * m_tracked.filteredCenter.x;
            m_tracked.filteredCenter.y = m_filterAlpha * primary.center.y + (1 - m_filterAlpha) * m_tracked.filteredCenter.y;
        }
        return true;
    } else {
        // 无检测，增加丢失计数器
        m_lostCounter++;
        m_tracked.lostCounter = m_lostCounter;
        // 在缓冲帧数内，仍视为跟踪中（使用上次滤波中心）
        if (m_lostCounter <= m_lostBufferFrames) {
            return true;
        } else {
            // 丢失超出缓冲，重置
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