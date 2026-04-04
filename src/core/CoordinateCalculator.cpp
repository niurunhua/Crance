#include "CoordinateCalculator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

CoordinateCalculator::CoordinateCalculator()
    : m_imageWidth(640)
    , m_imageHeight(480)
    , m_stableFrames(5)
    , m_maxDeviation(10.0f)
    , m_consecutiveStableFrames(0) {
    m_imageCenter = cv::Point2f(m_imageWidth / 2.0f, m_imageHeight / 2.0f);
}

// 设置图像尺寸
void CoordinateCalculator::setImageSize(int width, int height) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_imageWidth = width;
    m_imageHeight = height;
    m_imageCenter = cv::Point2f(width / 2.0f, height / 2.0f);
}

// 设置防抖参数
void CoordinateCalculator::setStabilizationParams(int stableFrames, float maxDeviation) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stableFrames = stableFrames;
    m_maxDeviation = maxDeviation;
}

// 更新目标状态
bool CoordinateCalculator::updateTarget(int classId, const cv::Point2f& centerInRoi,
                                       const cv::Rect& roiRect, float confidence) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // 坐标映射
    cv::Point2f center = mapToOriginal(centerInRoi, roiRect);

    // 创建新状态
    TargetState newState;
    newState.classId = classId;
    newState.center = center;
    newState.confidence = confidence;
    newState.valid = true;

    // 计算偏差
    auto deviation = calculateDeviation(center);
    newState.dx = deviation.first;
    newState.dy = deviation.second;

    // 更新当前状态
    m_currentState = newState;

    // 更新防抖状态机
    return updateStabilization(newState);
}

// 坐标映射
cv::Point2f CoordinateCalculator::mapToOriginal(const cv::Point2f& pointInRoi, const cv::Rect& roiRect) const {
    return cv::Point2f(
        pointInRoi.x + roiRect.x,
        pointInRoi.y + roiRect.y
    );
}

// 计算中心偏差
std::pair<int, int> CoordinateCalculator::calculateDeviation(const cv::Point2f& center) const {
    int dx = static_cast<int>(center.x - m_imageCenter.x);
    int dy = static_cast<int>(center.y - m_imageCenter.y);
    return {dx, dy};
}

// 防抖状态机更新
bool CoordinateCalculator::updateStabilization(const TargetState& newState) {
    // 添加到历史队列
    m_stateHistory.push_back(newState);

    // 保持队列大小
    while (m_stateHistory.size() > static_cast<size_t>(m_stableFrames)) {
        m_stateHistory.pop_front();
    }

    // 检查一致性
    bool consistent = checkConsistency();

    if (consistent) {
        m_consecutiveStableFrames++;

        // 如果连续稳定帧数达到要求，更新稳定状态
        if (m_consecutiveStableFrames >= m_stableFrames) {
            m_stableState = newState;
            return true;
        }
    } else {
        // 不一致，重置连续计数
        m_consecutiveStableFrames = 0;
    }

    return false;
}

// 检查状态一致性
bool CoordinateCalculator::checkConsistency() const {
    if (m_stateHistory.size() < static_cast<size_t>(m_stableFrames)) {
        return false;
    }

    // 检查类别ID是否一致
    int firstClassId = m_stateHistory.front().classId;
    for (const auto& state : m_stateHistory) {
        if (state.classId != firstClassId) {
            return false;
        }
    }

    // 检查位置是否在允许偏差范围内
    cv::Point2f avgCenter(0, 0);
    for (const auto& state : m_stateHistory) {
        avgCenter += state.center;
    }
    avgCenter.x /= m_stateHistory.size();
    avgCenter.y /= m_stateHistory.size();

    for (const auto& state : m_stateHistory) {
        float dx = std::abs(state.center.x - avgCenter.x);
        float dy = std::abs(state.center.y - avgCenter.y);
        if (dx > m_maxDeviation || dy > m_maxDeviation) {
            return false;
        }
    }

    return true;
}

// 获取当前稳定目标状态
TargetState CoordinateCalculator::getStableTarget() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_stableState;
}

// 获取当前原始目标状态
TargetState CoordinateCalculator::getCurrentTarget() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentState;
}

// 重置状态机
void CoordinateCalculator::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stateHistory.clear();
    m_currentState = TargetState();
    m_stableState = TargetState();
    m_consecutiveStableFrames = 0;
}

// 在图像上绘制视觉反馈
void CoordinateCalculator::drawVisualFeedback(cv::Mat& frame, bool drawCrosshair, bool drawText) const {
    if (frame.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // 绘制十字准星
    if (drawCrosshair) {
        // 中心十字
        cv::line(frame,
                cv::Point(static_cast<int>(m_imageCenter.x), 0),
                cv::Point(static_cast<int>(m_imageCenter.x), frame.rows),
                cv::Scalar(0, 255, 255), 1);  // 黄色

        cv::line(frame,
                cv::Point(0, static_cast<int>(m_imageCenter.y)),
                cv::Point(frame.cols, static_cast<int>(m_imageCenter.y)),
                cv::Scalar(0, 255, 255), 1);

        // 如果目标有效，绘制目标十字
        if (m_currentState.valid) {
            cv::Point center(static_cast<int>(m_currentState.center.x),
                            static_cast<int>(m_currentState.center.y));

            // 目标十字（红色）
            int crossSize = 20;
            cv::line(frame,
                    cv::Point(center.x - crossSize, center.y),
                    cv::Point(center.x + crossSize, center.y),
                    cv::Scalar(0, 0, 255), 2);

            cv::line(frame,
                    cv::Point(center.x, center.y - crossSize),
                    cv::Point(center.x, center.y + crossSize),
                    cv::Scalar(0, 0, 255), 2);

            // 连接目标和中心（绿色虚线）
            cv::Point imgCenter(static_cast<int>(m_imageCenter.x),
                               static_cast<int>(m_imageCenter.y));
            cv::line(frame, imgCenter, center, cv::Scalar(0, 255, 0), 1);
        }
    }

    // 绘制文本信息
    if (drawText) {
        std::vector<std::string> texts;

        // 当前偏差
        if (m_currentState.valid) {
            texts.push_back("dx: " + std::to_string(m_currentState.dx) +
                           " dy: " + std::to_string(m_currentState.dy));

            // 类别信息
            if (m_currentState.classId >= 0) {
                texts.push_back("Class: " + std::to_string(m_currentState.classId));
            }

            // 置信度
            if (m_currentState.confidence > 0) {
                texts.push_back("Confidence: " +
                               std::to_string(static_cast<int>(m_currentState.confidence * 100)) + "%");
            }
        }

        // 稳定状态
        if (m_stableState.valid && isTargetStable()) {
            texts.push_back("Status: Stable");
            texts.push_back("Stable Class: " + std::to_string(m_stableState.classId));
        } else if (m_currentState.valid) {
            texts.push_back("Status: Detecting");
        } else {
            texts.push_back("Status: No Target");
        }

        // 历史帧数
        texts.push_back("History: " + std::to_string(m_stateHistory.size()) +
                       "/" + std::to_string(m_stableFrames));

        // 绘制文本
        int yOffset = 30;
        for (const auto& text : texts) {
            cv::putText(frame, text, cv::Point(frame.cols - 200, yOffset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            yOffset += 20;
        }
    }
}

// 检查目标是否稳定
bool CoordinateCalculator::isTargetStable() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_consecutiveStableFrames >= m_stableFrames;
}