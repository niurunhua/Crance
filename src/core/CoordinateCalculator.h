#ifndef COORDINATE_CALCULATOR_H
#define COORDINATE_CALCULATOR_H

#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <atomic>

/**
 * @brief 目标状态结构体
 */
struct TargetState {
    int classId;                // 类别ID
    cv::Point2f center;         // 中心点坐标
    float confidence;           // 置信度
    int dx;                     // X方向偏差
    int dy;                     // Y方向偏差
    bool valid;                 // 是否有效

    TargetState() : classId(-1), center(0, 0), confidence(0.0f), dx(0), dy(0), valid(false) {}
};

/**
 * @brief 坐标计算与防抖状态机
 *
 * 负责将ROI内的坐标映射回原图，计算中心偏差，
 * 实现"连续5帧一致"的防抖逻辑，并在原图上绘制视觉反馈。
 */
class CoordinateCalculator {
public:
    CoordinateCalculator();
    ~CoordinateCalculator() = default;

    /**
     * @brief 设置图像尺寸（用于中心点计算）
     * @param width 图像宽度
     * @param height 图像高度
     */
    void setImageSize(int width, int height);

    /**
     * @brief 设置防抖参数
     * @param stableFrames 需要连续稳定的帧数
     * @param maxDeviation 最大允许偏差（像素）
     */
    void setStabilizationParams(int stableFrames = 5, float maxDeviation = 10.0f);

    /**
     * @brief 更新目标状态
     * @param classId 检测到的类别ID
     * @param centerInRoi ROI内的中心点坐标
     * @param roiRect ROI在原图中的位置
     * @param confidence 置信度
     * @return true 目标状态已更新，false 目标无效或未稳定
     */
    bool updateTarget(int classId, const cv::Point2f& centerInRoi,
                      const cv::Rect& roiRect, float confidence = 1.0f);

    /**
     * @brief 获取当前稳定目标状态
     * @return 目标状态结构体
     */
    TargetState getStableTarget() const;

    /**
     * @brief 获取当前原始目标状态（未防抖）
     * @return 目标状态结构体
     */
    TargetState getCurrentTarget() const;

    /**
     * @brief 重置状态机（目标丢失时调用）
     */
    void reset();

    /**
     * @brief 在图像上绘制视觉反馈
     * @param frame 目标图像
     * @param drawCrosshair 是否绘制十字准星
     * @param drawText 是否绘制文本信息
     */
    void drawVisualFeedback(cv::Mat& frame, bool drawCrosshair = true, bool drawText = true) const;

    /**
     * @brief 检查目标是否稳定
     * @return true 目标已稳定，false 目标未稳定
     */
    bool isTargetStable() const;

private:
    /**
     * @brief 坐标映射：将ROI内坐标转换为原图坐标
     * @param pointInRoi ROI内坐标
     * @param roiRect ROI矩形
     * @return 原图坐标
     */
    cv::Point2f mapToOriginal(const cv::Point2f& pointInRoi, const cv::Rect& roiRect) const;

    /**
     * @brief 计算中心偏差
     * @param center 目标中心点
     * @return 偏差值（dx, dy）
     */
    std::pair<int, int> calculateDeviation(const cv::Point2f& center) const;

    /**
     * @brief 防抖状态机更新
     * @param newState 新检测到的目标状态
     * @return true 状态已更新并稳定，false 状态未稳定
     */
    bool updateStabilization(const TargetState& newState);

    /**
     * @brief 检查状态一致性
     * @return true 最近几帧状态一致，false 状态不一致
     */
    bool checkConsistency() const;

private:
    // 图像尺寸
    int m_imageWidth;
    int m_imageHeight;
    cv::Point2f m_imageCenter;

    // 防抖参数
    int m_stableFrames;          // 需要连续稳定的帧数
    float m_maxDeviation;        // 最大允许偏差

    // 状态队列（用于防抖）
    std::deque<TargetState> m_stateHistory;
    TargetState m_currentState;   // 当前检测到的状态
    TargetState m_stableState;    // 稳定后的状态

    // 统计信息
    int m_consecutiveStableFrames;
    mutable std::mutex m_mutex;
};

#endif // COORDINATE_CALCULATOR_H