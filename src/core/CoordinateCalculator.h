#ifndef COORDINATE_CALCULATOR_H
#define COORDINATE_CALCULATOR_H

#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <atomic>

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
 * 坐标计算与防抖状态
 *负责将ROI内的坐标映射回原图，计算中心偏差，
 */
class CoordinateCalculator {
public:
    CoordinateCalculator();
    ~CoordinateCalculator() = default;

    /**
     * 设置图像尺寸（用于中心点计算）
     *  图像宽度
     *  图像高度
     */
    void setImageSize(int width, int height);

    /**
     * 设置防抖参数
     *  需要连续稳定的帧数
     *  最大允许偏差（像素）
     */
    void setStabilizationParams(int stableFrames = 5, float maxDeviation = 10.0f);

    /**
     * 更新目标状态
     *  检测到的类别ID
     *  ROI内的中心点坐标
     * ROI在原图中的位置
     *  置信度
     * 目标状态已更新，false 目标无效或未稳定
     */
    bool updateTarget(int classId, const cv::Point2f& centerInRoi,
                      const cv::Rect& roiRect, float confidence = 1.0f);

    /**
     * 获取当前稳定目标状态
     * 目标状态结构体
     */
    TargetState getStableTarget() const;

    /**
     * 获取当前原始目标状态（未防抖）
     *  目标状态结构体
     */
    TargetState getCurrentTarget() const;

    /**
     * 重置状态机
     */
    void reset();

    /**
     * 在图像上绘制视觉反馈
     *  目标图像
     *是否绘制十字准星
     * 是否绘制文本信息
     */
    void drawVisualFeedback(cv::Mat& frame, bool drawCrosshair = true, bool drawText = true) const;

    /**
     * 检查目标是否稳定
     * true 目标已稳定，false 目标未稳定
     */
    bool isTargetStable() const;

private:
    /**
     * 坐标映射：将ROI内坐标转换为原图坐标
     * ROI内坐标
     * ROI矩形
     *  原图坐标
     */
    cv::Point2f mapToOriginal(const cv::Point2f& pointInRoi, const cv::Rect& roiRect) const;

    /**
     * 计算中心偏差
     * 目标中心点
     * 偏差值（dx, dy）
     */
    std::pair<int, int> calculateDeviation(const cv::Point2f& center) const;

    /**
     * 防抖状态机更新
     *  新检测到的目标状态
     * true 状态已更新并稳定，false 状态未稳定
     */
    bool updateStabilization(const TargetState& newState);

    /**
     * 检查状态一致性
     * true 最近几帧状态一致，false 状态不一致
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