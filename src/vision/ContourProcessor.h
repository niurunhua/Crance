#ifndef CONTOUR_PROCESSOR_H
#define CONTOUR_PROCESSOR_H

#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 前向声明
class Detector;

/**
 * @brief 轮廓处理结果结构体
 */
struct ContourResult {
    bool success;                       // 处理是否成功
    std::vector<cv::Rect> contours;     // 找到的轮廓（最多5个）
    std::vector<cv::Rect> rois;         // 对应的ROI区域（已防越界处理）
    cv::Mat debugImage;                 // 调试图像（可选）
    std::string errorMsg;               // 错误信息（如果success为false）

    ContourResult() : success(false) {}
};

/**
 * @brief 轮廓防爆与ROI局部推理处理器
 *
 * 负责预处理图像、寻找白纸轮廓、防爆处理（轮廓过多时跳过）、
 * ROI提取和防越界裁剪，最后调用YOLO进行局部识别。
 */
class ContourProcessor {
public:
    ContourProcessor();
    ~ContourProcessor() = default;

    /**
     * @brief 设置参数
     * @param threshold 二值化阈值
     * @param minArea 最小轮廓面积
     * @param roiMargin ROI外扩边距
     * @param maxContours 最大轮廓数（防爆阈值）
     */
    void setParams(int threshold, int minArea, int roiMargin = 20, int maxContours = 100);

    /**
     * @brief 处理单帧图像，提取ROI并调用检测器
     * @param frame 输入图像
     * @param detector 检测器指针
     * @param result 输出结果
     * @return true 处理成功，false 处理失败（跳过此帧）
     */
    bool process(const cv::Mat& frame, Detector* detector, ContourResult& result);

    /**
     * @brief 仅提取轮廓（不调用检测器）
     * @param frame 输入图像
     * @param result 输出结果
     * @return true 提取成功，false 提取失败
     */
    bool extractContours(const cv::Mat& frame, ContourResult& result);

    /**
     * @brief 在图像上绘制轮廓和ROI
     * @param frame 目标图像
     * @param result 轮廓处理结果
     * @param drawContours 是否绘制轮廓
     * @param drawRois 是否绘制ROI矩形
     */
    static void drawResults(cv::Mat& frame, const ContourResult& result,
                           bool drawContours = true, bool drawRois = true);

private:
    /**
     * @brief 预处理图像（灰度化、二值化）
     * @param frame 输入图像
     * @param binary 输出二值图像
     */
    void preprocess(const cv::Mat& frame, cv::Mat& binary);

    /**
     * @brief 寻找白纸轮廓
     * @param binary 二值图像
     * @param contours 输出轮廓
     * @return 找到的轮廓数量
     */
    int findPaperContours(const cv::Mat& binary, std::vector<std::vector<cv::Point>>& contours);

    /**
     * @brief 轮廓防爆检查
     * @param contourCount 轮廓数量
     * @return true 通过检查，false 轮廓过多需要跳过
     */
    bool contourExplosionCheck(int contourCount) const;

    /**
     * @brief 按面积筛选轮廓
     * @param contours 输入轮廓
     * @param filteredRects 输出矩形（最多5个）
     */
    void filterContoursByArea(const std::vector<std::vector<cv::Point>>& contours,
                             std::vector<cv::Rect>& filteredRects) const;

    /**
     * @brief 生成ROI区域（外扩并防越界）
     * @param rect 原始矩形
     * @param imageSize 图像尺寸
     * @return 防越界后的ROI
     */
    cv::Rect generateSafeROI(const cv::Rect& rect, const cv::Size& imageSize) const;

private:
    // 处理参数（使用Config中的静态值）
    int m_threshold;
    int m_minArea;
    int m_roiMargin;
    int m_maxContours;  // 防爆阈值（轮廓数超过此值则跳过）
};

#endif // CONTOUR_PROCESSOR_H