#include "ContourProcessor.h"
#include "Detector.h"
#include "../core/Config.h"
#include <algorithm>
#include <iostream>

// 构造函数
ContourProcessor::ContourProcessor()
    : m_threshold(Config::THRESHOLD)
    , m_minArea(Config::MIN_AREA)
    , m_roiMargin(20)
    , m_maxContours(100) {
}

// 设置参数
void ContourProcessor::setParams(int threshold, int minArea, int roiMargin, int maxContours) {
    m_threshold = threshold;
    m_minArea = minArea;
    m_roiMargin = roiMargin;
    m_maxContours = maxContours;
}

// 处理单帧图像
bool ContourProcessor::process(const cv::Mat& frame, Detector* detector, ContourResult& result) {
    result.success = false;
    result.errorMsg.clear();

    if (frame.empty()) {
        result.errorMsg = "输入图像为空";
        return false;
    }

    // 1. 预处理图像
    cv::Mat binary;
    preprocess(frame, binary);

    // 2. 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    int contourCount = findPaperContours(binary, contours);

    // 3. 轮廓防爆检查
    if (!contourExplosionCheck(contourCount)) {
        result.errorMsg = "轮廓过多(" + std::to_string(contourCount) + ")，跳过此帧";
        return false;
    }

    // 4. 按面积筛选轮廓
    std::vector<cv::Rect> filteredRects;
    filterContoursByArea(contours, filteredRects);

    if (filteredRects.empty()) {
        result.errorMsg = "未找到符合条件的轮廓";
        return false;
    }

    // 5. 生成安全的ROI区域
    result.contours = filteredRects;
    for (const auto& rect : filteredRects) {
        cv::Rect roi = generateSafeROI(rect, frame.size());
        result.rois.push_back(roi);
    }

    // 6. 调用检测器进行局部识别
    if (detector) {
        // 这里可以遍历每个ROI进行检测
        // 目前只处理第一个ROI作为示例
        if (!result.rois.empty()) {
            cv::Rect roi = result.rois[0];
            cv::Mat roiImage = frame(roi).clone();

            std::vector<Detection> detections;
            detector->detect(roiImage, detections);

            // 将检测结果映射回原图坐标
            for (auto& det : detections) {
                det.box.x += roi.x;
                det.box.y += roi.y;
                det.center.x += roi.x;
                det.center.y += roi.y;
            }

            // 可以在这里处理检测结果...
        }
    }

    // 7. 创建调试图像
    result.debugImage = frame.clone();
    drawResults(result.debugImage, result, true, true);

    result.success = true;
    return true;
}

// 仅提取轮廓
bool ContourProcessor::extractContours(const cv::Mat& frame, ContourResult& result) {
    result.success = false;

    if (frame.empty()) {
        return false;
    }

    // 预处理
    cv::Mat binary;
    preprocess(frame, binary);

    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    int contourCount = findPaperContours(binary, contours);

    // 轮廓防爆检查
    if (!contourExplosionCheck(contourCount)) {
        return false;
    }

    // 按面积筛选
    std::vector<cv::Rect> filteredRects;
    filterContoursByArea(contours, filteredRects);

    if (filteredRects.empty()) {
        return false;
    }

    // 生成安全ROI
    result.contours = filteredRects;
    for (const auto& rect : filteredRects) {
        cv::Rect roi = generateSafeROI(rect, frame.size());
        result.rois.push_back(roi);
    }

    result.success = true;
    return true;
}

// 预处理图像
void ContourProcessor::preprocess(const cv::Mat& frame, cv::Mat& binary) {
    // 转换为灰度图
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }

    // 二值化
    cv::threshold(gray, binary, m_threshold, 255, cv::THRESH_BINARY);

    // 可选：形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
}

// 寻找白纸轮廓
int ContourProcessor::findPaperContours(const cv::Mat& binary, std::vector<std::vector<cv::Point>>& contours) {
    // 寻找所有轮廓
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤掉面积太小的轮廓
    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= m_minArea) {
            filteredContours.push_back(contour);
        }
    }

    contours = std::move(filteredContours);
    return static_cast<int>(contours.size());
}

// 轮廓防爆检查
bool ContourProcessor::contourExplosionCheck(int contourCount) const {
    if (contourCount > m_maxContours) {
        std::cerr << "轮廓防爆: 检测到 " << contourCount
                  << " 个轮廓，超过阈值 " << m_maxContours << "，跳过此帧" << std::endl;
        return false;
    }
    return true;
}

// 按面积筛选轮廓
void ContourProcessor::filterContoursByArea(const std::vector<std::vector<cv::Point>>& contours,
                                           std::vector<cv::Rect>& filteredRects) const {
    // 计算每个轮廓的边界矩形和面积
    struct ContourInfo {
        std::vector<cv::Point> contour;
        cv::Rect rect;
        double area;
    };

    std::vector<ContourInfo> contourInfos;
    for (const auto& contour : contours) {
        ContourInfo info;
        info.contour = contour;
        info.rect = cv::boundingRect(contour);
        info.area = cv::contourArea(contour);
        contourInfos.push_back(info);
    }

    // 按面积从大到小排序
    std::sort(contourInfos.begin(), contourInfos.end(),
              [](const ContourInfo& a, const ContourInfo& b) {
                  return a.area > b.area;
              });

    // 取前5个（或更少）
    int maxCount = std::min(5, static_cast<int>(contourInfos.size()));
    for (int i = 0; i < maxCount; ++i) {
        filteredRects.push_back(contourInfos[i].rect);
    }
}

// 生成安全的ROI区域
cv::Rect ContourProcessor::generateSafeROI(const cv::Rect& rect, const cv::Size& imageSize) const {
    // 外扩边距
    int margin = m_roiMargin;
    int x = rect.x - margin;
    int y = rect.y - margin;
    int width = rect.width + 2 * margin;
    int height = rect.height + 2 * margin;

    // 防越界裁剪
    x = std::max(0, x);
    y = std::max(0, y);
    width = std::min(width, imageSize.width - x);
    height = std::min(height, imageSize.height - y);

    // 确保宽度和高度为正数
    width = std::max(1, width);
    height = std::max(1, height);

    return cv::Rect(x, y, width, height) & cv::Rect(0, 0, imageSize.width, imageSize.height);
}

// 绘制轮廓和ROI
void ContourProcessor::drawResults(cv::Mat& frame, const ContourResult& result,
                                  bool drawContours, bool drawRois) {
    if (frame.empty() || !result.success) {
        return;
    }

    // 绘制轮廓
    if (drawContours && !result.contours.empty()) {
        for (const auto& rect : result.contours) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        }
    }

    // 绘制ROI
    if (drawRois && !result.rois.empty()) {
        for (const auto& roi : result.rois) {
            cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);

            // 在ROI上标注编号
            cv::putText(frame, "ROI", cv::Point(roi.x + 5, roi.y + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }
    }

    // 绘制信息文本
    std::string info = "Contours: " + std::to_string(result.contours.size());
    cv::putText(frame, info, cv::Point(10, frame.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
}