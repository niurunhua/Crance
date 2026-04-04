#include "Detector.h"
#include "../core/Config.h"
#include <fstream>
#include <sstream>
#include <iostream>

Detector::Detector(const std::string& modelPath,
    const std::string& classesFile,
    int netWidth,
    int netHeight,
    float confThreshold,
    float nmsThreshold)
    : m_modelPath(modelPath),
    m_classesFile(classesFile),
    m_netWidth(netWidth),
    m_netHeight(netHeight),
    m_confThreshold(confThreshold),
    m_nmsThreshold(nmsThreshold),
    m_threshold(Config::THRESHOLD),
    m_minArea(Config::MIN_AREA) {}

bool Detector::init() {
    // 加载模型
    m_net = cv::dnn::readNetFromONNX(m_modelPath);
    if (m_net.empty()) {
        std::cerr << "Failed to load model from " << m_modelPath << std::endl;
        return false;
    }
    // 设置优先后端（CPU）
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 加载类别名称
    if (!m_classesFile.empty()) {
        std::ifstream file(m_classesFile);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                m_classNames.push_back(line);
            }
            file.close();
        }
    }
    // 如果未加载类别名称，创建默认名称
    if (m_classNames.empty()) {
        for (int i = 0; i < Config::NUM_CLASSES; ++i) {
            m_classNames.push_back("class_" + std::to_string(i));
        }
    }

    // 获取输出层名称（YOLOv11可能只有单个输出 \"output0\"）
    m_outputNames = m_net.getUnconnectedOutLayersNames();
    if (m_outputNames.empty()) {
        // 备用方案：假设输出名称为 \"output0\"
        m_outputNames.push_back("output0");
    }

    std::cout << "Detector initialized with " << m_classNames.size() << " classes." << std::endl;
    return true;
}

void Detector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    // 从图像创建blob，无缩放（0-255），调整到网络尺寸，无均值减法
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(m_netWidth, m_netHeight), cv::Scalar(), true, false);
}

void Detector::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, std::vector<Detection>& detections) {
    detections.clear();
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outputs) {
        cv::Mat outMat;

        // 1. 将3D张量缩减为2D矩阵
        if (output.dims == 3) {
            outMat = cv::Mat(output.size[1], output.size[2], CV_32F, output.data);
        }
        else if (output.dims == 2) {
            outMat = output;
        }
        else {
            std::cerr << "Unexpected output dimension: " << output.dims << std::endl;
            continue;
        }

        // 2. 核心修复：转置矩阵以确保安全的行读取
        if (outMat.rows < outMat.cols) {
            cv::transpose(outMat, outMat);
        }

        const int numAnchors = outMat.rows;     // 例如：8400
        const int numAttributes = outMat.cols;  // 例如：9（4个坐标 + 5个类别）
        const int numClasses = numAttributes - 4;

        if (numClasses <= 0) continue;

        // 3. 安全遍历所有预测框
        for (int i = 0; i < numAnchors; ++i) {
            const float* data = outMat.ptr<float>(i);

            // 归一化坐标
            const float x_center = data[0];
            const float y_center = data[1];
            const float width = data[2];
            const float height = data[3];

            // 找到最大置信度的类别
            int classId = -1;
            float maxConf = 0.0f;
            for (int c = 0; c < numClasses; ++c) {
                float conf = data[4 + c];
                if (conf > maxConf) {
                    maxConf = conf;
                    classId = c;
                }
            }

            if (maxConf >= m_confThreshold) {
                // 转换回真实像素坐标
                const int left = static_cast<int>((x_center - width / 2) * frame.cols);
                const int top = static_cast<int>((y_center - height / 2) * frame.rows);
                const int w = static_cast<int>(width * frame.cols);
                const int h = static_cast<int>(height * frame.rows);

                cv::Rect box(left, top, w, h);
                // 确保框在图像边界内
                box = box & cv::Rect(0, 0, frame.cols, frame.rows);

                // 仅保留有效框
                if (box.width > 2 && box.height > 2) {
                    classIds.push_back(classId);
                    confidences.push_back(maxConf);
                    boxes.push_back(box);
                }
            }
        }
    }

    // 应用非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    // 创建检测对象
    for (int idx : indices) {
        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        det.center = cv::Point2f(boxes[idx].x + boxes[idx].width / 2.0f,
            boxes[idx].y + boxes[idx].height / 2.0f);
        detections.push_back(det);
    }
}

std::vector<cv::Rect> Detector::findWhiteRegions(const cv::Mat& frame) {
    std::vector<cv::Rect> whiteRects;
    if (frame.empty()) return whiteRects;

    // 1. 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 2. 阈值提取亮白纸张
    cv::Mat binary;
    cv::threshold(gray, binary, m_threshold, 255, cv::THRESH_BINARY);

    // 3. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. 过滤轮廓
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < m_minArea) continue; // 噪声

        cv::Rect rect = cv::boundingRect(contour);
        float aspectRatio = static_cast<float>(rect.width) / rect.height;
        if (aspectRatio < 0.7f || aspectRatio > 1.3f) continue; // 非正方形

        // 添加边距（每边15像素）
        int margin = 15;
        cv::Rect expanded(
            rect.x - margin,
            rect.y - margin,
            rect.width + 2 * margin,
            rect.height + 2 * margin
        );
        // 确保在图像边界内
        expanded = expanded & cv::Rect(0, 0, frame.cols, frame.rows);
        whiteRects.push_back(expanded);
    }
    return whiteRects;
}

void Detector::detect(cv::Mat& frame, std::vector<Detection>& detections) {
    detections.clear();
    // 步骤1：查找白色纸张区域
    std::vector<cv::Rect> whiteRects = findWhiteRegions(frame);
    if (whiteRects.empty()) {
        // 未找到白色纸张，跳过YOLO推理
        return;
    }

    // 步骤2：对每个白色区域，裁剪ROI并运行YOLO
    for (const cv::Rect& roiRect : whiteRects) {
        // 确保ROI有效
        if (roiRect.width <= 0 || roiRect.height <= 0) continue;

        cv::Mat roiImg = frame(roiRect);
        cv::Mat blob;
        preprocess(roiImg, blob);
        m_net.setInput(blob);
        std::vector<cv::Mat> outputs;
        m_net.forward(outputs, m_outputNames);

        // 相对于ROI的临时检测结果
        std::vector<Detection> roiDetections;
        postprocess(roiImg, outputs, roiDetections);

        // 将坐标映射回原图
        for (auto& det : roiDetections) {
            det.box.x += roiRect.x;
            det.box.y += roiRect.y;
            det.center.x += roiRect.x;
            det.center.y += roiRect.y;
            detections.push_back(det);
        }
    }

    // 是否对不同ROI的检测结果应用非极大值抑制？
    // 由于每个ROI代表独立的纸张，跨ROI的NMS可能不需要。
    // 但如果多个ROI重叠，可以应用NMS。
    // 为简单起见，我们保留所有检测结果。
}

void Detector::detectROI(const cv::Mat& frame, const cv::Rect& roi, std::vector<Detection>& detections) {
    detections.clear();

    // 检查ROI有效性
    if (roi.width <= 0 || roi.height <= 0 || roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > frame.cols || roi.y + roi.height > frame.rows) {
        std::cerr << "警告：无效的ROI区域，跳过检测" << std::endl;
        return;
    }

    // 裁剪ROI区域
    cv::Mat roiImg = frame(roi);
    cv::Mat blob;
    preprocess(roiImg, blob);

    // 网络推理
    m_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_outputNames);

    // 后处理（ROI坐标系）
    std::vector<Detection> roiDetections;
    postprocess(roiImg, outputs, roiDetections);

    // 将坐标映射回原图
    for (auto& det : roiDetections) {
        det.box.x += roi.x;
        det.box.y += roi.y;
        det.center.x += roi.x;
        det.center.y += roi.y;
        detections.push_back(det);
    }

    // 可选：对检测结果进行NMS
    // 注意：由于只有一个ROI，可能不需要NMS
}

void Detector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 绘制矩形框
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        // 标注类别名称和置信度
        // 确保类别ID在有效范围内
        std::string className = "Unknown";
        if (det.classId >= 0 && det.classId < static_cast<int>(m_classNames.size())) {
            className = m_classNames[det.classId];
        }
        std::string label = className + " " + std::to_string(det.confidence).substr(0, 4);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(det.box.y, labelSize.height);
        cv::rectangle(frame, cv::Point(det.box.x, top - labelSize.height),
            cv::Point(det.box.x + labelSize.width, top + baseLine),
            cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x, top),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // 绘制中心点
        cv::circle(frame, det.center, 3, cv::Scalar(0, 0, 255), -1);
    }
}