#include "Detector.h"
#include "../core/Config.h"
#include <fstream>
#include <iostream>
#include <chrono>

Detector::Detector(const std::string& modelPath,
    const std::string& classesFile,
    int netWidth, int netHeight,
    float confThreshold, float nmsThreshold,
    bool useWhiteRegionDetection)
    : m_modelPath(modelPath), m_classesFile(classesFile),
      m_netWidth(netWidth), m_netHeight(netHeight),
      m_confThreshold(confThreshold), m_nmsThreshold(nmsThreshold),
      m_threshold(Config::THRESHOLD), m_minArea(Config::MIN_AREA),
      m_useWhiteRegionDetection(useWhiteRegionDetection) {}

bool Detector::init() {
    try {
        std::cout << "Loading model: " << m_modelPath << std::endl;

        // 加载ONNX模型
        m_model = m_core.read_model(m_modelPath);
        auto inputs = m_model->inputs();
        auto outputs = m_model->outputs();

        if (inputs.empty() || outputs.empty()) {
            std::cerr << "Model has no inputs or outputs!" << std::endl;
            return false;
        }

        m_input_name = inputs[0].get_any_name();
        m_output_name = outputs[0].get_any_name();

        // 设置输入形状
        ov::Shape input_shape = {1, 3, (size_t)m_netHeight, (size_t)m_netWidth};
        m_model->reshape({{m_input_name, input_shape}});

        // 编译模型 (Intel CPU使用oneDNN优化)
        m_compiled_model = m_core.compile_model(m_model, "CPU");
        m_infer_request = m_compiled_model.create_infer_request();

        std::cout << "OpenVINO model loaded on CPU" << std::endl;

        // 加载类别名称
        if (!m_classesFile.empty()) {
            std::ifstream file(m_classesFile);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    m_classNames.push_back(line);
                }
            }
        }

        if (m_classNames.empty()) {
            for (int i = 0; i < Config::NUM_CLASSES; ++i) {
                m_classNames.push_back("class_" + std::to_string(i));
            }
        }

        std::cout << "Detector initialized: " << m_classNames.size() << " classes" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "OpenVINO init failed: " << e.what() << std::endl;
        return false;
    }
}

// 后处理：解析模型输出
void Detector::postprocess(const cv::Mat& frame, ov::Tensor& output_tensor, std::vector<Detection>& detections) {
    detections.clear();

    float* data = output_tensor.data<float>();
    auto shape = output_tensor.get_shape();
    int num_attributes = shape[1];  // 4 + 类别数
    int num_anchors = shape[2];     // 锚点数 (如8400)
    int num_classes = num_attributes - 4;

    // 坐标缩放比例
    float scaleX = (float)frame.cols / m_netWidth;
    float scaleY = (float)frame.rows / m_netHeight;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // 遍历所有锚点
    for (int i = 0; i < num_anchors; ++i) {
        float x = data[0 * num_anchors + i];
        float y = data[1 * num_anchors + i];
        float w = data[2 * num_anchors + i];
        float h = data[3 * num_anchors + i];

        // 找最大置信度类别
        int classId = -1;
        float maxConf = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float conf = data[(4 + c) * num_anchors + i];
            if (conf > maxConf) {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf >= m_confThreshold) {
            int left = (int)((x - w / 2) * scaleX);
            int top = (int)((y - h / 2) * scaleY);
            int width = (int)(w * scaleX);
            int height = (int)(h * scaleY);

            cv::Rect box(left, top, width, height);
            box = box & cv::Rect(0, 0, frame.cols, frame.rows);  // 裁剪到图像边界

            if (box.width > 2 && box.height > 2) {
                classIds.push_back(classId);
                confidences.push_back(maxConf);
                boxes.push_back(box);
            }
        }
    }

    // 非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

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

// 查找白色纸张区域 (用于数字检测)
std::vector<cv::Rect> Detector::findWhiteRegions(const cv::Mat& frame) {
    std::vector<cv::Rect> whiteRects;
    if (frame.empty()) return whiteRects;

    cv::Mat gray, binary;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, m_threshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < m_minArea) continue;

        cv::Rect rect = cv::boundingRect(contour);
        float aspectRatio = (float)rect.width / rect.height;
        if (aspectRatio < 0.7f || aspectRatio > 1.3f) continue;  // 过滤非正方形

        // 添加边距
        int margin = 15;
        cv::Rect expanded(
            rect.x - margin, rect.y - margin,
            rect.width + 2 * margin, rect.height + 2 * margin
        );
        expanded = expanded & cv::Rect(0, 0, frame.cols, frame.rows);
        whiteRects.push_back(expanded);
    }
    return whiteRects;
}

void Detector::detect(cv::Mat& frame, std::vector<Detection>& detections) {
    detections.clear();

    if (!m_useWhiteRegionDetection) {
        // 直接全图检测
        detectROI(frame, cv::Rect(0, 0, frame.cols, frame.rows), detections);
        return;
    }

    // 白纸区域检测
    std::vector<cv::Rect> whiteRects = findWhiteRegions(frame);
    if (whiteRects.empty()) return;

    for (const cv::Rect& roiRect : whiteRects) {
        if (roiRect.width <= 0 || roiRect.height <= 0) continue;

        cv::Mat roiImg = frame(roiRect);
        std::vector<Detection> roiDetections;
        detectROI(roiImg, cv::Rect(0, 0, roiImg.cols, roiImg.rows), roiDetections);

        // 坐标映射回原图
        for (auto& det : roiDetections) {
            det.box.x += roiRect.x;
            det.box.y += roiRect.y;
            det.center.x += roiRect.x;
            det.center.y += roiRect.y;
            detections.push_back(det);
        }
    }
}

void Detector::detectROI(const cv::Mat& frame, const cv::Rect& roi, std::vector<Detection>& detections) {
    detections.clear();

    if (roi.width <= 0 || roi.height <= 0 || roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > frame.cols || roi.y + roi.height > frame.rows) {
        return;
    }

    cv::Mat roiImg = frame(roi);

    try {
        // 预处理：缩放并归一化
        cv::Mat blob;
        cv::resize(roiImg, blob, cv::Size(m_netWidth, m_netHeight));
        blob.convertTo(blob, CV_32F, 1.0 / 255.0);

        // 填充输入张量 (NCHW格式)
        auto input_tensor = m_infer_request.get_input_tensor();
        float* input_data = input_tensor.data<float>();
        std::vector<cv::Mat> channels(3);
        cv::split(blob, channels);

        int channel_size = m_netWidth * m_netHeight;
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data + c * channel_size, channels[c].data, channel_size * sizeof(float));
        }

        // 推理
        auto start = std::chrono::high_resolution_clock::now();
        m_infer_request.infer();
        auto end = std::chrono::high_resolution_clock::now();
        m_lastInferenceTime = std::chrono::duration<double, std::milli>(end - start).count();

        // 后处理
        auto output_tensor = m_infer_request.get_output_tensor();
        postprocess(roiImg, output_tensor, detections);

        // 坐标映射回原图
        for (auto& det : detections) {
            det.box.x += roi.x;
            det.box.y += roi.y;
            det.center.x += roi.x;
            det.center.y += roi.y;
        }

    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}

void Detector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        std::string className = (det.classId >= 0 && det.classId < (int)m_classNames.size())
            ? m_classNames[det.classId] : "Unknown";
        std::string label = className + " " + std::to_string(det.confidence).substr(0, 4);

        // 绘制标签背景
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
