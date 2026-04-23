#include "Detector.h"
#include "../core/Config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

Detector::Detector(const std::string& modelPath,
    const std::string& classesFile,
    int netWidth,
    int netHeight,
    float confThreshold,
    float nmsThreshold,
    bool useWhiteRegionDetection)
    : m_modelPath(modelPath),
    m_classesFile(classesFile),
    m_netWidth(netWidth),
    m_netHeight(netHeight),
    m_confThreshold(confThreshold),
    m_nmsThreshold(nmsThreshold),
    m_threshold(Config::THRESHOLD),
    m_minArea(Config::MIN_AREA),
    m_useWhiteRegionDetection(useWhiteRegionDetection) {}

bool Detector::init() {
    try {
        std::cout << "Loading model with OpenVINO: " << m_modelPath << std::endl;

        // Load model
        m_model = m_core.read_model(m_modelPath);

        // Get input/output names
        auto inputs = m_model->inputs();
        auto outputs = m_model->outputs();

        if (inputs.empty() || outputs.empty()) {
            std::cerr << "Model has no inputs or outputs!" << std::endl;
            return false;
        }

        m_input_name = inputs[0].get_any_name();
        m_output_name = outputs[0].get_any_name();

        std::cout << "Input: " << m_input_name << ", Output: " << m_output_name << std::endl;

        // Set input shape (batch=1, channels=3, height, width)
        ov::Shape input_shape = {1, 3, (size_t)m_netHeight, (size_t)m_netWidth};
        auto input_port = inputs[0];
        m_model->reshape({{m_input_name, input_shape}});

        // Compile model for CPU (Intel CPU will use oneDNN optimization)
        m_compiled_model = m_core.compile_model(m_model, "CPU");

        // Create infer request
        m_infer_request = m_compiled_model.create_infer_request();

        std::cout << "OpenVINO model loaded successfully on CPU" << std::endl;

        // Load class names
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

        if (m_classNames.empty()) {
            for (int i = 0; i < Config::NUM_CLASSES; ++i) {
                m_classNames.push_back("class_" + std::to_string(i));
            }
        }

        std::cout << "Detector initialized with " << m_classNames.size() << " classes." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "OpenVINO init failed: " << e.what() << std::endl;
        return false;
    }
}

void Detector::postprocess(const cv::Mat& frame, ov::Tensor& output_tensor, std::vector<Detection>& detections) {
    detections.clear();

    float* data = output_tensor.data<float>();
    auto shape = output_tensor.get_shape();

    int num_attributes = shape[1];  // e.g., 4 + num_classes
    int num_anchors = shape[2];     // e.g., 8400

    int num_classes = num_attributes - 4;

    const float scaleX = static_cast<float>(frame.cols) / m_netWidth;
    const float scaleY = static_cast<float>(frame.rows) / m_netHeight;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < num_anchors; ++i) {
        float x_center = data[0 * num_anchors + i];
        float y_center = data[1 * num_anchors + i];
        float width = data[2 * num_anchors + i];
        float height = data[3 * num_anchors + i];

        // Find max class confidence
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
            int left = static_cast<int>((x_center - width / 2) * scaleX);
            int top = static_cast<int>((y_center - height / 2) * scaleY);
            int w = static_cast<int>(width * scaleX);
            int h = static_cast<int>(height * scaleY);

            cv::Rect box(left, top, w, h);
            box = box & cv::Rect(0, 0, frame.cols, frame.rows);

            if (box.width > 2 && box.height > 2) {
                classIds.push_back(classId);
                confidences.push_back(maxConf);
                boxes.push_back(box);
            }
        }
    }

    // NMS
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

std::vector<cv::Rect> Detector::findWhiteRegions(const cv::Mat& frame) {
    std::vector<cv::Rect> whiteRects;
    if (frame.empty()) return whiteRects;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat binary;
    cv::threshold(gray, binary, m_threshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < m_minArea) continue;

        cv::Rect rect = cv::boundingRect(contour);
        float aspectRatio = static_cast<float>(rect.width) / rect.height;
        if (aspectRatio < 0.7f || aspectRatio > 1.3f) continue;

        int margin = 15;
        cv::Rect expanded(
            rect.x - margin,
            rect.y - margin,
            rect.width + 2 * margin,
            rect.height + 2 * margin
        );
        expanded = expanded & cv::Rect(0, 0, frame.cols, frame.rows);
        whiteRects.push_back(expanded);
    }
    return whiteRects;
}

void Detector::detect(cv::Mat& frame, std::vector<Detection>& detections) {
    detections.clear();

    if (!m_useWhiteRegionDetection) {
        detectROI(frame, cv::Rect(0, 0, frame.cols, frame.rows), detections);
        return;
    }

    std::vector<cv::Rect> whiteRects = findWhiteRegions(frame);
    if (whiteRects.empty()) {
        return;
    }

    for (const cv::Rect& roiRect : whiteRects) {
        if (roiRect.width <= 0 || roiRect.height <= 0) continue;

        cv::Mat roiImg = frame(roiRect);
        std::vector<Detection> roiDetections;
        detectROI(roiImg, cv::Rect(0, 0, roiImg.cols, roiImg.rows), roiDetections);

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
        // Preprocess: resize and convert to blob
        cv::Mat blob;
        cv::resize(roiImg, blob, cv::Size(m_netWidth, m_netHeight));
        blob.convertTo(blob, CV_32F, 1.0 / 255.0);

        // Get input tensor
        auto input_tensor = m_infer_request.get_input_tensor();

        // Copy data to input tensor (NCHW format)
        float* input_data = input_tensor.data<float>();
        std::vector<cv::Mat> channels(3);
        cv::split(blob, channels);

        int channel_size = m_netWidth * m_netHeight;
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data + c * channel_size, channels[c].data, channel_size * sizeof(float));
        }

        // Inference
        auto start = std::chrono::high_resolution_clock::now();
        m_infer_request.infer();
        auto end = std::chrono::high_resolution_clock::now();
        m_lastInferenceTime = std::chrono::duration<double, std::milli>(end - start).count();

        // Get output
        auto output_tensor = m_infer_request.get_output_tensor();

        // Postprocess
        postprocess(roiImg, output_tensor, detections);

        // Map coordinates back to original frame
        for (auto& det : detections) {
            det.box.x += roi.x;
            det.box.y += roi.y;
            det.center.x += roi.x;
            det.center.y += roi.y;
        }

        // Inference time logging (disabled)
        // if (m_lastInferenceTime > 30) {
        //     std::cout << "[OpenVINO Inference: " << m_lastInferenceTime << "ms]" << std::endl;
        // }

    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
    }
}

void Detector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

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

        cv::circle(frame, det.center, 3, cv::Scalar(0, 0, 255), -1);
    }
}
