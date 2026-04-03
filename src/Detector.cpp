#include "Detector.h"
#include "Config.h"
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
    m_nmsThreshold(nmsThreshold) {}

bool Detector::init() {
    // Load model
    m_net = cv::dnn::readNetFromONNX(m_modelPath);
    if (m_net.empty()) {
        std::cerr << "Failed to load model from " << m_modelPath << std::endl;
        return false;
    }
    // Set preferable backend (CPU)
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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
    // If class names not loaded, create default names
    if (m_classNames.empty()) {
        for (int i = 0; i < Config::NUM_CLASSES; ++i) {
            m_classNames.push_back("class_" + std::to_string(i));
        }
    }

    // Get output layer names (YOLOv11 may have single output "output0")
    m_outputNames = m_net.getUnconnectedOutLayersNames();
    if (m_outputNames.empty()) {
        // Fallback: assume output name "output0"
        m_outputNames.push_back("output0");
    }

    std::cout << "Detector initialized with " << m_classNames.size() << " classes." << std::endl;
    return true;
}

void Detector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    // Create blob from image, no scaling (0-255), resize to net size, no mean subtraction
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(m_netWidth, m_netHeight), cv::Scalar(), true, false);
}

void Detector::postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, std::vector<Detection>& detections) {
    detections.clear();
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outputs) {
        cv::Mat outMat;

        // 1. Reduce 3D Tensor to 2D matrix
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

        // 2. Core fix: transpose matrix to ensure safe row-wise reading
        if (outMat.rows < outMat.cols) {
            cv::transpose(outMat, outMat);
        }

        const int numAnchors = outMat.rows;     // e.g., 8400
        const int numAttributes = outMat.cols;  // e.g., 9 (4 coordinates + 5 classes)
        const int numClasses = numAttributes - 4;

        if (numClasses <= 0) continue;

        // 3. Safely traverse all prediction boxes
        for (int i = 0; i < numAnchors; ++i) {
            const float* data = outMat.ptr<float>(i);

            // Normalized coordinates
            const float x_center = data[0];
            const float y_center = data[1];
            const float width = data[2];
            const float height = data[3];

            // Find class with maximum confidence
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
                // Convert back to real pixel coordinates
                const int left = static_cast<int>((x_center - width / 2) * frame.cols);
                const int top = static_cast<int>((y_center - height / 2) * frame.rows);
                const int w = static_cast<int>(width * frame.cols);
                const int h = static_cast<int>(height * frame.rows);

                cv::Rect box(left, top, w, h);
                // Ensure box is within image boundaries
                box = box & cv::Rect(0, 0, frame.cols, frame.rows);

                // Keep only valid boxes
                if (box.width > 2 && box.height > 2) {
                    classIds.push_back(classId);
                    confidences.push_back(maxConf);
                    boxes.push_back(box);
                }
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    // Create Detection objects
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

    // 1. Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 2. Threshold to extract bright white paper
    cv::Mat binary;
    cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY);

    // 3. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. Filter contours
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 1500) continue; // noise

        cv::Rect rect = cv::boundingRect(contour);
        float aspectRatio = static_cast<float>(rect.width) / rect.height;
        if (aspectRatio < 0.7f || aspectRatio > 1.3f) continue; // not square-like

        // Add margin (15 pixels each side)
        int margin = 15;
        cv::Rect expanded(
            rect.x - margin,
            rect.y - margin,
            rect.width + 2 * margin,
            rect.height + 2 * margin
        );
        // Ensure within image bounds
        expanded = expanded & cv::Rect(0, 0, frame.cols, frame.rows);
        whiteRects.push_back(expanded);
    }
    return whiteRects;
}

void Detector::detect(cv::Mat& frame, std::vector<Detection>& detections) {
    detections.clear();
    // Step 1: find white paper regions
    std::vector<cv::Rect> whiteRects = findWhiteRegions(frame);
    if (whiteRects.empty()) {
        // No white paper found, skip YOLO inference
        return;
    }

    // Step 2: for each white region, crop ROI and run YOLO
    for (const cv::Rect& roiRect : whiteRects) {
        // Ensure ROI is valid
        if (roiRect.width <= 0 || roiRect.height <= 0) continue;

        cv::Mat roiImg = frame(roiRect);
        cv::Mat blob;
        preprocess(roiImg, blob);
        m_net.setInput(blob);
        std::vector<cv::Mat> outputs;
        m_net.forward(outputs, m_outputNames);

        // Temporary detections relative to ROI
        std::vector<Detection> roiDetections;
        postprocess(roiImg, outputs, roiDetections);

        // Map coordinates back to original frame
        for (auto& det : roiDetections) {
            det.box.x += roiRect.x;
            det.box.y += roiRect.y;
            det.center.x += roiRect.x;
            det.center.y += roiRect.y;
            detections.push_back(det);
        }
    }

    // Apply Non-Maximum Suppression across all detections from different ROIs?
    // Since each ROI is separate paper, NMS across ROIs may not be needed.
    // But if multiple ROIs overlap, we could apply NMS.
    // For simplicity, we keep all detections.
}

void Detector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Draw rectangle
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        // Label with class name and confidence
        // Ensure classId is within bounds
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

        // Draw center point
        cv::circle(frame, det.center, 3, cv::Scalar(0, 0, 255), -1);
    }
}