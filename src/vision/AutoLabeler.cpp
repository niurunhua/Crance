#include "AutoLabeler.h"
#include "../core/Config.h"
#include <fstream>
#include <iostream>
#include <sys/stat.h> // for mkdir
#include <chrono>
#include <sstream> 
#include <iomanip>

#ifdef _WIN32
#include <direct.h> // for _mkdir
#endif

AutoLabeler::AutoLabeler(const std::string& outputDir, float confidenceThreshold)
    : m_outputDir(outputDir),
      m_confidenceThreshold(confidenceThreshold),
      m_savedCount(0) {
    ensureDirectoryExists(m_outputDir);
    ensureDirectoryExists(m_outputDir + "/images");
    ensureDirectoryExists(m_outputDir + "/labels");
}

void AutoLabeler::setOutputDir(const std::string& outputDir) {
    m_outputDir = outputDir;
    ensureDirectoryExists(m_outputDir);
    ensureDirectoryExists(m_outputDir + "/images");
    ensureDirectoryExists(m_outputDir + "/labels");
}

void AutoLabeler::ensureDirectoryExists(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

int AutoLabeler::process(const cv::Mat& frame, const std::vector<Detection>& detections) {
    // Filter detections by confidence threshold
    std::vector<Detection> highConfDetections;
    float maxConf = 0;
    for (const auto& det : detections) {
        if (det.confidence > maxConf) maxConf = det.confidence;
        if (det.confidence >= m_confidenceThreshold) {
            highConfDetections.push_back(det);
        }
    }

    static int callCount = 0;
    if (++callCount <= 5) {
        std::cout << "[AutoLabeler] 检测数:" << detections.size()
                  << " 最高置信度:" << maxConf
                  << " 阈值:" << m_confidenceThreshold
                  << " 达标数:" << highConfDetections.size() << std::endl;
    }

    if (highConfDetections.empty()) {
        return 0;
    }

    // Generate unique filename based on timestamp
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();
    std::stringstream ss;
    ss << "sample_" << std::setfill('0') << std::setw(10) << value;
    std::string baseName = ss.str();

    saveImageAndLabel(frame, highConfDetections, baseName);
    m_savedCount++;
    return static_cast<int>(highConfDetections.size());
}

void AutoLabeler::saveImageAndLabel(const cv::Mat& frame, const std::vector<Detection>& detections, const std::string& baseName) {
    // Save image
    std::string imagePath = m_outputDir + "/images/" + baseName + ".jpg";
    cv::imwrite(imagePath, frame);

    // Save label file (YOLO format)
    std::string labelPath = m_outputDir + "/labels/" + baseName + ".txt";
    std::ofstream labelFile(labelPath);
    if (!labelFile.is_open()) {
        std::cerr << "Failed to open label file: " << labelPath << std::endl;
        return;
    }

    const float imgWidth = static_cast<float>(frame.cols);
    const float imgHeight = static_cast<float>(frame.rows);

    for (const auto& det : detections) {
        // Convert bounding box to normalized coordinates
        float x_center = (det.box.x + det.box.width / 2.0f) / imgWidth;
        float y_center = (det.box.y + det.box.height / 2.0f) / imgHeight;
        float width = det.box.width / imgWidth;
        float height = det.box.height / imgHeight;

        // Clamp to [0,1]
        x_center = std::max(0.0f, std::min(1.0f, x_center));
        y_center = std::max(0.0f, std::min(1.0f, y_center));
        width = std::max(0.0f, std::min(1.0f, width));
        height = std::max(0.0f, std::min(1.0f, height));

        labelFile << det.classId << " " << x_center << " " << y_center << " " << width << " " << height << "\n";
    }
    labelFile.close();
    std::cout << "Auto-labeled: " << baseName << ".jpg with " << detections.size() << " objects." << std::endl;
}