#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
    cv::Point2f center; // 框中心点
};

class Detector {
public:
    Detector(const std::string& modelPath,
             const std::string& classesFile,
             int netWidth,
             int netHeight,
             float confThreshold,
             float nmsThreshold,
             bool useWhiteRegionDetection = true);
    ~Detector() = default;

    /**
     * @brief 初始化检测器（加载模型、类别等）
     * @return 成功返回true，失败返回false
     */
    bool init();

    /**
     * @brief 在整帧图像上进行检测
     * @param frame 输入帧（将调整到网络尺寸）
     * @param detections 输出检测结果向量
     */
    void detect(cv::Mat& frame, std::vector<Detection>& detections);

    /**
     * @brief 在ROI区域内进行检测（局部推理）
     * @param frame 完整输入帧
     * @param roi 感兴趣区域（将裁剪此区域进行检测）
     * @param detections 输出检测结果向量（坐标已映射回原图）
     */
    void detectROI(const cv::Mat& frame, const cv::Rect& roi, std::vector<Detection>& detections);

    /**
     * @brief 在帧上绘制检测结果
     */
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

    /**
     * @brief 设置白色区域检测的二值化阈值
     * @param threshold 二值化阈值（0-255）
     */
    void setThreshold(int threshold) { m_threshold = threshold; }

    /**
     * @brief 设置白色区域检测的最小轮廓面积
     * @param minArea 最小轮廓面积（像素数）
     */
    void setMinArea(int minArea) { m_minArea = minArea; }

    /**
     * @brief 获取类别名称列表
     */
    std::vector<std::string> getClassNames() const { return m_classNames; }

private:
    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    void postprocess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, std::vector<Detection>& detections);
    std::vector<cv::Rect> findWhiteRegions(const cv::Mat& frame);

    std::string m_modelPath;
    std::string m_classesFile;
    int m_netWidth;
    int m_netHeight;
    float m_confThreshold;
    float m_nmsThreshold;
    cv::dnn::Net m_net;
    std::vector<std::string> m_classNames;
    std::vector<std::string> m_outputNames;
    int m_threshold;      // 二值化阈值（使用Config中的静态值）
    int m_minArea;        // 最小轮廓面积（使用Config中的静态值）
    bool m_useWhiteRegionDetection; // 是否使用白纸区域检测
};

#endif // DETECTOR_H