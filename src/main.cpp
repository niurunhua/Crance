#include "Config.h"
#include "Detector.h"
#include "FrameQueue.h"
#include "Tracker.h"
#include "SerialPort.h"
#include "AutoLabeler.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <deque>
#include <cmath>
#include <opencv2/opencv.hpp>

// Global flags
std::atomic<bool> g_running{ true };
FrameQueue g_frameQueue(2); // max 2 frames in queue

// Producer thread: capture frames from camera
void producerThread(int cameraIndex = 0) {
    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cameraIndex << std::endl;
        g_running = false;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, Config::INPUT_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::INPUT_HEIGHT);
    // Optionally set FPS
    // cap.set(cv::CAP_PROP_FPS, 30);

    std::cout << "Producer started, capturing from camera " << cameraIndex << std::endl;
    cv::Mat frame;
    while (g_running) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Captured empty frame." << std::endl;
            break;
        }
        // Resize to input dimensions if needed
        if (frame.size() != cv::Size(Config::INPUT_WIDTH, Config::INPUT_HEIGHT)) {
            cv::resize(frame, frame, cv::Size(Config::INPUT_WIDTH, Config::INPUT_HEIGHT));
        }
        bool pushed = g_frameQueue.push(frame);
        if (!pushed) {
            // Queue full, drop frame
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    cap.release();
    std::cout << "Producer thread exited." << std::endl;
}


int main(int argc, char** argv) {
    cv::setNumThreads(std::thread::hardware_concurrency());
    std::cout << "YOLOv11 Async Detection System" << std::endl;
    std::cout << "Press ESC to exit." << std::endl;

    // Initialize detector
    Detector detector(Config::MODEL_PATH,
        Config::CLASSES_FILE,
        Config::NETWORK_WIDTH,
        Config::NETWORK_HEIGHT,
        Config::CONFIDENCE_THRESHOLD,
        Config::NMS_THRESHOLD);
    if (!detector.init()) {
        std::cerr << "Detector initialization failed." << std::endl;
        return -1;
    }
    // Get class names for stability display
    std::vector<std::string> classNames = detector.getClassNames();

    // Initialize tracker
    Tracker tracker(Config::FILTER_ALPHA, Config::LOST_BUFFER_FRAMES);

    // Initialize serial port
    SerialPort serial;
    if (!serial.open(Config::SERIAL_PORT, Config::SERIAL_BAUD)) {
        std::cerr << "Failed to open serial port " << Config::SERIAL_PORT
            << ", continuing without serial output." << std::endl;
    }

    // Initialize auto-labeler
    AutoLabeler autoLabeler("dataset", Config::AUTO_LABEL_THRESH);

    // Start producer thread (camera)
    std::thread producer(producerThread, 0); // camera index 0

    // Main thread consumer loop
    cv::Mat frame;
    std::vector<Detection> detections;
    uint8_t heartbeat = 0;
    auto lastTime = std::chrono::steady_clock::now();

    // Frame skipping optimization
    const int SKIP_FRAMES = 2; // Process detection every 2 frames
    int frameCounter = 0;
    bool tracked = false;
    int dx = 0, dy = 0;
    TrackedObject lastTrackedObj;

    // Category stability queue
    std::deque<int> recentResults;
    const int QUEUE_SIZE = 5;
    int stableResult = -1;
    // Smoothed FPS
    float smoothedFps = 0.0f;
    const float FPS_ALPHA = 0.1f;

    while (g_running) {
        try {
            // Calculate FPS
            auto currentTime = std::chrono::steady_clock::now();
            auto deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();
            if (deltaTime == 0) deltaTime = 1; // avoid division by zero
            float fps = 1000.0f / deltaTime;
            // Smooth FPS
            smoothedFps = FPS_ALPHA * fps + (1 - FPS_ALPHA) * smoothedFps;
            lastTime = currentTime;

        // Pop frame (blocking with timeout) and check if empty
        if (!g_frameQueue.pop(frame)) continue;
        if (frame.empty()) continue;

        // Frame skipping: detect only every SKIP_FRAMES
        bool shouldDetect = (frameCounter % SKIP_FRAMES == 0);
        bool tracked = false;
        int dx = 0, dy = 0;
        TrackedObject obj;

        if (shouldDetect) {
            // Perform detection on this frame
            detections.clear();
            auto start = std::chrono::steady_clock::now();
            detector.detect(frame, detections);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // std::cout << "Detection time: " << elapsed << " ms" << std::endl;

            // Update tracker with new detections
            tracked = tracker.update(detections, frame.size());
        } else {
            // Use tracker state without new detection
            tracked = (tracker.isDetected() || tracker.getTrackedObject().lostCounter < Config::LOST_BUFFER_FRAMES);
        }

        // Get offset and tracked object if tracked
        if (tracked) {
            tracker.getOffset(dx, dy);
            obj = tracker.getTrackedObject();
            // Send via serial port (only on detection frames to avoid flooding)
            if (shouldDetect && serial.isOpen()) {
                serial.sendTrackedObject(dx, dy, obj.classId, heartbeat++);
            }
            // Auto-labeling if confidence high (only on detection frames)
            if (shouldDetect && obj.confidence >= Config::AUTO_LABEL_THRESH) {
                autoLabeler.process(frame, detections);
            }
            lastTrackedObj = obj;

            // Update recent results queue with valid classId
            if (obj.classId >= 0) {
                recentResults.push_back(obj.classId);
                // Keep queue size limited
                if (recentResults.size() > QUEUE_SIZE) {
                    recentResults.pop_front();
                }
                // Check consistency when queue is full
                if (recentResults.size() == QUEUE_SIZE) {
                    bool allEqual = true;
                    int first = recentResults.front();
                    for (int val : recentResults) {
                        if (val != first) {
                            allEqual = false;
                            break;
                        }
                    }
                    if (allEqual) {
                        stableResult = first;
                    }
                }
            }
        }

        frameCounter++;

        // Draw visualizations
        if (!detections.empty()) {
            // Draw actual detections from current frame
            detector.drawDetections(frame, detections);
        } else if (tracked) {
            // Draw tracked object from previous detection
            // Create a temporary detection from tracked object
            std::vector<Detection> tempDetections;
            Detection det;
            det.classId = obj.classId;
            det.confidence = obj.confidence;
            // Create a small box around tracked center for visualization
            int boxSize = 40;
            det.box = cv::Rect(obj.filteredCenter.x - boxSize/2, obj.filteredCenter.y - boxSize/2, boxSize, boxSize);
            det.center = obj.filteredCenter;
            tempDetections.push_back(det);
            detector.drawDetections(frame, tempDetections);
        }
        // Draw center crosshair
        cv::line(frame, cv::Point(Config::SCREEN_CENTER_X, 0),
            cv::Point(Config::SCREEN_CENTER_X, Config::INPUT_HEIGHT),
            cv::Scalar(0, 255, 255), 1);
        cv::line(frame, cv::Point(0, Config::SCREEN_CENTER_Y),
            cv::Point(Config::INPUT_WIDTH, Config::SCREEN_CENTER_Y),
            cv::Scalar(0, 255, 255), 1);
        // Draw offset text
        std::string offsetText = "dx: " + std::to_string(dx) + " dy: " + std::to_string(dy);
        cv::putText(frame, offsetText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // Draw stable class
        std::string classText = "Class: ";
        if (stableResult >= 0 && stableResult < static_cast<int>(classNames.size())) {
            classText += classNames[stableResult];
        } else {
            classText += "Unknown";
        }
        cv::putText(frame, classText, cv::Point(10, 45),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        // Draw FPS
        float fpsToShow = smoothedFps;
        if (!std::isfinite(fpsToShow) || fpsToShow < 0) {
            fpsToShow = fps; // fallback to raw fps
        }
        std::string fpsText = "FPS: " + std::to_string(fpsToShow).substr(0, 4);
        cv::putText(frame, fpsText, cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

        // Show frame
        cv::imshow("YOLOv11 Detection", frame);
        if (cv::waitKey(1) == 27) { // ESC to exit
            g_running = false;
            break;
        }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception: " << e.what() << std::endl;
            std::cerr << "File: " << e.file << " Line: " << e.line << std::endl;
            g_running = false;
            break;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception: " << e.what() << std::endl;
            g_running = false;
            break;
        } catch (...) {
            std::cerr << "Unknown exception occurred!" << std::endl;
            g_running = false;
            break;
        }
    }

    // Wait for threads to finish
    producer.join();

    // Cleanup
    serial.close();
    cv::destroyAllWindows();
    std::cout << "Program terminated." << std::endl;
    return 0;
}