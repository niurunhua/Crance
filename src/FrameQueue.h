#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

/**
 * @brief Thread-safe queue for frames between producer and consumer.
 */
class FrameQueue {
public:
    FrameQueue(size_t maxSize = 10);
    ~FrameQueue() = default;

    /**
     * @brief Push a frame into the queue.
     * @param frame The frame to push.
     * @return true if successful, false if queue is full and cannot wait.
     */
    bool push(const cv::Mat& frame);

    /**
     * @brief Pop a frame from the queue (blocking).
     * @param frame Output frame.
     * @return true if successful, false if queue is empty and cannot wait.
     */
    bool pop(cv::Mat& frame);

    /**
     * @brief Get current size of the queue.
     */
    size_t size() const;

    /**
     * @brief Clear all frames in the queue.
     */
    void clear();

    /**
     * @brief Set maximum size of the queue.
     */
    void setMaxSize(size_t maxSize);

    /**
     * @brief Check if queue is empty.
     */
    bool empty() const;

    /**
     * @brief Check if queue is full.
     */
    bool full() const;

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_condNotEmpty;
    std::condition_variable m_condNotFull;
    std::queue<cv::Mat> m_queue;
    size_t m_maxSize;
};

#endif // FRAME_QUEUE_H