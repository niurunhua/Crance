#include "FrameQueue.h"
#include <chrono>

FrameQueue::FrameQueue(size_t maxSize) : m_maxSize(maxSize) {}

bool FrameQueue::push(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // If queue is full, wait until not full (with timeout to avoid deadlock)
    if (m_queue.size() >= m_maxSize) {
        // Instead of waiting indefinitely, we can drop the oldest frame
        // to maintain real-time performance.
        m_queue.pop();
    }
    m_queue.push(frame.clone()); // clone to avoid reference issues
    lock.unlock();
    m_condNotEmpty.notify_one();
    return true;
} 

bool FrameQueue::pop(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // Wait with timeout (1ms) to avoid blocking indefinitely
    if (!m_condNotEmpty.wait_for(lock, std::chrono::milliseconds(1), [this]() { return !m_queue.empty(); })) {
        return false; // timeout, queue still empty
    }
    frame = m_queue.front();
    m_queue.pop();
    lock.unlock();
    m_condNotFull.notify_one();
    return true;
}

size_t FrameQueue::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

void FrameQueue::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
    m_condNotFull.notify_all();
}

void FrameQueue::setMaxSize(size_t maxSize) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_maxSize = maxSize;
}

bool FrameQueue::empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.empty();
}

bool FrameQueue::full() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size() >= m_maxSize;
}