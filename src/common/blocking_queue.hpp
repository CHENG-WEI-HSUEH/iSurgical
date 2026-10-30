#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

// Bounded thread-safe queue shared between pipeline stages.
//   drop_oldest = false: Push blocks while the queue is full (no frame is lost)
//   drop_oldest = true : Push discards the oldest item instead (consumer always sees the newest data)
// Close() wakes every waiter; Pop keeps returning queued items until the queue is empty.
template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity, bool drop_oldest = false)
        : capacity_(capacity > 0 ? capacity : 1), drop_oldest_(drop_oldest) {}

    // Returns false if the queue was closed
    bool Push(T item) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            if (drop_oldest_) {
                if (closed_) return false;
                while (queue_.size() >= capacity_) {
                    queue_.pop_front();
                    ++dropped_;
                }
            } else {
                not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
                if (closed_) return false;
            }
            queue_.push_back(std::move(item));
        }
        not_empty_.notify_one();
        return true;
    }

    // Returns false once the queue is closed and drained
    bool Pop(T& item) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });
            if (queue_.empty()) return false;
            item = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();
        return true;
    }

    void Close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t Dropped() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_;
    }

private:
    const size_t capacity_;
    const bool drop_oldest_;
    std::deque<T> queue_;
    mutable std::mutex mtx_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
    size_t dropped_ = 0;
};
