#include "periodic_counter.hpp"

#include <chrono>

void PeriodicCounterOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(interval_ms_, "interval_ms", "Interval (ms)",
             "Timer interval in milliseconds", int64_t(1000));
}

void PeriodicCounterOp::start() {
  async_condition_ = condition<holoscan::AsynchronousCondition>("async_condition");

  is_running_ = true;
  count_ = 0;

  timer_thread_ = std::thread([this]() {
    auto interval = std::chrono::milliseconds(interval_ms_.get());

    while (is_running_) {
      std::unique_lock<std::mutex> lock(stop_mutex_);
      if (stop_cv_.wait_for(lock, interval, [this]() { return !is_running_.load(); })) {
        break;
      }

      async_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
    }
  });

  HOLOSCAN_LOG_INFO("PeriodicCounterOp started with interval {}ms", interval_ms_.get());
}

void PeriodicCounterOp::stop() {
  {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    is_running_ = false;
  }
  stop_cv_.notify_all();

  if (timer_thread_.joinable()) {
    timer_thread_.join();
  }
  HOLOSCAN_LOG_INFO("PeriodicCounterOp stopped. Final count: {}", count_.load());
}

void PeriodicCounterOp::compute(holoscan::InputContext&,
                                holoscan::OutputContext&,
                                holoscan::ExecutionContext&) {
  ++count_;
  HOLOSCAN_LOG_INFO("PeriodicCounterOp count: {}", count_.load());

  async_condition_->event_state(holoscan::AsynchronousEventState::WAIT);
}
