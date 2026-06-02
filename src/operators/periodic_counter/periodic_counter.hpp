#ifndef _PERIODIC_COUNTER_HPP_
#define _PERIODIC_COUNTER_HPP_

#include <thread>
#include <atomic>

#include <holoscan/holoscan.hpp>

class PeriodicCounterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PeriodicCounterOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<int64_t> interval_ms_;
  std::shared_ptr<holoscan::AsynchronousCondition> async_condition_;

  std::thread timer_thread_;
  std::atomic<bool> is_running_{false};
  std::atomic<uint64_t> count_{0};
};

#endif // _PERIODIC_COUNTER_HPP_
