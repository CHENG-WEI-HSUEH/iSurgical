#ifndef TEMPORAL_PHASE_POSTPROCESSOR_HPP
#define TEMPORAL_PHASE_POSTPROCESSOR_HPP

#include <holoscan/holoscan.hpp>
#include <string>
#include <vector>
#include <deque>
#include <memory>
#include <functional>

class TemporalPhasePostprocessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TemporalPhasePostprocessorOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

  void setPhaseCallback(std::function<void(const std::string&)> cb) {
    phase_callback_ = std::move(cb);
  }

 private:
  holoscan::Parameter<int> n_phases_;
  holoscan::Parameter<std::vector<std::string>> phase_names_;

  // Robust State Tracking
  int current_phase_idx_{0}; // Hardcoded start at 0 ("preparation")
  
  // History buffer: stores the softmax probability vector for the last N evaluations
  std::deque<std::vector<float>> probability_history_;
  
  // 3 minutes window / 10 seconds per sample = 18 samples max history
  static constexpr size_t kMaxHistorySize = 18; 

  std::function<void(const std::string&)> phase_callback_;
};

#endif // TEMPORAL_PHASE_POSTPROCESSOR_HPP