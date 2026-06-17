#ifndef TEMPORAL_PHASE_POSTPROCESSOR_HPP
#define TEMPORAL_PHASE_POSTPROCESSOR_HPP

#include <holoscan/holoscan.hpp>
#include <chrono>
#include <deque>
#include <string>
#include <vector>
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
  // Holoscan Parameters loaded from the YAML file
  holoscan::Parameter<int> seq_len_;
  holoscan::Parameter<int> n_phases_;
  holoscan::Parameter<std::vector<std::string>> phase_names_;
  holoscan::Parameter<int> report_interval_sec_;

  // Sliding history buffer of past frame classification logits or embeddings
  // Storing raw logit vectors for smoothing
  std::deque<std::vector<float>> logit_history_;

  // Longer-term accumulator: summed per-class softmax probability since the
  // last report, used to log the highest-probability phase once per
  // report_interval_sec_ regardless of the (much shorter) smoothing window above.
  std::vector<float> minute_prob_sum_;
  int minute_tick_count_ = 0;
  std::chrono::steady_clock::time_point last_report_time_;

  std::function<void(const std::string&)> phase_callback_;
};

#endif // TEMPORAL_PHASE_POSTPROCESSOR_HPP