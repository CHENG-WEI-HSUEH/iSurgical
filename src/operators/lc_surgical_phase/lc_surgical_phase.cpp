#include "lc_surgical_phase.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

void TemporalPhasePostprocessorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("inference_output");
  spec.output<std::string>("smoothed_phase").condition(holoscan::ConditionType::kNone);

  spec.param(n_phases_, "NumPhases", "Number of Phases", "Total phase categories", 13);
  spec.param(phase_names_, "PhaseName", "Phase Names", "List of string identifiers");
}

void TemporalPhasePostprocessorOp::start() {
  HOLOSCAN_LOG_INFO("TemporalPhasePostprocessorOp started with 3-minute sliding window smoother.");
  current_phase_idx_ = 0; // Deterministic initial state: preparation
  probability_history_.clear();
}

void TemporalPhasePostprocessorOp::compute(holoscan::InputContext& input,
                                           holoscan::OutputContext& output,
                                           holoscan::ExecutionContext&) {
  auto msg = input.receive<holoscan::TensorMap>("inference_output");
  if (!msg || msg.value().find("phase_logits") == msg.value().end()) {
    return; 
  }

  auto tensor = msg.value()["phase_logits"];
  int classes = n_phases_.get();
  const auto& names = phase_names_.get();

  // 1. Copy inference output from GPU to CPU
  std::vector<float> logits(classes);
  cudaError_t cuda_status = cudaMemcpy(logits.data(), tensor->data(),
                                       classes * sizeof(float), cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("[PostProcessor] cudaMemcpy failed: {}", cudaGetErrorString(cuda_status));
    return;
  }

  // 2. Compute Stable Softmax Probabilities
  float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probabilities(classes);
  float sum_exp = 0.0f;
  for (int i = 0; i < classes; ++i) {
    probabilities[i] = std::exp(logits[i] - max_logit);
    sum_exp += probabilities[i];
  }
  for (int i = 0; i < classes; ++i) {
    probabilities[i] /= sum_exp;
  }

  // 3. Update the 3-minute Sliding History Buffer
  probability_history_.push_back(probabilities);
  if (probability_history_.size() > kMaxHistorySize) {
    probability_history_.pop_front();
  }

  // 4. Accumulate probabilities across the window for robust voting
  std::vector<float> accumulated_probs(classes, 0.0f);
  for (const auto& window_probs : probability_history_) {
    for (int i = 0; i < classes; ++i) {
      accumulated_probs[i] += window_probs[i];
    }
  }

  // 5. Apply the Monotonic Constraint: Find the best phase equal to or forward from the current state
  int best_phase_idx = current_phase_idx_;
  float max_accumulated_score = accumulated_probs[current_phase_idx_];

  for (int i = current_phase_idx_ + 1; i < classes; ++i) {
    if (accumulated_probs[i] > max_accumulated_score) {
      max_accumulated_score = accumulated_probs[i];
      best_phase_idx = i;
    }
  }

  // 6. State Transition Verification
  if (best_phase_idx > current_phase_idx_) {
    HOLOSCAN_LOG_WARN("[PostProcessor] Stable transition detected over history window: {} -> {}", 
                      names[current_phase_idx_], names[best_phase_idx]);
    current_phase_idx_ = best_phase_idx;
  }

  // 7. Output Result Handling
  std::string finalized_phase_str = (current_phase_idx_ < static_cast<int>(names.size())) ? names[current_phase_idx_] : "Unknown";
  int current_frame_raw_winner = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

  HOLOSCAN_LOG_INFO("[PostProcessor] Locked State: {} | Instantaneous Frame Result: {} ({:.2f}%) | Buffer: {}/{}", 
                    finalized_phase_str, names[current_frame_raw_winner], probabilities[current_frame_raw_winner] * 100.0f,
                    probability_history_.size(), kMaxHistorySize);

  output.emit(finalized_phase_str, "smoothed_phase");

  if (phase_callback_) {
    phase_callback_(finalized_phase_str);
  }
}