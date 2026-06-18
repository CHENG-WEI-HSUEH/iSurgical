#include "lc_surgical_phase.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

void TemporalPhasePostprocessorOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("inference_output");

  spec.param(n_phases_, "NumPhases", "Number of Phases", "Total phase categories", 13);
  spec.param(phase_names_, "PhaseName", "Phase Names", "List of string identifiers");
}

void TemporalPhasePostprocessorOp::start() {
  HOLOSCAN_LOG_INFO("TemporalPhasePostprocessorOp started with Unconstrained Instant-Forward Lock.");
  current_phase_idx_ = 0; // Cold-start initialization: always begin at preparation
}

void TemporalPhasePostprocessorOp::compute(holoscan::InputContext& input,
                                           holoscan::OutputContext&,
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

  // 3. Find the raw winner of the single instantaneous frame
  int raw_winner_idx = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
  float raw_confidence = probabilities[raw_winner_idx];

  // 4. UNCONSTRAINED INSTANT FORWARD STEP
  // If the model predicts ANY phase further ahead in the sequence with a confidence threshold 
  // (e.g., >= 35%), we fast-forward directly to that phase, skipping any intermediate steps.
  if (raw_winner_idx > current_phase_idx_ && raw_confidence >= 0.35f) {
    HOLOSCAN_LOG_WARN("[PostProcessor] !!! FORCE INSTANT LEAP DETECTED !!! Jumped: {} -> {}", 
                      names[current_phase_idx_], names[raw_winner_idx]);
    
    current_phase_idx_ = raw_winner_idx; // Move forward instantly
  }
  // Backward Lock: Any frame result where raw_winner_idx <= current_phase_idx_ is completely ignored.

  // 5. Output logging and dispatch via callback
  std::string finalized_phase_str = (current_phase_idx_ < static_cast<int>(names.size())) ? names[current_phase_idx_] : "Unknown";

  HOLOSCAN_LOG_INFO("[PostProcessor] Monotonic Lock State: {} | Instant Frame Winner: {} ({:.2f}%)", 
                    finalized_phase_str, names[raw_winner_idx], raw_confidence * 100.0f);

  if (phase_callback_) {
    phase_callback_(finalized_phase_str);
  }
}