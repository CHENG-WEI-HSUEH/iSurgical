#include "lc_surgical_phase.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

void TemporalPhasePostprocessorOp::setup(holoscan::OperatorSpec& spec) {
  // Input received from downstream of an inference operator (e.g., YoloSeg or Phase CNN)
  spec.input<holoscan::gxf::Entity>("inference_output");
  // Output port if you want to pass the smoothed string or index downstream.
  // Nothing currently connects to it (the gRPC client is wired via a native
  // callback instead), so disable the default downstream-affordable condition
  // or this operator would never be scheduled to tick.
  spec.output<std::string>("smoothed_phase").condition(holoscan::ConditionType::kNone);

  spec.param(seq_len_, "SequenceLength", "Sequence Length",
             "Temporal window frame size for smoothing", 15);
  spec.param(n_phases_, "NumPhases", "Number of Phases",
             "Total phase categories", 13);
  spec.param(phase_names_, "PhaseName", "Phase Names",
             "List of string identifiers for the phases");
  spec.param(report_interval_sec_, "ReportIntervalSec", "Report Interval (s)",
             "How often to log the highest-probability phase over the recent window", 60);
}

void TemporalPhasePostprocessorOp::start() {
  HOLOSCAN_LOG_INFO("TemporalPhasePostprocessorOp started with window size: {}", seq_len_.get());
  if (phase_names_.get().size() != static_cast<size_t>(n_phases_.get())) {
    HOLOSCAN_LOG_WARN("Mismatch between NumPhases ({}) and PhaseName list size ({})!",
                      n_phases_.get(), phase_names_.get().size());
  }

  minute_prob_sum_.assign(n_phases_.get(), 0.0f);
  minute_tick_count_ = 0;
  last_report_time_ = std::chrono::steady_clock::now();
}

void TemporalPhasePostprocessorOp::compute(holoscan::InputContext& input,
                                            holoscan::OutputContext& output,
                                            holoscan::ExecutionContext&) {
  auto msg = input.receive<holoscan::TensorMap>("inference_output");
  if (!msg || msg.value().find("phase_logits") == msg.value().end()) {
    return; // Upstream inference hasn't published logits yet or tensor key mismatched
  }

  auto tensor = msg.value()["phase_logits"];
  int classes = n_phases_.get();
  
  // Inference output tensors live in device memory; copy them to host before use
  std::vector<float> current_logits(classes);
  cudaError_t cuda_status = cudaMemcpy(current_logits.data(), tensor->data(),
                                        classes * sizeof(float), cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("[PostProcessor] cudaMemcpy failed: {}", cudaGetErrorString(cuda_status));
    return;
  }

  // 0. Accumulate softmax probabilities into the longer-term (e.g. 1 minute) report window,
  // independent of the short sliding-window smoothing below.
  {
    float max_logit = *std::max_element(current_logits.begin(), current_logits.end());
    std::vector<float> probs(classes);
    float sum_exp = 0.0f;
    for (int c = 0; c < classes; ++c) {
      probs[c] = std::exp(current_logits[c] - max_logit);
      sum_exp += probs[c];
    }
    for (int c = 0; c < classes; ++c) {
      minute_prob_sum_[c] += probs[c] / sum_exp;
    }
    ++minute_tick_count_;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed_sec =
      std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time_).count();
  if (elapsed_sec >= report_interval_sec_.get() && minute_tick_count_ > 0) {
    int top_idx = std::distance(minute_prob_sum_.begin(),
                                 std::max_element(minute_prob_sum_.begin(), minute_prob_sum_.end()));
    const auto& names = phase_names_.get();
    float avg_confidence = minute_prob_sum_[top_idx] / static_cast<float>(minute_tick_count_);
    std::string top_name = (top_idx >= 0 && top_idx < static_cast<int>(names.size()))
                                ? names[top_idx]
                                : "Unknown";
    HOLOSCAN_LOG_INFO(
        "[PostProcessor] Highest-probability phase over last {}s ({} samples): {} (avg confidence {:.1f}%)",
        elapsed_sec, minute_tick_count_, top_name, avg_confidence * 100.0f);

    std::fill(minute_prob_sum_.begin(), minute_prob_sum_.end(), 0.0f);
    minute_tick_count_ = 0;
    last_report_time_ = now;
  }

  // 1. Maintain sliding window buffer history
  logit_history_.push_back(current_logits);
  if (static_cast<int>(logit_history_.size()) > seq_len_.get()) {
    logit_history_.pop_front();
  }

  // 2. Perform Temporal Pooling (Temporal Smoothing Strategy)
  // Strategy: Average the logits across the sliding window to filter high-frequency noise
  std::vector<float> pooled_logits(classes, 0.0f);
  for (const auto& individual_logits : logit_history_) {
    for (int c = 0; c < classes; ++c) {
      pooled_logits[c] += individual_logits[c];
    }
  }

  // 3. Extract final ArgMax from the smoothed window profile
  int best_phase_idx = std::distance(
      pooled_logits.begin(), 
      std::max_element(pooled_logits.begin(), pooled_logits.end())
  );

  // 4. Dispatch Phase Label
  const auto& names = phase_names_.get();
  if (best_phase_idx >= 0 && best_phase_idx < static_cast<int>(names.size())) {
    std::string smoothed_phase_name = names[best_phase_idx];
    
    HOLOSCAN_LOG_INFO("[PostProcessor] Smoothed Phase: {}", smoothed_phase_name);

    // Emit via Holoscan downstream pipeline
    output.emit(smoothed_phase_name, "smoothed_phase");

    // Fire application runtime native callback if registered
    if (phase_callback_) {
      phase_callback_(smoothed_phase_name);
    }
  }
}