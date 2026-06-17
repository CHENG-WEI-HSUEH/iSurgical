#ifndef _PHASE_FRAME_SAMPLER_HPP_
#define _PHASE_FRAME_SAMPLER_HPP_

#include <holoscan/holoscan.hpp>

// Decouples surgical phase recognition from the live video frame rate.
// Paired with a PeriodicCondition on this operator and a kPop input queue
// policy, only the most recently arrived frame is kept between ticks; older
// frames are dropped instead of backing up. This lets the phase branch run
// at a fixed low rate (e.g. 5 Hz) without ever applying backpressure to the
// real-time segmentation path sharing the same source operator.
class PhaseFrameSamplerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PhaseFrameSamplerOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;
};

#endif  // _PHASE_FRAME_SAMPLER_HPP_
