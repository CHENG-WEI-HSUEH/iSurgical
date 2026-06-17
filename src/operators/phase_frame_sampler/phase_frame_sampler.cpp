#include "phase_frame_sampler.hpp"

void PhaseFrameSamplerOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>(
      "in", holoscan::IOSpec::IOSize(1), holoscan::IOSpec::QueuePolicy::kPop);
  spec.output<holoscan::gxf::Entity>("out");
}

void PhaseFrameSamplerOp::compute(holoscan::InputContext& input,
                                   holoscan::OutputContext& output,
                                   holoscan::ExecutionContext&) {
  auto msg = input.receive<holoscan::gxf::Entity>("in");
  if (!msg) return;
  output.emit(msg.value(), "out");
}
