#ifndef PHASE_GRPC_CLIENT_HPP
#define PHASE_GRPC_CLIENT_HPP

#include <holoscan/holoscan.hpp>

#include <grpcpp/grpcpp.h>
#include "surgical_phase.grpc.pb.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <string>

class PhaseGrpcClientOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PhaseGrpcClientOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& input,
               holoscan::OutputContext& output,
               holoscan::ExecutionContext& context) override;

  void sendPhase(const std::string& phase);

 private:
  holoscan::Parameter<std::string> server_address_;

  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<surgical_phase::SurgicalPhaseService::Stub> stub_;

  std::thread sender_thread_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<std::string> phase_queue_;
  std::atomic<bool> is_running_{false};

  void senderLoop();
};

#endif // PHASE_GRPC_CLIENT_HPP
