#include "phase_grpc_client.hpp"
#include <chrono>
#include <iostream>

PhaseGrpcClient::PhaseGrpcClient(const std::string& server_address) {
  channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  stub_ = surgical_phase::SurgicalPhaseService::NewStub(channel_);

  is_running_ = true;
  sender_thread_ = std::thread(&PhaseGrpcClient::senderLoop, this);

  std::cout << "[PhaseGrpcClient] Started (server=" << server_address << ")" << std::endl;
}

PhaseGrpcClient::~PhaseGrpcClient() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    is_running_ = false;
  }
  queue_cv_.notify_all();

  if (sender_thread_.joinable()) {
    sender_thread_.join();
  }
  std::cout << "[PhaseGrpcClient] Stopped" << std::endl;
}

void PhaseGrpcClient::sendPhase(const std::string& phase) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    phase_queue_.push(phase);
  }
  queue_cv_.notify_one();
}

void PhaseGrpcClient::senderLoop() {
  while (true) {
    std::string phase;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() {
        return !phase_queue_.empty() || !is_running_.load();
      });
      if (!is_running_ && phase_queue_.empty()) return;
      phase = phase_queue_.front();
      phase_queue_.pop();
    }

    surgical_phase::PhaseRequest request;
    request.set_phase(phase);
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    request.set_timestamp_ms(ms);

    surgical_phase::PhaseResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(2));

    grpc::Status status = stub_->SendPhase(&context, request, &response);
    if (status.ok()) {
      std::cout << "[PhaseGrpcClient] Sent phase '" << phase
                << "' -> success=" << response.success() << std::endl;
    } else {
      std::cerr << "[PhaseGrpcClient] RPC FAILED (" << static_cast<int>(status.error_code())
                << "): " << status.error_message() << std::endl;
    }
  }
}
