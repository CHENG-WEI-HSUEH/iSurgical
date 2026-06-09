import time
from concurrent import futures

import grpc
from PySide6.QtCore import QObject, Signal

from generated import surgical_phase_pb2
from generated import surgical_phase_pb2_grpc


class PhaseSignalBridge(QObject):
    phase_received = Signal(str, int)
    surgery_started = Signal(str, str, str, int)


class SurgicalPhaseServicer(surgical_phase_pb2_grpc.SurgicalPhaseServiceServicer):
    def __init__(self, bridge: PhaseSignalBridge):
        self._bridge = bridge

    def SendPhase(self, request, context):
        self._bridge.phase_received.emit(request.phase, request.timestamp_ms)
        return surgical_phase_pb2.PhaseResponse(success=True)

    def StartSurgery(self, request, context):
        self._bridge.surgery_started.emit(
            request.doctor_name,
            request.patient_id,
            request.procedure,
            request.start_time_ms,
        )
        return surgical_phase_pb2.PhaseResponse(success=True)


def create_server(bridge: PhaseSignalBridge, port: int = 50051) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    surgical_phase_pb2_grpc.add_SurgicalPhaseServiceServicer_to_server(
        SurgicalPhaseServicer(bridge), server
    )
    server.add_insecure_port(f"[::]:{port}")
    return server
