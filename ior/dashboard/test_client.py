"""Test client to simulate phase updates from iSurgical."""
import time
import grpc
from generated import surgical_phase_pb2, surgical_phase_pb2_grpc


def main():
    channel = grpc.insecure_channel("localhost:50051")
    stub = surgical_phase_pb2_grpc.SurgicalPhaseServiceStub(channel)

    # Start surgery
    now_ms = int(time.time() * 1000)
    stub.StartSurgery(surgical_phase_pb2.SurgeryInfo(
        doctor_name="Dr. John Smith",
        patient_id="Jane Doe",
        procedure="Laparoscopic Cholecystectomy",
        start_time_ms=now_ms,
    ))
    print("Surgery started")

    phases = [
        "Preparation",
        "CalotTriangleDissection",
        "ClippingAndCutting",
        "GallbladderDissection",
        "GallbladderPackaging",
        "CleaningAndCoagulation",
        "GallbladderRetraction",
    ]

    for phase in phases:
        time.sleep(5)
        ts = int(time.time() * 1000)
        stub.SendPhase(surgical_phase_pb2.PhaseRequest(
            phase=phase, timestamp_ms=ts
        ))
        print(f"Sent phase: {phase}")

    print("Done")


if __name__ == "__main__":
    main()
