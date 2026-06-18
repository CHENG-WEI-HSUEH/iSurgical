#!/usr/bin/env python3
import os
import sys
import datetime
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton, QFormLayout,
    QMessageBox, QGraphicsDropShadowEffect, QFrame
)
from PySide6.QtGui import QColor, QFont
from ruamel.yaml import YAML

import grpc

sys.path.insert(0, str(Path(__file__).resolve().parent / "generated"))
import surgical_phase_pb2
import surgical_phase_pb2_grpc

DOCKER_IMAGE = "isurgical-app:latest"

class ORDashboard(QMainWindow):
    def __init__(self, app_yaml_path: Path, dashboard_yaml_path: Path):
        super().__init__()
        self.app_yaml_path = app_yaml_path
        self.dashboard_yaml_path = dashboard_yaml_path
        
        self.yaml_handler = YAML()
        self.yaml_handler.preserve_quotes = True
        self.yaml_handler.allow_duplicate_keys = True
        self.yaml_handler.indent(mapping=2, sequence=4, offset=2)
        
        self.departments = []
        self.procedures = []
        
        self.load_yaml_configurations()
        self.init_ui()
        
    def load_yaml_configurations(self):
        """Dynamically populates items using target external YAML configuration nodes."""
        if not self.app_yaml_path.exists():
            QMessageBox.critical(self, "Error", f"Missing pipeline configuration at:\n{self.app_yaml_path}")
            sys.exit(1)
            
        try:
            # 1. Load User UI Dropdowns from configs/dashboard.yaml
            if self.dashboard_yaml_path.exists():
                with open(self.dashboard_yaml_path, 'r', encoding='utf-8') as f:
                    dash_data = self.yaml_handler.load(f) or {}
                dash_settings = dash_data.get("DashboardSettings", {})
                self.departments = dash_settings.get("AvailableDepartments", ["General Surgery"])
                ui_fallback_procedures = dash_settings.get("FallbackProcedures", ["Laparoscopic_Cholecystectomy"])
                grpc_settings = dash_data.get("gRpcSettings", {})
                self.grpc_server_address = grpc_settings.get("ServerAddress", "127.0.0.1:50051")
            else:
                self.departments = ["General Surgery"]
                ui_fallback_procedures = ["Laparoscopic_Cholecystectomy"]
                self.grpc_server_address = "127.0.0.1:50051"

            # 2. Load Pipeline configurations from configs/app.yaml
            with open(self.app_yaml_path, 'r', encoding='utf-8') as f:
                self.config_data = self.yaml_handler.load(f)
            
            # Clean duplicate keys automatically on load phase
            with open(self.app_yaml_path, 'w', encoding='utf-8') as f:
                self.yaml_handler.dump(self.config_data, f)
                
            proc_config = self.config_data.get("ProcedureConfig", {})
            self.procedures = sorted(list(proc_config.keys())) if proc_config else ui_fallback_procedures
            
        except Exception as e:
            QMessageBox.critical(self, "YAML Configuration Error", f"Failed loading YAML configurations:\n{str(e)}")
            sys.exit(1)

    def create_blanket_frame(self):
        """Helper to create a unified fully dark layout card container (Blanket)"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #1e222b;
                border: 1px solid #3e4451;
                border-radius: 8px;
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 90))
        shadow.setOffset(0, 4)
        frame.setGraphicsEffect(shadow)
        return frame

    def init_ui(self):
        self.setWindowTitle("iSurgical - Intake Management Control Dashboard")
        self.setMinimumSize(650, 620)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1c23;
            }
            QLabel {
                color: #abb2bf;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
            QLabel#header_title {
                color: #61afef;
                font-weight: bold;
                font-size: 24px;
                margin-bottom: 5px;
            }
            QLineEdit, QComboBox {
                background-color: #21252b;
                color: #ffffff;
                border: 1px solid #3e4451;
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #61afef;
                background-color: #21252b;
            }
            QLineEdit::placeholder {
                color: #5c6370;
            }
            QComboBox QAbstractItemView {
                background-color: #21252b;
                color: #ffffff;
                selection-background-color: #3e4451;
                border: 1px solid #3e4451;
            }
            QCheckBox {
                color: #abb2bf;
                font-size: 13px;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 5px;
                border: 1px solid #3e4451;
                background-color: #21252b;
            }
            QCheckBox::indicator:checked {
                background-color: #98c379;
                border: 1px solid #98c379;
            }
            QPushButton#submit_btn {
                background-color: #61afef;
                color: #1e222b;
                font-weight: bold;
                font-size: 15px;
                border: none;
                border-radius: 6px;
                padding: 14px;
            }
            QPushButton#submit_btn:hover {
                background-color: #4fa4e3;
            }
            QPushButton#submit_btn:pressed {
                background-color: #3b8ec2;
            }
        """)

        master_layout = QVBoxLayout()
        master_layout.setContentsMargins(25, 25, 25, 25)
        master_layout.setSpacing(16)

        self.title_label = QLabel("Operating Room Session Setup")
        self.title_label.setObjectName("header_title")
        master_layout.addWidget(self.title_label)

        # ── BLANKET 1: OPERATIONS & STAFF LOGISTICS ──────────────────────
        b1_frame = self.create_blanket_frame()
        b1_layout = QFormLayout(b1_frame)
        b1_layout.setContentsMargins(20, 20, 20, 20)
        b1_layout.setSpacing(14)
        
        self.or_name_input = QLineEdit()
        self.or_name_input.setPlaceholderText("e.g. OR-Integrated-04")
        b1_layout.addRow("Operating Room Name:", self.or_name_input)
        
        self.surgeons_input = QLineEdit()
        self.surgeons_input.setPlaceholderText("e.g. Dr. Mason, Dr. Vance")
        b1_layout.addRow("Surgeons Name(s):", self.surgeons_input)
        
        self.dept_combo = QComboBox()
        self.dept_combo.addItems(self.departments)
        b1_layout.addRow("Department:", self.dept_combo)
        master_layout.addWidget(b1_frame)

        # ── BLANKET 2: PATIENT & PROCEDURE SPECS ────────────────────────
        b2_frame = self.create_blanket_frame()
        b2_layout = QFormLayout(b2_frame)
        b2_layout.setContentsMargins(20, 20, 20, 20)
        b2_layout.setSpacing(14)
        
        self.patient_input = QLineEdit()
        self.patient_input.setPlaceholderText("e.g. Patient #402-A")
        b2_layout.addRow("Patient ID / Ref:", self.patient_input)
        
        self.proc_combo = QComboBox()
        self.proc_combo.addItems(self.procedures)
        b2_layout.addRow("Procedure Type:", self.proc_combo)
        master_layout.addWidget(b2_frame)

        # ── BLANKET 3: LIVE SYSTEM INFORMATION & STREAM LAYER ───────────
        b3_frame = self.create_blanket_frame()
        b3_layout = QFormLayout(b3_frame)
        b3_layout.setContentsMargins(20, 20, 20, 20)
        b3_layout.setSpacing(14)
        
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: #e5c07b; font-weight: bold; font-family: monospace; font-size: 15px;")
        b3_layout.addRow("System Time & Date:", self.time_label)
        
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_system_timestamp)
        self.clock_timer.start(1000)
        self.update_system_timestamp()
        
        self.record_checkbox = QCheckBox("Enable Async Recording Layer for this pipeline stream")
        
        # Locked onto verified binding key: 'IsRecorded'
        initial_record_state = self.config_data.get("IsRecorded", False) if hasattr(self, 'config_data') else False
        self.record_checkbox.setChecked(bool(initial_record_state))
        b3_layout.addRow("Session Recording:", self.record_checkbox)
        master_layout.addWidget(b3_frame)

        self.submit_button = QPushButton("Commit Configuration & Launch Session")
        self.submit_button.setObjectName("submit_btn")
        self.submit_button.clicked.connect(self.handle_form_submission)
        master_layout.addWidget(self.submit_button)

        outer_container = QWidget()
        outer_container.setLayout(master_layout)
        self.setCentralWidget(outer_container)

    def update_system_timestamp(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.time_label.setText(current_time)

    def handle_form_submission(self):
        if not self.or_name_input.text().strip() or not self.surgeons_input.text().strip():
            QMessageBox.warning(self, "Validation Failed", "Please populate both Operating Room Name and Surgeon configurations.")
            return

        is_record_checked = self.record_checkbox.isChecked()
        selected_procedure = self.proc_combo.currentText()

        confirmation_summary = (
            f"Operating Room: {self.or_name_input.text().strip()}\n"
            f"Surgeon(s): {self.surgeons_input.text().strip()}\n"
            f"Department: {self.dept_combo.currentText()}\n"
            f"Patient ID / Ref: {self.patient_input.text().strip()}\n"
            f"Procedure Type: {selected_procedure}\n"
            f"Session Recording: {'Enabled' if is_record_checked else 'Disabled'}"
        )
        first_confirm = QMessageBox.question(
            self, "Confirm Session Details",
            f"Please review the session details below:\n\n{confirmation_summary}\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if first_confirm != QMessageBox.Yes:
            return

        second_confirm = QMessageBox.question(
            self, "Final Confirmation",
            "This will commit the configuration and notify the phase server.\n\nAre you sure you want to launch this session?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if second_confirm != QMessageBox.Yes:
            return

        try:
            with open(self.app_yaml_path, 'r', encoding='utf-8') as f:
                current_yaml_tree = self.yaml_handler.load(f)

            # Locked target key: IsRecorded
            current_yaml_tree["IsRecorded"] = is_record_checked
            current_yaml_tree["Procedure"] = selected_procedure
            
            current_yaml_tree["SessionMetadata"] = {
                "OperatingRoom": self.or_name_input.text().strip(),
                "Surgeons": [s.strip() for s in self.surgeons_input.text().split(",")],
                "Department": self.dept_combo.currentText(),
                "PatientID": self.patient_input.text().strip(),
                "Timestamp": datetime.datetime.now().isoformat()
            }

            with open(self.app_yaml_path, 'w', encoding='utf-8') as f:
                self.yaml_handler.dump(current_yaml_tree, f)

            self.notify_grpc_server(current_yaml_tree["SessionMetadata"], selected_procedure)
            container_launched = self.launch_isurgical_container()

            QMessageBox.information(
                self, "Success",
                f"Configuration applied successfully!\n\nIsRecorded: {is_record_checked}\nProcedure: {selected_procedure}"
                + ("\n\niSurgical pipeline container is launching." if container_launched else "")
            )
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Failed modifying properties inside app.yaml file:\n{str(e)}")

    def launch_isurgical_container(self) -> bool:
        """Launches the iSurgical inference pipeline container via 'docker run', mirroring the README's Run instructions."""
        repo_root = self.app_yaml_path.parent.parent

        try:
            subprocess.run(["xhost", "+local:docker"], check=False)
        except FileNotFoundError:
            pass  # xhost not available on this host; proceed and let docker surface any display errors

        docker_cmd = [
            "docker", "run", "--rm",
            "--runtime", "nvidia",
            "--privileged",
            "--network", "host",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-e", f"DISPLAY={os.environ.get('DISPLAY', ':1')}",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
            "--device", "/dev/video0:/dev/video0",
            "-v", f"{repo_root / 'configs'}:/workspace/iSurgical/configs",
            "-v", f"{repo_root / 'data'}:/workspace/iSurgical/data",
            "-v", f"{Path.home() / '.ssh'}:/root/.ssh-host:ro",
            DOCKER_IMAGE,
            "./bin/iSurgical", "configs/app.yaml",
        ]

        try:
            subprocess.Popen(docker_cmd, cwd=str(repo_root), start_new_session=True)
            return True
        except FileNotFoundError:
            QMessageBox.critical(self, "Docker Not Found", "Could not locate the 'docker' executable on PATH.")
        except Exception as e:
            QMessageBox.critical(self, "Container Launch Failed", f"Failed to launch the iSurgical container:\n{str(e)}")
        return False

    def notify_grpc_server(self, session_metadata: dict, procedure: str):
        """Sends the submitted session info to the surgical phase gRPC server."""
        start_time_ms = int(datetime.datetime.fromisoformat(session_metadata["Timestamp"]).timestamp() * 1000)
        surgery_info = surgical_phase_pb2.SurgeryInfo(
            doctor_name=", ".join(session_metadata["Surgeons"]),
            patient_id=session_metadata["PatientID"],
            procedure=procedure,
            start_time_ms=start_time_ms,
        )

        try:
            with grpc.insecure_channel(self.grpc_server_address) as channel:
                stub = surgical_phase_pb2_grpc.SurgicalPhaseServiceStub(channel)
                response = stub.StartSurgery(surgery_info, timeout=5)
            if not response.success:
                QMessageBox.warning(self, "gRPC Notification Failed", "Phase server rejected the StartSurgery request.")
        except grpc.RpcError as e:
            QMessageBox.warning(self, "gRPC Connection Failed", f"Could not reach phase server at {self.grpc_server_address}:\n{e.details() if hasattr(e, 'details') else str(e)}")

def main():
    base_dir = Path(__file__).resolve().parents[3]
    app_config = base_dir / "configs" / "app.yaml"
    dash_config = base_dir / "configs" / "dashboard.yaml"
    
    app = QApplication(sys.argv)
    window = ORDashboard(app_config, dash_config)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()