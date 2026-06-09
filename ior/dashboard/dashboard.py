import sys
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

from server import PhaseSignalBridge, create_server

LC_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingAndCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningAndCoagulation",
    "GallbladderRetraction",
]

PHASE_DISPLAY_NAMES = {
    "Preparation": "Preparation",
    "CalotTriangleDissection": "Calot Triangle Dissection",
    "ClippingAndCutting": "Clipping & Cutting",
    "GallbladderDissection": "Gallbladder Dissection",
    "GallbladderPackaging": "Gallbladder Packaging",
    "CleaningAndCoagulation": "Cleaning & Coagulation",
    "GallbladderRetraction": "Gallbladder Retraction",
}

PHASE_COLORS = {
    "Preparation": "#4FC3F7",
    "CalotTriangleDissection": "#FFB74D",
    "ClippingAndCutting": "#E57373",
    "GallbladderDissection": "#81C784",
    "GallbladderPackaging": "#BA68C8",
    "CleaningAndCoagulation": "#FFD54F",
    "GallbladderRetraction": "#4DB6AC",
}


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def deidentify_name(name: str) -> str:
    if not name:
        return "---"
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0][0] + "***"
    return parts[0][0] + "*** " + parts[-1][0] + "***"


class PhaseCard(QFrame):
    def __init__(self, phase_key: str, parent=None):
        super().__init__(parent)
        self._phase_key = phase_key
        self._elapsed_seconds = 0.0
        self._active = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self._color_bar = QFrame()
        self._color_bar.setFixedWidth(6)
        self._color_bar.setStyleSheet(
            f"background-color: {PHASE_COLORS[phase_key]}; border-radius: 3px;"
        )
        layout.addWidget(self._color_bar)

        self._name_label = QLabel(PHASE_DISPLAY_NAMES[phase_key])
        self._name_label.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self._name_label, stretch=1)

        self._time_label = QLabel("--:--")
        self._time_label.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._time_label)

        self._update_style()

    def set_active(self, active: bool):
        self._active = active
        self._update_style()

    def set_elapsed(self, seconds: float):
        self._elapsed_seconds = seconds
        self._time_label.setText(format_duration(seconds))

    def add_elapsed(self, delta: float):
        self._elapsed_seconds += delta
        self._time_label.setText(format_duration(self._elapsed_seconds))

    def _update_style(self):
        if self._active:
            self.setStyleSheet(
                f"PhaseCard {{ background-color: {PHASE_COLORS[self._phase_key]}22; "
                f"border: 2px solid {PHASE_COLORS[self._phase_key]}; border-radius: 8px; }}"
            )
            self._name_label.setStyleSheet(f"color: {PHASE_COLORS[self._phase_key]}; font-weight: bold;")
        else:
            self.setStyleSheet(
                "PhaseCard { background-color: #2A2A2A; border: 1px solid #444; border-radius: 8px; }"
            )
            self._name_label.setStyleSheet("color: #CCC;")


class SurgeryDashboard(QMainWindow):
    def __init__(self, bridge: PhaseSignalBridge):
        super().__init__()
        self._bridge = bridge
        self._current_phase = None
        self._phase_start_time = None
        self._surgery_start_ms = None
        self._phase_cards: dict[str, PhaseCard] = {}

        self.setWindowTitle("iSurgical - Surgery Dashboard")
        self.setMinimumSize(700, 680)
        self._setup_dark_theme()
        self._build_ui()

        bridge.phase_received.connect(self._on_phase_received)
        bridge.surgery_started.connect(self._on_surgery_started)

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(1000)

    def _setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#1E1E1E"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#EEEEEE"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#252525"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#EEEEEE"))
        self.setPalette(palette)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header
        header = QLabel("Surgery Dashboard")
        header.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #EEEEEE;")
        main_layout.addWidget(header)

        # Info panel
        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background-color: #2A2A2A; border: 1px solid #444; border-radius: 8px; }"
        )
        info_layout = QGridLayout(info_frame)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setHorizontalSpacing(24)

        lbl_style = "color: #888; font-size: 10px;"
        val_style = "color: #EEE; font-size: 13px; font-weight: bold;"

        info_layout.addWidget(self._make_label("DOCTOR", lbl_style), 0, 0)
        self._doctor_label = self._make_label("---", val_style)
        info_layout.addWidget(self._doctor_label, 1, 0)

        info_layout.addWidget(self._make_label("PATIENT", lbl_style), 0, 1)
        self._patient_label = self._make_label("---", val_style)
        info_layout.addWidget(self._patient_label, 1, 1)

        info_layout.addWidget(self._make_label("PROCEDURE", lbl_style), 0, 2)
        self._procedure_label = self._make_label("---", val_style)
        info_layout.addWidget(self._procedure_label, 1, 2)

        info_layout.addWidget(self._make_label("ELAPSED", lbl_style), 0, 3)
        self._elapsed_label = self._make_label("--:--:--", val_style)
        info_layout.addWidget(self._elapsed_label, 1, 3)

        main_layout.addWidget(info_frame)

        # Current phase highlight
        self._current_phase_label = QLabel("Waiting for surgery to start...")
        self._current_phase_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._current_phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._current_phase_label.setStyleSheet("color: #4FC3F7; padding: 8px;")
        main_layout.addWidget(self._current_phase_label)

        # Phase cards
        for phase_key in LC_PHASES:
            card = PhaseCard(phase_key)
            self._phase_cards[phase_key] = card
            main_layout.addWidget(card)

        main_layout.addStretch()

    def _make_label(self, text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _on_surgery_started(self, doctor: str, patient_id: str, procedure: str, start_time_ms: int):
        self._doctor_label.setText(doctor)
        self._patient_label.setText(deidentify_name(patient_id))
        self._procedure_label.setText(procedure)
        self._surgery_start_ms = start_time_ms
        self._current_phase_label.setText("Surgery in progress...")
        self._current_phase_label.setStyleSheet("color: #81C784; padding: 8px;")

    def _on_phase_received(self, phase: str, timestamp_ms: int):
        if phase == self._current_phase:
            return

        now = time.time()

        if self._current_phase and self._current_phase in self._phase_cards:
            self._phase_cards[self._current_phase].set_active(False)

        self._current_phase = phase
        self._phase_start_time = now

        if phase in self._phase_cards:
            self._phase_cards[phase].set_active(True)
            self._current_phase_label.setText(
                f"Current Phase: {PHASE_DISPLAY_NAMES.get(phase, phase)}"
            )
            color = PHASE_COLORS.get(phase, "#FFFFFF")
            self._current_phase_label.setStyleSheet(f"color: {color}; padding: 8px;")

    def _tick(self):
        if self._surgery_start_ms:
            elapsed = time.time() - (self._surgery_start_ms / 1000.0)
            self._elapsed_label.setText(format_duration(max(0, elapsed)))

        if self._current_phase and self._phase_start_time:
            card = self._phase_cards.get(self._current_phase)
            if card:
                card.add_elapsed(1.0)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    bridge = PhaseSignalBridge()
    grpc_server = create_server(bridge, port=50051)
    grpc_server.start()

    window = SurgeryDashboard(bridge)
    window.show()

    ret = app.exec()
    grpc_server.stop(grace=2)
    sys.exit(ret)


if __name__ == "__main__":
    main()
