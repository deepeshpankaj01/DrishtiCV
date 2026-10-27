"""Main desktop window for DrishtiCV.

Constructs the complete computer-vision workspace: header, controls sidebar,
central video canvas with playback controls, analytics dashboard, and status bar.
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QIcon, QFont, QAction, QColor
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QCheckBox,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QScrollArea,
    QFrame,
    QSplitter,
    QStatusBar,
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
)

from drishticv.config import (
    APP_NAME,
    APP_FULL_TITLE,
    APP_SUBTITLE,
    APP_VERSION,
    DEVELOPER_NAME,
    APP_TAGLINE,
    APP_CATEGORY,
    MIN_WINDOW_WIDTH,
    MIN_WINDOW_HEIGHT,
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAMES,
    AVAILABLE_TRACKERS,
    COLOR_BG_MAIN,
    COLOR_BG_PANEL,
    COLOR_BG_CARD,
    COLOR_ACCENT,
    COLOR_BORDER,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_SUCCESS,
    DARK_THEME_QSS,
)
from drishticv.detection.detector import DrishtiDetector, DetectionResult
from drishticv.core.events import (
    LineCrossingTracker,
    RegionMonitor,
    LineCrossingEvent,
    RegionMonitoringEvent,
)
from drishticv.core.event_logger import SessionManager, EventLogger
from drishticv.core.recording import VideoRecorder
from drishticv.ui.video_canvas import VideoCanvas
from drishticv.ui.workers import VideoProcessingWorker, list_available_cameras

logger = logging.getLogger(__name__)


class ClassFilterDialog(QDialog):
    """Dialog to select which YOLO object classes to detect and track."""

    def __init__(self, all_classes: Dict[int, str], selected_ids: Optional[List[int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Detection Classes — DrishtiCV")
        self.setMinimumSize(360, 420)
        self.all_classes = all_classes

        layout = QVBoxLayout(self)

        lbl = QLabel("Select object classes to detect (unselected classes are ignored):")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        # Quick buttons
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Clear All")
        btn_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._clear_all)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        for cid, name in sorted(all_classes.items(), key=lambda x: x[0]):
            item = QListWidgetItem(f"{name} (ID: {cid})")
            item.setData(Qt.UserRole, cid)
            checked = True if (selected_ids is None or cid in selected_ids) else False
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def _clear_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def get_selected_classes(self) -> Optional[List[int]]:
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        if len(selected) == len(self.all_classes):
            return None  # All classes selected
        return selected


class MainWindow(QMainWindow):
    """Main desktop application window for DrishtiCV."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_FULL_TITLE)
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.setStyleSheet(DARK_THEME_QSS)

        # Core State & Systems
        self.session_manager = SessionManager()
        self.event_logger = EventLogger(self.session_manager)
        self.detector = DrishtiDetector(DEFAULT_MODEL_NAME)
        self.line_tracker = LineCrossingTracker()
        self.region_monitor = RegionMonitor()
        self.recorder = VideoRecorder()

        # Input and stream state
        self.current_source_type: str = "WEBCAM"
        self.current_source_name: str = "Webcam 0"
        self.is_streaming: bool = False
        self.selected_classes: Optional[List[int]] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[DetectionResult] = None
        self.total_frames_processed: int = 0

        # Build UI Architecture
        self._init_ui()

        # Initialize Background Worker
        self.worker = VideoProcessingWorker(
            self.detector,
            self.line_tracker,
            self.region_monitor,
            self.recorder,
        )
        self._connect_worker_signals()

        # Load initial model
        self._load_initial_model()

        # Refresh camera devices
        self._refresh_camera_devices()

    def _init_ui(self) -> None:
        """Construct central layout containing header, sidebar, canvas, and analytics."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)

        # 1. Header Area
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # 2. Main Content Splitter (Left Sidebar | Center Canvas | Right Analytics)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        sidebar = self._create_sidebar()
        center_area = self._create_center_area()
        analytics_panel = self._create_analytics_panel()

        splitter.addWidget(sidebar)
        splitter.addWidget(center_area)
        splitter.addWidget(analytics_panel)

        # Proportions: Sidebar ~280px, Canvas ~580px, Analytics ~280px
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        main_layout.addWidget(splitter, 1)

        # 3. Status Bar
        self._create_status_bar()

    # Layout Builders
    def _create_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet(
            f"background-color: {COLOR_BG_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 6px 12px;"
        )
        layout = QHBoxLayout(header)
        layout.setContentsMargins(4, 2, 4, 2)

        # Brand / Title
        title_box = QVBoxLayout()
        title_lbl = QLabel(f"<b>{APP_NAME}</b> <span style='color: {COLOR_ACCENT}; font-size: 11px;'>v{APP_VERSION}</span>")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFFFF;")
        sub_lbl = QLabel(f"{APP_SUBTITLE} — {APP_TAGLINE}")
        sub_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        layout.addLayout(title_box)

        layout.addStretch(1)

        # Developer Credit
        dev_lbl = QLabel(f"Developer: <b>{DEVELOPER_NAME}</b>")
        dev_lbl.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_MAIN}; background-color: {COLOR_BG_CARD}; padding: 4px 10px; border-radius: 4px; border: 1px solid {COLOR_BORDER};")
        layout.addWidget(dev_lbl)

        # Status Pill Badge
        self.status_pill = QLabel("IDLE")
        self.status_pill.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {COLOR_BG_MAIN}; background-color: {COLOR_TEXT_SECONDARY}; padding: 4px 12px; border-radius: 10px;"
        )
        layout.addWidget(self.status_pill)

        # Session Folder Button
        btn_session = QPushButton("Open Session Folder")
        btn_session.setStyleSheet(f"font-size: 11px; padding: 4px 10px;")
        btn_session.clicked.connect(self._open_session_folder)
        layout.addWidget(btn_session)

        return header

    def _create_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(360)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        # A. Source Selection Tabs
        source_group = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        self.source_tabs = QTabWidget()
        self.source_tabs.currentChanged.connect(self._on_source_tab_changed)

        # Tab 1: Webcam
        tab_webcam = QWidget()
        w_layout = QVBoxLayout(tab_webcam)
        w_layout.setContentsMargins(4, 8, 4, 4)

        cam_row = QHBoxLayout()
        self.cam_combo = QComboBox()
        btn_refresh_cam = QPushButton("Refresh")
        btn_refresh_cam.setMaximumWidth(70)
        btn_refresh_cam.clicked.connect(self._refresh_camera_devices)
        cam_row.addWidget(self.cam_combo, 1)
        cam_row.addWidget(btn_refresh_cam)
        w_layout.addLayout(cam_row)

        self.btn_webcam_stream = QPushButton("Start Webcam")
        self.btn_webcam_stream.setObjectName("accentButton")
        self.btn_webcam_stream.clicked.connect(self._toggle_webcam_stream)
        w_layout.addWidget(self.btn_webcam_stream)
        self.source_tabs.addTab(tab_webcam, "Webcam")

        # Tab 2: Video File
        tab_video = QWidget()
        v_layout = QVBoxLayout(tab_video)
        v_layout.setContentsMargins(4, 8, 4, 4)

        btn_browse_vid = QPushButton("Select Video File...")
        btn_browse_vid.clicked.connect(self._browse_video_file)
        self.lbl_selected_video = QLabel("No video selected")
        self.lbl_selected_video.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.lbl_selected_video.setWordWrap(True)

        self.btn_video_stream = QPushButton("Play Video")
        self.btn_video_stream.setObjectName("accentButton")
        self.btn_video_stream.setEnabled(False)
        self.btn_video_stream.clicked.connect(self._toggle_video_stream)

        v_layout.addWidget(btn_browse_vid)
        v_layout.addWidget(self.lbl_selected_video)
        v_layout.addWidget(self.btn_video_stream)
        self.source_tabs.addTab(tab_video, "Video File")

        # Tab 3: Single Image
        tab_image = QWidget()
        i_layout = QVBoxLayout(tab_image)
        i_layout.setContentsMargins(4, 8, 4, 4)

        btn_browse_img = QPushButton("Select Image...")
        btn_browse_img.clicked.connect(self._browse_image_file)
        self.lbl_selected_image = QLabel("No image selected")
        self.lbl_selected_image.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.lbl_selected_image.setWordWrap(True)

        self.btn_run_image = QPushButton("Detect Objects")
        self.btn_run_image.setObjectName("accentButton")
        self.btn_run_image.setEnabled(False)
        self.btn_run_image.clicked.connect(self._run_image_detection)

        i_layout.addWidget(btn_browse_img)
        i_layout.addWidget(self.lbl_selected_image)
        i_layout.addWidget(self.btn_run_image)
        self.source_tabs.addTab(tab_image, "Image")

        source_layout.addWidget(self.source_tabs)
        layout.addWidget(source_group)

        # B. Model Selection & Loading
        model_group = QGroupBox("Object Detection Model")
        model_layout = QVBoxLayout(model_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems([DEFAULT_MODEL_NAME] + FALLBACK_MODEL_NAMES)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)

        btn_custom_model = QPushButton("Browse Custom .pt File...")
        btn_custom_model.clicked.connect(self._browse_custom_model)
        model_layout.addWidget(btn_custom_model)

        self.lbl_model_status = QLabel("Checking model...")
        self.lbl_model_status.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px;")
        self.lbl_model_status.setWordWrap(True)
        model_layout.addWidget(self.lbl_model_status)

        layout.addWidget(model_group)

        # C. Inference & Tracking Settings
        settings_group = QGroupBox("Detection & Tracking Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Confidence Slider
        conf_row = QHBoxLayout()
        conf_lbl = QLabel("Confidence:")
        self.lbl_conf_val = QLabel("40%")
        self.lbl_conf_val.setStyleSheet(f"color: {COLOR_ACCENT}; font-weight: bold;")
        conf_row.addWidget(conf_lbl)
        conf_row.addStretch(1)
        conf_row.addWidget(self.lbl_conf_val)
        settings_layout.addLayout(conf_row)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 95)
        self.slider_conf.setValue(40)
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        settings_layout.addWidget(self.slider_conf)

        # Tracking Checkbox & Tracker selector
        self.chk_tracking = QCheckBox("Enable Multi-Object Tracking")
        self.chk_tracking.setChecked(True)
        self.chk_tracking.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.chk_tracking)

        tracker_row = QHBoxLayout()
        tracker_row.addWidget(QLabel("Tracker:"))
        self.combo_tracker = QComboBox()
        for name in AVAILABLE_TRACKERS.keys():
            self.combo_tracker.addItem(name)
        self.combo_tracker.currentTextChanged.connect(self._on_settings_changed)
        tracker_row.addWidget(self.combo_tracker, 1)
        settings_layout.addLayout(tracker_row)

        # Movement Trails
        self.chk_trails = QCheckBox("Show Movement Trails")
        self.chk_trails.setChecked(True)
        self.chk_trails.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.chk_trails)

        # Annotations options
        self.chk_labels = QCheckBox("Show Object Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.chk_labels)

        self.chk_scores = QCheckBox("Show Confidence Scores")
        self.chk_scores.setChecked(True)
        self.chk_scores.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.chk_scores)

        # Class Filter Dialog Button
        btn_filter_classes = QPushButton("Filter Detection Classes...")
        btn_filter_classes.clicked.connect(self._open_class_filter_dialog)
        settings_layout.addWidget(btn_filter_classes)
        self.lbl_filter_summary = QLabel("All classes enabled")
        self.lbl_filter_summary.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        settings_layout.addWidget(self.lbl_filter_summary)

        layout.addWidget(settings_group)

        # D. Virtual Line & Monitored Zone Setup
        analytics_setup_group = QGroupBox("Scene Analytics Setup")
        a_setup_layout = QVBoxLayout(analytics_setup_group)

        # Line Controls
        line_row = QHBoxLayout()
        self.btn_set_line = QPushButton("Define Line (2 Clicks)")
        self.btn_set_line.clicked.connect(self._start_line_setup)
        self.btn_clear_line = QPushButton("Clear")
        self.btn_clear_line.setMaximumWidth(60)
        self.btn_clear_line.clicked.connect(self._clear_line)
        line_row.addWidget(self.btn_set_line, 1)
        line_row.addWidget(self.btn_clear_line)
        a_setup_layout.addLayout(line_row)

        self.lbl_line_status = QLabel("Virtual line: Not configured")
        self.lbl_line_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        a_setup_layout.addWidget(self.lbl_line_status)

        # Region Controls
        region_row = QHBoxLayout()
        self.btn_set_region = QPushButton("Define Monitored Zone")
        self.btn_set_region.clicked.connect(self._start_region_setup)
        self.btn_clear_region = QPushButton("Clear")
        self.btn_clear_region.setMaximumWidth(60)
        self.btn_clear_region.clicked.connect(self._clear_region)
        region_row.addWidget(self.btn_set_region, 1)
        region_row.addWidget(self.btn_clear_region)
        a_setup_layout.addLayout(region_row)

        self.lbl_region_status = QLabel("Monitored zone: Not configured")
        self.lbl_region_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        a_setup_layout.addWidget(self.lbl_region_status)

        layout.addWidget(analytics_setup_group)

        # E. Capture & Recording Actions
        capture_group = QGroupBox("Capture & Recording")
        cap_layout = QVBoxLayout(capture_group)

        self.btn_snapshot = QPushButton("Save Snapshot")
        self.btn_snapshot.clicked.connect(self._save_snapshot)
        cap_layout.addWidget(self.btn_snapshot)

        self.btn_record = QPushButton("Start Recording Video")
        self.btn_record.clicked.connect(self._toggle_recording)
        cap_layout.addWidget(self.btn_record)

        self.lbl_record_status = QLabel("Recorder idle")
        self.lbl_record_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        cap_layout.addWidget(self.lbl_record_status)

        layout.addWidget(capture_group)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _create_center_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        # Video Canvas
        self.canvas = VideoCanvas()
        self.canvas.line_defined.connect(self._on_line_defined)
        self.canvas.region_defined.connect(self._on_region_defined)
        self.canvas.interaction_cancelled.connect(self._on_interaction_cancelled)
        layout.addWidget(self.canvas, 1)

        # Video Playback Controls Bar
        self.playback_bar = QFrame()
        self.playback_bar.setStyleSheet(
            f"background-color: {COLOR_BG_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 2px 8px;"
        )
        bar_layout = QHBoxLayout(self.playback_bar)
        bar_layout.setContentsMargins(4, 2, 4, 2)

        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.setMaximumWidth(70)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        bar_layout.addWidget(self.btn_play_pause)

        self.btn_stop_stream = QPushButton("Stop")
        self.btn_stop_stream.setMaximumWidth(60)
        self.btn_stop_stream.setObjectName("dangerButton")
        self.btn_stop_stream.clicked.connect(self._stop_current_stream)
        bar_layout.addWidget(self.btn_stop_stream)

        # Seek Slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider_moved)
        bar_layout.addWidget(self.seek_slider, 1)

        # Frame/Time Indicator
        self.lbl_time_code = QLabel("00:00 / 00:00")
        self.lbl_time_code.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; min-width: 90px;")
        bar_layout.addWidget(self.lbl_time_code)

        layout.addWidget(self.playback_bar)
        self.playback_bar.setVisible(False)  # Hidden when idle or in image mode

        return container

    def _create_analytics_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(360)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(12)

        # A. Performance & Source Card
        perf_group = QGroupBox("Stream & Processing Metrics")
        perf_layout = QGridLayout(perf_group)
        perf_layout.setSpacing(6)

        perf_layout.addWidget(QLabel("Source:"), 0, 0)
        self.lbl_m_source = QLabel("Idle")
        self.lbl_m_source.setStyleSheet(f"color: {COLOR_TEXT_MAIN}; font-weight: bold;")
        perf_layout.addWidget(self.lbl_m_source, 0, 1)

        perf_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.lbl_m_res = QLabel("—")
        perf_layout.addWidget(self.lbl_m_res, 1, 1)

        perf_layout.addWidget(QLabel("Processing FPS:"), 2, 0)
        self.lbl_m_fps = QLabel("0.0 FPS")
        self.lbl_m_fps.setStyleSheet(f"color: {COLOR_ACCENT}; font-weight: bold;")
        perf_layout.addWidget(self.lbl_m_fps, 2, 1)

        perf_layout.addWidget(QLabel("Inference Latency:"), 3, 0)
        self.lbl_m_latency = QLabel("0.0 ms")
        perf_layout.addWidget(self.lbl_m_latency, 3, 1)

        perf_layout.addWidget(QLabel("Frames Processed:"), 4, 0)
        self.lbl_m_frames = QLabel("0")
        perf_layout.addWidget(self.lbl_m_frames, 4, 1)

        layout.addWidget(perf_group)

        # B. Detections & Tracking (Current Frame)
        obj_group = QGroupBox("Frame Detections & Active Tracks")
        obj_layout = QGridLayout(obj_group)
        obj_layout.setSpacing(6)

        obj_layout.addWidget(QLabel("Current Detections:"), 0, 0)
        self.lbl_curr_detections = QLabel("0")
        self.lbl_curr_detections.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        obj_layout.addWidget(self.lbl_curr_detections, 0, 1)

        obj_layout.addWidget(QLabel("Active Tracked IDs:"), 1, 0)
        self.lbl_active_tracks = QLabel("0")
        self.lbl_active_tracks.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {COLOR_ACCENT};")
        obj_layout.addWidget(self.lbl_active_tracks, 1, 1)

        note_lbl = QLabel("(Detections = this frame; Tracks = persistent IDs across frames)")
        note_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px;")
        note_lbl.setWordWrap(True)
        obj_layout.addWidget(note_lbl, 2, 0, 1, 2)

        layout.addWidget(obj_group)

        # C. Cumulative Scene Metrics (Virtual Line & Monitored Zone)
        counts_group = QGroupBox("Cumulative Scene Analytics")
        counts_layout = QGridLayout(counts_group)
        counts_layout.setSpacing(6)

        counts_layout.addWidget(QLabel("Inbound Crossings (A→B):"), 0, 0)
        self.lbl_cross_in = QLabel("0")
        self.lbl_cross_in.setStyleSheet(f"font-weight: bold; color: {COLOR_ACCENT};")
        counts_layout.addWidget(self.lbl_cross_in, 0, 1)

        counts_layout.addWidget(QLabel("Outbound Crossings (B→A):"), 1, 0)
        self.lbl_cross_out = QLabel("0")
        self.lbl_cross_out.setStyleSheet(f"font-weight: bold; color: {COLOR_WARNING};")
        counts_layout.addWidget(self.lbl_cross_out, 1, 1)

        counts_layout.addWidget(QLabel("Total Line Crossings:"), 2, 0)
        self.lbl_cross_total = QLabel("0")
        self.lbl_cross_total.setStyleSheet("font-weight: bold; color: #FFFFFF;")
        counts_layout.addWidget(self.lbl_cross_total, 2, 1)

        counts_layout.addWidget(QLabel("Monitored Zone Entries:"), 3, 0)
        self.lbl_zone_entries = QLabel("0")
        self.lbl_zone_entries.setStyleSheet(f"font-weight: bold; color: {COLOR_ACCENT};")
        counts_layout.addWidget(self.lbl_zone_entries, 3, 1)

        btn_reset_counts = QPushButton("Reset Cumulative Counters")
        btn_reset_counts.setStyleSheet("font-size: 11px;")
        btn_reset_counts.clicked.connect(self._reset_cumulative_counts)
        counts_layout.addWidget(btn_reset_counts, 4, 0, 1, 2)

        layout.addWidget(counts_group)

        # D. Detected Classes Breakdown Table
        class_group = QGroupBox("Detected Classes Breakdown")
        class_layout = QVBoxLayout(class_group)

        self.table_classes = QTableWidget(0, 2)
        self.table_classes.setHorizontalHeaderLabels(["Class", "Count"])
        self.table_classes.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_classes.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.setMaximumHeight(140)
        class_layout.addWidget(self.table_classes)

        layout.addWidget(class_group)

        # E. Recent Scene Events Table
        events_group = QGroupBox("Recent Scene Events")
        events_layout = QVBoxLayout(events_group)

        self.table_events = QTableWidget(0, 4)
        self.table_events.setHorizontalHeaderLabels(["Time", "Event", "ID", "Class"])
        self.table_events.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_events.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_events.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_events.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_events.verticalHeader().setVisible(False)
        self.table_events.setMaximumHeight(180)
        events_layout.addWidget(self.table_events)

        layout.addWidget(events_group)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _create_status_bar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.sb_model = QLabel(f"Model: {DEFAULT_MODEL_NAME}")
        self.sb_source = QLabel("Source: Idle")
        self.sb_rec = QLabel("Rec: Inactive")
        self.sb_dir = QLabel(f"Output: {self.session_manager.session_dir.name}")

        self.status_bar.addWidget(self.sb_model, 1)
        self.status_bar.addWidget(self.sb_source, 1)
        self.status_bar.addWidget(self.sb_rec, 1)
        self.status_bar.addWidget(self.sb_dir, 2)

    # Worker Signal Connections
    def _connect_worker_signals(self) -> None:
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.source_started.connect(self._on_source_started)
        self.worker.source_finished.connect(self._on_source_finished)
        self.worker.source_error.connect(self._on_source_error)
        self.worker.crossing_event_occurred.connect(self._on_crossing_event)
        self.worker.region_event_occurred.connect(self._on_region_event)
        self.worker.status_changed.connect(self._on_worker_status_changed)

    # Model Loading Logic
    def _load_initial_model(self) -> None:
        """Load default YOLO26 checkpoint."""
        self.lbl_model_status.setText("Loading model...")
        self.status_pill.setText("LOADING MODEL")
        self.status_pill.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: #0B1015; background-color: {COLOR_WARNING}; padding: 4px 12px; border-radius: 10px;"
        )

        success, msg = self.detector.load_model(DEFAULT_MODEL_NAME)
        if success:
            self.lbl_model_status.setText(f"Active: {DEFAULT_MODEL_NAME} ({len(self.detector.class_names)} classes)")
            self.lbl_model_status.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 11px;")
            self.sb_model.setText(f"Model: {DEFAULT_MODEL_NAME}")
            self.status_pill.setText("READY")
            self.status_pill.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: #0B1015; background-color: {COLOR_ACCENT}; padding: 4px 12px; border-radius: 10px;"
            )
        else:
            self.lbl_model_status.setText("Failed to load model.")
            self.lbl_model_status.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 11px;")
            self.status_pill.setText("MODEL ERROR")
            self.status_pill.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: #FFFFFF; background-color: {COLOR_ERROR}; padding: 4px 12px; border-radius: 10px;"
            )
            QMessageBox.critical(self, "Model Load Error", msg)

    def _on_model_changed(self, model_name: str) -> None:
        if not model_name:
            return
        was_running = self.worker.isRunning()
        if was_running:
            self.worker.stop()

        self.lbl_model_status.setText(f"Loading {model_name}...")
        success, msg = self.detector.load_model(model_name)
        if success:
            self.lbl_model_status.setText(f"Active: {model_name} ({len(self.detector.class_names)} classes)")
            self.lbl_model_status.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 11px;")
            self.sb_model.setText(f"Model: {model_name}")
            # Reset class filter if model changed
            self.selected_classes = None
            self.lbl_filter_summary.setText("All classes enabled")
        else:
            self.lbl_model_status.setText(f"Failed to load {model_name}")
            self.lbl_model_status.setStyleSheet(f"color: {COLOR_ERROR}; font-size: 11px;")
            QMessageBox.warning(self, "Model Switch Failed", msg)

        if was_running:
            self.worker.start()

    def _browse_custom_model(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom YOLO Weights (.pt)",
            str(self.session_manager.base_dir.parent),
            "PyTorch Weights (*.pt);;All Files (*.*)",
        )
        if filepath:
            filename = Path(filepath).name
            if self.model_combo.findText(filename) == -1:
                self.model_combo.addItem(filename)
            self.model_combo.setCurrentText(filename)
            success, msg = self.detector.load_model(filepath)
            if success:
                self.lbl_model_status.setText(f"Active Custom: {filename} ({len(self.detector.class_names)} classes)")
                self.lbl_model_status.setStyleSheet(f"color: {COLOR_SUCCESS}; font-size: 11px;")
                self.sb_model.setText(f"Model: {filename}")
                self.selected_classes = None
                self.lbl_filter_summary.setText("All classes enabled")
                QMessageBox.information(self, "Custom Model Loaded", f"Successfully loaded custom model: {filename}")
            else:
                QMessageBox.critical(self, "Custom Model Load Failed", msg)

    # Camera Device Management
    def _refresh_camera_devices(self) -> None:
        self.cam_combo.clear()
        devices = list_available_cameras(max_to_test=4)
        if devices:
            for idx in devices:
                self.cam_combo.addItem(f"Webcam Device {idx}", idx)
            self.btn_webcam_stream.setEnabled(True)
        else:
            self.cam_combo.addItem("No cameras found", -1)
            self.btn_webcam_stream.setEnabled(False)

    # Input Mode Actions
    def _on_source_tab_changed(self, index: int) -> None:
        tab_names = ["WEBCAM", "VIDEO", "IMAGE"]
        self.current_source_type = tab_names[index]

    def _toggle_webcam_stream(self) -> None:
        if self.worker.isRunning() and self.worker.source_mode == "WEBCAM":
            self._stop_current_stream()
            return

        cam_idx = self.cam_combo.currentData()
        if cam_idx is None or cam_idx < 0:
            QMessageBox.warning(self, "No Webcam", "Please select a valid webcam device.")
            return

        self._stop_current_stream()
        self.worker.set_source_webcam(cam_idx)
        self.current_source_name = f"Webcam #{cam_idx}"
        self._sync_worker_settings()
        self.btn_webcam_stream.setText("Stop Webcam")
        self.btn_webcam_stream.setObjectName("dangerButton")
        self.btn_webcam_stream.setStyle(self.btn_webcam_stream.style())
        self.playback_bar.setVisible(True)
        self.seek_slider.setEnabled(False)  # Live stream cannot seek
        self.worker.start()

    def _browse_video_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)",
        )
        if filepath:
            self.lbl_selected_video.setText(Path(filepath).name)
            self.btn_video_stream.setEnabled(True)
            self.current_video_path = filepath

    def _toggle_video_stream(self) -> None:
        if self.worker.isRunning() and self.worker.source_mode == "VIDEO":
            self._stop_current_stream()
            return

        if not hasattr(self, "current_video_path") or not os.path.isfile(self.current_video_path):
            QMessageBox.warning(self, "Video Error", "Selected video file does not exist.")
            return

        self._stop_current_stream()
        self.worker.set_source_video(self.current_video_path)
        self.current_source_name = Path(self.current_video_path).name
        self._sync_worker_settings()
        self.btn_video_stream.setText("Stop Video")
        self.btn_video_stream.setObjectName("dangerButton")
        self.btn_video_stream.setStyle(self.btn_video_stream.style())
        self.playback_bar.setVisible(True)
        self.seek_slider.setEnabled(True)
        self.worker.start()

    def _browse_image_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*.*)",
        )
        if filepath:
            self.lbl_selected_image.setText(Path(filepath).name)
            self.btn_run_image.setEnabled(True)
            self.current_image_path = filepath

    def _run_image_detection(self) -> None:
        if not hasattr(self, "current_image_path") or not os.path.isfile(self.current_image_path):
            QMessageBox.warning(self, "Image Error", "Selected image file does not exist.")
            return

        self._stop_current_stream()
        self.worker.set_source_image(self.current_image_path)
        self.current_source_name = Path(self.current_image_path).name
        self._sync_worker_settings()
        self.playback_bar.setVisible(False)
        self.worker.start()

    def _stop_current_stream(self) -> None:
        if self.recorder.is_recording:
            self._toggle_recording()

        if self.worker.isRunning():
            self.worker.stop()

        self.btn_webcam_stream.setText("Start Webcam")
        self.btn_webcam_stream.setObjectName("accentButton")
        self.btn_webcam_stream.setStyle(self.btn_webcam_stream.style())

        self.btn_video_stream.setText("Play Video")
        self.btn_video_stream.setObjectName("accentButton")
        self.btn_video_stream.setStyle(self.btn_video_stream.style())

        self.playback_bar.setVisible(False)
        self.status_pill.setText("IDLE")
        self.status_pill.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {COLOR_BG_MAIN}; background-color: {COLOR_TEXT_SECONDARY}; padding: 4px 12px; border-radius: 10px;"
        )
        self.sb_source.setText("Source: Idle")

    def _toggle_play_pause(self) -> None:
        if not self.worker.isRunning():
            return
        if self.worker.is_paused():
            self.worker.resume()
            self.btn_play_pause.setText("Pause")
        else:
            self.worker.pause()
            self.btn_play_pause.setText("Resume")

    def _on_seek_slider_moved(self, value: int) -> None:
        if self.worker.isRunning() and self.worker.source_mode == "VIDEO":
            self.worker.request_seek(value)

    # Worker Settings Sync
    def _sync_worker_settings(self) -> None:
        conf = self.slider_conf.value() / 100.0
        classes = self.selected_classes
        tracking_on = self.chk_tracking.isChecked()
        tracker_key = self.combo_tracker.currentText()
        tracker_yaml = AVAILABLE_TRACKERS.get(tracker_key, "bytetrack.yaml")
        draw_trails = self.chk_trails.isChecked()
        draw_labels = self.chk_labels.isChecked()
        draw_conf = self.chk_scores.isChecked()
        draw_tracks = self.chk_tracking.isChecked()

        self.worker.update_settings(
            conf_threshold=conf,
            allowed_classes=classes,
            tracking_enabled=tracking_on,
            tracker_type=tracker_yaml,
            draw_trails=draw_trails,
            draw_labels=draw_labels,
            draw_conf=draw_conf,
            draw_tracks=draw_tracks,
        )

    def _on_conf_changed(self, value: int) -> None:
        self.lbl_conf_val.setText(f"{value}%")
        self._sync_worker_settings()

    def _on_settings_changed(self) -> None:
        self._sync_worker_settings()

    def _open_class_filter_dialog(self) -> None:
        if not self.detector.class_names:
            QMessageBox.information(self, "Classes", "No model classes loaded yet.")
            return

        dlg = ClassFilterDialog(self.detector.class_names, self.selected_classes, self)
        if dlg.exec() == QDialog.Accepted:
            self.selected_classes = dlg.get_selected_classes()
            if self.selected_classes is None:
                self.lbl_filter_summary.setText("All classes enabled")
            else:
                self.lbl_filter_summary.setText(f"{len(self.selected_classes)} of {len(self.detector.class_names)} classes enabled")
            self._sync_worker_settings()

    # Scene Analytics Setup Actions
    def _start_line_setup(self) -> None:
        self.canvas.start_line_setup()
        self.lbl_line_status.setText("Click start and end points on canvas...")
        self.lbl_line_status.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 11px;")

    def _clear_line(self) -> None:
        self.line_tracker.clear_line()
        self.canvas.set_virtual_line(None)
        self.lbl_line_status.setText("Virtual line: Not configured")
        self.lbl_line_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

    def _on_line_defined(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        self.line_tracker.set_line(p1, p2)
        self.canvas.set_virtual_line((p1, p2))
        self.lbl_line_status.setText(f"Line set: ({int(p1[0])},{int(p1[1])}) → ({int(p2[0])},{int(p2[1])})")
        self.lbl_line_status.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 11px;")
        self.event_logger.log_event(
            event_type="LINE_CONFIGURED",
            source_name=self.current_source_name,
            details=f"Coordinates: ({int(p1[0])},{int(p1[1])}) -> ({int(p2[0])},{int(p2[1])})",
        )

    def _start_region_setup(self) -> None:
        self.canvas.start_region_setup()
        self.lbl_region_status.setText("Click and drag box on canvas...")
        self.lbl_region_status.setStyleSheet(f"color: {COLOR_WARNING}; font-size: 11px;")

    def _clear_region(self) -> None:
        self.region_monitor.clear_region()
        self.canvas.set_monitored_region(None)
        self.lbl_region_status.setText("Monitored zone: Not configured")
        self.lbl_region_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

    def _on_region_defined(self, rect: Tuple[float, float, float, float]) -> None:
        self.region_monitor.set_region(rect)
        self.canvas.set_monitored_region(rect)
        x1, y1, x2, y2 = map(int, rect)
        self.lbl_region_status.setText(f"Zone set: [{x1},{y1}] to [{x2},{y2}]")
        self.lbl_region_status.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 11px;")
        self.event_logger.log_event(
            event_type="REGION_CONFIGURED",
            source_name=self.current_source_name,
            details=f"Bounding box: [{x1},{y1}] to [{x2},{y2}]",
        )

    def _on_interaction_cancelled(self) -> None:
        if self.line_tracker.is_configured():
            assert self.line_tracker.line_p1 and self.line_tracker.line_p2
            p1, p2 = self.line_tracker.line_p1, self.line_tracker.line_p2
            self.lbl_line_status.setText(f"Line set: ({int(p1[0])},{int(p1[1])}) → ({int(p2[0])},{int(p2[1])})")
        else:
            self.lbl_line_status.setText("Virtual line: Not configured")

        if self.region_monitor.is_configured():
            assert self.region_monitor.region_rect
            x1, y1, x2, y2 = map(int, self.region_monitor.region_rect)
            self.lbl_region_status.setText(f"Zone set: [{x1},{y1}] to [{x2},{y2}]")
        else:
            self.lbl_region_status.setText("Monitored zone: Not configured")

    def _reset_cumulative_counts(self) -> None:
        self.line_tracker.reset_all()
        self.region_monitor.reset_all()
        self.lbl_cross_in.setText("0")
        self.lbl_cross_out.setText("0")
        self.lbl_cross_total.setText("0")
        self.lbl_zone_entries.setText("0")
        self.event_logger.log_event("COUNTS_RESET", source_name=self.current_source_name, details="User initiated")

    # Capture & Recording
    def _save_snapshot(self) -> None:
        if self.latest_frame is None:
            QMessageBox.warning(self, "Snapshot Error", "No frame available to save.")
            return

        dest = self.session_manager.generate_snapshot_path("snapshot")
        success, res_path = VideoRecorder.save_snapshot(self.latest_frame, dest)
        if success:
            self.event_logger.log_snapshot_saved(res_path, self.current_source_name)
            self._add_event_to_table(
                time.strftime("%H:%M:%S"),
                "SNAPSHOT",
                "—",
                Path(res_path).name,
            )
            self.status_bar.showMessage(f"Snapshot saved: {Path(res_path).name}", 4000)
        else:
            QMessageBox.critical(self, "Snapshot Failed", res_path)

    def _toggle_recording(self) -> None:
        if not self.recorder.is_recording:
            if self.latest_frame is None:
                QMessageBox.warning(self, "Record Error", "Cannot start recording: No active video feed.")
                return

            h, w = self.latest_frame.shape[:2]
            dest = self.session_manager.generate_recording_path("recording", ".mp4")
            success, msg = self.recorder.start_recording(dest, w, h, fps=30.0)
            if success:
                self.btn_record.setText("Stop Recording")
                self.btn_record.setObjectName("dangerButton")
                self.btn_record.setStyle(self.btn_record.style())
                self.lbl_record_status.setText(f"Recording: {dest.name}")
                self.lbl_record_status.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: bold;")
                self.sb_rec.setText(f"Rec: {dest.name}")
                self.sb_rec.setStyleSheet(f"color: {COLOR_ERROR}; font-weight: bold;")
                self.event_logger.log_recording_started(str(dest), self.current_source_name)
            else:
                QMessageBox.critical(self, "Recording Failed to Start", msg)
        else:
            success, path, duration, frames = self.recorder.stop_recording()
            self.btn_record.setText("Start Recording Video")
            self.btn_record.setObjectName("")
            self.btn_record.setStyle(self.btn_record.style())
            self.lbl_record_status.setText("Recorder idle")
            self.lbl_record_status.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
            self.sb_rec.setText("Rec: Inactive")
            self.sb_rec.setStyleSheet("")

            if success and path:
                self.event_logger.log_recording_stopped(str(path), duration, frames)
                self.status_bar.showMessage(f"Recording saved: {path.name} ({duration:.1f}s)", 5000)

    def _open_session_folder(self) -> None:
        self.session_manager.open_session_directory_in_explorer()

    # Worker Event Handlers
    @Slot(object, object, dict)
    def _on_frame_ready(self, frame: np.ndarray, result: DetectionResult, metrics: dict) -> None:
        self.latest_frame = frame
        self.latest_result = result
        self.total_frames_processed = metrics.get("frame_index", 0)

        # Update Video Canvas
        self.canvas.set_frame(frame)

        # Update Analytics Cards
        self.lbl_m_fps.setText(f"{metrics['fps']:.1f} FPS")
        self.lbl_m_latency.setText(f"{metrics['inference_ms']:.1f} ms")
        self.lbl_m_frames.setText(str(metrics["frame_index"]))
        self.lbl_curr_detections.setText(str(metrics["object_count"]))
        self.lbl_active_tracks.setText(str(metrics["tracked_count"]))

        # Update Video Seek Bar
        tot_frames = metrics.get("total_frames", 0)
        curr_frame = metrics.get("frame_index", 0)
        if tot_frames > 0:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setRange(0, tot_frames)
            self.seek_slider.setValue(curr_frame)
            self.seek_slider.blockSignals(False)

            sec_curr = int(curr_frame / 30.0)
            sec_tot = int(tot_frames / 30.0)
            self.lbl_time_code.setText(f"{sec_curr//60:02d}:{sec_curr%60:02d} / {sec_tot//60:02d}:{sec_tot%60:02d}")

        # Update Classes Breakdown Table
        counts = result.counts_by_class
        self.table_classes.setRowCount(len(counts))
        for row, (c_name, c_cnt) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
            item_name = QTableWidgetItem(c_name)
            item_count = QTableWidgetItem(str(c_cnt))
            item_count.setTextAlignment(Qt.AlignCenter)
            self.table_classes.setItem(row, 0, item_name)
            self.table_classes.setItem(row, 1, item_count)

    @Slot(str, int, int, float)
    def _on_source_started(self, name: str, w: int, h: int, fps: float) -> None:
        self.lbl_m_source.setText(name)
        self.lbl_m_res.setText(f"{w} × {h}")
        self.sb_source.setText(f"Source: {name} ({w}x{h})")
        self.status_pill.setText("STREAMING")
        self.status_pill.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: #0B1015; background-color: {COLOR_ACCENT}; padding: 4px 12px; border-radius: 10px;"
        )
        self.event_logger.log_source_started(name, f"{w}x{h} @ {fps:.1f}fps")

    @Slot()
    def _on_source_finished(self) -> None:
        self._stop_current_stream()
        self.status_bar.showMessage("End of stream reached.", 4000)

    @Slot(str, str)
    def _on_source_error(self, err_msg: str, hint: str) -> None:
        self._stop_current_stream()
        full_text = f"{err_msg}\n\nHint: {hint}"
        self.canvas.set_error_state(full_text)
        self.status_pill.setText("ERROR")
        self.status_pill.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: #FFFFFF; background-color: {COLOR_ERROR}; padding: 4px 12px; border-radius: 10px;"
        )
        self.event_logger.log_error("SOURCE", err_msg, self.current_source_name)

    @Slot(object)
    def _on_crossing_event(self, evt: LineCrossingEvent) -> None:
        # Update Cumulative UI Counts
        self.lbl_cross_in.setText(str(self.line_tracker.count_a_to_b))
        self.lbl_cross_out.setText(str(self.line_tracker.count_b_to_a))
        self.lbl_cross_total.setText(str(self.line_tracker.count_a_to_b + self.line_tracker.count_b_to_a))

        # Log CSV Event
        self.event_logger.log_line_crossing(evt, self.current_source_name)

        # Add to recent events table
        self._add_event_to_table(
            time.strftime("%H:%M:%S", time.localtime(evt.timestamp)),
            evt.human_direction,
            f"#{evt.track_id}",
            evt.class_name,
        )

    @Slot(object)
    def _on_region_event(self, evt: RegionMonitoringEvent) -> None:
        # Update Cumulative UI Counts
        self.lbl_zone_entries.setText(str(self.region_monitor.total_entries))

        # Log CSV Event
        self.event_logger.log_region_event(evt, self.current_source_name)

        # Add to recent events table
        self._add_event_to_table(
            time.strftime("%H:%M:%S", time.localtime(evt.timestamp)),
            evt.event_type,
            f"#{evt.track_id}",
            evt.class_name,
        )

    @Slot(str)
    def _on_worker_status_changed(self, status: str) -> None:
        if status == "STREAMING":
            self.status_pill.setText("STREAMING")
            self.status_pill.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: #0B1015; background-color: {COLOR_ACCENT}; padding: 4px 12px; border-radius: 10px;"
            )
        elif status == "IDLE":
            self.status_pill.setText("IDLE")
            self.status_pill.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: {COLOR_BG_MAIN}; background-color: {COLOR_TEXT_SECONDARY}; padding: 4px 12px; border-radius: 10px;"
            )

    def _add_event_to_table(self, ts: str, event_type: str, tid: str, class_name: str) -> None:
        row = self.table_events.rowCount()
        self.table_events.insertRow(row)

        item_ts = QTableWidgetItem(ts)
        item_ev = QTableWidgetItem(event_type)
        item_id = QTableWidgetItem(tid)
        item_cls = QTableWidgetItem(class_name)

        # Visual highlights
        if "Inbound" in event_type or "ENTRY" in event_type:
            item_ev.setForeground(QColor(COLOR_ACCENT))
        elif "Outbound" in event_type or "EXIT" in event_type:
            item_ev.setForeground(QColor(COLOR_WARNING))

        self.table_events.setItem(row, 0, item_ts)
        self.table_events.setItem(row, 1, item_ev)
        self.table_events.setItem(row, 2, item_id)
        self.table_events.setItem(row, 3, item_cls)
        self.table_events.scrollToBottom()

    def closeEvent(self, event) -> None:
        """Gracefully release camera, worker thread, recorder, and log files on shutdown."""
        logger.info("Application shutdown initiated. Releasing resources...")
        if self.recorder.is_recording:
            self.recorder.stop_recording()

        if self.worker.isRunning():
            self.worker.stop()

        self.event_logger.log_event("APP_SHUTDOWN", source_name=self.current_source_name, details="Clean exit")
        event.accept()
