"""Application configuration and styling constants for DrishtiCV.

Developed by Deepesh Kumar Pankaj.
"See Objects. Track Movement. Understand the Scene."
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

# Identity
APP_NAME: str = "DrishtiCV"
APP_FULL_TITLE: str = "DrishtiCV — Real-Time Object Detection & Tracking"
APP_SUBTITLE: str = "Local Computer-Vision Workspace"
APP_VERSION: str = "1.0.0"
DEVELOPER_NAME: str = "Deepesh Kumar Pankaj"
APP_TAGLINE: str = "See Objects. Track Movement. Understand the Scene."
APP_CATEGORY: str = "Computer Vision / Artificial Intelligence / Machine Learning"

# File System Paths
WORKSPACE_ROOT: Path = Path(__file__).resolve().parent.parent.parent
MODELS_DIR: Path = WORKSPACE_ROOT / "models"
OUTPUTS_DIR: Path = WORKSPACE_ROOT / "outputs"
DOCS_DIR: Path = WORKSPACE_ROOT / "docs"

# Model Defaults
DEFAULT_MODEL_NAME: str = "yolo26n.pt"
FALLBACK_MODEL_NAMES: list[str] = ["yolo11n.pt", "yolov8n.pt"]
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_TRACKER: str = "bytetrack.yaml"
AVAILABLE_TRACKERS: dict[str, str] = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}

# Window & Video Display
MIN_WINDOW_WIDTH: int = 1120
MIN_WINDOW_HEIGHT: int = 720
DEFAULT_FPS: float = 30.0
MAX_TRAIL_LENGTH: int = 32

# Dark Technical Theme Palette
COLOR_BG_MAIN: str = "#101419"
COLOR_BG_PANEL: str = "#181F28"
COLOR_BG_CARD: str = "#1E2632"
COLOR_BG_INPUT: str = "#141A22"
COLOR_BORDER: str = "#283444"
COLOR_BORDER_LIGHT: str = "#36455A"

COLOR_ACCENT: str = "#37D6B2"       # Muted Teal Primary Accent
COLOR_ACCENT_HOVER: str = "#2CB898"
COLOR_ACCENT_DIM: str = "#1D5E51"

COLOR_TEXT_MAIN: str = "#E6EDF3"    # Soft White
COLOR_TEXT_SECONDARY: str = "#8B949E" # Cool Grey
COLOR_TEXT_MUTED: str = "#545D68"

COLOR_WARNING: str = "#D29922"      # Amber
COLOR_ERROR: str = "#F85149"        # Red
COLOR_SUCCESS: str = "#3FB950"      # Green
COLOR_INFO: str = "#58A6FF"         # Soft Blue

# Overlay BGR Colors (OpenCV format)
CV_COLOR_ACCENT: Tuple[int, int, int] = (178, 214, 55)   # #37D6B2 in BGR
CV_COLOR_LINE: Tuple[int, int, int] = (0, 215, 255)      # Amber/Yellow in BGR
CV_COLOR_REGION: Tuple[int, int, int] = (255, 178, 55)   # Light cyan/blue in BGR
CV_COLOR_TRAIL: Tuple[int, int, int] = (178, 214, 55)
CV_COLOR_ALERT: Tuple[int, int, int] = (73, 81, 248)     # Red in BGR

# Base QSS Stylesheet
DARK_THEME_QSS: str = f"""
QMainWindow, QDialog {{
    background-color: {COLOR_BG_MAIN};
    color: {COLOR_TEXT_MAIN};
}}

QWidget {{
    font-family: "Segoe UI", "SF Pro Display", -apple-system, system-ui, sans-serif;
    font-size: 13px;
    color: {COLOR_TEXT_MAIN};
}}

QToolTip {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    padding: 4px 8px;
    border-radius: 4px;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: {COLOR_BG_MAIN};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLOR_BORDER};
    min-height: 20px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLOR_ACCENT_DIM};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background: {COLOR_BG_MAIN};
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {COLOR_BORDER};
    min-width: 20px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLOR_ACCENT_DIM};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Group Boxes & Panels */
QGroupBox {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    margin-top: 18px;
    padding-top: 14px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {COLOR_ACCENT};
}}

/* Push Buttons */
QPushButton {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {COLOR_BORDER};
    border-color: {COLOR_BORDER_LIGHT};
}}
QPushButton:pressed {{
    background-color: {COLOR_BG_INPUT};
}}
QPushButton:disabled {{
    background-color: {COLOR_BG_INPUT};
    color: {COLOR_TEXT_MUTED};
    border-color: {COLOR_BG_CARD};
}}

/* Accent Button */
QPushButton#accentButton {{
    background-color: {COLOR_ACCENT};
    color: #0B1015;
    font-weight: 600;
    border: 1px solid {COLOR_ACCENT_HOVER};
}}
QPushButton#accentButton:hover {{
    background-color: {COLOR_ACCENT_HOVER};
}}
QPushButton#accentButton:pressed {{
    background-color: #24977C;
}}

/* Danger Button */
QPushButton#dangerButton {{
    background-color: rgba(248, 81, 73, 0.15);
    color: {COLOR_ERROR};
    border: 1px solid rgba(248, 81, 73, 0.4);
}}
QPushButton#dangerButton:hover {{
    background-color: rgba(248, 81, 73, 0.25);
    border-color: {COLOR_ERROR};
}}

/* Inputs, Combos, Spinners */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {COLOR_BG_INPUT};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 5px 8px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLOR_ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_PANEL};
    border: 1px solid {COLOR_BORDER};
    selection-background-color: {COLOR_ACCENT_DIM};
    selection-color: {COLOR_TEXT_MAIN};
}}

/* Checkboxes */
QCheckBox {{
    spacing: 8px;
    color: {COLOR_TEXT_MAIN};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    background-color: {COLOR_BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLOR_BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_TEXT_MAIN};
    border: 2px solid {COLOR_ACCENT};
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    background-color: {COLOR_BG_PANEL};
    border-radius: 5px;
}}
QTabBar::tab {{
    background-color: {COLOR_BG_MAIN};
    color: {COLOR_TEXT_SECONDARY};
    padding: 6px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid {COLOR_BORDER};
    border-bottom: none;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {COLOR_BG_PANEL};
    color: {COLOR_ACCENT};
    font-weight: 600;
}}

/* Tables */
QTableWidget, QTableView {{
    background-color: {COLOR_BG_INPUT};
    border: 1px solid {COLOR_BORDER};
    gridline-color: {COLOR_BORDER};
    selection-background-color: {COLOR_ACCENT_DIM};
    selection-color: {COLOR_TEXT_MAIN};
    border-radius: 4px;
}}
QHeaderView::section {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_SECONDARY};
    border: 1px solid {COLOR_BORDER};
    padding: 4px 8px;
    font-weight: 600;
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLOR_BG_PANEL};
    border-top: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_SECONDARY};
}}
"""
