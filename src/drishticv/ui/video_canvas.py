"""Interactive video canvas widget for DrishtiCV.

Renders annotated video frames with aspect-ratio preservation, supports interactive
mouse-based virtual-line and monitored-zone definition, and displays clear empty states.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRectF, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QImage,
    QPixmap,
    QFont,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from drishticv.config import (
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
)


class VideoCanvas(QWidget):
    """Custom high-performance canvas widget for video rendering and interactive overlays."""

    line_defined = Signal(tuple, tuple)  # (p1, p2)
    region_defined = Signal(tuple)        # (x1, y1, x2, y2)
    interaction_cancelled = Signal()

    MODE_NORMAL = "NORMAL"
    MODE_LINE_P1 = "LINE_P1"
    MODE_LINE_P2 = "LINE_P2"
    MODE_REGION_DRAG = "REGION_DRAG"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.current_frame: Optional[np.ndarray] = None
        self.frame_width: int = 0
        self.frame_height: int = 0

        # Video placement rect inside widget (accounting for letterbox/pillarbox)
        self.video_rect = QRectF()

        # Interaction Mode
        self.interactive_mode: str = self.MODE_NORMAL
        self.temp_p1: Optional[Tuple[float, float]] = None
        self.temp_mouse: Optional[Tuple[float, float]] = None

        # Configured Overlays (in native video coordinates)
        self.virtual_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.monitored_region: Optional[Tuple[float, float, float, float]] = None

        # Empty state message
        self.empty_title: str = "DrishtiCV Workspace Idle"
        self.empty_subtitle: str = "Select a Webcam, Video File, or Image from the left sidebar."
        self.error_message: Optional[str] = None

    def set_frame(self, frame: np.ndarray) -> None:
        """Update current displayed frame (expects BGR format from OpenCV)."""
        self.current_frame = frame
        self.error_message = None
        h, w = frame.shape[:2]
        if w != self.frame_width or h != self.frame_height:
            self.frame_width = w
            self.frame_height = h
        self.update()

    def set_error_state(self, error_text: str) -> None:
        """Set error message display when video source fails."""
        self.error_message = error_text
        self.current_frame = None
        self.update()

    def clear_canvas(self) -> None:
        """Reset canvas to empty idle state."""
        self.current_frame = None
        self.error_message = None
        self.frame_width = 0
        self.frame_height = 0
        self.update()

    def start_line_setup(self) -> None:
        """Enter interactive two-click line definition mode."""
        self.interactive_mode = self.MODE_LINE_P1
        self.temp_p1 = None
        self.temp_mouse = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def start_region_setup(self) -> None:
        """Enter interactive click-and-drag region definition mode."""
        self.interactive_mode = self.MODE_REGION_DRAG
        self.temp_p1 = None
        self.temp_mouse = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def cancel_interaction(self) -> None:
        """Cancel interactive setup and revert to normal view."""
        self.interactive_mode = self.MODE_NORMAL
        self.temp_p1 = None
        self.temp_mouse = None
        self.setCursor(Qt.ArrowCursor)
        self.interaction_cancelled.emit()
        self.update()

    def set_virtual_line(self, line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        self.virtual_line = line
        self.update()

    def set_monitored_region(self, region: Optional[Tuple[float, float, float, float]]) -> None:
        self.monitored_region = region
        self.update()

    # Coordinate transformation between Widget space and Frame space
    def _compute_video_rect(self) -> QRectF:
        """Compute the aspect-ratio preserved rectangle for frame inside widget."""
        w_w = self.width()
        w_h = self.height()
        if self.frame_width <= 0 or self.frame_height <= 0:
            return QRectF(0, 0, w_w, w_h)

        aspect_video = self.frame_width / float(self.frame_height)
        aspect_widget = w_w / float(w_h)

        if aspect_widget > aspect_video:
            # Pillarbox (black bars on left/right)
            target_h = w_h
            target_w = target_h * aspect_video
            target_x = (w_w - target_w) / 2.0
            target_y = 0.0
        else:
            # Letterbox (black bars on top/bottom)
            target_w = w_w
            target_h = target_w / aspect_video
            target_x = 0.0
            target_y = (w_h - target_h) / 2.0

        return QRectF(target_x, target_y, target_w, target_h)

    def widget_to_frame(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        """Map a mouse position in the widget to native video frame coordinates."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return None

        vr = self._compute_video_rect()
        if not vr.contains(pos.x(), pos.y()):
            # Clamp to video rect boundaries
            clamped_x = max(vr.left(), min(vr.right(), float(pos.x())))
            clamped_y = max(vr.top(), min(vr.bottom(), float(pos.y())))
        else:
            clamped_x = float(pos.x())
            clamped_y = float(pos.y())

        fx = (clamped_x - vr.left()) * (self.frame_width / vr.width())
        fy = (clamped_y - vr.top()) * (self.frame_height / vr.height())
        return (fx, fy)

    def frame_to_widget(self, frame_pt: Tuple[float, float]) -> QPoint:
        """Map native video coordinates to widget pixel coordinates."""
        vr = self._compute_video_rect()
        if self.frame_width <= 0 or self.frame_height <= 0:
            return QPoint(int(frame_pt[0]), int(frame_pt[1]))

        wx = vr.left() + (frame_pt[0] / float(self.frame_width)) * vr.width()
        wy = vr.top() + (frame_pt[1] / float(self.frame_height)) * vr.height()
        return QPoint(int(round(wx)), int(round(wy)))

    # Mouse Events
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.cancel_interaction()
            return

        if event.button() != Qt.LeftButton:
            return

        pt = self.widget_to_frame(event.position().toPoint())
        if pt is None:
            return

        if self.interactive_mode == self.MODE_LINE_P1:
            self.temp_p1 = pt
            self.interactive_mode = self.MODE_LINE_P2
            self.update()
        elif self.interactive_mode == self.MODE_LINE_P2:
            if self.temp_p1 is not None and pt != self.temp_p1:
                p1, p2 = self.temp_p1, pt
                self.interactive_mode = self.MODE_NORMAL
                self.setCursor(Qt.ArrowCursor)
                self.line_defined.emit(p1, p2)
                self.temp_p1 = None
                self.temp_mouse = None
                self.update()
        elif self.interactive_mode == self.MODE_REGION_DRAG:
            self.temp_p1 = pt
            self.temp_mouse = pt
            self.update()

    def mouseMoveEvent(self, event) -> None:
        pt = self.widget_to_frame(event.position().toPoint())
        if pt is not None:
            self.temp_mouse = pt
            if self.interactive_mode in (self.MODE_LINE_P2, self.MODE_REGION_DRAG):
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.interactive_mode == self.MODE_REGION_DRAG:
            pt = self.widget_to_frame(event.position().toPoint())
            if self.temp_p1 is not None and pt is not None:
                x1, y1 = self.temp_p1
                x2, y2 = pt
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    self.interactive_mode = self.MODE_NORMAL
                    self.setCursor(Qt.ArrowCursor)
                    self.region_defined.emit(rect)
                self.temp_p1 = None
                self.temp_mouse = None
                self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.cancel_interaction()
        else:
            super().keyPressEvent(event)

    # Painting
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # 1. Fill base canvas background
        painter.fillRect(self.rect(), QColor(COLOR_BG_MAIN))

        # 2. Check states
        if self.error_message:
            self._draw_error_state(painter)
            return

        if self.current_frame is None:
            self._draw_empty_state(painter)
            return

        # 3. Render video frame
        self.video_rect = self._compute_video_rect()

        # Convert BGR OpenCV image to RGB QImage
        h, w, ch = self.current_frame.shape
        bytes_per_line = ch * w
        # Note: using cv2.cvtColor to ensure correct color channels
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        painter.drawImage(self.video_rect, qimg)

        # 4. Render persistent configured overlays
        if self.monitored_region is not None:
            self._draw_region_overlay(painter, self.monitored_region, is_temp=False)

        if self.virtual_line is not None:
            self._draw_line_overlay(painter, self.virtual_line[0], self.virtual_line[1], is_temp=False)

        # 5. Render interactive setup preview
        if self.interactive_mode == self.MODE_LINE_P1:
            self._draw_instruction_banner(painter, "Step 1/2: Click on video to place Line Start Point (Esc to cancel)")
        elif self.interactive_mode == self.MODE_LINE_P2 and self.temp_p1 is not None and self.temp_mouse is not None:
            self._draw_line_overlay(painter, self.temp_p1, self.temp_mouse, is_temp=True)
            self._draw_instruction_banner(painter, "Step 2/2: Click to place Line End Point (Esc to cancel)")
        elif self.interactive_mode == self.MODE_REGION_DRAG:
            if self.temp_p1 is not None and self.temp_mouse is not None:
                x1, y1 = self.temp_p1
                x2, y2 = self.temp_mouse
                rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                self._draw_region_overlay(painter, rect, is_temp=True)
            self._draw_instruction_banner(painter, "Click and drag to define Monitored Zone (Esc to cancel)")

    def _draw_empty_state(self, painter: QPainter) -> None:
        """Render a restrained, technical dark idle screen."""
        w, h = self.width(), self.height()

        # Draw subtle grid pattern
        pen_grid = QPen(QColor(COLOR_BORDER))
        pen_grid.setStyle(Qt.DotLine)
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)

        step = 48
        for x in range(0, w, step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            painter.drawLine(0, y, w, y)

        # Center target reticle
        cx, cy = w // 2, h // 2
        pen_reticle = QPen(QColor(COLOR_ACCENT))
        pen_reticle.setWidth(2)
        painter.setPen(pen_reticle)
        reticle_r = 36
        painter.drawEllipse(cx - reticle_r, cy - reticle_r, reticle_r * 2, reticle_r * 2)
        painter.drawLine(cx - reticle_r - 12, cy, cx - 12, cy)
        painter.drawLine(cx + 12, cy, cx + reticle_r + 12, cy)
        painter.drawLine(cx, cy - reticle_r - 12, cx, cy - 12)
        painter.drawLine(cx, cy + 12, cx, cy + reticle_r + 12)

        # Title & Guidance Text
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.setPen(QColor(COLOR_TEXT_MAIN))
        painter.drawText(QRectF(0, cy + 50, w, 32), Qt.AlignCenter, self.empty_title)

        painter.setFont(QFont("Segoe UI", 11))
        painter.setPen(QColor(COLOR_TEXT_SECONDARY))
        painter.drawText(QRectF(0, cy + 85, w, 24), Qt.AlignCenter, self.empty_subtitle)

    def _draw_error_state(self, painter: QPainter) -> None:
        """Render readable error card overlay."""
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        card_w, card_h = min(560, w - 40), 160
        card_rect = QRectF(cx - card_w // 2, cy - card_h // 2, card_w, card_h)

        painter.setBrush(QBrush(QColor(COLOR_BG_CARD)))
        painter.setPen(QPen(QColor(COLOR_ERROR), 2))
        painter.drawRoundedRect(card_rect, 6, 6)

        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        painter.setPen(QColor(COLOR_ERROR))
        painter.drawText(
            QRectF(card_rect.left() + 20, card_rect.top() + 16, card_w - 40, 26),
            Qt.AlignLeft,
            "Source Access Notice",
        )

        painter.setFont(QFont("Segoe UI", 11))
        painter.setPen(QColor(COLOR_TEXT_MAIN))
        painter.drawText(
            QRectF(card_rect.left() + 20, card_rect.top() + 48, card_w - 40, card_h - 60),
            Qt.AlignLeft | Qt.TextWordWrap,
            self.error_message or "An error occurred accessing the video source.",
        )

    def _draw_instruction_banner(self, painter: QPainter, message: str) -> None:
        """Render technical floating banner during setup interactions."""
        w = self.width()
        banner_w = min(520, w - 40)
        banner_rect = QRectF((w - banner_w) / 2.0, 16, banner_w, 34)

        painter.setBrush(QBrush(QColor(COLOR_BG_CARD)))
        painter.setPen(QPen(QColor(COLOR_ACCENT), 1.5))
        painter.drawRoundedRect(banner_rect, 4, 4)

        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(COLOR_ACCENT))
        painter.drawText(banner_rect, Qt.AlignCenter, message)

    def _draw_line_overlay(
        self,
        painter: QPainter,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        is_temp: bool = False,
    ) -> None:
        """Render virtual counting line with direction indicator and side badges."""
        wp1 = self.frame_to_widget(p1)
        wp2 = self.frame_to_widget(p2)

        color = QColor(COLOR_WARNING) if not is_temp else QColor(COLOR_ACCENT)
        pen = QPen(color, 3 if not is_temp else 2, Qt.SolidLine)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(wp1, wp2)

        # Draw endpoint handles
        painter.setBrush(QBrush(color))
        painter.drawEllipse(wp1, 5, 5)
        painter.drawEllipse(wp2, 5, 5)

        # Side labels (Side A: Left of vector, Side B: Right of vector)
        if not is_temp:
            dx = wp2.x() - wp1.x()
            dy = wp2.y() - wp1.y()
            length = math.hypot(dx, dy)
            if length > 20:
                mid_x = (wp1.x() + wp2.x()) / 2.0
                mid_y = (wp1.y() + wp2.y()) / 2.0

                # Perpendicular normal vectors
                nx = -dy / length
                ny = dx / length
                offset = 18

                pos_a = QPoint(int(mid_x + nx * offset), int(mid_y + ny * offset))
                pos_b = QPoint(int(mid_x - nx * offset), int(mid_y - ny * offset))

                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))

                # Badge A
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(16, 20, 25, 220)))
                painter.drawRoundedRect(QRectF(pos_a.x() - 12, pos_a.y() - 10, 24, 20), 3, 3)
                painter.setPen(QColor(COLOR_ACCENT))
                painter.drawText(QRectF(pos_a.x() - 12, pos_a.y() - 10, 24, 20), Qt.AlignCenter, "A")

                # Badge B
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(16, 20, 25, 220)))
                painter.drawRoundedRect(QRectF(pos_b.x() - 12, pos_b.y() - 10, 24, 20), 3, 3)
                painter.setPen(QColor(COLOR_WARNING))
                painter.drawText(QRectF(pos_b.x() - 12, pos_b.y() - 10, 24, 20), Qt.AlignCenter, "B")

    def _draw_region_overlay(
        self,
        painter: QPainter,
        rect: Tuple[float, float, float, float],
        is_temp: bool = False,
    ) -> None:
        """Render monitored region with translucent fill and corner tags."""
        x1, y1, x2, y2 = rect
        p_top_left = self.frame_to_widget((x1, y1))
        p_bottom_right = self.frame_to_widget((x2, y2))

        w_rect = QRectF(
            min(p_top_left.x(), p_bottom_right.x()),
            min(p_top_left.y(), p_bottom_right.y()),
            abs(p_bottom_right.x() - p_top_left.x()),
            abs(p_bottom_right.y() - p_top_left.y()),
        )

        fill_color = QColor(55, 214, 178, 35) if not is_temp else QColor(55, 214, 178, 55)
        painter.fillRect(w_rect, fill_color)

        pen = QPen(QColor(COLOR_ACCENT), 2, Qt.DashLine if is_temp else Qt.SolidLine)
        painter.setPen(pen)
        painter.drawRect(w_rect)

        # Region Title Badge
        if not is_temp and w_rect.width() > 60:
            badge_h = 20
            badge_w = 110
            badge_rect = QRectF(w_rect.left(), w_rect.top() - badge_h if w_rect.top() >= badge_h else w_rect.top(), badge_w, badge_h)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(16, 20, 25, 220)))
            painter.drawRoundedRect(badge_rect, 3, 3)

            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(QColor(COLOR_ACCENT))
            painter.drawText(badge_rect, Qt.AlignCenter, "Monitored Zone")
