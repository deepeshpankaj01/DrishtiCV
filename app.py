"""DrishtiCV Application Entrypoint.

Real-Time Object Detection, Multi-Object Tracking & Scene Analytics.
Developed by Deepesh Kumar Pankaj.
"""

import sys
import logging
from pathlib import Path

# Add src to Python module path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from drishticv.config import APP_NAME, DEVELOPER_NAME, APP_FULL_TITLE
from drishticv.ui.main_window import MainWindow


def configure_logging() -> None:
    """Configure base console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Initialize and run DrishtiCV desktop application."""
    configure_logging()
    logger = logging.getLogger("drishticv")
    logger.info(f"Starting {APP_FULL_TITLE}")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(DEVELOPER_NAME)
    app.setApplicationDisplayName(APP_FULL_TITLE)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
