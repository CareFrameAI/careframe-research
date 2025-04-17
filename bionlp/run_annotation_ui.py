import sys
from PyQt6.QtWidgets import QApplication
from bionlp.annotation_ui import StandaloneBioNlpUI

def launch_bionlp_annotation_ui():
    """Launch the UMLS Medical Term Viewer as a standalone application"""
    app = QApplication(sys.argv if not QApplication.instance() else [])
    window = StandaloneBioNlpUI()
    window.show()
    
    # Only run the exec() if there's no existing application
    if not QApplication.instance():
        sys.exit(app.exec())
    else:
        return window

if __name__ == "__main__":
    launch_bionlp_annotation_ui() 