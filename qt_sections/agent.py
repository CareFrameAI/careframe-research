from datetime import datetime
import os
import sys
import asyncio
import time
import pandas as pd
import numpy as np
import re
import html
import json
import traceback
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import copy
import threading
import psutil
import math
import ast

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, 
    QParallelAnimationGroup, QSequentialAnimationGroup, pyqtSignal, pyqtSlot,
    QThread, QObject, QRect, QByteArray, QPointF
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, 
    QLabel, QLineEdit, QSplitter, QFrame, QSizePolicy, QSpacerItem,
    QTextEdit, QProgressBar, QGraphicsOpacityEffect, QStackedWidget, QFileDialog, QDialog,
    QGraphicsDropShadowEffect, QCheckBox, QRadioButton, QButtonGroup, QTabWidget, QGridLayout, QSpinBox,
    QApplication, QComboBox
)
from PyQt6.QtGui import (
    QIcon, QColor, QPainter, QPen, QBrush, QFont, QPalette, 
    QLinearGradient, QFontMetrics, QConicalGradient
)
from PyQt6.QtSvgWidgets import QSvgWidget

# Add necessary imports
from qasync import asyncSlot
from llms.client import call_llm_sync, call_llm_async

# Import TaskStatus from common module
from common.status import TaskStatus

# Import the icon loader
from helpers.load_icon import load_bootstrap_icon

# Define pastel color palette
PASTEL_COLORS = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7', '#B3FFF7']
PASTEL_CMAP = sns.color_palette(PASTEL_COLORS)
# Configure SIP to avoid bad catcher results
# To install SIP: pip install PyQt6-sip
try:
    import PyQt6.sip as sip
    # Check if the attribute exists before calling it
    if hasattr(sip, 'setdestroyonexit'):
        sip.setdestroyonexit(False)  # Don't destroy C++ objects on exit
    # Override the bad catcher result function if accessible
    if hasattr(sip, 'setBadCatcherResult'):
        # Pass None directly instead of a lambda function
        sip.setBadCatcherResult(None)
except ImportError:
    pass

# Set matplotlib backend
matplotlib.use('Agg', force=True)  # Force Agg backend

# Disable interactive mode
plt.ioff()

# Disable Qt-specific backend features
plt.switch_backend('Agg')

# Neutralize problematic socket handling in matplotlib
import socket
socket.socket = socket.socket


class TimelineTask:
    """Represents a task in the timeline."""
    def __init__(self, 
                 name: str, 
                 description: str = "", 
                 status: TaskStatus = TaskStatus.PENDING):
        self.name = name
        self.description = description
        self.status = status
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
    
    def validate_transition(self, new_status: TaskStatus) -> bool:
        """
        Validate if a status transition is allowed.
        """
        # If trying to set the same status, always allow it
        if self.status == new_status:
            return True
            
        # Define valid transitions
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.WARNING],
            TaskStatus.RUNNING: [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.WARNING],
            TaskStatus.COMPLETED: [],  # Terminal state
            TaskStatus.FAILED: [],     # Terminal state
            TaskStatus.WARNING: [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]
        }
        
        return new_status in valid_transitions.get(self.status, [])
        
    def start(self):
        """Start the task."""
        if not self.validate_transition(TaskStatus.RUNNING):
            return False
            
        self.start_time = datetime.now()
        self.status = TaskStatus.RUNNING
        return True
        
    def complete(self):
        """Mark the task as completed."""
        if not self.validate_transition(TaskStatus.COMPLETED):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.COMPLETED
        return True
        
    def fail(self):
        """Mark the task as failed."""
        if not self.validate_transition(TaskStatus.FAILED):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.FAILED
        return True
        
    def warn(self):
        """Mark the task with a warning."""
        if not self.validate_transition(TaskStatus.WARNING):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.WARNING
        return True
        
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.end_time:
            return self.duration
        else:
            return (datetime.now() - self.start_time).total_seconds()
            
    def format_elapsed_time(self) -> str:
        """Format elapsed time as a string."""
        seconds = self.elapsed_time()
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remainder = seconds % 60
            return f"{int(minutes)}m {int(remainder)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"


class SpinnerWidget(QWidget):
    """Custom spinner animation widget."""
    def __init__(self, parent=None, size=24, color=Qt.GlobalColor.blue):
        super().__init__(parent)
        self.size = size
        self.color = color
        self.angle = 0
        self.setFixedSize(size, size)
        
        # Create timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(40)  # Update slightly faster (was 50ms)
        
    def rotate(self):
        """Rotate the spinner."""
        self.angle = (self.angle + 12) % 360  # Slightly faster rotation (was 10)
        self.update()
        
    def paintEvent(self, event):
        """Paint the spinner."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate center and radius
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) - 1
        
        # Calculate spinner dimensions for the new compact design
        outer_radius = radius * 0.9
        dot_radius = radius * 0.2
        orbit_radius = radius * 0.6
        
        # Set up painter
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Create a more modern spinner with dots
        dot_count = 8
        for i in range(dot_count):
            # Calculate position on the circle
            angle = (self.angle + i * (360 / dot_count)) % 360
            rad_angle = angle * 3.14159 / 180
            
            # Calculate opacity based on position (fade effect)
            opacity = 0.25 + 0.75 * (1 - (i / dot_count))
            
            # Calculate color with opacity
            color = QColor(self.color)
            color.setAlphaF(opacity)
            painter.setBrush(color)
            
            # Calculate dot position
            x = center_x + orbit_radius * math.cos(rad_angle)
            y = center_y + orbit_radius * math.sin(rad_angle)
            
            # Draw the dot
            painter.drawEllipse(QPointF(x, y), dot_radius, dot_radius)


class TimelineTaskWidget(QFrame):
    """Widget that represents a task in the timeline with animated borders."""
    def __init__(self, task: TimelineTask, parent=None):
        super().__init__(parent)
        self.task = task
        self.spinner = None
        self.highlight_animation = None
        self.border_animation = None
        self.shadow_effect = None
        self.effects_container = None  # Container to hold our widget with effects
        
        # Create a timer to update elapsed time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        
        # Initialize the UI
        self.init_ui()
        self.setup_animations()
        
        # Update every second if task is running
        if task.status == TaskStatus.RUNNING:
            self.timer.start(1000)
            self.start_border_animation()
            
    def init_ui(self):
        """Initialize the UI with grid layout."""
        self.setObjectName("timelineTask")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        
        # Add shadow effect - but don't apply it directly to the widget yet
        # We'll handle it in setup_animations to avoid conflicts with highlight effect
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(8)
        self.shadow_effect.setColor(QColor(0, 0, 0, 40))
        self.shadow_effect.setOffset(0, 2)
        
        # Use compact grid layout 
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(4)
        
        # Status container with background circle for icon
        status_container = QWidget()
        status_container.setFixedSize(32, 32)
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add status icon at left with circle background
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(24, 24)
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_icon)
        
        # Add status container to main layout
        grid.addWidget(status_container, 0, 0, 2, 1, Qt.AlignmentFlag.AlignCenter)
        
        # Add task name at top row
        self.name_label = QLabel(self.task.name)
        self.name_label.setObjectName("taskName")
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        self.name_label.setFont(font)
        # Limit name length with ellipsis if too long
        self.name_label.setMaximumWidth(170)
        metrics = QFontMetrics(self.name_label.font())
        elided_text = metrics.elidedText(self.task.name, Qt.TextElideMode.ElideRight, 160)
        self.name_label.setText(elided_text)
        grid.addWidget(self.name_label, 0, 1)
        
        # Add task description as a smaller, lighter text if it's not empty
        if self.task.description and self.task.description.strip():
            self.desc_label = QLabel(self.task.description)
            self.desc_label.setObjectName("taskDescription")
            self.desc_label.setWordWrap(True)
            self.desc_label.setStyleSheet("font-size: 8pt; ")
            # Truncate long descriptions
            elided_desc = metrics.elidedText(self.task.description, Qt.TextElideMode.ElideRight, 160)
            self.desc_label.setText(elided_desc)
            grid.addWidget(self.desc_label, 1, 1)
        
        # Add elapsed time at bottom right
        self.time_label = QLabel(self.task.format_elapsed_time())
        self.time_label.setObjectName("taskTime")
        font = QFont()
        font.setPointSize(7)
        self.time_label.setFont(font)
        grid.addWidget(self.time_label, 2, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        
        # Update the status icon based on current status
        self.update_status_icon()
        
        # Set a fixed width to make the grid more uniform
        self.setMinimumWidth(180)
        self.setMaximumWidth(220)
        
        # Enhanced chip-like styling
        self.setStyleSheet("""
            QFrame#timelineTask {
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            QFrame#timelineTask:hover {
                border-color: #aaa;
            }
            QFrame#timelineTask[status="running"] {
                border-color: #4a6ee0;
            }
            QFrame#timelineTask[status="completed"] {
                border-color: #28a745;
            }
            QFrame#timelineTask[status="failed"] {
                border-color: #dc3545;
            }
            QFrame#timelineTask[status="warning"] {
                border-color: #ffc107;
            }
            QLabel#taskName {
                font-weight: bold;
            }
        """)
        
    def setup_animations(self):
        """Set up animations for the task widget."""
        # Use a different approach for handling both effects
        # Instead of setting graphics effects directly on the widget, we'll use property animations directly
        
        # Setup opacity animation for highlighting
        self.highlight_animation = QPropertyAnimation(self, b"windowOpacity")
        self.highlight_animation.setDuration(300)  # Faster animation
        self.highlight_animation.setStartValue(1.0)
        self.highlight_animation.setEndValue(0.7)
        self.highlight_animation.setLoopCount(2)  # Less loops for subtle effect
        
        # Setup shadow animation
        self.shadow_animation = QPropertyAnimation(self.shadow_effect, b"blurRadius")
        self.shadow_animation.setDuration(1500)
        self.shadow_animation.setStartValue(5)
        self.shadow_animation.setEndValue(15)
        self.shadow_animation.setLoopCount(-1)  # infinite
        self.shadow_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Apply shadow effect at the end of setup
        self.setGraphicsEffect(self.shadow_effect)
            
    def start_border_animation(self):
        """Start the border pulse animation for running tasks."""
        # Start shadow animation for running tasks
        if self.shadow_animation and self.shadow_effect:
            self.shadow_animation.stop()
            self.shadow_animation.start()
            
    def stop_border_animation(self):
        """Stop the border animation."""
        # Stop shadow animation
        if hasattr(self, 'shadow_animation') and self.shadow_animation:
            self.shadow_animation.stop()
            # Reset shadow to normal
            if hasattr(self, 'shadow_effect') and self.shadow_effect:
                self.shadow_effect.setBlurRadius(8)
            
    def update_time(self):
        """Update the elapsed time display."""
        self.time_label.setText(self.task.format_elapsed_time())
        
    def update_status_icon(self):
        """Update the status icon based on task status."""
        if self.task.status == TaskStatus.PENDING:
            # Pending icon (clock)
            icon_path = load_bootstrap_icon("clock")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.RUNNING:
            # Running icon (spinner)
            if not self.spinner:
                self.spinner = SpinnerWidget(self, size=24, color=QColor(74, 110, 224))  # Use theme blue color
                layout = self.layout()
                # Find the status icon inside the layout
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item.widget() and isinstance(item.widget(), QWidget) and item.widget().layout():
                        status_layout = item.widget().layout()
                        for j in range(status_layout.count()):
                            widget = status_layout.itemAt(j).widget()
                            if widget == self.status_icon:
                                # Hide the status icon when spinner is shown
                                self.status_icon.setVisible(False)
                                status_layout.replaceWidget(self.status_icon, self.spinner)
                                self.spinner.show()
                                break
            
            # Start timer if not already running
            if not self.timer.isActive():
                self.timer.start(1000)
                
            # Start border animation
            self.start_border_animation()
                
        elif self.task.status == TaskStatus.COMPLETED:
            # Completed icon (check)
            icon_path = load_bootstrap_icon("check-circle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.FAILED:
            # Failed icon (x)
            icon_path = load_bootstrap_icon("x-circle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.WARNING:
            # Warning icon (!)
            icon_path = load_bootstrap_icon("exclamation-triangle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        # Update property for styling
        self.setProperty("status", self.task.status.name.lower())
        self.style().polish(self)
    
    def update_status(self, status: TaskStatus):
        """Update the task status and refresh the UI."""
        if self.task.validate_transition(status):
            # Update task status
            self.task.status = status
            
            # Update UI
            self.update_status_icon()
            self.update_time()
            
            # Mark end time if status is terminal
            if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED or status == TaskStatus.WARNING:
                self.task.end_time = datetime.now()
                self.task.duration = (self.task.end_time - self.task.start_time).total_seconds()


class TimelineWidget(QScrollArea):
    """Widget that displays a timeline of tasks."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_widgets = {}  # Map of task name -> widget
        self.column_count = 3  # Default number of chips per row
        self.current_row = 0
        self.current_col = 0
        self.init_ui()  # Call init_ui after initializing properties
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create container widget
        self.container = QWidget()
        self.setWidget(self.container)
        
        # Create grid layout instead of vertical layout
        self.layout = QGridLayout(self.container)
        self.layout.setContentsMargins(12, 12, 12, 12)  # Slightly more padding around edges
        self.layout.setSpacing(10)  # Moderate spacing between items
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Add a header label
        self.header_label = QLabel("Task Timeline")
        self.header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
        """)
        self.layout.addWidget(self.header_label, 0, 0, 1, self.column_count)
        
        # Start tasks at row 1 (after header)
        self.current_row = 1
        
        # Add a spacer item at the bottom to push content up
        self.spacer_item = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)  # High row number to ensure it's at the bottom
        
        # Connect resize event to adjust column count
        self.container.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Handle resize events to adjust column count."""
        if obj == self.container and event.type() == event.Type.Resize:
            self.adjust_column_count()
        return super().eventFilter(obj, event)
    
    def adjust_column_count(self):
        """Adjust column count based on container width."""
        width = self.container.width()
        # Calculate optimal columns based on chip width + spacing
        chip_width = 220  # Maximum width of chips
        spacing = self.layout.spacing()
        
        # Calculate how many chips can fit in the width (minimum 1)
        new_column_count = max(1, int(width / (chip_width + spacing)))
        
        # Only reorganize if column count changed
        if new_column_count != self.column_count:
            self.column_count = new_column_count
            self.reorganize_layout()
    
    def reorganize_layout(self):
        """Reorganize the task layout after column count change."""
        # Store widgets in order
        widgets = []
        for name, widget in self.task_widgets.items():
            widgets.append(widget)
        
        # Remove widgets from layout
        for widget in widgets:
            self.layout.removeWidget(widget)
        
        # Remove spacer
        self.layout.removeItem(self.spacer_item)
        
        # Reset position
        self.current_row = 1  # Start after header
        self.current_col = 0
        
        # Re-add widgets
        for widget in widgets:
            self.layout.addWidget(widget, self.current_row, self.current_col)
            self.current_col += 1
            if self.current_col >= self.column_count:
                self.current_col = 0
                self.current_row += 1
                
        # Update header span
        self.layout.removeWidget(self.header_label)
        self.layout.addWidget(self.header_label, 0, 0, 1, self.column_count)
        
        # Add spacer back
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)
        
    def add_task(self, task: TimelineTask):
        """
        Add a task to the timeline.
        
        Args:
            task: The task to add
        """
        # Create task widget
        task_widget = TimelineTaskWidget(task, self)
        
        # Store reference
        self.task_widgets[task.name] = task_widget
        
        # Remove spacer first
        self.layout.removeItem(self.spacer_item)
        
        # Add task widget to layout at the current position
        self.layout.addWidget(task_widget, self.current_row, self.current_col)
        
        # Update position for next task
        self.current_col += 1
        if self.current_col >= self.column_count:
            self.current_col = 0
            self.current_row += 1
        
        # Add spacer back at the bottom
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)
        
        # Scroll to show the new task
        self.scroll_to_task(task.name)
        
        return task_widget
        
    def update_task_status(self, task_name: str, status: TaskStatus):
        """
        Update a task's status.
        
        Args:
            task_name: The name of the task
            status: The new status
        """
        if task_name in self.task_widgets:
            self.task_widgets[task_name].update_status(status)
            
    def scroll_to_task(self, task_name: str):
        """
        Scroll to ensure a task is visible.
        
        Args:
            task_name: The name of the task
        """
        if task_name in self.task_widgets:
            task_widget = self.task_widgets[task_name]
            
            # Highlight the task
            self._highlight_task(task_widget)
            
            # Scroll to the widget
            task_geometry = task_widget.geometry()
            viewport_rect = self.viewport().rect()
            
            if task_geometry.top() < 0 or task_geometry.bottom() > viewport_rect.height():
                # Get the widget position in scrollarea coordinates
                task_pos = task_widget.pos()
                
                # Scroll to position
                self.ensureWidgetVisible(task_widget)
            
    def _highlight_task(self, task_widget):
        """
        Highlight a task briefly.
        
        Args:
            task_widget: The widget to highlight
        """
        # Start highlight animation
        if hasattr(task_widget, 'highlight_animation') and task_widget.highlight_animation:
            task_widget.highlight_animation.stop()
            task_widget.highlight_animation.start()


class RoundedChatWidget(QWidget):
    """Chat widget with rounded corners."""
    messageSent = pyqtSignal(str)
    fileUploadRequested = pyqtSignal(str)  # Signal for file upload
    suggestionClicked = pyqtSignal(str)  # Signal for suggestion clicks
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.typing_indicator_visible = False
        self.suggestions = []
        self.markdown = None
        
    def init_ui(self):
        """Initialize the UI with grid layouts."""
        # Set up main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create settings panel at the very top
        settings_panel = QWidget()
        settings_panel.setObjectName("settingsPanel")
        
        # Use horizontal layout for settings
        settings_layout = QHBoxLayout(settings_panel)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(15)
        
        # Create settings title
        settings_title = QLabel("Analysis Settings")
        settings_title.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(settings_title)
        
        # Plan depth control - already in horizontal layout
        plan_layout = QHBoxLayout()
        plan_layout.setSpacing(8)
        plan_label = QLabel("Plan Depth:")
        plan_label.setObjectName("settingLabel")
        plan_label.setFixedWidth(70)
        self.plan_depth_spinbox = QSpinBox()
        self.plan_depth_spinbox.setObjectName("settingSpinbox")
        self.plan_depth_spinbox.setMinimum(1)
        self.plan_depth_spinbox.setMaximum(10)
        self.plan_depth_spinbox.setValue(5)
        self.plan_depth_spinbox.setFixedWidth(45)
        self.plan_depth_spinbox.setToolTip("Analysis plan depth")
        plan_layout.addWidget(plan_label)
        plan_layout.addWidget(self.plan_depth_spinbox)
        
        # Max graphs control - already in horizontal layout
        graphs_layout = QHBoxLayout()
        graphs_layout.setSpacing(8)
        graphs_label = QLabel("Max Graphs:")
        graphs_label.setObjectName("settingLabel")
        graphs_label.setFixedWidth(70)
        self.max_graphs_spinbox = QSpinBox()
        self.max_graphs_spinbox.setObjectName("settingSpinbox")
        self.max_graphs_spinbox.setMinimum(1)
        self.max_graphs_spinbox.setMaximum(20)
        self.max_graphs_spinbox.setValue(10)
        self.max_graphs_spinbox.setFixedWidth(45)
        self.max_graphs_spinbox.setToolTip("Maximum graphs per step")
        graphs_layout.addWidget(graphs_label)
        graphs_layout.addWidget(self.max_graphs_spinbox)
        
        # Add both controls to the horizontal layout
        settings_layout.addLayout(plan_layout)
        settings_layout.addLayout(graphs_layout)
        settings_layout.addStretch()
        
        # Add settings panel to main layout
        main_layout.addWidget(settings_panel)
        
        # Create main content area for chat
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Create the chat display area
        self.chat_display = QScrollArea()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setWidgetResizable(True)
        self.chat_display.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create the container for chat messages
        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(15)
        
        # Add an expanding spacer to push content to the top
        self.chat_layout.addStretch()
        
        # Set the container as the scroll area's widget
        self.chat_display.setWidget(self.chat_container)
        
        # Add chat display to content layout with stretch
        content_layout.addWidget(self.chat_display, 1)
        
        # Add content widget to main layout
        main_layout.addWidget(content_widget, 1)
        
        # Create the input area with rounded corners
        input_container = QWidget()
        input_container.setObjectName("inputContainer")
        
        # Use grid layout for input area
        input_grid = QGridLayout(input_container)
        input_grid.setContentsMargins(10, 5, 10, 5)
        input_grid.setSpacing(5)
        
        # Create thinking area for agent status indicators
        self.thinking_area = QWidget()
        self.thinking_area.setObjectName("thinkingArea")
        self.thinking_area.setMaximumHeight(80)  # Limit height but allow for multiple lines
        self.thinking_area.setMinimumHeight(40)
        
        thinking_layout = QVBoxLayout(self.thinking_area)
        thinking_layout.setContentsMargins(8, 5, 8, 5)
        thinking_layout.setSpacing(3)
        thinking_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Status indicator for agent actions
        self.status_indicator = QLabel()
        self.status_indicator.setObjectName("statusIndicator")
        self.status_indicator.setMaximumWidth(800)  # Prevent extreme stretching
        self.status_indicator.setWordWrap(True)     # Important: Enable word wrapping
        self.status_indicator.setStyleSheet("""
            #statusIndicator {
                
                font-style: italic;
                font-size: 0.9em;
                padding-left: 5px;
            }
        """)
        self.status_indicator.setVisible(False)
        thinking_layout.addWidget(self.status_indicator)
        
        # Add the thinking area to input grid, spanning the full width
        input_grid.addWidget(self.thinking_area, 0, 0, 1, 3)
        
        # Create typing indicator
        self.typing_indicator = QLabel("Agent is typing...")
        self.typing_indicator.setObjectName("typingIndicator")
        self.typing_indicator.setVisible(False)
        input_grid.addWidget(self.typing_indicator, 1, 0, 1, 3)
        
        # Create suggestion container
        self.suggestion_container = QWidget()
        self.suggestion_container.setObjectName("suggestionContainer")
        suggestion_layout = QHBoxLayout(self.suggestion_container)
        suggestion_layout.setContentsMargins(0, 0, 0, 10)
        suggestion_layout.setSpacing(5)
        suggestion_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        input_grid.addWidget(self.suggestion_container, 2, 0, 1, 3)
        self.suggestion_container.setVisible(False)
        
        # Create message input with rounded corners
        self.message_input = QTextEdit()
        self.message_input.setObjectName("messageInput")
        self.message_input.setMinimumHeight(50)
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("Type a message...")
        self.message_input.textChanged.connect(self.adjust_input_height)
        input_grid.addWidget(self.message_input, 3, 1, 1, 1)
        
        # Create the file upload button
        self.upload_button = QPushButton()
        self.upload_button.setObjectName("uploadButton")
        self.upload_button.setToolTip("Upload File")
        self.upload_button.setFixedSize(36, 36)
        
        # Set icon for upload button
        upload_icon_path = load_bootstrap_icon("paperclip")
        if upload_icon_path:
            self.upload_button.setIcon(QIcon(upload_icon_path))
            self.upload_button.setIconSize(QSize(20, 20))
        
        self.upload_button.clicked.connect(self.select_file)
        input_grid.addWidget(self.upload_button, 3, 0, 1, 1)
        
        # Create the send button with rounded corners
        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setToolTip("Send Message")
        self.send_button.setFixedSize(36, 36)
        
        # Set icon for send button
        send_icon_path = load_bootstrap_icon("send-fill")
        if send_icon_path:
            self.send_button.setIcon(QIcon(send_icon_path))
            self.send_button.setIconSize(QSize(20, 20))
        
        self.send_button.clicked.connect(self.send_message)
        input_grid.addWidget(self.send_button, 3, 2, 1, 1)
        
        # Add widgets to main layout (content area and input container)
        main_layout.addWidget(content_widget, 1)  # Stretch factor
        main_layout.addWidget(input_container, 0) # No stretch
        
        # Add styling for the thinking area and status indicators
        self.setStyleSheet(self.styleSheet() + """
            #thinkingArea {
                border-radius: 4px;
            }
            
            #statusIndicator {
                font-style: italic;
                
            }
            
            #settingLabel {
                font-size: 12px;
            }
            
            #settingSpinbox {
                border-radius: 3px;
                padding: 2px;
                max-height: 24px;
            }
        """)
        
    def update_status(self, text, is_thinking=True):
        """Update the status indicator with agent's current action."""
        # Ensure text is not too long for display
        if len(text) > 100:
            text = text[:97] + "..."
            
        # Set the text and make visible
        if is_thinking:
            self.status_indicator.setText(f"<span style='color: #4a6ee0;'>⚙️</span> {text}")
        else:
            self.status_indicator.setText(text)
            
        self.status_indicator.setVisible(True)
        
        # Ensure the thinking area is visible
        self.thinking_area.setVisible(True)
        
        # Auto-hide after some time if not thinking
        if not is_thinking:
            QTimer.singleShot(5000, lambda: self.status_indicator.setVisible(False))
    
    def set_suggestions(self, suggestions):
        """
        Set suggestion chips.
        
        Args:
            suggestions: List of suggestion strings
        """
        # Clear current suggestions
        self.suggestions = suggestions
        
        # Clear suggestion container
        while self.suggestion_container.layout().count():
            item = self.suggestion_container.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new suggestions
        for suggestion in suggestions:
            chip = self.create_suggestion_chip(suggestion)
            self.suggestion_container.layout().addWidget(chip)
        
        # Show/hide container based on suggestions
        self.suggestion_container.setVisible(bool(suggestions))
    
    def create_suggestion_chip(self, text):
        """
        Create a clickable suggestion chip.
        
        Args:
            text: The suggestion text
            
        Returns:
            QPushButton: The suggestion chip button
        """
        chip = QPushButton(text)
        chip.setObjectName("suggestionChip")
        
        # Calculate width based on text length (approximate)
        font_metrics = QFontMetrics(chip.font())
        text_width = font_metrics.horizontalAdvance(text)
        
        # Set button size (adjust padding as needed)
        chip.setMinimumWidth(min(text_width + 20, 200))
        chip.setMaximumWidth(min(text_width + 40, 300))
        chip.setMinimumHeight(30)
        chip.setMaximumHeight(30)
        
        # Connect click signal
        chip.clicked.connect(lambda: self.handle_suggestion_click(text))
        
        return chip
    
    def handle_suggestion_click(self, suggestion):
        """
        Handle click on a suggestion chip.
        
        Args:
            suggestion: The suggestion text
        """
        # Emit signal with suggestion text
        self.suggestionClicked.emit(suggestion)
        
        # Clear suggestions
        self.set_suggestions([])
    
    def adjust_input_height(self):
        """Adjust the input height based on content."""
        # Calculate new height based on document size
        doc_height = self.message_input.document().size().height()
        
        # Ensure height is within min and max
        new_height = max(50, min(doc_height + 20, 100))
        
        # Set new height
        self.message_input.setMinimumHeight(int(new_height))
        self.message_input.setMaximumHeight(int(new_height))
    
    def select_file(self):
        """Open file dialog to select a file."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Data files (*.csv *.tsv *.xlsx *.txt)")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.fileUploadRequested.emit(file_path)
            
    def send_message(self):
        """Send the current message."""
        # Get message text
        message = self.message_input.toPlainText().strip()
        
        # Clear input
        self.message_input.clear()
        
        # Emit signal if message is not empty
        if message:
            # Only emit the signal, don't add user message here as the receiver will handle that
            self.messageSent.emit(message)
    
    def add_user_message(self, message: str):
        """Add a user message to the chat display."""
        self.clean_json_formatting(message)
        self._add_message_bubble(message, is_user=True)
    
    def clean_json_formatting(self, text: str) -> str:
        """Clean up JSON content in the text."""
        # First handle JSON wrapped in triple backticks
        json_block_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
        
        def format_json_block(match):
            try:
                # Try to parse and pretty-print the JSON
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2)
                return f"```json\n{formatted}\n```"
            except json.JSONDecodeError:
                # If not valid JSON, return the original
                return match.group(0)
        
        # Replace JSON blocks with pretty-printed versions
        text = re.sub(json_block_pattern, format_json_block, text)
        
        # Also try to handle inline JSON objects (more challenging)
        # This is tricky because we need to avoid false positives
        inline_json_pattern = r'({[\s\S]*?})'
        
        def format_inline_json(match):
            try:
                # Try to parse the potential JSON
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                
                # If it parsed successfully and is non-trivial, format it
                if isinstance(parsed, dict) and len(parsed) > 1:
                    formatted = json.dumps(parsed, indent=2)
                    return f"```json\n{formatted}\n```"
                else:
                    return match.group(0)
            except json.JSONDecodeError:
                # Not valid JSON, return original
                return match.group(0)
        
        # Only apply to large JSON-like blocks to avoid false positives
        if len(text) > 100 and '{' in text and '}' in text:
            # Check if the text is not already in a code block
            if '```' not in text:
                text = re.sub(inline_json_pattern, format_inline_json, text)
        
        return text
    
    def _add_message_bubble(self, message: str, is_user: bool):
        """
        Add a message bubble to the chat.
        
        Args:
            message: The message text
            is_user: True if the message is from the user, False if from the agent
        """
        # Calculate position to insert message (before the spacer)
        spacer_idx = self.chat_layout.count() - 1
        
        # Check if this is a plan or summary and use dedicated widget
        is_plan_or_summary = False
        if not is_user:
            # Check for plan content
            if ("<h1>Analysis Plan" in message or 
                "<h2>Analysis Plan" in message or 
                "# Analysis Plan" in message):
                self._add_professional_plan(message, spacer_idx)
                return
            
            # Check for summary content
            if ("<h1>Analysis Summary" in message or 
                "<h2>Analysis Summary" in message or 
                "# Analysis Summary" in message or
                "<h1>Summary of Analysis" in message or
                "# Summary of Analysis" in message):
                self._add_professional_summary(message, spacer_idx)
                return
                
        # Create the message bubble frame for normal messages
        bubble = QFrame()
        bubble.setObjectName("userBubble" if is_user else "agentBubble")
        
        # Check if this is a markdown message with headers (likely analysis plan or summary)
        is_special_content = False
        if not is_user and ("<h1>" in message or "<h2>" in message or 
                          "# " in message or "## " in message):
            is_special_content = True
            # Special styling for analysis plans and summaries
            bubble.setObjectName("specialContentBubble")
            bubble.setStyleSheet("""
                QFrame#specialContentBubble {
                    border: 1px solid #d1d5db;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 5px;
                    min-width: 95%;
                    max-width: 98%;
                }
            """)
        
        # Create layout for the bubble
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 10, 15, 10)
        
        # Create a QLabel for the message - simple plain text for user messages
        message_label = QLabel()
        message_label.setObjectName("messageLabel")
        message_label.setWordWrap(True)
        message_label.setOpenExternalLinks(True)
        message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        
        # Use different text format based on sender
        if is_user:
            # User messages are always plain text
            message_label.setTextFormat(Qt.TextFormat.PlainText)
            message_label.setText(message)
        else:
            # Agent messages might be HTML from markdown conversion
            message_label.setTextFormat(Qt.TextFormat.RichText)
            message_label.setText(message)
        
        # Add to bubble layout
        bubble_layout.addWidget(message_label)
        
        # Create container for alignment
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Align based on sender and content type
        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        elif is_special_content:
            # Center special content (plans, summaries) with slight flexibility
            container_layout.addStretch(1)
            container_layout.addWidget(bubble, 10)  # Give bubble more stretch weight
            container_layout.addStretch(1)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()
        
        # Insert into chat layout
        self.chat_layout.insertWidget(spacer_idx, container)
        
        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def _add_professional_plan(self, content, insert_position):
        """Add a professionally formatted analysis plan using Qt widgets."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QFrame, QHBoxLayout, QScrollArea
        
        # Create plan frame
        plan_frame = QFrame()
        plan_frame.setObjectName("planFrame")
        plan_frame.setFrameShape(QFrame.Shape.StyledPanel)
        plan_frame.setStyleSheet("""
            QFrame#planFrame {
                border: 1px solid #4a6ee0;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 5px;
            }
        """)
        
        # Create frame layout
        plan_layout = QVBoxLayout(plan_frame)
        plan_layout.setContentsMargins(15, 15, 15, 15)
        plan_layout.setSpacing(10)
        
        # Add title
        title = QLabel("Analysis Plan")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        plan_layout.addWidget(title)
        
        # Create scroll area for steps
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setMaximumHeight(400)
        
        # Parse the plan content
        if content.startswith("# Analysis Plan") or content.startswith("<h1>Analysis Plan"):
            # Extract from markdown or HTML
            raw_content = ""
            if content.startswith("#"):
                # Markdown format
                raw_content = content.replace("# Analysis Plan", "").strip()
            else:
                # HTML format - strip HTML tags
                raw_content = re.sub(r'<[^>]*>', '', content)
                raw_content = raw_content.replace("Analysis Plan", "").strip()
            
            # Extract steps from content
            steps_container = QWidget()
            steps_layout = QVBoxLayout(steps_container)
            steps_layout.setContentsMargins(0, 0, 0, 0)
            steps_layout.setSpacing(10)
            
            # Find all steps using regex
            step_pattern = r"(?:Step|)\s*(\d+)[\.:\)]\s*(.*?)(?=(?:Step|)\s*\d+[\.:\)]|$)"
            steps = re.findall(step_pattern, raw_content, re.DOTALL)
            
            if steps:
                for step_num, step_desc in steps:
                    # Clean up the description
                    step_desc = step_desc.strip()
                    
                    # Check if it's a graph step
                    is_graph_step = "[GRAPH]" in step_desc
                    if is_graph_step:
                        # Keep it simple - no emojis
                        step_desc = step_desc.replace("[GRAPH]", "").strip()
                        step_label = QLabel(f"Step {step_num}: {step_desc}")
                        step_label.setStyleSheet("font-size: 15px; font-weight: bold;")
                    else:
                        step_label = QLabel(f"Step {step_num}: {step_desc}")
                        step_label.setStyleSheet("font-size: 15px; font-weight: bold;")
                    
                    step_label.setWordWrap(True)
                    steps_layout.addWidget(step_label)
            else:
                # No steps found, show raw content
                content_label = QLabel(raw_content)
                content_label.setWordWrap(True)
                content_label.setStyleSheet("font-size: 14px;")
                steps_layout.addWidget(content_label)
            
            # Set container as scroll area widget
            scroll_area.setWidget(steps_container)
            
            # Add scroll area to plan layout
            plan_layout.addWidget(scroll_area)
        else:
            # Fallback if parsing fails
            content_label = QLabel(content)
            content_label.setWordWrap(True)
            plan_layout.addWidget(content_label)
        
        # Create alignment container
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(plan_frame, 10)
        container_layout.addStretch(1)
        
        # Add to chat layout
        self.chat_layout.insertWidget(insert_position, container)
        
        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def _add_professional_summary(self, summary_data: dict):
        """Add a professionally formatted analysis summary using Qt widgets from parsed JSON data."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QFrame, QHBoxLayout, QScrollArea, QWidget
        
        # Calculate position to insert message (before the spacer)
        insert_position = self.chat_layout.count() - 1
        
        # Create summary frame
        summary_frame = QFrame()
        summary_frame.setObjectName("summaryFrame")
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setMinimumWidth(600)
        summary_frame.setStyleSheet("""
            QFrame#summaryFrame {
                border: 1px solid #28a745;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 5px;
            }
        """)
        
        # Create frame layout
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(15, 15, 15, 15)
        summary_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("Analysis Summary")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        summary_layout.addWidget(title_label)
        
        # --- Populate sections from JSON data --- 
        
        # Executive Summary
        exec_summary = summary_data.get("executive_summary", "Not available")
        exec_label = QLabel(f"<b>Executive Summary:</b> {exec_summary}")
        exec_label.setWordWrap(True)
        exec_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        summary_layout.addWidget(exec_label)
        
        # Scroll area for remaining details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setMaximumHeight(400) # Keep height limit
        
        # Container for scrollable content
        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(15)
        
        # Key Findings
        key_findings = summary_data.get("key_findings", [])
        if key_findings:
            findings_label = QLabel("<b>Key Findings:</b>")
            findings_label.setStyleSheet("font-size: 15px; margin-bottom: 5px;")
            details_layout.addWidget(findings_label)
            for finding in key_findings:
                bullet_container = QWidget()
                bullet_layout = QHBoxLayout(bullet_container)
                bullet_layout.setContentsMargins(10, 0, 0, 0)
                bullet_layout.setSpacing(10)
                bullet = QLabel("•")
                bullet.setStyleSheet("font-weight: bold;")
                content = QLabel(finding)
                content.setWordWrap(True)
                content.setStyleSheet("font-size: 14px;")
                bullet_layout.addWidget(bullet, 0)
                bullet_layout.addWidget(content, 1)
                details_layout.addWidget(bullet_container)

        # Methodology
        methodology = summary_data.get("methodology", "Not available")
        method_label = QLabel(f"<b>Methodology:</b> {methodology}")
        method_label.setWordWrap(True)
        method_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(method_label)
        
        # Detailed Results (Handle potential markdown)
        detailed_results = summary_data.get("detailed_results", "Not available")
        results_label = QLabel("<b>Detailed Results:</b>")
        results_label.setStyleSheet("font-size: 15px; margin-bottom: 5px;")
        details_layout.addWidget(results_label)
        results_content = QLabel(self.markdown_to_html(detailed_results)) # Reuse markdown converter if needed
        results_content.setWordWrap(True)
        results_content.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(results_content)

        # Visualizations Summary
        viz_summary = summary_data.get("visualizations_summary", "Not available")
        viz_label = QLabel(f"<b>Visualizations Summary:</b> {viz_summary}")
        viz_label.setWordWrap(True)
        viz_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(viz_label)

        # Limitations
        limitations = summary_data.get("limitations", "Not available")
        limit_label = QLabel(f"<b>Limitations:</b> {limitations}")
        limit_label.setWordWrap(True)
        limit_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(limit_label)
        
        # --- End of sections --- 
        
        details_layout.addStretch() # Push content up
        scroll_area.setWidget(details_container)
        summary_layout.addWidget(scroll_area)
        
        # Create alignment container for the whole summary frame
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(summary_frame, 10) # Give frame more stretch weight
        container_layout.addStretch(1)
        
        # Add to chat layout at the correct position
        self.chat_layout.insertWidget(insert_position, container)
        
        # Extract and set 'data_analysis_next_steps' as suggestions
        next_steps = summary_data.get("data_analysis_next_steps", [])
        if isinstance(next_steps, list) and all(isinstance(step, str) for step in next_steps):
            self.set_suggestions(next_steps)
        else:
            # Clear suggestions if data is invalid or missing
            self.set_suggestions([])
            print("Warning: 'data_analysis_next_steps' not found or invalid in summary data.")
        
        # Scroll to bottom (optional, as context/suggestions are added after)
        # QTimer.singleShot(50, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the chat display to the bottom."""
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_agent_message(self, message: str):
        """Add an agent message to the chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.add_agent_message(message)
        else:
            # Detect special content (analysis plans and summaries)
            is_special_content = False
            
            # Look for common patterns in analysis plans and summaries
            if message.startswith("# Analysis Plan") or message.startswith("# Analysis Summary") or \
               message.startswith("## Analysis Plan") or message.startswith("## Analysis Summary") or \
               message.startswith("# Summary of Analysis") or "## Step " in message:
                is_special_content = True
                
            # Process markdown before adding message
            # Check if content appears to be markdown
            if is_special_content or ("```" in message or "#" in message or "*" in message or 
                message.strip().startswith(">") or "- " in message):
                # Convert markdown to HTML
                html_content = self.markdown_to_html(message)
                self._add_message_bubble(html_content, is_user=False)
            else:
                # Plain text, no conversion needed
                self._add_message_bubble(message, is_user=False)
    
    def markdown_to_html(self, text: str) -> str:
        """Convert markdown to HTML using professional libraries."""
        try:
            # Use Python-Markdown with standard extensions
            import markdown
            
            # Basic set of extensions without custom processors
            extensions = [
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
                'markdown.extensions.tables', 
                'markdown.extensions.nl2br',
                'markdown.extensions.sane_lists'
            ]
            
            # Add custom styling for headings to enhance readability
            text = self._preprocess_markdown_headings(text)
            
            # Convert markdown to HTML with standard settings
            html = markdown.markdown(text, extensions=extensions)
            
            # Apply additional styling to HTML elements
            html = self._postprocess_html(html)
            
            return html
        except Exception as e:
            # Log error and fall back to simple conversion
            print(f"Error converting markdown: {str(e)}")
            return text
            
    def _preprocess_markdown_headings(self, text: str) -> str:
        """Preprocess markdown headings to enhance their appearance."""
        # Add a class to heading lines for better styling
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('# '):
                lines[i] = line + ' {.h1-heading}'
            elif line.startswith('## '):
                lines[i] = line + ' {.h2-heading}'
            elif line.startswith('### '):
                lines[i] = line + ' {.h3-heading}'
                
        return '\n'.join(lines)
        
    def _postprocess_html(self, html: str) -> str:
        """Apply additional styling to HTML elements."""
        # Style headings for better visibility
        html = html.replace('<h1>', '<h1 style="font-size: 1.8em; margin-top: 20px; margin-bottom: 15px; text-align: center;">')
        html = html.replace('<h2>', '<h2 style="font-size: 1.5em; margin-top: 15px; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;">')
        html = html.replace('<h3>', '<h3 style="font-size: 1.3em; margin-top: 10px; margin-bottom: 8px;">')
        
        # Style code blocks for better readability
        html = html.replace('<pre><code>', '<pre style="border: 1px solid #e1e4e8; border-radius: 4px; padding: 10px; overflow-x: auto;"><code style="font-family: monospace;">')
        
        # Style lists for better spacing
        html = html.replace('<ul>', '<ul style="margin-left: 15px; line-height: 1.4;">')
        html = html.replace('<ol>', '<ol style="margin-left: 15px; line-height: 1.4;">')
        
        return html
    
    def set_typing_indicator(self, visible: bool):
        """Show or hide the typing indicator."""
        self.typing_indicator.setVisible(visible)
        self.typing_indicator_visible = visible


class GraphWidget(QWidget):
    """Widget for displaying graphs and visualizations."""
    
    def __init__(self, parent=None):
        """Initialize the graph widget."""
        super().__init__(parent)
        
        # Initialize properties
        self.current_zoom = 1.0
        self.current_figure = None
        self.figure = None
        self.svg_string = None
        self.svg_widget = None
        self.canvas = None
        self.aspect_widget = None
        self.layout = None  # Will be set in init_ui
        
        # Test QSvgWidget is functioning with better error handling
        try:
            test_svg = QSvgWidget()
            # Check if the widget was created successfully
            if test_svg is None:
                raise Exception("Failed to create QSvgWidget instance")
                
            # Create a minimal valid SVG to test if rendering works
            minimal_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg">
  <rect width="10" height="10" fill="gray"/>
</svg>"""
            test_svg.renderer().load(QByteArray(minimal_svg.encode('utf-8')))
            print("SVG widget initialized and functioning correctly")
        except Exception as e:
            print(f"Warning: SVG widget initialization issue: {e}")
            print("Will use fallback rendering methods for visualizations")
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        # Create main layout and store as instance variable
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Create empty widget as placeholder
        self.placeholder = QLabel("No visualization available")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("""
            font-size: 16px;
            font-style: italic;
            border: 1px dashed;
            border-radius: 5px;
            padding: 40px;
            margin: 20px;
        """)
        
        self.layout.addWidget(self.placeholder)
    
    def clear(self):
        """Clear current visualization."""
        try:
            # Only clear layout if it exists and has items
            if hasattr(self, 'layout') and self.layout:
                # Remove all widgets from the layout
                while self.layout.count():
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            else:
                # Create layout if none exists
                self.layout = QVBoxLayout(self)
                self.layout.setContentsMargins(10, 10, 10, 10)
            
            # Add placeholder back
            self.placeholder = QLabel("No visualization available")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setStyleSheet("""
                font-size: 16px;
                font-style: italic;
                border: 1px dashed;
                border-radius: 5px;
                padding: 40px;
                margin: 20px;
            """)
            
            self.layout.addWidget(self.placeholder)
            
            # Reset references
            self.figure = None
            self.svg_string = None
            self.svg_widget = None
            self.canvas = None
        except Exception as e:
            print(f"Error clearing graph widget: {str(e)}")
            traceback.print_exc()
    
    def show_matplotlib_figure(self, figure):
        """Display a matplotlib figure using SVG for better threading safety.
        
        This method:
        1. Clears the current layout
        2. Applies pastel colors for better visibility
        3. Converts the figure to SVG (thread-safe format)
        4. Creates an SVG widget for display
        5. Adds maximize button in top-right
        
        Args:
            figure (matplotlib.figure.Figure): The figure to display
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clear existing content but only if we have the expected properties
            self.clear()
            
            # Store the figure reference
            self.current_figure = figure
            
            # Apply pastel colors for better visibility
            self._apply_pastel_colors(figure)
            
            # Create a BytesIO buffer for the SVG data
            buf = io.BytesIO()
            
            # Ensure the figure has a reasonable size before saving to SVG
            original_size = figure.get_size_inches()
            figure.set_size_inches(10, 7)  # Standardize size for consistency
            
            # Save figure as SVG to buffer with explicit size
            figure.savefig(buf, format='svg', bbox_inches='tight', dpi=100)
            buf.seek(0)
            svg_data = buf.getvalue().decode('utf-8')

            buf.close()
            
            # Restore original figure size
            figure.set_size_inches(original_size)
            
            # Create main layout - only create if no layout exists
            if not self.layout:
                self.layout = QVBoxLayout(self)
                self.layout.setContentsMargins(10, 10, 10, 10)
            else:
                # Remove all items from the layout 
                while self.layout.count() > 0:
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            
            # Create a frame for the SVG content
            svg_frame = QFrame()
            svg_frame.setFrameShape(QFrame.Shape.StyledPanel)
            svg_frame.setObjectName("svgFrame")
            # Set minimum size to ensure visibility
            svg_frame.setMinimumSize(400, 300)
            
            # Create layout for the SVG content with title and maximize button
            svg_layout = QVBoxLayout(svg_frame)
            svg_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create top bar with title and maximize button
            top_bar = QWidget()
            top_layout = QHBoxLayout(top_bar)
            top_layout.setContentsMargins(0, 0, 0, 5)
            
            # Add title (if figure has a title)
            title_text = "Visualization"
            for ax in figure.get_axes():
                if ax.get_title():
                    title_text = ax.get_title()
                    break
                    
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            top_layout.addWidget(title_label)
            
            # Add spacer to push maximize button to the right
            top_layout.addStretch()
            
            # Add maximize button
            maximize_btn = QPushButton()
            maximize_btn.setIcon(QIcon(load_bootstrap_icon("arrows-fullscreen")))
            maximize_btn.setToolTip("Maximize")
            maximize_btn.setFixedSize(24, 24)
            maximize_btn.setFlat(True)
            
            maximize_btn.clicked.connect(self._show_fullscreen_dialog)
            top_layout.addWidget(maximize_btn)
            
            # Add top bar to svg layout
            svg_layout.addWidget(top_bar)
            
            # Create the SVG widget and load the data
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Create aspect ratio maintaining container
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            
            # Add to layout
            svg_layout.addWidget(aspect_widget)
            
            # Add to main layout 
            self.layout.addWidget(svg_frame)
            
            # Store references to widgets for later manipulation
            self.svg_widget = svg_widget
            self.svg_data = svg_data  # Store the SVG data for fullscreen display
            self.aspect_widget = aspect_widget
            
            return True
        
        except Exception as e:
            error_layout = QVBoxLayout()
            error_label = QLabel(f"Error displaying figure: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            
            # Only set layout if none exists
            if not self.layout:
                self.setLayout(error_layout)
            else:
                # Clear and add error
                while self.layout.count():
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.layout.addWidget(error_label)
                
            traceback.print_exc()
            return False
    
    def _apply_pastel_colors(self, figure):
        """Apply pastel colors to the figure for better visibility."""
        for ax in figure.get_axes():
            # Check if it's a plot with lines
            lines = ax.get_lines()
            if lines:
                for i, line in enumerate(lines):
                    color_idx = i % len(PASTEL_COLORS)
                    line.set_color(PASTEL_COLORS[color_idx])
            
            # Check if it's a bar chart
            patches = ax.patches
            if patches:
                for i, patch in enumerate(patches):
                    color_idx = i % len(PASTEL_COLORS)
                    patch.set_facecolor(PASTEL_COLORS[color_idx])
    
    def _show_fullscreen_dialog(self):
        """Show the current figure in fullscreen."""
        try:
            if not self.current_figure:
                return False
                
            # Create a fullscreen dialog
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
            dialog = QDialog(self)
            dialog.setWindowTitle("Visualization - Fullscreen")
            dialog.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
            dialog.resize(1200, 900)  # Larger default size
            
            # Create layout
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Use the stored SVG data if available
            if hasattr(self, 'svg_data') and self.svg_data:
                svg_widget = QSvgWidget()
                svg_widget.renderer().load(QByteArray(self.svg_data.encode('utf-8')))
                svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                
                # Create aspect ratio maintaining container
                aspect_widget = SVGAspectRatioWidget(svg_widget)
                
                # Add to layout
                layout.addWidget(aspect_widget)
            else:
                # Generate SVG data from the figure
                try:
                    buf = io.BytesIO()
                    self.current_figure.savefig(buf, format='svg', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    svg_data = buf.getvalue().decode('utf-8')
                    buf.close()
                    
                    # Create SVG widget
                    svg_widget = QSvgWidget()
                    svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
                    svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    
                    # Create aspect ratio maintaining container
                    aspect_widget = SVGAspectRatioWidget(svg_widget)
                    
                    # Add to layout
                    layout.addWidget(aspect_widget)
                except Exception as e:
                    error_label = QLabel(f"Error displaying figure: {str(e)}")
                    error_label.setStyleSheet("color: red; font-weight: bold;")
                    layout.addWidget(error_label)
            
            # Add close button
            close_button = QPushButton("Close")
          
            close_button.clicked.connect(dialog.close)
            close_button.setFixedWidth(100)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            # Display modal dialog
            dialog.exec()
            
            return True
        except Exception as e:
            traceback.print_exc()
            return False


class ReflectionWidget(QFrame):
    """Widget for displaying agent reflections with special formatting."""
    
    def __init__(self, parent=None, has_error=False):
        super().__init__(parent)
        self.has_error = has_error
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        self.setObjectName("reflectionWidget")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Set border color based on error status
        border_color = "#e74c3c" if self.has_error else "#f39c12"
        self.setStyleSheet(f"""
            QFrame#reflectionWidget {{
                border: 2px solid {border_color};
                border-radius: 4px;
                margin: 8px 0px;
                padding: 15px;
            }}
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Add header
        header_layout = QHBoxLayout()
        
        # Add thinking icon
        thinking_icon = QLabel()
        icon_path = load_bootstrap_icon("lightbulb")
        if icon_path:
            thinking_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
        
        # Add header label
        header_text = "Error Analysis" if self.has_error else "Reflection"
        header_label = QLabel(header_text)
        header_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 14px;
            color: {border_color};
        """)
        
        header_layout.addWidget(thinking_icon)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Add content label
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("""
            font-style: italic;
            line-height: 1.4;
            padding: 10px 0;
        """)
        self.content_label.setTextFormat(Qt.TextFormat.RichText)
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        # Add widgets to layout
        layout.addLayout(header_layout)
        layout.addWidget(self.content_label)
        
    def set_content(self, text):
        """Set the reflection content."""
        # Format the text with line breaks for better readability
        formatted_text = text.replace('\n', '<br>')
        self.content_label.setText(formatted_text)


class DynamicContentWidget(QStackedWidget):
    """Widget for displaying dynamic content like graphs and data tables."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget_names = {}  # Map of name -> index
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create an empty placeholder widget
        placeholder = QWidget()
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_label = QLabel("No content to display")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("""
            font-size: 16px;
            font-style: italic;
            margin: 20px;
            padding: 20px;
        """)
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add the placeholder widget
        self.addWidget(placeholder)
        self.widget_names["placeholder"] = 0
    
    def add_widget(self, widget, name):
        """
        Add a widget to the stacked widget.
        
        Args:
            widget: The widget to add
            name: A name for the widget
        """
        # If we already have a widget with this name, remove it
        if name in self.widget_names:
            self.remove_widget(self.widget(self.widget_names[name]))
        
        # Add the new widget
        index = self.addWidget(widget)
        self.widget_names[name] = index
        
        # Show the new widget
        self.setCurrentIndex(index)
        
        return index
    
    def remove_widget(self, widget):
        """
        Remove a widget from the stacked widget.
        
        Args:
            widget: The widget to remove
        """
        # Find the widget index
        index = self.indexOf(widget)
        if index != -1:
            # Find the name
            name = None
            for key, value in self.widget_names.items():
                if value == index:
                    name = key
                    break
            
            # Remove the widget
            self.removeWidget(widget)
            
            # Update the map
            if name:
                del self.widget_names[name]
                
            # Update indices in the map
            for key, value in self.widget_names.items():
                if value > index:
                    self.widget_names[key] = value - 1
            
            # Delete the widget later
            widget.deleteLater()


class DataAnalysisAgent:
    """Agent for performing iterative data analysis based on user queries."""
    
    # Messages types for output formatting
    MSG_THINKING = "thinking"  # Thinking/control messages
    MSG_EXECUTING = "executing"  # Execution messages
    MSG_CODE = "code"  # Code snippets
    MSG_OUTPUT = "output"  # Execution output
    MSG_REFLECTION = "reflection"  # Agent reflection
    MSG_SUMMARY = "summary"  # Final summary
    
    def __init__(self):
        # Force Agg backend for matplotlib to prevent GUI-related errors
        plt.switch_backend('Agg')
        
        # Ensure we use non-interactive mode every time
        plt.rcParams['interactive'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        self.dataframes = {}  # Storage for dataframes between executions
        self.execution_history = []
        self.step_count = 0  # Counter for execution steps (individual code runs)
        self.plan_step = 0   # Counter for logical plan steps
        self.step_mapping = {}  # Maps execution steps to plan steps
        self.plan_steps = []  # List of plan steps in order
        self.current_plan_step_index = -1  # Index of current plan step
        self.graph_steps = []  # List of plan steps that should create graphs
from datetime import datetime
import os
import sys
import asyncio
import time
import pandas as pd
import numpy as np
import re
import html
import json
import traceback
import io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import copy
import threading
import psutil
import math
import ast

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, 
    QParallelAnimationGroup, QSequentialAnimationGroup, pyqtSignal, pyqtSlot,
    QThread, QObject, QRect, QByteArray, QPointF
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QScrollArea, 
    QLabel, QLineEdit, QSplitter, QFrame, QSizePolicy, QSpacerItem,
    QTextEdit, QProgressBar, QGraphicsOpacityEffect, QStackedWidget, QFileDialog, QDialog,
    QGraphicsDropShadowEffect, QCheckBox, QRadioButton, QButtonGroup, QTabWidget, QGridLayout, QSpinBox,
    QApplication, QComboBox
)
from PyQt6.QtGui import (
    QIcon, QColor, QPainter, QPen, QBrush, QFont, QPalette, 
    QLinearGradient, QFontMetrics, QConicalGradient
)
from PyQt6.QtSvgWidgets import QSvgWidget

# Add necessary imports
from qasync import asyncSlot
from llms.client import call_llm_sync, call_llm_async

# Import TaskStatus from common module
from common.status import TaskStatus

# Import the icon loader
from helpers.load_icon import load_bootstrap_icon

# Define pastel color palette
PASTEL_COLORS = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFB3F7', '#B3FFF7']
PASTEL_CMAP = sns.color_palette(PASTEL_COLORS)
# Configure SIP to avoid bad catcher results
# To install SIP: pip install PyQt6-sip
try:
    import PyQt6.sip as sip
    # Check if the attribute exists before calling it
    if hasattr(sip, 'setdestroyonexit'):
        sip.setdestroyonexit(False)  # Don't destroy C++ objects on exit
    # Override the bad catcher result function if accessible
    if hasattr(sip, 'setBadCatcherResult'):
        # Pass None directly instead of a lambda function
        sip.setBadCatcherResult(None)
except ImportError:
    pass

# Set matplotlib backend
matplotlib.use('Agg', force=True)  # Force Agg backend

# Disable interactive mode
plt.ioff()

# Disable Qt-specific backend features
plt.switch_backend('Agg')

# Neutralize problematic socket handling in matplotlib
import socket
socket.socket = socket.socket


class TimelineTask:
    """Represents a task in the timeline."""
    def __init__(self, 
                 name: str, 
                 description: str = "", 
                 status: TaskStatus = TaskStatus.PENDING):
        self.name = name
        self.description = description
        self.status = status
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
    
    def validate_transition(self, new_status: TaskStatus) -> bool:
        """
        Validate if a status transition is allowed.
        """
        # If trying to set the same status, always allow it
        if self.status == new_status:
            return True
            
        # Define valid transitions
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.WARNING],
            TaskStatus.RUNNING: [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.WARNING],
            TaskStatus.COMPLETED: [],  # Terminal state
            TaskStatus.FAILED: [],     # Terminal state
            TaskStatus.WARNING: [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.FAILED]
        }
        
        return new_status in valid_transitions.get(self.status, [])
        
    def start(self):
        """Start the task."""
        if not self.validate_transition(TaskStatus.RUNNING):
            return False
            
        self.start_time = datetime.now()
        self.status = TaskStatus.RUNNING
        return True
        
    def complete(self):
        """Mark the task as completed."""
        if not self.validate_transition(TaskStatus.COMPLETED):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.COMPLETED
        return True
        
    def fail(self):
        """Mark the task as failed."""
        if not self.validate_transition(TaskStatus.FAILED):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.FAILED
        return True
        
    def warn(self):
        """Mark the task with a warning."""
        if not self.validate_transition(TaskStatus.WARNING):
            return False
            
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = TaskStatus.WARNING
        return True
        
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.end_time:
            return self.duration
        else:
            return (datetime.now() - self.start_time).total_seconds()
            
    def format_elapsed_time(self) -> str:
        """Format elapsed time as a string."""
        seconds = self.elapsed_time()
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remainder = seconds % 60
            return f"{int(minutes)}m {int(remainder)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"


class SpinnerWidget(QWidget):
    """Custom spinner animation widget."""
    def __init__(self, parent=None, size=24, color=Qt.GlobalColor.blue):
        super().__init__(parent)
        self.size = size
        self.color = color
        self.angle = 0
        self.setFixedSize(size, size)
        
        # Create timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.timer.start(40)  # Update slightly faster (was 50ms)
        
    def rotate(self):
        """Rotate the spinner."""
        self.angle = (self.angle + 12) % 360  # Slightly faster rotation (was 10)
        self.update()
        
    def paintEvent(self, event):
        """Paint the spinner."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate center and radius
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(center_x, center_y) - 1
        
        # Calculate spinner dimensions for the new compact design
        outer_radius = radius * 0.9
        dot_radius = radius * 0.2
        orbit_radius = radius * 0.6
        
        # Set up painter
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Create a more modern spinner with dots
        dot_count = 8
        for i in range(dot_count):
            # Calculate position on the circle
            angle = (self.angle + i * (360 / dot_count)) % 360
            rad_angle = angle * 3.14159 / 180
            
            # Calculate opacity based on position (fade effect)
            opacity = 0.25 + 0.75 * (1 - (i / dot_count))
            
            # Calculate color with opacity
            color = QColor(self.color)
            color.setAlphaF(opacity)
            painter.setBrush(color)
            
            # Calculate dot position
            x = center_x + orbit_radius * math.cos(rad_angle)
            y = center_y + orbit_radius * math.sin(rad_angle)
            
            # Draw the dot
            painter.drawEllipse(QPointF(x, y), dot_radius, dot_radius)


class TimelineTaskWidget(QFrame):
    """Widget that represents a task in the timeline with animated borders."""
    def __init__(self, task: TimelineTask, parent=None):
        super().__init__(parent)
        self.task = task
        self.spinner = None
        self.highlight_animation = None
        self.border_animation = None
        self.shadow_effect = None
        self.effects_container = None  # Container to hold our widget with effects
        
        # Create a timer to update elapsed time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        
        # Initialize the UI
        self.init_ui()
        self.setup_animations()
        
        # Update every second if task is running
        if task.status == TaskStatus.RUNNING:
            self.timer.start(1000)
            self.start_border_animation()
            
    def init_ui(self):
        """Initialize the UI with grid layout."""
        self.setObjectName("timelineTask")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        
        # Add shadow effect - but don't apply it directly to the widget yet
        # We'll handle it in setup_animations to avoid conflicts with highlight effect
        self.shadow_effect = QGraphicsDropShadowEffect()
        self.shadow_effect.setBlurRadius(8)
        self.shadow_effect.setColor(QColor(0, 0, 0, 40))
        self.shadow_effect.setOffset(0, 2)
        
        # Use compact grid layout 
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(4)
        
        # Status container with background circle for icon
        status_container = QWidget()
        status_container.setFixedSize(32, 32)
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add status icon at left with circle background
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(24, 24)
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_icon)
        
        # Add status container to main layout
        grid.addWidget(status_container, 0, 0, 2, 1, Qt.AlignmentFlag.AlignCenter)
        
        # Add task name at top row
        self.name_label = QLabel(self.task.name)
        self.name_label.setObjectName("taskName")
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        self.name_label.setFont(font)
        # Limit name length with ellipsis if too long
        self.name_label.setMaximumWidth(170)
        metrics = QFontMetrics(self.name_label.font())
        elided_text = metrics.elidedText(self.task.name, Qt.TextElideMode.ElideRight, 160)
        self.name_label.setText(elided_text)
        grid.addWidget(self.name_label, 0, 1)
        
        # Add task description as a smaller, lighter text if it's not empty
        if self.task.description and self.task.description.strip():
            self.desc_label = QLabel(self.task.description)
            self.desc_label.setObjectName("taskDescription")
            self.desc_label.setWordWrap(True)
            self.desc_label.setStyleSheet("font-size: 8pt; ")
            # Truncate long descriptions
            elided_desc = metrics.elidedText(self.task.description, Qt.TextElideMode.ElideRight, 160)
            self.desc_label.setText(elided_desc)
            grid.addWidget(self.desc_label, 1, 1)
        
        # Add elapsed time at bottom right
        self.time_label = QLabel(self.task.format_elapsed_time())
        self.time_label.setObjectName("taskTime")
        font = QFont()
        font.setPointSize(7)
        self.time_label.setFont(font)
        grid.addWidget(self.time_label, 2, 1, 1, 1, Qt.AlignmentFlag.AlignRight)
        
        # Update the status icon based on current status
        self.update_status_icon()
        
        # Set a fixed width to make the grid more uniform
        self.setMinimumWidth(180)
        self.setMaximumWidth(220)
        
        # Enhanced chip-like styling
        self.setStyleSheet("""
            QFrame#timelineTask {
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            QFrame#timelineTask:hover {
                border-color: #aaa;
            }
            QFrame#timelineTask[status="running"] {
                border-color: #4a6ee0;
            }
            QFrame#timelineTask[status="completed"] {
                border-color: #28a745;
            }
            QFrame#timelineTask[status="failed"] {
                border-color: #dc3545;
            }
            QFrame#timelineTask[status="warning"] {
                border-color: #ffc107;
            }
            QLabel#taskName {
                font-weight: bold;
            }
        """)
        
    def setup_animations(self):
        """Set up animations for the task widget."""
        # Use a different approach for handling both effects
        # Instead of setting graphics effects directly on the widget, we'll use property animations directly
        
        # Setup opacity animation for highlighting
        self.highlight_animation = QPropertyAnimation(self, b"windowOpacity")
        self.highlight_animation.setDuration(300)  # Faster animation
        self.highlight_animation.setStartValue(1.0)
        self.highlight_animation.setEndValue(0.7)
        self.highlight_animation.setLoopCount(2)  # Less loops for subtle effect
        
        # Setup shadow animation
        self.shadow_animation = QPropertyAnimation(self.shadow_effect, b"blurRadius")
        self.shadow_animation.setDuration(1500)
        self.shadow_animation.setStartValue(5)
        self.shadow_animation.setEndValue(15)
        self.shadow_animation.setLoopCount(-1)  # infinite
        self.shadow_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Apply shadow effect at the end of setup
        self.setGraphicsEffect(self.shadow_effect)
            
    def start_border_animation(self):
        """Start the border pulse animation for running tasks."""
        # Start shadow animation for running tasks
        if self.shadow_animation and self.shadow_effect:
            self.shadow_animation.stop()
            self.shadow_animation.start()
            
    def stop_border_animation(self):
        """Stop the border animation."""
        # Stop shadow animation
        if hasattr(self, 'shadow_animation') and self.shadow_animation:
            self.shadow_animation.stop()
            # Reset shadow to normal
            if hasattr(self, 'shadow_effect') and self.shadow_effect:
                self.shadow_effect.setBlurRadius(8)
            
    def update_time(self):
        """Update the elapsed time display."""
        self.time_label.setText(self.task.format_elapsed_time())
        
    def update_status_icon(self):
        """Update the status icon based on task status."""
        if self.task.status == TaskStatus.PENDING:
            # Pending icon (clock)
            icon_path = load_bootstrap_icon("clock")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.RUNNING:
            # Running icon (spinner)
            if not self.spinner:
                self.spinner = SpinnerWidget(self, size=24, color=QColor(74, 110, 224))  # Use theme blue color
                layout = self.layout()
                # Find the status icon inside the layout
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item.widget() and isinstance(item.widget(), QWidget) and item.widget().layout():
                        status_layout = item.widget().layout()
                        for j in range(status_layout.count()):
                            widget = status_layout.itemAt(j).widget()
                            if widget == self.status_icon:
                                # Hide the status icon when spinner is shown
                                self.status_icon.setVisible(False)
                                status_layout.replaceWidget(self.status_icon, self.spinner)
                                self.spinner.show()
                                break
            
            # Start timer if not already running
            if not self.timer.isActive():
                self.timer.start(1000)
                
            # Start border animation
            self.start_border_animation()
                
        elif self.task.status == TaskStatus.COMPLETED:
            # Completed icon (check)
            icon_path = load_bootstrap_icon("check-circle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.FAILED:
            # Failed icon (x)
            icon_path = load_bootstrap_icon("x-circle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        elif self.task.status == TaskStatus.WARNING:
            # Warning icon (!)
            icon_path = load_bootstrap_icon("exclamation-triangle-fill")
            self.status_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
            self.status_icon.setVisible(True)
            
            # Stop timer if it's running
            if self.timer.isActive():
                self.timer.stop()
                
            # Stop border animation
            self.stop_border_animation()
                
            # Remove spinner if it exists
            if self.spinner:
                self.spinner.deleteLater()
                self.spinner = None
                
        # Update property for styling
        self.setProperty("status", self.task.status.name.lower())
        self.style().polish(self)
    
    def update_status(self, status: TaskStatus):
        """Update the task status and refresh the UI."""
        if self.task.validate_transition(status):
            # Update task status
            self.task.status = status
            
            # Update UI
            self.update_status_icon()
            self.update_time()
            
            # Mark end time if status is terminal
            if status == TaskStatus.COMPLETED or status == TaskStatus.FAILED or status == TaskStatus.WARNING:
                self.task.end_time = datetime.now()
                self.task.duration = (self.task.end_time - self.task.start_time).total_seconds()


class TimelineWidget(QScrollArea):
    """Widget that displays a timeline of tasks."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_widgets = {}  # Map of task name -> widget
        self.column_count = 3  # Default number of chips per row
        self.current_row = 0
        self.current_col = 0
        self.init_ui()  # Call init_ui after initializing properties
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create container widget
        self.container = QWidget()
        self.setWidget(self.container)
        
        # Create grid layout instead of vertical layout
        self.layout = QGridLayout(self.container)
        self.layout.setContentsMargins(12, 12, 12, 12)  # Slightly more padding around edges
        self.layout.setSpacing(10)  # Moderate spacing between items
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        # Add a header label
        self.header_label = QLabel("Task Timeline")
        self.header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 8px;
        """)
        self.layout.addWidget(self.header_label, 0, 0, 1, self.column_count)
        
        # Start tasks at row 1 (after header)
        self.current_row = 1
        
        # Add a spacer item at the bottom to push content up
        self.spacer_item = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)  # High row number to ensure it's at the bottom
        
        # Connect resize event to adjust column count
        self.container.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        """Handle resize events to adjust column count."""
        if obj == self.container and event.type() == event.Type.Resize:
            self.adjust_column_count()
        return super().eventFilter(obj, event)
    
    def adjust_column_count(self):
        """Adjust column count based on container width."""
        width = self.container.width()
        # Calculate optimal columns based on chip width + spacing
        chip_width = 220  # Maximum width of chips
        spacing = self.layout.spacing()
        
        # Calculate how many chips can fit in the width (minimum 1)
        new_column_count = max(1, int(width / (chip_width + spacing)))
        
        # Only reorganize if column count changed
        if new_column_count != self.column_count:
            self.column_count = new_column_count
            self.reorganize_layout()
    
    def reorganize_layout(self):
        """Reorganize the task layout after column count change."""
        # Store widgets in order
        widgets = []
        for name, widget in self.task_widgets.items():
            widgets.append(widget)
        
        # Remove widgets from layout
        for widget in widgets:
            self.layout.removeWidget(widget)
        
        # Remove spacer
        self.layout.removeItem(self.spacer_item)
        
        # Reset position
        self.current_row = 1  # Start after header
        self.current_col = 0
        
        # Re-add widgets
        for widget in widgets:
            self.layout.addWidget(widget, self.current_row, self.current_col)
            self.current_col += 1
            if self.current_col >= self.column_count:
                self.current_col = 0
                self.current_row += 1
                
        # Update header span
        self.layout.removeWidget(self.header_label)
        self.layout.addWidget(self.header_label, 0, 0, 1, self.column_count)
        
        # Add spacer back
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)
        
    def add_task(self, task: TimelineTask):
        """
        Add a task to the timeline.
        
        Args:
            task: The task to add
        """
        # Create task widget
        task_widget = TimelineTaskWidget(task, self)
        
        # Store reference
        self.task_widgets[task.name] = task_widget
        
        # Remove spacer first
        self.layout.removeItem(self.spacer_item)
        
        # Add task widget to layout at the current position
        self.layout.addWidget(task_widget, self.current_row, self.current_col)
        
        # Update position for next task
        self.current_col += 1
        if self.current_col >= self.column_count:
            self.current_col = 0
            self.current_row += 1
        
        # Add spacer back at the bottom
        self.layout.addItem(self.spacer_item, 9999, 0, 1, self.column_count)
        
        # Scroll to show the new task
        self.scroll_to_task(task.name)
        
        return task_widget
        
    def update_task_status(self, task_name: str, status: TaskStatus):
        """
        Update a task's status.
        
        Args:
            task_name: The name of the task
            status: The new status
        """
        if task_name in self.task_widgets:
            self.task_widgets[task_name].update_status(status)
            
    def scroll_to_task(self, task_name: str):
        """
        Scroll to ensure a task is visible.
        
        Args:
            task_name: The name of the task
        """
        if task_name in self.task_widgets:
            task_widget = self.task_widgets[task_name]
            
            # Highlight the task
            self._highlight_task(task_widget)
            
            # Scroll to the widget
            task_geometry = task_widget.geometry()
            viewport_rect = self.viewport().rect()
            
            if task_geometry.top() < 0 or task_geometry.bottom() > viewport_rect.height():
                # Get the widget position in scrollarea coordinates
                task_pos = task_widget.pos()
                
                # Scroll to position
                self.ensureWidgetVisible(task_widget)
            
    def _highlight_task(self, task_widget):
        """
        Highlight a task briefly.
        
        Args:
            task_widget: The widget to highlight
        """
        # Start highlight animation
        if hasattr(task_widget, 'highlight_animation') and task_widget.highlight_animation:
            task_widget.highlight_animation.stop()
            task_widget.highlight_animation.start()


class RoundedChatWidget(QWidget):
    """Chat widget with rounded corners."""
    messageSent = pyqtSignal(str)
    fileUploadRequested = pyqtSignal(str)  # Signal for file upload
    suggestionClicked = pyqtSignal(str)  # Signal for suggestion clicks
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.typing_indicator_visible = False
        self.suggestions = []
        self.markdown = None
        
    def init_ui(self):
        """Initialize the UI with grid layouts."""
        # Set up main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create settings panel at the very top
        settings_panel = QWidget()
        settings_panel.setObjectName("settingsPanel")
        
        # Use horizontal layout for settings
        settings_layout = QHBoxLayout(settings_panel)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(15)
        
        # Create settings title
        settings_title = QLabel("Analysis Settings")
        settings_title.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(settings_title)
        
        # Plan depth control - already in horizontal layout
        plan_layout = QHBoxLayout()
        plan_layout.setSpacing(8)
        plan_label = QLabel("Plan Depth:")
        plan_label.setObjectName("settingLabel")
        plan_label.setFixedWidth(70)
        self.plan_depth_spinbox = QSpinBox()
        self.plan_depth_spinbox.setObjectName("settingSpinbox")
        self.plan_depth_spinbox.setMinimum(1)
        self.plan_depth_spinbox.setMaximum(10)
        self.plan_depth_spinbox.setValue(5)
        self.plan_depth_spinbox.setFixedWidth(45)
        self.plan_depth_spinbox.setToolTip("Analysis plan depth")
        plan_layout.addWidget(plan_label)
        plan_layout.addWidget(self.plan_depth_spinbox)
        
        # Max graphs control - already in horizontal layout
        graphs_layout = QHBoxLayout()
        graphs_layout.setSpacing(8)
        graphs_label = QLabel("Max Graphs:")
        graphs_label.setObjectName("settingLabel")
        graphs_label.setFixedWidth(70)
        self.max_graphs_spinbox = QSpinBox()
        self.max_graphs_spinbox.setObjectName("settingSpinbox")
        self.max_graphs_spinbox.setMinimum(1)
        self.max_graphs_spinbox.setMaximum(20)
        self.max_graphs_spinbox.setValue(10)
        self.max_graphs_spinbox.setFixedWidth(45)
        self.max_graphs_spinbox.setToolTip("Maximum graphs per step")
        graphs_layout.addWidget(graphs_label)
        graphs_layout.addWidget(self.max_graphs_spinbox)
        
        # Add both controls to the horizontal layout
        settings_layout.addLayout(plan_layout)
        settings_layout.addLayout(graphs_layout)
        settings_layout.addStretch()
        
        # Add settings panel to main layout
        main_layout.addWidget(settings_panel)
        
        # Create main content area for chat
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Create the chat display area
        self.chat_display = QScrollArea()
        self.chat_display.setObjectName("chatDisplay")
        self.chat_display.setWidgetResizable(True)
        self.chat_display.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create the container for chat messages
        self.chat_container = QWidget()
        self.chat_container.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(15)
        
        # Add an expanding spacer to push content to the top
        self.chat_layout.addStretch()
        
        # Set the container as the scroll area's widget
        self.chat_display.setWidget(self.chat_container)
        
        # Add chat display to content layout with stretch
        content_layout.addWidget(self.chat_display, 1)
        
        # Add content widget to main layout
        main_layout.addWidget(content_widget, 1)
        
        # Create the input area with rounded corners
        input_container = QWidget()
        input_container.setObjectName("inputContainer")
        
        # Use grid layout for input area
        input_grid = QGridLayout(input_container)
        input_grid.setContentsMargins(10, 5, 10, 5)
        input_grid.setSpacing(5)
        
        # Create thinking area for agent status indicators
        self.thinking_area = QWidget()
        self.thinking_area.setObjectName("thinkingArea")
        self.thinking_area.setMaximumHeight(80)  # Limit height but allow for multiple lines
        self.thinking_area.setMinimumHeight(40)
        
        thinking_layout = QVBoxLayout(self.thinking_area)
        thinking_layout.setContentsMargins(8, 5, 8, 5)
        thinking_layout.setSpacing(3)
        thinking_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        
        # Status indicator for agent actions
        self.status_indicator = QLabel()
        self.status_indicator.setObjectName("statusIndicator")
        self.status_indicator.setMaximumWidth(800)  # Prevent extreme stretching
        self.status_indicator.setWordWrap(True)     # Important: Enable word wrapping
        self.status_indicator.setStyleSheet("""
            #statusIndicator {
                
                font-style: italic;
                font-size: 0.9em;
                padding-left: 5px;
            }
        """)
        self.status_indicator.setVisible(False)
        thinking_layout.addWidget(self.status_indicator)
        
        # Add the thinking area to input grid, spanning the full width
        input_grid.addWidget(self.thinking_area, 0, 0, 1, 3)
        
        # Create typing indicator
        self.typing_indicator = QLabel("Agent is typing...")
        self.typing_indicator.setObjectName("typingIndicator")
        self.typing_indicator.setVisible(False)
        input_grid.addWidget(self.typing_indicator, 1, 0, 1, 3)
        
        # Create suggestion container
        self.suggestion_container = QWidget()
        self.suggestion_container.setObjectName("suggestionContainer")
        suggestion_layout = QHBoxLayout(self.suggestion_container)
        suggestion_layout.setContentsMargins(0, 0, 0, 10)
        suggestion_layout.setSpacing(5)
        suggestion_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        input_grid.addWidget(self.suggestion_container, 2, 0, 1, 3)
        self.suggestion_container.setVisible(False)
        
        # Create message input with rounded corners
        self.message_input = QTextEdit()
        self.message_input.setObjectName("messageInput")
        self.message_input.setMinimumHeight(50)
        self.message_input.setMaximumHeight(100)
        self.message_input.setPlaceholderText("Type a message...")
        self.message_input.textChanged.connect(self.adjust_input_height)
        input_grid.addWidget(self.message_input, 3, 1, 1, 1)
        
        # Create the file upload button
        self.upload_button = QPushButton()
        self.upload_button.setObjectName("uploadButton")
        self.upload_button.setToolTip("Upload File")
        self.upload_button.setFixedSize(36, 36)
        
        # Set icon for upload button
        upload_icon_path = load_bootstrap_icon("paperclip")
        if upload_icon_path:
            self.upload_button.setIcon(QIcon(upload_icon_path))
            self.upload_button.setIconSize(QSize(20, 20))
        
        self.upload_button.clicked.connect(self.select_file)
        input_grid.addWidget(self.upload_button, 3, 0, 1, 1)
        
        # Create the send button with rounded corners
        self.send_button = QPushButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setToolTip("Send Message")
        self.send_button.setFixedSize(36, 36)
        
        # Set icon for send button
        send_icon_path = load_bootstrap_icon("send-fill")
        if send_icon_path:
            self.send_button.setIcon(QIcon(send_icon_path))
            self.send_button.setIconSize(QSize(20, 20))
        
        self.send_button.clicked.connect(self.send_message)
        input_grid.addWidget(self.send_button, 3, 2, 1, 1)
        
        # Add widgets to main layout (content area and input container)
        main_layout.addWidget(content_widget, 1)  # Stretch factor
        main_layout.addWidget(input_container, 0) # No stretch
        
        # Add styling for the thinking area and status indicators
        self.setStyleSheet(self.styleSheet() + """
            #thinkingArea {
                border-radius: 4px;
            }
            
            #statusIndicator {
                font-style: italic;
                
            }
            
            #settingLabel {
                font-size: 12px;
            }
            
            #settingSpinbox {
                border-radius: 3px;
                padding: 2px;
                max-height: 24px;
            }
        """)
        
    def update_status(self, text, is_thinking=True):
        """Update the status indicator with agent's current action."""
        # Ensure text is not too long for display
        if len(text) > 100:
            text = text[:97] + "..."
            
        # Set the text and make visible
        if is_thinking:
            self.status_indicator.setText(f"<span style='color: #4a6ee0;'>⚙️</span> {text}")
        else:
            self.status_indicator.setText(text)
            
        self.status_indicator.setVisible(True)
        
        # Ensure the thinking area is visible
        self.thinking_area.setVisible(True)
        
        # Auto-hide after some time if not thinking
        if not is_thinking:
            QTimer.singleShot(5000, lambda: self.status_indicator.setVisible(False))
    
    def set_suggestions(self, suggestions):
        """
        Set suggestion chips.
        
        Args:
            suggestions: List of suggestion strings
        """
        # Clear current suggestions
        self.suggestions = suggestions
        
        # Clear suggestion container
        while self.suggestion_container.layout().count():
            item = self.suggestion_container.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new suggestions
        for suggestion in suggestions:
            chip = self.create_suggestion_chip(suggestion)
            self.suggestion_container.layout().addWidget(chip)
        
        # Show/hide container based on suggestions
        self.suggestion_container.setVisible(bool(suggestions))
    
    def create_suggestion_chip(self, text):
        """
        Create a clickable suggestion chip.
        
        Args:
            text: The suggestion text
            
        Returns:
            QPushButton: The suggestion chip button
        """
        chip = QPushButton(text)
        chip.setObjectName("suggestionChip")
        
        # Calculate width based on text length (approximate)
        font_metrics = QFontMetrics(chip.font())
        text_width = font_metrics.horizontalAdvance(text)
        
        # Set button size (adjust padding as needed)
        chip.setMinimumWidth(min(text_width + 20, 200))
        chip.setMaximumWidth(min(text_width + 40, 300))
        chip.setMinimumHeight(30)
        chip.setMaximumHeight(30)
        
        # Connect click signal
        chip.clicked.connect(lambda: self.handle_suggestion_click(text))
        
        return chip
    
    def handle_suggestion_click(self, suggestion):
        """
        Handle click on a suggestion chip.
        
        Args:
            suggestion: The suggestion text
        """
        # Emit signal with suggestion text
        self.suggestionClicked.emit(suggestion)
        
        # Clear suggestions
        self.set_suggestions([])
    
    def adjust_input_height(self):
        """Adjust the input height based on content."""
        # Calculate new height based on document size
        doc_height = self.message_input.document().size().height()
        
        # Ensure height is within min and max
        new_height = max(50, min(doc_height + 20, 100))
        
        # Set new height
        self.message_input.setMinimumHeight(int(new_height))
        self.message_input.setMaximumHeight(int(new_height))
    
    def select_file(self):
        """Open file dialog to select a file."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Data files (*.csv *.tsv *.xlsx *.txt)")
        
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.fileUploadRequested.emit(file_path)
            
    def send_message(self):
        """Send the current message."""
        # Get message text
        message = self.message_input.toPlainText().strip()
        
        # Clear input
        self.message_input.clear()
        
        # Emit signal if message is not empty
        if message:
            # Only emit the signal, don't add user message here as the receiver will handle that
            self.messageSent.emit(message)
    
    def add_user_message(self, message: str):
        """Add a user message to the chat display."""
        self.clean_json_formatting(message)
        self._add_message_bubble(message, is_user=True)
    
    def clean_json_formatting(self, text: str) -> str:
        """Clean up JSON content in the text."""
        # First handle JSON wrapped in triple backticks
        json_block_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```'
        
        def format_json_block(match):
            try:
                # Try to parse and pretty-print the JSON
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                formatted = json.dumps(parsed, indent=2)
                return f"```json\n{formatted}\n```"
            except json.JSONDecodeError:
                # If not valid JSON, return the original
                return match.group(0)
        
        # Replace JSON blocks with pretty-printed versions
        text = re.sub(json_block_pattern, format_json_block, text)
        
        # Also try to handle inline JSON objects (more challenging)
        # This is tricky because we need to avoid false positives
        inline_json_pattern = r'({[\s\S]*?})'
        
        def format_inline_json(match):
            try:
                # Try to parse the potential JSON
                json_str = match.group(1).strip()
                parsed = json.loads(json_str)
                
                # If it parsed successfully and is non-trivial, format it
                if isinstance(parsed, dict) and len(parsed) > 1:
                    formatted = json.dumps(parsed, indent=2)
                    return f"```json\n{formatted}\n```"
                else:
                    return match.group(0)
            except json.JSONDecodeError:
                # Not valid JSON, return original
                return match.group(0)
        
        # Only apply to large JSON-like blocks to avoid false positives
        if len(text) > 100 and '{' in text and '}' in text:
            # Check if the text is not already in a code block
            if '```' not in text:
                text = re.sub(inline_json_pattern, format_inline_json, text)
        
        return text
    
    def _add_message_bubble(self, message: str, is_user: bool):
        """
        Add a message bubble to the chat.
        
        Args:
            message: The message text
            is_user: True if the message is from the user, False if from the agent
        """
        # Calculate position to insert message (before the spacer)
        spacer_idx = self.chat_layout.count() - 1
        
        # Check if this is a plan or summary and use dedicated widget
        is_plan_or_summary = False
        if not is_user:
            # Check for plan content
            if ("<h1>Analysis Plan" in message or 
                "<h2>Analysis Plan" in message or 
                "# Analysis Plan" in message):
                self._add_professional_plan(message, spacer_idx)
                return
            
            # Check for summary content
            if ("<h1>Analysis Summary" in message or 
                "<h2>Analysis Summary" in message or 
                "# Analysis Summary" in message or
                "<h1>Summary of Analysis" in message or
                "# Summary of Analysis" in message):
                self._add_professional_summary(message, spacer_idx)
                return
                
        # Create the message bubble frame for normal messages
        bubble = QFrame()
        bubble.setObjectName("userBubble" if is_user else "agentBubble")
        
        # Check if this is a markdown message with headers (likely analysis plan or summary)
        is_special_content = False
        if not is_user and ("<h1>" in message or "<h2>" in message or 
                          "# " in message or "## " in message):
            is_special_content = True
            # Special styling for analysis plans and summaries
            bubble.setObjectName("specialContentBubble")
            bubble.setStyleSheet("""
                QFrame#specialContentBubble {
                    border: 1px solid #d1d5db;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 5px;
                    min-width: 95%;
                    max-width: 98%;
                }
            """)
        
        # Create layout for the bubble
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 10, 15, 10)
        
        # Create a QLabel for the message - simple plain text for user messages
        message_label = QLabel()
        message_label.setObjectName("messageLabel")
        message_label.setWordWrap(True)
        message_label.setOpenExternalLinks(True)
        message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        
        # Use different text format based on sender
        if is_user:
            # User messages are always plain text
            message_label.setTextFormat(Qt.TextFormat.PlainText)
            message_label.setText(message)
        else:
            # Agent messages might be HTML from markdown conversion
            message_label.setTextFormat(Qt.TextFormat.RichText)
            message_label.setText(message)
        
        # Add to bubble layout
        bubble_layout.addWidget(message_label)
        
        # Create container for alignment
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Align based on sender and content type
        if is_user:
            container_layout.addStretch()
            container_layout.addWidget(bubble)
        elif is_special_content:
            # Center special content (plans, summaries) with slight flexibility
            container_layout.addStretch(1)
            container_layout.addWidget(bubble, 10)  # Give bubble more stretch weight
            container_layout.addStretch(1)
        else:
            container_layout.addWidget(bubble)
            container_layout.addStretch()
        
        # Insert into chat layout
        self.chat_layout.insertWidget(spacer_idx, container)
        
        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def _add_professional_plan(self, content, insert_position):
        """Add a professionally formatted analysis plan using Qt widgets."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QFrame, QHBoxLayout, QScrollArea
        
        # Create plan frame
        plan_frame = QFrame()
        plan_frame.setObjectName("planFrame")
        plan_frame.setFrameShape(QFrame.Shape.StyledPanel)
        plan_frame.setStyleSheet("""
            QFrame#planFrame {
                border: 1px solid #4a6ee0;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 5px;
            }
        """)
        
        # Create frame layout
        plan_layout = QVBoxLayout(plan_frame)
        plan_layout.setContentsMargins(15, 15, 15, 15)
        plan_layout.setSpacing(10)
        
        # Add title
        title = QLabel("Analysis Plan")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        plan_layout.addWidget(title)
        
        # Create scroll area for steps
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setMaximumHeight(400)
        
        # Parse the plan content
        if content.startswith("# Analysis Plan") or content.startswith("<h1>Analysis Plan"):
            # Extract from markdown or HTML
            raw_content = ""
            if content.startswith("#"):
                # Markdown format
                raw_content = content.replace("# Analysis Plan", "").strip()
            else:
                # HTML format - strip HTML tags
                raw_content = re.sub(r'<[^>]*>', '', content)
                raw_content = raw_content.replace("Analysis Plan", "").strip()
            
            # Extract steps from content
            steps_container = QWidget()
            steps_layout = QVBoxLayout(steps_container)
            steps_layout.setContentsMargins(0, 0, 0, 0)
            steps_layout.setSpacing(10)
            
            # Find all steps using regex
            step_pattern = r"(?:Step|)\s*(\d+)[\.:\)]\s*(.*?)(?=(?:Step|)\s*\d+[\.:\)]|$)"
            steps = re.findall(step_pattern, raw_content, re.DOTALL)
            
            if steps:
                for step_num, step_desc in steps:
                    # Clean up the description
                    step_desc = step_desc.strip()
                    
                    # Check if it's a graph step
                    is_graph_step = "[GRAPH]" in step_desc
                    if is_graph_step:
                        # Keep it simple - no emojis
                        step_desc = step_desc.replace("[GRAPH]", "").strip()
                        step_label = QLabel(f"Step {step_num}: {step_desc}")
                        step_label.setStyleSheet("font-size: 15px; font-weight: bold;")
                    else:
                        step_label = QLabel(f"Step {step_num}: {step_desc}")
                        step_label.setStyleSheet("font-size: 15px; font-weight: bold;")
                    
                    step_label.setWordWrap(True)
                    steps_layout.addWidget(step_label)
            else:
                # No steps found, show raw content
                content_label = QLabel(raw_content)
                content_label.setWordWrap(True)
                content_label.setStyleSheet("font-size: 14px;")
                steps_layout.addWidget(content_label)
            
            # Set container as scroll area widget
            scroll_area.setWidget(steps_container)
            
            # Add scroll area to plan layout
            plan_layout.addWidget(scroll_area)
        else:
            # Fallback if parsing fails
            content_label = QLabel(content)
            content_label.setWordWrap(True)
            plan_layout.addWidget(content_label)
        
        # Create alignment container
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(plan_frame, 10)
        container_layout.addStretch(1)
        
        # Add to chat layout
        self.chat_layout.insertWidget(insert_position, container)
        
        # Scroll to bottom
        QTimer.singleShot(50, self.scroll_to_bottom)
    
    def _add_professional_summary(self, summary_data: dict):
        """Add a professionally formatted analysis summary using Qt widgets from parsed JSON data."""
        from PyQt6.QtWidgets import QLabel, QVBoxLayout, QFrame, QHBoxLayout, QScrollArea, QWidget
        
        # Calculate position to insert message (before the spacer)
        insert_position = self.chat_layout.count() - 1
        
        # Create summary frame
        summary_frame = QFrame()
        summary_frame.setObjectName("summaryFrame")
        summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_frame.setMinimumWidth(600)
        summary_frame.setStyleSheet("""
            QFrame#summaryFrame {
                border: 1px solid #28a745;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 5px;
            }
        """)
        
        # Create frame layout
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(15, 15, 15, 15)
        summary_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("Analysis Summary")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        summary_layout.addWidget(title_label)
        
        # --- Populate sections from JSON data --- 
        
        # Executive Summary
        exec_summary = summary_data.get("executive_summary", "Not available")
        exec_label = QLabel(f"<b>Executive Summary:</b> {exec_summary}")
        exec_label.setWordWrap(True)
        exec_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        summary_layout.addWidget(exec_label)
        
        # Scroll area for remaining details
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setMaximumHeight(400) # Keep height limit
        
        # Container for scrollable content
        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(15)
        
        # Key Findings
        key_findings = summary_data.get("key_findings", [])
        if key_findings:
            findings_label = QLabel("<b>Key Findings:</b>")
            findings_label.setStyleSheet("font-size: 15px; margin-bottom: 5px;")
            details_layout.addWidget(findings_label)
            for finding in key_findings:
                bullet_container = QWidget()
                bullet_layout = QHBoxLayout(bullet_container)
                bullet_layout.setContentsMargins(10, 0, 0, 0)
                bullet_layout.setSpacing(10)
                bullet = QLabel("•")
                bullet.setStyleSheet("font-weight: bold;")
                content = QLabel(finding)
                content.setWordWrap(True)
                content.setStyleSheet("font-size: 14px;")
                bullet_layout.addWidget(bullet, 0)
                bullet_layout.addWidget(content, 1)
                details_layout.addWidget(bullet_container)

        # Methodology
        methodology = summary_data.get("methodology", "Not available")
        method_label = QLabel(f"<b>Methodology:</b> {methodology}")
        method_label.setWordWrap(True)
        method_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(method_label)
        
        # Detailed Results (Handle potential markdown)
        detailed_results = summary_data.get("detailed_results", "Not available")
        results_label = QLabel("<b>Detailed Results:</b>")
        results_label.setStyleSheet("font-size: 15px; margin-bottom: 5px;")
        details_layout.addWidget(results_label)
        results_content = QLabel(self.markdown_to_html(detailed_results)) # Reuse markdown converter if needed
        results_content.setWordWrap(True)
        results_content.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(results_content)

        # Visualizations Summary
        viz_summary = summary_data.get("visualizations_summary", "Not available")
        viz_label = QLabel(f"<b>Visualizations Summary:</b> {viz_summary}")
        viz_label.setWordWrap(True)
        viz_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(viz_label)

        # Limitations
        limitations = summary_data.get("limitations", "Not available")
        limit_label = QLabel(f"<b>Limitations:</b> {limitations}")
        limit_label.setWordWrap(True)
        limit_label.setStyleSheet("font-size: 14px;")
        details_layout.addWidget(limit_label)
        
        # --- End of sections --- 
        
        details_layout.addStretch() # Push content up
        scroll_area.setWidget(details_container)
        summary_layout.addWidget(scroll_area)
        
        # Create alignment container for the whole summary frame
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(summary_frame, 10) # Give frame more stretch weight
        container_layout.addStretch(1)
        
        # Add to chat layout at the correct position
        self.chat_layout.insertWidget(insert_position, container)
        
        # Extract and set 'data_analysis_next_steps' as suggestions
        next_steps = summary_data.get("data_analysis_next_steps", [])
        if isinstance(next_steps, list) and all(isinstance(step, str) for step in next_steps):
            self.set_suggestions(next_steps)
        else:
            # Clear suggestions if data is invalid or missing
            self.set_suggestions([])
            print("Warning: 'data_analysis_next_steps' not found or invalid in summary data.")
        
        # Scroll to bottom (optional, as context/suggestions are added after)
        # QTimer.singleShot(50, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the chat display to the bottom."""
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_agent_message(self, message: str):
        """Add an agent message to the chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.add_agent_message(message)
        else:
            # Detect special content (analysis plans and summaries)
            is_special_content = False
            
            # Look for common patterns in analysis plans and summaries
            if message.startswith("# Analysis Plan") or message.startswith("# Analysis Summary") or \
               message.startswith("## Analysis Plan") or message.startswith("## Analysis Summary") or \
               message.startswith("# Summary of Analysis") or "## Step " in message:
                is_special_content = True
                
            # Process markdown before adding message
            # Check if content appears to be markdown
            if is_special_content or ("```" in message or "#" in message or "*" in message or 
                message.strip().startswith(">") or "- " in message):
                # Convert markdown to HTML
                html_content = self.markdown_to_html(message)
                self._add_message_bubble(html_content, is_user=False)
            else:
                # Plain text, no conversion needed
                self._add_message_bubble(message, is_user=False)
    
    def markdown_to_html(self, text: str) -> str:
        """Convert markdown to HTML using professional libraries."""
        try:
            # Use Python-Markdown with standard extensions
            import markdown
            
            # Basic set of extensions without custom processors
            extensions = [
                'markdown.extensions.fenced_code',
                'markdown.extensions.codehilite',
                'markdown.extensions.tables', 
                'markdown.extensions.nl2br',
                'markdown.extensions.sane_lists'
            ]
            
            # Add custom styling for headings to enhance readability
            text = self._preprocess_markdown_headings(text)
            
            # Convert markdown to HTML with standard settings
            html = markdown.markdown(text, extensions=extensions)
            
            # Apply additional styling to HTML elements
            html = self._postprocess_html(html)
            
            return html
        except Exception as e:
            # Log error and fall back to simple conversion
            print(f"Error converting markdown: {str(e)}")
            return text
            
    def _preprocess_markdown_headings(self, text: str) -> str:
        """Preprocess markdown headings to enhance their appearance."""
        # Add a class to heading lines for better styling
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('# '):
                lines[i] = line + ' {.h1-heading}'
            elif line.startswith('## '):
                lines[i] = line + ' {.h2-heading}'
            elif line.startswith('### '):
                lines[i] = line + ' {.h3-heading}'
                
        return '\n'.join(lines)
        
    def _postprocess_html(self, html: str) -> str:
        """Apply additional styling to HTML elements."""
        # Style headings for better visibility
        html = html.replace('<h1>', '<h1 style="font-size: 1.8em; margin-top: 20px; margin-bottom: 15px; text-align: center;">')
        html = html.replace('<h2>', '<h2 style="font-size: 1.5em; margin-top: 15px; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;">')
        html = html.replace('<h3>', '<h3 style="font-size: 1.3em; margin-top: 10px; margin-bottom: 8px;">')
        
        # Style code blocks for better readability
        html = html.replace('<pre><code>', '<pre style="border: 1px solid #e1e4e8; border-radius: 4px; padding: 10px; overflow-x: auto;"><code style="font-family: monospace;">')
        
        # Style lists for better spacing
        html = html.replace('<ul>', '<ul style="margin-left: 15px; line-height: 1.4;">')
        html = html.replace('<ol>', '<ol style="margin-left: 15px; line-height: 1.4;">')
        
        return html
    
    def set_typing_indicator(self, visible: bool):
        """Show or hide the typing indicator."""
        self.typing_indicator.setVisible(visible)
        self.typing_indicator_visible = visible


class GraphWidget(QWidget):
    """Widget for displaying graphs and visualizations."""
    
    def __init__(self, parent=None):
        """Initialize the graph widget."""
        super().__init__(parent)
        
        # Initialize properties
        self.current_zoom = 1.0
        self.current_figure = None
        self.figure = None
        self.svg_string = None
        self.svg_widget = None
        self.canvas = None
        self.aspect_widget = None
        self.layout = None  # Will be set in init_ui
        
        # Test QSvgWidget is functioning with better error handling
        try:
            test_svg = QSvgWidget()
            # Check if the widget was created successfully
            if test_svg is None:
                raise Exception("Failed to create QSvgWidget instance")
                
            # Create a minimal valid SVG to test if rendering works
            minimal_svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="10" height="10" xmlns="http://www.w3.org/2000/svg">
  <rect width="10" height="10" fill="gray"/>
</svg>"""
            test_svg.renderer().load(QByteArray(minimal_svg.encode('utf-8')))
            print("SVG widget initialized and functioning correctly")
        except Exception as e:
            print(f"Warning: SVG widget initialization issue: {e}")
            print("Will use fallback rendering methods for visualizations")
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        # Create main layout and store as instance variable
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Create empty widget as placeholder
        self.placeholder = QLabel("No visualization available")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("""
            font-size: 16px;
            font-style: italic;
            border: 1px dashed;
            border-radius: 5px;
            padding: 40px;
            margin: 20px;
        """)
        
        self.layout.addWidget(self.placeholder)
    
    def clear(self):
        """Clear current visualization."""
        try:
            # Only clear layout if it exists and has items
            if hasattr(self, 'layout') and self.layout:
                # Remove all widgets from the layout
                while self.layout.count():
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            else:
                # Create layout if none exists
                self.layout = QVBoxLayout(self)
                self.layout.setContentsMargins(10, 10, 10, 10)
            
            # Add placeholder back
            self.placeholder = QLabel("No visualization available")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.placeholder.setStyleSheet("""
                font-size: 16px;
                font-style: italic;
                border: 1px dashed;
                border-radius: 5px;
                padding: 40px;
                margin: 20px;
            """)
            
            self.layout.addWidget(self.placeholder)
            
            # Reset references
            self.figure = None
            self.svg_string = None
            self.svg_widget = None
            self.canvas = None
        except Exception as e:
            print(f"Error clearing graph widget: {str(e)}")
            traceback.print_exc()
    
    def show_matplotlib_figure(self, figure):
        """Display a matplotlib figure using SVG for better threading safety.
        
        This method:
        1. Clears the current layout
        2. Applies pastel colors for better visibility
        3. Converts the figure to SVG (thread-safe format)
        4. Creates an SVG widget for display
        5. Adds maximize button in top-right
        
        Args:
            figure (matplotlib.figure.Figure): The figure to display
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clear existing content but only if we have the expected properties
            self.clear()
            
            # Store the figure reference
            self.current_figure = figure
            
            # Apply pastel colors for better visibility
            self._apply_pastel_colors(figure)
            
            # Create a BytesIO buffer for the SVG data
            buf = io.BytesIO()
            
            # Ensure the figure has a reasonable size before saving to SVG
            original_size = figure.get_size_inches()
            figure.set_size_inches(10, 7)  # Standardize size for consistency
            
            # Save figure as SVG to buffer with explicit size
            figure.savefig(buf, format='svg', bbox_inches='tight', dpi=100)
            buf.seek(0)
            svg_data = buf.getvalue().decode('utf-8')

            buf.close()
            
            # Restore original figure size
            figure.set_size_inches(original_size)
            
            # Create main layout - only create if no layout exists
            if not self.layout:
                self.layout = QVBoxLayout(self)
                self.layout.setContentsMargins(10, 10, 10, 10)
            else:
                # Remove all items from the layout 
                while self.layout.count() > 0:
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
            
            # Create a frame for the SVG content
            svg_frame = QFrame()
            svg_frame.setFrameShape(QFrame.Shape.StyledPanel)
            svg_frame.setObjectName("svgFrame")
            # Set minimum size to ensure visibility
            svg_frame.setMinimumSize(400, 300)
            
            # Create layout for the SVG content with title and maximize button
            svg_layout = QVBoxLayout(svg_frame)
            svg_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create top bar with title and maximize button
            top_bar = QWidget()
            top_layout = QHBoxLayout(top_bar)
            top_layout.setContentsMargins(0, 0, 0, 5)
            
            # Add title (if figure has a title)
            title_text = "Visualization"
            for ax in figure.get_axes():
                if ax.get_title():
                    title_text = ax.get_title()
                    break
                    
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            top_layout.addWidget(title_label)
            
            # Add spacer to push maximize button to the right
            top_layout.addStretch()
            
            # Add maximize button
            maximize_btn = QPushButton()
            maximize_btn.setIcon(QIcon(load_bootstrap_icon("arrows-fullscreen")))
            maximize_btn.setToolTip("Maximize")
            maximize_btn.setFixedSize(24, 24)
            maximize_btn.setFlat(True)
            
            maximize_btn.clicked.connect(self._show_fullscreen_dialog)
            top_layout.addWidget(maximize_btn)
            
            # Add top bar to svg layout
            svg_layout.addWidget(top_bar)
            
            # Create the SVG widget and load the data
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Create aspect ratio maintaining container
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            
            # Add to layout
            svg_layout.addWidget(aspect_widget)
            
            # Add to main layout 
            self.layout.addWidget(svg_frame)
            
            # Store references to widgets for later manipulation
            self.svg_widget = svg_widget
            self.svg_data = svg_data  # Store the SVG data for fullscreen display
            self.aspect_widget = aspect_widget
            
            return True
        
        except Exception as e:
            error_layout = QVBoxLayout()
            error_label = QLabel(f"Error displaying figure: {str(e)}")
            error_label.setStyleSheet("color: red;")
            error_layout.addWidget(error_label)
            
            # Only set layout if none exists
            if not self.layout:
                self.setLayout(error_layout)
            else:
                # Clear and add error
                while self.layout.count():
                    item = self.layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.layout.addWidget(error_label)
                
            traceback.print_exc()
            return False
    
    def _apply_pastel_colors(self, figure):
        """Apply pastel colors to the figure for better visibility."""
        for ax in figure.get_axes():
            # Check if it's a plot with lines
            lines = ax.get_lines()
            if lines:
                for i, line in enumerate(lines):
                    color_idx = i % len(PASTEL_COLORS)
                    line.set_color(PASTEL_COLORS[color_idx])
            
            # Check if it's a bar chart
            patches = ax.patches
            if patches:
                for i, patch in enumerate(patches):
                    color_idx = i % len(PASTEL_COLORS)
                    patch.set_facecolor(PASTEL_COLORS[color_idx])
    
    def _show_fullscreen_dialog(self):
        """Show the current figure in fullscreen."""
        try:
            if not self.current_figure:
                return False
                
            # Create a fullscreen dialog
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
            dialog = QDialog(self)
            dialog.setWindowTitle("Visualization - Fullscreen")
            dialog.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowMaximizeButtonHint)
            dialog.resize(1200, 900)  # Larger default size
            
            # Create layout
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Use the stored SVG data if available
            if hasattr(self, 'svg_data') and self.svg_data:
                svg_widget = QSvgWidget()
                svg_widget.renderer().load(QByteArray(self.svg_data.encode('utf-8')))
                svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                
                # Create aspect ratio maintaining container
                aspect_widget = SVGAspectRatioWidget(svg_widget)
                
                # Add to layout
                layout.addWidget(aspect_widget)
            else:
                # Generate SVG data from the figure
                try:
                    buf = io.BytesIO()
                    self.current_figure.savefig(buf, format='svg', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    svg_data = buf.getvalue().decode('utf-8')
                    buf.close()
                    
                    # Create SVG widget
                    svg_widget = QSvgWidget()
                    svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
                    svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    
                    # Create aspect ratio maintaining container
                    aspect_widget = SVGAspectRatioWidget(svg_widget)
                    
                    # Add to layout
                    layout.addWidget(aspect_widget)
                except Exception as e:
                    error_label = QLabel(f"Error displaying figure: {str(e)}")
                    error_label.setStyleSheet("color: red; font-weight: bold;")
                    layout.addWidget(error_label)
            
            # Add close button
            close_button = QPushButton("Close")
          
            close_button.clicked.connect(dialog.close)
            close_button.setFixedWidth(100)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            # Display modal dialog
            dialog.exec()
            
            return True
        except Exception as e:
            traceback.print_exc()
            return False


class ReflectionWidget(QFrame):
    """Widget for displaying agent reflections with special formatting."""
    
    def __init__(self, parent=None, has_error=False):
        super().__init__(parent)
        self.has_error = has_error
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        self.setObjectName("reflectionWidget")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Set border color based on error status
        border_color = "#e74c3c" if self.has_error else "#f39c12"
        self.setStyleSheet(f"""
            QFrame#reflectionWidget {{
                border: 2px solid {border_color};
                border-radius: 4px;
                margin: 8px 0px;
                padding: 15px;
            }}
        """)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Add header
        header_layout = QHBoxLayout()
        
        # Add thinking icon
        thinking_icon = QLabel()
        icon_path = load_bootstrap_icon("lightbulb")
        if icon_path:
            thinking_icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
        
        # Add header label
        header_text = "Error Analysis" if self.has_error else "Reflection"
        header_label = QLabel(header_text)
        header_label.setStyleSheet(f"""
            font-weight: bold;
            font-size: 14px;
            color: {border_color};
        """)
        
        header_layout.addWidget(thinking_icon)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Add content label
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("""
            font-style: italic;
            line-height: 1.4;
            padding: 10px 0;
        """)
        self.content_label.setTextFormat(Qt.TextFormat.RichText)
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        # Add widgets to layout
        layout.addLayout(header_layout)
        layout.addWidget(self.content_label)
        
    def set_content(self, text):
        """Set the reflection content."""
        # Format the text with line breaks for better readability
        formatted_text = text.replace('\n', '<br>')
        self.content_label.setText(formatted_text)


class DynamicContentWidget(QStackedWidget):
    """Widget for displaying dynamic content like graphs and data tables."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget_names = {}  # Map of name -> index
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Create an empty placeholder widget
        placeholder = QWidget()
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_label = QLabel("No content to display")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("""
            font-size: 16px;
            font-style: italic;
            margin: 20px;
            padding: 20px;
        """)
        placeholder_layout.addWidget(placeholder_label)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add the placeholder widget
        self.addWidget(placeholder)
        self.widget_names["placeholder"] = 0
    
    def add_widget(self, widget, name):
        """
        Add a widget to the stacked widget.
        
        Args:
            widget: The widget to add
            name: A name for the widget
        """
        # If we already have a widget with this name, remove it
        if name in self.widget_names:
            self.remove_widget(self.widget(self.widget_names[name]))
        
        # Add the new widget
        index = self.addWidget(widget)
        self.widget_names[name] = index
        
        # Show the new widget
        self.setCurrentIndex(index)
        
        return index
    
    def remove_widget(self, widget):
        """
        Remove a widget from the stacked widget.
        
        Args:
            widget: The widget to remove
        """
        # Find the widget index
        index = self.indexOf(widget)
        if index != -1:
            # Find the name
            name = None
            for key, value in self.widget_names.items():
                if value == index:
                    name = key
                    break
            
            # Remove the widget
            self.removeWidget(widget)
            
            # Update the map
            if name:
                del self.widget_names[name]
                
            # Update indices in the map
            for key, value in self.widget_names.items():
                if value > index:
                    self.widget_names[key] = value - 1
            
            # Delete the widget later
            widget.deleteLater()


class DataAnalysisAgent:
    """Agent for performing iterative data analysis based on user queries."""
    
    # Messages types for output formatting
    MSG_THINKING = "thinking"  # Thinking/control messages
    MSG_EXECUTING = "executing"  # Execution messages
    MSG_CODE = "code"  # Code snippets
    MSG_OUTPUT = "output"  # Execution output
    MSG_REFLECTION = "reflection"  # Agent reflection
    MSG_SUMMARY = "summary"  # Final summary
    
    def __init__(self):
        # Force Agg backend for matplotlib to prevent GUI-related errors
        plt.switch_backend('Agg')
        
        # Ensure we use non-interactive mode every time
        plt.rcParams['interactive'] = False
        plt.rcParams['figure.max_open_warning'] = 0
        
        self.dataframes = {}  # Storage for dataframes between executions
        self.execution_history = []
        self.step_count = 0  # Counter for execution steps (individual code runs)
        self.plan_step = 0   # Counter for logical plan steps
        self.step_mapping = {}  # Maps execution steps to plan steps
        self.plan_steps = []  # List of plan steps in order
        self.current_plan_step_index = -1  # Index of current plan step
        self.graph_steps = []  # List of plan steps that should create graphs
        self.plan_depth = 5  # Default number of plan steps to generate
        
        # Add reflection tracking
        self.reflection_mapping = {}  # Maps step_number -> reflection_content
        
        # Override default show function to avoid GUI interactions
        self._original_show = plt.show
        plt.show = lambda *args, **kwargs: None
    
    async def execute(self, task, output_callback=None):
        """Execute an analysis on the given data."""
        is_continuation = "ANALYSIS_CONTINUATION:" in task
        
        # Initialize output callback if not provided
        if output_callback is None:
            output_callback = lambda message, msg_type=None: print(message)
            
        # Initialize state tracking
        state = {
            "task": task,
            "plan": "",
            "completed_steps": [],
            "current_step": 0,
            "step_count": 0,
            "graph_steps": [],
            "graph_types": [],
            "generated_graphs": [],
            "is_continuation": is_continuation
        }
        
        # Extract previous summary if this is a continuation
        previous_summary = ""
        if is_continuation:
            # Use the robust regex pattern
            summary_match = re.search(r'PREVIOUS_SUMMARY:[\s\n]*(.*?)(?=\n\n\w+:|$)', task, re.DOTALL)
            if summary_match:
                previous_summary = summary_match.group(1)
                state["previous_summary"] = previous_summary
                
            # Extract continuation question with flexible pattern
            question_match = re.search(r'(?:Follow-up|Continuation).*?question:[\s\n]*(.*?)(?=\n\w+:|$)', task, re.IGNORECASE | re.DOTALL)
            if question_match:
                state["continuation_question"] = question_match.group(1).strip()
                
            # Extract context level
            context_match = re.search(r'Context level:[\s\n]*(.*?)(?=\n\w+:|$)', task, re.IGNORECASE | re.DOTALL)
            if context_match:
                state["context_level"] = context_match.group(1).strip()
                
            # Check if continuation mode is explicitly mentioned
            continuation_mode_match = re.search(r'Continuation mode:[\s\n]*(.*?)(?=\n\w+:|$)', task, re.IGNORECASE | re.DOTALL)
            if continuation_mode_match:
                mode = continuation_mode_match.group(1).strip().lower()
                # Override is_continuation if explicitly disabled
                if mode in ["disabled", "off", "false", "no"]:
                    is_continuation = False
                    state["is_continuation"] = False
                    print("Continuation mode explicitly disabled")
        
        # Reset step tracking for new execution
        self.step_count = 0
        self.plan_step = 0
        self.step_mapping = {}
        self.plan_steps = []
        self.current_plan_step_index = -1
        self.graph_steps = []
        
        # Clear reflection mapping
        self.reflection_mapping = {}
        
        # Initial planning step
        print("Getting analysis plan...")
        plan_data = await self._get_analysis_plan(task)
        if output_callback:
            plan_str = plan_data["plan"] if isinstance(plan_data, dict) else str(plan_data)
            # Ensure the plan is properly sent to the UI
            print(f"Sending plan to UI, length: {len(plan_str)}")
            output_callback(f"🔍 Analysis Plan:\n{plan_str}\n", "plan")
        
        # Extract plan steps and their graph indicators
        if isinstance(plan_data, dict) and "plan" in plan_data:
            plan_text = plan_data["plan"]
            self._extract_plan_steps(plan_text)
            
            # Also copy the graph_steps if provided in the plan data
            if "graph_steps" in plan_data and isinstance(plan_data["graph_steps"], list):
                # Filter out step 0 from graph steps
                self.graph_steps = [step for step in plan_data["graph_steps"] if step != 0]
        
        # Track our current state and progress
        current_state = {
            "task": task,
            "plan": plan_data,
            "dataframes": self.dataframes.copy(),  # Copy existing dataframes to ensure they're included
            "current_step": 0,
            "completed_steps": [],
            "outputs": [],
            "graph_steps": self.graph_steps.copy(),
            "generated_graphs": set(),
            "graph_types": {},
            "plan_step_mapping": {},  # Maps execution steps to plan steps
            "current_plan_step": 0,   # Current logical plan step being executed
            "reflections": {}         # Track reflections for each step
        }
        
        # Move to the first plan step
        self._advance_to_next_plan_step()
        
        # Output initial plan step info
        if output_callback:
            output_callback(f"\nStarting with Plan Step {self.plan_step}\n", self.MSG_THINKING)
                
        # Execute steps until completion
        while True:
            self.step_count += 1
            
            # Map this execution step to the current plan step
            self.step_mapping[self.step_count] = self.plan_step
            current_state["plan_step_mapping"][self.step_count] = self.plan_step
            
            if output_callback:
                output_callback(f"\nExecuting Step {self.step_count} (Plan Step {self.plan_step})...\n", self.MSG_EXECUTING)
            
            # Debug dataframes before get_next_code
            print(f"Dataframes before step {self.step_count}: {list(self.dataframes.keys())}")
            
            # Make sure current state has latest dataframes
            current_state["dataframes"] = self.dataframes.copy()
            
            # Get next code to execute
            code_to_execute = await self._get_next_code(current_state)
            
            if "ANALYSIS_COMPLETE" in code_to_execute:
                if output_callback:
                    output_callback("\nAnalysis Complete\n", self.MSG_THINKING)
                break
                
            # Show the code we're about to execute
            if output_callback:
                output_callback(f"```python\n{code_to_execute}\n```\n", self.MSG_CODE)
                
            # Execute the code
            result, output, error = self._execute_code(code_to_execute)
            
            # Debug dataframes after execution
            print(f"Dataframes after step {self.step_count}: {list(self.dataframes.keys())}")
            
            # Update state with execution results
            current_state["outputs"].append({
                "step": self.step_count,
                "plan_step": self.plan_step,
                "code": code_to_execute,
                "output": output,
                "error": error
            })
            
            # Make sure state has latest dataframes
            current_state["dataframes"] = self.dataframes.copy()
            
            # Print output and errors
            if output and output_callback:
                output_callback(f"Output:\n{output}\n", self.MSG_OUTPUT)
            if error and output_callback:
                output_callback(f"Error:\n{error}\n", self.MSG_OUTPUT)
                
            # Reflect on the results and plan next step
            reflection = await self._reflect_on_execution(current_state)
            
            # Store the reflection specifically associated with this step
            if reflection:
                reflection_text = ""
                if isinstance(reflection, dict) and "reflection" in reflection:
                    reflection_text = reflection["reflection"]
                elif isinstance(reflection, str):
                    reflection_text = reflection
                    
                if reflection_text.strip():
                    # Store in agent's reflection mapping
                    self.reflection_mapping[self.step_count] = reflection_text
                    # Also store in current state
                    current_state["reflections"][self.step_count] = reflection_text
            
            if output_callback:
                output_callback(f"Reflection:\n{reflection}\n", self.MSG_REFLECTION)
                
            current_state["completed_steps"].append({
                "step": self.step_count,
                "plan_step": self.plan_step,
                "code": code_to_execute,
                "output": output,
                "error": error,
                "reflection": reflection
            })
            current_state["current_step"] += 1
            
            # Check if we should advance to the next plan step
            # Only advance if execution was successful (no error) and we're not in recovery
            recovery_needed = False
            if isinstance(reflection, dict) and "recovery_needed" in reflection:
                recovery_needed = reflection["recovery_needed"]
            
            if not error and not recovery_needed and self._should_advance_plan_step(current_state):
                self._advance_to_next_plan_step()
                if output_callback:
                    if self.plan_step <= len(self.plan_steps):
                        output_callback(f"\n🔍 Moving to Plan Step {self.plan_step}\n", self.MSG_THINKING)
            
        # Final summary
        summary = await self._get_final_summary(current_state)
        if output_callback:
            output_callback(f"\n{summary}\n", self.MSG_SUMMARY)
        
        return summary
        
    def _extract_plan_steps(self, plan_text):
        """Extract the plan steps and determine which should generate graphs."""
        self.plan_steps = []
        self.graph_steps = []
        
        # Extract step numbers and determine if they're graph steps
        step_pattern = r"(?:Step\s*)?(\d+)[\.:\)]\s*(.*?)(?=(?:Step\s*)?(?:\d+)[\.:\)]|$)"
        
        for match in re.finditer(step_pattern, plan_text, re.DOTALL):
            plan_step_num = int(match.group(1))
            step_text = match.group(2).strip()
            
            # Skip step 0 as it's typically just planning/reflection
            if plan_step_num == 0:
                print(f"Skipping step 0 (planning step)")
                continue
                
            self.plan_steps.append(plan_step_num)
            
            # Check if this is a graph step
            is_graph_step = "[GRAPH]" in step_text or any(word in step_text.lower() 
                                                         for word in ["plot", "chart", "graph", "visual", 
                                                                     "figure", "histogram", "scatter", "bar"])
            if is_graph_step and plan_step_num not in self.graph_steps:
                self.graph_steps.append(plan_step_num)
        
        # Sort the steps 
        self.plan_steps.sort()
        self.graph_steps.sort()
        
        # If no valid steps found (rare case), add a default step 1
        if not self.plan_steps:
            self.plan_steps = [1]
            
        print(f"Extracted plan steps: {self.plan_steps}")
        print(f"Identified graph steps: {self.graph_steps}")
    
    def _advance_to_next_plan_step(self):
        """Advance to the next plan step."""
        self.current_plan_step_index += 1
        if self.current_plan_step_index < len(self.plan_steps):
            self.plan_step = self.plan_steps[self.current_plan_step_index]
        else:
            # If we've gone beyond defined steps, just increment
            self.plan_step += 1
        print(f"Advanced to plan step {self.plan_step} (index {self.current_plan_step_index})")
    
    def _should_advance_plan_step(self, state):
        """Determine if we should advance to the next plan step."""
        # Get the last step reflection
        if not state["completed_steps"]:
            return False
            
        last_step = state["completed_steps"][-1]
        
        # Don't advance if error in the last execution
        if last_step.get("error"):
            print(f"Not advancing plan step due to error in step {self.step_count}")
            return False
            
        # Don't advance if in recovery mode
        reflection = last_step.get("reflection", {})
        if isinstance(reflection, dict) and reflection.get("recovery_needed", False):
            print(f"Not advancing plan step due to recovery needed in step {self.step_count}")
            return False
        
        # Check if the current plan step was one that should generate graphs
        if self.plan_step in self.graph_steps:
            print(f"Current plan step {self.plan_step} is a graph step")
            
            # If this was a graph step, check if "figure" was created successfully
            # Look in the output for signs of a successful visualization
            output = last_step.get("output", "")
            if isinstance(output, str):
                # If figures were explicitly created, allow advancement
                if "[FIGURE:" in output:
                    print(f"Advancing plan step {self.plan_step} because figures were created")
                    return True
                # Or if there's output indicating a visualization was created
                if ("Figure" in output or "figure" in output or "plot" in output or 
                    "chart" in output or "graph" in output or "visualization" in output):
                    print(f"Advancing plan step {self.plan_step} due to visualization keywords in output")
                    return True
                    
            # Check code for visualization indicators
            code = last_step.get("code", "")
            if isinstance(code, str):
                # If code has visualization commands and execution was successful, likely completed
                viz_terms = ["plt.figure", "plt.plot", "plt.bar", "plt.scatter", 
                              "sns.heatmap", "sns.lineplot", "ax.plot"]
                if any(viz_term in code for viz_term in viz_terms) and not last_step.get("error"):
                    print(f"Advancing plan step {self.plan_step} due to visualization code")
                    return True
                
            # If execution steps in this plan step exceed limit, advance anyway
            execution_steps_in_current_plan = 0
            for step in state["completed_steps"]:
                if step.get("plan_step") == self.plan_step:
                    execution_steps_in_current_plan += 1
            
            if execution_steps_in_current_plan >= 3:
                print(f"Advancing plan step {self.plan_step} after {execution_steps_in_current_plan} attempts")
                return True
                    
            print(f"Not advancing graph step {self.plan_step} yet (no visualization detected)")
            return False  # Don't advance if graph step hasn't produced a graph yet
            
        # For non-graph steps, check if there's a clear completion indicator
        output = last_step.get("output", "")
        if isinstance(output, str):
            completion_indicators = [
                "completed", "finished", "done", "analysis", "processed",
                "calculated", "computed", "results:", "summary:", "statistics:"
            ]
            if any(indicator in output.lower() for indicator in completion_indicators):
                print(f"Advancing plan step {self.plan_step} due to completion indicators")
                return True
        
        # For non-graph steps without clear completion, make a reasonable guess
        # based on accumulated steps within this plan step
        execution_steps_in_current_plan = 0
        for step in state["completed_steps"]:
            if step.get("plan_step") == self.plan_step:
                execution_steps_in_current_plan += 1
        
        # If we've done more than 2 executions in this plan step and the last one succeeded,
        # it's reasonable to advance to the next plan step
        if execution_steps_in_current_plan >= 2:
            print(f"Advancing plan step {self.plan_step} after {execution_steps_in_current_plan} executions")
            return True
            
        # Default case - don't advance
        return False
    
    def needs_plan_step_increment(self, state):
        """Determine if we should increment the plan step counter based on state.
        
        Note: This is superseded by _should_advance_plan_step and is kept for compatibility.
        """
        return self._should_advance_plan_step(state)
        
    async def _get_analysis_plan(self, task):
        """Generate an analysis plan for the given task."""
        # Check if this is a continuation request
        is_continuation = "ANALYSIS_CONTINUATION:" in task
        
        # Create a prompt for the analysis plan
        if is_continuation:
            # Extract previous summary with more robust regex
            previous_summary = ""
            summary_match = re.search(r'PREVIOUS_SUMMARY:[\s\n]*(.*?)(?=\n\n\w+:|$)', task, re.DOTALL)
            if summary_match:
                previous_summary = summary_match.group(1)
                
            # Extract continuation question with more flexible pattern
            continuation_question = "Continue the analysis"
            question_match = re.search(r'(?:Follow-up|Continuation).*?question:[\s\n]*(.*?)(?=\n\w+:|$)', task, re.IGNORECASE | re.DOTALL)
            if question_match:
                continuation_question = question_match.group(1).strip()
                
            # Extract context level
            context_level = "all"
            context_match = re.search(r'Context level:[\s\n]*(.*?)(?=\n\w+:|$)', task, re.IGNORECASE | re.DOTALL)
            if context_match:
                context_level = context_match.group(1).strip()
            
            print(f"Continuation analysis with question: '{continuation_question}', context level: {context_level}")
            
            # Create continuation prompt with context level awareness
            context_instruction = ""
            if context_level == "last":
                context_instruction = "Focus ONLY on the most recent summary without considering older analysis steps."
            elif context_level == "last2" or context_level == "last3":
                context_instruction = f"Focus primarily on the {context_level.replace('last', '')} most recent summaries."
            else:
                context_instruction = "Consider the full context from all previous analysis steps."
            
            # Create continuation prompt - modified to explicitly request numbered steps
            prompt = f"""Based on this follow-up question: "{continuation_question}"
            
            Create a detailed analysis plan that continues from the previous analysis.
            
            Previous analysis summary:
            {previous_summary}
            
            Context instruction: {context_instruction}
            
            Create a detailed analysis plan with exactly {self.plan_depth} numbered steps (Step 1, Step 2, etc.).
            For each step:
            1. Provide a clear description of what to analyze
            2. Indicate with [GRAPH] tag if the step should generate a visualization
            3. Make sure each step builds on previous findings
            4. Each step must be clearly numbered for execution
            
            IMPORTANT: Format as numbered steps, with each step having a clear, actionable description. 
            ALWAYS prefix visualization steps with [GRAPH].
            """
        else:
            # Create a new analysis plan prompt
            prompt = f"""Create a detailed data analysis plan with exactly {self.plan_depth} steps that addresses this task:
            {task}
            
            Each step should be clearly numbered (Step 1, Step 2, etc.) and include:
            1. A clear description of what to analyze
            2. The appropriate methodology to use
            3. Indicate with [GRAPH] tag if the step should generate a visualization
            
            IMPORTANT: Format as numbered steps, with each step having a clear, actionable description.
            ALWAYS prefix visualization steps with [GRAPH].
            """
        
        # Parse configuration settings from task message if present
        plan_depth = 5  # Default plan depth
        max_graphs = 10  # Default max graphs per step
        
        # Look for configuration section in the task message
        config_section = re.search(r'ANALYSIS_CONFIG:(.*?)(?:\n\n|\Z)', task, re.DOTALL)
        if config_section:
            config_text = config_section.group(1)
            
            # Extract plan depth
            plan_depth_match = re.search(r'Plan depth:\s*(\d+)', config_text)
            if plan_depth_match:
                plan_depth = int(plan_depth_match.group(1))
                
            # Extract max visualizations
            max_viz_match = re.search(r'Max visualizations:\s*(\d+)', config_text)
            if max_viz_match:
                max_graphs = int(max_viz_match.group(1))
        
        # Update prompt with extracted configuration values
        if "[PLACEHOLDER_PLAN_DEPTH]" in prompt:
            prompt = prompt.replace("[PLACEHOLDER_PLAN_DEPTH]", str(plan_depth))
        if "[PLACEHOLDER_MAX_GRAPHS]" in prompt:
            prompt = prompt.replace("[PLACEHOLDER_MAX_GRAPHS]", str(max_graphs))
        
        response_json = await call_llm_async(prompt)
        
        # Store the raw response for fallback in case of parsing issues
        raw_response = response_json
        
        try:
            # First attempt: Try to parse as JSON directly
            try:
                response = json.loads(response_json)
                # Clean up step formatting if needed
                if "plan" in response and isinstance(response["plan"], str):
                    response["plan"] = response["plan"].replace("\\n", "\n")
                print("Successfully parsed analysis plan as JSON")
                return response
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                # If JSON parsing fails, continue to next approach
                pass
            
            # Second attempt: Try to extract JSON from a potentially larger text response
            json_pattern = r'\{[\s\S]*?\}'
            matches = re.findall(json_pattern, response_json)
            if matches:
                for potential_json in matches:
                    try:
                        response = json.loads(potential_json)
                        if "plan" in response:
                            if isinstance(response["plan"], str):
                                response["plan"] = response["plan"].replace("\\n", "\n")
                            print("Successfully extracted and parsed JSON from response")
                            return response
                    except:
                        continue
            
            # Third attempt: Look for plan content directly in the response
            if "step 1:" in response_json.lower() or "step 1." in response_json.lower():
                print("Extracting plan directly from text response")
                plan_text = self._extract_plan_from_text(response_json)
                
                # Try to detect visualization steps
                graph_steps = []
                graph_types = []
                step_pattern = r"(?:Step\s*)?(\d+)[\.:\)]\s*(.*?)(?=(?:Step\s*)?(?:\d+)[\.:\)]|$)"
                for match in re.finditer(step_pattern, plan_text, re.DOTALL):
                    step_num = int(match.group(1))
                    step_text = match.group(2).strip()
                    if any(word in step_text.lower() for word in ["plot", "chart", "graph", "visual", "figure", "histogram", "scatter"]):
                        graph_steps.append(step_num)
                        # Try to extract the visualization type
                        type_patterns = ["histogram", "scatter plot", "bar chart", "line graph", "heatmap", "box plot", "pie chart"]
                        for type_pattern in type_patterns:
                            if type_pattern in step_text.lower():
                                graph_types.append(type_pattern)
                                break
                        else:
                            graph_types.append("plot")  # Generic fallback
                
                # Add [GRAPH] labels to the plan text
                for step_num in graph_steps:
                    plan_text = re.sub(
                        f"(?:Step\s*)?{step_num}[\.:\)]\s*", 
                        f"Step {step_num}: [GRAPH] ", 
                        plan_text
                    )
                
                return {
                    "plan": plan_text,
                    "graph_steps": graph_steps,
                    "graph_count": len(graph_steps),
                    "graph_types": graph_types,
                    "is_continuation": is_continuation,
                    "raw_response": raw_response  # Include the raw response for debugging
                }
        except Exception as e:
            print(f"Error processing analysis plan: {str(e)}")
            # Log stack trace for debugging
            traceback.print_exc()
        
        # Ultimate fallback: If everything fails, return the raw response
        print("All parsing attempts failed. Using fallback for analysis plan.")
        return {
            "plan": raw_response,
            "graph_steps": [],
            "graph_count": 0,
            "graph_types": [],
            "parsing_failed": True,
            "is_continuation": is_continuation,
            "raw_response": raw_response
        }
    
    def _extract_plan_from_text(self, text):
        """Extract plan from text if JSON parsing fails."""
        if "Step 1:" in text or "1." in text:
            return text
        else:
            return "Could not generate structured plan. Proceeding with analysis."
    
    async def _get_next_code(self, state):
        """Generate the next code to execute based on the current state."""
        # Create a summary of available dataframes
        df_summary = ""
        
        # Debug print for diagnosing dataframes issue
        print(f"Available dataframes: {len(self.dataframes)} - {list(self.dataframes.keys())}")
        
        # Check if we have any dataframes
        if not self.dataframes:
            print("WARNING: No dataframes available. Analysis may fail.")
            df_summary = "WARNING: No dataframes are currently loaded. You may need to load data first."
        else:
            for name, df in self.dataframes.items():
                if isinstance(df, pd.DataFrame):
                    # Add detailed dataframe information, including more sample data for better context
                    df_summary += f"\nDataFrame '{name}': {df.shape[0]} rows, {df.shape[1]} columns"
                    df_summary += f"\nColumns: {list(df.columns)}"
                    df_summary += f"\nData types: {df.dtypes.to_dict()}"
                    df_summary += f"\nSample data (first 3 rows):\n{df.head(3).to_dict()}\n"
                    df_summary += f"\nSummary statistics:\n{df.describe().to_dict()}\n"
                    
                    # Check for missing values
                    missing_values = df.isnull().sum().to_dict()
                    df_summary += f"\nMissing values: {missing_values}\n"
        
        # Build execution history
        history = ""
        for step in state["completed_steps"]:
            history += f"Step {step['step']}:\n```python\n{step['code']}\n```\n"
            if step["output"]:
                history += f"Output: {step['output'][:200]}"
                if len(step["output"]) > 200:
                    history += "...(truncated)"
                history += "\n"
            if step["error"]:
                history += f"Error: {step['error']}\n"
        
        # Determine if visualization is needed in this step
        needs_visualization = False
        current_step = state["current_step"] + 1
        if "plan" in state:
            plan_text = state["plan"]
            if isinstance(plan_text, str):
                step_lines = re.findall(r"(?:Step|)\s*" + str(current_step) + r"[\.:]\s*(.*?)(?:\n|$)", plan_text)
                if step_lines:
                    step_text = step_lines[0].lower()
                    viz_keywords = ["plot", "chart", "graph", "visual", "figure", "histogram", "scatter", "bar", "pie"]
                    needs_visualization = any(keyword in step_text for keyword in viz_keywords)
        
        # Create prompt for code generation with enhanced visualization instructions if needed
        viz_instructions = ""
        if needs_visualization:
            viz_instructions = """
            When creating visualizations, follow these guidelines:
            1. Use plt.figure(figsize=(10, 6)) for better default sizing
            2. Apply meaningful titles, labels, and annotations
            3. Include plt.tight_layout() for proper spacing
            4. Ensure axis labels clearly describe the data with units if applicable
            5. Use a legend when multiple data series are shown
            6. Set appropriate axis limits to focus on the important data
            7. Apply a pastel color palette for better aesthetics
            8. Include a brief description as a plt.figtext() explaining key insights
            9. Set figure transparency with fig.patch.set_alpha(0.0)
            10. For complex plots, consider using subplots for better organization
            """
        
        # Create prompt for code generation
        prompt = f"""You are a data science agent that generates Python code.
        Output ONLY valid Python code that can be executed to perform the next step in the analysis plan.
        DO NOT include explanations outside of code comments. Use pandas, numpy, scipy, matplotlib, 
        scikit-learn, statsmodels, or other data analysis libraries as needed.
        
        Generate Python code that is ready to run. The code should:
        1. Be self-contained for the current step
        2. Use appropriate error handling (try/except blocks where necessary)
        3. Print or display meaningful results
        4. Return 'ANALYSIS_COMPLETE' in a print statement if all steps are completed
        
        Task: {state['task']}
        
        Plan: {state['plan']}
        
        Current step: {state["current_step"] + 1}
        
        Previous execution history:
        {history}
        
        Available dataframes:
        {df_summary}
        
        {viz_instructions}
        
        Generate ONLY the Python code for the next step in the analysis. The code must:
        1. Be valid Python that can run directly
        2. Reference existing dataframes by name if they exist
        3. Create meaningful visualizations where appropriate
        4. Include print statements to show progress and important results
        5. Print 'ANALYSIS_COMPLETE' if this is the final step
        """
        
        response_json = await call_llm_async(prompt)
        try:
            code = json.loads(response_json)["code"]
        except:
            # Extract code block if JSON parsing fails
            code = self._extract_code_from_text(response_json)
        
        return code
    
    def _extract_code_from_text(self, text):
        """Extract code block from text if JSON parsing fails."""
        # Look for code block between triple backticks
        code_match = re.search(r'```(?:python)?(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        else:
            # Return the whole text if no code block found
            return text
    
    def _execute_code(self, code):
        """Execute the generated code and capture outputs."""
        # Create string buffer to capture print outputs
        output_buffer = io.StringIO()
        error_message = None
        result = None
        
        # Get the current dataframes into local variables
        local_vars = self.dataframes.copy()
        
        # Add matplotlib with correct non-GUI settings to locals
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        local_vars['matplotlib'] = matplotlib
        local_vars['plt'] = plt
        
        # Also add numpy, pandas, and seaborn with correct settings
        import numpy as np
        import pandas as pd
        import seaborn as sns
        local_vars['np'] = np
        local_vars['pd'] = pd
        local_vars['sns'] = sns
        sns.set(style="whitegrid")
        
        # Add scipy with stats module
        import scipy
        from scipy import stats
        from scipy import signal, optimize, interpolate, linalg, spatial
        local_vars['scipy'] = scipy
        local_vars['stats'] = stats
        local_vars['signal'] = signal
        local_vars['optimize'] = optimize
        local_vars['interpolate'] = interpolate
        local_vars['linalg'] = linalg
        local_vars['spatial'] = spatial
        
        # Add statsmodels for statistical modeling
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            local_vars['sm'] = sm
            local_vars['smf'] = smf
        except ImportError:
            print("Warning: statsmodels not available.")
        
        # Add scikit-learn for machine learning
        try:
            import sklearn
            from sklearn import preprocessing, model_selection, metrics
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.cluster import KMeans
            local_vars['sklearn'] = sklearn
            local_vars['preprocessing'] = preprocessing
            local_vars['model_selection'] = model_selection
            local_vars['metrics'] = metrics
            local_vars['LinearRegression'] = LinearRegression
            local_vars['LogisticRegression'] = LogisticRegression
            local_vars['RandomForestClassifier'] = RandomForestClassifier
            local_vars['RandomForestRegressor'] = RandomForestRegressor
            local_vars['KMeans'] = KMeans
        except ImportError:
            print("Warning: scikit-learn not available.")
        
        # Track figures before execution
        initial_figs = plt.get_fignums()
        print(f"Initial figures before execution: {initial_figs}")
        
        # Track whether this is a graph-generating step
        is_graph_step = self.plan_step in self.graph_steps
        
        try:
            # Replace potentially problematic matplotlib code
            if "plt.show(" in code:
                code = code.replace("plt.show()", "# plt.show() - disabled in thread")
                print("Note: plt.show() calls have been disabled to prevent thread issues")
            
            # Limit the number of figure creations for graph steps to prevent duplicate renderings
            if is_graph_step:
                # Add code to close any existing figures first to prevent accumulation
                code = "plt.close('all')\n" + code
                # Check if we're creating too many graphs and limit them
                if "plt.figure(" in code:
                    figure_count = code.count("plt.figure(")
                    if figure_count > 3:
                        output_buffer.write(f"WARNING: Code trying to create {figure_count} figures. Limiting to prevent duplication.\n")
                        # Only allow the first 3 figure creations
                        code_lines = code.split("\n")
                        figure_counter = 0
                        for i, line in enumerate(code_lines):
                            if "plt.figure(" in line:
                                figure_counter += 1
                                if figure_counter > 3:
                                    code_lines[i] = "# " + line + " # Limited by framework"
                        code = "\n".join(code_lines)
            
            # Adjust indentation of the user's code to ensure it works inside our wrapper
            # Dedent the code first to remove any common leading spaces
            import textwrap
            code = textwrap.dedent(code)
            # Indent each line with exactly 4 spaces to fit inside our try block
            code_lines = code.split('\n')
            indented_code = '\n'.join(['    ' + line for line in code_lines])
            
            # Add error handling for any GUI operations without indentation issues
            safe_code = """
# Ensure we're in non-interactive mode
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()
plt.rcParams['interactive'] = False

try:
{}
except Exception as e:
    if "sip" in str(e).lower() or "qt" in str(e).lower() or "thread" in str(e).lower() or "timer" in str(e).lower() or "killTimer" in str(e):
        print("WARNING: Qt or threading-related error detected and suppressed. This is normal when running in a background thread.")
        print(f"Original error: {{str(e)}}")
    else:
        raise  # Re-raise other errors
""".format(indented_code)
            
            # Make dataframes directly accessible by name in the execution scope
            # This allows the code to use dataframe names directly without self.dataframes references
            for df_name, df_value in self.dataframes.items():
                exec(f"{df_name} = local_vars['{df_name}']")
            
            # If 'df' doesn't exist but other dataframes do, make the first one available as 'df'
            if 'df' not in local_vars and len(local_vars) > 0:
                # Find the first DataFrame
                for var_name, var_value in local_vars.items():
                    if isinstance(var_value, pd.DataFrame):
                        local_vars['df'] = var_value
                        exec(f"df = local_vars['df']")
                        print(f"Note: Using dataframe '{var_name}' as 'df' for compatibility")
                        break
            
            # For debugging, print the code that will be executed
            print("\nExecuting code:\n")
            print(safe_code)
            
            # Execute code and capture standard output
            sys.stdout = output_buffer
            exec(safe_code, globals(), local_vars)
            result = "Success"
            
            # Get any new or modified dataframes
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame):
                    self.dataframes[var_name] = var_value
                    
            # Check for new figures created during execution
            current_figs = plt.get_fignums()
            new_figs = set(current_figs) - set(initial_figs)
            
            # DON'T save figures to temporary files - instead, track them for direct display
            if new_figs:
                
                # If this is a graph step, tag the figures with the current plan step
                if is_graph_step:
                    pass
                
                for fig_num in new_figs:
                    try:
                        # Just get a reference to the figure
                        fig = plt.figure(fig_num)
                        
                        # Set transparent background for better display
                        fig.patch.set_alpha(0.0)
                        for ax in fig.get_axes():
                            ax.patch.set_alpha(0.0)
                        
                        # Tag the figure with the current plan step
                        fig.plan_step = self.plan_step
                        
                        # Store figure for direct SVG rendering
                        if not hasattr(self, 'figures'):
                            self.figures = []
                        self.figures.append(fig)
                        
                        # Just log the figure in output for reference
                    except Exception as e:
                        output_buffer.write(f"Error processing figure {fig_num}: {str(e)}\n")
                
                # Note: Don't close figures here, as we want to use them for displaying
                # The UI will handle the figures directly via the AgentWorker thread
            
        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result = "Error"
        finally:
            # Restore standard output
            sys.stdout = sys.__stdout__
            # We don't close figures here to allow them to be displayed later
        
        return result, output_buffer.getvalue(), error_message
    
    async def _reflect_on_execution(self, state):
        """Reflect on the execution results and suggest course corrections if needed."""
        last_output = state["outputs"][-1] if state["outputs"] else {}
        
        
        # Create prompt for structured reflection to detect error status
        prompt = f"""You are a data analysis expert. 
        Reflect on the code execution result and provide a brief assessment of:
        1. Whether the step was successful
        2. Key insights or findings
        3. Any issues that need to be addressed
        4. Whether to continue with the original plan or adjust
        
        Task: {state['task']}
        
        Code executed:
        ```python
        {last_output.get('code', 'No code executed')}
        ```
        
        Output:
        {last_output.get('output', 'No output')}
        
        Error:
        {last_output.get('error', 'No errors')}
        
        Attempt number: {last_output.get('recovery_attempt', 0) + 1}
        
        Provide your reflection in JSON format with these fields:
        {{
            "reflection": "Your 2-4 sentence reflection here",
            "has_error": true/false,
            "recovery_needed": true/false
        }}
        
        Set "has_error" to true if you detect any error in the output.
        Set "recovery_needed" to true if you think the step should be retried with improved code.
        """
        
        response_json = await call_llm_async(prompt)
        response_json = response_json.replace("```json", "").replace("```", "")
        try:
            # First try to parse the entire response as JSON
            parsed_json = json.loads(response_json)
            
            # Check if we got a properly structured response with all required fields
            if all(k in parsed_json for k in ["reflection", "has_error", "recovery_needed"]):
                return parsed_json
            # If missing fields but has reflection, use it and infer the rest
            elif "reflection" in parsed_json:
                return {
                    "reflection": parsed_json["reflection"],
                    "has_error": last_output.get('error', '') != '',
                    "recovery_needed": last_output.get('error', '') != '' and "fix" in parsed_json.get("reflection", "").lower()
                }
        except json.JSONDecodeError:
            # Not valid JSON, try to extract JSON from text response
            json_pattern = r'```json\s*(.*?)\s*```|{[\s\S]*"reflection"[\s\S]*}|{\s*"reflection":[\s\S]*}'
            json_match = re.search(json_pattern, response_json, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                # Clean up the extracted JSON
                json_str = re.sub(r'```json|```', '', json_str).strip()
                
                try:
                    parsed_json = json.loads(json_str)
                    if "reflection" in parsed_json:
                        # Extract required fields with defaults
                        return {
                            "reflection": parsed_json.get("reflection", ""),
                            "has_error": parsed_json.get("has_error", last_output.get('error', '') != ''),
                            "recovery_needed": parsed_json.get("recovery_needed", False)
                        }
                except:
                    pass
        
        # Fall back to using the raw text as reflection
        # Remove any markdown code block formatting
        clean_text = re.sub(r'```(?:json)?(.*?)```', r'\1', response_json, flags=re.DOTALL)
        clean_text = re.sub(r'^{|}$', '', clean_text.strip())
        
        # If it looks like valid content, use it
        return {
            "reflection": clean_text,
            "has_error": last_output.get('error', '') != '',
            "recovery_needed": last_output.get('error', '') != '' and "fix" in clean_text.lower()
        }
    
    async def _get_final_summary(self, state):
        """Generate a final summary of the analysis."""
        # Build execution history
        history = ""
        for step in state["completed_steps"]:
            history += f"Step {step['step']}:\n```python\n{step['code']}\n```\n"
            if step["output"]:
                output_preview = step["output"][:300] if isinstance(step["output"], str) else str(step["output"])[:300]
                if isinstance(step["output"], str) and len(step["output"]) > 300:
                    output_preview += "...(truncated)"
                history += f"Output: {output_preview}\n"
            if step["reflection"]:
                # Handle both string and dictionary reflections
                if isinstance(step["reflection"], dict) and "reflection" in step["reflection"]:
                    reflection_text = step["reflection"]["reflection"]
                    reflection_preview = reflection_text[:300] if isinstance(reflection_text, str) else str(reflection_text)[:300]
                    history += f"Reflection: {reflection_preview}\n\n"
                elif isinstance(step["reflection"], str):
                    reflection_preview = step["reflection"][:300]
                    history += f"Reflection: {reflection_preview}\n\n"
                else:
                    # For any other type, convert to string
                    history += f"Reflection: {str(step['reflection'])[:300]}\n\n"
        
        # Count completed graph steps
        graph_steps = state.get("graph_steps", [])
        graph_types = state.get("graph_types", [])
        generated_graphs = state.get("generated_graphs", [])
        
        graph_summary = ""
        if graph_steps:
            graph_summary = f"\nVisualizations created: {len(generated_graphs)} of {len(graph_steps)} planned\n"
            for i, step in enumerate(graph_steps):
                graph_type = graph_types[i] if i < len(graph_types) else "visualization"
                status = "✅ Created" if step in generated_graphs else "❌ Not created"
                graph_summary += f"- Step {step}: {graph_type} - {status}\n"
        
        # Create prompt for summary with explicit markdown section instructions
        prompt = f"""You are a data science expert. Summarize the completed analysis.
        
        Original task: {state['task']}
        Plan that was followed:
        {state['plan']}
        {graph_summary}
        Execution steps:
        {history}
        
        Provide a comprehensive summary as a JSON object. 
        **IMPORTANT:** Structure the JSON with the following exact keys:
        
        {{ 
          "executive_summary": "(string: 1-2 sentence high-level overview)",
          "key_findings": [
            "(string: bullet point 1)",
            "(string: bullet point 2)",
            "... (3-5 key discoveries)"
          ],
          "methodology": "(string: Brief description of analysis approach)",
          "detailed_results": "(string: More in-depth description of findings, can use markdown)",
          "visualizations_summary": "(string: Brief discussion of visualizations created)",
          "limitations": "(string: Honest assessment of constraints or caveats)",
          "data_analysis_steps": [
            "(string: specific next DATA ANALYSIS step the AI could perform, e.g., 'Run correlation analysis')",
            "... (Optional, 1-3 steps. If none, provide an empty list [])"
          ],
          "study_related_steps": [
            "(string: specific next STUDY-RELATED step for the USER, e.g., 'Collect more data', 'Consult domain expert')",
            "... (Optional, 1-3 steps. If none, provide an empty list [])"
          ]
        }}
        
        Ensure the output is a single, valid JSON object only, with no surrounding text or markdown formatting.
        Distinguish clearly between data analysis steps the AI can perform next and broader study-related steps for the user.
        """
        
        response_text = await call_llm_async(prompt)
        
        # Attempt to parse the JSON response more robustly
        try:
            # Clean potential markdown code blocks and whitespace
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"): 
                cleaned_text = re.sub(r'^```json\s*|\s*```$', '', cleaned_text).strip()
            elif cleaned_text.startswith("```"): # Handle generic code blocks too
                 cleaned_text = re.sub(r'^```\s*|\s*```$', '', cleaned_text).strip()
                 
            # Handle potential nesting issue observed in logs
            # Try parsing directly first
            try:
                summary_json = json.loads(cleaned_text)
                # Check if the actual summary is nested inside
                if isinstance(summary_json.get("executive_summary"), str) and summary_json["executive_summary"].strip().startswith("{"):
                     nested_json_str = summary_json["executive_summary"].strip()
                     try:
                         summary_json = json.loads(nested_json_str)
                         print("Successfully parsed nested JSON summary.")
                     except json.JSONDecodeError as nested_e:
                         print(f"Warning: Detected nested JSON structure but failed to parse: {nested_e}")
                         # Proceed with outer structure if parsing nested fails
                
                # Basic validation: check if essential keys exist
                # Check for either next_steps or the new keys
                has_required_keys = ("executive_summary" in summary_json and 
                                   "key_findings" in summary_json and
                                   ("next_steps" in summary_json or 
                                    ("data_analysis_steps" in summary_json and "study_related_steps" in summary_json)))
                                    
                if has_required_keys:
                    print("Successfully parsed summary JSON.")
                    # Ensure compatibility: if old 'next_steps' exists, map it
                    if "next_steps" in summary_json and "data_analysis_steps" not in summary_json:
                        summary_json["data_analysis_steps"] = summary_json.pop("next_steps")
                        summary_json["study_related_steps"] = [] # Assume old steps were analysis steps
                    elif "data_analysis_steps" not in summary_json:
                         summary_json["data_analysis_steps"] = [] # Add empty list if missing
                    if "study_related_steps" not in summary_json:
                         summary_json["study_related_steps"] = [] # Add empty list if missing
                         
                    return summary_json
                else:
                    print("Warning: Parsed JSON missing expected keys.")
                    raise json.JSONDecodeError("Missing expected keys", cleaned_text, 0)
                    
            except json.JSONDecodeError as e:
                 # If direct parsing fails, re-raise to be caught by the outer try-except
                 raise e
                 
        except json.JSONDecodeError as e:
             print(f"Error parsing summary JSON: {e}")
             print(f"Received text: {response_text}")
             # Fallback: return a basic error structure if JSON parsing fails
             # Ensure all keys are present in the fallback
             return {
                 "executive_summary": "Error: Could not generate or parse summary.",
                 "key_findings": [f"Failed to parse LLM response: {e}"],
                 "methodology": "N/A",
                 "detailed_results": f"Raw response:\n{response_text}",
                 "visualizations_summary": "N/A",
                 "limitations": "N/A",
                 "data_analysis_steps": [], 
                 "study_related_steps": [] 
             }
    
    async def _generate_graph(self, state, step_number, data_description=None, graph_type=None):
        """Generate a specialized visualization."""
        # Create a summary of available dataframes
        df_summary = ""
        for name, df in self.dataframes.items():
            if isinstance(df, pd.DataFrame):
                df_summary += f"\nDataFrame '{name}': {df.shape[0]} rows, {df.shape[1]} columns"
                # Convert column names to strings before joining
                column_names = [str(col) for col in df.columns[:10]]
                df_summary += f"\n- Columns: {', '.join(column_names)}"
                if len(df.columns) > 10:
                    df_summary += f" and {len(df.columns) - 10} more"
                # Include sample data
                df_summary += f"\n- Sample data: {df.head(2).to_dict()}\n"
        
        # Get the context from previous steps
        context = ""
        for step in state["completed_steps"]:
            if "output" in step and step["output"]:
                context += f"Step {step['step']} Output: {step['output'][:300]}\n"
            if "reflection" in step and step["reflection"]:
                context += f"Reflection: {step['reflection'][:300]}\n"
        
        # Create a scientific visualization prompt
        prompt = f"""You are a data visualization expert specializing in scientific and statistical graphics.
        Generate Python code to create a publication-quality visualization for step {step_number} of the analysis.
        
        Analysis task: {state['task']}
        
        Current step description: {data_description or ""}
        
        {'Suggested visualization type: ' + graph_type if graph_type else ''}
        
        Available dataframes:
        {df_summary}
        
        Context from previous steps:
        {context}
        
        Your visualization should follow these scientific visualization best practices:
        1. MEANINGFUL TITLE: Use a descriptive title that communicates the main finding or question
        2. CLEAR LABELING: All axes must be clearly labeled with units when applicable
        3. APPROPRIATE VISUALIZATION: Choose the graph type that best reveals the patterns in the data
        4. COLOR USAGE: Use color strategically to highlight important patterns, not just for decoration
        5. STATISTICAL ANNOTATIONS: Include error bars, trend lines, or statistical tests when appropriate
        6. PROPER SCALING: Use appropriate transformations (log, etc.) if data is skewed
        7. DATA-INK RATIO: Maximize the data-to-ink ratio by removing unnecessary elements
        8. ACCESSIBILITY: Use colorblind-friendly palettes and adequate font sizes
        9. CLARITY: Avoid cluttering the visualization with too many elements
        10. CONTEXT: Provide a caption or annotation explaining key insights
        
        Technical requirements for your code:
        1. Create the visualization using matplotlib and seaborn
        2. Use a figure size of at least (10, 6) for visibility
        3. Set appropriate DPI (at least 100) for clarity
        4. Apply the pastel color palette defined in the codebase
        5. Set transparent backgrounds for both figure and axes
        6. Include plt.tight_layout() for proper spacing
        7. Add plt.figtext() with a brief explanation of key findings
        8. For complex relationships, consider using facet plots or small multiples
        9. The code should be self-contained and create exactly one figure
        10. Use meaningful variable names and include comments
        
        Return ONLY the Python code to generate this visualization, nothing else.
        """
        
        response_json = await call_llm_async(prompt)
        try:
            visualization_code = json.loads(response_json)["code"]
        except:
            # Extract code block if JSON parsing fails
            visualization_code = self._extract_code_from_text(response_json)
        
        return visualization_code

    async def _get_recovery_code(self, state, code, error, output, reflection):
        """Generate code to fix errors from previous execution attempt."""
        # Create a detailed prompt for error recovery
        prompt = f"""You are a data science debugging expert.
        
        The previous code execution attempt encountered an error. Your task is to fix the code to make it work.
        
        Original task: {state['task']}
        
        Code that caused the error:
        ```python
        {code}
        ```
        
        Error message:
        {error or "No specific error message, but the code didn't work as expected."}
        
        Output from execution:
        {output or "No output was generated."}
        
        Reflection on the error:
        {reflection}
        
        Available dataframes:
        """
        
        # Add dataframe information
        for name, df in self.dataframes.items():
            if isinstance(df, pd.DataFrame):
                prompt += f"\nDataFrame '{name}': {df.shape[0]} rows, {df.shape[1]} columns"
                # Convert column names to strings to handle tuple column names
                column_list = [str(col) for col in df.columns]
                prompt += f"\nColumns: {column_list}"
                
        # Add instructions for the fix
        prompt += """
        
        Please generate FIXED code that:
        1. Addresses the specific error or issue identified
        2. Uses better error handling if needed
        3. Is complete and ready to run
        4. Accomplishes the original goal of this step
        
        Only output the fixed Python code, nothing else. Do not include explanations outside of code comments.
        """
        
        # Get the fix from the LLM
        response_json = await call_llm_async(prompt)
        try:
            code = json.loads(response_json)["code"]
        except:
            # Extract code block if JSON parsing fails
            code = self._extract_code_from_text(response_json)
            
        return code


class AgentWorker(QObject):
    """Worker thread for running the agent asynchronously."""
    finished = pyqtSignal()
    progress = pyqtSignal(str, str)  # Signal for when progress is made: (message, msg_type)
    figure_created = pyqtSignal(object)  # Signal for when a matplotlib figure is created
    
    # Constants
    MAX_FIGURES_PER_SESSION = 100  # Increased limit for maximum number of figures to capture in a session
    
    def __init__(self, agent, task):
        super().__init__()
        self.agent = agent
        self.task = task
        self.figure_count = 0  # Counter to track number of figures
        
        # Patch matplotlib to capture figures
        self._patch_matplotlib()
        
    def _patch_matplotlib(self):
        """Patch matplotlib to capture figures safely."""
        # Store original functions
        original_figure = plt.figure
        original_subplots = plt.subplots
        original_show = plt.show  # Store the original show function
        
        # Flag to track if warning has been emitted
        max_figures_warning_emitted = [False]
        
        # Patch plt.figure
        def patched_figure(*args, **kwargs):
            # Use a lock to prevent reentrant calls causing issues
            try:
                # Remove limit check since we want to show all figures
                # Just limit figure size to reasonable dimensions to prevent memory issues
                if 'figsize' in kwargs:
                    # Ensure figure size is not too large
                    max_width, max_height = 12, 10  # Maximum reasonable figure size
                    current_width, current_height = kwargs['figsize']
                    if current_width > max_width or current_height > max_height:
                        # Scale down to reasonable size
                        kwargs['figsize'] = (min(current_width, max_width), min(current_height, max_height))
                    
                # Set DPI limit for reasonable memory usage
                kwargs['dpi'] = min(kwargs.get('dpi', 100), 150)
                
                # Create the figure but don't render it yet
                fig = original_figure(*args, **kwargs)
                
                # Apply pastel colors by default to all plots
                plt.style.use('default')
                
                # Increment the figure count
                self.figure_count += 1
                
                # Save plan step info directly on the figure for better integration
                current_plan_step = getattr(self.agent, 'plan_step', 1)
                fig.plan_step = current_plan_step
                
                try:
                    # Signal that a figure was created with the plan step information
                    self.figure_created.emit((fig, f"Plan Step {current_plan_step}"))
                except Exception as e:
                    print(f"Error emitting figure signal: {str(e)}")
                
                return fig
            except Exception as e:
                # Log error but continue with normal figure creation
                print(f"Error in patched_figure: {str(e)}")
                return original_figure(*args, **kwargs)
        
        # Patch plt.subplots
        def patched_subplots(*args, **kwargs):
            """Patch for plt.subplots to capture figure creation."""
            try:
                # Call original function
                fig, axs = original_subplots(*args, **kwargs)
                
                # Apply pastel colors to the figure
                plt.style.use('default')
                
                # Increment figure count
                self.figure_count += 1
                
                # IMMEDIATE DISPLAY: Get the plan step directly (prefer direct plan_step over mapping)
                current_plan_step = getattr(self.agent, 'plan_step', 1)
                
                if hasattr(self.agent, 'step_mapping') and hasattr(self.agent, 'step_count'):
                    current_exec_step = self.agent.step_count
                    mapped_plan_step = self.agent.step_mapping.get(current_exec_step, current_plan_step)
                    # If there's inconsistency between direct and mapped plan step, prefer the direct one
                    if mapped_plan_step != current_plan_step:
                        pass
                
                # Use the plan step for title - this ensures logical grouping
                step_title = f"Plan Step {current_plan_step}"
                
                # Add attribute to figure for internal tracking
                fig.plan_step = current_plan_step
                
                # Signal that a figure was created - do this synchronously
                try:
                    self.figure_created.emit((fig, step_title))
                except Exception as e:
                    print(f"Error emitting figure signal: {str(e)}")
                
                return fig, axs
            except Exception as e:
                print(f"Error in patched_subplots: {str(e)}")
                return original_subplots(*args, **kwargs)  # Fall back to original
        
        # Patch plt.show
        def patched_show(*args, **kwargs):
            """Safer patched version of plt.show()
            
            This will maintain backward compatibility for scripts that call plt.show()
            but will not actually display the window, preventing GUI-related errors.
            """
            # Just do nothing instead of showing a window
            # This works better than closing figures which can cause other issues
            return None  
        
        # Apply the patches
        plt.figure = patched_figure
        plt.subplots = patched_subplots
        plt.show = patched_show
    
    async def run_agent(self):
        """Run the agent and emit progress signals."""
        try:
            # Process any figures from the agent at regular intervals - but avoid QTimer
            # Instead, check for figures directly at key points using a simple flag
            figure_processing_due = True
            
            def process_pending_figures():
                if hasattr(self.agent, 'figures') and self.agent.figures:
                    # Make a copy of the figures to avoid modification during iteration
                    figures_to_process = self.agent.figures.copy()
                    self.agent.figures = []  # Clear immediately to prevent concurrent modification
                    
                    for fig in figures_to_process:
                        # Process each figure individually
                        try:
                            self.figure_created.emit((fig, f"Plan Step {getattr(self.agent, 'plan_step', 1)}"))
                        except Exception as e:
                            print(f"Error processing figure: {str(e)}")
            
            # Create a wrapper function for the callback to handle message types
            def progress_callback(message, msg_type="standard"):
                self.progress.emit(message, msg_type)
                # Process figures at each callback to avoid timer usage
                if figure_processing_due:
                    process_pending_figures()
            
            # Run the agent with progress signal
            await self.agent.execute(self.task, progress_callback)
            
            # Final check for any remaining figures
            process_pending_figures()
            
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}\n{traceback.format_exc()}", "standard")
        finally:
            self.finished.emit()
    
    def run(self):
        """Run the agent in the current thread."""
        asyncio.run(self.run_agent())


class AgentInterface(QWidget):
    """Main agent interface with conversational and data analysis modes."""
    
    file_upload_requested = pyqtSignal(str)  # Signal for file upload requests
    
    # Static figure storage to prevent garbage collection
    processed_figures = []  # Store all successfully processed figures
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Check matplotlib backend
        import matplotlib
        print(f"🔧 Current Matplotlib backend: {matplotlib.get_backend()}")
        # Ensure we're using Agg backend for better thread safety
        if matplotlib.get_backend() != 'Agg':
            print("🔧 Setting Matplotlib backend to Agg for better thread safety")
            matplotlib.use('Agg')
        
        # Figure processing flags
        self.figure_processing = False
        self.pending_figures = []
        
        # Set up the main window reference
        self.main_window = parent
        
        # Connect to theme change signal if parent has one
        if parent and hasattr(parent, 'theme_changed'):
            parent.theme_changed.connect(self.apply_theme)
        
        # Initialize agents
        self.conversational_agent = None  # Will be initialized when needed
        self.data_analysis_agent = DataAnalysisAgent()
        # Set parent reference so agent can directly add figures to UI
        self.data_analysis_agent.parent = self
        print("Set parent reference on data_analysis_agent")
        
        # Track loaded data files
        self.loaded_files = []
        
        # Initialize UI
        self.init_ui()
        
        # Current mode (conversational or data analysis)
        self.current_mode = "conversational"
        
        # Apply theme initially
        self.apply_theme()
        
    def init_ui(self):
        """Initialize the user interface with grid layouts."""
        # Set up the main grid layout
        main_grid = QGridLayout(self)
        main_grid.setContentsMargins(0, 0, 0, 0)
        main_grid.setSpacing(0)
        
        # Create mode selection at the top
        mode_panel = QWidget()
        mode_panel.setObjectName("modePanel")
        mode_panel.setMaximumHeight(50)
        
        mode_layout = QHBoxLayout(mode_panel)
        mode_layout.setContentsMargins(10, 5, 10, 5)
        
        # Create mode selection radio buttons
        self.mode_group = QButtonGroup(self)
        
        self.conversational_mode_rb = QRadioButton("Conversational Mode")
        self.conversational_mode_rb.setObjectName("modeRadioButton")
        self.conversational_mode_rb.setChecked(True)
        self.conversational_mode_rb.toggled.connect(self.on_mode_changed)
        
        self.data_analysis_mode_rb = QRadioButton("Data Analysis Mode")
        self.data_analysis_mode_rb.setObjectName("modeRadioButton")
        self.data_analysis_mode_rb.toggled.connect(self.on_mode_changed)
        
        self.mode_group.addButton(self.conversational_mode_rb)
        self.mode_group.addButton(self.data_analysis_mode_rb)
        
        mode_layout.addWidget(self.conversational_mode_rb)
        mode_layout.addWidget(self.data_analysis_mode_rb)
        mode_layout.addStretch()
        
        # Add New Chat button
        new_chat_button = QPushButton("New Chat")
        new_chat_button.setObjectName("newChatButton")
        new_chat_button.setToolTip("Start a new chat session")
        new_chat_button.setStyleSheet("""
            QPushButton#newChatButton {
                border-radius: 4px;
                padding: 5px 10px;
                font-weight: bold;
            }
        """)
        new_chat_button.clicked.connect(self.new_chat)
        mode_layout.addWidget(new_chat_button)
        
        # Add mode panel to main grid
        main_grid.addWidget(mode_panel, 0, 0, 1, 2)
        
        # Create content grid for the main area
        content_widget = QWidget()
        content_grid = QGridLayout(content_widget)
        content_grid.setContentsMargins(0, 0, 0, 0)
        content_grid.setSpacing(10)
        
        # Create the timeline widget
        self.timeline = self.create_timeline_widget()
        content_grid.addWidget(self.timeline, 0, 0)
        
        # Create the chat widget
        self.chat_widget = self.create_chat_widget()
        content_grid.addWidget(self.chat_widget, 1, 0)
        
        # Set row stretch factors for timeline and chat
        content_grid.setRowStretch(0, 2)  # Timeline gets 2 parts
        content_grid.setRowStretch(1, 3)  # Chat gets 3 parts
        
        # Create right panel with tabs
        right_panel = self.create_right_panel()
        content_grid.addWidget(right_panel, 0, 1, 2, 1)  # Spans both rows
        
        # Set column stretch factors for left and right sides
        content_grid.setColumnStretch(0, 3)  # Left side gets 3 parts
        content_grid.setColumnStretch(1, 2)  # Right side gets 2 parts
        
        # Add content widget to main grid
        main_grid.addWidget(content_widget, 1, 0, 1, 2)
        main_grid.setRowStretch(1, 1)
        
        # Dictionary to track active tasks
        self.active_tasks = {}
        
        # Dictionary to track step tasks
        self.step_tasks = {}
    
    def create_timeline_widget(self):
        """Create and configure the timeline widget."""
        timeline = TimelineWidget(self)
        timeline.setObjectName("timelineWidget")
        timeline.setMinimumHeight(200)
        return timeline
    
    def create_chat_widget(self):
        """Create and configure the chat widget."""
        chat = RoundedChatWidget(self)
        chat.setObjectName("chatWidget")
        chat.messageSent.connect(self.handle_user_message)
        chat.fileUploadRequested.connect(self.handle_file_upload)
        chat.suggestionClicked.connect(self.handle_suggestion_click)
        return chat
    
    def create_right_panel(self):
        """Create the right panel with tabs."""
        # Create tab widget for the right panel
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("rightTabs")
        self.right_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.setVisible(True)  # Explicitly set visibility
        print("Created right_tabs widget")
        
        # Create outputs widget as the first tab
        self.step_outputs = StepOutputsWidget(self)
        self.right_tabs.addTab(self.step_outputs, "Outputs")
        print("Added Outputs tab at index 0")
        
        # Create graph gallery widget as the second tab
        self.graph_gallery = GraphGalleryWidget(self)
        self.right_tabs.addTab(self.graph_gallery, "Graphs")
        print("Added Graphs tab at index 1")
        
        # Make sure tabs are visible
        for i in range(self.right_tabs.count()):
            self.right_tabs.setTabVisible(i, True)
            print(f"Tab {i} visibility set to True")
        
        # Create graph widget for data analysis
        self.graph_widget = GraphWidget(self)
        
        # Connect tab change signal for debugging
        self.right_tabs.currentChanged.connect(self._on_tab_changed)
        
        # Add reference count printing
        print(f"Graph gallery created: {self.graph_gallery}")
        print(f"Graph widget created: {self.graph_widget}")
        print(f"Right tabs count: {self.right_tabs.count()}")
        
        return self.right_tabs
        
    def _on_tab_changed(self, index):
        """Debug method to track tab changes."""
        tab_names = ["Outputs", "Graphs"]
        if 0 <= index < len(tab_names):
            print(f"Tab changed to: {tab_names[index]} (index {index})")
        else:
            print(f"Tab changed to unknown index: {index}")
    
    def handle_suggestion_click(self, suggestion):
        """Handle a click on a suggestion chip."""
        # Process the suggestion like a user message
        self.handle_user_message(suggestion)
    
    def clear_current_analysis(self):
        """Clear current analysis state to prepare for a new run."""
        # Clear any displayed graphs
        if hasattr(self, 'graph_widget'):
            self.graph_widget.clear()
        
        # Clear graph gallery if it exists
        if hasattr(self, 'graph_gallery') and self.graph_gallery:
            # Clear the gallery by creating a new instance
            self.graph_gallery = GraphGalleryWidget(self)
            if hasattr(self, 'right_tabs') and self.right_tabs:
                # Replace the tab at index 1 (Graphs tab)
                if self.right_tabs.count() > 1:
                    old_widget = self.right_tabs.widget(1)
                    if old_widget:
                        old_widget.deleteLater()
                    self.right_tabs.removeTab(1)
                    self.right_tabs.insertTab(1, self.graph_gallery, "Graphs")
        
        # Clear step outputs
        if hasattr(self, 'step_outputs') and self.step_outputs:
            self.step_outputs = StepOutputsWidget(self)
            if hasattr(self, 'right_tabs') and self.right_tabs:
                # Replace the tab at index 0 (Outputs tab)
                if self.right_tabs.count() > 0:
                    old_widget = self.right_tabs.widget(0)
                    if old_widget:
                        old_widget.deleteLater()
                    self.right_tabs.removeTab(0)
                    self.right_tabs.insertTab(0, self.step_outputs, "Outputs")
        
        # Reset timeline or add a separator
        if hasattr(self, 'timeline') and self.timeline:
            # We don't clear the timeline completely, but we could add a separator
            pass
        
        # Reset the agent's state
        if hasattr(self, 'data_analysis_agent'):
            self.data_analysis_agent.step_count = 0
            self.data_analysis_agent.plan_step = 0
            self.data_analysis_agent.step_mapping = {}
            self.data_analysis_agent.plan_steps = []
            self.data_analysis_agent.current_plan_step_index = -1
            self.data_analysis_agent.graph_steps = []
            self.data_analysis_agent.reflection_mapping = {}
            
            # Keep the dataframes for continuity
            # This allows analyzing the same data with different settings
        
        # Force UI update to ensure changes take effect
        if hasattr(self, 'right_tabs'):
            self.right_tabs.setCurrentIndex(0)  # Switch to Outputs tab
            self.right_tabs.repaint()
    
    def on_mode_changed(self):
        """Handle mode change between conversational and data analysis."""
        # Check which mode is selected
        if self.conversational_mode_rb.isChecked():
            # Conversational mode suggestions
            self.set_suggestions([
                "Can you help me with a coding problem?",
                "Explain how this algorithm works",
                "Write a Python function to sort a list",
                "What's the difference between list and tuple?"
            ])
        else:
            # Data analysis mode suggestions
            self.set_suggestions([
                "Analyze this dataset",
                "Create visualizations of key variables",
                "Find correlations between features",
                "Run a regression analysis" 
            ])
    
    def handle_user_message(self, message: str):
        """Process a user message."""
        # Create a task for this message
        task_name = f"Query: {message[:30]}{'...' if len(message) > 30 else ''}"
        task = self.add_task(task_name, message)
        
        # Add the message to the chat display
        self.chat_widget.add_user_message(message)
        
        # If we're in continuation mode, use the continuation process
        if hasattr(self, 'is_continuing_analysis') and self.is_continuing_analysis:
            self.is_continuing_analysis = False  # Reset flag
            
            # Always enable continuation by default
            self.process_continuation(message, task, True)
        else:
            # Regular message handling
            if self.conversational_mode_rb.isChecked():
                self.handle_conversational_message(message, task)
            else:
                # Check if we have previous analysis results to continue from
                has_previous_analysis = hasattr(self, 'latest_summary_content') and self.latest_summary_content
                
                if has_previous_analysis:
                    # Use continuation as the default behavior
                    self.process_continuation(message, task, True)
                else:
                    # First-time analysis or no previous results
                    self.process_data_analysis(message, task)
    
    def handle_conversational_message(self, message: str, task: TimelineTask):
        """Process conversational messages with an LLM."""
        self.chat_widget.set_typing_indicator(True)
        
        # TODO: Implement conversational agent
        # For now, just echo back a simple response
        response = f"You said: {message}\n\nI'm currently in development for conversational mode. Please switch to data analysis mode to analyze your data."
        
        # Simulate typing delay
        QTimer.singleShot(1000, lambda: self.chat_widget.add_agent_message(response))
        QTimer.singleShot(1000, lambda: self.chat_widget.set_typing_indicator(False))
        QTimer.singleShot(1000, lambda: self.update_task_status(task.name, TaskStatus.COMPLETED))
    
    def process_data_analysis(self, message: str, task: TimelineTask):
        """Process data analysis requests."""
        # Clear current analysis state
        if hasattr(self, 'clear_current_analysis'):
            self.clear_current_analysis()
        
        self.chat_widget.set_typing_indicator(True)
        
        # Switch to the outputs tab
        self.right_tabs.setCurrentIndex(0)
        
        # Configure analysis settings from spinboxes
        plan_depth = self.chat_widget.plan_depth_spinbox.value()
        max_graphs = self.chat_widget.max_graphs_spinbox.value()
        
        # Update UI to reflect these settings
        self.chat_widget.update_status(f"Analyzing with plan depth {plan_depth} and max {max_graphs} graphs per step", is_thinking=True)
        
        # Add settings to the message in a structured format that's easier for the LLM to parse
        enhanced_message = f"""{message}

ANALYSIS_CONFIG:
- Plan depth: {plan_depth} (number of analysis steps to perform)
- Max visualizations: {max_graphs} (maximum visualizations per step)
- Create clear, visually appealing graphs with informative titles and labels
- Focus on the most important insights in the data
"""
        
        print(f"Starting data analysis with message: {enhanced_message[:100]}...")
        print(f"Using plan depth: {plan_depth}, max graphs: {max_graphs}")
        
        # Create worker thread
        self.worker_thread = QThread()
        self.worker = AgentWorker(self.data_analysis_agent, enhanced_message)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals with protection against thread crashes
        self.worker.progress.connect(self.update_analysis_output)
        self.worker.figure_created.connect(self.handle_new_figure)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.started.connect(self.worker.run)
        
        # Add proper thread cleanup to prevent crashes
        self.worker_thread.finished.connect(self.worker.deleteLater)  # Properly clean up worker
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # Properly clean up thread
        
        # Add timeout detection to handle hanging threads
        def check_for_timeout():
            # Update status to completed or show warning if needed
            if task.name in self.active_tasks and self.active_tasks[task.name].status == TaskStatus.RUNNING:
                self.update_task_status(task.name, TaskStatus.COMPLETED)
                
        # Connect with a small delay
        self.worker_thread.finished.connect(lambda: QTimer.singleShot(100, check_for_timeout))
        
        # Start the worker thread
        self.worker_thread.start()
        
        # Update the task status
        self.update_task_status(task.name, TaskStatus.RUNNING)
        
        # Add initial thinking status
        self.chat_widget.update_status("Creating analysis plan...", is_thinking=True)
    
    def handle_file_upload(self, file_path: str):
        """Handle file upload."""
        # Check if file exists
        if not os.path.exists(file_path):
            self.chat_widget.add_agent_message(f"Error: File {file_path} not found.")
            return
            
        # Create task for file loading
        file_name = os.path.basename(file_path)
        task = self.add_task(f"Loading: {file_name}", "Uploading and parsing file")
        self.update_task_status(task.name, TaskStatus.RUNNING)
        
        try:
            # Determine file type and load
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                file_type = 'CSV'
            elif file_path.lower().endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
                file_type = 'TSV'
            elif file_path.lower().endswith('.xlsx'):
                df = pd.read_excel(file_path)
                file_type = 'Excel'
            else:
                self.update_task_status(task.name, TaskStatus.FAILED)
                self.chat_widget.add_agent_message(f"Unsupported file format: {file_path}")
                return
                
            # Process file name to get variable name
            df_name = os.path.splitext(file_name)[0].replace(" ", "_")
            
            # Store the dataframe
            self.data_analysis_agent.dataframes[df_name] = df
            
            # Add to list of loaded files
            self.loaded_files.append((file_name, df_name, file_type))
            
            # Complete the task
            self.update_task_status(task.name, TaskStatus.COMPLETED)
            
        except Exception as e:
            self.update_task_status(task.name, TaskStatus.FAILED)
            self.chat_widget.add_agent_message(f"Error loading file: {str(e)}")
    
    @pyqtSlot(str, str)
    def update_analysis_output(self, text: str, msg_type: str = None):
        """Update the analysis output in the chat."""
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QLabel
        
        self.chat_widget.set_typing_indicator(False)
        
        # Print debugging info for each message
        print(f"Received message of type '{msg_type}', length: {len(text)}")
        if len(text) > 100:
            print(f"Message preview: {text[:100]}...")
        
        # Default to standard message if no type specified
        if msg_type is None:
            # Try to detect message type from content
            if text.startswith("🤔 Reflection:") or text.startswith("Reflection:"):
                msg_type = self.data_analysis_agent.MSG_REFLECTION
            elif text.startswith("```python"):
                msg_type = self.data_analysis_agent.MSG_CODE
            elif text.startswith("Output:"):
                msg_type = self.data_analysis_agent.MSG_OUTPUT
            elif text.startswith("📊 Executing Step") or text.startswith("Executing Step"):
                msg_type = self.data_analysis_agent.MSG_EXECUTING
            elif text.startswith("🔍 Moving to Plan Step") or text.startswith("Moving to Plan Step"):
                msg_type = self.data_analysis_agent.MSG_THINKING
            elif text.startswith("🔍 Analysis Plan:") or text.startswith("Analysis Plan:"):
                msg_type = "plan"
            elif text.startswith("📑 Final Summary:") or text.startswith("Final Summary:"):
                msg_type = self.data_analysis_agent.MSG_SUMMARY
            else:
                msg_type = "standard"  # Default type
        
        # Enhanced plan detection - check for multiple patterns that indicate this is a plan message
        is_plan = False
        if msg_type == "plan":
            is_plan = True
        elif "Analysis Plan:" in text:
            is_plan = True
        elif "analysis plan" in text.lower():
            is_plan = True
        elif "plan:" in text.lower() and ("step 1" in text.lower() or "step 1:" in text.lower()):
            is_plan = True
        elif text.strip().startswith("{") and "plan" in text.lower() and "steps" in text.lower():
            is_plan = True
        elif re.search(r'step\s*1[\.:)]', text.lower()) and re.search(r'step\s*2[\.:)]', text.lower()):
            is_plan = True
            
        if is_plan:
            msg_type = "plan"
            print("Message identified as plan using enhanced detection")
        
        # Handle message based on type
        if msg_type == self.data_analysis_agent.MSG_THINKING:
            # Show thinking messages in the status indicator above input
            # Extract just the thinking content
            thinking_text = text
            if ":" in text:
                thinking_text = text.split(":", 1)[1].strip()
            
            # Update the status with this thinking content
            self.chat_widget.update_status(thinking_text, is_thinking=True)
            
            # Don't display control messages in chat
            return
        
        elif msg_type == self.data_analysis_agent.MSG_EXECUTING:
            # Show execution messages as status indicators
            step_info = re.search(r'Executing Step (\d+) \(Plan Step (\d+)\)', text)
            if step_info:
                exec_step, plan_step = step_info.groups()
                
                # Update status indicator
                self.chat_widget.update_status(f"Executing Plan Step {plan_step}", is_thinking=True)
                
                # Switch to the outputs tab to show processing
                QTimer.singleShot(0, lambda: self.right_tabs.setCurrentIndex(0))
            else:
                # Fallback if no step info found
                self.chat_widget.update_status(text, is_thinking=True)
            
            # Don't display execution messages in chat
            return
        
        elif msg_type == self.data_analysis_agent.MSG_CODE or msg_type == self.data_analysis_agent.MSG_OUTPUT:
            # Skip code and output in the chat - they're already in the tabs
            # Just update corresponding step outputs widget
            current_step = self.data_analysis_agent.step_count
            
            if msg_type == self.data_analysis_agent.MSG_CODE:
                # Extract code content
                code_match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
                if code_match:
                    code_content = code_match.group(1)
                    QTimer.singleShot(0, lambda: self.step_outputs.add_step(current_step, code=code_content))
            
            elif msg_type == self.data_analysis_agent.MSG_OUTPUT:
                # Extract output content
                output_content = text
                if text.startswith("Output:"):
                    output_content = text.split("Output:", 1)[1].strip()
                elif text.startswith("Error:"):
                    output_content = text.split("Error:", 1)[1].strip()
                    QTimer.singleShot(0, lambda: self.step_outputs.add_step(current_step, error=output_content))
                    
                    # Display error in chat using standard markdown
                    error_markdown = f"**Error:**\n```\n{output_content}\n```"
                    html = self.chat_widget.markdown_to_html(error_markdown)
                    self.chat_widget.add_agent_message(html)
                    return
                
                QTimer.singleShot(0, lambda: self.step_outputs.add_step(current_step, output=output_content))
        
        elif msg_type == self.data_analysis_agent.MSG_REFLECTION:
            # Reflection data should be a dictionary now
            if isinstance(text, dict):
                reflection_content = text.get("reflection_text", "Reflection not available.")
                has_error = text.get("has_error", False)
                # Add reflection to step outputs
                current_step = self.data_analysis_agent.step_count
                QTimer.singleShot(0, lambda: self.step_outputs.add_step(
                    current_step, 
                    reflection=reflection_content, 
                    error=True if has_error else None # Optionally mark the step as error if reflection indicates it
                ))
                
                # Store the reflection in the agent's mapping
                if hasattr(self.data_analysis_agent, 'reflection_mapping'):
                     self.data_analysis_agent.reflection_mapping[current_step] = reflection_content
                     
                # Optionally, display reflection briefly in status bar or a dedicated panel?
                # For now, just log it.
                print(f"Step {current_step} Reflection: {reflection_content[:100]}...")
                
            else:
                # Handle case where reflection is unexpectedly not a dict
                print(f"Warning: Received reflection message but data was not a dictionary: {text}")
                # Display raw text as fallback in step output
                current_step = self.data_analysis_agent.step_count
                QTimer.singleShot(0, lambda: self.step_outputs.add_step(
                    current_step, 
                    reflection=f"[Display Error] {str(text)[:200]}"
                ))
                
            # Don't add raw reflection dictionary to main chat
            return 
            
        elif msg_type == "plan":
            # Format Analysis Plan with Qt widgets instead of markdown
            from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QScrollArea, QWidget
            
            # Debug what we're receiving
            print(f"Received plan with content length: {len(text)}")
            
            # Handle plan text extraction
            if ":" in text:
                plan_content = text.split(":", 1)[1].strip()
            else:
                plan_content = text
            
            # Store the plan for toggling later
            self.analysis_plan = plan_content
            
            # Create professional plan display frame
            plan_frame = QFrame()
            plan_frame.setObjectName("planFrame")
            plan_frame.setFrameShape(QFrame.Shape.StyledPanel)
            plan_frame.setMinimumWidth(600)  # Set minimum width to ensure it's not too narrow
            plan_frame.setStyleSheet("""
                QFrame#planFrame {
                    border: 1px solid #4a6ee0;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 5px;
                }
            """)
            
            # Create plan layout
            plan_layout = QVBoxLayout(plan_frame)
            plan_layout.setContentsMargins(15, 15, 15, 15)
            plan_layout.setSpacing(10)
            
            # Add title
            title_label = QLabel("Analysis Plan")
            title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            plan_layout.addWidget(title_label)
            
            # Create a scroll area for the plan content
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.Shape.NoFrame)
            scroll_area.setMaximumHeight(400)  # Limit height
            
            # Create container for raw content
            content_container = QWidget()
            content_layout = QVBoxLayout(content_container)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(10)
            
            # Fix for escaped newlines in the plan content
            plan_content = plan_content.replace("\\n", "\n")
            
            # Create a label for raw content
            raw_content_label = QLabel()
            raw_content_label.setWordWrap(True)
            raw_content_label.setTextFormat(Qt.TextFormat.RichText)
            raw_content_label.setStyleSheet("font-size: 14px;")
            
            # Set HTML content with proper formatting from markdown
            formatted_content = self.chat_widget.markdown_to_html(plan_content)
            raw_content_label.setText(formatted_content)
            
            # Add the label to the layout
            content_layout.addWidget(raw_content_label)
            
            # Add stretcher to push content to top
            content_layout.addStretch()
            
            # Set the container widget as the scroll area's widget
            scroll_area.setWidget(content_container)
            plan_layout.addWidget(scroll_area)
            
            # Force setting vertical scroll bar policy
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            # Insert the plan frame into the chat widget
            chat_layout = self.chat_widget.chat_layout
            chat_layout.addWidget(plan_frame)
            
            # Directly scroll to bottom after adding the widget
            self.chat_widget.scroll_to_bottom()
        
        # Changes to the MSG_SUMMARY handler section around line 3956
        elif msg_type == self.data_analysis_agent.MSG_SUMMARY:
            # Create a professional summary with Qt widgets using the parsed JSON
            from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget, QPushButton, QComboBox
            import json
            
            # The 'text' received should now be the JSON object or the fallback error dict
            summary_data = text # Assuming text is already the dictionary
            
            # If text is somehow still a string, try parsing it (shouldn't happen with the agent changes)
            if isinstance(text, str):
                try:
                    if text.strip().startswith("```json"):
                        text = re.sub(r'^```json\s*|\s*```$', '', text.strip())
                    summary_data = json.loads(text)
                except json.JSONDecodeError:
                    print(f"Error: Received summary as string but failed to parse JSON: {text[:100]}...")
                    # Use the fallback error structure
                    summary_data = {
                        "executive_summary": "Error: Could not display summary.",
                        "key_findings": [f"Received invalid summary format."],
                        "methodology": "N/A",
                        "detailed_results": f"Raw response:\n{text}",
                        "visualizations_summary": "N/A",
                        "limitations": "N/A",
                        "next_steps": []
                    }

            # Ensure summary_data is a dictionary
            if not isinstance(summary_data, dict):
                print("Error: summary_data is not a dictionary!")
                summary_data = {
                    "executive_summary": "Error: Invalid summary data structure.",
                    "key_findings": [], "methodology": "", "detailed_results": "", 
                    "visualizations_summary": "", "limitations": "", "next_steps": []
                }
                
            # Store the structured summary content for potential continuation
            # We might want to store the whole dict or just the text parts
            self.latest_summary_content = summary_data.get("detailed_results", "") 
            # Extract data analysis steps for suggestions
            potential_next_steps = summary_data.get("data_analysis_steps", [])
            
            # Call the dedicated summary widget creation method
            # Pass the parsed dictionary directly
            self.chat_widget._add_professional_summary(summary_data)
            
            # Add context selection and continue button (outside the summary widget)
            context_container = QWidget()
            context_layout = QHBoxLayout(context_container)
            context_layout.setContentsMargins(10, 10, 10, 10)
            
            context_label = QLabel("Context for continuation:")
            context_layout.addWidget(context_label)
            
            self.context_dropdown = QComboBox()
            self.context_dropdown.addItems([
                "Use all analysis history", 
                "Use only last summary", 
                "Use last 2 summaries",
                "Use last 3 summaries"
            ])
            context_layout.addWidget(self.context_dropdown)
            context_layout.addStretch()
            
            continue_button = QPushButton("Continue")
            continue_button.setObjectName("continueButton")
            continue_button.setMinimumHeight(36)
            continue_button.setStyleSheet("""
                QPushButton#continueButton {
                    border-radius: 4px;
                    font-weight: bold;
                    padding: 8px 16px;
                }
            """)
            continue_button.clicked.connect(self.continue_analysis)
            context_layout.addWidget(continue_button)
            
            # Add context/continue container below the summary
            spacer_idx = self.chat_widget.chat_layout.count() - 1 # Before the final spacer
            self.chat_widget.chat_layout.insertWidget(spacer_idx, context_container)

            # Add suggestion chips if we have potential next steps
            if potential_next_steps:
                self.potential_next_steps = potential_next_steps
                suggestions_label = QLabel("Suggested next analysis steps:")
                suggestions_label.setStyleSheet("font-weight: bold; margin-top: 10px; margin-left: 10px;")
                self.chat_widget.chat_layout.insertWidget(spacer_idx + 1, suggestions_label) # After context
                
                suggestions_container = QWidget()
                suggestions_layout = QHBoxLayout(suggestions_container)
                suggestions_layout.setContentsMargins(10, 5, 10, 5)
                suggestions_layout.setSpacing(10)
                
                for step in potential_next_steps[:3]:
                    chip = QPushButton(step)
                    chip.setProperty("suggestion", step)
                    chip.setStyleSheet("""
                        QPushButton {
                            border: 1px solid #ddd;
                            border-radius: 16px;
                            padding: 6px 12px;
                            font-size: 13px;
                        }
                        QPushButton:hover {
                            border-color: #aaa;
                        }
                    """)
                    chip.clicked.connect(lambda checked, text=step: self.continue_analysis_with_suggestion(text))
                    suggestions_layout.addWidget(chip)
                
                suggestions_layout.addStretch()
                self.chat_widget.chat_layout.insertWidget(spacer_idx + 2, suggestions_container) # After label
            
            # Clear thinking indicator
            self.chat_widget.update_status("Analysis complete", is_thinking=False)
            
            # Scroll to bottom to show the summary
            QTimer.singleShot(100, self.chat_widget.scroll_to_bottom)
        
    # Add two new methods to the AgentInterface class
    def continue_analysis(self):
        """Open an input prompt for continuing the analysis."""
        # Create a task for continuation
        task = self.add_task("Continue Analysis", "Continuing from previous analysis")
        
        # Continuation is always enabled by default
        continuation_enabled = True
        
        # Set message for continuation
        prompt_msg = "How would you like to continue the analysis? Please enter your follow-up question."
        
        # Set a prompt in chat asking for the follow-up question
        self.chat_widget.add_agent_message(prompt_msg)
        
        # Add a flag to indicate we're continuing analysis
        self.is_continuing_analysis = True
        
        # Store the task for reference
        self.continuation_task = task
        
        # Store the continuation mode for reference (always enabled)
        self.continuation_mode_enabled = True
        
        # Set appropriate suggestions based on potential next steps
        if hasattr(self, 'potential_next_steps') and self.potential_next_steps:
            self.chat_widget.set_suggestions(self.potential_next_steps[:3])

    def continue_analysis_with_suggestion(self, suggestion_text):
        """Continue analysis with a specific suggestion."""
        # Continuation is always enabled
        continuation_enabled = True
        
        # Create task description for continuation
        task_description = f"Continuing with: {suggestion_text}"
        
        # Create a task for continuation
        task = self.add_task("Continue Analysis", task_description)
        
        # Set the flag for continuation
        self.is_continuing_analysis = True
        
        # Store the task and mode (always enabled)
        self.continuation_task = task
        self.continuation_mode_enabled = True
        
        # Add the suggestion as a user message
        self.chat_widget.add_user_message(suggestion_text)
        
        # Process the message for continuation
        self.process_continuation(suggestion_text, task, True)

    # Add a new method to handle the continuation logic
    def process_continuation(self, message, task, continuation_enabled=None):
        """Process a continuation of the previous analysis."""
        # Clear current tracking state but maintain previous results
        if hasattr(self, 'clear_current_analysis'):
            # Only partially clear state to keep history
            self.step_outputs.setVisible(True)  # Make sure outputs are visible
            # Don't clear graphs
        
        self.chat_widget.set_typing_indicator(True)
        
        # Switch to the outputs tab
        self.right_tabs.setCurrentIndex(0)
        
        # Configure analysis settings
        plan_depth = self.chat_widget.plan_depth_spinbox.value()
        max_graphs = self.chat_widget.max_graphs_spinbox.value()
        
        # Update UI
        self.chat_widget.update_status(f"Continuing analysis with plan depth {plan_depth}", is_thinking=True)
        
        # Get context level from dropdown
        context_level = "all"
        if hasattr(self, 'context_dropdown'):
            selected_option = self.context_dropdown.currentText()
            if "only last" in selected_option:
                context_level = "last"
            elif "last 2" in selected_option:
                context_level = "last2"
            elif "last 3" in selected_option:
                context_level = "last3"
        
        # Continuation is always enabled
        continuation_enabled = True
        
        print(f"Continuing analysis...")
        
        # Format message as a continuation
        enhanced_message = f"""ANALYSIS_CONTINUATION:
- Continue from previous analysis 
- Follow-up question: {message}
- Context level: {context_level}
- Continuation mode: enabled
ANALYSIS_CONFIG:
- Plan depth: {plan_depth} (number of analysis steps to perform)
- Max visualizations: {max_graphs} (maximum visualizations per step)
- Create clear, visually appealing graphs with informative titles and labels
- Focus on the most important insights in the data
"""
        
        # Add the summary content as context
        if hasattr(self, 'latest_summary_content'):
            enhanced_message += f"\nPREVIOUS_SUMMARY:\n{self.latest_summary_content}\n"
        
        print(f"Continuing analysis with message: {enhanced_message[:100]}...")
        
        # Create worker thread with continuation flag
        self.worker_thread = QThread()
        self.worker = AgentWorker(self.data_analysis_agent, enhanced_message)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.progress.connect(self.update_analysis_output)
        self.worker.figure_created.connect(self.handle_new_figure)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.started.connect(self.worker.run)
        
        # Cleanup
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        # Add timeout detection
        def check_for_timeout():
            if task.name in self.active_tasks and self.active_tasks[task.name].status == TaskStatus.RUNNING:
                self.update_task_status(task.name, TaskStatus.COMPLETED)
        
        self.worker_thread.finished.connect(lambda: QTimer.singleShot(100, check_for_timeout))
        
        # Start the worker thread
        self.worker_thread.start()
        
        # Update task status
        self.update_task_status(task.name, TaskStatus.RUNNING)
        
        # Add initial thinking status
        self.chat_widget.update_status("Continuing analysis...", is_thinking=True)

    
    def handle_new_figure(self, figure_data):
        """Handle a new figure from the worker thread."""
        try:
            # Figure data can be either a figure object or a tuple of (figure, step_title)
            if isinstance(figure_data, tuple) and len(figure_data) == 2:
                figure, step_title = figure_data
            else:
                figure = figure_data
                step_title = f"Step {len(AgentInterface.processed_figures) + 1}"
                
            # Ensure interactive mode is off
            plt.ioff()
            
            # Store reference to prevent garbage collection
            if not hasattr(AgentInterface, 'processed_figures'):
                AgentInterface.processed_figures = []
            
            # Set a flag to indicate we're currently processing this figure
            # This helps prevent concurrent modification issues
            if not hasattr(self, '_figure_processing_lock'):
                self._figure_processing_lock = False
                
            if self._figure_processing_lock:
                # Store for later processing
                if not hasattr(self, '_pending_figures'):
                    self._pending_figures = []
                self._pending_figures.append((figure, step_title))
                return True
                
            self._figure_processing_lock = True
            
            # This is crucial: store figure reference in class variable to prevent garbage collection
            AgentInterface.processed_figures.append(figure)
            
            # Use direct method call instead of QTimer for thread safety
            try:
                # First make sure graph tab exists and is visible
                if not hasattr(self, 'right_tabs') or not self.right_tabs:
                    self._figure_processing_lock = False
                    return False
                
                # Make sure graph gallery exists
                if not hasattr(self, 'graph_gallery') or not self.graph_gallery:
                    self._figure_processing_lock = False
                    return False
                
                # Make graph tab visible
                graph_tab_index = 1  # Index of the Graphs tab
                self.right_tabs.setTabVisible(graph_tab_index, True)
                
                # Force redraw of right tabs
                self.right_tabs.update()
                
                # Switch to graph tab - use direct method call
                self.right_tabs.setCurrentIndex(graph_tab_index)
                
                # Force UI update to ensure tab switch occurs
                self.right_tabs.repaint()
                QApplication.processEvents()
                
                # Add figure directly to graph gallery with step title
                gallery_index = self.graph_gallery.add_graph(figure, step_title)
                
                # Show in main graph widget if it exists
                if hasattr(self, 'graph_widget') and self.graph_widget:
                    success = self.graph_widget.show_matplotlib_figure(figure)
                
                # Force UI update to ensure visibility
                self.graph_gallery.update()
                self.graph_gallery.repaint()
                QApplication.processEvents()
                
                # Process any pending figures after a short delay using a Python-native approach
                # to avoid QTimer-related thread issues
                if hasattr(self, '_pending_figures') and self._pending_figures:
                    # Get the next pending figure
                    next_figure, next_title = self._pending_figures.pop(0)
                    # Release lock to allow next figure to be processed
                    self._figure_processing_lock = False
                    # Process it
                    self.handle_new_figure((next_figure, next_title))
                else:
                    # No pending figures, release the lock
                    self._figure_processing_lock = False
                
                return True
            except Exception as e:
                self._figure_processing_lock = False
                return False
                
        except Exception as e:
            if hasattr(self, '_figure_processing_lock'):
                self._figure_processing_lock = False
            return False
    
    def _add_figure_to_gallery(self, figure, step_number):
        """DEPRECATED: This method is no longer used. Functionality moved to handle_new_figure."""
        print("WARNING: _add_figure_to_gallery is deprecated. Use handle_new_figure instead.")
        pass
    
    def _ensure_graph_tab_visible(self):
        """Force the graph tab to be visible and selected."""
        try:
            print("📊 Ensuring graph tab is visible and selected")
            
            # Define graph tab index (1)
            graph_tab_index = 1
            
            # Make sure all tabs are visible
            self.right_tabs.setVisible(True)
            for i in range(self.right_tabs.count()):
                self.right_tabs.setTabVisible(i, True)
            
            # Force repaint of tabs
            self.right_tabs.repaint()
            
            # Switch to graph tab forcefully
            self.right_tabs.setCurrentIndex(graph_tab_index)
            
            # Process events to ensure tab change takes effect
            QApplication.processEvents()
            
            # Verify that the tab change worked
            current_tab = self.right_tabs.currentIndex()
            print(f"📊 Current tab index after switch: {current_tab}")
            
            # Make sure graph gallery and widget are visible
            if hasattr(self, 'graph_gallery'):
                self.graph_gallery.setVisible(True)
                self.graph_gallery.repaint()
                
            if hasattr(self, 'graph_widget'):
                self.graph_widget.setVisible(True)
                
            return True
        except Exception as e:
            print(f"❌ Error ensuring graph tab visible: {str(e)}")
            traceback.print_exc()
            return False
    
    def _check_figure_thread_timeout(self):
        """Check if the figure processing thread is still running after timeout."""
        if hasattr(self, 'figure_thread') and self.figure_thread.isRunning():
            print("Figure processing timeout detected - forcing quit")
            # Force quit the thread
            self.figure_thread.quit()
            # Show error message
            self.chat_widget.add_agent_message(
                "<div style='color: #e74c3c;'>Figure processing timed out. This may indicate a complex visualization that couldn't be rendered.</div>"
            )
    
    def _handle_processed_figure(self, fig_copy, step_number):
        """Handle a successfully processed figure by displaying it directly in the graph tab."""
        try:
            print(f"📊 Showing figure directly in graph tab")
            
            # Use QTimer to handle UI operations safely from worker threads
            from PyQt6.QtCore import QTimer
            
            # Store figure in static list to prevent garbage collection
            if not hasattr(AgentInterface, 'processed_figures'):
                AgentInterface.processed_figures = []
            AgentInterface.processed_figures.append(fig_copy)
            
            # Create a function to run in the main thread
            def update_ui_in_main_thread():
                try:
                    # Switch to graph tab BEFORE adding the figure
                    # This ensures the tab is active and ready to receive the figure
                    if hasattr(self, '_switch_to_graph_tab'):
                        self._switch_to_graph_tab()
                    else:
                        # Fallback if method is missing
                        graph_tab_index = 1
                        if hasattr(self, 'right_tabs') and self.right_tabs:
                            self.right_tabs.setCurrentIndex(graph_tab_index)
                            self.right_tabs.repaint()
            
                    # Force application to process events to ensure tab change is complete
                    QApplication.processEvents()
            
                    # Add figure to gallery for thumbnail view
                    if hasattr(self, 'graph_gallery') and self.graph_gallery:
                        # Use proper step title format instead of "Graph X" to ensure proper grouping
                        step_num = getattr(self.agent, 'step_count', 1)  # Default to step 1 
                        title = f"Step {step_num}"
                        self.graph_gallery.add_graph(fig_copy, title)
            
                    # Display in main graph widget
                    if hasattr(self, 'graph_widget') and self.graph_widget:
                        self.graph_widget.show_matplotlib_figure(fig_copy)
            
                    # Emit message to chat widget about the figure
                    if hasattr(self, 'add_agent_message'):
                        pass  # Skip adding notification message
                    
                    # Force UI update only if we have the widgets
                    if hasattr(self, 'graph_gallery'):
                        self.graph_gallery.repaint()
                    if hasattr(self, 'right_tabs'):
                        self.right_tabs.repaint()
                    QApplication.processEvents()
            
                    print(f"✅ Figure successfully added to graph tab")
                except Exception as e:
                    print(f"❌ Error in main thread UI update: {str(e)}")
                    traceback.print_exc()
            
            # Schedule the UI update for the main thread
            QTimer.singleShot(0, update_ui_in_main_thread)
            
            return True
        except Exception as e:
            print(f"❌ Error handling processed figure: {str(e)}")
            traceback.print_exc()
            return False

    def _switch_to_graph_tab(self):
        """Switch to the graph tab."""
        try:
            print("📊 Switching to graph tab")
            
            # Define graph tab index (1)
            graph_tab_index = 1
            
            # Make sure all tabs are visible
            self.right_tabs.setVisible(True)
            for i in range(self.right_tabs.count()):
                self.right_tabs.setTabVisible(i, True)
            
            # Repaint tabs
            self.right_tabs.repaint()
            
            # Switch to graph tab
            self.right_tabs.setCurrentIndex(graph_tab_index)
            
            return True
        except Exception as e:
            print(f"❌ Error switching to graph tab: {str(e)}")
            traceback.print_exc()
            return False
    
    def _handle_figure_processing_error(self, error_message):
        """Handle figure processing error."""
        self.chat_widget.add_agent_message(f"<div style='color: #e74c3c;'>{error_message}</div>")
    
    def add_task(self, name: str, description: str = "") -> TimelineTask:
        """Add a task to the timeline."""
        # Create the task
        task = TimelineTask(name, description)
        
        # Add to timeline widget
        self.timeline.add_task(task)
        
        # Track active tasks
        self.active_tasks[name] = task
        
        return task
    
    def update_task_status(self, task_name: str, status: TaskStatus):
        """Update a task's status."""
        if task_name in self.active_tasks:
            # Update in timeline
            self.timeline.update_task_status(task_name, status)
            
            # Update our record
            self.active_tasks[task_name].status = status

    def get_task(self, task_name: str) -> TimelineTask:
        """Get a task by name."""
        return self.active_tasks.get(task_name)
    
    def clear_graphs(self):
        """Clear any displayed graphs."""
        if hasattr(self, 'graph_widget'):
            self.graph_widget.clear()
            
    def show_message(self, message: str):
        """Show a message in the chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.add_agent_message(message)
            
    def set_suggestions(self, suggestions: list):
        """Set suggestion chips in the chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.set_suggestions(suggestions)
    
    def closeEvent(self, event):
        """Handle close event."""
        # Clean up any running worker threads
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            print("Cleaning up worker thread")
            self.worker_thread.quit()
            # Wait with timeout
            if not self.worker_thread.wait(1000):  # 1 second timeout
                print("Force terminating worker thread")
                self.worker_thread.terminate()
        
        # Clean up any running agent threads
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            print("Cleaning up agent thread")
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
            
        event.accept()

    def add_agent_message(self, message: str):
        """Add an agent message to the chat."""
        if hasattr(self, 'chat_widget'):
            self.chat_widget.add_agent_message(message)

    def apply_theme(self, is_dark=None):
        """Apply the current theme to the agent interface."""
        # Get the dark mode status
        if is_dark is None and hasattr(self.main_window, 'current_theme'):
            is_dark = self.main_window.current_theme == "dark"
        elif is_dark is None:
            is_dark = False
        
        # Get the stylesheet (colors have been removed here)
        theme_stylesheet = get_agent_stylesheet()
        
        # Add settings styling based on theme
        if is_dark:
            settings_style = """
                #settingLabel {
                    font-size: 11px;
                }
                #settingSpinbox {
                    border-radius: 3px;
                    padding: 2px;
                    max-height: 24px;
                }
                #settingsBar {
                    background-color: transparent;
                    margin-bottom: 5px;
                }
            """
        else:
            settings_style = """
                #settingLabel {
                    font-size: 11px;
                }
                #settingSpinbox {
                    border-radius: 3px;
                    padding: 2px;
                    max-height: 24px;
                }
                #settingsBar {
                    background-color: transparent;
                    margin-bottom: 5px;
                }
            """
        
        # Apply stylesheet to main widget and ensure it propagates to children
        self.setStyleSheet(theme_stylesheet + settings_style)
        
        # Force style refresh on key components
        if hasattr(self, 'chat_widget'):
            # Force update of chat container background
            self.chat_widget.chat_container.setProperty("theme", "dark" if is_dark else "light")
            self.chat_widget.chat_container.style().unpolish(self.chat_widget.chat_container)
            self.chat_widget.chat_container.style().polish(self.chat_widget.chat_container)
            
            # Force update of input area
            self.chat_widget.message_input.style().unpolish(self.chat_widget.message_input)
            self.chat_widget.message_input.style().polish(self.chat_widget.message_input)
            
            # Force update send and upload buttons
            self.chat_widget.send_button.style().unpolish(self.chat_widget.send_button)
            self.chat_widget.send_button.style().polish(self.chat_widget.send_button)
            self.chat_widget.upload_button.style().unpolish(self.chat_widget.upload_button)
            self.chat_widget.upload_button.style().polish(self.chat_widget.upload_button)
            
            # Force update spinboxes
            if hasattr(self.chat_widget, 'plan_depth_spinbox'):
                self.chat_widget.plan_depth_spinbox.style().unpolish(self.chat_widget.plan_depth_spinbox)
                self.chat_widget.plan_depth_spinbox.style().polish(self.chat_widget.plan_depth_spinbox)
            
            if hasattr(self.chat_widget, 'max_graphs_spinbox'):
                self.chat_widget.max_graphs_spinbox.style().unpolish(self.chat_widget.max_graphs_spinbox)
                self.chat_widget.max_graphs_spinbox.style().polish(self.chat_widget.max_graphs_spinbox)
        
            # Force update chat bubbles - iterate through all existing bubbles
            for i in range(self.chat_widget.chat_layout.count()):
                item = self.chat_widget.chat_layout.itemAt(i)
                if item and item.widget():
                    container = item.widget()
                    # Look for bubbles inside the container
                    if hasattr(container, 'layout'):
                        container_layout = container.layout()
                        for j in range(container_layout.count()):
                            bubble_item = container_layout.itemAt(j)
                            if bubble_item and bubble_item.widget():
                                bubble = bubble_item.widget()
                                if isinstance(bubble, QFrame) and (
                                    bubble.objectName() == "userBubble" or 
                                    bubble.objectName() == "agentBubble"
                                ):
                                    bubble.style().unpolish(bubble)
                                    bubble.style().polish(bubble)
        
            # Force update suggestion chips if visible
            if self.chat_widget.suggestion_container.isVisible():
                suggestion_layout = self.chat_widget.suggestion_container.layout()
                for i in range(suggestion_layout.count()):
                    item = suggestion_layout.itemAt(i)
                    if item and item.widget():
                        chip = item.widget()
                        if chip.objectName() == "suggestionChip":
                            chip.style().unpolish(chip)
                            chip.style().polish(chip)
        
        # Update tab icons if necessary (icons themselves may be updated externally)
        
    # First, add a new method to the class to implement the new chat functionality
    def new_chat(self):
        """Reset the interface for a new chat/analysis session."""
        # Clear analysis state
        if hasattr(self, 'clear_current_analysis'):
            self.clear_current_analysis()
        
        # Clear chat history completely
        if hasattr(self, 'chat_widget') and self.chat_widget:
            # Store a reference to the chat layout
            chat_layout = self.chat_widget.chat_layout
            
            # Remove all widgets from chat layout
            while chat_layout.count() > 0:
                item = chat_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Re-add the spacer at the end
            chat_layout.addStretch()
        
        # Reset any continuation flags
        self.is_continuing_analysis = False
        if hasattr(self, 'latest_summary_content'):
            self.latest_summary_content = None
        
        # Clear any tracked next steps
        if hasattr(self, 'potential_next_steps'):
            self.potential_next_steps = []
        
        # Add welcome message based on current mode
        if self.conversational_mode_rb.isChecked():
            welcome_msg = "Welcome to Chat Assistant! How can I help you today?"
        else:
            welcome_msg = "Ready for data analysis. Upload data files and enter your analysis query."
        
        self.chat_widget.add_agent_message(welcome_msg)
        
        # Reset mode-specific suggestions
        self.on_mode_changed()
        
        # Update UI
        if hasattr(self, 'right_tabs'):
            self.right_tabs.setCurrentIndex(0)  # Switch to Outputs tab
        
        # Process pending events to ensure UI updates
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()


# Create new class for graph gallery
class GraphGalleryWidget(QWidget):
    """Widget for displaying multiple graphs with thumbnails."""
    
    graphSelected = pyqtSignal(int)  # Signal when a graph is selected
    
    def __init__(self, parent=None):
        """Initialize the gallery."""
        super().__init__(parent)
        self.graphs = []  # Will hold dictionaries with figure and frame info
        self.selected_index = -1  # No graph selected by default
        self.seen_figures = set()  # Track figure "fingerprints" to avoid duplicates
        self.step_counters = {}  # Track number of figures per step
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Create title label
        title_label = QLabel("Graph Gallery")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.main_layout.addWidget(title_label)
        
        # Create scroll area for thumbnails
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create container for scrolling
        scroll_container = QWidget()
        
        # Create layout for graphs
        self.layout = QVBoxLayout(scroll_container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(25)  # Increased spacing between step sections
        
        # Add empty message
        self.empty_label = QLabel("No graphs yet. Run data analysis to generate visualizations.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(" padding: 20px;")
        self.layout.addWidget(self.empty_label)
        
        # Set scroll area widget
        scroll_area.setWidget(scroll_container)
        
        # Add scroll area to main layout
        self.main_layout.addWidget(scroll_area)
        
        print("✅ Graph gallery initialized")
        
    def fingerprint_figure(self, figure):
        """Create a fingerprint to identify duplicate figures.
        
        Returns a tuple that should be the same for duplicate figures.
        """
        # Get all titles and labels from the figure
        titles = []
        labels = []
        for ax in figure.get_axes():
            if ax.get_title():
                titles.append(ax.get_title())
            if ax.get_xlabel():
                labels.append(ax.get_xlabel())
            if ax.get_ylabel():
                labels.append(ax.get_ylabel())
        
        # Get data from the first 5 lines/collections (if any)
        data_samples = []
        for ax in figure.get_axes():
            # Check lines
            for i, line in enumerate(ax.get_lines()[:5]):
                # Sample first and last few data points
                xdata = line.get_xdata()
                ydata = line.get_ydata()
                if len(xdata) > 0 and len(ydata) > 0:
                    data_samples.append((
                        str(xdata[0]),
                        str(ydata[0]),
                        str(xdata[-1]) if len(xdata) > 1 else "",
                        str(ydata[-1]) if len(ydata) > 1 else ""
                    ))
            
            # Check patches
            for i, patch in enumerate(ax.patches[:5]):
                data_samples.append(str(patch.get_height() if hasattr(patch, 'get_height') else 0))
        
        # Create tuple of all collected attributes
        return (tuple(titles), tuple(labels), tuple(data_samples))
        
    def is_duplicate_figure(self, figure):
        """Check if a figure is likely a duplicate of one we've already seen."""
        fingerprint = self.fingerprint_figure(figure)
        if fingerprint in self.seen_figures:
            return True
        self.seen_figures.add(fingerprint)
        return False
    
    def clean_empty_steps(self):
        """Remove any empty step containers."""
        items_to_remove = []
        
        # First pass to find empty containers
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if (hasattr(widget, 'property') and 
                    widget.property("widget_type") == "container" and
                    hasattr(widget, 'layout') and
                    widget.layout().count() == 0):
                    
                    step_number = widget.property("step_number")
                    items_to_remove.append((i, step_number, "container"))
        
        # Remove empty containers
        for index, step_number, widget_type in reversed(items_to_remove):
            item = self.layout.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
        
        # Find and remove orphaned headers (headers without containers)
        step_containers = set()
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if (hasattr(widget, 'property') and 
                    widget.property("widget_type") == "container"):
                    step_containers.add(widget.property("step_number"))
        
        # Find headers without containers
        items_to_remove = []
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if (hasattr(widget, 'property') and 
                    widget.property("widget_type") == "header"):
                    step_number = widget.property("step_number")
                    if step_number not in step_containers:
                        items_to_remove.append((i, step_number, "header"))
        
        # Remove orphaned headers
        for index, step_number, widget_type in reversed(items_to_remove):
            item = self.layout.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
        
        # Update the UI
        self.repaint()
        QApplication.processEvents()
    
    def add_graph(self, figure, title=None):
        """Add a graph to the gallery and make it immediately visible."""
        try:
            # Check for duplicate figures to avoid adding the same graph multiple times
            if self.is_duplicate_figure(figure):
                return -1
            
            # Apply pastel colors for better visibility
            self._apply_pastel_colors(figure)
            
            # Parse step information from title or directly from figure
            step_number = "1"  # Default to step 1 instead of 0
            
            # First try to get plan_step directly from figure if it exists
            if hasattr(figure, 'plan_step'):
                # Skip step 0
                if figure.plan_step == 0:
                    figure.plan_step = 1
                step_number = str(figure.plan_step)
            # Then try to parse from title
            elif title:
                # First look for "Plan Step X" format
                plan_step_match = re.search(r'Plan\s+Step\s+(\d+)', title)
                if plan_step_match:
                    step_num = int(plan_step_match.group(1))
                    # Skip step 0
                    if step_num == 0:
                        step_num = 1
                    step_number = str(step_num)
                # Fall back to older "Step X" format if needed
                elif "Step" in title:
                    step_match = re.search(r'Step\s+(\d+)', title)
                    if step_match:
                        step_num = int(step_match.group(1))
                        # Skip step 0
                        if step_num == 0:
                            step_num = 1
                        step_number = str(step_num)
            
            # If this step has too many graphs already, use the last plan step instead
            # This helps spread out graphs across steps
            if step_number in self.step_counters and self.step_counters[step_number] >= 10:
                # Find the last available plan step
                valid_steps = [int(s) for s in self.step_counters.keys() if s.isdigit() and int(s) > 0]
                if valid_steps:
                    last_step = max(valid_steps)
                    if int(step_number) < last_step:
                        step_number = str(last_step)
                        print(f"⚠️ Step has too many graphs, reallocating to step {step_number}")
            
            # Update step counter
            self.step_counters[step_number] = self.step_counters.get(step_number, 0) + 1
            
            # We need to ensure each step's widgets are kept together - header followed by container
            # First, map out the existing step headers and containers
            step_widgets = {}  # Maps step_number -> (header_widget, container_widget, header_pos)
            
            # Discover existing step headers and containers
            for i in range(self.layout.count()):
                item = self.layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if hasattr(widget, 'property') and widget.property("step_number"):
                        widget_step = widget.property("step_number")
                        widget_type = widget.property("widget_type")
                        
                        # Initialize the tuple if we haven't seen this step yet
                        if widget_step not in step_widgets:
                            step_widgets[widget_step] = [None, None, -1]
                            
                        # Store header and its position
                        if widget_type == "header":
                            step_widgets[widget_step][0] = widget
                            step_widgets[widget_step][2] = i
                        # Store container
                        elif widget_type == "container":
                            step_widgets[widget_step][1] = widget
            
            # Check if we already have a header or container for this step
            header_exists = False
            container_exists = False
            if step_number in step_widgets:
                header_exists = step_widgets[step_number][0] is not None
                container_exists = step_widgets[step_number][1] is not None
                
            step_header = None
            step_container = None
            
            # Get the existing header if it exists
            if header_exists:
                step_header = step_widgets[step_number][0]
            
            # Get the existing container if it exists
            if container_exists:
                step_container = step_widgets[step_number][1]
            
            # Create the step header if it doesn't exist
            if not step_header:
                step_header = QFrame()
                step_header.setProperty("step_number", step_number)
                step_header.setProperty("widget_type", "header")
                step_header.setObjectName("stepHeader")
                step_header.setStyleSheet("""
                    QFrame#stepHeader {
                        border: 1px solid #dee2e6;
                        border-bottom: 2px solid #4a6ee0;
                        border-radius: 4px;
                        padding: 8px 15px;
                        margin-top: 15px;
                        margin-bottom: 5px;
                    }
                """)
                
                header_layout = QHBoxLayout(step_header)
                header_layout.setContentsMargins(5, 5, 5, 5)
                header_layout.setSpacing(10)
                
                # Create step number badge for better visibility
                step_badge = QLabel(step_number)
                step_badge.setObjectName("stepBadge")
                step_badge.setStyleSheet("""
                    QLabel#stepBadge {
                        font-weight: bold;
                        font-size: 14px;
                        padding: 2px 8px;
                        border-radius: 10px;
                        min-width: 20px;
                        max-width: 30px;
                        text-align: center;
                    }
                """)
                step_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                header_layout.addWidget(step_badge)
                
                # Always use "Plan Step" format for consistency
                step_label = QLabel("Plan Step")
                step_label.setStyleSheet("font-weight: bold; font-size: 14px;")
                header_layout.addWidget(step_label)
                
                # Add stretch to push toggle button to the right
                header_layout.addStretch()
                
                # Add collapse/expand button
                toggle_btn = QPushButton()
                toggle_btn.setIcon(QIcon(load_bootstrap_icon("arrows-collapse")))
                toggle_btn.setFixedSize(16, 16)
                toggle_btn.setFlat(True)
                toggle_btn.setProperty("state", "expanded")
                
                # Function to toggle container visibility
                def toggle_container():
                    # First find the container associated with this step
                    step_container_to_toggle = None
                    for i in range(self.layout.count()):
                        item = self.layout.itemAt(i)
                        if item and item.widget():
                            widget = item.widget()
                            if hasattr(widget, 'property') and widget.property("step_number") == step_number and widget.property("widget_type") == "container":
                                step_container_to_toggle = widget
                                break
                    
                    # If found, toggle visibility
                    if step_container_to_toggle:
                        current_visible = step_container_to_toggle.isVisible()
                        print(f"Toggling container visibility: {current_visible} -> {not current_visible}")
                        step_container_to_toggle.setVisible(not current_visible)
                        
                        # Update button icon
                        if current_visible:
                            toggle_btn.setIcon(QIcon(load_bootstrap_icon("arrows-expand")))
                            toggle_btn.setProperty("state", "collapsed")
                        else:
                            toggle_btn.setIcon(QIcon(load_bootstrap_icon("arrows-collapse")))
                            toggle_btn.setProperty("state", "expanded")
                        
                        # Force UI update
                        self.repaint()
                        QApplication.processEvents()
                
                toggle_btn.clicked.connect(toggle_container)
                header_layout.addWidget(toggle_btn)
            
            # Create the step container if it doesn't exist
            if not step_container:
                step_container = QWidget()
                step_container.setProperty("step_number", step_number)
                step_container.setProperty("widget_type", "container")
                
                # Use a grid layout instead of horizontal layout to arrange graphs better
                container_layout = QGridLayout(step_container)
                container_layout.setContentsMargins(5, 10, 5, 20)  # Increased bottom margin
                container_layout.setSpacing(15)  # Increased spacing between graphs
            
            # If neither header nor container exist, we need to add them at the correct position
            if not header_exists and not container_exists:
                # Find the right position to insert based on step number
                # Get all steps and their positions
                step_positions = []
                for s, (_, _, pos) in step_widgets.items():
                    if pos >= 0:  # Only consider headers with valid positions
                        try:
                            step_positions.append((int(s), pos))
                        except ValueError:
                            # Handle non-numeric steps
                            step_positions.append((999, pos))
                
                # Sort by step number
                step_positions.sort()
                
                # Find the right position to insert this step
                insert_pos = 0
                current_step_int = int(step_number)
                
                for other_step, pos in step_positions:
                    if current_step_int < other_step:
                        # Found a step with higher number, insert before it
                        insert_pos = pos
                        break
                    else:
                        # This step is already past our current one, so we need to
                        # insert after the container of this step if it exists
                        step_str = str(other_step)
                        if step_str in step_widgets and step_widgets[step_str][1] is not None:
                            # Find the position of this step's container
                            for i in range(self.layout.count()):
                                item = self.layout.itemAt(i)
                                if item and item.widget() == step_widgets[step_str][1]:
                                    insert_pos = i + 1
                                    break
                        else:
                            # If no container, use the next position after this header
                            insert_pos = pos + 1
                
                # If no insertion point was found (empty or all steps > current), insert at end
                if insert_pos == 0 and len(step_positions) > 0:
                    insert_pos = self.layout.count() - 1  # Before the stretch at the end
                
                # Insert the header and container in order
                self.layout.insertWidget(insert_pos, step_header)
                self.layout.insertWidget(insert_pos + 1, step_container)
                
                # Update for future reference
                step_widgets[step_number] = [step_header, step_container, insert_pos]
            
            # If header exists but container doesn't, add container right after header
            elif header_exists and not container_exists:
                header_pos = step_widgets[step_number][2]
                if header_pos >= 0:
                    self.layout.insertWidget(header_pos + 1, step_container)
                    step_widgets[step_number][1] = step_container
            
            # If container exists but header doesn't (unusual), add header before container
            elif not header_exists and container_exists:
                # Find container position
                container_pos = -1
                for i in range(self.layout.count()):
                    item = self.layout.itemAt(i)
                    if item and item.widget() == step_container:
                        container_pos = i
                        break
                
                if container_pos >= 0:
                    self.layout.insertWidget(container_pos, step_header)
                    step_widgets[step_number][0] = step_header
                    step_widgets[step_number][2] = container_pos
            
            # Create frame for this graph
            frame = QFrame()
            frame.setObjectName("graphFrame")
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            # Set minimum size to ensure visibility
            frame.setMinimumSize(200, 200)
            frame.setStyleSheet("""
                QFrame#graphFrame {
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    background-color: transparent;
                }
                QFrame#graphFrame:hover {
                    border-color: #4a6ee0;
                }
            """)
            
            # Create a wrapper layout for the frame content
            wrapper_layout = QVBoxLayout(frame)
            wrapper_layout.setContentsMargins(8, 8, 8, 8)
            wrapper_layout.setSpacing(5)
            
            # Extract a meaningful title from the figure
            graph_title_text = None
            
            # First try to get title from figure's axes
            for ax in figure.get_axes():
                if ax.get_title():
                    graph_title_text = ax.get_title()
                    break
            
            # If no title from axes, try from the provided title
            if not graph_title_text and title:
                # Remove the "Plan Step X" or "Step X" prefix if present
                plan_step_prefix = re.search(r'Plan\s+Step\s+\d+:?\s*', title)
                if plan_step_prefix:
                    graph_title_text = title[plan_step_prefix.end():]
                else:
                    step_prefix = re.search(r'Step\s+\d+:?\s*', title)
                    if step_prefix:
                        graph_title_text = title[step_prefix.end():]
                    else:
                        graph_title_text = title
                        
            # Only add title if it adds information beyond the step number
            if graph_title_text and f"Plan Step {step_number}" not in graph_title_text and f"Step {step_number}" not in graph_title_text:
                graph_title = QLabel(graph_title_text)
                graph_title.setStyleSheet("font-weight: bold; font-size: 12px;")
                graph_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                wrapper_layout.addWidget(graph_title)
            
            # Image container
            image_container = QFrame()
            image_container.setObjectName("imageContainer")
            image_container.setStyleSheet("""
                QFrame#imageContainer {
                    border: none;
                    background-color: transparent;
                }
            """)
            
            # Use a simple vertical layout for the image container
            image_layout = QVBoxLayout(image_container)
            image_layout.setContentsMargins(0, 0, 0, 0)
            
            # Convert to SVG for consistent display
            buf = io.BytesIO()
            figure.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            svg_data = buf.getvalue().decode('utf-8')
            buf.close()
            
            # Create SVG widget
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            # Increase minimum height for better visibility
            svg_widget.setMinimumHeight(180)
            
            # Create aspect ratio widget
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            
            # Add SVG widget to image layout
            image_layout.addWidget(aspect_widget)
            
            # Add a maximize button in the top-right corner
            maximize_btn = QPushButton()
            maximize_btn.setIcon(QIcon(load_bootstrap_icon("arrows-fullscreen")))
            maximize_btn.setToolTip("Maximize")
            maximize_btn.setFixedSize(24, 24)
            maximize_btn.setFlat(True)
            # Position the maximize button at the top-right corner of the frame
            maximize_btn.setParent(frame)
            maximize_btn.move(frame.width() - 30, 10)
            frame.resizeEvent = lambda event, btn=maximize_btn, f=frame: btn.move(f.width() - 30, 10)
            
            # Store the index to pass to the fullscreen function
            index = len(self.graphs)
            
            # Connect the maximize button to show fullscreen dialog ONLY when clicked
            maximize_btn.clicked.connect(lambda: self._show_fullscreen_dialog(index))
            
            # Add image container to wrapper
            wrapper_layout.addWidget(image_container)
            
            # If step container has a layout, add the frame to it using grid layout
            if step_container and hasattr(step_container, 'layout'):
                container_layout = step_container.layout()
                if isinstance(container_layout, QGridLayout):
                    # Calculate grid position based on number of existing items
                    item_count = container_layout.count()
                    row = item_count // 2  # 2 columns
                    col = item_count % 2
                    
                    # Create a descriptive subtitle for the graph (useful when multiple graphs in one step)
                    graph_subtitle = QLabel(f"Fig {item_count+1}")
                    graph_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    graph_subtitle.setStyleSheet(" font-size: 10px;")
                    wrapper_layout.addWidget(graph_subtitle)
                    
                    # Add the frame to the grid
                    container_layout.addWidget(frame, row, col)
                else:
                    # Fallback for any other layout type
                    container_layout.addWidget(frame)
                
                # Make sure container is visible
                step_container.setVisible(True)
            
            # Hide empty label if it exists
            if hasattr(self, 'empty_label') and self.empty_label and self.empty_label.isVisible():
                self.empty_label.setVisible(False)
            
            # Add to graphs list
            stored_title = graph_title_text if graph_title_text else f"Graph {len(self.graphs) + 1}"
            self.graphs.append({
                'figure': figure,
                'frame': frame,
                'title': stored_title,
                'svg_data': svg_data,  # Save SVG data for better fullscreen display
                'step': step_number
            })
            
            # If this is the first graph, use it to update the main display
            if len(self.graphs) == 1 and hasattr(self.parent(), 'graph_widget'):
                # Force a repaint to ensure the widget is properly initialized
                self.repaint()
                QApplication.processEvents()
                self.parent().graph_widget.show_matplotlib_figure(figure)
            
            
            # Clean up any empty steps every few graphs
            if len(self.graphs) % 5 == 0:
                self.clean_empty_steps()
            
            # Force UI update to ensure visibility
            self.repaint()
            QApplication.processEvents()
            
            return len(self.graphs) - 1
        except Exception as e:
            print(f"❌ Error adding graph to gallery: {str(e)}")
            traceback.print_exc()
            return -1
            
    def _show_fullscreen_dialog(self, index):
        """Show the graph at the specified index in a fullscreen dialog."""
        try:
            if index < 0 or index >= len(self.graphs):
                return
                
            # Get the selected figure and SVG data
            graph_info = self.graphs[index]
            figure = graph_info['figure']
            title = graph_info['title']
            svg_data = graph_info.get('svg_data', None)
            
            # Create a fullscreen dialog
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Visualization - {title}")
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
            dialog.resize(1200, 900)  # Larger default size
            
            # Create layout
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Create SVG widget - use the stored SVG data if available
            svg_widget = QSvgWidget()
            
            if svg_data:
                # Use the stored SVG data directly
                svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
            else:
                # Generate new SVG data if needed
                try:
                    # Save figure to SVG in memory
                    buf = io.BytesIO()
                    figure.savefig(buf, format='svg', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    svg_data = buf.getvalue().decode('utf-8')
                    buf.close()
                    
                    # Load SVG data into widget
                    svg_widget.renderer().load(QByteArray(svg_data.encode('utf-8')))
                except Exception as e:
                    # Fallback to using a minimal SVG
                    error_svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="#f0f0f0"/>
<text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif" font-size="24px" fill="#ff0000">Error displaying figure</text>
</svg>'''
                    svg_widget.renderer().load(QByteArray(error_svg.encode('utf-8')))
            
            # Set SVG widget size policy for expanding
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            
            # Create aspect ratio maintaining container
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            
            # Add to layout
            layout.addWidget(aspect_widget)
            
            # Add close button
            close_button = QPushButton("Close")
            
            close_button.clicked.connect(dialog.close)
            close_button.setFixedWidth(100)
            
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            # Display modal dialog (blocks until closed)
            dialog.exec()
            
            return True
        except Exception as e:
            return False
    
    def _apply_pastel_colors(self, figure):
        """Apply pastel colors to the figure."""
        # Make sure we're working with the right figure
        plt.figure(figure.number)
        
        # Set figure background to white for better readability
        figure.patch.set_facecolor('white')
        
        # Iterate through axes
        for ax in figure.get_axes():
            # Set white background for each axis
            ax.set_facecolor('white')
            
            # Process lines with pastel colors
            for line in ax.get_lines():
                line.set_alpha(0.8)
            
            # Process patches (like bars in bar charts)
            for patch in ax.patches:
                # Reduce the saturation a bit
                patch.set_alpha(0.8)
            
            # Process collections (like scatter plots)
            for collection in ax.collections:
                collection.set_alpha(0.8)
    
    def select_graph(self, index):
        """Select a graph and display it in the main widget.
        
        Args:
            index: Index of the graph to select
        """
        try:
            # Check if index is valid
            if index < 0 or index >= len(self.graphs):
                return
            
            # Update selected index
            self.selected_index = index
            graph_info = self.graphs[index]
            
            # Remove empty label if it exists
            if self.empty_label and self.empty_label.parent():
                self.empty_label.setVisible(False)
            
            # Update all frames to show selection state
            for i, graph in enumerate(self.graphs):
                if i == index:
                    # Selected frame
                    graph['frame'].setStyleSheet("""
                        QFrame#graphFrame {
                            border: 2px solid #4a6ee0;
                            border-radius: 8px;
                        }
                    """)
                else:
                    # Unselected frame
                    graph['frame'].setStyleSheet("""
                        QFrame#graphFrame {
                            border: 1px solid #dee2e6;
                            border-radius: 8px;
                        }
                       
                    """)
            
            # Try to notify parent if it has a method to handle this
            if hasattr(self.parent(), 'graph_widget') and hasattr(self.parent().graph_widget, 'show_matplotlib_figure'):
                try:
                    print(f"📊 Showing selected graph in parent's graph widget")
                    self.parent().graph_widget.show_matplotlib_figure(graph_info['figure'])
                except Exception as e:
                    print(f"❌ Error showing figure in parent widget: {str(e)}")
            
            # Force UI update
            self.repaint()
            QApplication.processEvents()
            
            print(f"✅ Successfully selected graph at index {index}")
            
            # Emit signal for other components to know about the selection
            self.graphSelected.emit(index)
        except Exception as e:
            print(f"❌ Error selecting graph: {str(e)}")
            traceback.print_exc()

    def ensure_graph_tab_visible(self):
        """Make absolutely sure the graph tab is visible and selected."""
        try:
            print("🔎 Ensuring graph tab is visible and selected")
            
            # First make sure all tabs are visible
            self.right_tabs.setVisible(True)
            
            # Check if we have enough tabs
            tab_count = self.right_tabs.count()
            print(f"Tab count: {tab_count}")
            
            if tab_count >= 2:  # Graph tab should be at index 1
                # Make sure the tab is visible
                self.right_tabs.setTabVisible(1, True)
                print("Set graph tab visibility to True")
                
                # Force select the tab
                self.right_tabs.setCurrentIndex(1)
                print("Set current tab index to 1 (Graphs)")
                
                # Make the tab widget update
                self.right_tabs.repaint()
                
                # Process events to ensure UI updates
                QApplication.processEvents()
                
                # Bring window to front
                window = self.window()
                if window:
                    window.activateWindow()
                    window.raise_()
                
                print("Graph tab should now be visible and selected")
                return True
            else:
                print(f"ERROR: Not enough tabs ({tab_count})")
                return False
        except Exception as e:
            print(f"Error ensuring graph tab visibility: {e}")
            traceback.print_exc()
            return False


class SVGAspectRatioWidget(QWidget):
    """Widget that maintains SVG aspect ratio while allowing resizing."""
    
    def __init__(self, svg_widget):
        super().__init__()
        self.svg_widget = svg_widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(svg_widget)
        
        # Get the default size from the SVG renderer
        default_size = svg_widget.renderer().defaultSize()
        
        # Check for invalid size and provide a fallback
        if default_size.width() <= 0 or default_size.height() <= 0:
            self.aspect_ratio = 4/3  # Default fallback ratio
        else:
            self.aspect_ratio = default_size.width() / default_size.height()
        
        # Set minimum size constraints
        self.setMinimumHeight(150)
        self.setMinimumWidth(int(150 * self.aspect_ratio))
    
    def resizeEvent(self, event):
        """Maintain aspect ratio during resize."""
        # Call parent implementation first
        super().resizeEvent(event)
        
        # Get current dimensions
        width = event.size().width()
        height = event.size().height()
        
        # Calculate dimensions that maintain aspect ratio
        if height > 0:
            width_by_height = width / height
            if width_by_height > self.aspect_ratio:
                # Too wide, adjust width based on height
                new_width = height * self.aspect_ratio
                new_height = height
            else:
                # Too tall, adjust height based on width
                new_width = width
                new_height = width / self.aspect_ratio
        else:
            # Handle zero height case
            new_width = width
            new_height = width / self.aspect_ratio
        
        # Calculate center position
        x = max(0, (width - new_width) / 2)
        y = max(0, (height - new_height) / 2)
        
        # Set the geometry of the SVG widget
        self.svg_widget.setGeometry(
            int(x), int(y), int(new_width), int(new_height)
        )
        
        # Make sure SVG widget is visible
        self.svg_widget.setVisible(True)


def fig_to_svg(fig):
    """Convert a matplotlib figure to SVG string."""
    # Create a BytesIO buffer to save the figure to
    buf = io.BytesIO()
    
    try:
        # Save figure as SVG to the buffer
        fig.savefig(buf, format='svg', bbox_inches='tight')
        
        # Get the SVG string
        buf.seek(0)
        svg_bytes = buf.getvalue()
        svg_string = svg_bytes.decode('utf-8')
        
        # Check if SVG data appears valid
        if not svg_string.startswith('<?xml') and '<svg' not in svg_string:
            # Return a minimal valid SVG as fallback
            return '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="#f8f9fa"/>
<text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif" font-size="20px" fill="#dc3545">Error generating figure</text>
</svg>'''
        
        return svg_string
    except Exception as e:
        # Return a minimal valid SVG as fallback
        return '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
<rect width="100%" height="100%" fill="#f8f9fa"/>
<text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" font-family="sans-serif" font-size="20px" fill="#dc3545">Error generating figure</text>
</svg>'''
    finally:
        # Clean up
        buf.close()

class StepOutputWidget(QFrame):
    """Widget for displaying outputs organized by step."""
    
    def __init__(self, step_number, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.has_error = False
        self.has_reflection = False
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setObjectName("stepOutputWidget")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame#stepOutputWidget {
                border: 1px solid;
                border-radius: 8px;
                margin: 10px 0px;
            }
        """)
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 10, 15, 10)
        
        # Add header
        header_layout = QHBoxLayout()
        
        # Step label
        self.header_label = QLabel(f"Step {self.step_number}")
        self.header_label.setStyleSheet("""
            font-weight: bold;
            font-size: 16px;
        """)
        
        # Add collapse button
        self.collapse_button = QPushButton()
        self.collapse_button.setIcon(QIcon(load_bootstrap_icon("arrows-collapse")))
        self.collapse_button.setMaximumSize(24, 24)
        self.collapse_button.setFlat(True)
        self.collapse_button.clicked.connect(self.toggle_collapse)
        
        header_layout.addWidget(self.header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.collapse_button)
        
        # Content container
        self.content_container = QWidget()
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Code section
        self.code_label = QLabel("Code:")
        self.code_label.setStyleSheet("font-weight: bold;")
        self.code_text = QTextEdit()
        self.code_text.setReadOnly(True)
        self.code_text.setMaximumHeight(150)
        self.code_text.setStyleSheet("""
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-family: monospace;
        """)
        
        # Output section
        self.output_label = QLabel("Output:")
        self.output_label.setStyleSheet("font-weight: bold;")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(150)
        self.output_text.setStyleSheet("""
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-family: monospace;
        """)
        
        # Error section
        self.error_container = QWidget()
        self.error_layout = QVBoxLayout(self.error_container)
        self.error_layout.setContentsMargins(0, 0, 0, 0)
        self.error_label = QLabel("Error:")
        self.error_label.setStyleSheet("font-weight: bold;")
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setMaximumHeight(150)
        self.error_text.setStyleSheet("""
            border: 1px solid #f5c2c7;
            border-radius: 4px;
            font-family: monospace;
        """)
        self.error_layout.addWidget(self.error_label)
        self.error_layout.addWidget(self.error_text)
        self.error_container.setVisible(False)
        
        # Reflection section - initialize but don't add to layout yet
        # This will be added dynamically when there's a reflection to show
        self.reflection_container = QWidget()
        self.reflection_layout = QVBoxLayout(self.reflection_container)
        self.reflection_layout.setContentsMargins(0, 0, 0, 0)
        self.reflection_label = QLabel("Reflection:")
        self.reflection_label.setStyleSheet("font-weight: bold;")
        self.reflection_layout.addWidget(self.reflection_label)
        self.reflection_frame = None
        self.reflection_container.setVisible(False)
        
        # Add sections to content layout in specific logical order:
        # 1. Code first (what we're going to run)
        # 2. Output next (what running the code produced)
        # 3. Error if there was one (what went wrong)
        # 4. Reflection last (analysis of what happened)
        self.content_layout.addWidget(self.code_label)
        self.content_layout.addWidget(self.code_text)
        self.content_layout.addWidget(self.output_label)
        self.content_layout.addWidget(self.output_text)
        self.content_layout.addWidget(self.error_container)
        self.content_layout.addWidget(self.reflection_container)
        
        # Add layouts to main layout
        self.layout.addLayout(header_layout)
        self.layout.addWidget(self.content_container)
        
        # Collapsed state
        self.is_collapsed = False
    
    def set_code(self, code):
        """Set the code text."""
        self.code_text.setText(code)
    
    def set_output(self, output):
        """Set the output text."""
        self.output_text.setText(output)
    
    def set_error(self, error):
        """Set the error text and show error section."""
        if error and error.strip():
            self.has_error = True
            self.error_text.setText(error)
            self.error_container.setVisible(True)
            
            # Update header style to indicate error
            self.header_label.setStyleSheet("""
                font-weight: bold;
                font-size: 16px;
            """)
            
            # Update widget border
            self.setStyleSheet("""
                QFrame#stepOutputWidget {
                    border: 1px solid #e74c3c;
                    border-radius: 8px;
                    margin: 10px 0px;
                }
            """)
    
    def set_reflection(self, reflection):
        """Add reflection widget."""
        if reflection and reflection.strip():
            self.has_reflection = True
            
            # Remove any existing reflection
            if self.reflection_frame:
                self.reflection_layout.removeWidget(self.reflection_frame)
                self.reflection_frame.deleteLater()
            
            # Create new reflection widget
            self.reflection_frame = ReflectionWidget(has_error=self.has_error)
            self.reflection_frame.set_content(reflection)
            
            # Add to reflection layout
            self.reflection_layout.addWidget(self.reflection_frame)
            
            # Show the reflection container
            self.reflection_container.setVisible(True)
    
    def toggle_collapse(self):
        """Toggle collapsed state."""
        self.is_collapsed = not self.is_collapsed
        self.content_container.setVisible(not self.is_collapsed)
        
        # Update button icon
        icon_name = "arrows-expand" if self.is_collapsed else "arrows-collapse"
        self.collapse_button.setIcon(QIcon(load_bootstrap_icon(icon_name)))

class StepOutputsWidget(QWidget):
    """Widget for displaying all step outputs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.step_widgets = {}  # Map of step_number -> widget
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container for step widgets
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(15)
        self.container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Add empty placeholder
        self.empty_label = QLabel("No outputs yet")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("padding: 20px; font-style: italic;")
        self.container_layout.addWidget(self.empty_label)
        
        # Set container as scroll area widget
        self.scroll_area.setWidget(self.container)
        
        # Add scroll area to layout
        self.layout.addWidget(self.scroll_area)
    
    def add_step(self, step_number, code=None, output=None, error=None, reflection=None):
        """Add or update a step output widget."""
        # Remove placeholder if this is the first step
        if self.empty_label and self.container_layout.count() == 1:
            self.empty_label.deleteLater()
            self.empty_label = None
        
        # Create widget if it doesn't exist
        if step_number not in self.step_widgets:
            step_widget = StepOutputWidget(step_number)
            self.step_widgets[step_number] = step_widget
            
            # Add to container in the right position (sorted by step number)
            insert_index = 0
            for i in range(self.container_layout.count()):
                widget = self.container_layout.itemAt(i).widget()
                if isinstance(widget, StepOutputWidget) and widget.step_number > step_number:
                    break
                insert_index += 1
            
            self.container_layout.insertWidget(insert_index, step_widget)
        else:
            step_widget = self.step_widgets[step_number]
        
        # Update the widget with content in the correct order
        # First update code, output, error sections
        if code:
            step_widget.set_code(code)
        if output:
            step_widget.set_output(output)
        if error:
            step_widget.set_error(error)
            
        # Reflections always come after other content
        if reflection:
            # Use a longer delay to ensure reflection appears last even if added at same time
            # This ensures reflections are always shown after all other content for this step
            # Fix: Properly capture reflection parameter in lambda to prevent variable binding issues
            QTimer.singleShot(200, lambda content=reflection: step_widget.set_reflection(content))
        
        # Scroll to this step
        QTimer.singleShot(250, lambda step=step_number: self.scroll_to_step(step))
        
        return step_widget
    
    def scroll_to_step(self, step_number):
        """Scroll to a specific step."""
        if step_number in self.step_widgets:
            widget = self.step_widgets[step_number]
            self.scroll_area.ensureWidgetVisible(widget)


class FigureProcessor(QObject):
    """Worker for processing matplotlib figures in a separate thread.
    
    This uses the QObject worker pattern (moveToThread) which is safer than subclassing QThread.
    """
    processingComplete = pyqtSignal(object, int)  # Emits processed figure and step number
    processingFailed = pyqtSignal(str)  # Emits error message
    finished = pyqtSignal()  # Signal for cleanup
    
    def __init__(self, figure, step_number):
        super().__init__()
        self.figure = figure
        self.step_number = step_number
        self.timeout = 10  # Timeout in seconds
        self.buf = None  # Will create when needed
    
    @pyqtSlot()
    def process(self):
        """Process the figure in a separate thread.
        
        This prepares the figure for direct display in the UI without saving to disk.
        """
        try:
            # CRITICAL: Create a deep copy of the figure to avoid threading issues
            # This prevents crashes when the original figure's C++ object gets deleted
            fig_copy = None
            try:
                # Create a completely new figure with the same data
                fig_copy = plt.figure(figsize=self.figure.get_size_inches())
                
                # Copy all axes and their contents
                for ax_src in self.figure.get_axes():
                    # Get position and add a new axis in the same position
                    pos = ax_src.get_position()
                    ax_dest = fig_copy.add_axes(pos.bounds)
                    
                    # Copy the axis content
                    for line in ax_src.get_lines():
                        ax_dest.plot(line.get_xdata(), line.get_ydata(), 
                                     color=line.get_color(), 
                                     linestyle=line.get_linestyle(),
                                     marker=line.get_marker())
                    
                    # Copy titles and labels
                    ax_dest.set_title(ax_src.get_title())
                    ax_dest.set_xlabel(ax_src.get_xlabel())
                    ax_dest.set_ylabel(ax_src.get_ylabel())
                    
                    # Copy legend if exists
                    if ax_src.get_legend():
                        ax_dest.legend()
                
                # If we successfully copied the figure, proceed with processing
                if fig_copy:
                    print("✅ Successfully copied figure for direct display")
                else:
                    raise Exception("Figure copy failed")
                    
                    
            except Exception as e:
                print(f"⚠️ Error copying figure: {str(e)}")
                print("Using the original figure as fallback")
                # Last resort - try using the original figure 
                # This is risky and might cause thread issues if the original gets deleted
                fig_copy = self.figure
            
            # Emit the processed figure back to the main thread for immediate display
            # Use QueuedConnection to ensure it's handled in the receiving thread
            self.processingComplete.emit(fig_copy, self.step_number)
        except Exception as e:
            # Catch and report all exceptions
            error_message = f"Figure processing error: {str(e)}"
            print(f"❌ {error_message}")
            traceback.print_exc()
            self.processingFailed.emit(error_message)
        finally:
            # Always signal that we're done even if there was an error
            # This helps prevent memory/resource leaks
            print("🏁 Figure processing thread finishing")
            self.finished.emit()
    
    def add_figure_directly(self, fig, step_number):
        """
        Add a figure directly to the gallery without using worker threads.
        This is a fallback method if the normal signal-based approach is failing.
        """
        try:
            print(f"⚡ DIRECT: Adding figure for step {step_number} directly")
            
            # Verify that gallery and graph widget exist
            if not hasattr(self, 'graph_gallery') or not self.graph_gallery:
                print("❌ ERROR: graph_gallery not found!")
                return
                
            if not hasattr(self, 'graph_widget') or not self.graph_widget:
                print("❌ ERROR: graph_widget not found!")
                return
            
            # Add figure to gallery
            current_graph_count = len(self.graph_gallery.figures)
            print(f"⚡ DIRECT: Adding figure to gallery (current count: {current_graph_count})")
            self.graph_gallery.add_graph(fig, f"Step {step_number}")
            
            # Show figure in main graph widget
            print(f"⚡ DIRECT: Showing figure in main graph widget")
            self.graph_widget.show_matplotlib_figure(fig)
            
            # Create a minimal message for the chat
            self.chat_widget.add_agent_message(
                f"<div style='text-align: center;'>"
                f"<p style='font-style: italic;'>Figure generated for Step {step_number} - "
                f"<a href='#' onclick='window.graphClicked({current_graph_count})'>View in gallery</a></p>"
                f"</div>"
            )
            
            # Switch to graph gallery tab
            print("⚡ DIRECT: Scheduling switch to graph gallery tab")
            QTimer.singleShot(100, lambda: self._switch_to_graph_tab())
            
            return True
        except Exception as e:
            print(f"❌ ERROR in add_figure_directly: {str(e)}")
            traceback.print_exc()
            return False
            
    def _switch_to_graph_tab(self):
        """Switch to the graph tab."""
        try:
            print("📊 Switching to graph tab")
            
            # Define graph tab index (1)
            graph_tab_index = 1
            
            # Make sure all tabs are visible
            self.right_tabs.setVisible(True)
            for i in range(self.right_tabs.count()):
                self.right_tabs.setTabVisible(i, True)
            
            # Repaint tabs
            self.right_tabs.repaint()
            
            # Switch to graph tab
            self.right_tabs.setCurrentIndex(graph_tab_index)
            
            return True
        except Exception as e:
            print(f"❌ Error switching to graph tab: {str(e)}")
            traceback.print_exc()
            return False


def get_agent_stylesheet():
    """Get the stylesheet for the agent interface, with spacing and padding improvements but without any color modifications."""
    return """
        QWidget#modePanel {
            padding: 5px;
        }

        QRadioButton#modeRadioButton {
            border-radius: 4px;
            padding: 4px 8px;
        }

        QFrame#timelineTask {
            border: 1px solid;
            border-radius: 8px;
            margin: 8px 0px;
        }

        QLabel#taskName {
            font-weight: bold;
            font-size: 10pt;
        }

        QLabel#taskTime {
            font-size: 8pt;
        }

        QWidget#chatContainer {
        }

        QFrame#userBubble {
            border: 1px solid;
            border-radius: 18px;
            padding: 5px;
        }

        QFrame#agentBubble {
            border: 1px solid;
            border-radius: 18px;
            padding: 5px;
            min-width: 60%;
            max-width: 95%;
        }
        
        QFrame#specialContentBubble {
            border: 1px solid;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 5px;
            min-width: 95%;
            max-width: 98%;
        }
        
        QFrame#specialContentBubble QLabel {
            font-size: 14px;
            line-height: 1.4;
        }

        QWidget#inputContainer {
            padding: 5px;
        }

        QLabel#typingIndicator {
            font-style: italic;
            padding-left: 10px;
            font-size: 12px;
        }

        QTextEdit#messageInput {
            border: 1px solid;
            border-radius: 18px;
            padding: 10px 15px;
            font-size: 14px;
        }

        QTextEdit#messageInput:focus {
            border: 1px solid;
        }

        QPushButton#uploadButton, QPushButton#sendButton {
            border: 1px solid;
            border-radius: 18px;
        }

        QPushButton#sendButton {
        }

        QPushButton#suggestionChip {
            border: 1px solid;
            border-radius: 15px;
            padding: 5px 10px;
            text-align: center;
            font-size: 12px;
        }

        QFrame#reflectionHistoryItem, QFrame#reflectionWidget {
            border: 1px solid;
            border-radius: 8px;
            margin: 4px 0px;
            padding: 15px;
        }

        QPushButton#normalThumb {
            border: 1px solid;
            border-radius: 4px;
        }

        QPushButton#selectedThumb {
            border: 2px solid;
            border-radius: 4px;
        }

        QFrame#graphFrame {
            border: 1px solid;
            border-radius: 4px;
            padding: 10px;
        }
    """


class SVGAspectRatioWidget(QWidget):
    """Widget that maintains SVG aspect ratio while allowing resizing."""
    
    def __init__(self, svg_widget):
        super().__init__()
        self.svg_widget = svg_widget
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(svg_widget)
        
        # Get the default size from the SVG renderer
        default_size = svg_widget.renderer().defaultSize()
        
        # Check for invalid size and provide a fallback
        if default_size.width() <= 0 or default_size.height() <= 0:
            self.aspect_ratio = 4/3  # Default fallback ratio
        else:
            self.aspect_ratio = default_size.width() / default_size.height()
        
        # Set minimum size constraints
        self.setMinimumHeight(150)
        self.setMinimumWidth(int(150 * self.aspect_ratio))
    
    def resizeEvent(self, event):
        """Maintain aspect ratio during resize."""
        # Call parent implementation first
        super().resizeEvent(event)
        
        # Get current dimensions
        width = event.size().width()
        height = event.size().height()
        
        # Calculate dimensions that maintain aspect ratio
        if height > 0:
            width_by_height = width / height
            if width_by_height > self.aspect_ratio:
                # Too wide, adjust width based on height
                new_width = height * self.aspect_ratio
                new_height = height
            else:
                # Too tall, adjust height based on width
                new_width = width
                new_height = width / self.aspect_ratio
        else:
            # Handle zero height case
            new_width = width
            new_height = width / self.aspect_ratio
        
        # Calculate center position
        x = max(0, (width - new_width) / 2)
        y = max(0, (height - new_height) / 2)
        
        # Set the geometry of the SVG widget
        self.svg_widget.setGeometry(
            int(x), int(y), int(new_width), int(new_height)
        )
        
        # Make sure SVG widget is visible
        self.svg_widget.setVisible(True)


class ReflectionHistoryWidget(QWidget):
    """Stub class for backward compatibility - no longer used."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.reflections = []
        self.init_ui()
        
    def init_ui(self):
        """Initialize minimal UI."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
    def add_reflection(self, text, step_number=None):
        """Stub method that does nothing."""
        pass
        
    def scroll_to_bottom(self):
        """Stub method that does nothing."""
    def scroll_to_bottom(self):
        """Stub method that does nothing."""
        pass

