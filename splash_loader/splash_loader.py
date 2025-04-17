# splash_loader/splash_loader.py

import sys
import math
import random
from PyQt6.QtWidgets import QWidget, QApplication, QDialog, QVBoxLayout, QToolTip, QPushButton
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, 
    QPoint, QPointF, QSize, QRect, QRectF, pyqtProperty, pyqtSignal,
    QSequentialAnimationGroup, QParallelAnimationGroup, QVariantAnimation,
    QElapsedTimer
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient, 
    QRadialGradient, QFont, QPainterPath, QPixmap, QTransform, QFontMetrics
)

class ResearchFlowLoader(QWidget):
    """
    An animated infographic loader that visualizes the main sections
    of the CareFrame application.
    """
    animationCompleted = pyqtSignal()  # Emitted when animation cycle completes
    goButtonClicked = pyqtSignal()     # Emitted when the Go button is clicked
    nodeClicked = pyqtSignal(str)      # Emitted when a node (cube) is clicked, passes phase name
    
    def __init__(self, parent=None, load_time=12000):  # Default to 12 seconds loading time
        super().__init__(parent)
        self.setMinimumSize(700, 550) # Increased height slightly for second row
        
        # Animation state
        self.progress = 0.0
        self.is_dark_theme = False
        self.animation_speed = 1.0
        self.hover_node = None
        self.hover_node_type = None # 'main' or 'tool'
        
        # View mode tracking - New
        self.current_view_mode = 'main'  # 'main' or a specific main phase name for detailed view
        self.previous_view_mode = 'main'
        
        # Menu mode - when used as a navigation menu
        self.menu_mode = False  # When True, hide Go button and bottom text
        
        # Enable mouseover effects
        self.setMouseTracking(True)
        
        # Timer and loading time
        self.load_time = load_time
        self.elapsed_timer = QElapsedTimer() # Initialize QElapsedTimer
        self.elapsed_timer.start() # Start the timer
        self.time_remaining = load_time / 1000
        
        # Text animation
        self.text_animation_progress = 0.0
        self.current_text_index = 0
        self.show_text = True
        self.text_fade = 1.0
        
        # Animation phases - Main Workflow
        self.main_phases = [
            "plan",          # Strategy: Plan, Hypotheses
            "manage",        # Manage: Design, Docs, Participants, Protocol
            "literature",    # Literature: Search, Rank, Claims
            "data",          # Data: Sources, Clean, Merge, Filter, Reshape
            "analysis",      # Analysis: Model, Evaluate, Assumptions, Subgroup, etc.
            "evidence"       # Evidence: Blockchain / Validation
        ]
        
        # Animation phases - Secondary Tools
        self.tool_phases = [
            "agent",
            "network",
            "database",
            "team",
            "settings"
        ]
        
        self.phases = self.main_phases + self.tool_phases # Combined list for easier lookup
        
        # Icons and positions for each phase
        self.phase_icons = {
            # Main
            "plan": "diagram-3",
            "manage": "clipboard-data",
            "literature": "book",
            "data": "database-gear", # Changed icon
            "analysis": "graph-up",
            "evidence": "shield-check", # Changed icon
            # Tools
            "agent": "robot",
            "network": "hdd-network",
            "database": "database",
            "team": "people",
            "settings": "gear",
            # Sub-phases
            "hypotheses": "lightbulb",
            "design": "pencil-square",
            "documentation": "file-text",
            "participants": "person-badge",
            "protocol": "list-check",
            "search": "search",
            "ranking": "sort-numeric-down",
            "claims": "journal-check", 
            "sources": "database",
            "cleaning": "brush",
            "reshaping": "table",
            "filtering": "funnel",
            "joining": "link",
            "evaluation": "check-circle",
            "assumptions": "question-circle",
            "subgroup": "people-fill",
            "mediation": "arrow-left-right",
            "sensitivity": "sliders",
            "interpret": "chat-text",
            "evidence": "shield-shaded"
        }
        
        # Phase descriptions (full text) - updated
        self.phase_descriptions = {
            # Main
            "plan": "Strategy & Planning",
            "manage": "Study Management",
            "literature": "Literature Review",
            "data": "Data Operations", 
            "analysis": "Statistical Analysis",
            "evidence": "Evidence & Validation",
            # Tools
            "agent": "AI Assistant",
            "network": "Network & Collaboration",
            "database": "Database Admin",
            "team": "Team Management",
            "settings": "Application Settings",
            # Sub-phases - adding explicitly
            "hypotheses": "Hypotheses",
            "design": "Study Design",
            "documentation": "Documentation",
            "participants": "Participants",
            "protocol": "Protocol",
            "search": "Literature Search",
            "ranking": "Literature Ranking",
            "claims": "Literature Claims",
            "sources": "Data Sources",
            "cleaning": "Data Cleaning",
            "reshaping": "Data Reshaping",
            "filtering": "Data Filtering",
            "joining": "Data Joining",
            "evaluation": "Test Evaluation",
            "assumptions": "Assumptions",
            "subgroup": "Subgroup Analysis",
            "mediation": "Mediation Analysis",
            "sensitivity": "Sensitivity Analysis",
            "interpret": "Interpretation",
            "evidence": "Evidence Blockchain"
        }
        
        # Also initialize a special mapping for sub-node labels when they share names with main nodes
        self.sub_node_labels = {
            "plan": "Planning" # Different label for 'plan' when it's a sub-node
        }
        
        # Texts to display for each phase - simplified
        self.phase_texts = {
            # Main
            "plan": ["Define goals and hypotheses", "Design research strategy"],
            "manage": ["Oversee study execution", "Manage participants and protocols"],
            "literature": ["Explore existing research", "Extract and rank evidence"],
            "data": ["Connect and integrate data sources", "Clean, transform, and filter data"],
            "analysis": ["Build models and run tests", "Evaluate and interpret results"],
            "evidence": ["Validate findings on blockchain", "Ensure research integrity"],
            # Tools (simpler descriptions)
            "agent": ["Interact with the AI research assistant"],
            "network": ["Manage connections and data sharing"],
            "database": ["Perform database operations"],
            "team": ["Manage team members and permissions"],
            "settings": ["Configure application settings"]
        }
        
        # Visual elements
        self.main_nodes = [] # Separate list for main workflow nodes
        self.tool_nodes = [] # Separate list for tool nodes
        self.connections = []
        self.decorative_elements = []
        
        # Node pulse animations
        self.pulse_factors = {}
        self.pulse_animation = QVariantAnimation()
        self.pulse_animation.setStartValue(0.0)
        self.pulse_animation.setEndValue(1.0)
        self.pulse_animation.setDuration(2000)
        self.pulse_animation.setLoopCount(-1) # Infinite loop
        self.pulse_animation.valueChanged.connect(self.update_pulse)
        self.pulse_animation.start()
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)  # ~60 fps
        
        # Tracking state for animation
        self.current_main_phase_index = 0 # Track main phase for animation cycling
        
        # Create gradient for background
        self.update_gradients()
        
        # Generate decorative elements
        self.generate_decorative_elements()
        
        # For radial circle animations
        self.ripple_animations = {}
        self.last_ripple_time = 0
        
        # For cube edge lighting animation
        self.cube_edge_lights = {}
        for i in range(12): # 12 edges in a cube
            self.cube_edge_lights[i] = 0.0 # Light intensity (0.0-1.0)
        self.current_edge = 0
        self.edge_light_direction = 1 # 1: increasing, -1: decreasing
        
        # Link animation
        self.link_flow_offset = 0.0 # For flowing dash pattern
        
        # Hover label animation
        self.hover_label_state = {"node_index": None, "node_type": None, "blinks": 0, "opacity": 0.0, "settled": False, "pos": QPointF(20, 40)}
        
        # App ready state
        self.app_ready = False
        
        # Create Go Button that appears when app is ready
        self.go_button = QPushButton("Open CareFrame", self)
        self.go_button.setFixedSize(220, 50)
        self.go_button.clicked.connect(self.on_go_button_clicked)
        self.go_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.go_button.hide() # Initially hidden
        self.go_button_animation_phase = 0.0

    def update_gradients(self):
        """Update gradients based on the current theme"""
        if self.is_dark_theme:
            # Dark theme
            self.bg_gradient = QLinearGradient(0, 0, 0, self.height())
            self.bg_gradient.setColorAt(0, QColor(30, 34, 42))
            self.bg_gradient.setColorAt(0.5, QColor(22, 24, 31))
            self.bg_gradient.setColorAt(1, QColor(18, 20, 25))
            
            self.main_node_colors = {
                "default_face": QColor(60, 65, 80),
                "face_0": QColor(75, 80, 95), "face_3": QColor(65, 70, 85), "face_4": QColor(80, 85, 100),
                "edge": QColor(90, 95, 110), "highlight": QColor(120, 125, 140)
            }
            self.main_active_colors = {
                "default_face": QColor(120, 60, 170),
                "face_0": QColor(150, 80, 200), "face_3": QColor(130, 70, 180), "face_4": QColor(160, 90, 210),
                "edge": QColor(180, 130, 220), "highlight": QColor(220, 130, 255)
            }
            self.main_hover_colors = {
                "default_face": QColor(80, 140, 200), 
                "face_0": QColor(100, 160, 220), "face_3": QColor(80, 140, 190), "face_4": QColor(120, 180, 230),
                "edge": QColor(140, 200, 255), "highlight": QColor(160, 220, 255)
            }
            
            # Secondary tool colors (e.g., greenish)
            self.tool_node_colors = {
                "default_face": QColor(50, 80, 70),
                "face_0": QColor(60, 95, 85), "face_3": QColor(55, 85, 75), "face_4": QColor(65, 100, 90),
                "edge": QColor(80, 110, 100), "highlight": QColor(100, 130, 120)
            }
            self.tool_active_colors = { # Using active color for hover on tools
                "default_face": QColor(60, 179, 113), # MediumSeaGreen base
                "face_0": QColor(70, 200, 130), "face_3": QColor(50, 160, 100), "face_4": QColor(80, 210, 140),
                "edge": QColor(100, 220, 160), "highlight": QColor(144, 238, 144) # LightGreen highlight
            }
            self.tool_hover_colors = self.tool_active_colors # Reuse active for hover

            self.highlight_color = QColor(156, 39, 176) # Purple for main active
            self.secondary_highlight = QColor(33, 150, 243) # Blue for main hover
            self.tool_highlight_color = QColor(60, 179, 113) # Green for tool hover/active
            
            self.text_color = QColor(230, 230, 230)
            self.connection_start = QColor(70, 70, 90, 150)
            self.connection_end = QColor(50, 50, 70, 150)
            self.glow_color = QColor(156, 39, 176, 60) # Main glow
            self.tool_glow_color = QColor(60, 179, 113, 50) # Tool glow
        else:
            # Light theme
            self.bg_gradient = QLinearGradient(0, 0, 0, self.height())
            self.bg_gradient.setColorAt(0, QColor(250, 250, 255))
            self.bg_gradient.setColorAt(0.5, QColor(245, 245, 250))
            self.bg_gradient.setColorAt(1, QColor(235, 235, 245))
            
            self.main_node_colors = {
                "default_face": QColor(200, 200, 210),
                "face_0": QColor(230, 230, 235), "face_3": QColor(210, 210, 220), "face_4": QColor(240, 240, 245),
                "edge": QColor(180, 180, 190), "highlight": QColor(210, 210, 220)
            }
            self.main_active_colors = {
                "default_face": QColor(160, 100, 210),
                "face_0": QColor(180, 130, 230), "face_3": QColor(160, 100, 210), "face_4": QColor(190, 150, 240),
                "edge": QColor(140, 80, 180), "highlight": QColor(200, 100, 240)
            }
            self.main_hover_colors = {
                 "default_face": QColor(120, 180, 240),
                 "face_0": QColor(150, 200, 255), "face_3": QColor(120, 170, 230), "face_4": QColor(170, 210, 255),
                 "edge": QColor(100, 160, 220), "highlight": QColor(130, 190, 255)
            }
            
            # Secondary tool colors (light theme - maybe light green/teal)
            self.tool_node_colors = {
                "default_face": QColor(200, 220, 210),
                "face_0": QColor(220, 235, 225), "face_3": QColor(210, 225, 215), "face_4": QColor(230, 240, 230),
                "edge": QColor(180, 200, 190), "highlight": QColor(200, 210, 200)
            }
            self.tool_active_colors = { # Using active color for hover on tools
                "default_face": QColor(102, 187, 106), # Light Green base
                "face_0": QColor(120, 200, 125), "face_3": QColor(90, 170, 95), "face_4": QColor(130, 210, 135),
                "edge": QColor(150, 220, 155), "highlight": QColor(174, 238, 174) # Lighter green highlight
            }
            self.tool_hover_colors = self.tool_active_colors

            self.highlight_color = QColor(156, 39, 176) # Purple
            self.secondary_highlight = QColor(33, 150, 243) # Blue
            self.tool_highlight_color = QColor(102, 187, 106) # Light Green

            self.text_color = QColor(60, 60, 70)
            self.connection_start = QColor(180, 180, 200, 150)
            self.connection_end = QColor(150, 150, 170, 150)
            self.glow_color = QColor(156, 39, 176, 40) # Main glow
            self.tool_glow_color = QColor(102, 187, 106, 35) # Tool glow
            
    # Removed generate_decorative_elements and update_decorative_elements for brevity
    def generate_decorative_elements(self):
        """Generate decorative background elements"""
        self.decorative_elements = []
        
        w, h = self.width(), self.height()
        for i in range(15):
            size = random.uniform(4, 10)
            margin = 50
            x = random.uniform(margin, w - margin)
            y = random.uniform(margin, h - margin)
            alpha = int(random.uniform(20, 80))
            
            if self.is_dark_theme:
                color = QColor(int(random.uniform(120, 220)), 
                              int(random.uniform(120, 220)), 
                              int(random.uniform(120, 240)), 
                              alpha)
            else:
                color = QColor(int(random.uniform(100, 180)), 
                              int(random.uniform(100, 180)), 
                              int(random.uniform(150, 220)), 
                              alpha)
            
            speed = random.uniform(0.2, 1.0)
            amplitude = random.uniform(5, 15)
            phase = random.uniform(0, 2 * math.pi)
            
            self.decorative_elements.append({
                "x": x, "y": y, "size": size, "color": color,
                "speed": speed, "amplitude": amplitude, "phase": phase
            })
            
    def update_decorative_elements(self):
        """Update positions of decorative elements"""
        for element in self.decorative_elements:
            element["phase"] += element["speed"] * 0.05
            if element["phase"] > 2 * math.pi: element["phase"] -= 2 * math.pi
            element["x_offset"] = math.sin(element["phase"]) * element["amplitude"]
            element["y_offset"] = math.cos(element["phase"]) * element["amplitude"]

    def set_theme(self, is_dark):
        """Set the theme (dark or light)"""
        self.is_dark_theme = is_dark
        self.update_gradients()
        self.generate_decorative_elements() # Regenerate with new theme colors
        self.update_button_style()
        self.update()
        
    def update_pulse(self, value):
        """Update pulse animation value"""
        # Pulse main nodes based on current main phase
        current_main_idx = self.current_main_phase_index
        for i, node in enumerate(self.main_nodes):
            phase_name = node["phase"]
            pulse_key = f"main_{phase_name}"
            if i == current_main_idx:
                self.pulse_factors[pulse_key] = 0.85 + 0.15 * math.sin(value * 2 * math.pi)
            else:
                self.pulse_factors[pulse_key] = 0.95 + 0.05 * math.sin((value + i * 0.1) * 2 * math.pi)
        
        # Apply subtle pulse to tool nodes (maybe based on overall progress)
        for i, node in enumerate(self.tool_nodes):
            phase_name = node["phase"]
            pulse_key = f"tool_{phase_name}"
            self.pulse_factors[pulse_key] = 0.97 + 0.03 * math.sin((value + i * 0.2 + 0.5) * 2 * math.pi)
            
    def update_animation(self):
        """Update animation state"""
        self.progress += 0.004 * self.animation_speed
        num_main_phases = len(self.main_phases)
        
        if self.progress >= 1.0:
            self.progress = 0.0
            old_phase_idx = self.current_main_phase_index
            self.current_main_phase_index = (self.current_main_phase_index + 1) % num_main_phases
            self.animationCompleted.emit() # Still emit for potential external use
            
            # Update node positions only if they haven't been initialized
            if not self.main_nodes:
                self.update_node_positions()

            # Add a ripple effect from the active main node when phase changes
            if self.main_nodes and self.current_main_phase_index < len(self.main_nodes):
                self.start_ripple_animation(self.current_main_phase_index, 'main')
            
            # Reset text animation
            self.text_animation_progress = 0.0
            self.text_fade = 0.7 
        
        # Update decorative elements
        self.update_decorative_elements()
        
        # Update ripple animations
        self.update_ripple_animations()
        
        # Update cube edge lights
        self.update_cube_lights()
        
        # Update link animation
        self.link_flow_offset = (self.link_flow_offset + 0.02) % 1.0
        
        # Update hover label animation
        self.update_hover_label_animation()
        
        # Update text animation
        self.text_animation_progress += 0.01
        if self.text_animation_progress >= 1.0:
            self.text_animation_progress = 0.0
            # Cycle through text options for the current main phase
            phase = self.main_phases[self.current_main_phase_index]
            texts = self.phase_texts.get(phase, [])
            if texts:
                self.current_text_index = (self.current_text_index + 1) % len(texts)
                self.text_fade = 0.7 
        
        # --- Update timer using QElapsedTimer ---
        elapsed_ms = self.elapsed_timer.elapsed()
        self.time_remaining = max(0, (self.load_time - elapsed_ms) / 1000)

        # Update text fade based on elapsed time (e.g., fade in over 200ms)
        # Use text_animation_progress to control fade-in for new text
        if self.text_animation_progress < 0.2: # Fade in over first 20% of text duration
            self.text_fade = 0.7 + (self.text_animation_progress / 0.2) * 0.3
        else:
            self.text_fade = 1.0
        
        # Update button animation phase (only for Go button now)
        if self.go_button.isVisible():
            self.go_button_animation_phase = (self.go_button_animation_phase + 0.02) % (2 * math.pi)
        
        # If app ready state hasn't been set but timer is low, consider app ready
        if not self.app_ready and self.time_remaining <= 0.1:
             self.set_app_ready(True)
             
        self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move events to detect hover"""
        pos = event.pos()
        self.hover_node = None
        self.hover_node_type = None
        found_hover = False

        # Check main nodes
        for i, node in enumerate(self.main_nodes):
            node_center = QPointF(node["x"], node["y"])
            distance = math.sqrt((pos.x() - node_center.x())**2 + (pos.y() - node_center.y())**2)
            if distance <= node["size"] * 1.2: # Slightly larger hitbox
                self.hover_node = i
                self.hover_node_type = 'main'
                found_hover = True
                break
        
        # Check tool nodes if no main node found
        if not found_hover:
            for i, node in enumerate(self.tool_nodes):
                node_center = QPointF(node["x"], node["y"])
                distance = math.sqrt((pos.x() - node_center.x())**2 + (pos.y() - node_center.y())**2)
                if distance <= node["size"] * 1.2:
                    self.hover_node = i
                    self.hover_node_type = 'tool'
                    found_hover = True
                    break

        # Update cursor
        if found_hover:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            QToolTip.hideText() # Don't show default tooltip
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            QToolTip.hideText()
        
        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events to detect clicks on nodes"""
        if event.button() == Qt.MouseButton.LeftButton and self.hover_node is not None:
            clicked_phase = None
            node_type = 'unknown'
            
            if self.hover_node_type == 'main' and self.hover_node < len(self.main_nodes):
                node = self.main_nodes[self.hover_node]
                clicked_phase = node["phase"]
                node_type = node.get("type", "main")
                
                # If we're in main view and this is a main workflow node, show detailed view 
                if self.current_view_mode == 'main' and not node.get("is_parent", False):
                    if clicked_phase in self.main_phases:
                        self.switch_to_detailed_view(clicked_phase)
                        return
                
                # If we're in detailed view and clicking the parent, go back to main view
                if self.current_view_mode != 'main' and node.get("is_parent", False):
                    self.switch_to_main_view()
                    return
                    
            elif self.hover_node_type == 'tool' and self.hover_node < len(self.tool_nodes):
                node = self.tool_nodes[self.hover_node]
                clicked_phase = node["phase"]
                node_type = "tool"
            
            # Process any node click that wasn't handled by special cases above
            if clicked_phase:
                print(f"DEBUG: Node clicked: {clicked_phase}, type: {node_type}")
                self.nodeClicked.emit(clicked_phase)
                # If app is ready, also trigger the Go button action
                if self.app_ready:
                    self.on_go_button_clicked()
        
        super().mousePressEvent(event) # Call base class method

    def resizeEvent(self, event):
        """Handle resize events to update node positions"""
        super().resizeEvent(event)
        self.update_gradients()
        self.update_node_positions()
        self.generate_decorative_elements()
        
        # Removed toggle button positioning
        
        # Position the Go button at the top right corner
        self.go_button.move(
            self.width() - self.go_button.width() - 40,
            30
        )
        
        self.update_button_style()
        
    def update_button_style(self):
        """Update the button styling based on current theme"""
        # Removed toggle button styling

        highlight_color = f"rgb({self.highlight_color.red()}, {self.highlight_color.green()}, {self.highlight_color.blue()})"
        
        # Style the Go button (using the style from set_app_ready directly)
        go_style = ""
        if self.is_dark_theme:
            go_style = """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                stop:0 #2ECC71, 
                                stop:1 #27AE60); /* Green gradient */
                    color: white;
                    border: 2px solid #2ECC71;
                    border-radius: 25px; /* More rounded */
                    padding: 10px 20px; /* Larger padding */
                    font-weight: bold;
                    font-size: 16px; /* Larger font */
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                stop:0 #27AE60, 
                                stop:1 #2ECC71); /* Slightly different hover gradient */
                    border: 3px solid #2ECC71; /* Thicker border on hover */
                }
                QPushButton:pressed {
                     background: #27AE60; /* Darker green when pressed */
                }
            """
        else: # Light theme
            go_style = """
                 QPushButton {
                     background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                 stop:0 #2ECC71, 
                                 stop:1 #27AE60); /* Same green gradient */
                     color: white;
                     border: 2px solid #2ECC71;
                     border-radius: 25px;
                     padding: 10px 20px;
                     font-weight: bold;
                     font-size: 16px;
                 }
                 QPushButton:hover {
                     background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                 stop:0 #27AE60, 
                                 stop:1 #2ECC71);
                     border: 3px solid #2ECC71;
                 }
                 QPushButton:pressed {
                      background: #27AE60; 
                 }
             """
        self.go_button.setStyleSheet(go_style)

    def update_node_positions(self):
        """Update the positions of nodes based on the widget size"""
        w, h = self.width(), self.height()
        
        self.main_nodes = []
        self.tool_nodes = []
        
        # Scale sizes based on window dimensions
        base_size_factor = min(w, h) / 1000 
        
        # --- Main Workflow Nodes ---
        num_main_nodes = len(self.main_phases)
        main_node_size = max(52, 52 * base_size_factor * 1.5) 
        link_width = max(4, 4 * base_size_factor * 1.2)
        center_y_main = h * 0.45 # Position main flow slightly above center
        
        self.current_link_width = link_width
        self.current_main_node_size = main_node_size
        
        # In main view, show all main nodes in a row
        if self.current_view_mode == 'main':
            for i in range(num_main_nodes):
                phase = self.main_phases[i]
                x_progress = (i / (num_main_nodes - 1)) if num_main_nodes > 1 else 0.5
                margin = max(100, w * 0.1)
                x = margin + (w - 2 * margin) * x_progress
                
                # Add slight elevation changes - small arc pattern
                mid_point = num_main_nodes / 2.0
                distance_from_mid = abs(i - mid_point) / mid_point if mid_point > 0 else 0
                y_offset = -25 * math.cos(distance_from_mid * math.pi) * base_size_factor * 1.5
                y = center_y_main + y_offset
                
                self.main_nodes.append({
                    "x": x, "y": y, "size": main_node_size, "phase": phase, "type": "main"
                })
                self.pulse_factors[f"main_{phase}"] = 1.0 # Initialize pulse factor
        else:
            # In detailed view, show only the selected main node and its sub-nodes
            selected_phase = self.current_view_mode
            
            # Display the parent node at the top center
            parent_x = w / 2
            parent_y = h * 0.3
            
            self.main_nodes.append({
                "x": parent_x, "y": parent_y, "size": main_node_size * 1.2,  
                "phase": selected_phase, "type": "main", "is_parent": True
            })
            self.pulse_factors[f"main_{selected_phase}"] = 1.0
            
            # Create a mapping of main phases to their sub-sections
            # Remove the parent node from the sub-sections list (first item)
            sub_sections = {
                "plan": ["plan", "hypotheses"],
                "manage": ["design", "documentation", "participants", "protocol"],
                "literature": ["search", "ranking", "claims"],
                "data": ["sources", "cleaning", "reshaping", "filtering", "joining"],
                "analysis": ["evaluation", "assumptions", "subgroup", "mediation", "sensitivity", "interpret"],
                "evidence": ["evidence"]
            }
            
            # Display sub-nodes in a row below the parent
            sub_nodes = sub_sections.get(selected_phase, [])
            if sub_nodes:
                num_sub_nodes = len(sub_nodes)
                sub_node_size = main_node_size * 0.8
                
                for i, sub_phase in enumerate(sub_nodes):
                    x_progress = (i / (num_sub_nodes - 1)) if num_sub_nodes > 1 else 0.5
                    margin = max(100, w * 0.15)
                    x = margin + (w - 2 * margin) * x_progress
                    y = h * 0.6  # Position sub-nodes below parent
                    
                    self.main_nodes.append({
                        "x": x, "y": y, "size": sub_node_size, 
                        "phase": sub_phase, "type": "sub", "parent": selected_phase
                    })
                    self.pulse_factors[f"sub_{sub_phase}"] = 1.0

        # Create connections between main nodes
        self.connections = []
        if self.current_view_mode == 'main':
            # In main view, connect sequential main nodes
            for i in range(num_main_nodes - 1):
                self.connections.append({"start": i, "end": i + 1})
        else:
            # In detailed view, connect parent to all sub-nodes
            for i in range(1, len(self.main_nodes)):
                self.connections.append({"start": 0, "end": i})
            
        # --- Secondary Tool Nodes ---
        num_tool_nodes = len(self.tool_phases)
        tool_node_size = max(36, 36 * base_size_factor * 1.3) # Smaller size
        center_y_tools = h * 0.75 # Position tools lower down
        self.current_tool_node_size = tool_node_size

        tool_margin = max(150, w * 0.2) # Wider margin for tools row
        
        for i in range(num_tool_nodes):
            phase = self.tool_phases[i]
            x_progress = (i / (num_tool_nodes - 1)) if num_tool_nodes > 1 else 0.5
            x = tool_margin + (w - 2 * tool_margin) * x_progress
            y = center_y_tools # Flat line for tools
            
            self.tool_nodes.append({
                "x": x, "y": y, "size": tool_node_size, "phase": phase, "type": "tool"
            })
            self.pulse_factors[f"tool_{phase}"] = 1.0 # Initialize pulse factor

    def get_current_text(self):
        """Get the text for the current main phase"""
        if self.current_view_mode == 'main':
            phase = self.main_phases[self.current_main_phase_index]
        else:
            # In detailed view, use the selected phase
            phase = self.current_view_mode
            
        texts = self.phase_texts.get(phase, ["Loading Application..."])
        
        # Select text based on animation progress within the phase
        text_idx = min(int(self.progress * len(texts)), len(texts) - 1)
        return texts[text_idx]

    def paintEvent(self, event):
        """Paint the animated infographic"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        
        painter.fillRect(self.rect(), self.bg_gradient)
        
        self.draw_decorative_elements(painter)
        
        if not self.main_nodes: # Check if nodes need initializing
            self.update_node_positions()
            
        # Draw connections (only for main nodes)
        self.draw_connections(painter)
        
        # Draw ripple animations
        self.draw_ripple_animations(painter)
        
        # Draw nodes (both main and tool)
        self.draw_nodes(painter)
        
        # Draw current phase text (simplified)
        self.draw_phase_text(painter)
        
        if self.go_button.isVisible():
            self.draw_go_button_glow(painter)

    # draw_decorative_elements remains the same
    def draw_decorative_elements(self, painter):
        """Draw decorative background elements"""
        for element in self.decorative_elements:
            x = element["x"] + element.get("x_offset", 0)
            y = element["y"] + element.get("y_offset", 0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(element["color"]))
            painter.drawEllipse(QRectF(x - element["size"]/2, y - element["size"]/2, element["size"], element["size"]))

    def draw_connections(self, painter):
        """Draw connections between main nodes only"""
        if not self.main_nodes: return

        for conn in self.connections:
            # Ensure indices are valid for main_nodes
            if conn["start"] >= len(self.main_nodes) or conn["end"] >= len(self.main_nodes):
                continue
                
            start_node = self.main_nodes[conn["start"]]
            end_node = self.main_nodes[conn["end"]]
            
            grad = QLinearGradient(start_node["x"], start_node["y"], end_node["x"], end_node["y"])
            if self.is_dark_theme:
                grad.setColorAt(0, QColor(70, 70, 100, 180))
                grad.setColorAt(0.5, QColor(90, 90, 120, 150))
                grad.setColorAt(1, QColor(50, 50, 80, 180))
            else:
                grad.setColorAt(0, QColor(180, 180, 220, 180))
                grad.setColorAt(0.5, QColor(200, 200, 240, 150))
                grad.setColorAt(1, QColor(160, 160, 200, 180))
                
            path = QPainterPath()
            path.moveTo(start_node["x"], start_node["y"])
            dx = end_node["x"] - start_node["x"]
            dy = end_node["y"] - start_node["y"]
            ctrl1_x = start_node["x"] + dx * 0.4
            ctrl1_y = start_node["y"] + dy * 0.1
            ctrl2_x = start_node["x"] + dx * 0.6
            ctrl2_y = start_node["y"] + dy * 0.9
            path.cubicTo(ctrl1_x, ctrl1_y, ctrl2_x, ctrl2_y, end_node["x"], end_node["y"])
            
            pen = QPen(QBrush(grad), self.current_link_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            
            # Draw animated flowing pattern if connection involves the active main phase
            is_conn_active = (conn["start"] == self.current_main_phase_index or conn["end"] == self.current_main_phase_index)
            if is_conn_active:
                glow_color = QColor(self.highlight_color) # Use main highlight
                glow_color.setAlpha(150)
                
                flow_pen = QPen(glow_color, self.current_link_width * 0.7)
                flow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                flow_pen.setStyle(Qt.PenStyle.DotLine)
                flow_pen.setDashOffset(self.link_flow_offset * 20)
                painter.setPen(flow_pen)
                painter.drawPath(path)
                
                # Draw flowing particles
                path_length = path.length()
                num_particles = int(path_length / 20)
                for j in range(num_particles):
                    pos_percent = (j / num_particles + self.link_flow_offset) % 1.0
                    path_point = path.pointAtPercent(pos_percent)
                    particle_size = 3
                    glow_grad = QRadialGradient(path_point, particle_size * 2)
                    glow_grad.setColorAt(0, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 180))
                    glow_grad.setColorAt(1, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 0))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(glow_grad))
                    painter.drawEllipse(path_point, particle_size, particle_size)

    def draw_ripple_animations(self, painter):
        """Draw ripple animations around nodes"""
        for ripple_id, ripple in self.ripple_animations.items():
            center = QPointF(ripple["center_x"], ripple["center_y"])
            radius = ripple["current_radius"]
            grad = QRadialGradient(center, radius)
            ripple_color = QColor(ripple["color"])
            ripple_color.setAlphaF(ripple["opacity"] * 0.5) 
            transparent_color = QColor(ripple_color)
            transparent_color.setAlphaF(0.0)
            grad.setColorAt(0.7, transparent_color) 
            grad.setColorAt(0.8, ripple_color)      
            grad.setColorAt(1.0, transparent_color) 
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(center, radius, radius)

    def start_ripple_animation(self, node_idx, node_type):
        """Start a ripple animation from the specified node (main or tool)"""
        node_list = self.main_nodes if node_type == 'main' else self.tool_nodes
        if node_idx >= len(node_list):
            return
            
        node = node_list[node_idx]
        ripple_id = f"ripple_{node_type}_{node_idx}_{len(self.ripple_animations)}"
        
        is_active = (node_type == 'main' and node_idx == self.current_main_phase_index)
        ripple_color = self.highlight_color if node_type == 'main' else self.tool_highlight_color
        
        ripple = {
            "center_x": node["x"], "center_y": node["y"],
            "start_radius": node["size"] * 0.8,
            "current_radius": node["size"] * 0.8,
            "max_radius": node["size"] * 4.0,
            "progress": 0.0, "opacity": 0.7,
            "color": ripple_color if is_active else QColor(150, 150, 150, 100)
        }
        self.ripple_animations[ripple_id] = ripple

    # update_ripple_animations remains the same
    def update_ripple_animations(self):
        """Update all active ripple animations"""
        ripples_to_remove = []
        for ripple_id, ripple in self.ripple_animations.items():
            ripple["progress"] += 0.02
            if ripple["progress"] >= 1.0:
                ripples_to_remove.append(ripple_id)
                continue
            t = ripple["progress"]
            eased_t = 1 - (1 - t) * (1 - t) # Ease out quad
            radius_range = ripple["max_radius"] - ripple["start_radius"]
            ripple["current_radius"] = ripple["start_radius"] + radius_range * eased_t
            ripple["opacity"] = 0.7 * (1 - eased_t)
        for ripple_id in ripples_to_remove:
            if ripple_id in self.ripple_animations:
                del self.ripple_animations[ripple_id]

    # update_cube_lights remains the same
    def update_cube_lights(self):
        """Update the lighting animation for cube edges"""
        current_light = self.cube_edge_lights[self.current_edge]
        current_light += 0.05 * self.edge_light_direction
        if current_light >= 1.0:
            current_light = 1.0
            self.edge_light_direction = -1
        elif current_light <= 0.0:
            current_light = 0.0
            self.edge_light_direction = 1
            self.current_edge = (self.current_edge + 1) % 12
        self.cube_edge_lights[self.current_edge] = current_light
        for edge in range(12):
            if edge != self.current_edge:
                self.cube_edge_lights[edge] *= 0.95 

    def update_hover_label_animation(self):
        """Update the hover label animation state"""
        if self.hover_node is not None:
            current_key = f"{self.hover_node_type}_{self.hover_node}"
            if "key" not in self.hover_label_state or self.hover_label_state["key"] != current_key:
                # Reset animation for new node
                self.hover_label_state["key"] = current_key
                self.hover_label_state["node_index"] = self.hover_node
                self.hover_label_state["node_type"] = self.hover_node_type
                
                self.hover_label_state["blinks"] = 0
                self.hover_label_state["opacity"] = 0.0
                self.hover_label_state["settled"] = False
                
                # Set position near the node
                node_list = self.main_nodes if self.hover_node_type == 'main' else self.tool_nodes
                if self.hover_node < len(node_list):
                    node = node_list[self.hover_node]
                    self.hover_label_state["pos"] = QPointF(node["x"], node["y"] + node["size"] * 1.6)
            else:
                if not self.hover_label_state["settled"]:
                    self.hover_label_state["opacity"] = min(1.0, self.hover_label_state["opacity"] + 0.1)
                    self.hover_label_state["blinks"] += 0.1
                    if self.hover_label_state["blinks"] >= 3:
                        self.hover_label_state["settled"] = True
                        self.hover_label_state["opacity"] = 1.0
        else:
            # Reset when no node is hovered
            self.hover_label_state["key"] = None
            self.hover_label_state["node_index"] = None
            self.hover_label_state["node_type"] = None
            self.hover_label_state["opacity"] = 0.0
            self.hover_label_state["settled"] = False

    def draw_phase_text(self, painter):
        """Draw simplified text area at the bottom"""
        # Skip drawing bottom text when in menu mode
        if self.menu_mode:
            return
            
        w, h = self.width(), self.height()
        base_size_factor = min(w, h) / 1000 
        
        text_rect = QRectF(20, h - 80, w - 40, 60) # Slightly smaller height
        bg_color = QColor(0, 0, 0, 140) if self.is_dark_theme else QColor(255, 255, 255, 200)
        
        painter.setPen(QPen(QColor(180, 180, 180, 60), 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(text_rect, 15, 15)
        
        # Display a generic loading text or current main phase name
        current_phase_name = self.phase_descriptions.get(self.main_phases[self.current_main_phase_index], "Loading...")
        # display_text = f"Loading: {current_phase_name}" 
        display_text = self.get_current_text() # Show cycling text for current phase

        painter.setPen(QColor(255, 255, 255) if self.is_dark_theme else QColor(50, 50, 60))
        font_size = max(15, int(15 * base_size_factor * 1.2))
        font = QFont("Arial", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Use fade effect
        text_opacity = max(0.7, self.text_fade)
        current_color = painter.pen().color()
        current_color.setAlphaF(text_opacity)
        painter.setPen(current_color)
        
        if self.is_dark_theme:
            shadow_color = QColor(0, 0, 0, int(100 * text_opacity))
            painter.setPen(shadow_color)
            shadow_rect = QRectF(text_rect)
            shadow_rect.translate(2, 2)
            painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignCenter, display_text)
            painter.setPen(current_color) # Restore main text color
        
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, display_text)
        
        # Keep Title drawing
        title_rect = QRectF(80, 15, w - 160, 50) 
        if self.is_dark_theme:
            shadow_color = QColor(0, 0, 0, 100)
            painter.setPen(shadow_color)
            shadow_rect = QRectF(title_rect); shadow_rect.translate(2, 2)
            title_font_size = max(20, int(20 * base_size_factor * 1.2))
            title_font = QFont("Arial", title_font_size, QFont.Weight.Bold)
            painter.setFont(title_font)
            painter.drawText(shadow_rect, Qt.AlignmentFlag.AlignCenter, "CareFrame")
        
        title_font_size = max(24, int(24 * base_size_factor * 1.2))
        title_font = QFont("Arial", title_font_size, QFont.Weight.Bold)
        painter.setFont(title_font)
        title_color = QColor(self.highlight_color)
        pulse = 0.8 + 0.2 * math.sin(self.progress * math.pi * 4)
        title_color.setRed(int(title_color.red() * pulse))
        title_color.setGreen(int(title_color.green() * pulse))
        title_color.setBlue(int(title_color.blue() * pulse))
        painter.setPen(title_color)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "CareFrame")
        
        # Keep Subtitle drawing
        subtitle_rect = QRectF(80, 60, w - 160, 30)
        subtitle_font_size = max(12, int(12 * base_size_factor * 1.2))
        subtitle_font = QFont("Arial", subtitle_font_size)
        painter.setFont(subtitle_font)
        subtitle_color = QColor(200, 200, 220) if self.is_dark_theme else QColor(80, 80, 100)
        painter.setPen(subtitle_color)
        full_subtitle = "Clinical Research Acceleration Platform" # Simplified subtitle
        # Ensure typing animation completes by 70% of the overall animation
        typing_progress = min(1.0, self.progress / 0.7)
        char_count = int(len(full_subtitle) * typing_progress)
        visible_subtitle = full_subtitle[:char_count]
        cursor_visible = int(self.progress * 8) % 2 == 0
        if cursor_visible and char_count < len(full_subtitle): visible_subtitle += "|"
        painter.drawText(subtitle_rect, Qt.AlignmentFlag.AlignCenter, visible_subtitle)
        
        # Keep Timer drawing
        self.draw_countdown_timer(painter)

    # draw_countdown_timer remains the same
    def draw_countdown_timer(self, painter):
        """Draw a countdown timer showing time remaining"""
        if self.go_button.isVisible() or self.menu_mode:
            return
        minutes = int(self.time_remaining) // 60
        seconds = int(self.time_remaining) % 60
        time_text = f"{minutes:02d}:{seconds:02d}"
        timer_rect = QRectF(self.width() - 120, 20, 100, 40)
        bg_gradient = QLinearGradient(timer_rect.topLeft(), timer_rect.bottomRight())
        if self.is_dark_theme:
            bg_gradient.setColorAt(0, QColor(60, 60, 80, 180)); bg_gradient.setColorAt(1, QColor(40, 40, 60, 180))
            timer_color = QColor(230, 230, 250)
        else:
            bg_gradient.setColorAt(0, QColor(240, 240, 255, 220)); bg_gradient.setColorAt(1, QColor(220, 220, 240, 220))
            timer_color = QColor(60, 60, 90)
            
        painter.setPen(QPen(QColor(180, 180, 180, 60), 1))
        painter.setBrush(QBrush(bg_gradient))
        painter.drawRoundedRect(timer_rect, 8, 8)
        
        completion_ratio = 1 - (self.time_remaining / (self.load_time / 1000)) if self.load_time > 0 else 1
        red_level = min(255, timer_color.red() + int(completion_ratio * (255 - timer_color.red())))
        green_level = max(50, timer_color.green() - int(completion_ratio * (timer_color.green() - 50)))
        blue_level = max(50, timer_color.blue() - int(completion_ratio * (timer_color.blue() - 50)))
        timer_color = QColor(red_level, green_level, blue_level)
        
        painter.setPen(timer_color)
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(timer_rect, Qt.AlignmentFlag.AlignCenter, time_text)
        
        loading_rect = QRectF(timer_rect.left(), timer_rect.bottom() + 5, timer_rect.width(), 20)
        dot_count = (int(self.progress * 10) % 4); dots = "." * dot_count
        loading_text = f"Loading{dots}"
        painter.setPen(timer_color)
        font = QFont("Arial", 10); painter.setFont(font)
        painter.drawText(loading_rect, Qt.AlignmentFlag.AlignCenter, loading_text)

    def draw_nodes(self, painter):
        """Draw nodes representing research stages and tools"""
        w, h = self.width(), self.height()
        base_size_factor = min(w, h) / 1000 
        
        # Combine nodes for easier iteration, but track type
        all_nodes = [dict(n, index=i, type='main') for i, n in enumerate(self.main_nodes)] + \
                    [dict(n, index=i, type='tool') for i, n in enumerate(self.tool_nodes)]

        for node_info in all_nodes:
            node_index = node_info["index"]
            node_type = node_info["type"]
            node = node_info # The actual node dictionary
            
            is_main_node = (node_type == 'main')
            is_active = is_main_node and (node_index == self.current_main_phase_index)
            is_hovered = (node_index == self.hover_node and node_type == self.hover_node_type)
            is_sub_node = node.get("type") == "sub"
            
            center = QPointF(node["x"], node["y"])
            phase = node["phase"]
            
            # For debugging
            if is_sub_node:
                print(f"DEBUG: Drawing sub-node: {phase}, has description: {phase in self.phase_descriptions}")
            
            # Determine pulse factors
            pulse_key = None
            if is_main_node and not is_sub_node:
                pulse_key = f"main_{phase}"
            elif is_sub_node:
                pulse_key = f"sub_{phase}"
            else:
                pulse_key = f"tool_{phase}"
                
            pulse_factor = self.pulse_factors.get(pulse_key, 1.0)
            effective_size = node["size"] * pulse_factor
            
            # --- Draw Glow ---
            if is_active or is_hovered:
                base_glow_color = self.glow_color if is_main_node else self.tool_glow_color
                highlight_glow_color = self.highlight_color if is_main_node else self.tool_highlight_color
                hover_glow_color = self.secondary_highlight if is_main_node else self.tool_highlight_color # Use tool highlight for hover too

                for glow_layer in range(3):
                    glow_size = effective_size * (1.5 + glow_layer * 0.4) 
                    glow_opacity = 0.6 - glow_layer * 0.2                
                    glow_grad = QRadialGradient(center, glow_size)
                    
                    current_glow_color = base_glow_color
                    if is_active:
                        current_glow_color = QColor(highlight_glow_color)
                        pulse_offset = math.sin(self.progress * math.pi * 4) * 0.2
                        glow_opacity = max(0.0, min(0.8, glow_opacity + pulse_offset))
                    elif is_hovered:
                        current_glow_color = QColor(hover_glow_color)
                    
                    inner_alpha = int(glow_opacity * 80)
                    current_glow_color.setAlpha(inner_alpha)
                    glow_grad.setColorAt(0.0, current_glow_color)
                    middle_color = QColor(current_glow_color); middle_color.setAlpha(int(inner_alpha * 0.7))
                    glow_grad.setColorAt(0.7, middle_color)
                    outer_color = QColor(current_glow_color); outer_color.setAlpha(0)
                    glow_grad.setColorAt(1.0, outer_color)
                    
                    painter.setBrush(QBrush(glow_grad))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(center, glow_size, glow_size)
                
                # Inner glow
                inner_glow_size = effective_size * 1.05
                inner_glow = QRadialGradient(center, inner_glow_size)
                if is_active: inner_color = QColor(200, 180, 255, 100) if is_main_node else QColor(180, 255, 200, 100) # Purple/Green
                else: inner_color = QColor(180, 210, 255, 80) if is_main_node else QColor(180, 230, 210, 80) # Blue/Teal
                inner_glow.setColorAt(0.5, QColor(inner_color.red(), inner_color.green(), inner_color.blue(), 0))
                inner_glow.setColorAt(0.8, inner_color)
                inner_glow.setColorAt(1.0, QColor(inner_color.red(), inner_color.green(), inner_color.blue(), 0))
                painter.setBrush(QBrush(inner_glow))
                painter.drawEllipse(center, inner_glow_size, inner_glow_size)

            # --- Select Cube Colors ---
            if is_active: face_colors = self.main_active_colors
            elif is_hovered: face_colors = self.main_hover_colors if is_main_node else self.tool_hover_colors
            else: face_colors = self.main_node_colors if is_main_node else self.tool_node_colors
            
            # Enhance edge lighting for active main node
            node_edge_lights = self.cube_edge_lights.copy()
            if is_active:
                 for edge in range(12): node_edge_lights[edge] = max(0.2, node_edge_lights[edge] * 1.5)
            
            # --- Draw Cube (No Rotation) ---
            self.draw_cube(
                painter, center, effective_size, 
                0, # Use fixed angle 0 - REMOVED SWAYING
                face_colors, node_edge_lights
            )
            
            # --- Draw Label and Icon ---
            # Always show a label for all nodes
            if phase in self.phase_descriptions and not is_hovered:
                # Get the label to show - check for special sub-node labels first
                if is_sub_node and hasattr(self, 'sub_node_labels') and phase in self.sub_node_labels:
                    label_to_show = self.sub_node_labels[phase]
                else:
                    # Use regular phase descriptions
                    label_to_show = self.phase_descriptions.get(phase, phase.title())
                
                label_color = QColor(255, 255, 255) if is_active else self.text_color
                painter.setPen(label_color)
                
                # Adjust font size based on node type
                label_font_size = max(10, int(10 * base_size_factor * (1.2 if is_main_node else 1.0)))
                font = QFont("Arial", label_font_size, QFont.Weight.Bold if is_active else QFont.Weight.Medium)
                painter.setFont(font)
                
                metrics = QFontMetrics(font)
                text_width = metrics.horizontalAdvance(label_to_show)
                text_x = node["x"] - text_width / 2
                text_y = node["y"] + node["size"] * 1.4 
                painter.drawText(QPointF(text_x, text_y), label_to_show)
                
            if phase in self.phase_icons:
                icon_text = "" # Use actual icon if available
                icon_name = self.phase_icons[phase]
                icon_color = QColor(255, 255, 255) if is_active else self.text_color.lighter(110 if is_main_node else 100)
                
                # For demo, draw icon text if no icon loaded
                if not icon_text: 
                    # Use first letter as fallback text
                    icon_text = self.phase_descriptions.get(phase, "?")[0].upper()

                    painter.setPen(icon_color)
                    icon_font_size = max(16, int(16 * base_size_factor * (1.2 if is_main_node else 1.0)))
                    icon_font = QFont("Arial", icon_font_size, QFont.Weight.Bold)
                    painter.setFont(icon_font)
                    
                    icon_center = QRectF(
                        node["x"] - effective_size/3, node["y"] - effective_size/3,
                        effective_size*2/3, effective_size*2/3
                    )
                    painter.drawText(icon_center, Qt.AlignmentFlag.AlignCenter, icon_text)
                # else: # Logic to draw loaded QPixmap icon would go here

        # --- Draw Hover Label ---
        if self.hover_node is not None and self.hover_label_state["opacity"] > 0.01:
            hover_node_index = self.hover_label_state["node_index"]
            hover_node_type = self.hover_label_state["node_type"]
            
            node_list = self.main_nodes if hover_node_type == 'main' else self.tool_nodes
            if hover_node_index < len(node_list):
                node = node_list[hover_node_index]
                phase = node["phase"]
                
                if phase:
                    opacity = self.hover_label_state["opacity"]
                    label_pos = self.hover_label_state["pos"]
                    
                    self.draw_hover_label(painter, hover_node_index, hover_node_type, phase, label_pos, opacity)

    def draw_hover_label(self, painter, node_index, node_type, phase, label_pos, opacity):
        """Draw hover tooltip with detailed information"""
        node_list = self.main_nodes if node_type == 'main' else self.tool_nodes
        is_sub_node = False
        if node_index < len(node_list):
            node = node_list[node_index]
            is_sub_node = node.get("type") == "sub"
            
        # Special case for plan sub-node
        detailed_key = phase
        if is_sub_node and phase == "plan" and hasattr(self, 'detailed_descriptions') and "plan_sub" in self.detailed_descriptions:
            detailed_key = "plan_sub"
        
        if hasattr(self, 'detailed_descriptions') and detailed_key in self.detailed_descriptions:
            # Use detailed descriptions if available
            details = self.detailed_descriptions[detailed_key].copy()
            
            # For special sub-nodes, update the title
            if is_sub_node and hasattr(self, 'sub_node_labels') and phase in self.sub_node_labels:
                details[0] = self.sub_node_labels[phase]
        else:
            # Fallback to regular phase descriptions and texts
            description = self.phase_descriptions.get(phase, phase.title())
            details = [description] + self.phase_texts.get(phase, [])
        
        if not details:
            return
            
        bg_color = QColor(30, 30, 40, int(180 * opacity)) if self.is_dark_theme else QColor(255, 255, 255, int(220 * opacity))
        text_color = QColor(255, 255, 255, int(255 * opacity)) if self.is_dark_theme else QColor(50, 50, 60, int(255 * opacity))
        
        base_size_factor = min(self.width(), self.height()) / 1000
        hover_font_size = max(12, int(12 * base_size_factor * 1.2))
        font = QFont("Arial", hover_font_size, QFont.Weight.Bold)
        metrics = QFontMetrics(font)
        
        # Get max width of the text
        max_width = 0
        for text in details:
            text_width = metrics.horizontalAdvance(text)
            max_width = max(max_width, text_width)
            
        text_height = len(details) * metrics.height() + 16
        
        bg_rect = QRectF(label_pos.x() - max_width/2 - 10, label_pos.y() - 10, max_width + 20, text_height)
        
        # Draw background
        painter.setPen(QPen(QColor(100, 100, 120, int(80 * opacity)), 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(bg_rect, 8, 8)
        
        # Draw title in bold
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        
        title_y = label_pos.y() + metrics.height()
        title_text = details[0]
        title_width = metrics.horizontalAdvance(title_text)
        painter.drawText(QPointF(label_pos.x() - title_width/2, title_y), title_text)
        
        # Draw details in regular font
        detail_font_size = max(10, int(10 * base_size_factor * 1.2))
        detail_font = QFont("Arial", detail_font_size)
        painter.setFont(detail_font)
        detail_metrics = QFontMetrics(detail_font)
        
        for i, detail in enumerate(details[1:], 1):  # Skip the first item (title)
            detail_width = detail_metrics.horizontalAdvance(detail)
            detail_y = title_y + i * metrics.height()
            painter.drawText(QPointF(label_pos.x() - detail_width/2, detail_y), detail)
        
        # Draw blinking highlight effect if not settled
        if not self.hover_label_state["settled"]:
            highlight_color = self.highlight_color if node_type == 'main' else self.tool_highlight_color
            pulse_opacity = math.sin(self.progress * math.pi * 20) * 0.3 + 0.7
            glow_color = QColor(highlight_color)
            glow_color.setAlpha(int(40 * opacity * pulse_opacity))
            pulse_pen = QPen(glow_color, 2)
            painter.setPen(pulse_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(bg_rect, 8, 8)

    # draw_cube remains the same
    def draw_cube(self, painter, center, size, angle, colors, edge_lights):
        """Draw a 3D cube with specified colors and edge lighting"""
        points = self.create_cube_points(center, size, angle)
        faces = [
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), 
            (1, 5, 6, 2), (3, 7, 6, 2), (0, 4, 5, 1)
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        visible_faces = [0, 3, 4] # Front, Right, Top
        
        for face_idx in visible_faces:
            face = faces[face_idx]
            path = QPainterPath(); path.moveTo(points[face[0]])
            for i in range(1, len(face)): path.lineTo(points[face[i]])
            path.closeSubpath()
            
            face_color = colors.get(f"face_{face_idx}", colors.get("default_face", QColor(100, 100, 100)))
            if face_idx == 0: # Front
                grad = QLinearGradient(points[0], points[2]); grad.setColorAt(0, face_color.lighter(120)); grad.setColorAt(1, face_color)
            elif face_idx == 3: # Right
                grad = QLinearGradient(points[1], points[6]); grad.setColorAt(0, face_color); grad.setColorAt(1, face_color.darker(110))
            elif face_idx == 4: # Top
                grad = QLinearGradient(points[3], points[6]); grad.setColorAt(0, face_color.lighter(130)); grad.setColorAt(1, face_color.lighter(110))
            else:
                grad = QLinearGradient(center, QPointF(center.x() + size, center.y() + size)); grad.setColorAt(0, face_color); grad.setColorAt(1, face_color.darker(120))
            
            painter.setBrush(QBrush(grad))
            painter.setPen(QPen(colors.get("edge", QColor(60, 60, 60)), 1.0))
            painter.drawPath(path)
            
        for edge_idx, (start_idx, end_idx) in enumerate(edges):
            light_intensity = edge_lights.get(edge_idx, 0.0)
            if light_intensity > 0.01:
                edge_color = colors.get("highlight", QColor(200, 100, 255))
                glow_color = QColor(edge_color.red(), edge_color.green(), edge_color.blue(), int(255 * light_intensity))
                glow_pen = QPen(glow_color, 2.5); glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(glow_pen); painter.drawLine(points[start_idx], points[end_idx])
                if light_intensity > 0.5:
                    inner_color = QColor(min(255, edge_color.red() + 50), min(255, edge_color.green() + 50), min(255, edge_color.blue() + 50), int(180 * light_intensity))
                    inner_pen = QPen(inner_color, 1.2); inner_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(inner_pen); painter.drawLine(points[start_idx], points[end_idx])

    # create_cube_points remains the same
    def create_cube_points(self, center, size, angle=0):
        """Create points for a 3D cube with perspective"""
        half_size = size * 0.7
        sin_angle = math.sin(math.radians(angle)) # Convert angle to radians
        cos_angle = math.cos(math.radians(angle))
        points_3d = [
            (-half_size, -half_size, half_size), (half_size, -half_size, half_size),
            (half_size, half_size, half_size), (-half_size, half_size, half_size),
            (-half_size, -half_size, -half_size), (half_size, -half_size, -half_size),
            (half_size, half_size, -half_size), (-half_size, half_size, -half_size)
        ]
        points_2d = []
        for x, y, z in points_3d:
            rotated_x = x * cos_angle - z * sin_angle
            rotated_z = x * sin_angle + z * cos_angle
            proj_x = center.x() + rotated_x - rotated_z * 0.3 # Isometric-like projection
            proj_y = center.y() + y * 0.8 - rotated_z * 0.4 # Adjust Y perspective slightly
            points_2d.append(QPointF(proj_x, proj_y))
        return points_2d

    # draw_button_glow removed (was for toggle button)
    
    # draw_go_button_glow remains the same
    def draw_go_button_glow(self, painter):
        """Draw a simplified glow around the Go button"""
        glow_rect = QRectF(self.go_button.x() - 5, self.go_button.y() - 5, self.go_button.width() + 10, self.go_button.height() + 10)
        glow_color = QColor(46, 204, 113, 60) # Greenish glow
        pen = QPen(glow_color, 2); pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen); painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(glow_rect, 25, 25) # Match button radius

    def on_go_button_clicked(self):
        """Handle Go button click"""
        self.goButtonClicked.emit()

    def set_app_ready(self, ready=True):
        """Set whether the app is ready to be used"""
        self.app_ready = ready
        if ready and not self.menu_mode:
            self.go_button.show()
            self.update_button_style() # Apply the final GO button style
            self.time_remaining = 0 # Stop countdown

    # animate_go_button removed (now handled by style setting)

    def switch_to_detailed_view(self, main_phase):
        """Switch to detailed view showing sub-nodes of the selected main phase"""
        if main_phase in self.main_phases:
            self.previous_view_mode = self.current_view_mode
            self.current_view_mode = main_phase
            self.update_node_positions()
            self.update()
            
    def switch_to_main_view(self):
        """Switch back to main workflow view"""
        self.previous_view_mode = self.current_view_mode
        self.current_view_mode = 'main'
        self.update_node_positions()
        self.update()

    # Updated section description texts for more detailed hover information
    def initialize_detailed_descriptions(self):
        """Initialize more detailed descriptions for hover tooltips"""
        self.detailed_descriptions = {
            # Main workflows
            "plan": [
                "Research Strategy & Planning", 
                "Define goals, research questions and hypotheses",
                "Plan your overall research strategy"
            ],
            "hypotheses": [
                "Hypotheses Management", 
                "Create and manage research hypotheses",
                "Link hypotheses to evidence and data"
            ],
            # Special case for 'plan' sub-node (different from main plan node)
            "plan_sub": [
                "Planning",  # This will be overridden if it's a sub-node
                "Design your research plan",
                "Develop strategies and approaches"
            ],
            "manage": [
                "Study Management", 
                "Manage all aspects of study execution",
                "Coordinate participants, protocols and documentation"
            ],
            "design": [
                "Study Design", 
                "Define study methodology and approach",
                "Specify experimental design parameters"
            ],
            "documentation": [
                "Study Documentation", 
                "Manage all study-related documents",
                "Create and organize research documentation"
            ],
            "participants": [
                "Participant Management", 
                "Recruit and manage study participants",
                "Track participant data and engagement"
            ],
            "protocol": [
                "Protocol Management", 
                "Define and manage research protocols",
                "Create standardized procedures"
            ],
            "literature": [
                "Literature Review", 
                "Search and explore scientific literature",
                "Find relevant papers for your research"
            ],
            "search": [
                "Literature Search",
                "Find scientific articles and papers",
                "Search across multiple literature databases"
            ],
            "ranking": [
                "Literature Ranking", 
                "Evaluate and rank collected papers",
                "Prioritize most relevant research"
            ],
            "claims": [
                "Literature Claims", 
                "Extract evidence from ranked literature",
                "Connect findings to hypotheses"
            ],
            "data": [
                "Data Collection", 
                "Gather data from various sources",
                "Import and organize research data"
            ],
            "sources": [
                "Data Sources",
                "Connect to and import data",
                "Manage data source connections"
            ],
            "cleaning": [
                "Data Cleaning", 
                "Detect and fix data quality issues",
                "Remove outliers and invalid entries"
            ],
            "reshaping": [
                "Data Reshaping", 
                "Transform data structure for analysis",
                "Pivot, transpose, or restructure datasets"
            ],
            "filtering": [
                "Data Filtering", 
                "Create subsets and filter data",
                "Apply criteria to focus analysis"
            ],
            "joining": [
                "Data Joining", 
                "Combine multiple datasets",
                "Merge related data sources"
            ],
            "analysis": [
                "Statistical Analysis", 
                "Analyze data using statistical methods",
                "Run tests and build models"
            ],
            "evaluation": [
                "Test Evaluation", 
                "Evaluate analysis results",
                "Interpret statistical findings"
            ],
            "assumptions": [
                "Statistical Assumptions", 
                "Verify test assumptions",
                "Ensure statistical validity"
            ],
            "subgroup": [
                "Subgroup Analysis", 
                "Analyze differences between groups",
                "Identify population-specific effects"
            ],
            "mediation": [
                "Mediation Analysis", 
                "Analyze causal pathways",
                "Identify mediating variables"
            ],
            "sensitivity": [
                "Sensitivity Analysis", 
                "Test result robustness",
                "Analyze how variations affect outcomes"
            ],
            "interpret": [
                "Results Interpretation",
                "Understand analysis results",
                "Generate insights and explanations"
            ],
            "evidence": [
                "Evidence & Validation", 
                "Validate research findings",
                "Record evidence on blockchain"
            ],
            # Tools
            "agent": [
                "AI Assistant", 
                "Interact with AI research assistant",
                "Get help with analysis and writing"
            ],
            "network": [
                "Network & Collaboration", 
                "Connect with other researchers",
                "Share data and collaborate"
            ],
            "database": [
                "Database Administration", 
                "Manage database connections",
                "Configure data storage"
            ],
            "team": [
                "Team Management", 
                "Manage research team",
                "Assign roles and permissions"
            ],
            "settings": [
                "Application Settings", 
                "Configure CareFrame settings",
                "Customize interface and preferences"
            ]
        }

# Demo function updated slightly if needed (seems fine)
def show_demo(auto_close=False, duration=12000):
    app = QApplication(sys.argv)
    dialog = QDialog()
    dialog.setWindowTitle("CareFrame Loader")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    
    loader = ResearchFlowLoader(load_time=duration)
    loader.set_theme(True)
    # Initialize detailed descriptions
    if hasattr(loader, 'initialize_detailed_descriptions'):
        loader.initialize_detailed_descriptions()
    layout.addWidget(loader)
    
    # App readiness is now handled internally by the loader based on timer
    # QTimer.singleShot(10000, lambda: loader.set_app_ready(True)) 
    
    loader.goButtonClicked.connect(dialog.close)
    
    # Connect node click to potentially close dialog as well (optional)
    # loader.nodeClicked.connect(lambda phase: dialog.close()) 
    
    if auto_close:
        QTimer.singleShot(duration, dialog.close)
        
    dialog.showMaximized()
    sys.exit(app.exec())

# Example usage
if __name__ == '__main__':
    show_demo(auto_close=False, duration=15000)