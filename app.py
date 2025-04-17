import asyncio
import sys
from PyQt6.QtGui import QAction, QIcon, QFont, QColor, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import QGraphicsOpacityEffect
from PyQt6.QtCore import QPropertyAnimation
import json
import requests
import os

# Import the BioNLP Annotation UI
from bionlp.annotation_ui import BioNlpAnnotationUI

from data.assumptions.assumptions import AssumptionsDisplayWidget
from data.cleaning.clean import DataCleaningWidget
from data.evaluation.evaluate import TestEvaluationWidget
from data.joins.join import DataJoinWidget
from data.mediation.mediate import MediationAnalysisWidget
from data.selection.select import DataTestingWidget
from data.sensitivity.sensitivity import SensitivityAnalysisWidget
from data.filtering.filter import DataFilteringWidget
from data.subgroups.subgroup import SubgroupAnalysisWidget
from exchange.exchange import BlockchainWidget
from exchange.validator_management import ValidatorManagementSection
from literature_search.evidence import LiteratureEvidenceSection
from literature_search.ranking import PaperRankingSection
from qt_sections.agent import AgentInterface
from common.status import TaskStatus
import websockets
from qasync import QEventLoop, asyncSlot
from PyQt6.QtCore import QThread, QTimer, Qt, QSize, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFrame, QToolBar, QMessageBox, QStackedWidget,
    QDialog, QLabel, QGridLayout, QSizePolicy, QTreeWidget, QTreeWidgetItem, QCheckBox,
    QMenu
)
from qt_sections.llm_manager import LlmManagerWidget
from qt_sections.participant_management import ParticipantManagementSection
from qt_sections.studies_manager import StudiesManagerWidget
from qt_workers.background_jobs import CouchDBStatusWorker, OllamaStatusWorker
from qt_sections.network import NetworkSection
from qt_sections.hypotheses_manager import HypothesesManagerWidget
from qt_sections.study_design import StudyDesignSection
from qt_sections.protocols import ProtocolSection
from plan.research_goals import ResearchPlanningWidget
from qt_sections.settings import SettingsSection
from qt_sections.teams import TeamSection
from qt_sections.user_access import UserAccess, LoginDialog
from qt_sections.database_ops import DatabaseSection
from literature_search.search import LiteratureSearchSection
from qt_sections.model_builder import ModelBuilder
from server import ConnectionManager
from study_model.studies_manager import StudiesManager
from websockets.protocol import State
from data.collection.collect import DataCollectionWidget, SourceConnection
from data.reshape.reshape import DataReshapeWidget
from data.interpretation.interpret import InterpretationWidget
from qt_material import apply_stylesheet
from helpers.load_icon import load_bootstrap_icon
from splash_loader.splash_loader import ResearchFlowLoader
from literature_search.evidence import LiteratureEvidenceSection
from literature_search.ranking import PaperRankingSection
from plan.hypothesis_generator import HypothesisGeneratorWidget

# Theme persistence handling
def save_theme_preference(theme_name, theme_file, invert_secondary=False):
    """Save theme preferences to a local file"""
    theme_data = {
        "theme_name": theme_name,
        "theme_file": theme_file,
        "invert_secondary": invert_secondary,
        "is_dark": "dark" in theme_name.lower()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.expanduser("~/.careframe"), exist_ok=True)
    
    # Save to JSON file
    with open(os.path.expanduser("~/.careframe/theme_preferences.json"), "w") as f:
        json.dump(theme_data, f)
    
def load_theme_preference():
    """Load theme preferences from local file"""
    try:
        with open(os.path.expanduser("~/.careframe/theme_preferences.json"), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default theme if file doesn't exist or is invalid
        return {
            "theme_name": "Purple Dark",
            "theme_file": "dark_purple.xml",
            "invert_secondary": False,
            "is_dark": True
        }

class ThemeSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theme Selector")
        self.setMinimumWidth(800)  # Increased width for better display
        self.setMinimumHeight(600)
        
        layout = QVBoxLayout(self)
        
        # Add header labels
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 10)
        
        light_label = QLabel("Light Themes")
        light_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        dark_label = QLabel("Dark Themes")
        dark_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        header_layout.addWidget(light_label)
        header_layout.addWidget(dark_label)
        layout.addWidget(header)
        
        # Create grid for themes
        grid = QGridLayout()
        grid.setSpacing(15)
        
        # Define available themes with their color palettes
        self.theme_pairs = [
            {
                "name": "Cyan",
                "light": {"file": "light_cyan_500.xml", "colors": ["#E0F7FA", "#00BCD4", "#006064"]},
                "dark": {"file": "dark_cyan.xml", "colors": ["#263238", "#00BCD4", "#E0F7FA"]}
            },
            {
                "name": "Blue",
                "light": {"file": "light_blue.xml", "colors": ["#E3F2FD", "#2196F3", "#0D47A1"]},
                "dark": {"file": "dark_blue.xml", "colors": ["#263238", "#2196F3", "#E3F2FD"]}
            },
            {
                "name": "Purple",
                "light": {"file": "light_purple.xml", "colors": ["#F3E5F5", "#9C27B0", "#4A148C"]},
                "dark": {"file": "dark_purple.xml", "colors": ["#263238", "#9C27B0", "#F3E5F5"]}
            },
            {
                "name": "Amber",
                "light": {"file": "light_amber.xml", "colors": ["#FFF8E1", "#FFC107", "#FF6F00"]},
                "dark": {"file": "dark_amber.xml", "colors": ["#263238", "#FFC107", "#FFF8E1"]}
            },
            {
                "name": "Teal",
                "light": {"file": "light_teal.xml", "colors": ["#E0F2F1", "#009688", "#004D40"]},
                "dark": {"file": "dark_teal.xml", "colors": ["#263238", "#009688", "#E0F2F1"]}
            },
            {
                "name": "Pink",
                "light": {"file": "light_pink.xml", "colors": ["#FCE4EC", "#E91E63", "#880E4F"]},
                "dark": {"file": "dark_pink.xml", "colors": ["#263238", "#E91E63", "#FCE4EC"]}
            },
            {
                "name": "Light Green",
                "light": {"file": "light_lightgreen.xml", "colors": ["#F1F8E9", "#8BC34A", "#33691E"]},
                "dark": {"file": "dark_lightgreen.xml", "colors": ["#263238", "#8BC34A", "#F1F8E9"]}
            },
            {
                "name": "Yellow",
                "light": {"file": "light_yellow.xml", "colors": ["#FFFDE7", "#FFEB3B", "#F57F17"]},
                "dark": {"file": "dark_yellow.xml", "colors": ["#263238", "#FFEB3B", "#FFFDE7"]}
            }
        ]
        
        for row, theme_pair in enumerate(self.theme_pairs):
            # Light theme
            light_widget = self.create_theme_widget(f"{theme_pair['name']} Light", theme_pair["light"])
            grid.addWidget(light_widget, row, 0)
            
            # Dark theme
            dark_widget = self.create_theme_widget(f"{theme_pair['name']} Dark", theme_pair["dark"])
            grid.addWidget(dark_widget, row, 1)
        
        layout.addLayout(grid)
        
        # Add invert secondary checkbox
        self.invert_secondary_checkbox = QCheckBox("Invert Secondary Colors")
        self.invert_secondary_checkbox.setToolTip("Changes the secondary color palette in the theme")
        self.invert_secondary_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                padding: 8px;
                margin-top: 10px;
            }
        """)
        layout.addWidget(self.invert_secondary_checkbox)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 14px;
                margin-top: 10px;
            }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def create_theme_widget(self, theme_name, theme_data):
        theme_widget = QWidget()
        theme_layout = QVBoxLayout(theme_widget)
        theme_layout.setSpacing(5)
        
        # Create color preview
        colors_widget = QWidget()
        colors_widget.setFixedHeight(40)
        colors_layout = QHBoxLayout(colors_widget)
        colors_layout.setSpacing(0)
        colors_layout.setContentsMargins(0, 0, 0, 0)
        
        # Display actual color swatches
        for color in theme_data["colors"]:
            color_preview = QFrame()
            color_preview.setStyleSheet(f"""
                background-color: {color};
                border: none;
                border-radius: 4px;
                margin: 2px;
            """)
            colors_layout.addWidget(color_preview)
        
        theme_layout.addWidget(colors_widget)
        
        # Add theme name button
        theme_btn = QPushButton(theme_name)
        theme_btn.setStyleSheet("""
            QPushButton {
                border: 2px solid #ccc;
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                font-size: 12px;
            }
            QPushButton:hover {
                border-color: #999;
                background-color: rgba(200, 200, 200, 0.2);
            }
        """)
        theme_btn.clicked.connect(lambda checked, t=theme_data: self.select_theme(theme_name, t))
        theme_layout.addWidget(theme_btn)
        
        theme_widget.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        return theme_widget
    
    def select_theme(self, theme_name, theme_data):
        self.selected_theme = theme_data
        self.selected_theme["name"] = theme_name
        self.selected_theme["invert_secondary"] = self.invert_secondary_checkbox.isChecked()
        self.accept()


class MainWindow(QMainWindow):
    # Add a theme changed signal
    theme_changed = pyqtSignal(bool)  # bool parameter indicates is_dark
    
    def __init__(self, show_immediately=True, existing_loader=None, existing_splash=None):
        """Initialize the main window and all components"""
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        
        self.setWindowTitle("CareFrame v0.1")
        
        self.initialization_complete = False
        self.main_window_ready = False
        self.pending_navigation_target = None  # Add this line for navigation after splash
        
        # Load saved theme preferences
        theme_prefs = load_theme_preference()
        self.current_theme = "dark" if theme_prefs["is_dark"] else "light"
        self.current_theme_file = theme_prefs["theme_file"]
        self.invert_secondary = theme_prefs.get("invert_secondary", False)
        
        # Use existing splash and loader if provided
        self.splash_dialog = existing_splash
        self.loader = existing_loader
        
        # Connect loader signals if it exists already
        if self.loader:
            try:
                # Disconnect and reconnect to ensure clean connection
                try:
                    self.loader.nodeClicked.disconnect()
                except:
                    pass
                print("DEBUG: Connecting loader nodeClicked signal to handle_loader_node_click")
                self.loader.nodeClicked.connect(self.handle_loader_node_click)
                print("DEBUG: Connection successful")
            except Exception as e:
                print(f"ERROR connecting nodeClicked signal: {e}")
        
        # Only create a new splash if we don't have one already
        if not self.splash_dialog:
            self.show_splash_screen()
        
        self.websocket_connection = None  # WebSocket connection attribute
        self.host = "127.0.0.1"
        self.port = 8889
        self.app_quiet_mode = False  # Add quiet mode state property

        # Initialize user access before creating sections
        self.user_access = UserAccess()
        self.studies_manager = StudiesManager()
        
        # Initialize default project and study if none exists
        self.initialize_default_project_study()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Add navigation state dictionary to track which groups are visible
        self.nav_groups_visible = {
            "Study": True,
            "Data": True,
            "Analysis": True,
            "Clinical": True
        }
        
        # Create the logo and title in the top left
        self.create_logo_and_title()
        
        # Create the side navigation first
        self.create_side_navigation()

        self.content_frame = QFrame()
        self.main_layout.addWidget(self.content_frame)
        self.content_layout = QVBoxLayout(self.content_frame)
        self.main_content_widget = QStackedWidget()
        self.content_layout.addWidget(self.main_content_widget)

        # Create sections
        self.create_sections()
        
        # Initialize the agentic system after creating sections
        self.initialize_agentic_system()
        
        # Theme setup
        # Set default values
        self.current_theme = "dark_blue"
        self.current_theme_file = "dark_blue.qss"
        self.current_icon_color = "#FFFFFF"  # Default for dark theme
        
        # Create corner buttons with consistent sizing and styles
        self.agent_button = QPushButton()
        self.agent_button.setIconSize(QSize(22, 22))
        self.agent_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.agent_button.setToolTip("AI Assistant")
        self.agent_button.clicked.connect(self.show_agent_section)
        self.agent_button.setProperty("icon_name", "chat-dots")
        
        self.exchange_button = QPushButton()
        self.exchange_button.setIconSize(QSize(22, 22))
        self.exchange_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exchange_button.setToolTip("Exchange Data")
        self.exchange_button.clicked.connect(self.show_evidence_section)
        self.exchange_button.setProperty("icon_name", "arrow-left-right")
        
        self.login_button = QPushButton()
        self.login_button.setIconSize(QSize(22, 22))
        self.login_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.login_button.setToolTip("Login/Logout")
        self.login_button.clicked.connect(self.show_login_dialog)
        self.login_button.setProperty("icon_name", "box-arrow-right")
        
        # User info label - will be updated with theme-appropriate colors later
        self.user_info_label = QLabel()
        
        # Create the menu bar first
        self.create_menu_bar()
        
        self.create_bottom_statusbar()
        self.initialize_ui(show_immediately)
        self.apply_theme()
        self.update_user_display()
        
        # Mark initialization as complete and update splash screen
        self.initialization_complete = True
        if hasattr(self, 'loader'):
            self.loader.set_app_ready(True)
    
    def create_logo_and_title(self):
        """Create and add the app logo and title to the top left of the window."""
        # Create a widget to hold the logo and title
        logo_widget = QWidget()
        logo_layout = QHBoxLayout(logo_widget)
        # Remove any stretching margins to prevent alignment issues
        logo_layout.setContentsMargins(5, 0, 5, 0) 
        logo_layout.setSpacing(5)
        
        # Load the SVG logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CareFrame.svg")
        
        if os.path.exists(logo_path):
            # Use QSvgWidget for proper SVG rendering
            from PyQt6.QtSvgWidgets import QSvgWidget
            
            # Create container for SVG to control hovering and sizing
            svg_container = QFrame()
            svg_container.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border: none;
                    padding: 4px 2px;
                    border-radius: 6px;
                }
                QFrame:hover {
                    background-color: rgba(150, 150, 150, 35);
                    border-radius: 6px;
                }
            """)

            # Use layout to properly center the SVG
            container_layout = QVBoxLayout(svg_container)
            container_layout.setContentsMargins(0, 0, 0, 0)

            # Create the SVG widget with fixed size and ensure transparency
            svg_widget = QSvgWidget(logo_path)
            svg_widget.setFixedSize(QSize(64, 26))
            svg_widget.setStyleSheet("background-color: transparent; border: none;")

            # Add to container layout
            container_layout.addWidget(svg_widget, 0, Qt.AlignmentFlag.AlignLeft)
            
            # Make clickable
            svg_container.setCursor(Qt.CursorShape.PointingHandCursor)
            svg_container.mousePressEvent = lambda e: self.show_logo_splash_loader()
            
            # Store references
            self.svg_widget = svg_widget
            
            # Add to main layout with left alignment
            logo_layout.addWidget(svg_container, 0, Qt.AlignmentFlag.AlignLeft)
        else:
            # Fallback if SVG not found
            fallback_label = QLabel("CareFrame")
            fallback_label.setStyleSheet("font-weight: bold; font-size: 16px;")
            logo_layout.addWidget(fallback_label, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Set a fixed size for the logo widget to prevent stretching
        logo_widget.setFixedWidth(74)
        
        # Store the logo widget to be added to the toolbar later
        self.logo_widget = logo_widget
    def show_logo_splash_loader(self):
        """Show the splash loader as a menu when the logo is clicked"""
        # Create a dialog to contain the loader
        dialog = QDialog(self)
        dialog.setWindowTitle("CareFrame Navigator")
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        
        # Apply rounded corners to the dialog
        dialog.setStyleSheet("""
            QDialog {
                background-color: transparent;
                border-radius: 20px;
            }
        """)
        
        # Set theme based on current preferences
        is_dark = self.is_dark_theme if hasattr(self, 'is_dark_theme') else True
        
        # Create a container widget with rounded corners for the loader
        container = QWidget(dialog)
        container.setObjectName("loaderContainer")
        
        container.setStyleSheet("""
            QWidget#loaderContainer {
                border-radius: 20px;
            }
        """)
        
        # Create layout for the dialog
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.addWidget(container)
        
        # Create layout for the container
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create a header widget for the close button and dragging
        header_widget = QWidget()
        header_widget.setFixedHeight(30)
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a title label that can be used for dragging
        title_label = QLabel("CareFrame Navigator")
        title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Add a close button to the top right corner
        close_button = QPushButton()
        close_button.setFixedSize(24, 24)
        close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_button.clicked.connect(dialog.close)
        
        # Style the close button
        close_icon = load_bootstrap_icon("x", size=16)
        close_button.setIcon(QIcon(close_icon.pixmap(16, 16)))
        close_button.setIconSize(QSize(16, 16))
        
        header_layout.addWidget(close_button)
        
        # Add header to container layout
        container_layout.addWidget(header_widget)
        
        # Create new ResearchFlowLoader instance
        from splash_loader.splash_loader import ResearchFlowLoader
        loader = ResearchFlowLoader(load_time=5000)  # Shorter load time for menu usage
        
        # Set loader theme
        loader.set_theme(is_dark)
        
        # Initialize detailed descriptions for hover tooltips
        if hasattr(loader, 'initialize_detailed_descriptions'):
            loader.initialize_detailed_descriptions()
        
        # Configure loader for menu mode: hide Go button and bottom labels
        loader.show_text = False  # Hide bottom labels/text
        loader.menu_mode = True   # Use as menu, no Go button
        
        # The loader is ready to use but we don't want the Go button
        # This sets a flag but doesn't show the button in menu mode
        loader.set_app_ready(True)
        
        # Connect signals
        loader.nodeClicked.connect(lambda phase: self.handle_splash_menu_click(phase, dialog))
        loader.goButtonClicked.connect(dialog.close)
        
        # Add loader to container layout (after the header)
        container_layout.addWidget(loader)
        
        # Show the dialog as a popup
        dialog.setMinimumSize(1000, 700)
        
        # Enable dragging by tracking mouse events on the dialog
        # Store initial position for drag calculation
        dialog._drag_pos = None
        
        # Override mousePressEvent to track when dragging starts
        def dialog_mouse_press(event):
            if event.button() == Qt.MouseButton.LeftButton:
                dialog._drag_pos = event.globalPosition().toPoint()
        
        # Override mouseMoveEvent to handle dragging
        def dialog_mouse_move(event):
            if dialog._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
                # Calculate the difference between current position and initial position
                delta = event.globalPosition().toPoint() - dialog._drag_pos
                # Move the dialog by this delta
                dialog.move(dialog.pos() + delta)
                # Update the initial position for the next move event
                dialog._drag_pos = event.globalPosition().toPoint()
        
        # Override mouseReleaseEvent to stop dragging
        def dialog_mouse_release(event):
            if event.button() == Qt.MouseButton.LeftButton:
                dialog._drag_pos = None
        
        # Attach the event handlers to the dialog
        dialog.mousePressEvent = dialog_mouse_press
        dialog.mouseMoveEvent = dialog_mouse_move
        dialog.mouseReleaseEvent = dialog_mouse_release
        
        dialog.show()
    def handle_splash_menu_click(self, phase, dialog):
        """Handle click on a node in the splash menu"""
        # Close the dialog first
        dialog.close()
        
        # Map phase names to their corresponding navigation methods
        phase_to_section = {
            # Main workflows
            "plan": self.show_planning_section,
            "manage": self.show_study_design_section,
            "literature": self.show_literature_search_section,
            "data": self.show_data_collection_section,
            "analysis": self.show_data_testing_section,
            "evidence": self.show_evidence_section,
            # Sub-phases (examples)
            "hypotheses": self.show_hypotheses_section,
            "design": self.show_study_design_section,
            "documentation": self.show_study_documentation_section,
            "participants": self.show_participant_management_section,
            "protocol": self.show_protocol_section,
            "search": self.show_literature_search_section,
            "ranking": self.show_literature_ranking_section,
            "claims": self.show_literature_evidence_section,
            "sources": self.show_data_collection_section,
            "cleaning": self.show_data_cleaning_section,
            "reshaping": self.show_data_reshaping_section,
            "filtering": self.show_data_filtering_section,
            "joining": self.show_data_joining_section,
            "evaluation": self.show_analysis_evaluation_section,
            "assumptions": self.show_analysis_assumptions_section,
            "subgroup": self.show_analysis_subgroup_section,
            "mediation": self.show_analysis_mediation_section,
            "sensitivity": self.show_analysis_sensitivity_section,
            "interpret": self.show_analysis_interpretation_section,
            # Tools
            "agent": self.show_agent_section,
            "network": self.show_network_section,
            "database": self.show_database_section,
            "team": self.show_team_section,
            "settings": self.show_settings_section
        }
        
        # Navigate to the selected section if it exists in the mapping
        if phase in phase_to_section:
            phase_to_section[phase]()

    def initialize_default_project_study(self):
        """Initialize a default project and study if none exist."""
        # Import necessary types
        from study_model.study_model import StudyDesign, StudyType
        import uuid
        from datetime import datetime
        
        # Check if there are any projects
        created_default = False
        if not self.studies_manager.projects:
            # Create a default project
            default_project = self.studies_manager.create_project(
                name="Project 1",
                description="Default project created automatically on startup"
            )
            
            # Create a basic study design
            study_design = StudyDesign(
                study_id=str(uuid.uuid4()),
                title="Title Placeholder",
                study_type=StudyType.BETWEEN_SUBJECTS,
                description=f"Default study created on {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            # Create a default study
            default_study = self.studies_manager.create_study(
                name="Study 1", 
                study_design=study_design,
                project_id=default_project.id
            )
            
            created_default = True
            
        # If we created a default project/study and the session manager is initialized,
        # refresh its display
        if created_default and hasattr(self, 'session_manager_section'):
            self.session_manager_section.refresh_projects_list()

    def initialize_agentic_system(self):
        """Initialize the agentic system and connect it to the agent interface."""
        try:
            # Set the agentic_initialized flag so the agent knows the system is available
            self.agent_section.agentic_initialized = True
            
            # Add a task to show the system is initialized
            task = self.agent_section.add_task(
                name="Initialize Agentic System",
                description="Loading clinical research assistance capabilities"
            )
            self.agent_section.update_task_status(task.name, TaskStatus.COMPLETED)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.agent_section.add_agent_message(
                "⚠️ There was an error initializing the advanced agent capabilities. "
                "Basic functionality is still available."
            )

    @asyncSlot()
    async def send_websocket_ping(self):
        """Sends a 'ping' message to the WebSocket server."""
        if self.websocket_connection and self.websocket_connection.state == State.OPEN:
            try:
                message = json.dumps({
                    "action": "ping",
                    "payload": {"message": "ping"}
                })
                await self.websocket_connection.send(message)
            except Exception as e:
                QMessageBox.critical(self, "WebSocket Error", f"Error sending message: {e}")
        else:
            QMessageBox.warning(self, "WebSocket", "Not connected to WebSocket server.")

    @asyncSlot()
    async def toggle_websocket_connection(self):
        """Connects or disconnects the WebSocket based on current state.
        
        If connected, disconnects.
        If disconnected, performs a health check and then connects.
        Once connected, schedules the listener task.
        """
        if self.websocket_connection and self.websocket_connection.state == State.OPEN:
            await self.disconnect_websocket()
        else:
            if hasattr(self, 'network_section') and self.network_section.is_host_mode:
                port = self.network_section.port
                host = "127.0.0.1"
            else:
                if not hasattr(self, 'network_section'):
                    return
                host = self.network_section.host_input.text()
                try:
                    port = int(self.network_section.host_port_input.text())
                except ValueError:
                    QMessageBox.warning(self, "Invalid Port", "Please enter a valid port number")
                    return

            await self.connect_websocket(host, port)
            if self.websocket_connection and self.websocket_connection.state == State.OPEN:
                # Schedule the websocket listener as a separate task
                self.listener_task = asyncio.create_task(self.websocket_listener())

    async def connect_websocket(self, ip, port):
        """Establishes a WebSocket connection after performing a health check."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, requests.get, f"http://{ip}:{port}/health"
            )
            if response.status_code != 200:
                raise Exception("Server is not responding")
        except requests.RequestException as e:
            self.update_status_indicator(self.websocket_status_label, False, "Server: Not Running", "hdd-network")
            QMessageBox.warning(self, "WebSocket", "Server is not running. Please start the server first.")
            return

        if self.websocket_connection:
            await self.disconnect_websocket()

        try:
            self.websocket_connection = await websockets.connect(f"ws://{ip}:{port}/ws", ping_interval=None)
            self.update_status_indicator(self.websocket_status_label, True, "Server: Connected", "hdd-network")
            QMessageBox.information(self, "WebSocket", "Connected successfully!")
            if hasattr(self, 'network_section'):
                self.network_section.update_websocket_status(True)
                if self.network_section.is_host_mode:
                    self.network_section.connect_button.setText("Disconnect")
                else:
                    self.network_section.connect_host_btn.setText("Disconnect")
                    self.network_section.subscribe_btn.setEnabled(True)
        except Exception as e:
            error_msg = str(e)
            self.update_status_indicator(self.websocket_status_label, False, f"Server: {error_msg}", "hdd-network")
            QMessageBox.warning(self, "WebSocket", f"Connection failed: {error_msg}")
            self.websocket_connection = None

    @asyncSlot()
    async def disconnect_websocket(self):
        """Disconnects from the WebSocket server."""
        if self.websocket_connection:
            try:
                await self.websocket_connection.close()
                if hasattr(self, 'listener_task'):
                    self.listener_task.cancel()
                    try:
                        await self.listener_task
                    except asyncio.CancelledError:
                        pass
            except Exception as e:
                pass
            finally:
                self.websocket_connection = None
                self.update_status_indicator(self.websocket_status_label, False, "Server: Disconnected", "hdd-network")
                if hasattr(self, 'network_section'):
                    self.network_section.update_websocket_status(False)
                    if self.network_section.is_host_mode:
                        self.network_section.connect_button.setText("Connect")
                    else:
                        self.network_section.connect_host_btn.setText("Connect to Host")
                        self.network_section.subscribe_btn.setEnabled(False)

    async def websocket_listener(self):
        """Listens for incoming messages on the WebSocket connection."""
        try:
            while True:
                if not self.websocket_connection or self.websocket_connection.state != State.OPEN:
                    break
                try:
                    message = await self.websocket_connection.recv()
                    data = json.loads(message)
                    if data.get("action") == "ping":
                        QMessageBox.information(self, "WebSocket", f"Received 'ping': {data}")
                    elif data.get("action") == "client_list":
                        if hasattr(self, 'network_section'):
                            self.network_section.handle_received_message(message)
                    elif data.get("action") == "message":
                        if hasattr(self, 'network_section'):
                            self.network_section.handle_received_message(message)
                    elif data.get("action") == "publish_ack":
                        payload = data.get("payload", {})
                        self.network_section.output_text.append(f"Publish Acknowledged: {payload.get('message', '')}")
                except websockets.exceptions.ConnectionClosedOK:
                    break
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    continue
        except Exception as e:
            pass
        finally:
            if self.websocket_connection:
                await self.websocket_connection.close()
            self.websocket_connection = None
            self.update_status_indicator(self.websocket_status_label, False, "Server: Disconnected", "hdd-network")

    def initialize_ui(self, show_immediately):
        # Set a reasonable default size first
        self.resize(1600, 1200)
        
        if hasattr(self, 'quiet_mode_toggle'):
            self.quiet_mode_toggle.setChecked(False)
            self.toggle_app_quiet_mode()

        # Show the projects section by default
        self.show_projects_section()
        
        # Only show the window immediately if requested
        if show_immediately:
            self.show()
            self.showMaximized()

    def show_splash_screen(self):
        """Show a splash screen with animated loader"""
        self.splash_dialog = QDialog(None, Qt.WindowType.FramelessWindowHint)
        self.splash_dialog.setStyleSheet("background-color: transparent;")
        splash_layout = QVBoxLayout(self.splash_dialog)
        splash_layout.setContentsMargins(0, 0, 0, 0)
        
        # Use existing loader if provided (for faster startup)
        if hasattr(self, 'loader') and self.loader:
            splash_layout.addWidget(self.loader)
        else:
            # Create animated loader
            from splash_loader.splash_loader import ResearchFlowLoader
            self.loader = ResearchFlowLoader(load_time=15000)
            
            # Set theme based on current preferences
            is_dark = self.is_dark_theme if hasattr(self, 'is_dark_theme') else True
            self.loader.set_theme(is_dark)
            
            # Initialize detailed descriptions for hover tooltips
            if hasattr(self.loader, 'initialize_detailed_descriptions'):
                self.loader.initialize_detailed_descriptions()
                
            # Connect signals
            self.splash_counter = 0  # Initialize counter for animation cycles
            self.loader.animationCompleted.connect(self.on_splash_animation_completed)
            self.loader.nodeClicked.connect(self.handle_loader_node_click)
            self.loader.goButtonClicked.connect(self.close_splash_and_show_main)
            
            splash_layout.addWidget(self.loader)
        
        # Show splash maximized immediately
        self.splash_dialog.showMaximized()
        QApplication.processEvents()  # Force immediate processing of events to show the splash
        
        # Start checking app readiness after splash is visible
        QTimer.singleShot(1000, self.check_app_readiness)
    
    def check_app_readiness(self):
        """Check if all subsystems are ready and mark the app as ready when true"""
        # Check both loading state and critical services
        if self.initialization_complete:
            ollama_ready = hasattr(self, 'ollama_status_label') and "Connected" in self.ollama_status_label.toolTip()
            db_ready = hasattr(self, 'database_status_label') and "Connected" in self.database_status_label.toolTip()
            
            # Set ready if either the app is fully initialized or critical services are connected
            if ollama_ready and db_ready:
                if hasattr(self, 'loader'):
                    self.loader.set_app_ready(True)
                return True
        
        # Check again in 500ms if not ready
        QTimer.singleShot(500, self.check_app_readiness)
        return False
    
    def close_splash_and_show_main(self):
        """Close the splash screen and show the main window."""
        print("DEBUG: close_splash_and_show_main called")
        
        # Save the navigation target first
        navigation_target = None
        if hasattr(self, 'pending_navigation_target') and self.pending_navigation_target:
            navigation_target = self.pending_navigation_target
            print(f"DEBUG: Saved navigation target: {navigation_target.__name__}")
            # Reset it immediately to avoid multiple calls
            self.pending_navigation_target = None
        
        try:
            self.close_splash_screen() # Close splash first
        except Exception as e:
            print(f"ERROR closing splash screen: {e}")

        if not self.isVisible(): # Only show if not already visible
            try:
                print("DEBUG: Showing main window")
                self.show()
                self.showMaximized()
                QApplication.processEvents()
                print("DEBUG: Main window shown")
            except Exception as e:
                 print(f"ERROR showing main window: {e}")
            
        self.main_window_ready = True
        
        # Navigate to the selected section if one was saved
        if navigation_target:
            try:
                print(f"DEBUG: Executing navigation to: {navigation_target.__name__}")
                # Use a timer with longer delay to ensure UI is fully ready before navigating
                QTimer.singleShot(500, navigation_target)
            except Exception as e:
                print(f"ERROR executing navigation: {e}")
        else:
            print("DEBUG: No navigation target to execute")

    def close_splash_screen(self):
        """Force close the splash screen if it's still open"""
        print("DEBUG: close_splash_screen called") # Added print
        if hasattr(self, 'splash_dialog') and self.splash_dialog:
            try:
                if hasattr(self, 'loader') and hasattr(self.loader, 'timer'):
                    self.loader.timer.stop()
                self.splash_dialog.close()
                self.splash_dialog = None 
                print("DEBUG: Splash dialog closed and set to None") # Added print
            except Exception as e:
                print(f"ERROR in close_splash_screen: {e}")
        else:
             print("DEBUG: No splash_dialog found or already None in close_splash_screen") # Added print

    def on_splash_animation_completed(self):
        """Handle splash animation completion"""
        self.splash_counter += 1
        if self.splash_counter >= 3:  # Close after 3 animation cycles if ready
            # Check if the app is ready before closing splash
            if self.initialization_complete and hasattr(self, 'loader') and self.loader.app_ready:
                self.close_splash_and_show_main()
            else:
                # If not ready yet but initialized, set app ready to show the GO button
                if self.initialization_complete and hasattr(self, 'loader'):
                    self.loader.set_app_ready(True)

    def create_sections(self):
        # Agent Interface
        self.agent_section = AgentInterface(self)
        self.main_content_widget.addWidget(self.agent_section)
        
        # Connect the file upload signal from agent to handler in main window
        self.agent_section.file_upload_requested.connect(self.handle_agent_file_upload)
        
        # BioNLP / UMLS Medical Term Viewer
        self.bionlp_section = BioNlpAnnotationUI()
        self.main_content_widget.addWidget(self.bionlp_section)
        
        # Hub
        from qt_sections.hub import HubSection
        self.projects_section = HubSection()
        
        # Connect the studies manager and set it
        if hasattr(self, 'studies_manager'):
            self.projects_section.set_studies_manager(self.studies_manager)
        
        self.main_content_widget.addWidget(self.projects_section)
        
        # Ensure the hub refreshes data when displayed
        def refresh_projects_on_show():
            current_widget = self.main_content_widget.currentWidget()
            if current_widget == self.projects_section:
                # Force hub section to refresh when shown
                self.projects_section.refresh_data()
        
        # Connect to current changed signal
        self.main_content_widget.currentChanged.connect(refresh_projects_on_show)
        
        # Add Data Collection and Harmonization widgets
        self.data_collection_widget = DataCollectionWidget()
        self.main_content_widget.addWidget(self.data_collection_widget)
        
        self.data_reshape_widget = DataReshapeWidget()
        self.main_content_widget.addWidget(self.data_reshape_widget)
        
        self.data_cleaning_widget = DataCleaningWidget()
        self.main_content_widget.addWidget(self.data_cleaning_widget)
        
        self.data_filtering_widget = DataFilteringWidget()
        self.main_content_widget.addWidget(self.data_filtering_widget)

        self.data_testing_widget = DataTestingWidget()
        self.data_testing_widget.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.data_testing_widget)
        
        self.data_join_widget = DataJoinWidget()
        self.main_content_widget.addWidget(self.data_join_widget)
        
        # Statistical Interpretation
        self.statistical_interpretation_widget = InterpretationWidget()
        self.statistical_interpretation_widget.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.statistical_interpretation_widget)
        
        # Test Evaluation
        self.test_evaluation_widget = TestEvaluationWidget()
        self.test_evaluation_widget.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.test_evaluation_widget)
        
        # Subgroup Analysis
        self.subgroup_analysis_widget = SubgroupAnalysisWidget()
        self.main_content_widget.addWidget(self.subgroup_analysis_widget)

        # Assumptions
        self.display_assumptions_widget = AssumptionsDisplayWidget()
        self.display_assumptions_widget.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.display_assumptions_widget)

        # Mediation Analysis
        self.mediation_analysis_widget = MediationAnalysisWidget()
        self.main_content_widget.addWidget(self.mediation_analysis_widget)
        
        # Sensitivity Analysis
        self.sensitivity_analysis_widget = SensitivityAnalysisWidget()
        self.main_content_widget.addWidget(self.sensitivity_analysis_widget)

        # Study Design
        self.study_design_section = StudyDesignSection(parent=self)
        self.main_content_widget.addWidget(self.study_design_section)
        
        # Participant Management
        self.participant_management_section = ParticipantManagementSection(parent=self)
        self.participant_management_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.participant_management_section)

        # Study Model Builder
        self.study_model_builder = ModelBuilder()
        self.study_model_builder.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.study_model_builder)

        # Studies Manager
        self.session_manager_section = StudiesManagerWidget()
        self.session_manager_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.session_manager_section)

        # Settings
        self.settings_section = SettingsSection()
        self.main_content_widget.addWidget(self.settings_section)

        # Hypotheses
        self.hypotheses_section = HypothesesManagerWidget()
        self.hypotheses_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.hypotheses_section)

        # Search
        self.literature_search_section = LiteratureSearchSection()
        self.literature_search_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.literature_search_section)

        # Ranking
        self.literature_ranking_section = PaperRankingSection()
        self.literature_ranking_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.literature_ranking_section)
        
        # Evidence
        self.literature_evidence_section = LiteratureEvidenceSection()
        self.literature_evidence_section.set_studies_manager(self.studies_manager)
        self.literature_evidence_section.set_hypotheses_manager(self.hypotheses_section)
        self.main_content_widget.addWidget(self.literature_evidence_section)

        # Connect search section's papersCollected signal to ranking section's set_papers method
        self.literature_search_section.papersCollected.connect(self.literature_ranking_section.set_papers)

        # Connect search section's papersCollected signal to evidence section's set_papers method
        self.literature_search_section.papersCollected.connect(self.literature_evidence_section.set_papers)

        # Connect ranking section's papersRanked signal to evidence section's set_papers method
        self.literature_ranking_section.papersRanked.connect(self.literature_evidence_section.set_papers)

        # Protocol
        self.protocol_section = ProtocolSection()
        self.main_content_widget.addWidget(self.protocol_section)

        # Evidence / Blockchain
        self.evidence_section = BlockchainWidget(self.studies_manager, self.user_access)
        self.main_content_widget.addWidget(self.evidence_section)

        # Validator Management
        self.validator_management_section = ValidatorManagementSection()
        self.validator_management_section.user_access = self.user_access
        self.validator_management_section.blockchain_api = self.evidence_section.blockchain_api
        self.main_content_widget.addWidget(self.validator_management_section)

        # Flow Diagram
        self.study_plan_section = ResearchPlanningWidget()
        self.study_plan_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.study_plan_section)

        # LLM Manager
        self.llm_manager_section = LlmManagerWidget()
        self.main_content_widget.addWidget(self.llm_manager_section)

        # Team
        self.team_section = TeamSection()
        self.team_section.user_access = self.user_access
        self.main_content_widget.addWidget(self.team_section)
        
        # Database Operations
        self.database_section = DatabaseSection()
        self.database_section.user_access = self.user_access
        self.main_content_widget.addWidget(self.database_section)

        # Network
        self.network_section = NetworkSection(self)
        self.main_content_widget.addWidget(self.network_section)

        # Hypothesis Generator
        self.hypothesis_generator_section = HypothesisGeneratorWidget()
        self.hypothesis_generator_section.set_studies_manager(self.studies_manager)
        self.main_content_widget.addWidget(self.hypothesis_generator_section)
        
        # Explicitly connect the testing widget to the hypothesis generator
        if hasattr(self, 'data_testing_widget') and self.data_testing_widget:
            self.hypothesis_generator_section.set_testing_widget(self.data_testing_widget)
        else:
            pass

        # Store references to all buttons
        self.all_nav_buttons = [
            {"text": "Portfolio", "icon": "home", "callback": self.show_projects_section, "group": "Clinical"},
            {"text": "Session", "icon": "save", "callback": self.show_session_manager_section, "group": "Study"},
            {"text": "Hypotheses", "icon": "lightbulb", "callback": self.show_hypotheses_section, "group": "Clinical"},
            {"text": "Validation", "icon": "check-square", "callback": self.show_validator_management_section, "group": "Evidence"},
            {"text": "Planning", "icon": "diagram-3", "callback": self.show_planning_section, "group": "Study"},
            {"text": "Design", "icon": "diagram-3", "callback": self.show_study_design_section, "group": "Study"},
            {"text": "Documentation", "icon": "project-diagram", "callback": self.show_study_documentation_section, "group": "Study"},
            {"text": "Sources", "icon": "shuffle", "callback": self.show_data_collection_section, "group": "Data"},
            {"text": "Clean", "icon": "shuffle", "callback": self.show_data_cleaning_section, "group": "Data"},
            {"text": "Filter", "icon": "shuffle", "callback": self.show_data_filtering_section, "group": "Data"},
            {"text": "Merge", "icon": "shuffle", "callback": self.show_data_joining_section, "group": "Data"},
            {"text": "Reshape", "icon": "shuffle", "callback": self.show_data_reshaping_section, "group": "Data"},
            {"text": "Model", "icon": "activity", "callback": self.show_data_testing_section, "group": "Analysis"},
            {"text": "Evaluation", "icon": "activity", "callback": self.show_analysis_evaluation_section, "group": "Analysis"},
            {"text": "Assumptions", "icon": "activity", "callback": self.show_analysis_assumptions_section, "group": "Analysis"},
            {"text": "Subgroup", "icon": "activity", "callback": self.show_analysis_subgroup_section, "group": "Analysis"},
            {"text": "Generator", "icon": "gear-wide-connected", "callback": self.show_hypothesis_generator_section, "group": "Clinical"},
            {"text": "Protocol", "icon": "file-text", "callback": self.show_protocol_section, "group": "Clinical"},
            {"text": "Search", "icon": "n-search", "callback": self.show_literature_search_section, "group": "Clinical"},
            {"text": "Mediation", "icon": "activity", "callback": self.show_analysis_mediation_section, "group": "Analysis"},
            {"text": "Sensitivity", "icon": "activity", "callback": self.show_analysis_sensitivity_section, "group": "Analysis"},
            {"text": "Interpret", "icon": "activity", "callback": self.show_analysis_interpretation_section, "group": "Analysis"},
            {"text": "Evidence", "icon": "blockchain", "callback": self.show_evidence_section, "group": "Evidence"},
            {"text": "Team", "icon": "users", "callback": self.show_team_section, "group": "Team"},
            {"text": "Database", "icon": "database", "callback": self.show_database_section, "group": "Database"},
            {"text": "Network", "icon": "network", "callback": self.show_network_section, "group": "Network"},
            {"text": "Agent", "icon": "robot", "callback": self.show_agent_section, "group": "Agent"},
        ]

    def create_side_navigation(self):
        # Create a vertical sidebar widget
        self.navigation_sidebar = QWidget()
        self.navigation_sidebar.setFixedWidth(200)  # Maintain the same width as before
        self.sidebar_is_collapsed = False  # Track if navigation is collapsed
        self.sidebar_icons_only = False  # Track if navigation is in icons-only mode
        
        # Get current theme info
        is_dark = hasattr(self, 'current_theme_file') and "dark" in self.current_theme_file
        strip_color = "#606060" if is_dark else "#aaaaaa"
        strip_hover_color = "#808080" if is_dark else "#888888"
        
        # Create main layout for the side navigation
        side_container = QHBoxLayout(self.navigation_sidebar)
        side_container.setContentsMargins(0, 0, 0, 0)
        side_container.setSpacing(0)
        
        # Create tree widget that will hold the navigation items
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)  # Hide the header
        self.nav_tree.setAnimated(True)  # Animate expanding/collapsing
        self.nav_tree.setIndentation(15)  # Reduce indentation to save space
        self.nav_tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Remove focus rectangle
        
        
        self.nav_tree.setStyleSheet(f"""
            QTreeWidget {{
                border: none;
                font-size: 12px;
            }}
            QTreeWidget::item {{
                padding: 4px 2px;
            }}
            QTreeWidget::item:selected {{
                border: none;
            }}
            QTreeWidget::item:hover {{
                background-color: rgba(150, 150, 150, 40);
            }}
        """)
        
        # Create the tree structure
        self.populate_nav_tree()
        
        # Connect the tree item selection signal
        self.nav_tree.itemClicked.connect(self.handle_nav_tree_selection)
        
        self.sidebar_toggle_strip = QFrame()
        self.sidebar_toggle_strip.setFixedWidth(15)
        self.sidebar_toggle_strip.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Add style with hover effect and theme-appropriate colors
        self.sidebar_toggle_strip.setStyleSheet(f"""
            QFrame {{
                border-radius: 8px;
                margin-right: 7px;
                margin-left: 4px;
                background: {strip_color};
            }}
            QFrame:hover {{
                margin-right: 10px; /* Expand to the right */
                background: {strip_hover_color};
            }}
        """)
        
        # Create hover event handlers for width change effect
        self.sidebar_toggle_strip.enterEvent = self.toggle_strip_enter
        self.sidebar_toggle_strip.leaveEvent = self.toggle_strip_leave
        self.sidebar_toggle_strip.mousePressEvent = self.cycle_sidebar_state
        
        # Add tree widget and toggle strip to main layout
        side_container.addWidget(self.sidebar_toggle_strip)
        side_container.addWidget(self.nav_tree)
        
        # Add to main layout
        self.main_layout.insertWidget(0, self.navigation_sidebar)

    def populate_nav_tree(self):
        """Create the tree structure for navigation."""
        # Get current theme info
        is_dark = hasattr(self, 'current_theme_file') and "dark" in self.current_theme_file
        text_color = "#FFFFFF" if is_dark else "#212121"
        
        # Define the groups and their icons
        group_icons = {
            "Home": "house",
            "Strategy": "diagram-3",
            "Manage": "clipboard-data",
            "Literature": "book",
            "Evidence": "diagram-3",
            "Data": "database",
            "Analysis": "graph-up",
            "Tools": "gear-wide"
        }
        
        # Define the groups and their items
        nav_structure = {
            "Home": [
                {"text": "Dashboard", "callback": self.show_projects_section, "icon": "folder2"},
                {"text": "Session", "callback": self.show_session_manager_section, "icon": "save"},
            ],
            "Strategy": [
                {"text": "Plan", "callback": self.show_planning_section, "icon": "diagram-3"},
                {"text": "Hypotheses", "callback": self.show_hypotheses_section, "icon": "lightbulb"},
            ],
            "Manage": [
                {"text": "Design", "callback": self.show_study_design_section, "icon": "pen"},
                {"text": "Documentation", "callback": self.show_study_documentation_section, "icon": "file-text"},
                {"text": "Participants", "callback": self.show_participant_management_section, "icon": "people"},
                {"text": "Protocol", "callback": self.show_protocol_section, "icon": "clipboard-check"}
            ],
            "Literature": [
                {"text": "Search", "callback": self.show_literature_search_section, "icon": "book"},
                {"text": "Rank", "callback": self.show_literature_ranking_section, "icon": "book"},
                {"text": "Claims", "callback": self.show_literature_evidence_section, "icon": "journal-check"},
            ],
            "Data": [
                {"text": "Sources", "callback": self.show_data_collection_section, "icon": "gear-wide-connected"},
                {"text": "Clean", "callback": self.show_data_cleaning_section, "icon": "brush"},
                {"text": "Merge", "callback": self.show_data_joining_section, "icon": "intersect"},
                {"text": "Filter", "callback": self.show_data_filtering_section, "icon": "funnel"},
                {"text": "Reshape", "callback": self.show_data_reshaping_section, "icon": "shuffle"},
            ],
            "Analysis": [
                {"text": "Model", "callback": self.show_data_testing_section, "icon": "boxes"},
                {"text": "Evaluate", "callback": self.show_analysis_evaluation_section, "icon": "check-square"},
                {"text": "Assumptions", "callback": self.show_analysis_assumptions_section, "icon": "exclamation-triangle"},
                {"text": "Subgroup", "callback": self.show_analysis_subgroup_section, "icon": "diagram-2"},
                {"text": "Mediation", "callback": self.show_analysis_mediation_section, "icon": "arrows"},
                {"text": "Sensitivity", "callback": self.show_analysis_sensitivity_section, "icon": "sliders"},
                {"text": "Interpret", "callback": self.show_analysis_interpretation_section, "icon": "chat-text"},
            ]
        }
        
        # Create the tree items
        for group_name, items in nav_structure.items():
            group_item = QTreeWidgetItem(self.nav_tree, [group_name])
            group_item.setExpanded(True)  # Expand by default
            
            # Set a flag to identify this as a header (non-clickable group)
            group_item.setData(0, Qt.ItemDataRole.UserRole, "header")
            
            # Add icon to group item
            if group_name in group_icons:
                icon_name = group_icons[group_name]
                group_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                # Store the icon name for later refreshing
                group_item.setData(0, Qt.ItemDataRole.UserRole + 1, icon_name)
            
            # Optional: Set a bold font for group headers
            font = QFont()
            font.setBold(True)
            group_item.setFont(0, font)
            
            # Add items to this group
            for item_info in items:
                child_item = QTreeWidgetItem(group_item, [item_info["text"]])
                # Store the callback function in the item data
                child_item.setData(0, Qt.ItemDataRole.UserRole, item_info["callback"])
                
                # Add icon to child item
                if "icon" in item_info:
                    icon_name = item_info["icon"]
                    child_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                    # Store the icon name for later refreshing
                    child_item.setData(0, Qt.ItemDataRole.UserRole + 1, icon_name)

    def handle_nav_tree_selection(self, item, column):
        """Handle the selection of a tree item."""
        # Get the callback function stored in the item data
        callback_data = item.data(0, Qt.ItemDataRole.UserRole)
        
        # Skip if this is a header/group item
        if callback_data == "header":
            return
            
        # If this is a child item with a callback, execute it
        if callback_data is not None:
            callback_data()

    def toggle_strip_enter(self, event):
        # Increase width when mouse enters
        self.sidebar_toggle_strip.setFixedWidth(25)  # Expand to 25px on hover

    def toggle_strip_leave(self, event):
        # Reset width when mouse leaves
        self.sidebar_toggle_strip.setFixedWidth(15)  # Return to 15px

    def cycle_sidebar_state(self, event=None):
        # Cycle through the three states: full -> icons-only -> collapsed -> full
        if not self.sidebar_is_collapsed and not self.sidebar_icons_only:
            # Full view -> Icons-only view
            self.sidebar_icons_only = True
            self.sidebar_is_collapsed = False
            self.navigation_sidebar.setFixedWidth(85)  # Width for icons only
            self.set_tree_icons_only_mode(True)
        elif self.sidebar_icons_only and not self.sidebar_is_collapsed:
            # Icons-only view -> Collapsed
            self.sidebar_icons_only = False
            self.sidebar_is_collapsed = True
            self.nav_tree.setVisible(False)
            self.navigation_sidebar.setFixedWidth(15)  # Collapse to just show the toggle strip
        else:
            # Collapsed -> Full view
            self.sidebar_icons_only = False
            self.sidebar_is_collapsed = False
            self.nav_tree.setVisible(True)
            self.navigation_sidebar.setFixedWidth(200)  # Expand to show all content
            self.set_tree_icons_only_mode(False)

    def set_tree_icons_only_mode(self, icons_only):
        """Set the navigation tree to icons-only mode or full mode."""
        # Get current theme info
        is_dark = "dark" in self.current_theme_file
        primary_color = self.get_theme_primary_color(is_dark)
        
        # Calculate contrasting colors for text
        text_color = "#FFFFFF" if is_dark else "#212121"
        hover_bg = f"rgba({','.join(str(int(primary_color[1:][i:i+2], 16)) for i in (0, 2, 4))}, 0.15)"
        selected_bg = f"rgba({','.join(str(int(primary_color[1:][i:i+2], 16)) for i in (0, 2, 4))}, 0.3)"
        
        # Set icon-only mode for all tree items
        root = self.nav_tree.invisibleRootItem()
        for i in range(root.childCount()):
            group_item = root.child(i)
            
            # Check if this is a header
            is_header = group_item.data(0, Qt.ItemDataRole.UserRole) == "header"
            
            # Set group item display mode
            if icons_only:
                # Store the text in the tooltip
                group_item.setToolTip(0, group_item.text(0))
                group_item.setText(0, "")
                
                # Get icon name and apply with theme color
                icon_name = group_item.data(0, Qt.ItemDataRole.UserRole + 1)
                if icon_name:
                    group_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                
                # For header items, disable selection and set a different background
                if is_header:
                    group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    group_item.setData(0, Qt.ItemDataRole.BackgroundRole, 
                                      QColor(150, 150, 150, 30 if is_dark else 15))
            else:
                # Restore text from group name (we can retrieve it from itemData if needed)
                group_name = group_item.toolTip(0)
                if group_name:
                    group_item.setText(0, group_name)
                
                # Set text color based on theme
                group_item.setForeground(0, QColor(text_color))
                
                # Update icon with theme color
                icon_name = group_item.data(0, Qt.ItemDataRole.UserRole + 1)
                if icon_name:
                    group_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                
                # Clear any background color
                if is_header:
                    group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    group_item.setData(0, Qt.ItemDataRole.BackgroundRole, QColor(0, 0, 0, 0))
            
            # Set children items display mode
            for j in range(group_item.childCount()):
                child_item = group_item.child(j)
                if icons_only:
                    # Store text in tooltip for icons-only mode
                    child_item.setToolTip(0, child_item.text(0))
                    child_item.setText(0, "")
                    
                    # Update icon with theme color
                    icon_name = child_item.data(0, Qt.ItemDataRole.UserRole + 1)
                    if icon_name:
                        child_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                    
                    # Ensure child items are selectable
                    child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsSelectable)
                else:
                    # Restore text from tooltip
                    item_text = child_item.toolTip(0)
                    if item_text:
                        child_item.setText(0, item_text)
                    
                    # Set text color based on theme
                    child_item.setForeground(0, QColor(text_color))
                    
                    # Update icon with theme color
                    icon_name = child_item.data(0, Qt.ItemDataRole.UserRole + 1)
                    if icon_name:
                        child_item.setIcon(0, load_bootstrap_icon(icon_name, text_color))
                    
                    # Ensure child items are selectable
                    child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsSelectable)
        
        # Hide horizontal scrollbar in icons-only mode
        if icons_only:
            self.nav_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            
            # Hide the branch lines completely in icons-only mode
            self.nav_tree.setRootIsDecorated(False)  # Hide the expand/collapse controls
            self.nav_tree.setItemsExpandable(False)  # Prevent expanding/collapsing
            
            # Adjust the tree style for icons-only mode with theme-aware colors
            self.nav_tree.setStyleSheet(f"""
                QTreeWidget {{
                    border: none;
                    background-color: transparent;
                    show-decoration-selected: 0;
                }}
                QTreeWidget::item {{
                    padding: 12px 4px;
                    margin: 3px;
                    text-align: center;
                    border-radius: 8px;
                    color: {text_color};
                }}
                QTreeWidget::item:selected {{
                    border: none;
                    background-color: {selected_bg};
                    border-radius: 8px;
                }}
                QTreeWidget::item:hover {{
                    background-color: {hover_bg};
                    border-radius: 8px;
                }}
                QTreeWidget::branch {{
                    background: transparent;
                    border: none;
                }}
                QTreeWidget::branch:selected {{
                    background: transparent;
                }}
                QTreeWidget::branch:has-siblings:!adjoins-item {{
                    border: none;
                    background: transparent;
                }}
                QTreeWidget::branch:has-siblings:adjoins-item {{
                    border: none;
                    background: transparent;
                }}
                QTreeWidget::branch:!has-children:!has-siblings:adjoins-item {{
                    border: none;
                    background: transparent;
                }}
                QScrollBar:horizontal {{
                    height: 0px;
                    background: transparent;
                }}
            """)
            # Reduce indentation in icons-only mode to zero
            self.nav_tree.setIndentation(0)
        else:
            # Restore normal scrollbar policy and branch display
            self.nav_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.nav_tree.setRootIsDecorated(True)  # Show the expand/collapse controls
            self.nav_tree.setItemsExpandable(True)  # Allow expanding/collapsing
            
            # Reset to original style with theme-aware colors
            self.nav_tree.setStyleSheet(f"""
                QTreeWidget {{
                    border: none;
                    font-size: 12px;
                    color: {text_color};
                }}
                QTreeWidget::item {{
                    padding: 4px 2px;
                    color: {text_color};
                }}
                QTreeWidget::item:selected {{
                    border: none;
                    background-color: {selected_bg};
                    border-radius: 3px;
                }}
                QTreeWidget::item:hover {{
                    background-color: {hover_bg};
                    border-radius: 3px;
                }}
            """)
            # Restore normal indentation
            self.nav_tree.setIndentation(15)

    def get_theme_primary_color(self, is_dark):
        """Extract primary theme color from the current theme file name."""
        if not hasattr(self, 'current_theme_file'):
            return None
            
        # Extract color from theme file name
        theme_file = self.current_theme_file.lower()
        
        # Look for color names in the theme file
        color_map = {
            "cyan": "#00BCD4",
            "blue": "#2196F3",
            "purple": "#9C27B0",
            "amber": "#FFC107",
            "teal": "#009688",
            "pink": "#E91E63", 
            "lightgreen": "#8BC34A",
            "green": "#4CAF50",
            "yellow": "#FFEB3B",
            "orange": "#FF9800",
            "red": "#F44336",
            "deeppurple": "#673AB7",
            "indigo": "#3F51B5",
            "lightblue": "#03A9F4",
            "lime": "#CDDC39",
            "deeporange": "#FF5722",
            "grey": "#9E9E9E",
            "bluegrey": "#607D8B"
        }
        
        # Find matching color name in theme file
        for color_name, color_value in color_map.items():
            if color_name in theme_file:
                return color_value
                
        # Default color if none found
        return "#3F51B5"  # Material Blue

    def adjust_color_brightness(self, hex_color, percent):
        # Utility to adjust color brightness for hover effect
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # Adjust brightness
        rgb_new = []
        for c in rgb:
            adjusted = c + (255 - c) * percent / 100 if percent > 0 else c * (1 + percent / 100)
            rgb_new.append(max(0, min(255, int(adjusted))))
        
        return '#{:02x}{:02x}{:02x}'.format(*rgb_new)
        
    def get_primary_color(self):
        return self.get_theme_primary_color(self.current_theme_is_dark)

    def apply_theme(self):
        # Load theme settings
        theme_prefs = load_theme_preference()
        theme_name = theme_prefs["theme_name"]
        theme_file = theme_prefs["theme_file"]
        invert_secondary = theme_prefs.get("invert_secondary", False)
        
        # Store current theme information for later reference
        self.current_theme = theme_name
        self.current_theme_file = theme_file
        self.invert_secondary = invert_secondary
        
        # Determine if this is a dark theme
        is_dark = "dark" in theme_file
        self.is_dark_theme = is_dark
        
        # Apply the stylesheet using qt_material
        apply_stylesheet(self, theme=theme_file, invert_secondary=invert_secondary)
        
        # Set primary and accent colors from the theme
        primary_color = self.get_theme_primary_color(is_dark) or "#3f51b5"  # Default blue if not specified
        if is_dark:
            hover_color = "rgba(100, 100, 100, 0.2)"
            selected_color = "rgba(100, 100, 100, 0.4)" 
        else:
            hover_color = "rgba(200, 200, 200, 0.3)"
            selected_color = "rgba(200, 200, 200, 0.5)"
            
        # Set contrast text color based on theme
        text_color = "#FFFFFF" if is_dark else "#212121"
        self.current_icon_color = text_color
        
        # Update SVG widget if it exists 
        if hasattr(self, 'svg_widget') and self.svg_widget:
            try:
                # Store last theme state to avoid redundant updates
                self.svg_theme_last = is_dark
            except Exception as e:
                print(f"Error updating SVG theme: {str(e)}")
        
        # Update top toolbar if it exists
        if hasattr(self, 'top_toolbar'):
            # Update any theme-specific styling for the toolbar
            self.top_toolbar.setStyleSheet("""
                QToolBar {
                    background-color: transparent;
                }
            """)
        
        # Update toggle strip colors
        if hasattr(self, 'sidebar_toggle_strip'):
            strip_color = "#606060" if is_dark else "#aaaaaa"
            strip_hover_color = "#808080" if is_dark else "#888888"
            self.sidebar_toggle_strip.setStyleSheet(f"""
                QFrame {{
                    border-radius: 8px;
                    margin-right: 7px;
                    margin-left: 4px;
                    background: {strip_color};
                }}
                QFrame:hover {{
                    margin-right: 10px;
                    background: {strip_hover_color};
                }}
            """)
        
        # Update tree styling if already initialized
        if hasattr(self, 'nav_tree'):
            # If in icon-only mode, apply that style with new colors
            if hasattr(self, 'sidebar_icons_only') and self.sidebar_icons_only:
                self.set_tree_icons_only_mode(True)
            else:
                # Otherwise apply regular mode with new colors
                self.nav_tree.setStyleSheet(f"""
                    QTreeWidget {{
                        border: none;
                        font-size: 12px;
                        color: {text_color};
                    }}
                    QTreeWidget::item {{
                        padding: 4px 2px;
                        color: {text_color};
                    }}
                    QTreeWidget::item:selected {{
                        border: none;
                        background-color: {selected_color};
                        border-radius: 3px;
                    }}
                    QTreeWidget::item:hover {{
                        background-color: {hover_color};
                        border-radius: 3px;
                    }}
                """)
        
        # Update chain studies widget theme
        if hasattr(self, 'study_plan_section'):
            self.study_plan_section.update_theme(is_dark)
        
        # Propagate theme to study model builder
        if hasattr(self, 'study_model_builder'):
            self.study_model_builder.update_theme(is_dark)

        # Update theme for specialized widgets
        if hasattr(self, 'display_assumptions_widget'):
            self.display_assumptions_widget.set_theme(is_dark)
        
        # Refresh ALL icons in the application to ensure they match the current theme
        self.refresh_all_icons()

        # Emit theme changed signal with current state
        self.theme_changed.emit(is_dark)
        
    def update_nav_tree_icons(self, color):
        """Update all navigation tree icons with the specified color."""
        if not hasattr(self, 'nav_tree'):
            return
            
        # Function to update icons recursively
        def update_item_icons(item):
            # Get current icon name from the item's data
            icon = item.icon(0)
            if not icon.isNull():
                # If there's an icon, try to update it with the new color
                # We need to store icon names in the items to properly refresh them
                icon_name = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if icon_name:
                    item.setIcon(0, load_bootstrap_icon(icon_name, color))
                
            # Process child items recursively
            for i in range(item.childCount()):
                update_item_icons(item.child(i))
        
        # Process all top-level items
        root = self.nav_tree.invisibleRootItem()
        for i in range(root.childCount()):
            update_item_icons(root.child(i))

    def create_bottom_statusbar(self):
        self.bottom_statusbar = QToolBar()
        self.bottom_statusbar.setMovable(False)  # Fix the toolbar in place
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.bottom_statusbar)
        self.bottom_statusbar.setIconSize(QSize(22, 22))  # Consistent with top toolbar

        status_style = """
            QLabel {
                padding: 2px 8px;
                border-radius: 3px;
                margin: 0 2px;
            }
        """
        
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(4)
        
        self.ollama_status_label = QLabel("Model")
        self.ollama_status_label.setStyleSheet(status_style)
        status_layout.addWidget(self.ollama_status_label)
        self.update_status_indicator(self.ollama_status_label, False, "Ollama: Not Connected", "cpu")

        self.websocket_status_label = QLabel("Network")
        self.websocket_status_label.setStyleSheet(status_style)
        status_layout.addWidget(self.websocket_status_label)
        self.update_status_indicator(self.websocket_status_label, False, "Server: Not Connected", "hdd-network")

        self.database_status_label = QLabel("Database")
        self.database_status_label.setStyleSheet(status_style)
        status_layout.addWidget(self.database_status_label)
        self.update_status_indicator(self.database_status_label, False, "Database: Not Connected", "database")

        self.bottom_statusbar.addWidget(status_widget)

        # Add a spacer to push the following items to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bottom_statusbar.addWidget(spacer)
        
        # Common button style matching the top toolbar buttons
        button_style = """
            QPushButton {
                border: none;
                border-radius: 4px;
                padding: 6px;
                margin: 0 2px;
            }
            QPushButton:hover {
                background-color: rgba(128, 128, 128, 0.2);
            }
        """
        
        # Add the quiet mode toggle
        self.quiet_mode_toggle = QPushButton()
        self.quiet_mode_toggle.setIconSize(QSize(22, 22))
        self.apply_themed_icon(self.quiet_mode_toggle, "volume-mute")
        self.quiet_mode_toggle.setCheckable(True)
        self.quiet_mode_toggle.setChecked(False)
        self.quiet_mode_toggle.setToolTip("Toggle quiet mode to suppress non-essential notifications")
        self.quiet_mode_toggle.clicked.connect(self.toggle_app_quiet_mode)
        self.quiet_mode_toggle.setStyleSheet(button_style)
        self.bottom_statusbar.addWidget(self.quiet_mode_toggle)

        theme_btn = QPushButton()
        theme_btn.setIconSize(QSize(22, 22))
        self.apply_themed_icon(theme_btn, "palette")
        theme_btn.setToolTip("Change application theme")
        theme_btn.setStyleSheet(button_style)
        theme_btn.clicked.connect(self.toggle_theme)
        self.bottom_statusbar.addWidget(theme_btn)

        self.start_status_checkers()

    def update_status_indicator(self, indicator, is_connected, tooltip, icon_name):
        """Updates a status indicator in the status bar using bootstrap icons."""
        # Get current theme info
        is_dark = hasattr(self, 'current_theme_file') and "dark" in self.current_theme_file
        
        # Store the icon name and connection status as properties for theme refresh
        indicator.setProperty("icon_name", icon_name)
        indicator.setProperty("is_connected", is_connected)
        indicator.setProperty("tooltip", tooltip)
        
        if is_connected:
            # Use a softer green for dark theme
            color = "#4CAF50" if not is_dark else "#81C784"
            status_icon = load_bootstrap_icon(icon_name, color)
        else:
            # Use a softer red for dark theme
            color = "#F44336" if not is_dark else "#E57373"
            status_icon = load_bootstrap_icon(icon_name, color)
        
        # Convert QIcon to QPixmap for QLabel
        pixmap = status_icon.pixmap(QSize(16, 16))
        indicator.setPixmap(pixmap)
        indicator.setToolTip(tooltip)

    def start_status_checkers(self):
        self.ollama_worker = OllamaStatusWorker()
        self.couchdb_worker = CouchDBStatusWorker()
        
        self.ollama_thread = QThread()
        self.couchdb_thread = QThread()
        
        self.ollama_worker.moveToThread(self.ollama_thread)
        self.couchdb_worker.moveToThread(self.couchdb_thread)
        
        self.ollama_worker.finished.connect(lambda msg: self.update_status_indicator(
            self.ollama_status_label, True, f"Ollama: {msg}", "cpu"))
        self.ollama_worker.error.connect(lambda msg: self.update_status_indicator(
            self.ollama_status_label, False, f"Ollama: {msg}", "cpu"))
        
        self.couchdb_worker.status.connect(lambda status: self.update_status_indicator(
            self.database_status_label, status, "Database: Connected" if status else "Database: Not Connected", "database"))
        
        self.ollama_thread.started.connect(self.ollama_worker.run)
        self.couchdb_thread.started.connect(self.couchdb_worker.run)
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_all_statuses)
        self.status_timer.start(30000)
        
        self.check_all_statuses()

    def check_all_statuses(self):
        if not self.ollama_thread.isRunning():
            self.ollama_thread.start()
        if not self.couchdb_thread.isRunning():
            self.couchdb_thread.start()
        
        is_connected = self.websocket_connection is not None and self.websocket_connection.state == State.OPEN
        self.update_status_indicator(
            self.websocket_status_label,
            is_connected,
            "Server: Connected" if is_connected else "Server: Not Connected",
            "hdd-network"
        )

    def create_menu_bar(self):
        # Replace menubar with a toolbar at the top
        top_toolbar = QToolBar()
        top_toolbar.setMovable(False)
        top_toolbar.setFloatable(False)  # Prevent floating
        top_toolbar.setIconSize(QSize(26, 26))  # Increased icon size from 22 to 26
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, top_toolbar)
        self.top_toolbar = top_toolbar
        
        # Create a main wrapper widget to control all alignment
        main_wrapper = QWidget()
        main_layout = QHBoxLayout(main_wrapper)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create a left-side container for all navigation buttons
        left_container = QWidget()
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(5, 0, 0, 0)
        left_layout.setSpacing(4)  # Increased spacing between buttons
        left_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Common button style for all toolbar buttons
        button_style = """
            QPushButton {
                border: none;
                border-radius: 4px;
                padding: 8px;  /* Increased padding */
                margin: 0 3px;  /* Increased margin */
            }
            QPushButton:hover {
                background-color: rgba(128, 128, 128, 0.2);
            }
        """
        
        # Add the logo widget to the left container if it exists
        if hasattr(self, 'logo_widget'):
            left_layout.addWidget(self.logo_widget, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Agent button - keep this first as primary interaction
        if hasattr(self, 'agent_button'):
            self.agent_button.setIconSize(QSize(26, 26))  # Increased size
            self.apply_themed_icon(self.agent_button, "robot")
            self.agent_button.setStyleSheet(button_style)
            left_layout.addWidget(self.agent_button, 0, Qt.AlignmentFlag.AlignLeft)
        
        # LLM button - moved up next to Agent as they're related
        llm_btn = QPushButton()
        llm_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(llm_btn, "cpu")
        llm_btn.setToolTip("LLM Manager")
        llm_btn.setStyleSheet(button_style)
        llm_btn.clicked.connect(self.show_llm_manager_section)
        left_layout.addWidget(llm_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Exchange button
        if hasattr(self, 'exchange_button'):
            self.exchange_button.setIconSize(QSize(26, 26))  # Increased size
            self.apply_themed_icon(self.exchange_button, "arrow-left-right")
            self.exchange_button.setStyleSheet(button_style)
            left_layout.addWidget(self.exchange_button, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Network button - related to exchange
        network_btn = QPushButton()
        network_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(network_btn, "globe")
        network_btn.setToolTip("Network")
        network_btn.clicked.connect(self.show_network_section)
        network_btn.setStyleSheet(button_style)
        left_layout.addWidget(network_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Database button - related to data storage
        database_btn = QPushButton()
        database_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(database_btn, "database")
        database_btn.setToolTip("View Database")
        database_btn.clicked.connect(self.show_database_section)
        database_btn.setStyleSheet(button_style)
        left_layout.addWidget(database_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Settings button - moved up for better access
        settings_btn = QPushButton()
        settings_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(settings_btn, "gear")
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self.show_settings_section)
        settings_btn.setStyleSheet(button_style)
        left_layout.addWidget(settings_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # BioNLP Annotation button - specialized tool
        bionlp_btn = QPushButton()
        bionlp_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(bionlp_btn, "heart-pulse")  # Medical-related icon
        bionlp_btn.setToolTip("Biomedical Annotation Tool")
        bionlp_btn.setStyleSheet(button_style)
        bionlp_btn.clicked.connect(self.show_bionlp_annotation_tool)
        left_layout.addWidget(bionlp_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Teams button - collaboration
        teams_btn = QPushButton()
        teams_btn.setIconSize(QSize(26, 26))  # Increased size
        self.apply_themed_icon(teams_btn, "people")
        teams_btn.setToolTip("Teams")
        teams_btn.clicked.connect(self.show_team_section)
        teams_btn.setStyleSheet(button_style)
        left_layout.addWidget(teams_btn, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Fix the size policy for the left container
        left_container.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Add the left container to the main layout
        main_layout.addWidget(left_container, 0, Qt.AlignmentFlag.AlignLeft)
        
        # Create a spacer widget to push remaining items to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(spacer)
        
        # Create a right-side container for user info
        right_container = QWidget()
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 5, 0)
        right_layout.setSpacing(2)
        
        # User info label on right
        if hasattr(self, 'user_info_label'):
            self.user_info_label.setStyleSheet("""
                margin-right: 6px;
                padding: 0px 4px;
            """)
            right_layout.addWidget(self.user_info_label)
        
        # Login button on right
        if hasattr(self, 'login_button'):
            self.login_button.setIconSize(QSize(22, 22))
            self.apply_themed_icon(self.login_button, "box-arrow-right")
            self.login_button.setStyleSheet(button_style)
            right_layout.addWidget(self.login_button)
        
        # Add the right container to the main layout
        main_layout.addWidget(right_container, 0, Qt.AlignmentFlag.AlignRight)
        
        # Add the main wrapper to the toolbar
        top_toolbar.addWidget(main_wrapper)

    def get_current_theme(self):
        return self.current_theme

    def show_projects_section(self):
        self.main_content_widget.setCurrentWidget(self.projects_section)

    def show_literature_search_section(self):
        self.main_content_widget.setCurrentWidget(self.literature_search_section)

    def show_literature_ranking_section(self):
        self.main_content_widget.setCurrentWidget(self.literature_ranking_section)

    def show_literature_evidence_section(self):
        self.main_content_widget.setCurrentWidget(self.literature_evidence_section)

    def show_protocol_section(self):
        """Show the protocol section with any generated protocol sections from the study plan."""
        # Check if study_plan_section has generated protocol sections and apply them
        if (hasattr(self, 'study_plan_section') and 
            hasattr(self.study_plan_section, 'generated_protocol_sections') and 
            self.study_plan_section.generated_protocol_sections):
            
            # Set the protocol section reference in study_plan_section if not set
            if not hasattr(self.study_plan_section, 'protocol_section') or not self.study_plan_section.protocol_section:
                self.study_plan_section.protocol_section = self.protocol_section
            
            # Set study type in protocol_section if available from study_plan_section
            if hasattr(self.study_plan_section, 'protocol_study_type') and self.study_plan_section.protocol_study_type:
                self.protocol_section.study_type = self.study_plan_section.protocol_study_type
                
            # Apply the generated sections
            self.study_plan_section._apply_generated_protocol_sections()
            
        self.main_content_widget.setCurrentWidget(self.protocol_section)

    def show_hypotheses_section(self):
        self.main_content_widget.setCurrentWidget(self.hypotheses_section)

    def show_data_collection_section(self):
        self.main_content_widget.setCurrentWidget(self.data_collection_widget)

    def show_data_reshaping_section(self):
        self.main_content_widget.setCurrentWidget(self.data_reshape_widget)

    def show_data_cleaning_section(self):
        self.main_content_widget.setCurrentWidget(self.data_cleaning_widget)

    def show_data_filtering_section(self):
        self.main_content_widget.setCurrentWidget(self.data_filtering_widget)

    def show_data_joining_section(self):
        self.main_content_widget.setCurrentWidget(self.data_join_widget)

    def show_data_testing_section(self):
        self.main_content_widget.setCurrentWidget(self.data_testing_widget)

    def show_analysis_interpretation_section(self):
        self.main_content_widget.setCurrentWidget(self.statistical_interpretation_widget)

    def show_analysis_evaluation_section(self):
        self.main_content_widget.setCurrentWidget(self.test_evaluation_widget)

    def show_analysis_assumptions_section(self):
        self.main_content_widget.setCurrentWidget(self.display_assumptions_widget)

    def show_analysis_subgroup_section(self):
        self.main_content_widget.setCurrentWidget(self.subgroup_analysis_widget)

    def show_analysis_mediation_section(self):
        self.main_content_widget.setCurrentWidget(self.mediation_analysis_widget)

    def show_analysis_sensitivity_section(self):
        self.main_content_widget.setCurrentWidget(self.sensitivity_analysis_widget)

    def show_evidence_section(self):
        self.main_content_widget.setCurrentWidget(self.evidence_section)

    def show_validator_management_section(self):
        self.main_content_widget.setCurrentWidget(self.validator_management_section)

    def show_planning_section(self):
        self.main_content_widget.setCurrentWidget(self.study_plan_section)

    def show_study_design_section(self):
        self.main_content_widget.setCurrentWidget(self.study_model_builder)

    def show_participant_management_section(self):
        self.main_content_widget.setCurrentWidget(self.participant_management_section)

    def show_study_documentation_section(self):
        self.main_content_widget.setCurrentWidget(self.study_design_section)

    def show_session_manager_section(self):
        # Refresh the studies list before showing the section to ensure it's up to date
        if hasattr(self, 'session_manager_section') and self.session_manager_section:
            # Set the user access in the studies widget
            if hasattr(self.session_manager_section, 'set_user_access') and hasattr(self, 'user_access'):
                self.session_manager_section.set_user_access(self.user_access)
            
            self.session_manager_section.refresh_projects_list()
        self.main_content_widget.setCurrentWidget(self.session_manager_section)

    def show_settings_section(self):
        self.main_content_widget.setCurrentWidget(self.settings_section)

    def show_team_section(self):
        self.main_content_widget.setCurrentWidget(self.team_section)

    def show_llm_manager_section(self):
        self.main_content_widget.setCurrentWidget(self.llm_manager_section)

    def show_database_section(self):
        self.main_content_widget.setCurrentWidget(self.database_section)

    def show_network_section(self):
        self.main_content_widget.setCurrentWidget(self.network_section)

    def show_agent_section(self):
        self.main_content_widget.setCurrentWidget(self.agent_section)

    def show_hypothesis_generator_section(self):
        """Show the hypothesis generator section."""
        self.main_content_widget.setCurrentWidget(self.hypothesis_generator_section)

    def toggle_theme(self):
        """Toggle between light and dark theme."""
        # Open the theme selector dialog
        theme_dialog = ThemeSelector(self)
        theme_dialog.setWindowTitle("Select Theme")
        if theme_dialog.exec():
            selected_theme = theme_dialog.selected_theme
            theme_name = selected_theme["name"]
            theme_file = selected_theme["file"]
            invert_secondary = selected_theme.get("invert_secondary", False)
            
            # Save the theme preference
            save_theme_preference(theme_name, theme_file, invert_secondary)
            
            # Update the current theme
            self.current_theme = theme_name
            self.current_theme_file = theme_file
            self.invert_secondary = invert_secondary
            
            # Apply the new theme
            self.apply_theme()

    def closeEvent(self, event):
        """Clean up resources when window is closed."""
        # Stop the server if it's running
        if hasattr(self, 'network_section'):
            self.network_section.stop_server()
            
        # Close websocket connection if open
        if hasattr(self, 'websocket_connection') and self.websocket_connection:
            try:
                async def close_websocket():
                    await self.websocket_connection.close()
                    
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(close_websocket())
                else:
                    loop.run_until_complete(close_websocket())
            except Exception:
                pass  # Silently handle any errors during cleanup
                
        # Accept the close event
        event.accept()

    def show_login_dialog(self):
        if hasattr(self, 'user_access') and self.user_access.current_user:
            logout_dialog = QMessageBox(self)
            logout_dialog.setWindowTitle("Logout")
            logout_dialog.setText("Are you sure you want to logout?")
            logout_dialog.setIcon(QMessageBox.Icon.Question)
            
            # Create custom icon using bootstrap
            icon_label = QLabel(logout_dialog)
            icon_label.setPixmap(load_bootstrap_icon("box-arrow-right", "#f44336").pixmap(32, 32))
            logout_dialog.setIconPixmap(icon_label.pixmap())
            
            logout_dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            logout_dialog.setDefaultButton(QMessageBox.StandardButton.No)
            
            if logout_dialog.exec() == QMessageBox.StandardButton.Yes:
                self.user_access.logout()
                # Update user display
                self.update_user_display()
        else:
            dialog = LoginDialog(self.user_access)
            dialog.loginSuccessful.connect(self.on_login_successful)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.update_user_display()

    def on_login_successful(self, user_data):
        self.user_access.current_user = user_data
        self.update_user_display()
        
        # Update user in studies widget if it exists
        if hasattr(self, 'session_manager_section') and self.session_manager_section:
            if hasattr(self.session_manager_section, 'set_user_access'):
                self.session_manager_section.set_user_access(self.user_access)

    def update_user_display(self):
        # Get current theme info
        is_dark = "dark" in self.current_theme_file if hasattr(self, 'current_theme_file') else False
        
        # Common button style matching the toolbar buttons
        button_style = """
            QPushButton {
                border: none;
                border-radius: 4px;
                padding: 6px;
                margin: 0 2px;
            }
            QPushButton:hover {
                background-color: rgba(128, 128, 128, 0.2);
            }
        """
        
        # Apply consistent styling to the buttons
        if hasattr(self, 'agent_button'):
            self.agent_button.setStyleSheet(button_style)
        if hasattr(self, 'exchange_button'):
            self.exchange_button.setStyleSheet(button_style)  
        if hasattr(self, 'login_button'):
            self.login_button.setStyleSheet(button_style)
        
        if hasattr(self, 'user_access') and self.user_access.current_user:
            user_data = self.user_access.current_user
            self.user_info_label.setText(f"{user_data.get('name', user_data['email'])}")
            # Style without explicit text color
            self.user_info_label.setStyleSheet("""
                margin-right: 6px;
                padding: 0px 4px;
            """)
            self.apply_themed_icon(self.login_button, "box-arrow-right")
            self.login_button.setToolTip("Logout")
            is_admin = user_data.get('is_admin', False)
            if hasattr(self, 'team_section'):
                self.team_section.reset_password_button.setEnabled(is_admin)
        else:
            self.user_info_label.setText("Not logged in")
            # Use opacity for "not logged in" state instead of explicit gray color
            self.user_info_label.setStyleSheet("""
                margin-right: 6px;
                padding: 0px 4px;
                opacity: 0.7;
            """)
            self.apply_themed_icon(self.login_button, "box-arrow-in-right")
            self.login_button.setToolTip("Login")
            if hasattr(self, 'team_section'):
                self.team_section.reset_password_button.setEnabled(False)

    async def subscribe_to_system_topic(self):
        """Subscribe to the system topic on the WebSocket server."""
        if self.websocket_connection:
            message = {
                "action": "subscribe",
                "payload": {"topic": ConnectionManager.SYSTEM_TOPIC}
            }
            await self.websocket_connection.send(json.dumps(message))
            self.network_section.output_text.append("Subscribed to system topic.")

    async def subscribe_host_to_topic(self, topic):
        """Subscribe the host to a specific topic."""
        if self.websocket_connection:
            message = {
                "action": "subscribe",
                "payload": {"topic": topic}
            }
            await self.websocket_connection.send(json.dumps(message))
            self.network_section.output_text.append(f"Host subscribed to topic: {topic}")

    async def unsubscribe_host_from_topic(self, topic):
        """Unsubscribe the host from a specific topic."""
        if self.websocket_connection:
            message = {
                "action": "unsubscribe",
                "payload": {"topic": topic}
            }
            await self.websocket_connection.send(json.dumps(message))
            self.network_section.output_text.append(f"Host unsubscribed from topic: {topic}")

    def show_data_collection(self):
        """Show the data collection section."""
        self.main_content_widget.setCurrentWidget(self.data_collection_widget)

    def show_data_reshape(self):
        """Show the data reshape section."""
        self.main_content_widget.setCurrentWidget(self.data_reshape_widget)

    def show_data_cleaning(self):
        """Show the data cleaning section."""
        self.main_content_widget.setCurrentWidget(self.data_cleaning_widget)

    def show_data_filtering(self):
        """Show the data filtering section."""
        self.main_content_widget.setCurrentWidget(self.data_filtering_widget)

    def show_data_testing(self):
        """Show the data testing section."""
        self.main_content_widget.setCurrentWidget(self.data_testing_widget)

    @asyncSlot()
    async def handle_agent_file_upload(self, source_type, connection_info, file_name, data_source):
        """Handle file upload requests from the agent interface."""
        try:
            # Load the data from the source
            df = await data_source.load_data()
            
            # Create source connection
            source_connection = SourceConnection(source_type, connection_info, file_name)
            
            # Add the source to the collection widget
            self.data_collection_widget.add_source(file_name, source_connection, df)
            
            # Show a success message
            QMessageBox.information(
                self, 
                "Upload Success", 
                f"The agent has successfully uploaded '{file_name}' to the data collection."
            )
            
            # Switch to the data collection view to show the new data
            self.show_data_collection()
        except Exception as e:
            # Show error message if something goes wrong
            QMessageBox.critical(
                self, 
                "Upload Error", 
                f"Failed to upload file through agent: {str(e)}"
            )

    def toggle_app_quiet_mode(self):
        """Toggle the application-wide quiet mode setting."""
        self.app_quiet_mode = self.quiet_mode_toggle.isChecked()
        
        # Update the button icon/tooltip based on the state
        if self.app_quiet_mode:
            self.apply_themed_icon(self.quiet_mode_toggle, "volume-mute-fill")
            self.quiet_mode_toggle.setToolTip("Quiet Mode: On - Click to restore notifications")
        else:
            self.apply_themed_icon(self.quiet_mode_toggle, "volume-up")
            self.quiet_mode_toggle.setToolTip("Quiet Mode: Off - Click to suppress notifications")
        
        # Propagate quiet mode setting to relevant widgets
        if hasattr(self, 'data_testing_widget'):
            self.data_testing_widget._quiet_mode = self.app_quiet_mode
        
        # Show brief notification instead of dialog box
        status_bar = self.statusBar()
        if status_bar:
            if self.app_quiet_mode:
                status_bar.showMessage("Quiet mode enabled. Non-essential notifications suppressed.", 3000)
            else:
                status_bar.showMessage("Quiet mode disabled. Normal notifications restored.", 3000)

    def apply_themed_icon(self, widget, icon_name, size=None):
        """Apply a themed icon to a widget based on current theme.
        
        Args:
            widget: The widget to apply the icon to (must have setIcon method)
            icon_name: The name of the bootstrap icon
            size: Optional icon size (default is widget's iconSize if available)
        """
        if not hasattr(widget, 'setIcon'):
            return False
            
        # Determine current theme color
        is_dark = "dark" in self.current_theme_file if hasattr(self, 'current_theme_file') else False
        text_color = "#FFFFFF" if is_dark else "#212121"
        
        # Store the icon name for later refreshing
        widget.setProperty("icon_name", icon_name)
        
        # Get the widget's icon size if available
        if size is None and hasattr(widget, 'iconSize'):
            size = max(widget.iconSize().width(), widget.iconSize().height())
            
        # Load the icon with theme color
        icon = load_bootstrap_icon(icon_name, text_color, size)
        
        # Apply the icon directly
        widget.setIcon(icon)
        
        return True

    def update_all_action_icons(self, widget=None, processed=None):
        """Recursively update all QAction and QPushButton icons in the UI.
        
        Args:
            widget: The widget to start search from (default is self)
            processed: Set to track already processed widgets
        """
        if processed is None:
            processed = set()
            
        if widget is None:
            widget = self
            
        if id(widget) in processed:
            return
            
        processed.add(id(widget))
        
        # Get current theme info
        is_dark = "dark" in self.current_theme_file if hasattr(self, 'current_theme_file') else False
        text_color = "#FFFFFF" if is_dark else "#212121"
        
        # Process QMenu and QToolBar actions
        if isinstance(widget, QMenu) or isinstance(widget, QToolBar):
            for action in widget.actions():
                if action.icon():
                    # Check if we've stored the icon name property
                    icon_name = action.property("icon_name")
                    if icon_name:
                        # Update with the stored icon name
                        action.setIcon(load_bootstrap_icon(icon_name, text_color))
        
        # Specifically handle QPushButton
        if isinstance(widget, QPushButton):
            # Check if button has an icon and stored icon_name
            if not widget.icon().isNull():
                icon_name = widget.property("icon_name")
                if icon_name:
                    size = max(widget.iconSize().width(), widget.iconSize().height())
                    new_icon = load_bootstrap_icon(icon_name, text_color, size)
                    widget.setIcon(QIcon(new_icon.pixmap(size, size)))
        
        # Recursively process children
        for child in widget.findChildren(QObject):
            # Only process widgets
            if isinstance(child, QWidget) or isinstance(child, QMenu) or isinstance(child, QToolBar):
                self.update_all_action_icons(child, processed)
    
    def refresh_all_icons(self):
        """Refresh all icons in the application with appropriate theme colors."""
        # Get theme info
        is_dark = "dark" in self.current_theme_file if hasattr(self, 'current_theme_file') else False
        text_color = getattr(self, 'current_icon_color', "#FFFFFF" if is_dark else "#212121")
        
        # Recursively update all widgets with icons
        self._refresh_widget_icons(self, text_color)
        
        # Update all menu actions
        self.update_all_action_icons()
        
        # Force an immediate repaint of the UI to show the changes
        self.repaint()
        QApplication.processEvents()
        
    def _refresh_widget_icons(self, parent_widget, text_color, processed=None):
        """Recursively update icons for all widgets."""
        if processed is None:
            processed = set()
            
        if id(parent_widget) in processed:
            return
            
        processed.add(id(parent_widget))
        
        # Process this widget if it has an icon
        if hasattr(parent_widget, 'icon') and callable(parent_widget.icon) and not parent_widget.icon().isNull():
            icon_name = parent_widget.property("icon_name")
            if icon_name:
                size = None
                if hasattr(parent_widget, 'iconSize'):
                    size = max(parent_widget.iconSize().width(), parent_widget.iconSize().height())
                icon = load_bootstrap_icon(icon_name, text_color, size)
                parent_widget.setIcon(icon)
        
        # Special handling for QMenu and QToolBar
        if isinstance(parent_widget, QMenu) or isinstance(parent_widget, QToolBar):
            for action in parent_widget.actions():
                if action.icon() and not action.icon().isNull():
                    icon_name = action.property("icon_name")
                    if icon_name:
                        action.setIcon(load_bootstrap_icon(icon_name, text_color))
        
        # If this is a QTreeWidget, process all top level items
        if isinstance(parent_widget, QTreeWidget):
            self.update_nav_tree_icons(text_color)
        
        # Process all child widgets
        for child in parent_widget.findChildren(QWidget):
            self._refresh_widget_icons(child, text_color, processed)
        
        # Process QActions directly owned by this widget
        if hasattr(parent_widget, 'actions'):
            for action in parent_widget.actions():
                if not action.icon().isNull():
                    icon_name = action.property("icon_name")
                    if icon_name:
                        action.setIcon(load_bootstrap_icon(icon_name, text_color))

    def handle_loader_node_click(self, section_name):
        """Handles clicks on loader nodes to close splash and navigate to selected section."""
        print(f"DEBUG: Node clicked: {section_name}")
        
        # Special cases for sub-sections that aren't being found
        special_cases = {
            'mediation': self.show_analysis_mediation_section,
            'evaluation': self.show_analysis_evaluation_section,
            'assumptions': self.show_analysis_assumptions_section,
            'subgroup': self.show_analysis_subgroup_section,
            'sensitivity': self.show_analysis_sensitivity_section,
            'interpret': self.show_analysis_interpretation_section,
            'sources': self.show_data_collection_section,
            'cleaning': self.show_data_cleaning_section,
            'reshaping': self.show_data_reshaping_section,
            'filtering': self.show_data_filtering_section,
            'joining': self.show_data_joining_section,
            'design': self.show_study_design_section,
            'documentation': self.show_study_documentation_section,
            'participants': self.show_participant_management_section,
            'protocol': self.show_protocol_section,
            'search': self.show_literature_search_section,
            'ranking': self.show_literature_ranking_section,
            'claims': self.show_literature_evidence_section,
            'validator': self.show_validator_management_section
        }
        
        if section_name in special_cases:
            print(f"DEBUG: Special case - forcing navigation to {section_name} section")
            self.pending_navigation_target = special_cases[section_name]
            self.close_splash_and_show_main()
            return
        
        # First, get the appropriate section method
        section_map = self.get_section_map()
        print(f"DEBUG: Available sections: {list(section_map.keys())}")
        print(f"DEBUG: Looking for section: '{section_name}'")
        
        # Add specific debug for section
        print(f"DEBUG: Section map contains '{section_name}': {section_name in section_map}")
        
        show_method = section_map.get(section_name)
        
        if show_method:
            print(f"DEBUG: Found navigation method for {section_name}: {show_method.__name__}")
            # Store the navigation target to execute after splash closes
            self.pending_navigation_target = show_method
        else:
            print(f"DEBUG: No navigation method found for '{section_name}'")
            self.pending_navigation_target = None
        
        # Close splash and show main window
        self.close_splash_and_show_main()

    def get_section_map(self):
        """Returns the mapping of section names to show methods."""
        return {
            # Main workflow sections
            "plan": self.show_planning_section,
            "hypotheses": self.show_hypotheses_section,
            "manage": self.show_session_manager_section, 
            "design": self.show_study_design_section,
            "documentation": self.show_study_documentation_section,
            "participants": self.show_participant_management_section,
            "protocol": self.show_protocol_section,
            "literature": self.show_literature_search_section,
            "search": self.show_literature_search_section,
            "ranking": self.show_literature_ranking_section,
            "claims": self.show_literature_evidence_section,
            "data": self.show_data_collection_section,
            "sources": self.show_data_collection_section,
            "cleaning": self.show_data_cleaning_section,
            "reshaping": self.show_data_reshaping_section,
            "filtering": self.show_data_filtering_section,
            "joining": self.show_data_joining_section,
            "analysis": self.show_data_testing_section,
            "evaluation": self.show_analysis_evaluation_section,
            "assumptions": self.show_analysis_assumptions_section,
            "subgroup": self.show_analysis_subgroup_section,
            "mediation": self.show_analysis_mediation_section,
            "sensitivity": self.show_analysis_sensitivity_section,
            "interpret": self.show_analysis_interpretation_section,
            "evidence": self.show_evidence_section,
            "validator": self.show_validator_management_section,
            
            # Tool sections
            "agent": self.show_agent_section,
            "network": self.show_network_section,
            "database": self.show_database_section,
            "team": self.show_team_section,
            "settings": self.show_settings_section,
        }
        
    def show_bionlp_annotation_tool(self):
        """Show the Biomedical Annotation Tool in the main content area"""
        self.main_content_widget.setCurrentWidget(self.bionlp_section)
        self.statusBar().showMessage("Biomedical Annotation Tool activated", 3000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Load theme preferences before creating splash
    theme_prefs = load_theme_preference()
    is_dark = theme_prefs["is_dark"]
    
    # Create the splash screen immediately
    splash_dialog = QDialog(None, Qt.WindowType.FramelessWindowHint)
    splash_dialog.setStyleSheet("background-color: transparent;")
    splash_layout = QVBoxLayout(splash_dialog)
    splash_layout.setContentsMargins(0, 0, 0, 0)
    
    # Create animated loader
    from splash_loader.splash_loader import ResearchFlowLoader
    loader = ResearchFlowLoader(load_time=15000)
    loader.set_theme(is_dark)  # Use theme from preferences
    splash_layout.addWidget(loader)
    
    # Show splash maximized immediately
    splash_dialog.showMaximized()
    app.processEvents()  # Force immediate processing of events to show the splash
    
    # Initialize MainWindow in the background after splash is visible
    def initialize_app():
        # Create the main window but don't show it yet
        main_window = MainWindow(show_immediately=False, existing_loader=loader, existing_splash=splash_dialog)
        
        # Connect Go button to close splash and show main window
        loader.goButtonClicked.connect(lambda: main_window.close_splash_and_show_main())
        
        # Start checking app readiness
        QTimer.singleShot(500, main_window.check_app_readiness)
    
    # Wait a tiny bit to ensure splash is visible, then start initialization
    QTimer.singleShot(100, initialize_app)
    
    with loop:
        loop.run_forever()


