import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QLabel, QLineEdit, QComboBox, QTextEdit, QFormLayout, 
    QGroupBox, QStackedWidget, QFrame, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QListWidget, QListWidgetItem,
    QMessageBox, QDialog, QDialogButtonBox, QSizePolicy, QStatusBar,
    QCheckBox, QTabWidget, QToolButton, QMenu, QInputDialog, QButtonGroup,
    QRadioButton
)
from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtCore import QSize
import asyncio
from llms.client import call_llm_async, call_llm_async_json
from qt_sections.llm_manager import llm_config
from qasync import asyncSlot
from helpers.load_icon import load_bootstrap_icon
from data.collection.collect import DataFrameDisplay
from study_model.studies_manager import StudiesManager

class DataPreprocessingWidget(QWidget):
    """Widget for preprocessing datasets to make them ready for statistical analysis"""
    
    dataset_updated = pyqtSignal(str, object)  # Signal emitted when a dataset is updated (name, dataframe)
    
    def __init__(self, studies_manager: StudiesManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Preprocessing")
        self.studies_manager = studies_manager # Store the studies manager instance
        
        # Store the current dataset and its original
        self.current_dataset_name = None
        self.current_dataset = None
        self.original_dataset = None
        
        # Store preprocessing steps and their status
        self.preprocessing_steps = []
        self.step_statuses = {}
        self.patient_id_column = None # Initialize patient ID column attribute
        self.grouping_variable = None # Initialize grouping variable attribute
        self.hypothesis = None # Initialize hypothesis attribute
        self.step_results = {} # <-- Add storage for results
        self.dataset_states = {} # <-- Add storage for dataset states
        
        # Initialize UI
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top toolbar with main actions
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side: Title with icon
        title_layout = QHBoxLayout()
        title_icon = QLabel()
        title_icon.setPixmap(load_bootstrap_icon("gear-fill").pixmap(20, 20))
        title_layout.addWidget(title_icon)
        
        title_label = QLabel("Data Preprocessing")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(title_label)
        toolbar_layout.addLayout(title_layout)
        
        toolbar_layout.addStretch()
        
        # Right side: Action buttons
        actions_frame = QFrame()
        actions_frame.setFrameShape(QFrame.Shape.Panel)
        actions_frame.setFrameShadow(QFrame.Shadow.Raised)
        actions_frame.setLineWidth(1)
        actions_frame.setStyleSheet("border: none;")
        actions_layout = QHBoxLayout(actions_frame)
        actions_layout.setSpacing(5)
        actions_layout.setContentsMargins(5, 2, 5, 2)
        
        # Refresh button (to refresh dataset list)
        refresh_button = QPushButton()
        refresh_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_button.setText("Refresh List") # Changed text
        refresh_button.setIconSize(QSize(18, 18))
        refresh_button.setToolTip("Refresh datasets from studies manager")
        refresh_button.clicked.connect(self.refresh_datasets) # Connects to updated refresh_datasets
        refresh_button.setFixedHeight(32)
        refresh_button.setMinimumWidth(110) # Adjusted width
        self.apply_icon_button_style(refresh_button)
        actions_layout.addWidget(refresh_button)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedHeight(24)
        actions_layout.addWidget(separator)
        
        # Auto Process button
        auto_process_button = QPushButton()
        auto_process_button.setIcon(load_bootstrap_icon("lightning-fill"))
        auto_process_button.setText("Auto Process")
        auto_process_button.setIconSize(QSize(18, 18))
        auto_process_button.setToolTip("Automatically apply all preprocessing steps")
        auto_process_button.clicked.connect(self.run_auto_process)
        auto_process_button.setFixedHeight(32)
        auto_process_button.setMinimumWidth(120)
        self.apply_icon_button_style(auto_process_button)
        actions_layout.addWidget(auto_process_button)
        
        # Save button
        save_button = QPushButton()
        save_button.setIcon(load_bootstrap_icon("save-fill"))
        save_button.setText("Save Processed")
        save_button.setIconSize(QSize(18, 18))
        save_button.setToolTip("Save the processed dataset")
        save_button.clicked.connect(self.save_processed_dataset)
        save_button.setFixedHeight(32)
        save_button.setMinimumWidth(120)
        self.apply_icon_button_style(save_button)
        actions_layout.addWidget(save_button)
        
        toolbar_layout.addWidget(actions_frame)
        main_layout.addLayout(toolbar_layout)
        
        # Add separator line below toolbar
        header_separator = QFrame()
        header_separator.setFrameShape(QFrame.Shape.HLine)
        header_separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(header_separator)
        
        # Split view with timeline on left and dataset display on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Now includes datasets list and steps
        left_panel = QWidget()
        left_panel.setMinimumWidth(300) # Increased width slightly
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5) # Add spacing between groups
        
        # --- New Dataset List Panel ---
        datasets_group = QGroupBox("Available Datasets")
        datasets_layout = QVBoxLayout(datasets_group)
        
        # Search box
        search_layout = QHBoxLayout()
        search_icon = QLabel()
        search_icon.setPixmap(load_bootstrap_icon("search").pixmap(16, 16))
        search_layout.addWidget(search_icon)
        self.search_datasets = QLineEdit()
        self.search_datasets.setPlaceholderText("Search datasets...")
        self.search_datasets.textChanged.connect(self.filter_datasets)
        search_layout.addWidget(self.search_datasets)
        datasets_layout.addLayout(search_layout)
        
        # Datasets list
        self.dataset_list = QListWidget()
        self.dataset_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.dataset_list.itemClicked.connect(self.on_dataset_selected)
        self.dataset_list.setIconSize(QSize(20, 20))
        datasets_layout.addWidget(self.dataset_list)
        
        left_layout.addWidget(datasets_group)
        # --- End New Dataset List Panel ---
        
        # --- Add Hypothesis Input Group ---
        hypothesis_group = QGroupBox("Study Hypothesis (Optional)")
        hypothesis_layout = QVBoxLayout(hypothesis_group)
        
        # Instructions label
        hypothesis_label = QLabel("Enter your research hypothesis to help guide the preprocessing steps:")
        hypothesis_label.setWordWrap(True)
        hypothesis_layout.addWidget(hypothesis_label)
        
        # Hypothesis text input
        self.hypothesis_input = QTextEdit()
        self.hypothesis_input.setPlaceholderText("Example: 'Patients treated with medication X show lower blood pressure than those given placebo.'")
        self.hypothesis_input.setMaximumHeight(80)
        self.hypothesis_input.textChanged.connect(self.on_hypothesis_changed)
        hypothesis_layout.addWidget(self.hypothesis_input)
        
        left_layout.addWidget(hypothesis_group)
        # --- End Hypothesis Input Group ---
        
        # Steps list with timeline visualization
        steps_group = QGroupBox("Preprocessing Steps")
        steps_layout = QVBoxLayout(steps_group)
        
        # Create timeline UI
        self.steps_list = QListWidget()
        self.steps_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.steps_list.itemClicked.connect(self.on_step_selected)
        steps_layout.addWidget(self.steps_list)
        
        left_layout.addWidget(steps_group)
        
        # Current step details
        self.step_details = QGroupBox("Step Details")
        step_details_layout = QVBoxLayout(self.step_details)
        
        self.step_title = QLabel("Select a step")
        self.step_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        step_details_layout.addWidget(self.step_title)
        
        self.step_description = QLabel("")
        self.step_description.setWordWrap(True)
        step_details_layout.addWidget(self.step_description)
        
        # Step status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.step_status = QLabel("Not Started")
        status_layout.addWidget(self.step_status)
        status_layout.addStretch()
        step_details_layout.addLayout(status_layout)
        
        # Step action button
        self.step_action_button = QPushButton("Run Step")
        self.step_action_button.setEnabled(False)
        self.step_action_button.clicked.connect(self.run_current_step)
        step_details_layout.addWidget(self.step_action_button)
        
        left_layout.addWidget(self.step_details)
        
        # Right panel - dataset display
        right_panel = QWidget()
        self.setup_dataset_display(right_panel)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650]) # Adjust sizes
        
        # Make splitter take maximum available space
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        main_layout.addWidget(splitter, 1)
        
        # Add status bar at bottom
        self.status_bar = QStatusBar()
        self.status_bar.setMaximumHeight(30)
        main_layout.addWidget(self.status_bar)
        
        # Populate dataset list initially
        self.populate_dataset_list()
        
        # Add preprocessing steps to the timeline
        self.add_preprocessing_steps()
    
    def setup_dataset_display(self, parent):
        """Setup the dataset display panel"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Dataset display header
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.Shape.StyledPanel)
        header_frame.setFrameShadow(QFrame.Shadow.Raised)
        header_frame.setLineWidth(1)
        header_frame.setStyleSheet("border: none;")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 3, 8, 3)
        header_layout.setSpacing(8)
        
        header_frame.setFixedHeight(45)
        
        # Dataset info section
        info_section = QHBoxLayout()
        info_section.setSpacing(10)
        
        # Dataset title with icon
        icon = load_bootstrap_icon("table")
        icon_pixmap = icon.pixmap(16, 16)
        
        dataset_icon = QLabel()
        dataset_icon.setPixmap(icon_pixmap)
        info_section.addWidget(dataset_icon)
        
        self.current_dataset_label = QLabel("No dataset loaded")
        self.current_dataset_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        info_section.addWidget(self.current_dataset_label)
        
        # Dataset info
        self.dataset_info_label = QLabel("")
        info_section.addWidget(self.dataset_info_label)
        
        # Preprocessing status
        self.preprocessing_status_label = QLabel("")
        info_section.addWidget(self.preprocessing_status_label)
        info_section.addStretch()
        
        header_layout.addLayout(info_section)
        
        # View options
        view_section = QHBoxLayout()
        
        # Toggle between original and processed
        self.view_toggle = QComboBox()
        self.view_toggle.addItems(["Current (Processed)", "Original"])
        self.view_toggle.setCurrentIndex(0)
        self.view_toggle.currentIndexChanged.connect(self.toggle_dataset_view)
        view_section.addWidget(self.view_toggle)
        
        header_layout.addLayout(view_section)
        
        layout.addWidget(header_frame)
        
        # Create dataset overview panel
        self.dataset_overview = QFrame()
        self.dataset_overview.setFrameShape(QFrame.Shape.StyledPanel)
        self.dataset_overview.setFrameShadow(QFrame.Shadow.Sunken)
        self.dataset_overview.setStyleSheet("QFrame { border: none; } .stat-value { font-weight: bold; }")
        overview_layout = QHBoxLayout(self.dataset_overview)
        overview_layout.setContentsMargins(10, 5, 10, 5)
        
        # Stats layout
        self.overview_stats = QHBoxLayout()
        self.overview_stats.setSpacing(15)
        
        overview_layout.addLayout(self.overview_stats)
        
        self.dataset_overview.setFixedHeight(80)
        self.dataset_overview.setVisible(False)
        
        layout.addWidget(self.dataset_overview)
        
        # Dataset display tabs
        self.display_tabs = QTabWidget()
        
        # Data tab
        self.dataset_display = DataFrameDisplay()
        self.display_tabs.addTab(self.dataset_display, "Data")
        
        # Analysis tab (will show preprocessing analysis)
        self.analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_widget)
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        analysis_layout.addWidget(self.analysis_text)
        self.display_tabs.addTab(self.analysis_widget, "Analysis")
        
        layout.addWidget(self.display_tabs, 1)

    def apply_icon_button_style(self, button):
        """Apply consistent style to icon buttons"""
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """)

    def add_preprocessing_steps(self):
        """Add the preprocessing steps to the timeline"""
        # Define the preprocessing steps
        steps = [
            {
                "id": "identify_patient_id",
                "title": "Step 1: Identify Patient Identifier",
                "description": "Identify columns that uniquely identify patients in the dataset.",
                "icon": "person-badge"
            },
            {
                "id": "normalize_rows",
                "title": "Step 2: Normalize Dataset",
                "description": "Ensure there is one row per patient in the dataset.",
                "icon": "arrow-down-up"
            },
            {
                "id": "check_grouper",
                "title": "Step 3: Check Grouping Variable",
                "description": "Verify dataset has a grouping variable for hypothesis testing.",
                "icon": "collection"
            },
            {
                "id": "validate_columns",
                "title": "Step 4: Validate Columns for Analysis",
                "description": "Ensure columns are properly formatted for statistical analysis.",
                "icon": "check-circle"
            },
            {
                "id": "final_validation",
                "title": "Step 5: Final Validation",
                "description": "Check balance of groups and overall dataset readiness.",
                "icon": "clipboard-check"
            }
        ]
        
        # Store steps
        self.preprocessing_steps = steps
        
        # Initialize step statuses
        for step in steps:
            self.step_statuses[step["id"]] = "Not Started"
        
        # Add steps to list with timeline visualization
        for i, step in enumerate(steps):
            item = QListWidgetItem(step["title"])
            item.setData(Qt.ItemDataRole.UserRole, step["id"])
            
            # Set icon
            icon = load_bootstrap_icon(step["icon"])
            item.setIcon(icon)
            
            # Style based on status (initially all not started)
            item.setForeground(QColor("#6c757d"))  # Gray for not started
            
            self.steps_list.addItem(item)

    def on_step_selected(self, item):
        """Handle step selection in the timeline. Displays stored results if available."""
        if not item: # Handle case where selection is cleared
            return

        step_id = item.data(Qt.ItemDataRole.UserRole)
        step_index = self.steps_list.row(item)
        step = next((s for s in self.preprocessing_steps if s["id"] == step_id), None)
        if not step:
            return

        # Update step details panel (always happens)
        self.step_title.setText(step["title"])
        self.step_description.setText(step["description"])
        current_status = self.step_statuses.get(step_id, "Not Started")
        self.step_status.setText(current_status)

        # Check if results are stored for this step
        if step_id in self.step_results:
            # Display stored results
            stored_result = self.step_results[step_id]
            stored_df = stored_result.get("dataframe")
            stored_analysis = stored_result.get("analysis", "Analysis results not found for this step.")

            if stored_df is not None:
                self.display_dataset(stored_df) # Show stored dataframe
                self.analysis_text.setText(stored_analysis) # Show stored analysis
                self.view_toggle.setEnabled(False) # Disable toggle for historical view
                self.current_dataset_label.setText(f"Dataset: {self.current_dataset_name} (Snapshot after Step {step_index + 1})")
                self.display_tabs.setCurrentIndex(1) # Show analysis tab by default when viewing history
            else:
                # Handle case where stored data is somehow invalid
                self.analysis_text.setText(f"Error: Could not load stored data for Step {step_index + 1}.")
                self.view_toggle.setEnabled(True) # Re-enable toggle
                self.view_toggle.setCurrentIndex(0)
                self.display_dataset(self.current_dataset) # Show current dataset
                self.current_dataset_label.setText(f"Dataset: {self.current_dataset_name}")

            # Disable run button when viewing history of a completed step
            self.step_action_button.setEnabled(False)

        else:
            # No stored results - show current state
            if self.current_dataset is not None:
                 # Display the latest processed dataset
                 self.display_dataset(self.current_dataset)
                 self.view_toggle.setEnabled(True) # Enable toggle
                 self.view_toggle.setCurrentIndex(0) # Ensure "Current" view is selected
                 self.current_dataset_label.setText(f"Dataset: {self.current_dataset_name}")
                 self.analysis_text.setText("(Analysis results will appear here after running the step)")

                 # Enable run button only if dataset loaded and step not completed/in progress
                 can_run = (current_status != "Completed" and current_status != "In Progress")
                 self.step_action_button.setEnabled(can_run)

            else:
                 # No dataset loaded
                 self.dataset_display.clear()
                 self.dataset_overview.setVisible(False)
                 self.analysis_text.clear()
                 self.view_toggle.setEnabled(False)
                 self.current_dataset_label.setText("No dataset loaded")
                 self.step_action_button.setEnabled(False)

    def populate_dataset_list(self):
        """Populate the list widget with available datasets"""
        self.dataset_list.clear()
        if not self.studies_manager:
            self.update_status("Error: Studies manager not available.")
            return

        # Use the correct method: get_datasets_from_active_study()
        datasets = self.studies_manager.get_datasets_from_active_study()
        if not datasets:
            # Add a placeholder item if no datasets are available
            placeholder_item = QListWidgetItem("No datasets in active study")
            placeholder_item.setForeground(QColor("#6c757d")) # Gray text
            placeholder_item.setFlags(Qt.ItemFlag.NoItemFlags) # Make it non-selectable
            self.dataset_list.addItem(placeholder_item)
            self.update_status("No datasets found in the active study.")
            return

        # Iterate through the list of (name, dataframe) tuples
        for name, dataframe in datasets:
            if name and isinstance(dataframe, pd.DataFrame): # Check if dataframe is valid
                item = QListWidgetItem(name)
                try:
                    # Set icon - use try-except for robustness
                    icon = load_bootstrap_icon("table")
                    item.setIcon(icon)
                except Exception as e:
                    print(f"Warning: Could not load icon for dataset list: {e}")
                self.dataset_list.addItem(item)

        self.update_status(f"Found {len(datasets)} datasets in active study.")
        # Apply initial filter if search box has text
        self.filter_datasets(self.search_datasets.text())

    def filter_datasets(self, text):
        """Filter the dataset list based on search text"""
        text = text.lower().strip()
        has_visible_items = False
        for i in range(self.dataset_list.count()):
            item = self.dataset_list.item(i)
            # Skip non-selectable items (like placeholders)
            if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
                 item.setHidden(True) # Always hide placeholders during search
                 continue

            is_match = text in item.text().lower()
            item.setHidden(not is_match)
            if is_match:
                has_visible_items = True

        # Optionally, show a message if no datasets match the filter
        # (Could add a label below the list for this)

    def on_dataset_selected(self, item):
        """Handle selection of a dataset from the list"""
        # Ignore selection if it's a placeholder item
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return

        selected_name = item.text()
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Studies manager is not available.")
            return

        # Get all datasets from the active study
        all_datasets = self.studies_manager.get_datasets_from_active_study()
        df = None
        # Find the dataframe with the matching name
        for name, dataframe in all_datasets:
            if name == selected_name:
                df = dataframe
                break # Found the dataset

        if df is None or not isinstance(df, pd.DataFrame):
            QMessageBox.warning(self, "Error", f"Could not load dataset: {selected_name}")
            self.update_status(f"Error loading dataset: {selected_name}")
            return

        # Load the dataset data into the preprocessing view
        self.load_dataset_data(selected_name, df)

    def save_current_dataset_state(self):
        """Saves the current preprocessing state for the active dataset."""
        if self.current_dataset_name:
            print(f"[DEBUG] Saving state for dataset: {self.current_dataset_name}")
            self.dataset_states[self.current_dataset_name] = {
                'step_statuses': self.step_statuses.copy(),
                'step_results': self.step_results.copy(),
                'patient_id_column': self.patient_id_column,
                'grouping_variable': self.grouping_variable,
                'hypothesis': self.hypothesis,
                # Store the actual current dataframe state as well, needed if steps modify it
                'current_dataframe': self.current_dataset.copy() if self.current_dataset is not None else None
            }

    def load_dataset_data(self, name, dataframe):
        """Load dataset data into the widget, preserving state."""
        print(f"[DEBUG] Switching dataset. Current: {self.current_dataset_name}, New: {name}")

        # 1. Save the state of the currently loaded dataset (if any)
        self.save_current_dataset_state()

        # 2. Set the new dataset details
        self.current_dataset_name = name
        self.original_dataset = dataframe.copy() # Keep original reference

        # 3. Load state for the new dataset if it exists, otherwise initialize
        if name in self.dataset_states:
            print(f"[DEBUG] Loading existing state for dataset: {name}")
            state = self.dataset_states[name]
            self.step_statuses = state.get('step_statuses', {}).copy()
            self.step_results = state.get('step_results', {}).copy()
            self.patient_id_column = state.get('patient_id_column')
            self.grouping_variable = state.get('grouping_variable')
            self.hypothesis = state.get('hypothesis')
            # Load the stored dataframe state as the 'current' working copy
            loaded_df = state.get('current_dataframe')
            if loaded_df is not None:
                 self.current_dataset = loaded_df.copy()
            else: # Fallback if stored dataframe is missing
                 self.current_dataset = dataframe.copy()
                 # Reset state if dataframe was missing, as stored state might be invalid
                 self.step_statuses = {step['id']: "Not Started" for step in self.preprocessing_steps}
                 self.step_results = {}
                 self.patient_id_column = None
                 self.grouping_variable = None
                 self.hypothesis = None
                 print(f"[WARN] Stored DataFrame missing for {name}, resetting state.")

            # Ensure step_statuses dictionary covers all defined steps
            for step in self.preprocessing_steps:
                if step['id'] not in self.step_statuses:
                    self.step_statuses[step['id']] = "Not Started"

        else:
            print(f"[DEBUG] Initializing new state for dataset: {name}")
            self.current_dataset = dataframe.copy() # Start with a fresh copy
            self.step_statuses = {step['id']: "Not Started" for step in self.preprocessing_steps}
            self.step_results = {}
            self.patient_id_column = None
            self.grouping_variable = None
            self.hypothesis = None

        # Set hypothesis input field if available (preserving any changes)
        if hasattr(self, 'hypothesis_input'):
            # Temporarily disconnect to avoid triggering on_hypothesis_changed
            self.hypothesis_input.blockSignals(True)
            self.hypothesis_input.setPlainText(self.hypothesis or "")
            self.hypothesis_input.blockSignals(False)

        print(f"[DEBUG] Loaded state for {name}. Patient ID: {self.patient_id_column}, Grouping Var: {self.grouping_variable}, Hypothesis: {self.hypothesis}")
        print(f"[DEBUG] Step Statuses: {self.step_statuses}")

        # 4. Update UI based on loaded/initialized state
        self.current_dataset_label.setText(f"Dataset: {name}")
        rows, cols = self.current_dataset.shape # Use current_dataset shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")

        # Determine overall status based on loaded step statuses
        if all(status == "Completed" for status in self.step_statuses.values()):
             # Check readiness score from final validation if available
             final_val_results = self.step_results.get("final_validation", {}).get("analysis", "")
             # Basic check for readiness score in text (could be improved)
             if "Readiness score (from LLM):" in final_val_results:
                  # Attempt to parse score - simplistic for now
                  try:
                      score_line = [line for line in final_val_results.split('\\n') if "Readiness score (from LLM):" in line][0]
                      score_part = score_line.split("**")[1].split("/")[0]
                      score = int(score_part)
                      if score >= 7: # Assuming 7+ is ready
                           self.preprocessing_status_label.setText(f"Status: Ready for Analysis ({score}/10)")
                           self.preprocessing_status_label.setStyleSheet("color: #28a745;") # Green
                      else:
                           self.preprocessing_status_label.setText(f"Status: Partially Ready ({score}/10)")
                           self.preprocessing_status_label.setStyleSheet("color: #ffc107;") # Yellow
                  except Exception:
                      self.preprocessing_status_label.setText("Status: Processed")
                      self.preprocessing_status_label.setStyleSheet("color: #17a2b8;") # Info blue
             else:
                  self.preprocessing_status_label.setText("Status: Processed")
                  self.preprocessing_status_label.setStyleSheet("color: #17a2b8;") # Info blue
        elif any(status != "Not Started" for status in self.step_statuses.values()):
             self.preprocessing_status_label.setText("Status: Partially Processed")
             self.preprocessing_status_label.setStyleSheet("color: #ffc107;") # Yellow/amber
        else:
             self.preprocessing_status_label.setText("Status: Not Processed")
             self.preprocessing_status_label.setStyleSheet("color: #6c757d;") # Gray

        self.update_step_list_status() # Update list colors

        # Display the loaded/current dataset state
        self.view_toggle.setEnabled(True)
        self.view_toggle.setCurrentIndex(0) # Default to current view
        self.display_dataset(self.current_dataset)

        # Select the first step or the last shown step for context
        # For simplicity, just select the first step for now
        if self.steps_list.count() > 0:
            current_selection = self.steps_list.currentRow()
            if current_selection == -1: # If nothing selected, select first
                 self.steps_list.setCurrentRow(0)
                 self.on_step_selected(self.steps_list.item(0))
            else: # Re-trigger selection to update display based on loaded state
                 self.on_step_selected(self.steps_list.currentItem())
        else:
             self.step_title.setText("Select a step")
             self.step_description.setText("")
             self.step_status.setText("Not Started")
             self.step_action_button.setEnabled(False)

        self.update_status(f"Loaded dataset: {name}")

    def display_dataset(self, dataframe):
        """Display dataset in the view"""
        # Display in table
        self.dataset_display.display_dataframe(dataframe)
        
        # Show overview
        self.update_dataset_overview(dataframe)
        self.dataset_overview.setVisible(True)
        
        # Reset analysis text
        self.analysis_text.setText("")
    
    def update_dataset_overview(self, dataframe):
        """Update the dataset overview panel"""
        # Clear previous stats
        for i in reversed(range(self.overview_stats.count())): 
            item = self.overview_stats.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
            self.overview_stats.removeItem(item)
        
        # Add basic stats
        rows, cols = dataframe.shape
        self.overview_stats.addWidget(self.create_stat_card(
            "Dataset Size", 
            f"{rows} rows × {cols} columns",
            "table"
        ))
        
        # Add missing data info
        missing_count = dataframe.isna().sum().sum()
        missing_pct = (missing_count / (rows * cols)) * 100 if rows * cols > 0 else 0
        self.overview_stats.addWidget(self.create_stat_card(
            "Missing Data",
            f"{missing_count} cells ({missing_pct:.1f}%)",
            "exclamation-triangle"
        ))
        
        # Add columns info (categorical vs numeric)
        num_cols = dataframe.select_dtypes(include=['number']).columns.size
        cat_cols = dataframe.select_dtypes(exclude=['number']).columns.size
        self.overview_stats.addWidget(self.create_stat_card(
            "Column Types",
            f"{num_cols} numeric, {cat_cols} categorical",
            "bar-chart-line"
        ))
    
    def create_stat_card(self, title, value, icon_name):
        """Create a card for displaying a statistic"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                border-radius: 4px;
            }
            QLabel[class="title"] {
                font-size: 12px;
            }
            QLabel[class="value"] {
                font-weight: bold;
                font-size: 13px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        
        # Title section with icon
        title_layout = QHBoxLayout()
        title_layout.setSpacing(5)
        
        # Create icon
        icon = load_bootstrap_icon(icon_name, color="#0d6efd")
        icon_pixmap = icon.pixmap(16, 16)
        
        icon_label = QLabel()
        icon_label.setPixmap(icon_pixmap)
        icon_label.setMinimumSize(16, 16)
        title_layout.addWidget(icon_label)
        
        title_label = QLabel(title)
        title_label.setProperty("class", "title")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Value
        value_label = QLabel(value)
        value_label.setProperty("class", "value")
        value_label.setWordWrap(True)
        layout.addWidget(value_label)
        
        return card
    
    def toggle_dataset_view(self, index):
        """Toggle between original and processed dataset views"""
        if index == 0 and self.current_dataset is not None:
            # Show processed dataset
            self.display_dataset(self.current_dataset)
        elif index == 1 and self.original_dataset is not None:
            # Show original dataset
            self.display_dataset(self.original_dataset)
    
    def update_step_list_status(self):
        """Update the status indicators in the steps list"""
        for i in range(self.steps_list.count()):
            item = self.steps_list.item(i)
            step_id = item.data(Qt.ItemDataRole.UserRole)
            status = self.step_statuses[step_id]
            
            # Set color based on status
            if status == "Completed":
                item.setForeground(QColor("#28a745"))  # Green for completed
            elif status == "In Progress":
                item.setForeground(QColor("#007bff"))  # Blue for in progress
            elif status == "Error":
                item.setForeground(QColor("#dc3545"))  # Red for error
            else:
                item.setForeground(QColor("#6c757d"))  # Gray for not started
    
    def update_status(self, message):
        """Update status bar with message"""
        if hasattr(self, 'status_bar') and self.status_bar is not None:
            self.status_bar.showMessage(message)
        else:
            print(f"Status bar not initialized yet: {message}")
    
    def refresh_datasets(self):
        """Refresh the dataset list from the studies manager"""
        current_name = self.current_dataset_name
        self.populate_dataset_list()
        self.update_status("Refreshed dataset list.")

        dataset_still_exists = False
        if current_name:
            for i in range(self.dataset_list.count()):
                if self.dataset_list.item(i).text() == current_name:
                    dataset_still_exists = True
                    self.dataset_list.setCurrentRow(i)
                    break

        if current_name and not dataset_still_exists:
            print(f"[DEBUG] Removing state for dataset: {current_name}")
            # Remove state for the dataset that no longer exists
            if current_name in self.dataset_states:
                del self.dataset_states[current_name]

            # Clear the UI as before
            self.current_dataset_name = None
            self.current_dataset = None
            self.original_dataset = None
            self.step_results = {} 
            self.patient_id_column = None
            self.grouping_variable = None
            self.hypothesis = None
            # Initialize step statuses for empty state
            self.step_statuses = {step['id']: "Not Started" for step in self.preprocessing_steps}
            
            # Reset UI elements
            self.current_dataset_label.setText("No dataset loaded")
            self.dataset_info_label.setText("")
            self.dataset_display.clear()
            self.dataset_overview.setVisible(False)
            self.view_toggle.setEnabled(True)
            self.view_toggle.setCurrentIndex(0)
            self.analysis_text.clear()
            self.update_step_list_status() # Update list colors for reset state

            if self.steps_list.count() > 0:
                 self.steps_list.setCurrentRow(-1)
                 self.step_title.setText("Select a step")
                 self.step_description.setText("")
                 self.step_status.setText("Not Started")
                 self.step_action_button.setEnabled(False)
            self.update_status("Current dataset removed or refreshed, please select a dataset.")
    
    def save_processed_dataset(self):
        """Save the processed dataset back to the studies manager"""
        print(f"[DEBUG] save_processed_dataset: Check - self.current_dataset is None: {self.current_dataset is None}, name: {self.current_dataset_name}") # DEBUG
        # Original simple check: relies on dataset being loaded beforehand
        if self.current_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first")
            return
            
        # Check if any preprocessing has been done
        if self.current_dataset.equals(self.original_dataset):
            response = QMessageBox.question(
                self,
                "No Changes",
                "The dataset has not been modified. Save anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if response != QMessageBox.StandardButton.Yes:
                return
                
        # Ask user if they want to save as a new dataset or overwrite
        save_dialog = QDialog(self)
        save_dialog.setWindowTitle("Save Processed Dataset")
        save_dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(save_dialog)
        
        # Ask for dataset name
        layout.addWidget(QLabel("Save processed dataset as:"))
        
        save_options = QButtonGroup(save_dialog)
        
        # Option 1: Save as new dataset
        new_dataset_radio = QRadioButton("Save as new dataset")
        new_dataset_radio.setChecked(True)
        save_options.addButton(new_dataset_radio, 1)
        layout.addWidget(new_dataset_radio)
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        
        new_name_input = QLineEdit(f"{self.current_dataset_name}_processed")
        name_layout.addWidget(new_name_input)
        layout.addLayout(name_layout)
        
        # Option 2: Overwrite existing
        overwrite_radio = QRadioButton("Overwrite existing dataset")
        save_options.addButton(overwrite_radio, 2)
        layout.addWidget(overwrite_radio)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(save_dialog.accept)
        button_box.rejected.connect(save_dialog.reject)
        layout.addWidget(button_box)
        
        if save_dialog.exec() == QDialog.DialogCode.Accepted:
            # Get the save option
            option = save_options.checkedId()
            
            if option == 1:  # Save as new
                new_name = new_name_input.text().strip()
                if not new_name:
                    QMessageBox.warning(self, "Invalid Name", "Please enter a valid dataset name")
                    return
                    
                # Save as new dataset
                self.save_as_new_dataset(new_name)
            else:  # Overwrite
                # Confirm overwrite
                confirm = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"Are you sure you want to overwrite the dataset '{self.current_dataset_name}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if confirm == QMessageBox.StandardButton.Yes:
                    self.overwrite_dataset()
    
    def save_as_new_dataset(self, new_name):
        """Save the processed dataset as a new dataset"""
        # Get reference to the studies manager instance
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return

        # Check if the new name already exists using self.studies_manager
        # Get existing dataset names from the active study
        existing_datasets = self.studies_manager.get_datasets_from_active_study()
        existing_names = [name for name, _ in existing_datasets]

        if new_name in existing_names:
            confirm = QMessageBox.question(
                self,
                "Dataset Exists",
                f"Dataset '{new_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
                
        # Add the dataset to the studies manager using the instance
        # Use add_dataset_to_active_study which handles both add and overwrite
        success = self.studies_manager.add_dataset_to_active_study(new_name, self.current_dataset)
        
        if success:
            self.update_status(f"Saved processed dataset as: {new_name}")
            QMessageBox.information(self, "Success", f"Processed dataset saved as '{new_name}'")
            # Refresh the dataset list to show the new dataset
            self.populate_dataset_list()
        else:
            QMessageBox.warning(self, "Error", f"Failed to save dataset as '{new_name}'")
    
    def overwrite_dataset(self):
        """Overwrite the existing dataset with the processed version"""
        # Get reference to the studies manager instance
        if not self.studies_manager:
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return

        # Update the dataset in the studies manager using the instance and method consistent with collect.py
        success = self.studies_manager.update_dataset_in_active_study(
            self.current_dataset_name, self.current_dataset
        )
        
        if success:
            # Update original dataset reference
            self.original_dataset = self.current_dataset.copy()
            
            self.update_status(f"Updated dataset: {self.current_dataset_name}")
            QMessageBox.information(self, "Success", f"Dataset '{self.current_dataset_name}' has been updated")
            
            # Emit the dataset_updated signal
            self.dataset_updated.emit(self.current_dataset_name, self.current_dataset)
        else:
            QMessageBox.warning(self, "Error", f"Failed to update dataset '{self.current_dataset_name}'")
    
    def run_current_step(self):
        """Run the currently selected step"""
        print(f"[DEBUG] run_current_step: Check - self.current_dataset is None: {self.current_dataset is None}, name: {self.current_dataset_name}") # DEBUG
        # Original simple check
        if self.current_dataset is None:
            QMessageBox.warning(self, "No Dataset", "Please load a dataset first")
            return
            
        # Get the selected step
        items = self.steps_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "No Step Selected", "Please select a preprocessing step")
            return
            
        step_id = items[0].data(Qt.ItemDataRole.UserRole)
        
        # Update step status
        self.step_statuses[step_id] = "In Progress"
        self.step_status.setText("In Progress")
        self.update_step_list_status()
        
        # Disable the run button while processing
        self.step_action_button.setEnabled(False)
        
        # Call the appropriate async method directly - @asyncSlot will handle it
        if step_id == "identify_patient_id":
            self.run_identify_patient_id()
        elif step_id == "normalize_rows":
            self.run_normalize_rows()
        elif step_id == "check_grouper":
            self.run_check_grouper()
        elif step_id == "validate_columns":
            self.run_validate_columns()
        elif step_id == "final_validation":
            self.run_final_validation()
    
    @asyncSlot()
    async def run_auto_process(self):
        """Run all preprocessing steps automatically"""
        if self.current_dataset is None:
            return
            
        # Update status
        self.update_status("Auto processing dataset...")
        
        # Run each step in sequence
        await self.run_identify_patient_id()
        # Check status before proceeding
        if self.step_statuses["identify_patient_id"] != "Completed":
             self.update_status("Auto processing stopped due to error in Step 1.")
             return
             
        await self.run_normalize_rows()
        # Check status before proceeding
        if self.step_statuses["normalize_rows"] != "Completed":
             self.update_status("Auto processing stopped due to error in Step 2.")
             return
             
        await self.run_check_grouper()
        # Check status before proceeding
        if self.step_statuses["check_grouper"] != "Completed":
             self.update_status("Auto processing stopped due to error in Step 3.")
             return
             
        await self.run_validate_columns() # <-- Added this call
        # Check status before proceeding
        if self.step_statuses["validate_columns"] != "Completed":
             self.update_status("Auto processing stopped due to error in Step 4.")
             return
             
        await self.run_final_validation()
        # Check status before proceeding
        if self.step_statuses["final_validation"] != "Completed":
             self.update_status("Auto processing stopped due to error in Step 5.")
             return
        
        # Update status
        self.update_status("Auto processing complete")
        
        # Show final validation results
        self.display_tabs.setCurrentIndex(1)  # Switch to Analysis tab
    
    @asyncSlot()
    async def run_identify_patient_id(self):
        """Run Step 1: Identify patient identifier columns"""
        step_id = "identify_patient_id"
        analysis = "" # Initialize analysis text
        try:
            self.update_status("Identifying patient identifiers...")
            self.step_statuses[step_id] = "In Progress"
            self.step_status.setText("In Progress")
            self.update_step_list_status()
            self.step_action_button.setEnabled(False) # Disable button during run

            analysis = "## Patient Identifier Analysis\\n\\n"
            df = self.current_dataset

            if df is None:
                 raise ValueError("Dataset not loaded.")

            # Use LLM to identify potential patient ID columns
            prompt = f"""
            I have a healthcare dataset with {len(df)} rows and {len(df.columns)} columns:
            
            Column names: {', '.join(df.columns.tolist())}
            
            Sample data:
            {df.head(5).to_string()}
            """
            
            # Include hypothesis if available
            if self.hypothesis:
                prompt += f"""
                
                Study hypothesis: {self.hypothesis}
                """
                
                # Add hypothesis to analysis
                analysis += f"Using study hypothesis for context: \"{self.hypothesis}\"\\n\\n"
            
            prompt += f"""
            Please identify which column(s) could be used as patient identifiers. 
            These are unique identifiers that can be used to track the same patient across different rows.
            
            Some examples of what could be patient identifiers:
            - Specific ID columns like 'patient_id', 'subject_id', 'participant_id'
            - Medical record numbers (MRN)
            - Patient number columns
            - Sometimes a combination of columns like 'first_name + last_name + dob'
            
            For each potential identifier column, indicate:
            1. The column name
            2. Why you think it's a patient identifier
            3. Whether it appears to be unique per patient
            4. What percentage of values are unique
            
            Then recommend the BEST single column or combination of columns to use as the patient identifier.
            
            Return the response in this JSON format:
            {{
                "potential_id_columns": [
                    {{
                        "column": "column_name",
                        "reason": "Why this column is a potential identifier",
                        "uniqueness": "Percentage of unique values",
                        "recommended": true/false
                    }},
                    ...
                ],
                "best_identifier": "column_name" or ["column1", "column2"],
                "recommendation_reason": "Why this is the best identifier"
            }}
            """
            
            # Call LLM using the JSON-specific function and default JSON model
            analysis_data = await call_llm_async_json(prompt, model=llm_config.default_json_model)

            # --- Existing code for processing analysis_data ---
            analysis += "### Potential Patient Identifiers\\n\\n"
            if analysis_data.get("potential_id_columns"):
                for col_info in analysis_data["potential_id_columns"]:
                    column = col_info.get("column", "")
                    reason = col_info.get("reason", "")
                    uniqueness = col_info.get("uniqueness", "")
                    recommended = col_info.get("recommended", False)
                    
                    analysis += f"**{column}**\\n"
                    analysis += f"- Reason: {reason}\\n"
                    analysis += f"- Uniqueness: {uniqueness}\\n"
                    if recommended:
                        analysis += f"- ✓ Recommended\\n"
                    analysis += "\\n"

            best_id = analysis_data.get("best_identifier", "")
            reason = analysis_data.get("recommendation_reason", "")
            
            if isinstance(best_id, list):
                best_id_str = ", ".join(best_id)
            else:
                best_id_str = best_id
            
            analysis += f"### Recommended Identifier\\n\\n"
            analysis += f"**{best_id_str}**\\n\\n"
            analysis += f"Reason: {reason}\\n\\n"
            
            # Store the recommended identifier for use in future steps
            self.patient_id_column = best_id
            
            # Add uniqueness check
            analysis += "### Uniqueness Check\\n\\n"
            
            # Calculate uniqueness for the recommended identifier
            if isinstance(best_id, list):
                # For compound identifiers, create a combined column
                combined_id = df[best_id].astype(str).agg('-'.join, axis=1)
                total_rows = len(df)
                unique_values = combined_id.nunique()
                uniqueness_pct = (unique_values / total_rows) * 100
                
                analysis += f"Combined identifier ({best_id_str}):\\n"
                analysis += f"- Total rows: {total_rows}\\n"
                analysis += f"- Unique values: {unique_values}\\n"
                analysis += f"- Uniqueness: {uniqueness_pct:.2f}%\\n\\n"
                
                if uniqueness_pct < 100:
                    analysis += "⚠️ **Warning**: The recommended identifier is not 100% unique.\\n"
                    analysis += "This might indicate duplicate patient records or multiple entries per patient.\\n\\n"
            else:
                # For single column identifiers
                if best_id in df.columns:
                    total_rows = len(df)
                    unique_values = df[best_id].nunique()
                    uniqueness_pct = (unique_values / total_rows) * 100
                    
                    analysis += f"Column '{best_id}':\\n"
                    analysis += f"- Total rows: {total_rows}\\n"
                    analysis += f"- Unique values: {unique_values}\\n"
                    analysis += f"- Uniqueness: {uniqueness_pct:.2f}%\\n\\n"
                    
                    if uniqueness_pct < 100:
                        analysis += "⚠️ **Warning**: The recommended identifier is not 100% unique.\\n"
                        analysis += "This might indicate duplicate patient records or multiple entries per patient.\\n\\n"
            # --- End of existing code ---

            # On success:
            self.step_statuses[step_id] = "Completed"
            self.step_status.setText("Completed")
            self.update_status("Patient identifier analysis completed")
            # Store results
            self.step_results[step_id] = {
                "dataframe": self.current_dataset.copy(), # Store copy of current state
                "analysis": analysis
            }
            # --- Save state AFTER successful completion ---
            self.save_current_dataset_state()
            self.select_next_step()

        except Exception as e:
            error_message = f"Error identifying patient identifiers: {str(e)}"
            print(f"[ERROR] {error_message}")
            analysis += f"\\n--- ERROR --- \\n{error_message}\\n"
            self.step_statuses[step_id] = "Error"
            self.step_status.setText("Error")
            self.update_status(error_message)

        finally:
            # Always update UI elements at the end
            self.analysis_text.setText(analysis)
            self.update_step_list_status()
            # Re-enable button only if the selected step is still this one and it's not completed
            items = self.steps_list.selectedItems()
            if items and items[0].data(Qt.ItemDataRole.UserRole) == step_id and self.step_statuses[step_id] != "Completed":
                self.step_action_button.setEnabled(True)
            elif not items or items[0].data(Qt.ItemDataRole.UserRole) != step_id :
                 # If selection changed, on_step_selected will handle button state
                 pass
            else: # Step is completed
                 self.step_action_button.setEnabled(False)
            self.display_tabs.setCurrentIndex(1)
    
    @asyncSlot()
    async def run_normalize_rows(self):
        """Run Step 2: Normalize dataset to one row per patient"""
        step_id = "normalize_rows"
        analysis = ""
        original_df_before_step = self.current_dataset.copy() if self.current_dataset is not None else None

        try:
            self.update_status("Normalizing dataset...")
            self.step_statuses[step_id] = "In Progress"
            self.step_status.setText("In Progress")
            self.update_step_list_status()
            self.step_action_button.setEnabled(False)

            analysis = "## Dataset Normalization Analysis\\n\\n"
            
            # Add hypothesis to analysis if available
            if self.hypothesis:
                analysis += f"Using study hypothesis for context: \"{self.hypothesis}\"\\n\\n"
                
            df = self.current_dataset

            if df is None:
                 raise ValueError("Dataset not loaded.")
            if not hasattr(self, 'patient_id_column') or not self.patient_id_column:
                 analysis += "⚠️ **Error**: No patient identifier was identified in Step 1. Cannot normalize.\\n\\n"
                 raise ValueError("Patient identifier not set.")

            # Get the patient identifier
            patient_id = self.patient_id_column
            patient_id_str = patient_id if isinstance(patient_id, str) else ", ".join(patient_id)

            # Check for duplicates
            unique_values = 0 # Initialize
            total_rows = len(df)
            if isinstance(patient_id, list):
                # For compound identifiers
                combined_id = df[patient_id].astype(str).agg('-'.join, axis=1)
                unique_values = combined_id.nunique()
                rows_per_patient = combined_id.value_counts()
                max_rows = rows_per_patient.max()
                patients_with_multiple_rows = (rows_per_patient > 1).sum()
                
                analysis += f"### Current Dataset Structure\\n\\n"
                analysis += f"- Patient Identifier: {patient_id_str}\\n"
                analysis += f"- Total Rows: {total_rows}\\n"
                analysis += f"- Unique Patients: {unique_values}\\n"
                analysis += f"- Maximum Rows per Patient: {max_rows}\\n"
                analysis += f"- Patients with Multiple Rows: {patients_with_multiple_rows}\\n\\n"
                
            else:
                # For single column identifiers
                if patient_id in df.columns:
                    unique_values = df[patient_id].nunique()
                    rows_per_patient = df[patient_id].value_counts()
                    max_rows = rows_per_patient.max()
                    patients_with_multiple_rows = (rows_per_patient > 1).sum()
                    
                    analysis += f"### Current Dataset Structure\\n\\n"
                    analysis += f"- Patient Identifier: {patient_id}\\n"
                    analysis += f"- Total Rows: {total_rows}\\n"
                    analysis += f"- Unique Patients: {unique_values}\\n"
                    analysis += f"- Maximum Rows per Patient: {max_rows}\\n"
                    analysis += f"- Patients with Multiple Rows: {patients_with_multiple_rows}\\n\\n"
                else:
                    analysis += f"⚠️ **Error**: Patient identifier column '{patient_id}' not found in the dataset.\\n\\n"
                    raise ValueError(f"Patient ID column '{patient_id}' not found.")

            # Check if normalization is needed
            if total_rows == unique_values:
                 analysis += "✓ **Dataset is already normalized**: There is exactly one row per patient.\\n\\n"
                 self.step_statuses[step_id] = "Completed"
                 self.step_status.setText("Completed")
                 self.update_status("Dataset is already normalized")
                 self.step_results[step_id] = {
                     "dataframe": self.current_dataset.copy(), # Store current state
                     "analysis": analysis
                 }
                 # Save state explicitly here as we return early
                 self.save_current_dataset_state()
                 self.select_next_step()
                 return

            # Normalization is needed
            analysis += "⚠️ **Normalization needed**: There are multiple rows per patient.\\n\\n"
            
            # Use LLM to suggest normalization strategy
            prompt = f"""
            I have a healthcare dataset with {len(df)} rows and {len(df.columns)} columns.
            
            Column names: {', '.join(df.columns.tolist())}
            
            Sample data:
            {df.head(5).to_string()}
            
            The patient identifier is: {patient_id_str}
            """
            
            # Include hypothesis if available
            if self.hypothesis:
                prompt += f"""
                
                Study hypothesis: {self.hypothesis}
                """
            
            prompt += f"""
            
            There are {unique_values} unique patients but {total_rows} total rows.
            This means some patients have multiple rows in the dataset.
            
            Please suggest a strategy to normalize this dataset to have one row per patient. 
            Consider these common approaches:
            
            1. First occurrence: Keep only the first row for each patient
            2. Last occurrence: Keep only the last row for each patient
            3. Aggregation: Combine multiple rows using aggregation functions for each column
               (e.g., mean for numeric, mode for categorical, first/last for identifiers)
            4. Pivoting: For time-series or visit data, create new columns with suffixes
               (e.g., bp_visit1, bp_visit2)
            
            For the aggregation or pivoting approach, please provide specific instructions for each column.
            
            Return the response in this JSON format:
            {{
                "recommended_approach": "first_occurrence|last_occurrence|aggregation|pivoting",
                "reason": "Explanation of why this approach is recommended",
                "implementation": {{
                    "approach_details": "How to implement the approach",
                    "column_specific_instructions": [
                        {{
                            "column": "column_name",
                            "method": "agg_function or new_column_name pattern",
                            "reason": "Why this method for this column"
                        }},
                        ...
                    ]
                }},
                "code": "Python code to implement the normalization (without explanation comments)"
            }}
            """
            
            # Call LLM using JSON function
            normalization_strategy = await call_llm_async_json(prompt, model=llm_config.default_json_model)
            
            if not isinstance(normalization_strategy, dict):
                analysis += f"⚠️ **Error**: LLM did not return a valid JSON object for normalization strategy. Response type: {type(normalization_strategy)}\\nRaw Response:\\n```\\n{normalization_strategy}\\n```\\n"
                raise TypeError("LLM response for normalization is not a dictionary.")

            # Add results to analysis
            recommended_approach = normalization_strategy.get("recommended_approach", "")
            reason = normalization_strategy.get("reason", "")
            
            analysis += f"### Recommended Normalization Approach: {recommended_approach}\\n\\n"
            analysis += f"{reason}\\n\\n"
            
            # Add implementation details
            implementation = normalization_strategy.get("implementation", {})
            approach_details = implementation.get("approach_details", "")
            
            analysis += f"### Implementation Details\\n\\n"
            analysis += f"{approach_details}\\n\\n"
            
            # Add column-specific instructions
            column_instructions = implementation.get("column_specific_instructions", [])
            
            if column_instructions:
                analysis += f"### Column-Specific Instructions\\n\\n"
                
                for instruction in column_instructions:
                    column = instruction.get("column", "")
                    method = instruction.get("method", "")
                    reason = instruction.get("reason", "")
                    
                    analysis += f"**{column}**\\n"
                    analysis += f"- Method: {method}\\n"
                    analysis += f"- Reason: {reason}\\n\\n"
            
            # Execute the normalization code
            normalize_code = normalization_strategy.get("code", "")
            
            if normalize_code:
                analysis += f"### Normalization Code\\n\\n"
                analysis += f"```python\\n{normalize_code}\\n```\\n\\n"
                
                # Create safe execution environment
                local_vars = {
                    "df": df.copy(),
                    "patient_id": patient_id,
                    "pd": pd,
                    "np": np
                }
                
                try:
                    # Execute the code
                    exec(normalize_code, {}, local_vars)
                    
                    # Get the result (should be stored in 'normalized_df')
                    if "normalized_df" in local_vars and isinstance(local_vars["normalized_df"], pd.DataFrame):
                        normalized_df = local_vars["normalized_df"]
                        
                        # Update the current dataset
                        self.current_dataset = normalized_df
                        self.display_dataset(normalized_df) # Update display immediately
                        
                        # Add normalization results
                        analysis += f"### Normalization Results\\n\\n"
                        analysis += f"- Original rows: {total_rows}\\n"
                        analysis += f"- Normalized rows: {len(normalized_df)}\\n"
                        # Recalculate unique patients based on normalized df
                        if isinstance(patient_id, list):
                            norm_combined_id = normalized_df[patient_id].astype(str).agg('-'.join, axis=1)
                            norm_unique_values = norm_combined_id.nunique()
                        else:
                            norm_unique_values = normalized_df[patient_id].nunique()
                        analysis += f"- Unique patients in normalized data: {norm_unique_values}\\n"
                        analysis += f"- Rows per patient: {len(normalized_df) / norm_unique_values:.2f} (if > 0 unique patients)\\n\\n"
                        
                        if len(normalized_df) == norm_unique_values:
                            analysis += "✓ **Success**: Dataset has been normalized to one row per patient.\\n\\n"
                        else:
                            analysis += "⚠️ **Warning**: Normalization did not achieve exactly one row per patient.\\n"
                            analysis += "Further normalization may be needed.\\n\\n"
                    else:
                        analysis += "⚠️ **Error**: Normalization code did not produce a valid DataFrame named 'normalized_df'.\\n\\n"
                        raise ValueError("Normalization code failed to produce 'normalized_df'")
                
                except Exception as e:
                    analysis += f"⚠️ **Error executing normalization code**: {str(e)}\\n\\n"
                    raise e # Re-raise the specific error
            else:
                analysis += "⚠️ **Warning**: LLM suggested normalization but provided no code.\\n\\n"
                # Treat lack of code as an error for this step
                raise ValueError("Normalization needed but no code provided.")
            

            # If code execution and processing succeeded:
            self.step_statuses[step_id] = "Completed"
            self.step_status.setText("Completed")
            self.update_status("Dataset normalization completed")
            # Store results AFTER potential modification by exec()
            self.step_results[step_id] = {
                "dataframe": self.current_dataset.copy(), # Store final state for this step
                "analysis": analysis
            }
            # --- Save state AFTER successful completion ---
            self.save_current_dataset_state()
            self.select_next_step()

        except Exception as e:
            error_message = f"Error normalizing dataset: {str(e)}"
            print(f"[ERROR] {error_message}")
            analysis += f"\\n--- ERROR --- \\n{error_message}\\n"
            self.step_statuses[step_id] = "Error"
            self.step_status.setText("Error")
            self.update_status(error_message)
            # Revert dataset if error occurred during processing
            if original_df_before_step is not None:
                self.current_dataset = original_df_before_step
                # No need to redisplay here, finally block handles UI update

        finally:
            # Always update UI elements at the end
            self.analysis_text.setText(analysis)
            self.update_step_list_status()
            # Update display in case of error/revert or if normalization wasn't needed
            if self.current_dataset is not None:
                 self.display_dataset(self.current_dataset)

            # Re-enable button logic (same as run_identify_patient_id)
            items = self.steps_list.selectedItems()
            if items and items[0].data(Qt.ItemDataRole.UserRole) == step_id and self.step_statuses[step_id] != "Completed":
                self.step_action_button.setEnabled(True)
            elif not items or items[0].data(Qt.ItemDataRole.UserRole) != step_id :
                 pass
            else: # Step is completed
                 self.step_action_button.setEnabled(False)
            self.display_tabs.setCurrentIndex(1)
    
    @asyncSlot()
    async def run_check_grouper(self):
        """
        Run Step 3: Identify candidates, let user select, generate/execute code,
        check balance, allow re-selection if imbalanced, set final grouper.
        """
        step_id = "check_grouper"
        analysis = "## Grouping Variable Analysis (User Selection & Balance Check)\\n\\n"
        original_df_before_step = self.current_dataset.copy() if self.current_dataset is not None else None
        original_grouper = self.grouping_variable
        
        # Add hypothesis to analysis if available
        if self.hypothesis:
            analysis += f"Using study hypothesis for context: \"{self.hypothesis}\"\\n\\n"
            
        # Store initial candidates here to allow removal
        high_suitability_candidates = [] 
        selected_candidate_col = None
        final_grouper_set = False
        
        try:
            self.update_status("Identifying grouping variable candidates...")
            self.step_statuses[step_id] = "In Progress"
            self.step_status.setText("In Progress")
            self.update_step_list_status()
            self.step_action_button.setEnabled(False)

            df = self.current_dataset # Use the current state at the start
            if df is None: raise ValueError("Dataset not loaded.")

            # --- 1. Initial LLM call to find candidates ---
            initial_prompt = f"""
            I have a healthcare dataset with {len(df)} rows and {len(df.columns)} columns:
            Column names: {', '.join(df.columns.tolist())}
            Sample data:
            {df.head(5).to_string()}
            """
            
            # Include hypothesis if available - this is especially important for grouping
            if self.hypothesis:
                initial_prompt += f"""
                
                Study hypothesis: {self.hypothesis}
                
                Based on this hypothesis, I need to identify appropriate grouping variables that would allow testing of this hypothesis.
                """
            
            initial_prompt += f"""
            
            Identify potential grouping variables suitable for hypothesis testing. Examples: treatment/control, demographics (gender, age bins), clinical conditions (diabetes, hypertension), severity scores (low/medium/high).
            
            For each potential variable, assess its suitability (High, Medium, Low) based on relevance and potential for creating distinct groups.
            """
            
            # If hypothesis available, add a specific instruction to prioritize variables relevant to it
            if self.hypothesis:
                initial_prompt += f"""
                
                Give 'High' suitability primarily to variables that directly relate to the study hypothesis.
                """
                
            initial_prompt += f"""
            
            Return ONLY the response in this JSON format:
            {{
                "potential_grouping_variables": [
                    {{"column": "column_name", "suitability": "High|Medium|Low", "reason": "Brief reason"}}, ...
                ]
            }}
            """
            
            initial_analysis = await call_llm_async_json(initial_prompt, model=llm_config.default_json_model)

            if not isinstance(initial_analysis, dict) or "potential_grouping_variables" not in initial_analysis:
                analysis += f"⚠️ **Error**: LLM did not return valid potential grouping variables.\\nResponse:\\n```\\n{initial_analysis}\\n```\\n"
                raise TypeError("LLM response for candidates is invalid.")

            potential_groupers = initial_analysis.get("potential_grouping_variables", [])
            analysis += "### Potential Grouping Variable Candidates\\n\\n"
            if not potential_groupers:
                analysis += "No potential variables identified by LLM.\\n"
                raise ValueError("No potential grouping variables identified.")

            # Populate initial list of high-suitability candidates
            for candidate in potential_groupers:
                col_name = candidate.get("column")
                suitability = candidate.get("suitability")
                if suitability == "High" and col_name in df.columns:
                    high_suitability_candidates.append(col_name)
            analysis += "\\n"

            if not high_suitability_candidates:
                 raise ValueError("No high suitability grouping candidates found.")
            analysis += f"Found {len(high_suitability_candidates)} high suitability candidate(s): {', '.join(high_suitability_candidates)}\\n\\n"

            # --- 2. Loop for Selection, Code Gen, Execution, Balance Check ---
            # Keep track of the dataframe state *before* trying a candidate's code
            current_processing_df = self.current_dataset.copy() 

            while not final_grouper_set and high_suitability_candidates:
                # --- User Selection ---
                if len(high_suitability_candidates) == 1:
                     selected_candidate_col = high_suitability_candidates[0]
                     analysis += f"Only one candidate remaining: `{selected_candidate_col}`. Proceeding automatically.\\n\\n"
                else:
                     selected_candidate_col, ok = QInputDialog.getItem(
                         self, "Select Candidate for Grouping",
                         "Select candidate column to generate grouping code for:",
                         high_suitability_candidates, 0, False
                     )
                     if not ok or not selected_candidate_col:
                         analysis += "User cancelled candidate selection.\\n"
                         # Revert to state before this step attempt
                         self.current_dataset = original_df_before_step
                         self.grouping_variable = original_grouper
                         self.step_statuses[step_id] = "Not Started"
                         self.step_status.setText("Not Started")
                         self.update_status("Grouping variable selection cancelled.")
                         return # Exit cleanly
                     analysis += f"User selected candidate: `{selected_candidate_col}`\\n\\n"

                # Remove selected candidate from list for next iteration (if needed)
                # Do this *after* selection but *before* potential error/skip
                if selected_candidate_col in high_suitability_candidates:
                     high_suitability_candidates.remove(selected_candidate_col)
                     
                # --- Generate and execute code ---
                analysis += f"### Processing Candidate: `{selected_candidate_col}`\\n\\n"
                self.update_status(f"Generating grouping code for: {selected_candidate_col}...")
                
                col_name = selected_candidate_col
                # Use current_processing_df for info gathering
                dtype = str(current_processing_df[col_name].dtype) 
                unique_vals = current_processing_df[col_name].nunique()
                sample_vals = current_processing_df[col_name].dropna().unique()[:5]

                code_gen_prompt = f"""
                Given a pandas DataFrame 'df' with a column named '{col_name}' (dtype: {dtype}, unique values: {unique_vals}, sample: {sample_vals}), write Python code to add a NEW categorical grouping column named '{col_name}_group' to the DataFrame.

                The new column should represent meaningful groups based on the '{col_name}' values. Examples:
                - If '{col_name}' is numeric/continuous (like age or a score), create bins (e.g., 'Low', 'Medium', 'High' or age ranges '<65', '65+'). Aim for 2-5 balanced groups if possible.
                - If '{col_name}' is categorical or boolean (like gender or hypertension status '0'/'1'), use the existing values to create descriptive category names (e.g., 'Male'/'Female', 'Hypertensive'/'Non-Hypertensive').
                - If '{col_name}' already represents good groups, create the new column by mapping the existing values to descriptive strings if necessary.

                The code MUST:
                1. Operate on a DataFrame assumed to be named 'df'.
                2. Add a new column named exactly '{col_name}_group'.
                3. Handle potential missing values in '{col_name}' appropriately (e.g., assign to a 'Missing' group or propagate NaN).
                4. Be self-contained (import pandas as pd if needed, but assume df exists).

                Return ONLY the Python code as a single string. Do not include explanations or example usage outside the code string.
                Example for a numeric score 'charlson_index':
                ```python
                import pandas as pd
                import numpy as np
                def create_groups(df):
                    bins = [-np.inf, 2, 5, np.inf]
                    labels = ['Low (0-2)', 'Medium (3-5)', 'High (6+)']
                    df['charlson_index_group'] = pd.cut(df['charlson_index'], bins=bins, labels=labels, right=True)
                    df['charlson_index_group'] = df['charlson_index_group'].astype(str).fillna('Missing') # Handle NaN after binning
                    return df
                df = create_groups(df.copy()) # Apply function
                ```
                Example for boolean 'hypertension' (0/1):
                ```python
                import pandas as pd
                def create_groups(df):
                    mapping = {{0: 'Non-Hypertensive', 1: 'Hypertensive'}}
                    df['hypertension_group'] = df['hypertension'].map(mapping).fillna('Missing')
                    return df
                df = create_groups(df.copy()) # Apply function
                ```
                """
                generated_code_raw = await call_llm_async(code_gen_prompt)

                # --- Clean the generated code --- 
                generated_code = generated_code_raw.strip()
                if generated_code.startswith("```python"):
                    generated_code = generated_code[9:] # Remove ```python
                if generated_code.startswith("```"):
                     generated_code = generated_code[3:] # Remove ```
                if generated_code.endswith("```"):
                     generated_code = generated_code[:-3] # Remove trailing ```
                generated_code = generated_code.strip()
                # --------------------------------

                if not generated_code: # Add better validation
                    analysis += f"⚠️ LLM did not return valid code for `{col_name}`. Skipping candidate.\\n"
                    print(f"[DEBUG] Invalid code for {col_name}")
                    continue # Skip to next candidate in the while loop

                analysis += f"Generated code. Attempting execution...\\n```python\\n{generated_code}\\n```\\n"
                print(f"[DEBUG] Cleaned code for {col_name}:\\n{generated_code}\\n")
                
                # Execute on a *copy* to easily revert if needed
                local_vars = {"df": current_processing_df.copy(), "pd": pd, "np": np} 
                new_col_name = f"{col_name}_group"
                
                try:
                    exec(generated_code, {"pd": pd, "np": np}, local_vars)
                    
                    if "df" not in local_vars or not isinstance(local_vars["df"], pd.DataFrame):
                         raise ValueError("Code execution did not result in a DataFrame.")
                    result_df_candidate = local_vars["df"] # This df has the new column (potentially)

                    if new_col_name not in result_df_candidate.columns:
                        raise ValueError(f"Code did not create expected column '{new_col_name}'.")

                    analysis += f"✓ Code executed successfully. Created column: `{new_col_name}`.\\n"
                    
                    # --- Calculate and Display Balance ---
                    analysis += f"\\n**Balance Check for `{new_col_name}`:**\\n"
                    counts = result_df_candidate[new_col_name].value_counts()
                    analysis += "  Group counts:\\n"
                    for group_name, count_val in counts.items():
                         analysis += f"    - {group_name}: {count_val}\\n"
                    
                    balance_ratio = 0.0
                    warning_note = "N/A (Less than 2 groups)"
                    is_imbalanced = False
                    if len(counts) >= 2:
                        min_count = counts.min()
                        max_count = counts.max()
                        balance_ratio = min_count / max_count if max_count > 0 else 0
                        analysis += f"\\n  Balance Ratio (smallest/largest): {balance_ratio:.2f}\\n"
                        if balance_ratio < 0.3:
                            warning_note = "⚠️ **Warning**: Groups highly imbalanced (ratio < 0.3). Affects analysis."
                            is_imbalanced = True
                        elif balance_ratio < 0.5:
                            warning_note = "⚠️ **Note**: Groups somewhat imbalanced (ratio < 0.5)."
                            is_imbalanced = True # Still prompt user even for moderate imbalance
                        else:
                            warning_note = "✓ **Good**: Groups reasonably balanced."
                    analysis += f"  {warning_note}\\n\\n"

                    # --- Prompt User if Imbalanced ---
                    user_choice = QMessageBox.StandardButton.Yes # Default to keep if not imbalanced
                    if is_imbalanced and high_suitability_candidates: # Only ask if other options remain
                        msgBox = QMessageBox(self)
                        msgBox.setWindowTitle("Imbalanced Grouping Created")
                        msgBox.setIcon(QMessageBox.Icon.Warning)
                        msgBox.setText(f"The created grouping column `{new_col_name}` is imbalanced (ratio: {balance_ratio:.2f}).\n\n{warning_note}")
                        msgBox.setInformativeText("Do you want to keep this grouping variable or select another candidate?")
                        keepButton = msgBox.addButton("Keep This Grouper", QMessageBox.ButtonRole.AcceptRole)
                        selectAnotherButton = msgBox.addButton("Select Another Candidate", QMessageBox.ButtonRole.RejectRole)
                        
                        msgBox.exec()
                        
                        if msgBox.clickedButton() == selectAnotherButton:
                             user_choice = QMessageBox.StandardButton.No # User wants to try again
                             analysis += f"User chose to select another candidate instead of `{new_col_name}`.\\n\\n"
                             # The loop will continue after removing this candidate earlier
                             continue # Skip to next iteration of while loop
                        else:
                             analysis += f"User chose to keep the imbalanced grouper `{new_col_name}`.\\n"
                             # Fall through to set this as the final grouper
                             
                    elif is_imbalanced: # Imbalanced but no other candidates left
                         analysis += "Grouper is imbalanced, but no other high-suitability candidates remain. Keeping this one.\\n"
                         QMessageBox.warning(self,"Imbalanced Grouper", f"The created grouping column `{new_col_name}` is imbalanced (ratio: {balance_ratio:.2f}), and no other candidates are available. This grouper will be used.")


                    # --- Keep This Grouper ---
                    if user_choice == QMessageBox.StandardButton.Yes:
                         self.current_dataset = result_df_candidate # Commit the df with the new column
                         self.grouping_variable = new_col_name
                         final_grouper_set = True # Exit the while loop
                         analysis += f"\\n**Final Grouping Variable Set To:** `{self.grouping_variable}`\\n"
                         # Loop will terminate

                except Exception as e:
                    exec_error_msg = f"Error processing candidate `{col_name}`: {str(e)}"
                    print(f"[ERROR] {exec_error_msg}")
                    analysis += f"⚠️ {exec_error_msg}. Trying next candidate if available.\\n\\n"
                    # If code execution fails, continue the loop to try the next candidate (if any)

            # --- End of While Loop ---
            
            if not final_grouper_set:
                 analysis += "\\nFailed to set a final grouping variable after trying all candidates.\\n"
                 raise ValueError("Could not create a suitable grouping variable.")

            # --- Completion ---
            self.update_status(f"Grouping variable set to: {self.grouping_variable}")
            self.step_statuses[step_id] = "Completed"
            self.step_status.setText("Completed")
            self.step_results[step_id] = {
                "dataframe": self.current_dataset.copy(), # Save final DF state
                "analysis": analysis
            }
            self.save_current_dataset_state() # Save overall state
            self.select_next_step()

        except Exception as e:
            # ... (Outer error handling - revert dataset/grouper) ...
            error_message = f"Error during grouping variable processing: {str(e)}"
            print(f"[ERROR] {error_message}")
            analysis += f"\\n--- ERROR --- \\n{error_message}\\n"
            self.step_statuses[step_id] = "Error"
            self.step_status.setText("Error")
            self.update_status(error_message)
            if original_df_before_step is not None:
                self.current_dataset = original_df_before_step
            self.grouping_variable = original_grouper

        finally:
            # ... (Final UI updates) ...
            self.analysis_text.setText(analysis)
            self.update_step_list_status()
            if self.current_dataset is not None:
                 self.display_dataset(self.current_dataset)
            items = self.steps_list.selectedItems()
            if self.step_statuses.get(step_id) == "Error" and items and items[0].data(Qt.ItemDataRole.UserRole) == step_id:
                 self.step_action_button.setEnabled(True)
            else:
                 self.step_action_button.setEnabled(False)
            self.display_tabs.setCurrentIndex(1)
    
    @asyncSlot()
    async def run_validate_columns(self):
        """Run Step 4: Validate columns for statistical analysis"""
        step_id = "validate_columns"
        analysis = ""
        original_df_before_step = self.current_dataset.copy() if self.current_dataset is not None else None
        code_executed_successfully = False

        try:
            self.update_status("Validating columns for analysis...")
            self.step_statuses[step_id] = "In Progress"
            self.step_status.setText("In Progress")
            self.update_step_list_status()
            self.step_action_button.setEnabled(False)

            analysis = "## Column Validation Analysis\\n\\n"
            
            # Add hypothesis to analysis if available
            if self.hypothesis:
                analysis += f"Using study hypothesis for context: \"{self.hypothesis}\"\\n\\n"
                
            df = self.current_dataset

            if df is None:
                 raise ValueError("Dataset not loaded.")

            # Get column info
            col_info = df.agg(['nunique', lambda x: x.isna().mean() * 100]).T
            col_info.columns = ['unique_count', 'missing_pct']
            col_info['dtype'] = df.dtypes
            
            # Use LLM to validate columns
            prompt = f"""
            I have a healthcare dataset prepared for analysis with {len(df)} rows and {len(df.columns)} columns.
            
            Column details:
            {col_info.to_string()}
            
            Sample data:
            {df.head(5).to_string()}
            """
            
            # Include hypothesis if available
            if self.hypothesis:
                prompt += f"""
                
                Study hypothesis: {self.hypothesis}
                
                Please consider this hypothesis when evaluating the columns and prioritize variables relevant to testing it.
                """
                
            prompt += f"""
            
            Please validate these columns for suitability in statistical analysis. Check for:
            
            1. Data Types: Identify columns with inappropriate data types (e.g., numbers stored as objects, categories as numbers). Suggest conversions.
            2. Missing Values: Flag columns with high percentages of missing data (e.g., > 30%). Suggest imputation or removal.
            3. Variance: Identify constant or near-constant columns (low variance). Suggest removal.
            4. Suitability: Recommend which columns are suitable for analysis as predictors or outcomes.
            5. Transformations: Suggest necessary transformations (e.g., scaling for numeric, encoding for categorical).
            """
            
            # If hypothesis is provided, add a specific instruction
            if self.hypothesis:
                prompt += f"""
            6. Hypothesis Relevance: Identify which columns are most relevant for testing the study hypothesis.
                """
                
            prompt += f"""
            
            Return the response in this JSON format:
            {{
                "column_issues": [
                    {{
                        "column": "column_name",
                        "issue_type": "DataType|MissingValue|LowVariance|Suitability",
                        "description": "Details of the issue",
                        "recommendation": "Suggested action (e.g., Convert to numeric, Remove, Impute, Scale, Encode)"
                    }},
                    ...
                ],
                "overall_recommendations": [
                    "General suggestions for column preparation"
                ],
                "code": "Python code to implement necessary basic transformations (e.g., type conversions) - ONLY IF ESSENTIAL and safe."
            """
            
            # Add hypothesis-specific field if available
            if self.hypothesis:
                prompt += """,
                "hypothesis_relevant_columns": [
                    {
                        "column": "column_name",
                        "relevance": "High|Medium|Low",
                        "reason": "Why this column is relevant to the hypothesis"
                    },
                    ...
                ]
            """
            
            prompt += """
            }
            """
            
            # Call LLM using the JSON-specific function
            validation_results = await call_llm_async_json(prompt, model=llm_config.default_json_model)
            
            if not isinstance(validation_results, dict):
                analysis += f"⚠️ **Error**: LLM did not return a valid JSON object for column validation. Response type: {type(validation_results)}\\nRaw Response:\\n```\\n{validation_results}\\n```\\n"
                raise TypeError("LLM response for validation is not a dictionary.")

            # --- Process the validation_results dictionary ---
            analysis += "### Column Issues Found\\n\\n"
            
            column_issues = validation_results.get("column_issues", [])
            
            if column_issues:
                for issue in column_issues:
                    column = issue.get("column", "")
                    issue_type = issue.get("issue_type", "")
                    description = issue.get("description", "")
                    recommendation = issue.get("recommendation", "")
                    
                    analysis += f"**{column}** ({issue_type})\\n"
                    analysis += f"- Issue: {description}\\n"
                    analysis += f"- Recommendation: {recommendation}\\n\\n"
            else:
                analysis += "✓ No major issues found with column formats or types.\\n\\n"
                
            # Add overall recommendations
            overall_recommendations = validation_results.get("overall_recommendations", [])
            
            if overall_recommendations:
                analysis += "### Overall Recommendations\\n\\n"
                for rec in overall_recommendations:
                    analysis += f"- {rec}\\n"
                analysis += "\\n"
                
            # Execute transformation code if provided
            transformation_code = validation_results.get("code", "")
            
            if transformation_code:
                analysis += f"### Applying Suggested Transformations\\n\\n"
                analysis += f"```python\\n{transformation_code}\\n```\\n\\n"
                
                # Create safe execution environment
                local_vars = {
                    "df": df.copy(),
                    "pd": pd,
                    "np": np
                }
                
                try:
                    # Execute the code
                    exec(transformation_code, {}, local_vars)
                    
                    # Get the result (should modify df in-place)
                    if "df" in local_vars and isinstance(local_vars["df"], pd.DataFrame):
                        transformed_df = local_vars["df"]
                        
                        # Check if DataFrame actually changed to avoid unnecessary updates
                        if not df.equals(transformed_df):
                            self.current_dataset = transformed_df # Update current dataset
                            self.display_dataset(transformed_df) # Update display immediately
                            analysis += "✓ Transformations applied successfully.\\n\\n"
                            code_executed_successfully = True
                        else:
                             analysis += "✓ Code executed, but no changes detected in the DataFrame.\\n\\n"
                             code_executed_successfully = True # Mark as success even if no change
                            
                    else:
                        analysis += "⚠️ **Error**: Transformation code did not produce a valid DataFrame in local_vars['df'].\\n\\n"
                        # Don't raise error, just report in analysis
                
                except Exception as e:
                    analysis += f"⚠️ **Error executing transformation code**: {str(e)}\\n\\n"
                    # Don't raise error, just report in analysis
            
            # Process hypothesis-relevant columns if available
            if self.hypothesis and "hypothesis_relevant_columns" in validation_results:
                relevant_columns = validation_results.get("hypothesis_relevant_columns", [])
                
                if relevant_columns:
                    analysis += "### Hypothesis-Relevant Columns\\n\\n"
                    analysis += "The following columns were identified as relevant to your hypothesis:\\n\\n"
                    
                    # Group by relevance for better organization
                    high_relevance = []
                    medium_relevance = []
                    low_relevance = []
                    
                    for col_info in relevant_columns:
                        column = col_info.get("column", "")
                        relevance = col_info.get("relevance", "")
                        reason = col_info.get("reason", "")
                        
                        if relevance == "High":
                            high_relevance.append((column, reason))
                        elif relevance == "Medium":
                            medium_relevance.append((column, reason))
                        elif relevance == "Low":
                            low_relevance.append((column, reason))
                    
                    if high_relevance:
                        analysis += "**High Relevance**\\n\\n"
                        for column, reason in high_relevance:
                            analysis += f"- **{column}**: {reason}\\n"
                        analysis += "\\n"
                        
                    if medium_relevance:
                        analysis += "**Medium Relevance**\\n\\n"
                        for column, reason in medium_relevance:
                            analysis += f"- **{column}**: {reason}\\n"
                        analysis += "\\n"
                        
                    if low_relevance:
                        analysis += "**Low Relevance**\\n\\n"
                        for column, reason in low_relevance:
                            analysis += f"- **{column}**: {reason}\\n"
                        analysis += "\\n"
                else:
                    analysis += "No columns specifically relevant to the hypothesis were identified.\\n\\n"
            
            # Validation step completes even if code fails, but logs issues
            self.step_statuses[step_id] = "Completed"
            self.step_status.setText("Completed")
            self.update_status("Column validation completed")
            # Store results AFTER potential modification by exec()
            self.step_results[step_id] = {
                "dataframe": self.current_dataset.copy(), # Store potentially modified state
                "analysis": analysis
            }
            # --- Save state AFTER successful completion ---
            self.save_current_dataset_state()
            self.select_next_step()

        except Exception as e:
            # Catch errors from LLM call or initial processing
            error_message = f"Error validating columns: {str(e)}"
            print(f"[ERROR] {error_message}")
            analysis += f"\\n--- ERROR --- \\n{error_message}\\n"
            self.step_statuses[step_id] = "Error"
            self.step_status.setText("Error")
            self.update_status(error_message)
            # Revert dataset only if error occurred before code execution attempt
            if original_df_before_step is not None and not code_executed_successfully: 
                self.current_dataset = original_df_before_step

        finally:
            # Always update UI elements at the end
            self.analysis_text.setText(analysis)
            self.update_step_list_status()
            # Update display in case of error/revert or if code modified df
            if self.current_dataset is not None:
                 self.display_dataset(self.current_dataset)

            # Re-enable button logic
            items = self.steps_list.selectedItems()
            if items and items[0].data(Qt.ItemDataRole.UserRole) == step_id and self.step_statuses[step_id] != "Completed":
                self.step_action_button.setEnabled(True)
            elif not items or items[0].data(Qt.ItemDataRole.UserRole) != step_id :
                 pass
            else: # Step is completed
                 self.step_action_button.setEnabled(False)
            self.display_tabs.setCurrentIndex(1)
    
    def select_next_step(self):
        """Select the next step in the timeline"""
        current_row = self.steps_list.currentRow()
        if current_row < self.steps_list.count() - 1:
            self.steps_list.setCurrentRow(current_row + 1)
            self.on_step_selected(self.steps_list.item(current_row + 1))

    def on_hypothesis_changed(self):
        """Save the hypothesis when it's changed by the user"""
        if hasattr(self, 'hypothesis_input'):
            self.hypothesis = self.hypothesis_input.toPlainText().strip()
            if self.current_dataset_name:
                # Just save the state whenever hypothesis changes
                self.save_current_dataset_state()
                self.update_status("Hypothesis updated")
                
    @asyncSlot()
    async def run_final_validation(self):
        """Run Step 5: Final validation of the dataset"""
        step_id = "final_validation"
        analysis = ""
        original_df_before_step = self.current_dataset.copy() if self.current_dataset is not None else None

        try:
            self.update_status("Performing final validation...")
            self.step_statuses[step_id] = "In Progress"
            self.step_status.setText("In Progress")
            self.update_step_list_status()
            self.step_action_button.setEnabled(False)

            analysis = "## Final Dataset Validation\\n\\n"

            df = self.current_dataset
            if df is None: raise ValueError("Dataset not loaded.")

            # Check if previous steps are completed and required info is available
            if self.step_statuses.get("identify_patient_id") != "Completed":
                 analysis += "⚠️ **Prerequisite Error**: Step 1 (Identify Patient ID) not completed.\\n"
                 raise ValueError("Step 1 not completed.")
            if self.step_statuses.get("normalize_rows") != "Completed":
                 analysis += "⚠️ **Prerequisite Error**: Step 2 (Normalize Rows) not completed.\\n"
                 raise ValueError("Step 2 not completed.")
            if self.step_statuses.get("check_grouper") != "Completed":
                 analysis += "⚠️ **Prerequisite Error**: Step 3 (Check Grouping Variable) not completed.\\n"
                 raise ValueError("Step 3 not completed.")
            if self.step_statuses.get("validate_columns") != "Completed":
                 analysis += "⚠️ **Prerequisite Error**: Step 4 (Validate Columns) not completed.\\n"
                 raise ValueError("Step 4 not completed.")

            if not self.patient_id_column:
                 analysis += "⚠️ **Error**: Patient Identifier is missing.\\n"
                 raise ValueError("Patient Identifier missing.")
            if not self.grouping_variable:
                 analysis += "⚠️ **Error**: Grouping Variable is missing.\\n"
                 raise ValueError("Grouping Variable missing.")

            analysis += f"Using Patient Identifier: `{self.patient_id_column}`\\n"
            analysis += f"Using Grouping Variable: `{self.grouping_variable}`\\n\\n"

            # Get basic info
            rows, cols = df.shape
            missing_pct = (df.isna().sum().sum() / (rows * cols)) * 100 if rows * cols > 0 else 0
            group_counts = df[self.grouping_variable].value_counts()
            balance_ratio = 0.0
            if len(group_counts) >= 2:
                min_count = group_counts.min()
                max_count = group_counts.max()
                balance_ratio = min_count / max_count if max_count > 0 else 0

            # Use LLM for final assessment
            prompt = f"""
            I have a preprocessed healthcare dataset with {rows} rows and {cols} columns, ready for final validation before statistical analysis.

            Key Information:
            - Patient Identifier: {self.patient_id_column}
            - Grouping Variable: {self.grouping_variable}
            - Missing Data: {missing_pct:.1f}% overall
            - Group Distribution ({self.grouping_variable}):
            {group_counts.to_string()}
            - Group Balance Ratio (min/max): {balance_ratio:.2f} (if applicable)

            Sample Data:
            {df.head(5).to_string()}
            """

            if self.hypothesis:
                prompt += f"""
                Study Hypothesis: {self.hypothesis}
                """

            prompt += f"""

            Please perform a final assessment of this dataset's readiness for statistical analysis, considering the hypothesis if provided.
            Evaluate:
            1. Final check for any critical issues (e.g., excessive missing data, severe imbalance).
            2. Suitability of the grouping variable for the analysis.
            3. Overall readiness for statistical modeling or hypothesis testing.
            4. Assign a readiness score from 1 (Not Ready) to 10 (Fully Ready).

            Return the response in this JSON format:
            {{
                "assessment_summary": "A brief text summary of the dataset's readiness.",
                "issues_found": [
                    {{"severity": "Critical|Warning|Info", "description": "Description of the issue"}}
                ],
                "grouping_variable_assessment": "Evaluation of the grouping variable's suitability.",
                "readiness_score": number (1-10),
                "recommendations": ["Final recommendations before analysis"]
            }}
            """

            validation_results = await call_llm_async_json(prompt, model=llm_config.default_json_model)

            if not isinstance(validation_results, dict):
                analysis += f"⚠️ **Error**: LLM did not return valid JSON for final validation. Response:\\n```\\n{validation_results}\\n```\\n"
                raise TypeError("LLM response for final validation is not a dictionary.")

            # --- Process LLM results ---
            summary = validation_results.get("assessment_summary", "No summary provided.")
            issues = validation_results.get("issues_found", [])
            grouper_eval = validation_results.get("grouping_variable_assessment", "No assessment.")
            score = validation_results.get("readiness_score", "N/A")
            recommendations = validation_results.get("recommendations", [])

            analysis += f"### LLM Assessment Summary\\n\\n{summary}\\n\\n"

            if issues:
                analysis += "### Issues Found\\n\\n"
                for issue in issues:
                     severity = issue.get('severity', 'Info')
                     desc = issue.get('description', 'No description.')
                     analysis += f"- **[{severity}]**: {desc}\\n"
                analysis += "\\n"
            else:
                 analysis += "✓ No critical issues identified by LLM.\\n\\n"

            analysis += f"### Grouping Variable Assessment\\n\\n{grouper_eval}\\n\\n"

            analysis += f"### Readiness Score (from LLM)\\n\\n**{score} / 10**\\n\\n"

            if recommendations:
                analysis += "### Final Recommendations\\n\\n"
                for rec in recommendations:
                     analysis += f"- {rec}\\n"
                analysis += "\\n"

            # Step completion
            self.step_statuses[step_id] = "Completed"
            self.step_status.setText("Completed")
            self.update_status("Final validation complete.")
            self.step_results[step_id] = {
                "dataframe": self.current_dataset.copy(),
                "analysis": analysis
            }
            self.save_current_dataset_state()
            # Potentially update overall status label based on score here
            self.update_overall_status_label() # Add call to update main status


        except Exception as e:
            error_message = f"Error during final validation: {str(e)}"
            print(f"[ERROR] {error_message}")
            analysis += f"\\n--- ERROR --- \\n{error_message}\\n"
            self.step_statuses[step_id] = "Error"
            self.step_status.setText("Error")
            self.update_status(error_message)
            # Revert dataset not needed here as no modifications are made

        finally:
            self.analysis_text.setText(analysis)
            self.update_step_list_status()
            if self.current_dataset is not None:
                 self.display_dataset(self.current_dataset) # Refresh display

            # Re-enable button only if error occurred
            items = self.steps_list.selectedItems()
            if self.step_statuses.get(step_id) == "Error" and items and items[0].data(Qt.ItemDataRole.UserRole) == step_id:
                 self.step_action_button.setEnabled(True)
            else:
                 self.step_action_button.setEnabled(False) # Disable on complete or if selection changed
            self.display_tabs.setCurrentIndex(1) # Show analysis tab

    def update_overall_status_label(self):
        """Updates the main preprocessing status label based on final validation."""
        if all(status == "Completed" for status in self.step_statuses.values()):
             # Check readiness score from final validation if available
             final_val_results = self.step_results.get("final_validation", {}).get("analysis", "")
             # Improved check for readiness score in text
             score_line = next((line for line in final_val_results.split('\\n') if "Readiness Score (from LLM)" in line), None)
             if score_line:
                  try:
                      # Extract score: find the bold part "**score / 10**"
                      score_part = score_line.split("**")[1].split("/")[0].strip()
                      score = int(score_part)
                      if score >= 7: # Assuming 7+ is ready
                           self.preprocessing_status_label.setText(f"Status: Ready for Analysis ({score}/10)")
                           self.preprocessing_status_label.setStyleSheet("color: #28a745;") # Green
                      else:
                           self.preprocessing_status_label.setText(f"Status: Partially Ready ({score}/10)")
                           self.preprocessing_status_label.setStyleSheet("color: #ffc107;") # Yellow
                  except (IndexError, ValueError, TypeError) as e:
                      print(f"[WARN] Could not parse readiness score from line: '{score_line}'. Error: {e}")
                      self.preprocessing_status_label.setText("Status: Processed (Score Parse Error)")
                      self.preprocessing_status_label.setStyleSheet("color: #17a2b8;") # Info blue
             else:
                  self.preprocessing_status_label.setText("Status: Processed (Score N/A)")
                  self.preprocessing_status_label.setStyleSheet("color: #17a2b8;") # Info blue
        elif any(status != "Not Started" for status in self.step_statuses.values()):
             self.preprocessing_status_label.setText("Status: Partially Processed")
             self.preprocessing_status_label.setStyleSheet("color: #ffc107;") # Yellow/amber
        else:
             self.preprocessing_status_label.setText("Status: Not Processed")
             self.preprocessing_status_label.setStyleSheet("color: #6c757d;") # Gray


