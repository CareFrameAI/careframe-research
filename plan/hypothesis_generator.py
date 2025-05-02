import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import uuid
from datetime import datetime

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QStackedWidget,
    QSplitter, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QFormLayout, QTextEdit, QDialog, QDialogButtonBox, QMessageBox,
    QLineEdit
)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette

from helpers.load_icon import load_bootstrap_icon
import logging
import asyncio

from plan.variable_selection import HypothesisVariableSelector

logger = logging.getLogger(__name__)

class TimelineStep(QWidget):
    """A single step in the hypothesis generation timeline"""
    
    clicked = pyqtSignal(int)  # Signal when step is clicked with step index
    
    def __init__(self, step_number: int, title: str, is_active: bool = False, parent=None):
        super().__init__(parent)
        self.step_number = step_number
        self.title = title
        self.is_active = is_active
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Step number circle
        self.number_label = QLabel(str(self.step_number))
        self.number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.number_label.setFixedSize(28, 28)
        self.number_label.setStyleSheet("""
            background-color: #dddddd;
            border-radius: 14px;
            color: #333333;
            font-weight: bold;
        """)
        layout.addWidget(self.number_label)
        
        # Step title
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("""
            padding-left: 5px;
            font-size: 14px;
        """)
        layout.addWidget(self.title_label)
        
        # Set initial active state
        self.set_active(self.is_active)
        
        # Make the widget clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def set_active(self, active: bool):
        """Set this step as active or inactive"""
        self.is_active = active
        
        if active:
            self.number_label.setStyleSheet("""
                background-color: #4CAF50;
                border-radius: 14px;
                color: white;
                font-weight: bold;
            """)
            self.title_label.setStyleSheet("""
                padding-left: 5px;
                font-size: 14px;
                font-weight: bold;
                color: #4CAF50;
            """)
        else:
            self.number_label.setStyleSheet("""
                background-color: #dddddd;
                border-radius: 14px;
                color: #333333;
                font-weight: bold;
            """)
            self.title_label.setStyleSheet("""
                padding-left: 5px;
                font-size: 14px;
                font-weight: normal;
                color: #333333;
            """)
    
    def mousePressEvent(self, event):
        """Handle mouse press to emit clicked signal"""
        self.clicked.emit(self.step_number)
        super().mousePressEvent(event)


class TimelineWidget(QWidget):
    """Widget displaying the timeline of hypothesis generation steps"""
    
    step_changed = pyqtSignal(int)  # Signal when active step changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = []
        self.active_step = 1
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(5)
        
        # Create steps
        step_titles = [
            "Data Sources",
            "Model Analysis",
            # "Relationship Analysis",
            # "Hypotheses Formulation"
        ]
        
        for i, title in enumerate(step_titles):
            step_number = i + 1
            step = TimelineStep(step_number, title, step_number == self.active_step)
            step.clicked.connect(self.on_step_clicked)
            layout.addWidget(step)
            
            # Add connector line except after the last step
            if i < len(step_titles) - 1:
                connector = QLabel("→")
                connector.setStyleSheet("color: #888888; font-size: 16px;")
                layout.addWidget(connector)
            
            # Store step widget
            self.steps.append(step)
        
        layout.addStretch()
    
    def on_step_clicked(self, step_number: int):
        """Handle click on timeline step"""
        if step_number != self.active_step:
            self.set_active_step(step_number)
    
    def set_active_step(self, step_number: int):
        """Set the active step"""
        if 1 <= step_number <= len(self.steps):
            # Update old active step
            if 1 <= self.active_step <= len(self.steps):
                self.steps[self.active_step - 1].set_active(False)
            
            # Set new active step
            self.active_step = step_number
            self.steps[step_number - 1].set_active(True)
            
            # Emit the step_changed signal
            self.step_changed.emit(step_number)


class DataSourceItem(QWidget):
    """Widget representing a selectable data source with metadata and preview."""
    
    # Signal emitted when the data source selection changes
    toggled = pyqtSignal(str, bool)  # (source_name, is_selected)
    
    def __init__(self, name, dataframe, metadata=None, parent=None):
        """Initialize the data source item widget.
        
        Args:
            name (str): Name of the data source
            dataframe (DataFrame): The pandas DataFrame
            metadata (dict): Additional metadata about the dataset
            parent: Parent widget
        """
        super().__init__(parent)
        self.name = name
        self.dataframe = dataframe
        self.metadata = metadata or {}
        
        self.setObjectName(f"data-source-{name}")
        self.setMinimumHeight(80)
        self.setMaximumHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Add a subtle highlight effect on hover
        self.setAutoFillBackground(True)
        self.normal_palette = self.palette()
        self.hovered_palette = QPalette(self.normal_palette)
        self.hovered_palette.setColor(QPalette.ColorRole.Window, QColor("#f5f5f5"))
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Checkbox for selection
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(self._on_checkbox_changed)
        main_layout.addWidget(self.checkbox)
        
        # Content area with name and details
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        
        # Data source name
        name_label = QLabel(self.name)
        name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        content_layout.addWidget(name_label)
        
        # Info layout with row and column counts
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(15)
        
        # Row count
        rows = len(self.dataframe)
        row_label = QLabel(f"{rows:,} {'row' if rows == 1 else 'rows'}")
        row_label.setStyleSheet("color: #666;")
        info_layout.addWidget(row_label)
        
        # Column count
        cols = len(self.dataframe.columns)
        col_label = QLabel(f"{cols:,} {'column' if cols == 1 else 'columns'}")
        col_label.setStyleSheet("color: #666;")
        info_layout.addWidget(col_label)
        
        # Add any tags or metadata indicators
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, bool) and value:
                    tag = QLabel(key.replace("_", " ").title())
                    tag.setStyleSheet("background-color: #e0e0e0; color: #555; padding: 2px 6px; border-radius: 3px;")
                    info_layout.addWidget(tag)
        
        info_layout.addStretch()
        content_layout.addLayout(info_layout)
        
        main_layout.addLayout(content_layout, 1)
        
        # Preview button
        self.preview_button = QPushButton()
        self.preview_button.setIcon(load_bootstrap_icon("eye", "#555555"))
        self.preview_button.setFixedSize(32, 32)
        self.preview_button.setToolTip("Preview data")
        self.preview_button.setStyleSheet("background-color: #f0f0f0; border: none; border-radius: 16px;")
        self.preview_button.setCursor(Qt.CursorShape.ArrowCursor)
        main_layout.addWidget(self.preview_button)
    
    def _on_checkbox_changed(self, state):
        """Handle checkbox state changes.
        
        Args:
            state (int): The new checkbox state
        """
        is_selected = state == Qt.CheckState.Checked
        self.toggled.emit(self.name, is_selected)
    
    def set_selected(self, selected):
        """Set the selection state of this item.
        
        Args:
            selected (bool): Whether this item should be selected
        """
        # Block signals to avoid emitting toggled twice
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(selected)
        self.checkbox.blockSignals(False)
        
        # Emit signal manually since we blocked signals
        self.toggled.emit(self.name, selected)
    
    def is_selected(self):
        """Return whether this data source is selected.
        
        Returns:
            bool: Whether this data source is selected
        """
        return self.checkbox.isChecked()
    
    def enterEvent(self, event):
        """Handle mouse enter events.
        
        Args:
            event: QEvent instance
        """
        self.setPalette(self.hovered_palette)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events.
        
        Args:
            event: QEvent instance
        """
        self.setPalette(self.normal_palette)
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press events to toggle selection.
        
        Args:
            event: QEvent instance
        """
        # Toggle checkbox when clicking anywhere except the preview button
        if not self.preview_button.geometry().contains(event.pos()):
            self.checkbox.setChecked(not self.checkbox.isChecked())
            # Explicitly emit the toggled signal to ensure selection is captured
            self.toggled.emit(self.name, self.checkbox.isChecked())
        
        super().mousePressEvent(event)
    
    def show_preview(self):
        """Show a preview dialog of the dataset."""
        
        # Create a dialog to preview the dataset
        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle(f"Preview: {self.name}")
        preview_dialog.resize(800, 600)
        
        # Create layout
        layout = QVBoxLayout(preview_dialog)
        
        # Add header
        header = QLabel(f"Dataset: {self.name}")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Add info about the dataframe
        info_text = f"Rows: {len(self.dataframe)}, Columns: {len(self.dataframe.columns)}"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        # Create a table view to display the data
        table = QTableWidget()
        table.setRowCount(min(100, len(self.dataframe)))  # Limit to 100 rows for performance
        table.setColumnCount(len(self.dataframe.columns))
        
        # Set headers
        table.setHorizontalHeaderLabels(self.dataframe.columns)
        
        # Populate the table with data
        for row in range(min(100, len(self.dataframe))):
            for col, column_name in enumerate(self.dataframe.columns):
                value = str(self.dataframe.iloc[row, col])
                item = QTableWidgetItem(value)
                table.setItem(row, col, item)
        
        # Add the table to the layout
        layout.addWidget(table, 1)  # Give the table a stretch factor of 1
        
        # Add a close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(preview_dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Show the dialog
        preview_dialog.exec()


class DataSourcesStepWidget(QWidget):
    """Widget for selecting data sources in a hypothesis generator workflow."""
    
    # Signal emitted when data sources are selected
    # Emits a list of (name, dataframe) tuples for the selected sources
    sources_selected = pyqtSignal(list)
    
    # Signal emitted when the user wants to proceed to the next step
    next_step = pyqtSignal()
    
    def __init__(self, data_sources=None, parent=None):
        """Initialize the data sources step widget.
        
        Args:
            data_sources (dict): Dictionary mapping data source names to dataframes
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Store data sources and currently selected sources
        self.data_sources = data_sources or {}
        self.selected_sources = set()  # Use a set for efficient lookups
        self.source_items = {}  # Map source names to DataSourceItem widgets
        
        self.init_ui()
        
        # Populate with data sources if provided
        if self.data_sources:
            for name, df in self.data_sources.items():
                self.add_source_item(name, df)
                
        # Update the selection summary
        self.update_selection_summary()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        
        # Header with title and search
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Available Data Sources")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title_label)
        
        # Spacer
        header_layout.addStretch()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search datasets...")
        self.search_box.setFixedWidth(200)
        self.search_box.textChanged.connect(self.filter_sources)
        self.search_box.setClearButtonEnabled(True)
        header_layout.addWidget(self.search_box)
        
        main_layout.addLayout(header_layout)
        
        # Selection summary
        self.selection_summary = QLabel("")
        self.selection_summary.setStyleSheet("color: #666; margin-bottom: 5px;")
        main_layout.addWidget(self.selection_summary)
        
        # Data sources container with scrolling
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container for data source items
        self.sources_container = QWidget()
        self.sources_layout = QVBoxLayout(self.sources_container)
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_layout.setSpacing(10)
        self.sources_layout.addStretch()  # Push items to the top
        
        self.scroll_area.setWidget(self.sources_container)
        main_layout.addWidget(self.scroll_area, 1)  # 1 = stretch factor
        
        # Create a bottom toolbar with continue button
        bottom_toolbar = QWidget()
        toolbar_layout = QHBoxLayout(bottom_toolbar)
        toolbar_layout.setContentsMargins(0, 10, 0, 0)
        
        # Add spacer to push the button to the right
        toolbar_layout.addStretch()
        
        # Add the continue button
        self.continue_btn = QPushButton("Continue →")
        self.continue_btn.setEnabled(False)  # Disabled until a dataset is selected
        # Connect the continue button to emit the next_step signal
        self.continue_btn.clicked.connect(self.next_step.emit)
        toolbar_layout.addWidget(self.continue_btn)
        
        # Add the toolbar to the main layout
        main_layout.addWidget(bottom_toolbar)
    
    def add_source_item(self, name, dataframe, metadata=None):
        """Add a data source item to the widget.
        
        Args:
            name (str): Name of the data source
            dataframe (DataFrame): Pandas DataFrame
            metadata (dict): Optional metadata about the dataset
        """
        # Create the data source item widget
        source_item = DataSourceItem(name, dataframe, metadata)
        source_item.toggled.connect(self.on_source_toggled)
        
        # Add to layout - at the beginning, before the stretch
        self.sources_layout.insertWidget(self.sources_layout.count() - 1, source_item)
        
        # Store reference
        self.source_items[name] = source_item
        
        # Connect preview button
        source_item.preview_button.clicked.connect(
            lambda: self.show_preview(name, dataframe)
        )
    
    def on_source_toggled(self, source_name, selected):
        """Handle toggling of a data source's selection.
        
        Args:
            source_name (str): Name of the toggled data source
            selected (bool): Whether the source is now selected
        """
        logger.debug(f"Source toggled: {source_name}, selected: {selected}")
        
        # Update internal selection state
        if selected:
            self.selected_sources.add(source_name)
        else:
            self.selected_sources.discard(source_name)
        
        logger.debug(f"Updated selected_sources set: {self.selected_sources}")
        
        # Update the selection summary
        self.update_selection_summary()
        
        # Emit selected sources
        self.emit_selected_sources()
        
        # Enable or disable the continue button based on selection
        if hasattr(self, 'continue_btn'):
            has_selection = len(self.selected_sources) > 0
            logger.debug(f"Setting continue button enabled: {has_selection}")
            self.continue_btn.setEnabled(has_selection)
    
    def update_selection_summary(self):
        """Update the selection summary label."""
        count = len(self.selected_sources)
        if count == 0:
            summary = "No datasets selected."
        elif count == 1:
            summary = "1 dataset selected."
        else:
            summary = f"{count} datasets selected."
        
        self.selection_summary.setText(summary)
    
    def emit_selected_sources(self):
        """Emit the sources_selected signal with the currently selected sources."""
        logger.debug(f"Emitting source selection: {list(self.selected_sources)}")
        
        # Get actual (name, dataframe) tuples for selected sources
        selected_data = self.get_selected_sources()
        
        logger.debug(f"Emitting {len(selected_data)} selected data sources")
        
        # Only emit sources that actually exist in our data sources
        self.sources_selected.emit(selected_data)
    
    def filter_sources(self, query):
        """Filter displayed sources based on search query.
        
        Args:
            query (str): The search query
        """
        query = query.lower()
        
        # Show/hide source items based on whether they match the query
        for name, item in self.source_items.items():
            match = not query or query in name.lower()
            item.setVisible(match)
    
    def get_selected_sources(self):
        """Get the list of selected data sources.
        
        Returns:
            list: List of (name, dataframe) tuples for selected sources
        """
        selected = []
        logger.debug(f"Getting selected sources from names: {self.selected_sources}")
        logger.debug(f"Available data sources: {list(self.data_sources.keys())}")
        
        for name in self.selected_sources:
            if name in self.data_sources:
                selected.append((name, self.data_sources[name]))
            else:
                logger.warning(f"Selected source '{name}' not found in data_sources dictionary")
        
        logger.debug(f"Returning selected sources: {[name for name, _ in selected]}")
        return selected
    
    def set_data_sources(self, data_sources):
        """Set or update the available data sources.
        
        Args:
            data_sources (dict): Dictionary mapping source names to dataframes
        """
        # Store new data sources
        self.data_sources = data_sources
        logger.debug(f"Setting data sources: {list(data_sources.keys())}")
        
        # Clear existing items
        for i in reversed(range(self.sources_layout.count())):
            item = self.sources_layout.itemAt(i)
            if item and item.widget():
                item.widget().deleteLater()
        
        # Reset source items dict
        self.source_items = {}
        
        # Add stretch for proper layout
        self.sources_layout.addStretch()
        
        # Add new data sources
        if self.data_sources:
            logger.debug(f"Adding {len(self.data_sources)} data sources to UI")
            for name, df in self.data_sources.items():
                self.add_source_item(name, df)
        
        # Clear selection since sources changed
        self.selected_sources.clear()
        self.update_selection_summary()
        logger.debug("Cleared selected sources after setting new data sources")
        
        # Emit signal with empty selection
        self.emit_selected_sources()
    
    def show_preview(self, name, dataframe):
        """Show a preview of the data source.
        
        Args:
            name (str): Name of the data source
            dataframe (DataFrame): The data to preview
        """
        # This would typically show a dialog with a table preview
        # Implement or connect to your data preview functionality
        pass
    
    def set_studies_manager(self, studies_manager):
        """Set the studies manager and load available datasets.
        
        Args:
            studies_manager: The studies manager object to get data from
        """
        # Store the studies manager
        self.studies_manager = studies_manager
        
        # Get datasets from the active study
        if hasattr(studies_manager, 'get_datasets_from_active_study'):
            datasets = studies_manager.get_datasets_from_active_study()
            
            if datasets:
                # Convert to the expected format (dictionary mapping names to dataframes)
                data_sources = {name: df for name, df in datasets}
                
                # Update the widget with these datasets
                self.set_data_sources(data_sources)
            else:
                logger.debug("No datasets found in active study")
        else:
            logger.debug("Studies manager does not provide get_datasets_from_active_study method")

    def on_sources_selected(self, selected_sources):
        """Handle selection of data sources
        
        Args:
            selected_sources (list): List of (name, dataframe) tuples for selected sources
        """
        logger.debug(f"HypothesisGeneratorWidget received sources selected: {[name for name, _ in selected_sources]}")
        
        # Store the selected datasets
        self.selected_datasets = selected_sources
        
        # Update the data_sources_step's continue button if it exists
        has_selection = len(selected_sources) > 0
        logger.debug(f"Selection status: {has_selection} ({len(selected_sources)} datasets)")
        
        if hasattr(self, 'data_sources_step') and hasattr(self.data_sources_step, 'continue_btn'):
            self.data_sources_step.continue_btn.setEnabled(has_selection)


class HypothesisGeneratorWidget(QWidget):
    """Main widget for generating research hypotheses based on available data sources"""
    
    hypothesis_tested = pyqtSignal(str, dict)  # Signal when a hypothesis has been tested (text, results)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.step_widgets = {}
        self.current_step = 1
        self.studies_manager = None
        self.selected_datasets = []
        self.testing_widget = None
        self.current_hypothesis_text = "Higher blood pressure leads to increased risk of heart disease"
        self.current_test_results = None
        self.status_bar = None  # Will store reference to status bar if available
        
        # Initialize the UI
        self.init_ui()
        
        # Attempt to find and set up the testing widget
        self.initialize_testing_widget()
    
    def initialize_testing_widget(self):
        """Try to initialize the testing widget from the main window"""
        # Only do this if testing_widget is not already set
        if self.testing_widget is not None:
            print(f"Testing widget already initialized: {self.testing_widget}")
            return True
            
        # Try to find the app instance first
        app_instance = self.get_app_instance()
        if app_instance and hasattr(app_instance, 'data_testing_widget') and app_instance.data_testing_widget:
            self.testing_widget = app_instance.data_testing_widget
            print(f"Found testing widget in app_instance: {self.testing_widget}")
            
            # Initialize in variable_selection_step if it exists
            self._set_testing_widget_in_step()
            return True
            
        # Fallback: try to get it from main window
        try:
            main_window = self.window()
            if main_window and hasattr(main_window, 'data_testing_widget') and main_window.data_testing_widget:
                self.testing_widget = main_window.data_testing_widget
                print(f"Found testing widget in main_window: {self.testing_widget}")
                
                # Initialize in variable_selection_step if it exists
                self._set_testing_widget_in_step()
                return True
        except Exception as e:
            print(f"Error finding testing_widget in main window: {e}")
            
        return False
        
    def _set_testing_widget_in_step(self):
        """Helper method to set testing_widget in variable_selection_step"""
        if hasattr(self, 'variable_selection_step'):
            # Set the testing widget directly
            self.variable_selection_step.testing_widget = self.testing_widget
            
            # Also initialize the selector
            if hasattr(self.variable_selection_step, 'setup_selector'):
                self.variable_selection_step.setup_selector(self.testing_widget)
                print(f"Set up testing_widget in variable_selection_step: {self.testing_widget}")
            
            # Re-set data if already selected
            if self.selected_datasets and len(self.selected_datasets) > 0:
                dataset_name, _ = self.selected_datasets[0]
                self.variable_selection_step.set_data(dataset_name, self.current_hypothesis_text)
                print(f"Updated Model Analysis with dataset {dataset_name}")
        else:
            print("Warning: variable_selection_step not created yet in _set_testing_widget_in_step")
    
    def init_ui(self):
        # Use the existing layout instead of creating a new one
        layout = self.main_layout
        
        # Header
        header = QLabel("Hypothesis Generator")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)
        
        # Timeline
        self.timeline = TimelineWidget()
        self.timeline.step_changed.connect(self.on_step_changed)
        layout.addWidget(self.timeline)
        
        # Step content container
        self.step_content = QStackedWidget()
        layout.addWidget(self.step_content, 1)
        
        # Create step widgets
        self.create_step_widgets()
        
        # Set initial step
        self.step_content.setCurrentIndex(0)
    
    def create_step_widgets(self):
        """Create widgets for all timeline steps"""
        # Step 1: Data Sources
        self.data_sources_step = DataSourcesStepWidget()
        self.data_sources_step.sources_selected.connect(self.on_sources_selected)
        self.data_sources_step.next_step.connect(lambda: self.timeline.set_active_step(2))
        self.step_content.addWidget(self.data_sources_step)
        
        # Step 2: Model Analysis
        from plan.variable_selection import VariableSelectionWidget
        
        self.variable_selection_step = VariableSelectionWidget()
        self.variable_selection_step.test_completed.connect(self.on_test_completed)
        self.variable_selection_step.next_step.connect(lambda: self.timeline.set_active_step(3))
        self.step_content.addWidget(self.variable_selection_step)
        
        # If testing_widget is already set, connect it to the variable_selection_step
        if self.testing_widget:
            self._set_testing_widget_in_step()
        
        # Step 3: Relationship Analysis (placeholder)
        step3_widget = QWidget()
        step3_layout = QVBoxLayout(step3_widget)
        
        # Add header
        header = QLabel("Step 3: Relationship Analysis")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        step3_layout.addWidget(header)
        
        # Add description
        description = QLabel("This step will analyze the relationships between variables in your dataset "
                            "to identify potential confounders, mediators, and moderators.")
        description.setWordWrap(True)
        step3_layout.addWidget(description)
        
        # Add "coming soon" label
        coming_soon = QLabel("This feature is coming soon. Please check back later.")
        coming_soon.setStyleSheet("color: #666; font-style: italic; margin-top: 20px;")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step3_layout.addWidget(coming_soon)
        
        # Add continue button that does nothing yet
        continue_btn = QPushButton("Continue to Step 4")
        continue_btn.setEnabled(False)
        step3_layout.addWidget(continue_btn)
        
        # Add stretch to push everything to the top
        step3_layout.addStretch()
        
        # self.step_content.addWidget(step3_widget)
        
        # Step 4: Hypotheses Formulation (placeholder)
        step4_widget = QWidget()
        step4_layout = QVBoxLayout(step4_widget)
        
        # Add header
        header = QLabel("Step 4: Hypotheses Formulation")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        step4_layout.addWidget(header)
        
        # Add description
        description = QLabel("This step will help you formulate and refine multiple hypotheses "
                            "based on your data analysis, variable relationships, and existing research.")
        description.setWordWrap(True)
        step4_layout.addWidget(description)
        
        # Add "coming soon" label
        coming_soon = QLabel("This feature is coming soon. Please check back later.")
        coming_soon.setStyleSheet("color: #666; font-style: italic; margin-top: 20px;")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step4_layout.addWidget(coming_soon)
        
        # Add finish button that does nothing yet
        finish_btn = QPushButton("Finish")
        finish_btn.setEnabled(False)
        step4_layout.addWidget(finish_btn)
        
        # Add stretch to push everything to the top
        step4_layout.addStretch()
        
        # self.step_content.addWidget(step4_widget)
    
    def set_studies_manager(self, studies_manager):
        """Set the studies manager and initialize data"""
        self.studies_manager = studies_manager
        
        # Pass the studies manager to our data sources step
        self.data_sources_step.set_studies_manager(studies_manager)
        
        # Find the app instance and get the existing DataTestingWidget
        app_instance = self.get_app_instance()
        if app_instance and hasattr(app_instance, 'data_testing_widget'):
            # Use the app's existing DataTestingWidget
            self.testing_widget = app_instance.data_testing_widget
            print(f"Found DataTestingWidget in app instance: {self.testing_widget}")
            
            # Now set up the selector in the Model Analysis step
            if hasattr(self, 'variable_selection_step'):
                self.variable_selection_step.setup_selector(self.testing_widget)
                print(f"Set up variable_selection_step selector with testing_widget: {self.testing_widget}")
        else:
            print("Could not find DataTestingWidget in app instance - will try fallback methods")
            # Try to access it via the main window
            try:
                from PyQt6.QtWidgets import QApplication
                main_window = QApplication.instance().activeWindow()
                if hasattr(main_window, 'data_testing_widget'):
                    self.testing_widget = main_window.data_testing_widget
                    print(f"Found DataTestingWidget in main window: {self.testing_widget}")
                    
                    # Set up the selector in the Model Analysis step
                    if hasattr(self, 'variable_selection_step'):
                        self.variable_selection_step.setup_selector(self.testing_widget)
                        print(f"Set up variable_selection_step selector with testing_widget from main window")
            except Exception as e:
                print(f"Error trying to find DataTestingWidget: {e}")
    
    def set_status_bar(self, status_bar):
        """Set the status bar reference for showing messages"""
        self.status_bar = status_bar
        
    def set_testing_widget(self, testing_widget):
        """Set the testing widget reference"""
        if testing_widget is None:
            print("Warning: set_testing_widget called with None")
            return
            
        # Store the testing widget directly
        self.testing_widget = testing_widget
        print(f"Set testing_widget: {testing_widget}")
        
        # Use helper method to set it in the variable_selection_step
        self._set_testing_widget_in_step()
    
    def get_app_instance(self):
        """Helper method to get the main app instance"""
        # Find the parent app instance
        parent = self.parent()
        while parent:
            # Check if this is the main app (has data_testing_widget attribute)
            if hasattr(parent, 'data_testing_widget'):
                return parent
            parent = parent.parent()
        return None
    
    def set_app_instance(self, app_instance):
        """Set the main application instance"""
        self.app_instance = app_instance
        # Try to get a reference to the status bar if app_instance is a MainWindow
        if hasattr(app_instance, 'statusBar'):
            self.status_bar = app_instance.statusBar()
    
    def show_status_message(self, message):
        """Safely show a status message if a status bar is available"""
        try:
            if self.status_bar:
                self.status_bar.showMessage(message)
                return
                
            # Try to get status bar from parent
            if self.parent() and hasattr(self.parent(), 'statusBar'):
                self.parent().statusBar().showMessage(message)
                return
                
            # Try to get from main window
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().showMessage(message)
                return
                
            # Log the message instead
            print(f"Status: {message}")
        except Exception as e:
            # If all else fails, just print to console
            print(f"Status (error showing message): {message}, Error: {str(e)}")

    def clear_status_message(self):
        """Safely clear the status message if a status bar is available"""
        try:
            if self.status_bar:
                self.status_bar.clearMessage()
                return
                
            if self.parent() and hasattr(self.parent(), 'statusBar'):
                self.parent().statusBar().clearMessage()
                return
                
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().clearMessage()
        except Exception as e:
            # Just log the error
            print(f"Error clearing status message: {str(e)}")

    def set_hypothesis_text(self, text):
        """Set the hypothesis text for testing"""
        self.current_hypothesis_text = text
        
        # If we're on the Model Analysis step, update it
        if hasattr(self, 'variable_selection_step') and self.timeline.active_step == 2:
            if self.selected_datasets:  # Make sure we have a dataset selected
                self.variable_selection_step.set_data(self.selected_datasets[0], text)
    
    def on_step_changed(self, step_number: int):
        """Handle changing the active step in the timeline"""
        # Print debug info
        print(f"Step changed to {step_number}, selected datasets: {self.selected_datasets}")
        
        # Get the current step widget
        current_widget = self.step_content.widget(step_number - 1)
        
        # Update the UI
        self.step_content.setCurrentIndex(step_number - 1)
        
        # Perform any needed initialization when advancing to a step
        if step_number == 2:  # Model Analysis step
            # Make sure our selected datasets are available for subsequent steps
            if not self.selected_datasets and hasattr(self, 'data_sources_step'):
                # Try to get selected datasets from the data sources step
                selected = self.data_sources_step.get_selected_sources()
                if selected:
                    self.selected_datasets = selected
                    print(f"Retrieved datasets from step: {[name for name, _ in self.selected_datasets]}")
            
            # Now setup the Model Analysis step
            self.setup_variable_selection()
    
    def setup_variable_selection(self):
        """Setup the Model Analysis step with the selected data sources"""
        print(f"Setting up Model Analysis with datasets: {self.selected_datasets}")
        
        # If no datasets are selected but we have a data sources step, try to get them
        if not self.selected_datasets and hasattr(self, 'data_sources_step'):
            self.selected_datasets = self.data_sources_step.get_selected_sources()
            print(f"Late retrieval of datasets: {self.selected_datasets}")
            
        if not self.selected_datasets:
            print("Warning: No datasets selected for Model Analysis step")
            return
            
        # First make sure the testing_widget is set up
        if self.testing_widget is None:
            # Try to initialize the testing widget again
            self.initialize_testing_widget()
        
        # Make sure the variable_selection_step has the testing_widget
        if hasattr(self, 'variable_selection_step'):
            var_step = self.variable_selection_step
            
            # Make sure the testing_widget is set in the variable_selection_step
            if hasattr(var_step, 'testing_widget'):
                var_step.testing_widget = self.testing_widget
            
            # Set up the selector with the testing_widget
            if hasattr(var_step, 'setup_selector') and self.testing_widget:
                print(f"Setting up testing_widget in variable_selection_step: {self.testing_widget}")
                var_step.setup_selector(self.testing_widget)
                
                # Make sure the testing_widget has the data loading method
                if self.testing_widget and hasattr(self.testing_widget, 'load_dataset_from_study'):
                    print("Refreshing datasets in testing_widget")
                    # Use asyncio to refresh datasets
                    import asyncio
                    if asyncio.get_event_loop().is_running():
                        asyncio.create_task(self.testing_widget.load_dataset_from_study())
                    else:
                        asyncio.get_event_loop().run_until_complete(self.testing_widget.load_dataset_from_study())
            
            # IMPORTANT: Disable the start button to prevent duplicate analysis runs
            if hasattr(var_step, 'start_btn'):
                var_step.start_btn.setVisible(False)
        
        # Use set_data method to update the Model Analysis step
        if len(self.selected_datasets) > 0:
            # Pass the first selected dataset name (not the DataFrame) to the Model Analysis step
            dataset_name, dataset_df = self.selected_datasets[0]
            if hasattr(self, 'variable_selection_step'):
                self.variable_selection_step.set_data(dataset_name, self.current_hypothesis_text)
                print(f"Set Model Analysis with dataset {dataset_name}")
                
                # Force a refresh of datasets in the Model Analysis widget
                if hasattr(self.variable_selection_step, 'refresh_datasets'):
                    print("Calling refresh_datasets on variable_selection_step")
                    self.variable_selection_step.refresh_datasets()
        
        # Reset any existing hypothesis configuration when setting up variables
        self.target_variable = None
        self.predictor_variables = []
    
    def on_sources_selected(self, selected_sources):
        """Handle selection of data sources
        
        Args:
            selected_sources (list): List of (name, dataframe) tuples for selected sources
        """
        logger.debug(f"HypothesisGeneratorWidget received sources selected: {[name for name, _ in selected_sources]}")
        
        # Store the selected datasets
        self.selected_datasets = selected_sources
        
        # Update the data_sources_step's continue button if it exists
        has_selection = len(selected_sources) > 0
        logger.debug(f"Selection status: {has_selection} ({len(selected_sources)} datasets)")
        
        if hasattr(self, 'data_sources_step') and hasattr(self.data_sources_step, 'continue_btn'):
            self.data_sources_step.continue_btn.setEnabled(has_selection)
    
    def on_test_completed(self, results: Dict):
        """Handle completion of the statistical test"""
        # Store the results
        self.current_test_results = results
        print(f"Test completed with results: {results}")
        
        # Update our hypothesis text from the Model Analysis widget
        if hasattr(self, 'variable_selection_step') and self.variable_selection_step.hypothesis_text:
            self.current_hypothesis_text = self.variable_selection_step.hypothesis_text
        
        # Emit signal with hypothesis text and results
        self.hypothesis_tested.emit(self.current_hypothesis_text, results)
        
        # If this is a dictionary with a 'p_value', display a brief message
        if isinstance(results, dict) and 'p_value' in results:
            p_value = results['p_value']
            if p_value < 0.05:
                QMessageBox.information(self, "Test Result", 
                                       f"The test was statistically significant (p={p_value:.4f}).\n\n"
                                       f"The data supports your hypothesis:\n{self.current_hypothesis_text}")
            else:
                QMessageBox.information(self, "Test Result", 
                                       f"The test was not statistically significant (p={p_value:.4f}).\n\n"
                                       f"The data does not support your hypothesis:\n{self.current_hypothesis_text}")

    def refresh_datasets(self):
        """
        Refresh available datasets from the studies manager
        and update the testing_widget's dataset dropdown
        """
        print("Refreshing datasets from studies manager")
        if not self.studies_manager:
            print("No studies manager available")
            return []
            
        # Get datasets from the active study
        datasets = self.studies_manager.get_datasets_from_active_study()
        if not datasets:
            print("No datasets found in active study")
            return []
            
        print(f"Found {len(datasets)} datasets in active study")
        
        # Update the testing widget's dataset dropdown if available
        if hasattr(self, 'variable_selection_step') and hasattr(self.variable_selection_step, 'testing_widget'):
            testing_widget = self.variable_selection_step.testing_widget
            
            # The DataTestingWidget has a dataset_selector, not dataset_combo
            if hasattr(testing_widget, 'dataset_selector'):
                # Clear and repopulate the dataset selector
                dataset_selector = testing_widget.dataset_selector
                dataset_selector.clear()
                for name, df in datasets:
                    # Store the dataframe as user data
                    dataset_selector.addItem(name, df)
                print(f"Updated testing widget dropdown with {dataset_selector.count()} datasets")
                
        # Update our own data sources step
        if hasattr(self, 'data_sources_step'):
            data_sources = {name: df for name, df in datasets}
            self.data_sources_step.set_data_sources(data_sources)
            print(f"Updated data sources step with {len(data_sources)} datasets")
            
        return datasets
        
    async def run_analysis_workflow(self, dataset_name=None, automated_mode=False):
        """
        Executes a complete analysis workflow with step indicators following select.py flow:
        1. Set studies_manager
        2. Load dataset from study 
        3. Select dataset
        4. Auto-select test (maps variables)
        5. Build model
        6. Run statistical test
        7. Generate hypothesis for test (if enabled)
        
        Args:
            dataset_name: Name of dataset to use
            automated_mode: When True, skip UI updates and dialog creation
        
        Returns:
            Dict containing test results and status
        """
        results = {
            "success": False,
            "steps_completed": [],
            "error": None,
            "test_result": None,
            "hypothesis": None
        }
        
        try:
            # Show status message for user feedback (skip in automated mode)
            if not automated_mode:
                self.show_status_message("Starting analysis workflow...")
            
            # Step 0: Verify we have the testing widget properly set up
            if not hasattr(self, 'variable_selection_step') and not automated_mode:
                print("Creating step widgets since variable_selection_step is missing")
                self.create_step_widgets()
                
            if not self.testing_widget:
                print("Testing widget not yet initialized - trying to get it from main window")
                main_window = self.window()
                if hasattr(main_window, 'data_testing_widget') and main_window.data_testing_widget:
                    self.testing_widget = main_window.data_testing_widget
                else:
                    print("❌ Testing widget not available")
                    results["error"] = "Testing widget not available"
                    return results
            
            # Make sure variable_selection_step has the testing_widget set up correctly (skip in automated mode)
            if hasattr(self, 'variable_selection_step') and not automated_mode:
                # Set testing_widget directly on the variable_selection_step
                self.variable_selection_step.testing_widget = self.testing_widget
                # Call setup_selector to create the selector
                self.variable_selection_step.setup_selector(self.testing_widget)
            
            testing_widget = self.testing_widget
            
            # Step 1: Set the studies manager
            print("Step 1: Setting up studies manager")
            print(f"  - Has studies_manager: {hasattr(self, 'studies_manager')}")
            print(f"  - Testing widget has studies_manager: {hasattr(testing_widget, 'studies_manager')}")
            
            if self.studies_manager and not hasattr(testing_widget, 'studies_manager'):
                testing_widget.set_studies_manager(self.studies_manager)
                print("✓ Studies manager connected")
            else:
                print("⚠️ Studies manager already set or not available")
            
            # Step 2: Load available datasets
            print("Step 2: Loading available datasets")
            print(f"  - Testing widget has load_dataset_from_study: {hasattr(testing_widget, 'load_dataset_from_study')}")
            
            testing_widget.load_dataset_from_study()
            await asyncio.sleep(0.5)  # Give UI time to update
            
            if hasattr(testing_widget, 'dataset_selector'):
                print(f"  - Dataset selector count: {testing_widget.dataset_selector.count()}")
                for i in range(testing_widget.dataset_selector.count()):
                    print(f"    - Dataset {i}: {testing_widget.dataset_selector.itemText(i)}")
                
            results["steps_completed"].append("datasets_loaded")
            print(f"✓ Loaded datasets")
            
            if hasattr(testing_widget, 'dataset_selector') and testing_widget.dataset_selector.count() == 0:
                print("❌ No datasets available")
                results["error"] = "No datasets available"
                return results
            
            # Step 3: Select the specific dataset if provided
            print(f"Step 3: Selecting dataset {dataset_name if dataset_name else '(first available)'}")
            dataset_selected = False
            
            if dataset_name and hasattr(testing_widget, 'dataset_selector'):
                # Find and select the dataset in the selector dropdown
                for i in range(testing_widget.dataset_selector.count()):
                    print(f"  - Checking dataset {i}: {testing_widget.dataset_selector.itemText(i)}")
                    if testing_widget.dataset_selector.itemText(i) == dataset_name:
                        print(f"  - Found match at index {i}")
                        testing_widget.dataset_selector.setCurrentIndex(i)
                        dataset_selected = True
                        print(f"✓ Selected dataset '{dataset_name}'")
                        await asyncio.sleep(0.5)  # Give time for on_dataset_changed to process
                        break
                        
                if not dataset_selected:
                    print(f"⚠️ Dataset '{dataset_name}' not found, using first available")
                    if testing_widget.dataset_selector.count() > 0:
                        testing_widget.dataset_selector.setCurrentIndex(0)
                        dataset_selected = True
                        print(f"✓ Selected first dataset: '{testing_widget.dataset_selector.currentText()}'")
                        await asyncio.sleep(0.5)  # Give time for on_dataset_changed to process
            elif hasattr(testing_widget, 'dataset_selector'):
                # Use first dataset if available
                if testing_widget.dataset_selector.count() > 0:
                    testing_widget.dataset_selector.setCurrentIndex(0)
                    dataset_selected = True
                    print(f"✓ Selected first dataset: '{testing_widget.dataset_selector.currentText()}'")
                    await asyncio.sleep(0.5)  # Give time for on_dataset_changed to process
            
            if not dataset_selected:
                print("❌ Failed to select a dataset")
                results["error"] = "Failed to select a dataset"
                return results
            
            # Verify a dataframe was loaded after selection
            print(f"  - Has current_dataframe: {hasattr(testing_widget, 'current_dataframe')}")
            if hasattr(testing_widget, 'current_dataframe'):
                print(f"  - current_dataframe is None: {testing_widget.current_dataframe is None}")
            
            if not hasattr(testing_widget, 'current_dataframe') or testing_widget.current_dataframe is None:
                print("❌ Failed to load dataframe")
                results["error"] = "Failed to load dataframe"
                return results
                
            results["steps_completed"].append("dataset_selected")
            

            await asyncio.sleep(0.5)  # Give time for variable mapping
            
            
            # Step 4: Build model
            print("Step 4: Building variable mapping model")
            print(f"  - Has build_model: {hasattr(testing_widget, 'build_model')}")
            
            # Try to build the model, handling possible signature mismatches
            try:
                await testing_widget.build_model()
            except TypeError as e:
                print(f"Error calling build_model: {str(e)}")
                print("Trying alternative approach...")
                
            
            await asyncio.sleep(0.5)  # Give time for model building
            
            results["steps_completed"].append("model_built")
            print("✓ Model built")

            # Step 5: Auto-select test
            print("Step 5: Auto-selecting test")
            testing_widget.auto_select_test()
            await asyncio.sleep(0.5)  # Give time for test selection
            results["steps_completed"].append("test_selected")
            print("✓ Test selected")

            # Step 6: Run statistical test
            print("Step 6: Running statistical test")
            print(f"  - Has run_statistical_test: {hasattr(testing_widget, 'run_statistical_test')}")
            
            # Try to run the statistical test, handling possible signature mismatches
            try:
                # Try without the quiet parameter first
                test_result = await testing_widget.run_statistical_test()
            except TypeError as e:
                print(f"Error calling run_statistical_test: {str(e)}")
                print("Trying alternative approach...")
                
                # If the method exists but has a signature mismatch, try to get results indirectly
                if hasattr(testing_widget, 'on_run_test'):
                    # Call the non-async button click handler if available
                    testing_widget.on_run_test()
                    # Wait a moment for it to complete
                    await asyncio.sleep(2.0)
                    # Try to get the result after running the test
                    if hasattr(testing_widget, 'last_test_results'):
                        test_result = testing_widget.last_test_results
                        print("Got test_result from last_test_results")
                    else:
                        test_result = None
                        print("No test results available after calling on_run_test")
                else:
                    test_result = None
                    print("No alternative method found to run the test")
            
            print(f"  - Test result type: {type(test_result)}")
            print(f"  - Test result: {test_result}")
            
            if test_result:
                results["test_result"] = test_result
                results["steps_completed"].append("test_run")
                
                p_value = test_result.get('p_value')
                if p_value is not None:
                    print(f"✓ Test result: p-value = {p_value:.4f}")
                    
                    # Check if the test was significant
                    alpha = 0.05  # Default alpha level
                    if p_value < alpha:
                        print("✓ Hypothesis SUPPORTED (p < 0.05)")
                    else:
                        print("✗ Hypothesis NOT supported (p >= 0.05)")
                else:
                    print("✓ Test completed")
            else:
                print("❌ Test failed to produce results")
                results["error"] = "Test failed to produce results"
                return results
            
            # Step 7: Generate hypothesis for the test
            print("Step 7: Generating hypothesis for test")
            if hasattr(testing_widget, 'generate_hypothesis_for_test'):
                print("Testing widget has generate_hypothesis_for_test method")
                
                # Get variables required for hypothesis generation
                outcome = None
                group = None
                subject_id = None
                time = None
                test_name = None
                study_type = None
                
                if hasattr(testing_widget, 'outcome_combo') and testing_widget.outcome_combo.isEnabled():
                    outcome = testing_widget.outcome_combo.currentText()
                    
                if hasattr(testing_widget, 'group_combo') and testing_widget.group_combo.isEnabled():
                    group = testing_widget.group_combo.currentText()
                    
                if hasattr(testing_widget, 'subject_id_combo') and testing_widget.subject_id_combo.isEnabled():
                    subject_id = testing_widget.subject_id_combo.currentText()
                    
                if hasattr(testing_widget, 'time_combo') and testing_widget.time_combo.isEnabled():
                    time = testing_widget.time_combo.currentText()
                    
                if hasattr(testing_widget, 'test_combo') and testing_widget.test_combo.isEnabled():
                    test_name = testing_widget.test_combo.currentText()
                
                if hasattr(testing_widget, 'design_type_combo') and testing_widget.design_type_combo.isEnabled():
                    study_type = testing_widget.design_type_combo.currentData()
                
                if outcome and test_name:
                    print(f"Generating hypothesis for outcome: {outcome}, test: {test_name}")
                    try:
                        if not automated_mode:
                            self.show_status_message("Generating hypothesis from test results...")
                            
                        # Call generate_hypothesis_for_test
                        hypothesis_data = await testing_widget.generate_hypothesis_for_test(
                            outcome=outcome,
                            group=group,
                            subject_id=subject_id,
                            time=time,
                            test_name=test_name,
                            study_type=study_type
                        )
                        
                        if hypothesis_data:
                            results["hypothesis"] = hypothesis_data
                            results["steps_completed"].append("hypothesis_generated")
                            print(f"✓ Hypothesis generated: {hypothesis_data.get('title')}")
                            
                            # Update the hypothsis text field if we have one
                            if hasattr(self, 'hypothesis_text_field'):
                                hypothesis_text = hypothesis_data.get('title', '')
                                hypothesis_text += f"\n\nNull hypothesis: {hypothesis_data.get('null_hypothesis', '')}"
                                hypothesis_text += f"\nAlternative hypothesis: {hypothesis_data.get('alternative_hypothesis', '')}"
                                hypothesis_text += f"\nStatus: {hypothesis_data.get('status', 'Unknown')}"
                                self.hypothesis_text_field.setText(hypothesis_text)
                                self.current_hypothesis_text = hypothesis_text
                        else:
                            print("⚠️ Failed to generate hypothesis")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"❌ Error generating hypothesis: {str(e)}")
                else:
                    print("⚠️ Missing required variables for hypothesis generation")
            else:
                print("⚠️ Testing widget does not have generate_hypothesis_for_test method")

            # Final status update
            results["success"] = True
            if not automated_mode:
                self.show_status_message("Analysis workflow completed successfully")
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            print(f"❌ Analysis workflow failed: {error_msg}")
            results["error"] = error_msg
            if not automated_mode:
                self.show_status_message(f"Analysis error: {error_msg[:50]}...")
            return results

    async def _run_generate_hypothesis(self, outcome, group, subject_id, time, test_name):
        """Run hypothesis generation async"""
        try:
            if not self.testing_widget:
                QMessageBox.warning(self, "Error", "Testing widget not initialized")
                return
            
            # Check if studies manager is available to save the hypothesis
            main_window = self.window()
            if not self.studies_manager and hasattr(main_window, 'studies_manager'):
                self.studies_manager = main_window.studies_manager
                
            # Safely show status message
            try:
                self.show_status_message("Generating hypothesis using LLM...")
            except Exception as e:
                print(f"Non-critical error showing status: {str(e)}")
                
            self.save_button.setEnabled(False)
            # Get data needed for the prompt
            prompt = f"""
            Given the following data analysis setup, please generate a formal, testable hypothesis.
            
            Outcome Variable: {outcome}
            Group/Treatment Variable: {group or 'Not specified'}
            Subject ID Variable: {subject_id or 'Not specified'}
            Time Variable: {time or 'Not specified'}
            Statistical Test: {test_name or 'Not specified'}
            
            Please generate a clear, concise hypothesis that would be appropriate to test with this statistical setup.
            Phrase it as a formal scientific hypothesis statement that makes a specific, testable prediction.
            For example: "Patients treated with Drug A will show a significantly greater reduction in blood pressure compared to patients treated with placebo."
            
            Return only the hypothesis statement with no additional explanation.
            """
            
            # Call LLM
            from llms.client import call_llm_async
            try:
                self.show_status_message("Waiting for hypothesis from LLM...")
            except Exception:
                pass
            
            hypothesis_text = await call_llm_async(prompt)
            hypothesis_text = hypothesis_text.strip()
            
            self.current_hypothesis_text = hypothesis_text
            
            # Update text field
            self.hypothesis_text_field.setText(hypothesis_text)
            
            # Now let's test this hypothesis directly with Model Analysis workflow
            # Create a selector object
            try:
                self.show_status_message("Testing hypothesis with selected variables...")
            except Exception:
                pass
            
            selector = HypothesisVariableSelector(self.testing_widget)
            selector.set_dataset(self.dataset_selection_combo.currentText())
            selector.set_hypothesis(hypothesis_text)
            
            # Now we pass the hypothesis directly to testing_widget's build_model when we call the workflow
            try:
                print("Running Model Analysis workflow...")
                await selector.run_workflow()
                print("Model Analysis workflow completed successfully")
            except Exception as e:
                print(f"Error in Model Analysis workflow: {str(e)}")
                # We'll continue instead of raising the error to avoid breaking the whole process
            
            # Enable the save button
            self.save_button.setEnabled(True)
            try:
                self.show_status_message("Hypothesis generation and testing complete")
            except Exception:
                pass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error generating hypothesis: {str(e)}")
            self.save_button.setEnabled(True)
            try:
                self.clear_status_message()
            except Exception:
                pass

    def generate_hypothesis(self):
        """Generate a hypothesis using the LLM"""
        if not self.testing_widget:
            QMessageBox.warning(self, "Error", "Testing widget not initialized")
            return
        
        if self.dataset_selection_combo.count() == 0:
            QMessageBox.warning(self, "Error", "No datasets available")
            return
            
        # Get current variables from the testing widget
        outcome = None
        group = None
        subject_id = None
        time = None
        test_name = None
        
        if hasattr(self.testing_widget, 'outcome_combo') and self.testing_widget.outcome_combo.isEnabled():
            outcome = self.testing_widget.outcome_combo.currentText()
            
        if hasattr(self.testing_widget, 'group_combo') and self.testing_widget.group_combo.isEnabled():
            group = self.testing_widget.group_combo.currentText()
            
        if hasattr(self.testing_widget, 'subject_id_combo') and self.testing_widget.subject_id_combo.isEnabled():
            subject_id = self.testing_widget.subject_id_combo.currentText()
            
        if hasattr(self.testing_widget, 'time_combo') and self.testing_widget.time_combo.isEnabled():
            time = self.testing_widget.time_combo.currentText()
            
        if hasattr(self.testing_widget, 'test_combo') and self.testing_widget.test_combo.isEnabled():
            test_name = self.testing_widget.test_combo.currentText()
        
        if not outcome:
            QMessageBox.warning(self, "Error", "No outcome variable selected")
            return
            
        if not test_name:
            QMessageBox.warning(self, "Error", "No statistical test selected")
            return
        
        # Show status message
        self.show_status_message("Generating and testing hypothesis...")
            
        # Call the async method
        asyncio.create_task(self._run_generate_hypothesis(outcome, group, subject_id, time, test_name))

    async def generate_and_test_hypothesis_for_variable(self, variable_name, dataset_name=None, test_name=None):
        """Generate and test a hypothesis for a specific variable.
        
        This method provides an easy way to test a specific variable and generate a hypothesis
        for it from external components. It uses the DataTestingWidget's functionality.
        
        Args:
            variable_name (str): The name of the variable to analyze (will be set as outcome)
            dataset_name (str, optional): The name of the dataset to use. If None, uses current dataset.
            test_name (str, optional): The name of the test to run. If None, uses auto-selected test.
            
        Returns:
            dict: A dictionary with results including hypothesis and test results
        """
        results = {
            "success": False,
            "variable": variable_name,
            "dataset": dataset_name,
            "test": None,
            "hypothesis": None,
            "test_result": None,
            "error": None
        }
        
        try:
            print(f"Generating and testing hypothesis for variable: {variable_name}")
            
            # Check if we have the testing widget
            if not self.testing_widget:
                print("Testing widget not initialized - trying to get it from main window")
                main_window = self.window()
                if hasattr(main_window, 'data_testing_widget') and main_window.data_testing_widget:
                    self.testing_widget = main_window.data_testing_widget
                else:
                    error_msg = "Testing widget not available"
                    print(f"❌ {error_msg}")
                    results["error"] = error_msg
                    return results
            
            testing_widget = self.testing_widget
            
            # Step 1: Set the dataset
            if dataset_name and hasattr(testing_widget, 'dataset_selector'):
                # Find and select the dataset in the selector dropdown
                dataset_found = False
                for i in range(testing_widget.dataset_selector.count()):
                    if testing_widget.dataset_selector.itemText(i) == dataset_name:
                        testing_widget.dataset_selector.setCurrentIndex(i)
                        dataset_found = True
                        print(f"✓ Selected dataset '{dataset_name}'")
                        await asyncio.sleep(0.5)  # Give time for on_dataset_changed to process
                        break
                
                if not dataset_found:
                    error_msg = f"Dataset '{dataset_name}' not found"
                    print(f"❌ {error_msg}")
                    results["error"] = error_msg
                    return results
            
            # Verify a dataframe was loaded after selection
            if not hasattr(testing_widget, 'current_dataframe') or testing_widget.current_dataframe is None:
                error_msg = "Failed to load dataframe"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                return results
                
            # Update the current dataset name in results
            if hasattr(testing_widget, 'current_name'):
                results["dataset"] = testing_widget.current_name
                
            # Step 2: Set the outcome variable to the requested variable
            if hasattr(testing_widget, 'outcome_combo'):
                variable_found = False
                for i in range(testing_widget.outcome_combo.count()):
                    if testing_widget.outcome_combo.itemText(i) == variable_name:
                        testing_widget.outcome_combo.setCurrentIndex(i)
                        variable_found = True
                        print(f"✓ Set outcome variable to '{variable_name}'")
                        break
                
                if not variable_found:
                    error_msg = f"Variable '{variable_name}' not found in dataset"
                    print(f"❌ {error_msg}")
                    results["error"] = error_msg
                    return results
            else:
                error_msg = "Cannot set outcome variable - widget doesn't have outcome_combo"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                return results
                
            # Step 3: Build the model and wait for it to complete
            print("Building variable mapping model...")
            try:
                await testing_widget.build_model()
                await asyncio.sleep(0.5)  # Give time for model building
                print("✓ Model built")
            except Exception as e:
                error_msg = f"Error building model: {str(e)}"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                return results
            
            # Step 4: Set or auto-select the test
            if test_name and hasattr(testing_widget, 'test_combo'):
                # Set specific test if requested
                test_found = False
                for i in range(testing_widget.test_combo.count()):
                    if testing_widget.test_combo.itemText(i) == test_name:
                        testing_widget.test_combo.setCurrentIndex(i)
                        test_found = True
                        print(f"✓ Set test to '{test_name}'")
                        break
                
                if not test_found:
                    print(f"⚠️ Test '{test_name}' not found, using auto-select instead")
                    testing_widget.auto_select_test()
                    await asyncio.sleep(0.5)  # Give time for test selection
            else:
                # Auto-select test if no specific test requested
                print("Auto-selecting test...")
                testing_widget.auto_select_test()
                await asyncio.sleep(0.5)  # Give time for test selection
            
            # Get the selected test
            if hasattr(testing_widget, 'test_combo'):
                selected_test = testing_widget.test_combo.currentText()
                results["test"] = selected_test
                print(f"✓ Selected test: {selected_test}")
            
            # Step 5: Run the statistical test
            print("Running statistical test...")
            try:
                test_result = await testing_widget.run_statistical_test()
                await asyncio.sleep(0.5)  # Give time for test to complete
                
                if test_result:
                    results["test_result"] = test_result
                    p_value = test_result.get('p_value')
                    if p_value is not None:
                        print(f"✓ Test result: p-value = {p_value:.4f}")
                    else:
                        print("✓ Test completed")
                else:
                    error_msg = "Test failed to produce results"
                    print(f"❌ {error_msg}")
                    results["error"] = error_msg
                    return results
            except Exception as e:
                error_msg = f"Error running test: {str(e)}"
                print(f"❌ {error_msg}")
                results["error"] = error_msg
                return results
            
            # Step 6: Generate hypothesis for the test
            print("Generating hypothesis for test...")
            if hasattr(testing_widget, 'generate_hypothesis_for_test'):
                # Get variables required for hypothesis generation
                outcome = variable_name
                group = None
                subject_id = None
                time = None
                study_type = None
                
                if hasattr(testing_widget, 'group_combo') and testing_widget.group_combo.isEnabled():
                    group = testing_widget.group_combo.currentText()
                    
                if hasattr(testing_widget, 'subject_id_combo') and testing_widget.subject_id_combo.isEnabled():
                    subject_id = testing_widget.subject_id_combo.currentText()
                    
                if hasattr(testing_widget, 'time_combo') and testing_widget.time_combo.isEnabled():
                    time = testing_widget.time_combo.currentText()
                
                if hasattr(testing_widget, 'design_type_combo') and testing_widget.design_type_combo.isEnabled():
                    study_type = testing_widget.design_type_combo.currentData()
                
                try:
                    self.show_status_message("Generating hypothesis from test results...")
                    
                    # Call generate_hypothesis_for_test
                    hypothesis_data = await testing_widget.generate_hypothesis_for_test(
                        outcome=outcome,
                        group=group,
                        subject_id=subject_id,
                        time=time,
                        test_name=selected_test,
                        study_type=study_type
                    )
                    
                    if hypothesis_data:
                        results["hypothesis"] = hypothesis_data
                        print(f"✓ Hypothesis generated: {hypothesis_data.get('title')}")
                        
                        # Update the hypothesis text field if we have one
                        if hasattr(self, 'hypothesis_text_field'):
                            hypothesis_text = hypothesis_data.get('title', '')
                            hypothesis_text += f"\n\nNull hypothesis: {hypothesis_data.get('null_hypothesis', '')}"
                            hypothesis_text += f"\nAlternative hypothesis: {hypothesis_data.get('alternative_hypothesis', '')}"
                            hypothesis_text += f"\nStatus: {hypothesis_data.get('status', 'Unknown')}"
                            self.hypothesis_text_field.setText(hypothesis_text)
                            self.current_hypothesis_text = hypothesis_text
                    else:
                        print("⚠️ Failed to generate hypothesis")
                except Exception as e:
                    error_msg = f"Error generating hypothesis: {str(e)}"
                    print(f"❌ {error_msg}")
                    results["error"] = error_msg
            else:
                error_msg = "Testing widget does not have generate_hypothesis_for_test method"
                print(f"⚠️ {error_msg}")
                results["error"] = error_msg
            
            # Final status update
            results["success"] = True
            print("✓ Variable analysis workflow completed successfully")
            self.show_status_message(f"Completed hypothesis for {variable_name}")
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            print(f"❌ Variable analysis failed: {error_msg}")
            results["error"] = error_msg
            return results
