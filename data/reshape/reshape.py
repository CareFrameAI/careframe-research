import os
import json
import pandas as pd
import numpy as np 
from datetime import datetime, date, timedelta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPlainTextEdit, QFormLayout, 
    QSpinBox, QMessageBox, QGroupBox, QStackedWidget, QRadioButton, QButtonGroup,
    QListWidget, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QStatusBar
)
from PyQt6.QtGui import QIcon
import re
import asyncio
from llms.client import call_llm_async
from qasync import asyncSlot
from helpers.load_icon import load_bootstrap_icon

# ---------------------------------------------------------------------
# Display widget for pandas DataFrames in a table
# ---------------------------------------------------------------------
class DataFrameDisplay(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
    def display_dataframe(self, df: pd.DataFrame):
        self.clear()
        if df is None or df.empty:
            return
        self.setRowCount(min(1000, len(df)))  # Limit to 1000 rows for performance
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels(df.columns)
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 1000:
                break
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.setItem(i, j, item)
        self.resizeColumnsToContents()


# ---------------------------------------------------------------------
# Main Reshape Widget (all transformation UIs integrated)
# ---------------------------------------------------------------------
class DataReshapeWidget(QWidget):
    # Signal emitted when a source (or transformed dataset) is selected
    source_selected = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Reshape")
        
        # Internal storage for data sources and datasets
        self.dataframes = {}
        self.current_name = ""
        self.current_dataframe = None
        self.format_cache = {}  # Cache detected formats
        self._preview_df = None

        # Build the UI
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Main splitter dividing top (options) and bottom (dataset view)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # --------------------------
        # Top section: Timepoint Options
        # --------------------------
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create timepoint options widget
        self.timepoint_options_widget = self.create_timepoint_options_widget()
        top_layout.addWidget(self.timepoint_options_widget)
        
        main_splitter.addWidget(top_section)
        
        # --------------------------
        # Bottom section: Dataset view
        # --------------------------
        bottom_section = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: (Previously used for source list; now removed obsolete add buttons)
        sources_panel = QWidget()
        sources_layout = QVBoxLayout(sources_panel)
        sources_layout.setContentsMargins(5, 5, 5, 5)
        sources_group = QGroupBox("Available Data Sources")
        sources_inner_layout = QVBoxLayout(sources_group)
        
        # Add refresh button at the top of the sources list
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh from Studies")
        refresh_button.setIcon(load_bootstrap_icon("arrow-clockwise"))
        refresh_button.setToolTip("Refresh datasets from active study")
        refresh_button.clicked.connect(self.refresh_datasets_from_studies_manager)
        refresh_layout.addWidget(refresh_button)
        refresh_layout.addStretch()
        sources_inner_layout.addLayout(refresh_layout)
        
        # Add a label to indicate how to load data
        sources_hint = QLabel("Click 'Refresh from Studies' to load datasets from the active study")
        sources_hint.setWordWrap(True)
        sources_hint.setStyleSheet("color: #666;")
        sources_inner_layout.addWidget(sources_hint)
        
        # Only a list is kept so users can select an already loaded dataset.
        self.sources_list = QListWidget()
        self.sources_list.itemClicked.connect(self.on_source_selected)
        sources_inner_layout.addWidget(self.sources_list)
        
        sources_layout.addWidget(sources_group)
        bottom_section.addWidget(sources_panel)
        
        # Right panel: Dataset view and transformation preview
        dataset_panel = QWidget()
        dataset_layout = QVBoxLayout(dataset_panel)
        dataset_layout.setContentsMargins(5, 5, 5, 5)
        
        self.data_tabs = QTabWidget()
        # Tab for original dataset
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)
        header_layout = QHBoxLayout()
        self.current_dataset_label = QLabel("No dataset selected")
        self.current_dataset_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.current_dataset_label)
        self.dataset_info_label = QLabel("")
        header_layout.addWidget(self.dataset_info_label)
        header_layout.addStretch()
        original_layout.addLayout(header_layout)
        self.dataset_display = DataFrameDisplay()
        original_layout.addWidget(self.dataset_display)
        self.data_tabs.addTab(original_tab, "Original Data")
        
        # Tab for transformation preview and save
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Transformation Preview"))
        preview_header.addStretch()
        preview_header.addWidget(QLabel("Save As:"))
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for transformed dataset")
        preview_header.addWidget(self.save_name_input)
        save_button = QPushButton("Save")
        save_button.setIcon(load_bootstrap_icon("save"))
        save_button.clicked.connect(self.save_transformation)
        preview_header.addWidget(save_button)
        preview_layout.addLayout(preview_header)
        self.preview_display = DataFrameDisplay()
        preview_layout.addWidget(self.preview_display)
        self.data_tabs.addTab(preview_tab, "Transformation Preview")
        
        dataset_layout.addWidget(self.data_tabs)
        bottom_section.addWidget(sources_panel)
        bottom_section.addWidget(dataset_panel)
        
        # Adjust sizes to give more space to data sources
        bottom_section.setSizes([400, 600])  # Increased space for sources panel
        
        main_splitter.addWidget(bottom_section)
        
        # Explicitly set better initial sizes for the splitters
        main_splitter.setSizes([250, 750])  # Less space to top section to be more compact
        
        main_layout.addWidget(main_splitter)
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Force the widget to have a reasonable minimum size
        self.setMinimumSize(900, 700)
    
    # ---------------------------------------------------------------
    # Create the timepoint management options page
    # ---------------------------------------------------------------
    def create_timepoint_options_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Title with tooltip for description
        title = QLabel("Format: Long ⟷ Wide")
        title.setStyleSheet("font-size: 12pt; font-weight: bold;")
        description = (
            "Format Types:\n"
            "- Long Format: One row per subject per timepoint\n"
            "- Wide Format: One row per subject with separate columns for each timepoint"
        )
        title.setToolTip(description)
        layout.addWidget(title)
        
        # Direction and Column Selection in the same row
        top_section = QHBoxLayout()
        
        # Direction Selection
        direction_group = QGroupBox("Transformation Direction")
        direction_layout = QHBoxLayout(direction_group)
        direction_layout.setContentsMargins(8, 8, 8, 8)
        self.transform_group = QButtonGroup(widget)
        self.long_to_wide_radio = QRadioButton("Long → Wide")
        self.long_to_wide_radio.toggled.connect(self.update_timepoint_options)
        self.transform_group.addButton(self.long_to_wide_radio)
        direction_layout.addWidget(self.long_to_wide_radio)
        self.wide_to_long_radio = QRadioButton("Wide → Long")
        self.wide_to_long_radio.toggled.connect(self.update_timepoint_options)
        self.transform_group.addButton(self.wide_to_long_radio)
        direction_layout.addWidget(self.wide_to_long_radio)
        top_section.addWidget(direction_group, 1)
        
        # Column Selection
        columns_group = QGroupBox("Column Selection")
        columns_layout = QFormLayout(columns_group)
        columns_layout.setContentsMargins(8, 8, 8, 8)
        columns_layout.setVerticalSpacing(4)
        self.subject_id_combo = QComboBox()
        columns_layout.addRow("Subject ID:", self.subject_id_combo)
        self.timepoint_column_combo = QComboBox()
        self.timepoint_column_combo.currentTextChanged.connect(self.on_timepoint_column_changed)
        columns_layout.addRow("Timepoint:", self.timepoint_column_combo)
        self.value_column_combo = QComboBox()
        columns_layout.addRow("Value:", self.value_column_combo)
        top_section.addWidget(columns_group, 2)
        
        layout.addLayout(top_section)
        
        # Date handling options with compact layout
        self.timepoint_group = QGroupBox("Date Handling Options")
        timepoint_layout = QVBoxLayout(self.timepoint_group)
        timepoint_layout.setContentsMargins(8, 8, 8, 8)
        timepoint_layout.setSpacing(4)
        
        # Date handling method selection in a row at the top
        date_options_layout = QHBoxLayout()
        date_options_layout.addWidget(QLabel("Date Handling Method:"))
        self.date_handling_combo = QComboBox()
        self.date_handling_combo.addItems(["Use as is", "Group by intervals", "Binned timepoints", "Regular intervals with window"])
        self.date_handling_combo.currentIndexChanged.connect(self.on_date_handling_changed)
        date_options_layout.addWidget(self.date_handling_combo, 1)
        timepoint_layout.addLayout(date_options_layout)
        
        # Stacked widget for different date handling options
        self.date_handling_stack = QStackedWidget()
        self.date_handling_stack.setMaximumHeight(120) # Limit height to save space
        
        # Option 1: Use as is (no additional controls needed)
        as_is_widget = QWidget()
        as_is_layout = QVBoxLayout(as_is_widget)
        as_is_layout.addWidget(QLabel("Using timepoint values as-is without modification."))
        as_is_layout.addStretch()
        self.date_handling_stack.addWidget(as_is_widget)
        
        # Option 2: Group by intervals
        interval_widget = QWidget()
        interval_layout = QHBoxLayout(interval_widget)
        interval_layout.setContentsMargins(0, 5, 0, 5)
        
        interval_layout.addWidget(QLabel("Group every:"))
        
        self.interval_value_spin = QSpinBox()
        self.interval_value_spin.setRange(1, 365)
        self.interval_value_spin.setValue(30)
        interval_layout.addWidget(self.interval_value_spin)
        
        self.interval_unit_combo = QComboBox()
        self.interval_unit_combo.addItems(["Days", "Weeks", "Months", "Years"])
        interval_layout.addWidget(self.interval_unit_combo)
        
        interval_layout.addStretch(1)
        
        self.date_handling_stack.addWidget(interval_widget)
        
        # Option 3: Binned timepoints
        binned_widget = QWidget()
        binned_layout = QGridLayout(binned_widget)
        binned_layout.setContentsMargins(0, 5, 0, 5)
        
        binned_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.bin_start_edit = QLineEdit()
        self.bin_start_edit.setPlaceholderText("YYYY-MM-DD")
        binned_layout.addWidget(self.bin_start_edit, 0, 1)
        
        binned_layout.addWidget(QLabel("End Date:"), 0, 2)
        self.bin_end_edit = QLineEdit()
        self.bin_end_edit.setPlaceholderText("YYYY-MM-DD")
        binned_layout.addWidget(self.bin_end_edit, 0, 3)
        
        binned_layout.addWidget(QLabel("Number of Bins:"), 1, 0)
        self.bin_count_spin = QSpinBox()
        self.bin_count_spin.setRange(2, 100)
        self.bin_count_spin.setValue(4)
        binned_layout.addWidget(self.bin_count_spin, 1, 1)
        
        binned_layout.setColumnStretch(1, 1)
        binned_layout.setColumnStretch(3, 1)
        
        self.date_handling_stack.addWidget(binned_widget)
        
        # Option 4: Regular intervals with window
        window_widget = QWidget()
        window_layout = QGridLayout(window_widget)
        window_layout.setContentsMargins(0, 5, 0, 5)
        
        window_layout.addWidget(QLabel("Every:"), 0, 0)
        self.window_interval_spin = QSpinBox()
        self.window_interval_spin.setRange(1, 52)
        self.window_interval_spin.setValue(4)
        window_layout.addWidget(self.window_interval_spin, 0, 1)
        
        self.window_unit_combo = QComboBox()
        self.window_unit_combo.addItems(["Days", "Weeks", "Months"])
        self.window_unit_combo.setCurrentText("Weeks")
        window_layout.addWidget(self.window_unit_combo, 0, 2)
        
        window_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(1, 90)
        self.window_size_spin.setValue(14)
        window_layout.addWidget(self.window_size_spin, 1, 1)
        
        self.window_size_unit_combo = QComboBox()
        self.window_size_unit_combo.addItems(["Days", "Weeks"])
        self.window_size_unit_combo.setCurrentText("Days")
        window_layout.addWidget(self.window_size_unit_combo, 1, 2)
        
        # Add encounter/value column selection for ordering
        self.encounter_id_check = QGroupBox("Use additional column for ordering")
        self.encounter_id_check.setCheckable(True)
        self.encounter_id_check.setChecked(False)
        encounter_layout = QHBoxLayout(self.encounter_id_check)
        
        encounter_layout.addWidget(QLabel("Order Column:"))
        self.encounter_id_combo = QComboBox()
        encounter_layout.addWidget(self.encounter_id_combo, 1)
        
        window_layout.addWidget(self.encounter_id_check, 2, 0, 1, 3)
        
        window_layout.setColumnStretch(1, 1)
        window_layout.setColumnStretch(2, 1)
        
        self.date_handling_stack.addWidget(window_widget)
        
        timepoint_layout.addWidget(self.date_handling_stack)
        
        # Aggregation options
        agg_layout = QHBoxLayout()
        agg_layout.addWidget(QLabel("When multiple values exist, use:"))
        self.aggregation_combo = QComboBox()
        self.aggregation_combo.addItems(["First value", "Last value", "Mean", "Median", "Min", "Max", "Count", "Distinct count",
                                         "Closest to interval start", "Closest to interval middle", "Closest to interval end"])
        agg_layout.addWidget(self.aggregation_combo, 1)
        
        timepoint_layout.addLayout(agg_layout)
        layout.addWidget(self.timepoint_group)
        
        # Action buttons in a single row
        button_row = QHBoxLayout()
        
        # Auto-detect button
        auto_detect_button = QPushButton("Auto-Detect Columns (AI)")
        auto_detect_button.setIcon(load_bootstrap_icon("search"))
        auto_detect_button.clicked.connect(self.auto_detect_timepoints)
        button_row.addWidget(auto_detect_button)
        
        # AI-assisted button
        ai_wide_to_long_button = QPushButton("AI-Assisted Wide to Long")
        ai_wide_to_long_button.setIcon(load_bootstrap_icon("cpu"))
        ai_wide_to_long_button.clicked.connect(self.ai_wide_to_long_conversion)
        button_row.addWidget(ai_wide_to_long_button)
        
        # Apply button
        apply_button = QPushButton("Apply Transformation")
        apply_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        apply_button.clicked.connect(self.apply_timepoint_transformation)
        button_row.addWidget(apply_button)
        
        layout.addLayout(button_row)
        layout.addStretch()
        return widget
    
    def on_date_handling_changed(self, index):
        """Update UI based on date handling selection"""
        if hasattr(self, 'date_handling_stack'):
            self.date_handling_stack.setCurrentIndex(index)
    
    @asyncSlot()
    async def auto_detect_timepoints(self):
        """Use Gemini LLM to identify timepoint and subject ID columns"""
        self.status_bar.showMessage("Detecting timepoint columns...")
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
            
        # Create a prompt for Gemini API
        prompt = f"""
        I need to identify timepoint-related columns in this dataset.
        
        Dataset Columns: {list(self.current_dataframe.columns)}
        Dataset Sample (first 3 rows):
        {self.current_dataframe.head(3).to_string()}
        
        Please identify:
        1. The column that likely represents subject/patient IDs
        2. The column that likely represents timepoints/visits
        3. A column that contains values of interest (numeric data)
        
        Return your response as a JSON object with the following structure:
        {{
            "subject_id_column": "column_name",
            "timepoint_column": "column_name",
            "value_column": "column_name",
            "format_type": "longitudinal|columnar",
            "explanation": "brief explanation of your recommendation"
        }}
        """
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset with AI. This may take a moment...")
        
        try:
            # Call Gemini API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                # Set the detected values in the UI
                subject_id = result.get("subject_id_column")
                timepoint = result.get("timepoint_column")
                value = result.get("value_column")
                format_type = result.get("format_type")
                explanation = result.get("explanation")
                
                # Update UI with detected values
                if subject_id in self.current_dataframe.columns:
                    self.subject_id_combo.setCurrentText(subject_id)
                if timepoint in self.current_dataframe.columns:
                    self.timepoint_column_combo.setCurrentText(timepoint)
                if value in self.current_dataframe.columns:
                    self.value_column_combo.setCurrentText(value)
                    
                # Update format detection
                if format_type:
                    self.format_cache[self.current_name] = format_type
                    
                    # Select appropriate radio button
                    if format_type == "longitudinal":
                        self.long_to_wide_radio.setChecked(True)
                    elif format_type == "columnar":
                        self.wide_to_long_radio.setChecked(True)
                    
                # Show explanation
                QMessageBox.information(self, "AI Recommendation", 
                                       f"Detected Columns:\n"
                                       f"Subject ID: {subject_id}\n"
                                       f"Timepoint: {timepoint}\n"
                                       f"Value: {value}\n"
                                       f"Format: {format_type}\n\n"
                                       f"Explanation: {explanation}")
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods to add and select sources (datasets)
    # ---------------------------------------------------------------
    def add_source(self, name, connection, dataframe):
        # Store the dataframe under a given name.
        self.dataframes[name] = dataframe
        # Add to list if not already present
        items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if not items:
            self.sources_list.addItem(name)
    
    def on_source_selected(self, item):
        name = item.text()
        if name in self.dataframes:
            dataframe = self.dataframes[name]
            self.current_name = name
            self.current_dataframe = dataframe
            self.display_dataset(name, dataframe)
            self.update_column_selections()
            self.data_tabs.setCurrentIndex(0)
    
    def display_dataset(self, name, dataframe):
        self.current_dataset_label.setText(f"Dataset: {name}")
        rows, cols = dataframe.shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")
        self.dataset_display.display_dataframe(dataframe)
        self.preview_display.clear()
        self._preview_df = None
        self.save_name_input.setText(f"{name}_transformed")
    
    def update_column_selections(self):
        if self.current_dataframe is None:
            return
        self.subject_id_combo.clear()
        self.timepoint_column_combo.clear()
        self.value_column_combo.clear()
        for column in self.current_dataframe.columns:
            self.subject_id_combo.addItem(column)
            self.timepoint_column_combo.addItem(column)
            self.value_column_combo.addItem(column)
        self.auto_select_defaults()
    
    def auto_select_defaults(self):
        if self.current_dataframe is None:
            return
        subject_id_cols = [col for col in self.current_dataframe.columns 
                           if 'subject' in col.lower() or 'patient' in col.lower() or col.lower().endswith('_id')]
        if subject_id_cols:
            self.subject_id_combo.setCurrentText(subject_id_cols[0])
        visit_cols = [col for col in self.current_dataframe.columns 
                      if col.lower() in ['visit', 'timepoint', 'visit_id', 'visit_date']]
        if visit_cols:
            self.timepoint_column_combo.setCurrentText(visit_cols[0])
        value_cols = [col for col in self.current_dataframe.columns 
                      if pd.api.types.is_numeric_dtype(self.current_dataframe[col]) 
                      and not col.lower().endswith('_id')]
        if value_cols:
            self.value_column_combo.setCurrentText(value_cols[0])
    
    def update_timepoint_options(self):
        if self.long_to_wide_radio.isChecked():
            self.timepoint_column_combo.setEnabled(True)
            self.value_column_combo.setEnabled(True)
            
            # Update the encounter_id_combo if it exists
            if hasattr(self, 'encounter_id_combo') and self.current_dataframe is not None:
                self.encounter_id_combo.clear()
                for column in self.current_dataframe.columns:
                    self.encounter_id_combo.addItem(column)
                
                # Try to select an appropriate default
                encounter_cols = [col for col in self.current_dataframe.columns 
                                if 'encounter' in col.lower() or 'event' in col.lower()]
                if encounter_cols:
                    self.encounter_id_combo.setCurrentText(encounter_cols[0])
        else:
            self.timepoint_column_combo.setEnabled(False)
            self.value_column_combo.setEnabled(False)
    
    def on_timepoint_column_changed(self, column_name):
        if not column_name or self.current_dataframe is None:
            return
        try:
            # Try to determine if this is a date column
            is_date = False
            col_data = self.current_dataframe[column_name]
            
            # Check if it's already a datetime type
            if pd.api.types.is_datetime64_any_dtype(col_data):
                is_date = True
            else:
                # Try to convert to datetime if it's a string that looks like a date
                try:
                    pd.to_datetime(col_data, errors='raise')
                    is_date = True
                except:
                    # If conversion fails, we'll still enable date handling
                    # since the user might want to use it anyway
                    is_date = True
            
            # Always enable date handling options - let the user decide if they're applicable
            self.timepoint_group.setEnabled(True)
            
            # If we have a date column, try to set appropriate defaults for binning
            if is_date:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(col_data):
                    try:
                        date_col = pd.to_datetime(col_data, errors='coerce')
                        # Set default bin start/end dates if valid dates exist
                        valid_dates = date_col.dropna()
                        if not valid_dates.empty:
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()
                            
                            # Format as strings for the UI
                            if hasattr(self, 'bin_start_edit') and isinstance(min_date, pd.Timestamp):
                                self.bin_start_edit.setText(min_date.strftime('%Y-%m-%d'))
                            if hasattr(self, 'bin_end_edit') and isinstance(max_date, pd.Timestamp):
                                self.bin_end_edit.setText(max_date.strftime('%Y-%m-%d'))
                    except:
                        pass  # If conversion fails, just skip setting the defaults
        except Exception as e:
            # In case of error, still enable date handling
            self.timepoint_group.setEnabled(True)
            print(f"Error checking timepoint column: {str(e)}")
    
    # ---------------------------------------------------------------
    # Timepoint transformation methods (long to wide and wide to long)
    # ---------------------------------------------------------------
    def apply_timepoint_transformation(self):
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        if self.long_to_wide_radio.isChecked():
            self.convert_long_to_wide()
        elif self.wide_to_long_radio.isChecked():
            self.convert_wide_to_long()
        else:
            QMessageBox.warning(self, "Error", "Please select a transformation direction")
    
    def convert_long_to_wide(self):
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        subject_id_col = self.subject_id_combo.currentText()
        timepoint_col = self.timepoint_column_combo.currentText()
        value_col = self.value_column_combo.currentText()
        if not all([subject_id_col, timepoint_col, value_col]):
            QMessageBox.warning(self, "Error", "Please select all required columns")
            return
        try:
            self.status_bar.showMessage("Converting from long to wide format...")
            df = self.current_dataframe.copy()
            
            # Always enable date handling, don't disable based on regex patterns
            is_date_column = False
            is_numeric_timepoint = False
            
            # Check if timepoint column is a date or numeric type - but don't restrict functionality
            try:
                if pd.api.types.is_numeric_dtype(df[timepoint_col]):
                    is_numeric_timepoint = True
                elif pd.api.types.is_datetime64_any_dtype(df[timepoint_col]):
                    is_date_column = True
                else:
                    try:
                        pd.to_datetime(df[timepoint_col], errors='raise')
                        is_date_column = True
                    except:
                        pass  # We'll leave is_date_column as False but won't restrict functionality
            except:
                pass
            
            # Process the timepoint column if it's a date
            if is_date_column and self.timepoint_group.isEnabled():
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[timepoint_col]):
                    df[timepoint_col] = pd.to_datetime(df[timepoint_col], errors='coerce')
                
                # Get selected date handling method
                date_handling = self.date_handling_combo.currentIndex()
                
                if date_handling == 1:  # Group by intervals
                    # Create bins based on interval settings
                    interval_value = self.interval_value_spin.value()
                    interval_unit = self.interval_unit_combo.currentText().lower()
                    
                    # Create a new timepoint column based on interval
                    if interval_unit == 'days':
                        freq = f'{interval_value}D'
                    elif interval_unit == 'weeks':
                        freq = f'{interval_value}W'
                    elif interval_unit == 'months':
                        freq = f'{interval_value}M'
                    else:  # years
                        freq = f'{interval_value}Y'
                    
                    # Create a binned column rounded to the specified frequency
                    df['timepoint_binned'] = df[timepoint_col].dt.to_period(freq).dt.start_time
                    # Use the binned column for pivoting
                    timepoint_col = 'timepoint_binned'
                    
                elif date_handling == 2:  # Binned timepoints
                    try:
                        start_date = pd.to_datetime(self.bin_start_edit.text())
                        end_date = pd.to_datetime(self.bin_end_edit.text())
                        bin_count = self.bin_count_spin.value()
                        
                        # Create bins
                        bins = pd.date_range(start=start_date, end=end_date, periods=bin_count + 1)
                        
                        # Cut the dates into bins
                        df['timepoint_binned'] = pd.cut(df[timepoint_col], bins=bins, labels=False)
                        # Label bins more nicely for better column names
                        df['timepoint_binned'] = df['timepoint_binned'].apply(
                            lambda x: f"Bin{int(x) + 1}" if pd.notnull(x) else "Other")
                        
                        # Use the binned column for pivoting
                        timepoint_col = 'timepoint_binned'
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to create date bins: {str(e)}")
                        return
                
                elif date_handling == 3:  # Regular intervals with window
                    try:
                        # Get the window parameters
                        interval_value = self.window_interval_spin.value()
                        interval_unit = self.window_unit_combo.currentText().lower()
                        window_size = self.window_size_spin.value()
                        window_unit = self.window_size_unit_combo.currentText().lower()
                        
                        # Convert units to days for calculation
                        if interval_unit == 'weeks':
                            interval_days = interval_value * 7
                        elif interval_unit == 'months':
                            interval_days = interval_value * 30  # Approximate
                        else:
                            interval_days = interval_value
                            
                        if window_unit == 'weeks':
                            window_days = window_size * 7
                        else:
                            window_days = window_size
                        
                        # Use encounter column for ordering within windows if selected
                        use_encounter = self.encounter_id_check.isChecked()
                        encounter_col = self.encounter_id_combo.currentText() if use_encounter else None
                        
                        # Find the min and max dates for the dataset
                        min_date = df[timepoint_col].min()
                        max_date = df[timepoint_col].max()
                        
                        # Create the regular interval timepoints
                        interval_points = pd.date_range(
                            start=min_date, 
                            end=max_date, 
                            freq=f"{interval_days}D"
                        )
                        
                        # Process each subject
                        result_rows = []
                        for subject, subject_data in df.groupby(subject_id_col):
                            # For each interval point
                            for interval_point in interval_points:
                                start_window = interval_point - pd.Timedelta(days=window_days/2)
                                end_window = interval_point + pd.Timedelta(days=window_days/2)
                                
                                # Find rows within the window
                                window_data = subject_data[
                                    (subject_data[timepoint_col] >= start_window) & 
                                    (subject_data[timepoint_col] <= end_window)
                                ]
                                
                                if not window_data.empty:
                                    # Apply aggregation based on selected method
                                    agg_method = self.aggregation_combo.currentText().lower()
                                    
                                    if use_encounter and encounter_col in window_data.columns:
                                        # Sort by encounter ID if available
                                        window_data = window_data.sort_values(by=encounter_col)
                                    
                                    # Apply aggregation
                                    if agg_method == 'first value':
                                        value = window_data.iloc[0][value_col]
                                    elif agg_method == 'last value':
                                        value = window_data.iloc[-1][value_col]
                                    elif agg_method == 'mean':
                                        value = window_data[value_col].mean()
                                    elif agg_method == 'median':
                                        value = window_data[value_col].median()
                                    elif agg_method == 'min':
                                        value = window_data[value_col].min()
                                    elif agg_method == 'max':
                                        value = window_data[value_col].max()
                                    elif agg_method == 'count':
                                        # Just count rows - completely ignore the value column
                                        value = len(window_data)
                                    elif agg_method == 'distinct count':
                                        # Count distinct values in the column
                                        value = window_data[value_col].nunique()
                                    elif agg_method == 'closest to interval start':
                                        window_data['dist_to_start'] = abs(window_data[timepoint_col] - start_window)
                                        value = window_data.loc[window_data['dist_to_start'].idxmin()][value_col]
                                    elif agg_method == 'closest to interval middle':
                                        window_data['dist_to_middle'] = abs(window_data[timepoint_col] - interval_point)
                                        value = window_data.loc[window_data['dist_to_middle'].idxmin()][value_col]
                                    elif agg_method == 'closest to interval end':
                                        window_data['dist_to_end'] = abs(window_data[timepoint_col] - end_window)
                                        value = window_data.loc[window_data['dist_to_end'].idxmin()][value_col]
                                    else:
                                        value = window_data.iloc[0][value_col]  # Default to first value
                                    
                                    # Add the result row
                                    result_rows.append({
                                        subject_id_col: subject,
                                        'interval_point': interval_point,
                                        'value': value
                                    })
                        
                        # Create a new dataframe with the windowed data
                        if result_rows:
                            df = pd.DataFrame(result_rows)
                            timepoint_col = 'interval_point'
                        else:
                            QMessageBox.warning(self, "Error", "No data found within specified windows")
                            return
                                
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Failed to create interval windows: {str(e)}")
                        return
            
            # Handle aggregation for duplicate timepoints (only if not already handled by window processing)
            if not hasattr(self, 'date_handling_combo') or self.date_handling_combo.currentIndex() != 3 or not is_date_column:
                aggregation_method = self.aggregation_combo.currentText().lower()
                
                # Check for duplicates in the combination of subject_id and timepoint
                has_duplicates = df.duplicated(subset=[subject_id_col, timepoint_col], keep=False).any()
                
                # Always process based on aggregation method even if there are no duplicates
                # This ensures consistent behavior with "Count" and other aggregation methods
                
                # Process based on the selected aggregation method
                if aggregation_method == 'first value':
                    df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                        subset=[subject_id_col, timepoint_col], keep='first')
                elif aggregation_method == 'last value':
                    df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                        subset=[subject_id_col, timepoint_col], keep='last')
                elif aggregation_method == 'count':
                    # Just count rows in each group, completely ignore value_col
                    # Create a new dataframe with only counts
                    df = df.groupby([subject_id_col, timepoint_col]).size().reset_index(name='count_value')
                    # Override the value_col to use our new count column
                    value_col = 'count_value'
                elif aggregation_method == 'distinct count':
                    # Count distinct values in each group
                    df = df.groupby([subject_id_col, timepoint_col])[value_col].nunique().reset_index(name='distinct_count')
                    # Override the value_col to use our new distinct count column 
                    value_col = 'distinct_count'
                elif aggregation_method == 'mean':
                    df = df.groupby([subject_id_col, timepoint_col])[value_col].mean().reset_index()
                elif aggregation_method == 'median':
                    df = df.groupby([subject_id_col, timepoint_col])[value_col].median().reset_index()
                elif aggregation_method == 'min':
                    df = df.groupby([subject_id_col, timepoint_col])[value_col].min().reset_index()
                elif aggregation_method == 'max':
                    df = df.groupby([subject_id_col, timepoint_col])[value_col].max().reset_index()
                elif aggregation_method in ['closest to interval start', 'closest to interval middle', 'closest to interval end']:
                    # These are only applicable for window-based processing, so use first value as fallback
                    df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                        subset=[subject_id_col, timepoint_col], keep='first')
                else:
                    # Default to first value for any unhandled method
                    df = df.sort_values(by=[subject_id_col, timepoint_col]).drop_duplicates(
                        subset=[subject_id_col, timepoint_col], keep='first')
            
            # Use pivot to convert long to wide
            # If the timepoint column is numeric (like 1, 2, 3), convert it to string to prevent date interpretation
            if is_numeric_timepoint:
                df[timepoint_col] = df[timepoint_col].astype(str)
                
            wide_df = df.pivot(index=subject_id_col, 
                              columns=timepoint_col, 
                              values='value' if hasattr(self, 'date_handling_combo') and 
                                            self.date_handling_combo.currentIndex() == 3 and 
                                            is_date_column else value_col)
            
            # Add standard naming for columns based on the source timepoint type
            if is_date_column and self.timepoint_group.isEnabled():
                if hasattr(self, 'date_handling_combo'):
                    date_handling = self.date_handling_combo.currentIndex()
                    if date_handling == 0:  # Use as is
                        # Format dates for column names
                        wide_df.columns = [f"{value_col}_{col.strftime('%Y-%m-%d')}" for col in wide_df.columns]
                    elif date_handling == 1 or date_handling == 3:  # Group by intervals or window
                        # Format dates for column names
                        wide_df.columns = [f"{value_col}_{col.strftime('%Y-%m-%d')}" if isinstance(col, pd.Timestamp) 
                                         else f"{value_col}_{col}" for col in wide_df.columns]
                    else:  # Binned timepoints
                        wide_df.columns = [f"{value_col}_{col}" for col in wide_df.columns]
                else:
                    # Format dates for column names if date_handling_combo isn't initialized
                    wide_df.columns = [f"{value_col}_{col.strftime('%Y-%m-%d')}" if isinstance(col, pd.Timestamp) 
                                     else f"{value_col}_{col}" for col in wide_df.columns]
            elif is_numeric_timepoint:
                # For numeric timepoints, use original numeric value for column naming
                wide_df.columns = [f"{value_col}_visit{col}" for col in wide_df.columns]
            else:
                # For any other type of column
                wide_df.columns = [f"{value_col}_{col}" for col in wide_df.columns]
            
            # Reset index to make subject_id a regular column
            wide_df.reset_index(inplace=True)
            
            # Display the result
            self.preview_display.display_dataframe(wide_df)
            
            # Set default save name
            self.save_name_input.setText(f"{self.current_name}_wide")
            
            # Store the result for saving later
            self._preview_df = wide_df
            
            # Update status
            agg_method = self.aggregation_combo.currentText().lower()
            if hasattr(self, 'date_handling_combo') and self.date_handling_combo.currentIndex() == 3 and is_date_column:
                window_desc = f"every {self.window_interval_spin.value()} {self.window_unit_combo.currentText().lower()}"
                self.status_bar.showMessage(f"Converted from long to wide format - {len(wide_df)} rows (window intervals {window_desc}, {agg_method})")
            else:
                self.status_bar.showMessage(f"Converted from long to wide format - {len(wide_df)} rows (with {agg_method})")
            
            # Show the preview tab
            self.data_tabs.setCurrentIndex(1)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def convert_wide_to_long(self):
        try:
            subject_id_col = self.subject_id_combo.currentText()
            if not subject_id_col:
                QMessageBox.warning(self, "Error", "Please select a subject ID column")
                return
            value_timepoint_cols = [col for col in self.current_dataframe.columns 
                                    if re.search(r'_visit\d+|_v\d+|_timepoint\d+|_t\d+', col.lower())]
            if not value_timepoint_cols:
                QMessageBox.warning(self, "Error", "No timepoint columns detected in this dataset")
                return
            long_df = pd.melt(self.current_dataframe, id_vars=[subject_id_col], 
                              value_vars=value_timepoint_cols, 
                              var_name='variable', value_name='value')
            long_df['measure'] = long_df['variable'].apply(
                lambda x: re.search(r'(.+)_(visit|v|timepoint|t)(\d+)', x.lower()).group(1))
            long_df['visit'] = long_df['variable'].apply(
                lambda x: int(re.search(r'(.+)_(visit|v|timepoint|t)(\d+)', x.lower()).group(3)))
            long_df.drop('variable', axis=1, inplace=True)
            self.preview_display.display_dataframe(long_df)
            self._preview_df = long_df
            self.save_name_input.setText(f"{self.current_name}_long")
            self.data_tabs.setCurrentIndex(1)
            self.status_bar.showMessage(f"Converted from wide to long format - {len(long_df)} rows")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
    
    @asyncSlot()
    async def ai_wide_to_long_conversion(self):
        """Use Gemini to analyze column structure and convert wide format to long format"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        subject_id_col = self.subject_id_combo.currentText()
        if not subject_id_col:
            QMessageBox.warning(self, "Error", "Please select a subject ID column")
            return
            
        # Create a prompt for Gemini API
        prompt = f"""
        I need to convert a wide-format dataset to long format. The dataset has the following structure:
        
        Dataset Name: {self.current_name}
        Dataset Columns: {list(self.current_dataframe.columns)}
        Dataset Sample (first 3 rows):
        {self.current_dataframe.head(3).to_string()}
        
        Subject ID column: {subject_id_col}
        
        Please analyze the column names and structure to:
        1. Identify which columns represent different timepoints/visits of the same measurement
        2. Generate Python code using pandas to convert this from wide to long format
        3. The code should handle various column naming patterns
        
        IMPORTANT: Your code MUST store the final long-format result in a variable named 'long_df'.
        
        Return your response as JSON with this structure:
        {{
            "explanation": "explanation of column patterns you found",
            "id_columns": ["list of columns to keep as identifiers"],
            "pattern_columns": ["list of columns to unpivot"],
            "python_code": "pandas code to perform the conversion that MUST create a variable named 'long_df'"
        }}
        """
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset with AI. This may take a moment...")
        
        try:
            # Call Gemini API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                explanation = result.get("explanation", "")
                id_columns = result.get("id_columns", [])
                pattern_columns = result.get("pattern_columns", [])
                python_code = result.get("python_code", "")
                
                # Ensure the code creates a 'long_df' variable
                if 'long_df' not in python_code:
                    # Attempt to fix the code by adding an explicit assignment
                    python_code += "\n# Ensure result is named 'long_df'\nlong_df = result_df if 'result_df' in locals() else df_long if 'df_long' in locals() else melted_df if 'melted_df' in locals() else None"
                
                # Show a confirmation dialog with the explanation
                msg_box = QMessageBox()
                msg_box.setWindowTitle("AI Analysis")
                msg_box.setText(f"Column Pattern Analysis:\n{explanation}")
                msg_box.setInformativeText(f"ID columns: {', '.join(id_columns)}\n"
                                          f"Pattern columns: {len(pattern_columns)} columns identified\n\n"
                                          f"Do you want to apply this transformation?")
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDetailedText(f"Python code to execute:\n{python_code}")
                
                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    # Execute the Python code
                    try:
                        namespace = {
                            "pd": pd,
                            "np": np,
                            "df": self.current_dataframe,
                            "re": re
                        }
                        exec(python_code, namespace)
                        
                        # Get the result dataframe
                        if 'long_df' in namespace:
                            result_df = namespace['long_df']
                            
                            # Check if result is valid
                            if result_df is None or not isinstance(result_df, pd.DataFrame) or result_df.empty:
                                QMessageBox.warning(self, "Error", "The AI-generated code produced an empty or invalid dataframe")
                                return
                                
                            # Display the result
                            self.preview_display.display_dataframe(result_df)
                            
                            # Set default save name
                            self.save_name_input.setText(f"{self.current_name}_long")
                            
                            # Store the result for saving later
                            self._preview_df = result_df
                            
                            # Show the preview tab
                            self.data_tabs.setCurrentIndex(1)
                            
                            self.status_bar.showMessage(f"Converted from wide to long format - {len(result_df)} rows")
                        else:
                            # Try to find other variable names that might contain the result
                            possible_result_vars = ['result_df', 'df_long', 'melted_df', 'tidy_df', 'long_format_df']
                            found_var = None
                            
                            for var_name in possible_result_vars:
                                if var_name in namespace and isinstance(namespace[var_name], pd.DataFrame):
                                    found_var = var_name
                                    break
                                
                            if found_var:
                                result_df = namespace[found_var]
                                
                                # Display the result
                                self.preview_display.display_dataframe(result_df)
                                
                                # Set default save name
                                self.save_name_input.setText(f"{self.current_name}_long")
                                
                                # Store the result for saving later
                                self._preview_df = result_df
                                
                                # Show the preview tab
                                self.data_tabs.setCurrentIndex(1)
                                
                                self.status_bar.showMessage(f"Converted from wide to long format - {len(result_df)} rows")
                                
                                QMessageBox.information(self, "Variable Name Correction", 
                                                      f"Found result in variable '{found_var}' instead of 'long_df'")
                            else:
                                # If no result dataframe found, show detailed error
                                QMessageBox.warning(self, "Error", 
                                                  "The AI-generated code did not produce a result dataframe named 'long_df'.\n\n"
                                                  "Variables in the namespace: " + ", ".join([k for k in namespace.keys() 
                                                                                            if not k.startswith('_') and k not in ['pd', 'np', 'df', 're']]))
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to execute code: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI analysis failed: {str(e)}")
    
    # ---------------------------------------------------------------
    # Save the transformed dataset
    # ---------------------------------------------------------------
    def save_transformation(self):
        new_name = self.save_name_input.text()
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the transformed dataset")
            return
        if self._preview_df is None:
            QMessageBox.warning(self, "Error", "No transformation to save")
            return
        # In this simplified implementation we add the new dataset as a source
        self.add_source(new_name, SourceConnection("transformed", {}, new_name), self._preview_df)
        
        # Add the dataset to the studies manager if available
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            main_window.studies_manager.add_dataset_to_active_study(new_name, self._preview_df)
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.status_bar.showMessage(f"Transformation saved as '{new_name}'")

    def refresh_datasets_from_studies_manager(self):
        # Get reference to the main app window
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        # Get datasets from active study
        datasets = main_window.studies_manager.get_datasets_from_active_study()
        
        if not datasets:
            QMessageBox.information(self, "Info", "No datasets available in the active study")
            return
        
        # Add each dataset to our sources
        count = 0
        for name, df in datasets:
            if name not in self.dataframes or not self.dataframes[name].equals(df):
                self.add_source(name, SourceConnection("study", {}, name), df)
                count += 1
        
        self.status_bar.showMessage(f"Refreshed {count} datasets from active study")

# ---------------------------------------------------------------------
# Simple class to hold source connection details (retained for compatibility)
# ---------------------------------------------------------------------
class SourceConnection:
    def __init__(self, source_type, connection_params, name):
        self.source_type = source_type  # e.g., transformed, application, etc.
        self.connection_params = connection_params
        self.name = name
