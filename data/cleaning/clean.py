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
    QTabWidget, QStatusBar, QDoubleSpinBox, QCheckBox, QFileDialog, QSlider,
    QTextEdit
)
from PyQt6.QtGui import QIcon, QFont, QColor
import re
import asyncio
from llms.client import call_llm_async, call_llm_async_json
from qasync import asyncSlot
from scipy import stats
import matplotlib.pyplot as plt
from io import BytesIO
from PyQt6.QtCore import QBuffer, QByteArray
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QTimer
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
# Main Data Cleaning Widget
# ---------------------------------------------------------------------
class DataCleaningWidget(QWidget):
    # Signal emitted when a source (or cleaned dataset) is selected
    source_selected = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Cleaning")
        
        # Internal storage for data sources and datasets
        self.dataframes = {}
        self.current_name = ""
        self.current_dataframe = None
        self._preview_df = None
        self.cleaning_history = []  # Track cleaning operations for reporting
        self.column_stats = {}  # Store column statistics
        self.applicable_processes = {}  # Store LLM-recommended processes

        # Build the UI
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Main splitter dividing left (dataset selection) and right (cleaning operations)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --------------------------
        # Left section: Dataset selection and info
        # --------------------------
        left_section = QWidget()
        left_layout = QVBoxLayout(left_section)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Dataset selection
        sources_group = QGroupBox("Available Data Sources")
        sources_layout = QVBoxLayout(sources_group)
        
        # Add refresh button at the top of the sources list
        refresh_layout = QHBoxLayout()
        refresh_button = QPushButton("Refresh from Studies")
        refresh_button.setIcon(load_bootstrap_icon("arrow-clockwise"))
        refresh_button.setToolTip("Refresh datasets from active study")
        refresh_button.clicked.connect(self.refresh_datasets_from_studies_manager)
        refresh_layout.addWidget(refresh_button)
        refresh_layout.addStretch()
        sources_layout.addLayout(refresh_layout)
        
        # Add a label to indicate how to load data
        sources_hint = QLabel("Click 'Refresh from Studies' to load datasets from the active study")
        sources_hint.setWordWrap(True)
        sources_hint.setStyleSheet("color: #666;")
        sources_layout.addWidget(sources_hint)
        
        # List of available datasets
        self.sources_list = QListWidget()
        self.sources_list.itemClicked.connect(self.on_source_selected)
        sources_layout.addWidget(self.sources_list)
        
        left_layout.addWidget(sources_group)
        
        # Dataset summary
        summary_group = QGroupBox("Dataset Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        left_layout.addWidget(summary_group)
        
        main_splitter.addWidget(left_section)
        
        # --------------------------
        # Right section: Cleaning operations and preview
        # --------------------------
        right_section = QWidget()
        right_layout = QVBoxLayout(right_section)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Dataset name and info
        header_layout = QHBoxLayout()
        self.current_dataset_label = QLabel("No dataset selected")
        self.current_dataset_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.current_dataset_label)
        self.dataset_info_label = QLabel("")
        header_layout.addWidget(self.dataset_info_label)
        header_layout.addStretch()
        right_layout.addLayout(header_layout)
        
        # Cleaning operations tabs
        self.cleaning_tabs = QTabWidget()
        
        # Tab 1: Duplicates
        duplicates_tab = self.create_duplicates_tab()
        self.cleaning_tabs.addTab(duplicates_tab, "Duplicates")
        
        # Tab 2: Missing Values
        missing_values_tab = self.create_missing_values_tab()
        self.cleaning_tabs.addTab(missing_values_tab, "Missing Values")
        
        # Tab 3: Outliers
        outliers_tab = self.create_outliers_tab()
        self.cleaning_tabs.addTab(outliers_tab, "Outliers")
        
        # Tab 4: Type Conversion
        type_conversion_tab = self.create_type_conversion_tab()
        self.cleaning_tabs.addTab(type_conversion_tab, "Type Conversion")
        
        # Tab 5: Column Operations
        column_operations_tab = self.create_column_operations_tab()
        self.cleaning_tabs.addTab(column_operations_tab, "Column Operations")
        
        # Tab 7: Text Cleaning
        text_cleaning_tab = self.create_text_cleaning_tab()
        self.cleaning_tabs.addTab(text_cleaning_tab, "Text Cleaning")
        
        # Tab 8: AI Assistance
        ai_assistance_tab = self.create_ai_assistance_tab()
        self.cleaning_tabs.addTab(ai_assistance_tab, "AI Assistance")
        
        # Tab 9: Cleaning History
        history_tab = self.create_history_tab()
        self.cleaning_tabs.addTab(history_tab, "Cleaning History")
        
        right_layout.addWidget(self.cleaning_tabs)
        
        # Preview and save section
        preview_group = QGroupBox("Preview & Save")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Transformation Preview"))
        preview_header.addStretch()
        preview_header.addWidget(QLabel("Save As:"))
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for cleaned dataset")
        preview_header.addWidget(self.save_name_input)
        save_button = QPushButton("Save")
        save_button.setIcon(load_bootstrap_icon("save"))
        save_button.clicked.connect(self.save_cleaned_dataset)
        preview_header.addWidget(save_button)
        preview_layout.addLayout(preview_header)
        
        self.preview_display = DataFrameDisplay()
        preview_layout.addWidget(self.preview_display)
        
        right_layout.addWidget(preview_group)
        
        main_splitter.addWidget(right_section)
        
        # Set initial sizes for the splitter
        main_splitter.setSizes([300, 700])
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Force the widget to have a reasonable minimum size
        self.setMinimumSize(1000, 700)
    
    # ---------------------------------------------------------------
    # Create tab for handling duplicates
    # ---------------------------------------------------------------
    def create_duplicates_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Identify and remove duplicate entries in your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Duplicate detection settings
        settings_group = QGroupBox("Duplicate Detection Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Column selection for duplicate checking
        columns_label = QLabel("Check for duplicates based on:")
        columns_label.setToolTip("Select which columns to consider when identifying duplicates")
        settings_layout.addWidget(columns_label)
        
        self.duplicate_columns_list = QListWidget()
        self.duplicate_columns_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        settings_layout.addWidget(self.duplicate_columns_list)
        
        # Keep options
        keep_layout = QHBoxLayout()
        keep_layout.addWidget(QLabel("When duplicates are found, keep:"))
        self.duplicate_keep_combo = QComboBox()
        self.duplicate_keep_combo.addItems(["First occurrence", "Last occurrence", "None (remove all)"])
        keep_layout.addWidget(self.duplicate_keep_combo)
        settings_layout.addLayout(keep_layout)
        
        # Ignore index option
        self.ignore_index_check = QCheckBox("Ignore index when detecting duplicates")
        self.ignore_index_check.setChecked(True)
        self.ignore_index_check.setToolTip("If checked, only consider the values in the selected columns, not the row index")
        settings_layout.addWidget(self.ignore_index_check)
        
        layout.addWidget(settings_group)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        find_button = QPushButton("Find Duplicates")
        find_button.setIcon(load_bootstrap_icon("search"))
        find_button.clicked.connect(self.find_duplicates)
        buttons_layout.addWidget(find_button)
        
        remove_button = QPushButton("Remove Duplicates")
        remove_button.setIcon(load_bootstrap_icon("trash"))
        remove_button.clicked.connect(self.remove_duplicates)
        buttons_layout.addWidget(remove_button)
        
        layout.addLayout(buttons_layout)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.duplicates_result_label = QLabel("No duplicates checked yet")
        results_layout.addWidget(self.duplicates_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for handling missing values
    # ---------------------------------------------------------------
    def create_missing_values_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Identify and handle missing values in your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Missing value detection
        detection_group = QGroupBox("Missing Value Detection")
        detection_layout = QVBoxLayout(detection_group)
        
        # Options for what to consider as missing
        consider_layout = QVBoxLayout()
        consider_label = QLabel("Consider as missing:")
        consider_layout.addWidget(consider_label)
        
        self.consider_na_check = QCheckBox("NaN / None / null")
        self.consider_na_check.setChecked(True)
        consider_layout.addWidget(self.consider_na_check)
        
        self.consider_empty_check = QCheckBox("Empty strings")
        self.consider_empty_check.setChecked(True)
        consider_layout.addWidget(self.consider_empty_check)
        
        self.consider_whitespace_check = QCheckBox("Whitespace-only strings")
        self.consider_whitespace_check.setChecked(True)
        consider_layout.addWidget(self.consider_whitespace_check)
        
        custom_layout = QHBoxLayout()
        self.consider_custom_check = QCheckBox("Custom values:")
        custom_layout.addWidget(self.consider_custom_check)
        self.custom_missing_input = QLineEdit()
        self.custom_missing_input.setPlaceholderText("e.g., 'NA', 'missing', '-999' (comma-separated)")
        custom_layout.addWidget(self.custom_missing_input)
        consider_layout.addLayout(custom_layout)
        
        detection_layout.addLayout(consider_layout)
        
        # Find missing values button
        find_missing_button = QPushButton("Find Missing Values")
        find_missing_button.setIcon(load_bootstrap_icon("search"))
        find_missing_button.clicked.connect(self.find_missing_values)
        detection_layout.addWidget(find_missing_button)
        
        layout.addWidget(detection_group)
        
        # Missing value handling
        handling_group = QGroupBox("Missing Value Handling")
        handling_layout = QGridLayout(handling_group)
        
        # Column selection
        handling_layout.addWidget(QLabel("Column:"), 0, 0)
        self.missing_column_combo = QComboBox()
        handling_layout.addWidget(self.missing_column_combo, 0, 1, 1, 3)
        
        # Method selection
        handling_layout.addWidget(QLabel("Method:"), 1, 0)
        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "Remove rows", 
            "Replace with mean", 
            "Replace with median", 
            "Replace with mode",
            "Replace with constant", 
            "Replace with interpolation", 
            "Replace with forward fill", 
            "Replace with backward fill"
        ])
        self.missing_method_combo.currentTextChanged.connect(self.on_missing_method_changed)
        handling_layout.addWidget(self.missing_method_combo, 1, 1, 1, 3)
        
        # Value for constant replacement
        self.constant_value_label = QLabel("Constant value:")
        handling_layout.addWidget(self.constant_value_label, 2, 0)
        self.constant_value_input = QLineEdit()
        handling_layout.addWidget(self.constant_value_input, 2, 1, 1, 3)
        
        # Apply button
        apply_missing_button = QPushButton("Apply")
        apply_missing_button.setIcon(load_bootstrap_icon("check"))
        apply_missing_button.clicked.connect(self.handle_missing_values)
        handling_layout.addWidget(apply_missing_button, 3, 3)
        
        layout.addWidget(handling_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.missing_result_label = QLabel("No missing value analysis yet")
        results_layout.addWidget(self.missing_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.constant_value_label.setVisible(False)
        self.constant_value_input.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for handling outliers
    # ---------------------------------------------------------------
    def create_outliers_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Detect and handle outliers in numerical data.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Outlier detection
        detection_group = QGroupBox("Outlier Detection")
        detection_layout = QGridLayout(detection_group)
        
        # Column selection
        detection_layout.addWidget(QLabel("Column:"), 0, 0)
        self.outlier_column_combo = QComboBox()
        self.outlier_column_combo.currentTextChanged.connect(self.on_outlier_column_changed)
        detection_layout.addWidget(self.outlier_column_combo, 0, 1, 1, 3)
        
        # Method selection
        detection_layout.addWidget(QLabel("Method:"), 1, 0)
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems([
            "Z-score", 
            "IQR (Interquartile Range)", 
            "Modified Z-score", 
            "Percentile"
        ])
        self.outlier_method_combo.currentTextChanged.connect(self.on_outlier_method_changed)
        detection_layout.addWidget(self.outlier_method_combo, 1, 1, 1, 3)
        
        # Threshold settings
        self.z_threshold_label = QLabel("Z-score threshold:")
        detection_layout.addWidget(self.z_threshold_label, 2, 0)
        self.z_threshold_input = QDoubleSpinBox()
        self.z_threshold_input.setRange(1.0, 10.0)
        self.z_threshold_input.setValue(3.0)
        self.z_threshold_input.setSingleStep(0.1)
        detection_layout.addWidget(self.z_threshold_input, 2, 1)
        
        self.iqr_multiplier_label = QLabel("IQR multiplier:")
        detection_layout.addWidget(self.iqr_multiplier_label, 2, 0)
        self.iqr_multiplier_input = QDoubleSpinBox()
        self.iqr_multiplier_input.setRange(0.5, 5.0)
        self.iqr_multiplier_input.setValue(1.5)
        self.iqr_multiplier_input.setSingleStep(0.1)
        detection_layout.addWidget(self.iqr_multiplier_input, 2, 1)
        
        self.percentile_label = QLabel("Percentile threshold:")
        detection_layout.addWidget(self.percentile_label, 2, 0)
        self.percentile_input = QDoubleSpinBox()
        self.percentile_input.setRange(90.0, 99.9)
        self.percentile_input.setValue(95.0)
        self.percentile_input.setSingleStep(0.1)
        detection_layout.addWidget(self.percentile_input, 2, 1)
        
        # Visualization
        self.outlier_visualization = QLabel("Select a numeric column to see distribution")
        self.outlier_visualization.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.outlier_visualization.setMinimumHeight(200)
        self.outlier_visualization.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd;")
        detection_layout.addWidget(self.outlier_visualization, 3, 0, 1, 4)
        
        # Detect button
        detect_outliers_button = QPushButton("Detect Outliers")
        detect_outliers_button.setIcon(load_bootstrap_icon("search"))
        detect_outliers_button.clicked.connect(self.detect_outliers)
        detection_layout.addWidget(detect_outliers_button, 4, 3)
        
        layout.addWidget(detection_group)
        
        # Outlier handling
        handling_group = QGroupBox("Outlier Handling")
        handling_layout = QGridLayout(handling_group)
        
        # Method selection
        handling_layout.addWidget(QLabel("Handling Method:"), 0, 0)
        self.outlier_handling_combo = QComboBox()
        self.outlier_handling_combo.addItems([
            "Remove outliers", 
            "Cap at threshold", 
            "Replace with mean", 
            "Replace with median", 
            "Replace with custom value"
        ])
        self.outlier_handling_combo.currentTextChanged.connect(self.on_outlier_handling_changed)
        handling_layout.addWidget(self.outlier_handling_combo, 0, 1, 1, 2)
        
        # Custom value
        self.outlier_custom_label = QLabel("Custom value:")
        handling_layout.addWidget(self.outlier_custom_label, 1, 0)
        self.outlier_custom_input = QLineEdit()
        handling_layout.addWidget(self.outlier_custom_input, 1, 1, 1, 2)
        
        # Apply button
        apply_outlier_button = QPushButton("Apply")
        apply_outlier_button.setIcon(load_bootstrap_icon("check"))
        apply_outlier_button.clicked.connect(self.handle_outliers)
        handling_layout.addWidget(apply_outlier_button, 2, 2)
        
        layout.addWidget(handling_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.outlier_result_label = QLabel("No outlier analysis yet")
        results_layout.addWidget(self.outlier_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.z_threshold_label.setVisible(True)
        self.z_threshold_input.setVisible(True)
        self.iqr_multiplier_label.setVisible(False)
        self.iqr_multiplier_input.setVisible(False)
        self.percentile_label.setVisible(False)
        self.percentile_input.setVisible(False)
        self.outlier_custom_label.setVisible(False)
        self.outlier_custom_input.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for type conversion
    # ---------------------------------------------------------------
    def create_type_conversion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Convert data types and formats for columns in your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Type conversion settings
        conversion_group = QGroupBox("Type Conversion")
        conversion_layout = QGridLayout(conversion_group)
        
        # Column selection
        conversion_layout.addWidget(QLabel("Column:"), 0, 0)
        self.conversion_column_combo = QComboBox()
        self.conversion_column_combo.currentTextChanged.connect(self.on_conversion_column_changed)
        conversion_layout.addWidget(self.conversion_column_combo, 0, 1, 1, 3)
        
        # Current type
        conversion_layout.addWidget(QLabel("Current type:"), 1, 0)
        self.current_type_label = QLabel("N/A")
        conversion_layout.addWidget(self.current_type_label, 1, 1, 1, 3)
        
        # Target type
        conversion_layout.addWidget(QLabel("Convert to:"), 2, 0)
        self.target_type_combo = QComboBox()
        self.target_type_combo.addItems([
            "Integer (int)", 
            "Float (decimal)", 
            "String (text)", 
            "Boolean (True/False)", 
            "Date/Time", 
            "Categorical"
        ])
        self.target_type_combo.currentTextChanged.connect(self.on_target_type_changed)
        conversion_layout.addWidget(self.target_type_combo, 2, 1, 1, 3)
        
        # Date format
        self.date_format_label = QLabel("Date format:")
        conversion_layout.addWidget(self.date_format_label, 3, 0)
        self.date_format_input = QLineEdit("%Y-%m-%d")
        self.date_format_input.setToolTip("Format string using directives: %Y (year), %m (month), %d (day), %H (hour), %M (minute), %S (second)")
        conversion_layout.addWidget(self.date_format_input, 3, 1, 1, 3)
        
        # Bool mapping
        self.bool_true_label = QLabel("True values:")
        conversion_layout.addWidget(self.bool_true_label, 3, 0)
        self.bool_true_input = QLineEdit("1, True, Yes, Y")
        conversion_layout.addWidget(self.bool_true_input, 3, 1, 1, 3)
        
        self.bool_false_label = QLabel("False values:")
        conversion_layout.addWidget(self.bool_false_label, 4, 0)
        self.bool_false_input = QLineEdit("0, False, No, N")
        conversion_layout.addWidget(self.bool_false_input, 4, 1, 1, 3)
        
        # Error handling
        conversion_layout.addWidget(QLabel("On error:"), 5, 0)
        self.error_handling_combo = QComboBox()
        self.error_handling_combo.addItems([
            "Coerce (convert errors to NaN)", 
            "Raise (stop on first error)"
        ])
        conversion_layout.addWidget(self.error_handling_combo, 5, 1, 1, 3)
        
        # Apply button
        convert_button = QPushButton("Apply Conversion")
        convert_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        convert_button.clicked.connect(self.apply_type_conversion)
        conversion_layout.addWidget(convert_button, 6, 3)
        
        layout.addWidget(conversion_group)
        
        # Batch conversions
        batch_group = QGroupBox("Batch Conversions")
        batch_layout = QVBoxLayout(batch_group)
        
        batch_buttons_layout = QHBoxLayout()
        
        auto_numeric_button = QPushButton("Auto-Convert Numeric")
        auto_numeric_button.setIcon(load_bootstrap_icon("123"))
        auto_numeric_button.setToolTip("Automatically convert string columns to numeric where possible")
        auto_numeric_button.clicked.connect(self.auto_convert_numeric)
        batch_buttons_layout.addWidget(auto_numeric_button)
        
        auto_date_button = QPushButton("Auto-Convert Dates")
        auto_date_button.setIcon(load_bootstrap_icon("calendar"))
        auto_date_button.setToolTip("Automatically detect and convert date columns")
        auto_date_button.clicked.connect(self.auto_convert_dates)
        batch_buttons_layout.addWidget(auto_date_button)
        
        optimize_button = QPushButton("Optimize Data Types")
        optimize_button.setIcon(load_bootstrap_icon("speedometer"))
        optimize_button.setToolTip("Optimize memory usage by converting to appropriate data types")
        optimize_button.clicked.connect(self.optimize_data_types)
        batch_buttons_layout.addWidget(optimize_button)
        
        batch_layout.addLayout(batch_buttons_layout)
        layout.addWidget(batch_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.conversion_result_label = QLabel("No conversions performed yet")
        results_layout.addWidget(self.conversion_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.date_format_label.setVisible(False)
        self.date_format_input.setVisible(False)
        self.bool_true_label.setVisible(False)
        self.bool_true_input.setVisible(False)
        self.bool_false_label.setVisible(False)
        self.bool_false_input.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for column operations
    # ---------------------------------------------------------------
    def create_column_operations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Perform operations on columns like renaming, scaling, normalization, or creating derived columns.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Renaming columns
        rename_group = QGroupBox("Rename Columns")
        rename_layout = QGridLayout(rename_group)
        
        rename_layout.addWidget(QLabel("Column:"), 0, 0)
        self.rename_column_combo = QComboBox()
        rename_layout.addWidget(self.rename_column_combo, 0, 1)
        
        rename_layout.addWidget(QLabel("New name:"), 1, 0)
        self.new_column_name = QLineEdit()
        rename_layout.addWidget(self.new_column_name, 1, 1)
        
        rename_button = QPushButton("Rename")
        rename_button.setIcon(load_bootstrap_icon("pencil"))
        rename_button.clicked.connect(self.rename_column)
        rename_layout.addWidget(rename_button, 2, 1)
        
        layout.addWidget(rename_group)
        
        # Column transformations
        transform_group = QGroupBox("Column Transformations")
        transform_layout = QGridLayout(transform_group)
        
        transform_layout.addWidget(QLabel("Column:"), 0, 0)
        self.transform_column_combo = QComboBox()
        self.transform_column_combo.currentTextChanged.connect(self.on_transform_column_changed)
        transform_layout.addWidget(self.transform_column_combo, 0, 1)
        
        transform_layout.addWidget(QLabel("Transformation:"), 1, 0)
        self.transform_type_combo = QComboBox()
        self.transform_type_combo.addItems([
            "Z-score normalization", 
            "Min-max scaling", 
            "Log transformation", 
            "Square root", 
            "Absolute value",
            "Bin into categories"
        ])
        self.transform_type_combo.currentTextChanged.connect(self.on_transform_type_changed)
        transform_layout.addWidget(self.transform_type_combo, 1, 1)
        
        # Bin settings
        self.bin_count_label = QLabel("Number of bins:")
        transform_layout.addWidget(self.bin_count_label, 2, 0)
        self.bin_count_spin = QSpinBox()
        self.bin_count_spin.setRange(2, 20)
        self.bin_count_spin.setValue(5)
        transform_layout.addWidget(self.bin_count_spin, 2, 1)
        
        # Output type
        transform_layout.addWidget(QLabel("Output:"), 3, 0)
        self.transform_output_combo = QComboBox()
        self.transform_output_combo.addItems([
            "Replace original column", 
            "Create new column"
        ])
        transform_layout.addWidget(self.transform_output_combo, 3, 1)
        
        # New column name
        self.transform_newcol_label = QLabel("New column name:")
        transform_layout.addWidget(self.transform_newcol_label, 4, 0)
        self.transform_newcol_input = QLineEdit()
        transform_layout.addWidget(self.transform_newcol_input, 4, 1)
        
        transform_button = QPushButton("Apply Transformation")
        transform_button.setIcon(load_bootstrap_icon("gear"))
        transform_button.clicked.connect(self.apply_column_transformation)
        transform_layout.addWidget(transform_button, 5, 1)
        
        layout.addWidget(transform_group)
        
        # Derived columns
        derived_group = QGroupBox("Create Derived Column")
        derived_layout = QVBoxLayout(derived_group)
        
        derived_layout.addWidget(QLabel("New column expression (use pandas syntax):"))
        self.derived_column_expr = QPlainTextEdit()
        self.derived_column_expr.setPlaceholderText("Examples:\n"
                                                   "df['A'] + df['B']  # Sum of columns A and B\n"
                                                   "df['Age'] > 18  # Boolean expression\n"
                                                   "pd.to_datetime(df['Date']).dt.year  # Extract year from date")
        self.derived_column_expr.setMaximumHeight(80)
        derived_layout.addWidget(self.derived_column_expr)
        
        new_name_layout = QHBoxLayout()
        new_name_layout.addWidget(QLabel("New column name:"))
        self.derived_column_name = QLineEdit()
        new_name_layout.addWidget(self.derived_column_name)
        derived_layout.addLayout(new_name_layout)
        
        create_button = QPushButton("Create Column")
        create_button.setIcon(load_bootstrap_icon("plus-circle"))
        create_button.clicked.connect(self.create_derived_column)
        derived_layout.addWidget(create_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(derived_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.column_ops_result_label = QLabel("No column operations performed yet")
        results_layout.addWidget(self.column_ops_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.bin_count_label.setVisible(False)
        self.bin_count_spin.setVisible(False)
        self.transform_newcol_label.setVisible(False)
        self.transform_newcol_input.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for text cleaning
    # ---------------------------------------------------------------
    def create_text_cleaning_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Clean and standardize text data in your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Text column selection
        column_layout = QHBoxLayout()
        column_layout.addWidget(QLabel("Text column:"))
        self.text_column_combo = QComboBox()
        self.text_column_combo.currentTextChanged.connect(self.on_text_column_changed)
        column_layout.addWidget(self.text_column_combo)
        layout.addLayout(column_layout)
        
        # Text cleaning operations
        cleaning_group = QGroupBox("Cleaning Operations")
        cleaning_layout = QVBoxLayout(cleaning_group)
        
        # Case normalization
        case_layout = QHBoxLayout()
        case_layout.addWidget(QLabel("Case:"))
        self.case_combo = QComboBox()
        self.case_combo.addItems(["No change", "Lowercase", "Uppercase", "Title Case"])
        case_layout.addWidget(self.case_combo)
        cleaning_layout.addLayout(case_layout)
        
        # Whitespace handling
        self.strip_whitespace_check = QCheckBox("Strip leading and trailing whitespace")
        self.strip_whitespace_check.setChecked(True)
        cleaning_layout.addWidget(self.strip_whitespace_check)
        
        self.normalize_whitespace_check = QCheckBox("Normalize internal whitespace (replace multiple spaces with single space)")
        self.normalize_whitespace_check.setChecked(True)
        cleaning_layout.addWidget(self.normalize_whitespace_check)
        
        # Special character handling
        self.remove_punctuation_check = QCheckBox("Remove punctuation")
        cleaning_layout.addWidget(self.remove_punctuation_check)
        
        self.remove_numbers_check = QCheckBox("Remove numeric characters")
        cleaning_layout.addWidget(self.remove_numbers_check)
        
        self.remove_special_chars_check = QCheckBox("Remove special characters")
        cleaning_layout.addWidget(self.remove_special_chars_check)
        
        # Advanced options
        advanced_layout = QHBoxLayout()
        self.custom_chars_check = QCheckBox("Remove custom characters:")
        advanced_layout.addWidget(self.custom_chars_check)
        self.custom_chars_input = QLineEdit()
        self.custom_chars_input.setPlaceholderText("e.g., $%#@")
        advanced_layout.addWidget(self.custom_chars_input)
        cleaning_layout.addLayout(advanced_layout)
        
        # Replacement options
        replace_layout = QGridLayout()
        self.find_replace_check = QCheckBox("Find and replace pattern:")
        replace_layout.addWidget(self.find_replace_check, 0, 0)
        replace_layout.addWidget(QLabel("Find:"), 1, 0)
        self.find_pattern_input = QLineEdit()
        replace_layout.addWidget(self.find_pattern_input, 1, 1)
        replace_layout.addWidget(QLabel("Replace with:"), 1, 2)
        self.replace_pattern_input = QLineEdit()
        replace_layout.addWidget(self.replace_pattern_input, 1, 3)
        cleaning_layout.addLayout(replace_layout)
        
        # Output options
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.text_output_combo = QComboBox()
        self.text_output_combo.addItems(["Replace original column", "Create new column"])
        self.text_output_combo.currentTextChanged.connect(self.on_text_output_changed)
        output_layout.addWidget(self.text_output_combo)
        cleaning_layout.addLayout(output_layout)
        
        new_col_layout = QHBoxLayout()
        self.text_newcol_label = QLabel("New column name:")
        new_col_layout.addWidget(self.text_newcol_label)
        self.text_newcol_input = QLineEdit()
        new_col_layout.addWidget(self.text_newcol_input)
        cleaning_layout.addLayout(new_col_layout)
        
        # Apply button
        apply_text_button = QPushButton("Apply Text Cleaning")
        apply_text_button.setIcon(load_bootstrap_icon("type"))
        apply_text_button.clicked.connect(self.apply_text_cleaning)
        cleaning_layout.addWidget(apply_text_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(cleaning_group)
        
        # Standardization
        standard_group = QGroupBox("Text Standardization")
        standard_layout = QVBoxLayout(standard_group)
        
        # Predefined standardization patterns
        standard_layout.addWidget(QLabel("Predefined patterns:"))
        self.predefined_patterns = QComboBox()
        self.predefined_patterns.addItems([
            "Select pattern...",
            "Phone numbers",
            "Email addresses",
            "URLs",
            "Dates",
            "Currency values",
            "US States",
            "Countries",
            "Yes/No variations"
        ])
        self.predefined_patterns.currentTextChanged.connect(self.on_predefined_pattern_changed)
        standard_layout.addWidget(self.predefined_patterns)
        
        # Apply standardization button
        apply_standard_button = QPushButton("Apply Standardization")
        apply_standard_button.setIcon(load_bootstrap_icon("list-check"))
        apply_standard_button.clicked.connect(self.apply_text_standardization)
        standard_layout.addWidget(apply_standard_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(standard_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.text_result_label = QLabel("No text cleaning applied yet")
        results_layout.addWidget(self.text_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.text_newcol_label.setVisible(False)
        self.text_newcol_input.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for AI assistance
    # ---------------------------------------------------------------
    def create_ai_assistance_tab(self):
        """Create the AI assistance tab with improved step-by-step execution"""
        ai_tab = QWidget()
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("This tab uses AI to analyze your dataset and create a custom cleaning plan.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create area for AI response
        self.ai_response_text = QTextEdit()
        self.ai_response_text.setReadOnly(True)
        self.ai_response_text.setMinimumHeight(300)
        layout.addWidget(self.ai_response_text)
        
        # Progress indicator
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("No cleaning plan generated yet")
        progress_layout.addWidget(self.progress_label)
        
        # Execution controls
        button_layout = QHBoxLayout()
        
        generate_button = QPushButton("Analyze Dataset & Generate Plan")
        generate_button.setIcon(load_bootstrap_icon("robot"))
        generate_button.clicked.connect(self.determine_applicable_processes)
        button_layout.addWidget(generate_button)
        
        self.execute_step_button = QPushButton("Execute Next Step")
        self.execute_step_button.setIcon(load_bootstrap_icon("play"))
        self.execute_step_button.clicked.connect(self.execute_current_step)
        self.execute_step_button.setEnabled(False)
        button_layout.addWidget(self.execute_step_button)
        
        progress_layout.addLayout(button_layout)
        layout.addWidget(progress_widget)
        
        # Store the current cleaning plan and step tracking
        self.current_cleaning_plan = None
        self.current_step = 0
        self.total_steps = 0
        
        # Add step navigation section
        step_nav_group = QGroupBox("Step Navigation")
        step_nav_layout = QVBoxLayout()
        
        # Step indicator
        self.ai_step_indicator = QLabel("No steps in progress")
        step_nav_layout.addWidget(self.ai_step_indicator)
        
        # Stay in AI tab checkbox
        self.stay_in_ai_tab_checkbox = QCheckBox("Stay in AI tab while executing steps")
        self.stay_in_ai_tab_checkbox.setChecked(True)
        step_nav_layout.addWidget(self.stay_in_ai_tab_checkbox)
        
        # Navigation buttons
        nav_buttons_layout = QHBoxLayout()
        self.prev_step_button = QPushButton("Previous Step")
        self.prev_step_button.setIcon(load_bootstrap_icon("arrow-left"))
        self.prev_step_button.setEnabled(False)
        self.prev_step_button.clicked.connect(self.go_to_previous_step)
        
        self.next_step_button = QPushButton("Next Step")
        self.next_step_button.setIcon(load_bootstrap_icon("arrow-right"))
        self.next_step_button.setEnabled(False)
        self.next_step_button.clicked.connect(self.execute_current_step)
        
        self.apply_all_button = QPushButton("Apply All Steps")
        self.apply_all_button.setIcon(load_bootstrap_icon("lightning"))
        self.apply_all_button.clicked.connect(self.apply_all_steps)
        
        nav_buttons_layout.addWidget(self.prev_step_button)
        nav_buttons_layout.addWidget(self.next_step_button)
        nav_buttons_layout.addWidget(self.apply_all_button)
        
        step_nav_layout.addLayout(nav_buttons_layout)
        step_nav_group.setLayout(step_nav_layout)
        
        # Add to main layout
        layout.addWidget(step_nav_group)
        
        ai_tab.setLayout(layout)
        return ai_tab
    
    # ---------------------------------------------------------------
    # Create tab for cleaning history
    # ---------------------------------------------------------------
    def create_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Track all cleaning operations performed on your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # History display
        self.history_text = QPlainTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText("No cleaning operations performed yet.")
        layout.addWidget(self.history_text)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        export_button = QPushButton("Export Cleaning Report")
        export_button.setIcon(load_bootstrap_icon("file-earmark-text"))
        export_button.clicked.connect(self.export_cleaning_report)
        buttons_layout.addWidget(export_button)
        
        reset_button = QPushButton("Reset Dataset")
        reset_button.setIcon(load_bootstrap_icon("arrow-counterclockwise"))
        reset_button.clicked.connect(self.reset_dataset)
        buttons_layout.addWidget(reset_button)
        
        layout.addLayout(buttons_layout)
        
        return tab
    
    # ---------------------------------------------------------------
    # Methods for handling dataset selection and display
    # ---------------------------------------------------------------
    def refresh_datasets_from_studies_manager(self):
        """Refresh datasets from the active study in the studies manager"""
        # Get reference to the main app window
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        # Get active study
        study = main_window.studies_manager.get_active_study()
        if not study:
            QMessageBox.warning(self, "Error", "No active study found")
            return
        
        # Check if study has datasets
        if not hasattr(study, 'available_datasets') or not study.available_datasets:
            QMessageBox.information(self, "Info", "No datasets available in the active study")
            return
        
        # Process each dataset
        count = 0
        for dataset in study.available_datasets:
            # Handle both dictionary and namedtuple formats
            if isinstance(dataset, dict):
                name = dataset.get('name')
                dataframe = dataset.get('data')
                metadata = dataset.get('metadata')
            else:
                # Handle legacy namedtuple format
                name = dataset.name
                dataframe = dataset.data
                metadata = None
            
            if name and isinstance(dataframe, pd.DataFrame):
                # Add source - DataCleaningWidget has a different structure than DataCollectionWidget
                # Just call add_source which will handle the dataset storage
                self.add_source(name, dataframe, metadata)
                count += 1
        
        if count > 0:
            QMessageBox.information(self, "Success", f"Successfully refreshed {count} datasets")
        else:
            QMessageBox.information(self, "Info", "All datasets are already up to date")
    
    def add_source(self, name, dataframe, metadata=None):
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
            self._preview_df = dataframe.copy()
            self.display_dataset(name, dataframe)
            self.update_column_selections()
            self.calculate_dataset_summary()
            self.cleaning_history = []
            self.update_history()
    
    def display_dataset(self, name, dataframe):
        self.current_dataset_label.setText(f"Dataset: {name}")
        rows, cols = dataframe.shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")
        self.preview_display.display_dataframe(dataframe)
        self.save_name_input.setText(f"{name}_cleaned")
    
    def calculate_dataset_summary(self):
        """Calculate and display summary statistics for the dataset"""
        if self.current_dataframe is None:
            return
        
        df = self.current_dataframe
        rows, cols = df.shape
        
        # Basic info
        summary = f"Dataset: {self.current_name}\n"
        summary += f"Rows: {rows}, Columns: {cols}\n\n"
        
        # Data types
        dtypes = df.dtypes.value_counts()
        summary += "Data Types:\n"
        for dtype, count in dtypes.items():
            summary += f"  {dtype}: {count} columns\n"
        
        # Missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        
        if not missing_cols.empty:
            summary += "\nMissing Values:\n"
            for col, count in missing_cols.items():
                pct = (count / rows) * 100
                summary += f"  {col}: {count} ({pct:.1f}%)\n"
        else:
            summary += "\nNo missing values detected.\n"
        
        # Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            summary += f"\nDuplicate Rows: {dup_count} ({(dup_count / rows) * 100:.1f}%)\n"
        else:
            summary += "\nNo duplicate rows detected.\n"
        
        # Calculate column stats for numeric columns
        self.column_stats = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            summary += "\nNumeric Column Statistics:\n"
            for col in numeric_cols:
                stats = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'std': df[col].std()
                }
                
                # Check for potential outliers using IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                
                stats['outliers'] = outlier_count
                self.column_stats[col] = stats
                
                summary += f"  {col}: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, "
                summary += f"Max={stats['max']:.2f}, Potential Outliers: {outlier_count}\n"
        
        self.summary_text.setPlainText(summary)
    
    def update_column_selections(self):
        """Update all combo boxes with column names"""
        if self.current_dataframe is None:
            return
            
        columns = list(self.current_dataframe.columns)
        
        # Update all combo boxes with column names
        for combo in [
            self.duplicate_columns_list, self.missing_column_combo, self.outlier_column_combo,
            self.conversion_column_combo, self.rename_column_combo, self.transform_column_combo,
        ]:
            if isinstance(combo, QComboBox):
                combo.clear()
                combo.addItems(columns)
            elif isinstance(combo, QListWidget):
                combo.clear()
                for col in columns:
                    combo.addItem(col)
        
        # Update column-specific UIs
        if self.current_dataframe is not None:
            self.update_missing_value_ui()
            self.update_outlier_ui()
            self.update_conversion_ui()
            self.update_transform_ui()
            self.update_text_ui()
    
    # ---------------------------------------------------------------
    # Methods for duplicates tab
    # ---------------------------------------------------------------
    def find_duplicates(self):
        """Find duplicate rows based on selected columns"""
        if self.current_dataframe is None:
            return
        
        # Get selected columns
        selected_items = self.duplicate_columns_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one column")
            return
        
        selected_columns = [item.text() for item in selected_items]
        
        # Check for duplicates
        df = self.current_dataframe
        duplicates = df.duplicated(subset=selected_columns, keep=False)
        dup_count = duplicates.sum()
        
        if dup_count > 0:
            # Show duplicates in the preview
            self.preview_display.display_dataframe(df[duplicates].sort_values(by=selected_columns))
            
            # Update results
            pct = (dup_count / len(df)) * 100
            self.duplicates_result_label.setText(
                f"Found {dup_count} duplicate rows ({pct:.2f}%) based on columns: {', '.join(selected_columns)}")
            
            # Suggest action
            keep_option = self.duplicate_keep_combo.currentText()
            suggestion = f"\nRecommended action: Remove duplicates, keeping {keep_option.lower()}."
            self.duplicates_result_label.setText(self.duplicates_result_label.text() + suggestion)
        else:
            self.duplicates_result_label.setText(
                f"No duplicates found based on columns: {', '.join(selected_columns)}")
            
            # Reset preview
            self.preview_display.display_dataframe(df)
    
    def remove_duplicates(self):
        """Remove duplicate rows based on selected columns"""
        if self.current_dataframe is None:
            return
        
        # Get selected columns
        selected_items = self.duplicate_columns_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one column")
            return
        
        selected_columns = [item.text() for item in selected_items]
        
        # Get keep option
        keep_option = self.duplicate_keep_combo.currentText()
        if keep_option == "First occurrence":
            keep = "first"
        elif keep_option == "Last occurrence":
            keep = "last"
        else:  # "None (remove all)"
            keep = False
        
        # Remove duplicates
        df = self.current_dataframe
        original_count = len(df)
        
        # Create a copy for preview
        result_df = df.drop_duplicates(subset=selected_columns, keep=keep, ignore_index=self.ignore_index_check.isChecked())
        removed_count = original_count - len(result_df)
        
        if removed_count > 0:
            # Update preview
            self.preview_display.display_dataframe(result_df)
            self._preview_df = result_df
            
            # Update results
            pct = (removed_count / original_count) * 100
            self.duplicates_result_label.setText(
                f"Removed {removed_count} duplicate rows ({pct:.2f}%) based on columns: {', '.join(selected_columns)}")
            
            # Add to history
            self.add_to_history(f"Removed {removed_count} duplicate rows based on columns: {', '.join(selected_columns)}, "
                               f"keeping {keep_option.lower()}")
        else:
            self.duplicates_result_label.setText(
                f"No duplicates found based on columns: {', '.join(selected_columns)}")
    
    # ---------------------------------------------------------------
    # Methods for missing values tab
    # ---------------------------------------------------------------
    def update_missing_value_ui(self):
        """Update UI elements for missing value handling"""
        if self.current_dataframe is None:
            return
        
        # Update missing value column selection
        self.missing_column_combo.clear()
        self.missing_column_combo.addItems(self.current_dataframe.columns)
    
    def on_missing_method_changed(self, method):
        """Show/hide relevant inputs based on selected method"""
        show_constant = method == "Replace with constant"
        self.constant_value_label.setVisible(show_constant)
        self.constant_value_input.setVisible(show_constant)
    
    def find_missing_values(self):
        """Find missing values in the dataset"""
        if self.current_dataframe is None:
            return
        
        df = self.current_dataframe
        
        # Determine what to consider as missing
        consider_na = self.consider_na_check.isChecked()
        consider_empty = self.consider_empty_check.isChecked()
        consider_whitespace = self.consider_whitespace_check.isChecked()
        consider_custom = self.consider_custom_check.isChecked()
        
        # Create a mask for missing values
        missing_mask = pd.Series(False, index=range(len(df)))
        
        if consider_na:
            missing_mask = missing_mask | df.isna().any(axis=1)
        
        if consider_empty or consider_whitespace:
            for col in df.select_dtypes(include=['object']):
                if consider_empty:
                    missing_mask = missing_mask | (df[col] == '')
                if consider_whitespace:
                    missing_mask = missing_mask | df[col].str.isspace().fillna(False)
        
        if consider_custom and self.custom_missing_input.text():
            custom_values = [v.strip() for v in self.custom_missing_input.text().split(',')]
            for col in df.columns:
                for val in custom_values:
                    missing_mask = missing_mask | (df[col].astype(str) == val)
        
        # Count missing values
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            # Show rows with missing values
            self.preview_display.display_dataframe(df[missing_mask])
            
            # Update results
            missing_pct = (missing_count / len(df)) * 100
            
            # Get column-wise missing counts
            col_missing = {}
            for col in df.columns:
                col_mask = pd.Series(False, index=range(len(df)))
                
                # Check for NaN/None
                if consider_na:
                    col_mask = col_mask | df[col].isna()
                
                # Check for empty strings and whitespace
                if isinstance(df[col].dtype, pd.StringDtype) or df[col].dtype == 'object':
                    if consider_empty:
                        col_mask = col_mask | (df[col] == '')
                    if consider_whitespace:
                        col_mask = col_mask | df[col].str.isspace().fillna(False)
                
                # Check for custom values
                if consider_custom and self.custom_missing_input.text():
                    custom_values = [v.strip() for v in self.custom_missing_input.text().split(',')]
                    for val in custom_values:
                        col_mask = col_mask | (df[col].astype(str) == val)
                
                col_missing_count = col_mask.sum()
                if col_missing_count > 0:
                    col_missing[col] = col_missing_count
            
            # Create result text
            result = f"Found {missing_count} rows ({missing_pct:.2f}%) with missing values\n\n"
            result += "Missing values by column:\n"
            
            for col, count in col_missing.items():
                col_pct = (count / len(df)) * 100
                result += f"  {col}: {count} missing ({col_pct:.2f}%)\n"
            
            self.missing_result_label.setText(result)
        else:
            self.missing_result_label.setText("No missing values found based on selected criteria")
            self.preview_display.display_dataframe(df)
    
    def handle_missing_values(self):
        """Apply selected method to handle missing values"""
        if self.current_dataframe is None:
            return
        
        # Get selected column and method
        column = self.missing_column_combo.currentText()
        method = self.missing_method_combo.currentText()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a column")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Apply selected method
        if method == "Remove rows":
            original_count = len(df)
            df = df.dropna(subset=[column])
            removed_count = original_count - len(df)
            
            self.add_to_history(f"Removed {removed_count} rows with missing values in column '{column}'")
            self.missing_result_label.setText(f"Removed {removed_count} rows with missing values")
        
        elif method == "Replace with mean":
            if pd.api.types.is_numeric_dtype(df[column]):
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)
                self.add_to_history(f"Replaced missing values in '{column}' with mean ({mean_value:.4f})")
                self.missing_result_label.setText(f"Replaced missing values with mean: {mean_value:.4f}")
            else:
                QMessageBox.warning(self, "Warning", "Mean replacement only works with numeric columns")
                return
        
        elif method == "Replace with median":
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                self.add_to_history(f"Replaced missing values in '{column}' with median ({median_value:.4f})")
                self.missing_result_label.setText(f"Replaced missing values with median: {median_value:.4f}")
            else:
                QMessageBox.warning(self, "Warning", "Median replacement only works with numeric columns")
                return
        
        elif method == "Replace with mode":
            mode_value = df[column].mode().iloc[0]
            df[column] = df[column].fillna(mode_value)
            self.add_to_history(f"Replaced missing values in '{column}' with mode ({mode_value})")
            self.missing_result_label.setText(f"Replaced missing values with mode: {mode_value}")
        
        elif method == "Replace with constant":
            constant = self.constant_value_input.text()
            
            # Try to convert constant to appropriate type for the column
            try:
                if pd.api.types.is_numeric_dtype(df[column]):
                    constant = float(constant)
                elif pd.api.types.is_bool_dtype(df[column]):
                    constant = constant.lower() in ('true', 'yes', 'y', '1')
            except:
                pass
                
            df[column] = df[column].fillna(constant)
            self.add_to_history(f"Replaced missing values in '{column}' with constant value: {constant}")
            self.missing_result_label.setText(f"Replaced missing values with constant: {constant}")
        
        elif method == "Replace with interpolation":
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].interpolate()
                self.add_to_history(f"Filled missing values in '{column}' using interpolation")
                self.missing_result_label.setText(f"Filled missing values using interpolation")
            else:
                QMessageBox.warning(self, "Warning", "Interpolation only works with numeric columns")
                return
        
        elif method == "Replace with forward fill":
            df[column] = df[column].ffill()
            self.add_to_history(f"Filled missing values in '{column}' using forward fill")
            self.missing_result_label.setText(f"Filled missing values using forward fill")
        
        elif method == "Replace with backward fill":
            df[column] = df[column].bfill()
            self.add_to_history(f"Filled missing values in '{column}' using backward fill")
            self.missing_result_label.setText(f"Filled missing values using backward fill")
        
        # Update preview
        self._preview_df = df
        self.preview_display.display_dataframe(df)
    
    # ---------------------------------------------------------------
    # Methods for outliers tab
    # ---------------------------------------------------------------
    def update_outlier_ui(self):
        """Update UI elements for outlier handling"""
        if self.current_dataframe is None:
            return
        
        # Update outlier column selection with only numeric columns
        self.outlier_column_combo.clear()
        numeric_cols = self.current_dataframe.select_dtypes(include=['number']).columns
        self.outlier_column_combo.addItems(numeric_cols)
    
    def on_outlier_column_changed(self, column):
        """Update visualization when column changes"""
        if not column or self.current_dataframe is None:
            return
        
        # Create a histogram or box plot for the selected column
        try:
            df = self.current_dataframe
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                self.create_distribution_plot(column)
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def create_distribution_plot(self, column):
        """Create a distribution plot for the selected column"""
        df = self.current_dataframe
        
        # Create a figure with both histogram and box plot
        plt.figure(figsize=(8, 5))
        
        # Create a subplot for the histogram
        plt.subplot(2, 1, 1)
        plt.hist(df[column].dropna(), bins=30, alpha=0.7, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.grid(True, alpha=0.3)
        
        # Create a subplot for the box plot
        plt.subplot(2, 1, 2)
        plt.boxplot(df[column].dropna(), vert=False)
        plt.title('Box Plot (with potential outliers)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to QPixmap
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        # Create QPixmap from buffer
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        
        # Display in QLabel
        self.outlier_visualization.setPixmap(pixmap)
        self.outlier_visualization.setScaledContents(True)
    
    def on_outlier_method_changed(self, method):
        """Show/hide relevant inputs based on selected method"""
        # Hide all threshold inputs first
        self.z_threshold_label.setVisible(False)
        self.z_threshold_input.setVisible(False)
        self.iqr_multiplier_label.setVisible(False)
        self.iqr_multiplier_input.setVisible(False)
        self.percentile_label.setVisible(False)
        self.percentile_input.setVisible(False)
        
        # Show the relevant input
        if method == "Z-score":
            self.z_threshold_label.setVisible(True)
            self.z_threshold_input.setVisible(True)
        elif method == "IQR (Interquartile Range)":
            self.iqr_multiplier_label.setVisible(True)
            self.iqr_multiplier_input.setVisible(True)
        elif method == "Percentile":
            self.percentile_label.setVisible(True)
            self.percentile_input.setVisible(True)
        # Modified Z-score doesn't need additional inputs, uses same as Z-score
    
    def on_outlier_handling_changed(self, method):
        """Show/hide custom value input based on selected method"""
        show_custom = method == "Replace with custom value"
        self.outlier_custom_label.setVisible(show_custom)
        self.outlier_custom_input.setVisible(show_custom)
    
    def detect_outliers(self):
        """Detect outliers in the selected column using the chosen method"""
        if self.current_dataframe is None:
            return
        
        column = self.outlier_column_combo.currentText()
        method = self.outlier_method_combo.currentText()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a column")
            return
        
        df = self.current_dataframe
        
        # Skip missing values for detection
        data = df[column].dropna()
        
        # Detect outliers based on selected method
        outlier_mask = pd.Series(False, index=df.index)
        
        if method == "Z-score":
            threshold = self.z_threshold_input.value()
            z_scores = (data - data.mean()) / data.std()
            outlier_mask[data.index] = abs(z_scores) > threshold
            
            result_text = f"Z-score method with threshold {threshold}:\n"
            
        elif method == "Modified Z-score":
            threshold = self.z_threshold_input.value()
            median = data.median()
            mad = (data - median).abs().median() * 1.4826  # Consistent with normal distribution
            modified_z = 0.6745 * (data - median) / mad
            outlier_mask[data.index] = abs(modified_z) > threshold
            
            result_text = f"Modified Z-score method with threshold {threshold}:\n"
            
        elif method == "IQR (Interquartile Range)":
            multiplier = self.iqr_multiplier_input.value()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (multiplier * iqr)
            upper_bound = q3 + (multiplier * iqr)
            outlier_mask[data.index] = (data < lower_bound) | (data > upper_bound)
            
            result_text = f"IQR method with multiplier {multiplier}:\n"
            result_text += f"Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}\n"
            
        elif method == "Percentile":
            percentile = self.percentile_input.value()
            lower_percentile = (100 - percentile) / 2
            upper_percentile = 100 - lower_percentile
            lower_bound = data.quantile(lower_percentile / 100)
            upper_bound = data.quantile(upper_percentile / 100)
            outlier_mask[data.index] = (data < lower_bound) | (data > upper_bound)
            
            result_text = f"Percentile method using {percentile}% central range:\n"
            result_text += f"Lower bound ({lower_percentile:.1f}%): {lower_bound:.4f}, "
            result_text += f"Upper bound ({upper_percentile:.1f}%): {upper_bound:.4f}\n"
        
        # Count outliers
        outlier_count = outlier_mask.sum()
        outlier_pct = (outlier_count / len(df)) * 100
        
        result_text += f"Found {outlier_count} outliers ({outlier_pct:.2f}%) in column '{column}'"
        
        # Show outliers in preview
        if outlier_count > 0:
            self.preview_display.display_dataframe(df[outlier_mask])
            # Create outlier stats
            if outlier_count > 0:
                outlier_values = df.loc[outlier_mask, column]
                result_text += f"\n\nOutlier Statistics:"
                result_text += f"\n  Min: {outlier_values.min():.4f}"
                result_text += f"\n  Max: {outlier_values.max():.4f}"
                result_text += f"\n  Mean: {outlier_values.mean():.4f}"
        else:
            self.preview_display.display_dataframe(df)
        
        self.outlier_result_label.setText(result_text)
        
        # Store outlier information for later use
        self._outlier_info = {
            'column': column,
            'method': method,
            'mask': outlier_mask
        }
    
    def handle_outliers(self):
        """Apply selected method to handle detected outliers"""
        if self.current_dataframe is None or not hasattr(self, '_outlier_info'):
            QMessageBox.warning(self, "Warning", "Please detect outliers first")
            return
        
        # Get outlier information
        column = self._outlier_info['column']
        outlier_mask = self._outlier_info['mask']
        outlier_count = outlier_mask.sum()
        
        if outlier_count == 0:
            QMessageBox.information(self, "Information", "No outliers to handle")
            return
        
        # Get handling method
        method = self.outlier_handling_combo.currentText()
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Apply selected method
        if method == "Remove outliers":
            original_count = len(df)
            df = df[~outlier_mask].reset_index(drop=True)
            removed_count = original_count - len(df)
            
            self.add_to_history(f"Removed {removed_count} outliers from column '{column}'")
            result_text = f"Removed {removed_count} outliers from dataset"
            
        elif method == "Cap at threshold":
            # Get the appropriate bounds based on the detection method
            if self._outlier_info['method'] == "Z-score":
                threshold = self.z_threshold_input.value()
                mean = df[column].mean()
                std = df[column].std()
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                
            elif self._outlier_info['method'] == "Modified Z-score":
                threshold = self.z_threshold_input.value()
                median = df[column].median()
                mad = (df[column] - median).abs().median() * 1.4826
                lower_bound = median - (threshold * mad / 0.6745)
                upper_bound = median + (threshold * mad / 0.6745)
                
            elif self._outlier_info['method'] == "IQR (Interquartile Range)":
                multiplier = self.iqr_multiplier_input.value()
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (multiplier * iqr)
                upper_bound = q3 + (multiplier * iqr)
                
            elif self._outlier_info['method'] == "Percentile":
                percentile = self.percentile_input.value()
                lower_percentile = (100 - percentile) / 2
                upper_percentile = 100 - lower_percentile
                lower_bound = df[column].quantile(lower_percentile / 100)
                upper_bound = df[column].quantile(upper_percentile / 100)
            
            # Cap the values
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
            
            self.add_to_history(f"Capped {outlier_count} outliers in column '{column}' "
                               f"(lower: {lower_bound:.4f}, upper: {upper_bound:.4f})")
            
            result_text = f"Capped {outlier_count} outliers at bounds: "
            result_text += f"lower={lower_bound:.4f}, upper={upper_bound:.4f}"
            
        elif method == "Replace with mean":
            mean_value = df.loc[~outlier_mask, column].mean()
            df.loc[outlier_mask, column] = mean_value
            
            self.add_to_history(f"Replaced {outlier_count} outliers in column '{column}' with mean ({mean_value:.4f})")
            result_text = f"Replaced {outlier_count} outliers with mean: {mean_value:.4f}"
            
        elif method == "Replace with median":
            median_value = df.loc[~outlier_mask, column].median()
            df.loc[outlier_mask, column] = median_value
            
            self.add_to_history(f"Replaced {outlier_count} outliers in column '{column}' with median ({median_value:.4f})")
            result_text = f"Replaced {outlier_count} outliers with median: {median_value:.4f}"
            
        elif method == "Replace with custom value":
            try:
                custom_value = float(self.outlier_custom_input.text())
                df.loc[outlier_mask, column] = custom_value
                
                self.add_to_history(f"Replaced {outlier_count} outliers in column '{column}' with custom value ({custom_value})")
                result_text = f"Replaced {outlier_count} outliers with custom value: {custom_value}"
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid number for custom value")
                return
        
        # Update preview
        self._preview_df = df
        self.preview_display.display_dataframe(df)
        self.outlier_result_label.setText(result_text)
    
    # ---------------------------------------------------------------
    # Methods for type conversion tab
    # ---------------------------------------------------------------
    def update_conversion_ui(self):
        """Update UI elements for type conversion"""
        if self.current_dataframe is None:
            return
        
        # Update current type when column changes
        if self.conversion_column_combo.currentText():
            self.on_conversion_column_changed(self.conversion_column_combo.currentText())
    
    def on_conversion_column_changed(self, column):
        """Update current type label when column changes"""
        if not column or self.current_dataframe is None:
            return
        
        dtype = self.current_dataframe[column].dtype
        self.current_type_label.setText(str(dtype))
    
    def on_target_type_changed(self, target_type):
        """Show/hide format options based on target type"""
        # Hide all format options first
        self.date_format_label.setVisible(False)
        self.date_format_input.setVisible(False)
        self.bool_true_label.setVisible(False)
        self.bool_true_input.setVisible(False)
        self.bool_false_label.setVisible(False)
        self.bool_false_input.setVisible(False)
        
        # Show relevant options
        if "Date/Time" in target_type:
            self.date_format_label.setVisible(True)
            self.date_format_input.setVisible(True)
        elif "Boolean" in target_type:
            self.bool_true_label.setVisible(True)
            self.bool_true_input.setVisible(True)
            self.bool_false_label.setVisible(True)
            self.bool_false_input.setVisible(True)
    
    def apply_type_conversion(self):
        """Convert column to selected data type"""
        if self.current_dataframe is None:
            return
        
        column = self.conversion_column_combo.currentText()
        target_type = self.target_type_combo.currentText()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a column")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Get error handling option
            errors = 'coerce' if self.error_handling_combo.currentText().startswith('Coerce') else 'raise'
            
            # Perform conversion based on target type
            if "Integer" in target_type:
                df[column] = pd.to_numeric(df[column], errors=errors, downcast='integer')
                type_name = "integer"
                
            elif "Float" in target_type:
                df[column] = pd.to_numeric(df[column], errors=errors, downcast='float')
                type_name = "float"
                
            elif "String" in target_type:
                df[column] = df[column].astype(str)
                type_name = "string"
                
            elif "Boolean" in target_type:
                # Get true/false values
                true_values = [v.strip() for v in self.bool_true_input.text().split(',')]
                false_values = [v.strip() for v in self.bool_false_input.text().split(',')]
                
                # Create a temporary series with string values
                temp_series = df[column].astype(str).str.lower()
                
                # Replace with boolean values
                df[column] = pd.Series(False, index=df.index)
                for val in true_values:
                    df.loc[temp_series == val.lower(), column] = True
                
                type_name = "boolean"
                
            elif "Date/Time" in target_type:
                date_format = self.date_format_input.text()
                # If format is empty, assume it's a timestamp (in seconds, milliseconds, or nanoseconds)
                if not date_format:
                    # Try to convert numeric timestamps to datetime
                    # First ensure it's numeric
                    if pd.api.types.is_numeric_dtype(df[column]):
                        # Determine timestamp scale based on value magnitude
                        sample_val = df[column].iloc[0] if len(df) > 0 else 0
                        if sample_val > 1e18:  # nanoseconds (typical pandas timestamp)
                            df[column] = pd.to_datetime(df[column], unit='ns', errors=errors)
                        elif sample_val > 1e15:  # microseconds
                            df[column] = pd.to_datetime(df[column], unit='us', errors=errors)
                        elif sample_val > 1e12:  # milliseconds
                            df[column] = pd.to_datetime(df[column], unit='ms', errors=errors)
                        else:  # seconds
                            df[column] = pd.to_datetime(df[column], unit='s', errors=errors)
                    else:
                        # If not numeric, just try regular conversion
                        df[column] = pd.to_datetime(df[column], errors=errors)
                else:
                    # Use the specified format
                    df[column] = pd.to_datetime(df[column], format=date_format, errors=errors)
                type_name = "datetime"
                
            elif "Categorical" in target_type:
                df[column] = df[column].astype('category')
                type_name = "categorical"
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Update current type
            new_dtype = df[column].dtype
            self.current_type_label.setText(str(new_dtype))
            
            # Add to history
            self.add_to_history(f"Converted column '{column}' to {type_name} type")
            
            # Update result label
            self.conversion_result_label.setText(f"Successfully converted column '{column}' to {type_name} type ({new_dtype})")
            
        except Exception as e:
            QMessageBox.warning(self, "Conversion Error", str(e))
            self.conversion_result_label.setText(f"Error: {str(e)}")
    
    def auto_convert_numeric(self):
        """Automatically convert string columns to numeric where possible"""
        if self.current_dataframe is None:
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Track conversions
        conversions = []
        
        # Try to convert each column that's not already numeric
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                # Try to convert to numeric
                try:
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    
                    # Only convert if we don't lose too much data
                    non_null_before = df[column].notna().sum()
                    non_null_after = numeric_series.notna().sum()
                    
                    # If we kept at least 90% of the data and at least some values were converted successfully
                    if non_null_after / non_null_before >= 0.9 and non_null_after > 0:
                        # Determine if integer or float is more appropriate
                        if (numeric_series.dropna() % 1 == 0).all():
                            df[column] = numeric_series.astype('Int64')  # Nullable integer type
                            conversions.append(f"{column} (to integer)")
                        else:
                            df[column] = numeric_series.astype('float')
                            conversions.append(f"{column} (to float)")
                except:
                    continue
        
        if conversions:
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            self.add_to_history(f"Auto-converted {len(conversions)} columns to numeric types")
            
            # Update result label
            result_text = f"Successfully converted {len(conversions)} columns to numeric types:\n"
            result_text += "\n".join(conversions)
            self.conversion_result_label.setText(result_text)
        else:
            self.conversion_result_label.setText("No suitable columns found for automatic numeric conversion")
    
    def auto_convert_dates(self):
        """Automatically detect and convert date columns"""
        if self.current_dataframe is None:
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Track conversions
        conversions = []
        
        # Try to convert each column that's not already datetime
        for column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                # Try to convert to datetime
                try:
                    date_series = pd.to_datetime(df[column], errors='coerce')
                    
                    # Only convert if we don't lose too much data
                    non_null_before = df[column].notna().sum()
                    non_null_after = date_series.notna().sum()
                    
                    # If we kept at least 80% of the data and at least 10 values were converted successfully
                    if non_null_after / non_null_before >= 0.8 and non_null_after >= 10:
                        df[column] = date_series
                        conversions.append(column)
                except:
                    continue
        
        if conversions:
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            self.add_to_history(f"Auto-converted {len(conversions)} columns to datetime types")
            
            # Update result label
            result_text = f"Successfully converted {len(conversions)} columns to datetime types:\n"
            result_text += "\n".join(conversions)
            self.conversion_result_label.setText(result_text)
        else:
            self.conversion_result_label.setText("No suitable columns found for automatic date conversion")
    
    def optimize_data_types(self):
        """Optimize memory usage by downcasting numeric types and categorizing strings"""
        if self.current_dataframe is None:
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Memory usage before optimization
        mem_before = df.memory_usage(deep=True).sum()
        
        # Track optimizations
        numeric_optimized = []
        categorical_optimized = []
        
        # Optimize numeric columns
        for column in df.select_dtypes(include=['number']).columns:
            # Check if float
            if pd.api.types.is_float_dtype(df[column]):
                # Try to downcast
                df[column] = pd.to_numeric(df[column], downcast='float')
                numeric_optimized.append(f"{column} (float)")
            
            # Check if integer
            elif pd.api.types.is_integer_dtype(df[column]):
                # Try to downcast
                df[column] = pd.to_numeric(df[column], downcast='integer')
                numeric_optimized.append(f"{column} (integer)")
        
        # Optimize string columns with limited unique values
        for column in df.select_dtypes(include=['object']):
            # Check unique ratio
            unique_ratio = df[column].nunique() / len(df)
            
            # If column has less than 50% unique values and at least 10 rows, convert to category
            if unique_ratio < 0.5 and df[column].nunique() >= 2 and len(df) >= 10:
                df[column] = df[column].astype('category')
                categorical_optimized.append(column)
        
        # Memory usage after optimization
        mem_after = df.memory_usage(deep=True).sum()
        savings = (1 - mem_after / mem_before) * 100
        
        # Update preview
        self._preview_df = df
        self.preview_display.display_dataframe(df)
        
        # Add to history
        self.add_to_history(f"Optimized data types, saving {savings:.1f}% memory")
        
        # Update result label
        result_text = f"Memory usage optimized: {mem_before/1e6:.2f} MB → {mem_after/1e6:.2f} MB ({savings:.1f}% saved)\n\n"
        
        if numeric_optimized:
            result_text += f"Optimized {len(numeric_optimized)} numeric columns:\n"
            result_text += "\n".join(numeric_optimized) + "\n\n"
            
        if categorical_optimized:
            result_text += f"Converted {len(categorical_optimized)} columns to categorical type:\n"
            result_text += "\n".join(categorical_optimized)
            
        self.conversion_result_label.setText(result_text)
    
    # ---------------------------------------------------------------
    # Methods for column operations tab
    # ---------------------------------------------------------------
    def update_transform_ui(self):
        """Update UI elements for column transformations"""
        if self.current_dataframe is None:
            return
        
        # Adjust UI for selected column
        if self.transform_column_combo.currentText():
            self.on_transform_column_changed(self.transform_column_combo.currentText())
    
    def on_transform_column_changed(self, column):
        """Update UI based on column data type"""
        if not column or self.current_dataframe is None:
            return
        
        df = self.current_dataframe
        
        # Enable/disable transformations based on data type
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        
        # Remove non-applicable transformations from combo box
        self.transform_type_combo.clear()
        
        if is_numeric:
            self.transform_type_combo.addItems([
                "Z-score normalization", 
                "Min-max scaling", 
                "Log transformation", 
                "Square root", 
                "Absolute value",
                "Bin into categories"
            ])
        else:
            self.transform_type_combo.addItems([
                "One-hot encoding"
            ])
    
    def on_transform_type_changed(self, transform_type):
        """Show/hide options based on transformation type"""
        self.bin_count_label.setVisible(False)
        self.bin_count_spin.setVisible(False)
        
        if transform_type == "Bin into categories":
            self.bin_count_label.setVisible(True)
            self.bin_count_spin.setVisible(True)
            
        # Show new column name input if "Create new column" is selected
        show_newcol = self.transform_output_combo.currentText() == "Create new column"
        self.transform_newcol_label.setVisible(show_newcol)
        self.transform_newcol_input.setVisible(show_newcol)
        
        # Set default new column name based on transformation
        if show_newcol and self.transform_column_combo.currentText():
            column = self.transform_column_combo.currentText()
            if transform_type == "Z-score normalization":
                self.transform_newcol_input.setText(f"{column}_zscore")
            elif transform_type == "Min-max scaling":
                self.transform_newcol_input.setText(f"{column}_scaled")
            elif transform_type == "Log transformation":
                self.transform_newcol_input.setText(f"{column}_log")
            elif transform_type == "Square root":
                self.transform_newcol_input.setText(f"{column}_sqrt")
            elif transform_type == "Absolute value":
                self.transform_newcol_input.setText(f"{column}_abs")
            elif transform_type == "Bin into categories":
                self.transform_newcol_input.setText(f"{column}_binned")
            elif transform_type == "One-hot encoding":
                self.transform_newcol_input.setText(f"{column}_onehot")
    
    def on_transform_output_changed(self, output_type):
        """Show/hide new column name input"""
        show_newcol = output_type == "Create new column"
        self.transform_newcol_label.setVisible(show_newcol)
        self.transform_newcol_input.setVisible(show_newcol)
        
        # Update the new column name if visible
        if show_newcol:
            self.on_transform_type_changed(self.transform_type_combo.currentText())
    
    def rename_column(self):
        """Rename the selected column"""
        if self.current_dataframe is None:
            return
        
        old_name = self.rename_column_combo.currentText()
        new_name = self.new_column_name.text()
        
        if not old_name or not new_name:
            QMessageBox.warning(self, "Warning", "Please select a column and provide a new name")
            return
        
        if new_name in self._preview_df.columns:
            QMessageBox.warning(self, "Warning", f"Column '{new_name}' already exists")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Rename column
        df = df.rename(columns={old_name: new_name})
        
        # Update preview
        self._preview_df = df
        self.preview_display.display_dataframe(df)
        
        # Add to history
        self.add_to_history(f"Renamed column '{old_name}' to '{new_name}'")
        
        # Update result label
        self.column_ops_result_label.setText(f"Renamed column '{old_name}' to '{new_name}'")
        
        # Update column combos
        self.update_column_selections()
    
    def apply_column_transformation(self):
        """Apply the selected transformation to the column"""
        if self.current_dataframe is None:
            return
        
        column = self.transform_column_combo.currentText()
        transform_type = self.transform_type_combo.currentText()
        output_type = self.transform_output_combo.currentText()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a column")
            return
        
        # Check if we need a new column name
        if output_type == "Create new column":
            new_column = self.transform_newcol_input.text()
            if not new_column:
                QMessageBox.warning(self, "Warning", "Please provide a name for the new column")
                return
        else:
            new_column = column
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Apply transformation
            if transform_type == "Z-score normalization":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Z-score normalization requires numeric data")
                    return
                
                # Calculate z-score
                mean = df[column].mean()
                std = df[column].std()
                transformed = (df[column] - mean) / std
                
                transform_desc = f"Z-score normalization (mean={mean:.4f}, std={std:.4f})"
                
            elif transform_type == "Min-max scaling":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Min-max scaling requires numeric data")
                    return
                
                # Calculate min-max scaling
                min_val = df[column].min()
                max_val = df[column].max()
                transformed = (df[column] - min_val) / (max_val - min_val)
                
                transform_desc = f"min-max scaling to range [0,1] (min={min_val:.4f}, max={max_val:.4f})"
                
            elif transform_type == "Log transformation":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Log transformation requires numeric data")
                    return
                
                # Check for non-positive values
                if (df[column] <= 0).any():
                    min_val = df[column].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        transformed = np.log(df[column] + offset)
                        transform_desc = f"log transformation with offset +{offset:.4f}"
                    else:
                        transformed = np.log(df[column])
                        transform_desc = "log transformation"
                else:
                    transformed = np.log(df[column])
                    transform_desc = "log transformation"
                
            elif transform_type == "Square root":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Square root transformation requires numeric data")
                    return
                
                # Check for negative values
                if (df[column] < 0).any():
                    min_val = df[column].min()
                    offset = abs(min_val)
                    transformed = np.sqrt(df[column] + offset)
                    transform_desc = f"square root transformation with offset +{offset:.4f}"
                else:
                    transformed = np.sqrt(df[column])
                    transform_desc = "square root transformation"
                
            elif transform_type == "Absolute value":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Absolute value requires numeric data")
                    return
                
                transformed = np.abs(df[column])
                transform_desc = "absolute value"
                
            elif transform_type == "Bin into categories":
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(df[column]):
                    QMessageBox.warning(self, "Warning", "Binning requires numeric data")
                    return
                
                # Get number of bins
                bins = self.bin_count_spin.value()
                
                # Create bins
                transformed = pd.cut(df[column], bins=bins, labels=[f"Bin_{i+1}" for i in range(bins)])
                
                transform_desc = f"binned into {bins} categories"
                
            elif transform_type == "One-hot encoding":
                # One-hot encode the column
                dummies = pd.get_dummies(df[column], prefix=column)
                
                # For one-hot encoding, we need to add the new columns
                for dummy_col in dummies.columns:
                    if dummy_col not in df.columns:
                        df[dummy_col] = dummies[dummy_col]
                
                # Set result to the original column to skip the normal assignment below
                transformed = df[column]
                
                # Skip further processing for one-hot encoding
                self._preview_df = df
                self.preview_display.display_dataframe(df)
                
                self.add_to_history(f"One-hot encoded column '{column}' into {len(dummies.columns)} columns")
                self.column_ops_result_label.setText(f"One-hot encoded column '{column}' into {len(dummies.columns)} columns")
                
                return
            
            # Assign transformed values
            if output_type == "Replace original column":
                df[column] = transformed
                result_text = f"Applied {transform_desc} to column '{column}'"
                history_text = f"Applied {transform_desc} to column '{column}'"
            else:
                df[new_column] = transformed
                result_text = f"Created new column '{new_column}' with {transform_desc} of '{column}'"
                history_text = f"Created new column '{new_column}' with {transform_desc} of '{column}'"
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            self.add_to_history(history_text)
            
            # Update result label
            self.column_ops_result_label.setText(result_text)
            
            # Update column combos if we added a new column
            if output_type == "Create new column":
                self.update_column_selections()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Transformation failed: {str(e)}")
            self.column_ops_result_label.setText(f"Error: {str(e)}")
    
    def create_derived_column(self):
        """Create a new column using the provided expression"""
        if self.current_dataframe is None:
            return
        
        expression = self.derived_column_expr.toPlainText()
        new_column = self.derived_column_name.text()
        
        if not expression or not new_column:
            QMessageBox.warning(self, "Warning", "Please provide an expression and a name for the new column")
            return
        
        if new_column in self._preview_df.columns:
            QMessageBox.warning(self, "Warning", f"Column '{new_column}' already exists")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Set up the namespace for evaluation
            namespace = {
                'df': df,
                'pd': pd,
                'np': np
            }
            
            # Evaluate the expression
            result = eval(expression, namespace)
            
            # Add the new column
            df[new_column] = result
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            self.add_to_history(f"Created derived column '{new_column}' using expression")
            
            # Update result label
            self.column_ops_result_label.setText(f"Created derived column '{new_column}'")
            
            # Update column combos
            self.update_column_selections()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Expression evaluation failed: {str(e)}")
            self.column_ops_result_label.setText(f"Error: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods for text cleaning tab
    # ---------------------------------------------------------------
    def update_text_ui(self):
        """Update UI elements for text cleaning"""
        if self.current_dataframe is None:
            return
        
        # Update text column combo with string columns
        self.text_column_combo.clear()
        string_cols = self.current_dataframe.select_dtypes(include=['object', 'string']).columns
        self.text_column_combo.addItems(string_cols)
    
    def on_text_column_changed(self, column):
        """Update UI when text column changes"""
        # No special updates needed at this time
        pass
    
    def on_text_output_changed(self, output_type):
        """Show/hide new column name input"""
        show_newcol = output_type == "Create new column"
        self.text_newcol_label.setVisible(show_newcol)
        self.text_newcol_input.setVisible(show_newcol)
        
        # Set default new column name if visible
        if show_newcol and self.text_column_combo.currentText():
            column = self.text_column_combo.currentText()
            self.text_newcol_input.setText(f"{column}_clean")
    
    def on_predefined_pattern_changed(self, pattern):
        """Pre-fill the custom pattern fields based on the selected predefined pattern"""
        # This will be used when applying standardization patterns
        pass
    
    def apply_text_cleaning(self):
        """Apply text cleaning operations to the selected column"""
        if self.current_dataframe is None:
            return
        
        column = self.text_column_combo.currentText()
        output_type = self.text_output_combo.currentText()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a text column")
            return
        
        # Check if we need a new column name
        if output_type == "Create new column":
            new_column = self.text_newcol_input.text()
            if not new_column:
                QMessageBox.warning(self, "Warning", "Please provide a name for the new column")
                return
            if new_column in self._preview_df.columns:
                QMessageBox.warning(self, "Warning", f"Column '{new_column}' already exists")
                return
        else:
            new_column = column
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Get the text series
            text_series = df[column].astype(str)
            
            # Apply selected cleaning operations
            operations = []
            
            # Case normalization
            case_option = self.case_combo.currentText()
            if case_option == "Lowercase":
                text_series = text_series.str.lower()
                operations.append("lowercase")
            elif case_option == "Uppercase":
                text_series = text_series.str.upper()
                operations.append("uppercase")
            elif case_option == "Title Case":
                text_series = text_series.str.title()
                operations.append("title case")
            
            # Whitespace handling
            if self.strip_whitespace_check.isChecked():
                text_series = text_series.str.strip()
                operations.append("stripped whitespace")
            
            if self.normalize_whitespace_check.isChecked():
                text_series = text_series.str.replace(r'\s+', ' ', regex=True)
                operations.append("normalized whitespace")
            
            # Special character handling
            if self.remove_punctuation_check.isChecked():
                text_series = text_series.str.replace(r'[^\w\s]', '', regex=True)
                operations.append("removed punctuation")
            
            if self.remove_numbers_check.isChecked():
                text_series = text_series.str.replace(r'\d', '', regex=True)
                operations.append("removed numbers")
            
            if self.remove_special_chars_check.isChecked():
                text_series = text_series.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                operations.append("removed special characters")
            
            # Custom character handling
            if self.custom_chars_check.isChecked() and self.custom_chars_input.text():
                chars = re.escape(self.custom_chars_input.text())
                text_series = text_series.str.replace(f'[{chars}]', '', regex=True)
                operations.append(f"removed custom characters '{self.custom_chars_input.text()}'")
            
            # Find and replace
            if self.find_replace_check.isChecked() and self.find_pattern_input.text():
                find_pattern = self.find_pattern_input.text()
                replace_with = self.replace_pattern_input.text()
                text_series = text_series.str.replace(find_pattern, replace_with, regex=True)
                operations.append(f"replaced '{find_pattern}' with '{replace_with}'")
            
            # Assign the cleaned text
            df[new_column] = text_series
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            if operations:
                history_text = f"Cleaned text in '{column}'"
                if output_type == "Create new column":
                    history_text += f" to new column '{new_column}'"
                history_text += f" ({', '.join(operations)})"
                self.add_to_history(history_text)
            
            # Update result label
            if operations:
                result_text = f"Applied text cleaning to '{column}'"
                if output_type == "Create new column":
                    result_text += f" and saved to '{new_column}'"
                result_text += ":"
                for op in operations:
                    result_text += "\n- " + op
                self.text_result_label.setText(result_text)
            else:
                self.text_result_label.setText("No cleaning operations were selected")
            
            # Update column combos if we added a new column
            if output_type == "Create new column":
                self.update_column_selections()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Text cleaning failed: {str(e)}")
            self.text_result_label.setText(f"Error: {str(e)}")
    
    def apply_text_standardization(self):
        """Apply standardization to text values based on predefined patterns"""
        if self.current_dataframe is None:
            return
        
        column = self.text_column_combo.currentText()
        pattern = self.predefined_patterns.currentText()
        output_type = self.text_output_combo.currentText()
        
        if not column or pattern == "Select pattern...":
            QMessageBox.warning(self, "Warning", "Please select a column and a pattern")
            return
        
        # Check if we need a new column name
        if output_type == "Create new column":
            new_column = self.text_newcol_input.text()
            if not new_column:
                QMessageBox.warning(self, "Warning", "Please provide a name for the new column")
                return
            if new_column in self._preview_df.columns:
                QMessageBox.warning(self, "Warning", f"Column '{new_column}' already exists")
                return
        else:
            new_column = column
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Get the text series
            text_series = df[column].astype(str)
            
            # Apply standardization based on selected pattern
            if pattern == "Phone numbers":
                # Remove all non-numeric characters and format as (XXX) XXX-XXXX
                clean_series = text_series.str.replace(r'[^\d]', '', regex=True)
                # Extract last 10 digits for US numbers
                clean_series = clean_series.apply(lambda x: x[-10:] if len(x) >= 10 else x)
                # Format as (XXX) XXX-XXXX if 10 digits
                clean_series = clean_series.apply(
                    lambda x: f"({x[0:3]}) {x[3:6]}-{x[6:10]}" if len(x) == 10 else x)
                
                standardization_desc = "standardized phone numbers"
                
            elif pattern == "Email addresses":
                # Convert to lowercase
                clean_series = text_series.str.lower()
                # Remove spaces
                clean_series = clean_series.str.replace(r'\s', '', regex=True)
                
                standardization_desc = "standardized email addresses"
                
            elif pattern == "URLs":
                # Ensure http:// or https:// prefix
                def standardize_url(url):
                    url = url.strip().lower()
                    if url and not url.startswith(('http://', 'https://')):
                        return f"http://{url}"
                    return url
                
                clean_series = text_series.apply(standardize_url)
                
                standardization_desc = "standardized URLs"
                
            elif pattern == "Dates":
                # Try to parse dates and standardize format to YYYY-MM-DD
                def standardize_date(date_str):
                    date_str = date_str.strip()
                    if not date_str:
                        return date_str
                    try:
                        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
                    except:
                        return date_str
                
                clean_series = text_series.apply(standardize_date)
                
                standardization_desc = "standardized dates to YYYY-MM-DD format"
                
            elif pattern == "Currency values":
                # Extract numeric part and standardize format
                def standardize_currency(value):
                    # Extract digits, decimal point, and negative sign
                    digits = re.sub(r'[^\d.-]', '', value)
                    try:
                        # Parse as float and format with 2 decimal places
                        amount = float(digits)
                        return f"${amount:.2f}"
                    except:
                        return value
                
                clean_series = text_series.apply(standardize_currency)
                
                standardization_desc = "standardized currency values"
                
            elif pattern == "US States":
                # Map state names to standard two-letter codes
                state_dict = {
                    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
                    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
                    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
                    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
                    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
                    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
                    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
                    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
                    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
                    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
                    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
                    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
                    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"
                }
                
                # Also add entries for abbreviations to standardize case
                for abbrev in list(state_dict.values()):
                    state_dict[abbrev.lower()] = abbrev
                
                def standardize_state(state):
                    state = state.strip().lower()
                    return state_dict.get(state, state)
                
                clean_series = text_series.apply(standardize_state)
                
                standardization_desc = "standardized US state names to two-letter codes"
                
            elif pattern == "Countries":
                # Standardize country names (common variations)
                country_dict = {
                    "usa": "United States", "us": "United States", "u.s.": "United States",
                    "united states of america": "United States", "u.s.a.": "United States",
                    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
                    "england": "United Kingdom",  # simplification
                    "uae": "United Arab Emirates", "u.a.e.": "United Arab Emirates",
                }
                
                def standardize_country(country):
                    country = country.strip().lower()
                    return country_dict.get(country, country.title())
                
                clean_series = text_series.apply(standardize_country)
                
                standardization_desc = "standardized country names"
                
            elif pattern == "Yes/No variations":
                # Map various yes/no responses to standard Y/N
                yes_variations = ["yes", "y", "yeah", "yep", "yup", "true", "t", "1", "positive"]
                no_variations = ["no", "n", "nope", "nah", "false", "f", "0", "negative"]
                
                def standardize_yes_no(value):
                    value = value.strip().lower()
                    if value in yes_variations:
                        return "Yes"
                    elif value in no_variations:
                        return "No"
                    else:
                        return value
                
                clean_series = text_series.apply(standardize_yes_no)
                
                standardization_desc = "standardized Yes/No variations"
            
            # Assign the standardized text
            df[new_column] = clean_series
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            history_text = f"Applied text standardization to '{column}'"
            if output_type == "Create new column":
                history_text += f" to new column '{new_column}'"
            history_text += f" ({standardization_desc})"
            self.add_to_history(history_text)
            
            # Update result label
            result_text = f"Applied text standardization ({pattern}) to '{column}'"
            if output_type == "Create new column":
                result_text += f" and saved to '{new_column}'"
            self.text_result_label.setText(result_text)
            
            # Update column combos if we added a new column
            if output_type == "Create new column":
                self.update_column_selections()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Text standardization failed: {str(e)}")
            self.text_result_label.setText(f"Error: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods for AI assistance tab
    # ---------------------------------------------------------------
    @asyncSlot()
    async def determine_applicable_processes(self):
        """
        Use LLM to analyze the current dataset and determine which cleaning processes
        would be most applicable based on data characteristics, then generate a detailed
        execution plan.
        """
        if self.current_dataframe is None:
            self.ai_response_text.setPlainText("Please select a dataset first.")
            return
            
        # Show processing status
        self.ai_response_text.setPlainText("Analyzing dataset and generating cleaning plan...")
        
        # Gather dataset information for the prompt - with proper conversion to Python native types
        try:
            # Convert shape tuple to regular Python list for JSON serialization
            shape = [int(x) for x in self.current_dataframe.shape]
            
            # Convert column names to regular Python list
            columns = [str(col) for col in self.current_dataframe.columns.tolist()]
            
            # Convert dtypes to strings (they already are strings, but ensure they're regular Python strings)
            dtypes = {str(col): str(dtype) for col, dtype in self.current_dataframe.dtypes.items()}
            
            # Convert missing values to regular Python dict with int values
            missing_values = {str(col): int(val) for col, val in self.current_dataframe.isna().sum().to_dict().items()}
            
            # For sample data, convert each value to a Python native type
            sample_data = {}
            sample_df = self.current_dataframe.head(5)
            for col in sample_df.columns:
                # Handle different data types appropriately
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    # Convert numeric types to Python int/float
                    sample_data[str(col)] = [float(x) if pd.notnull(x) else None for x in sample_df[col].tolist()]
                elif pd.api.types.is_datetime64_dtype(sample_df[col]):
                    # Convert datetime to ISO string format
                    sample_data[str(col)] = [x.isoformat() if pd.notnull(x) else None for x in sample_df[col].tolist()]
                else:
                    # For other types (like strings), convert to regular Python strings
                    sample_data[str(col)] = [str(x) if pd.notnull(x) else None for x in sample_df[col].tolist()]
            
            # Get duplicate count (convert to Python int)
            duplicated_rows = int(self.current_dataframe.duplicated().sum())
            
            # Build the data_info dict with all Python native types
            data_info = {
                "dataset_name": str(self.current_name),
                "shape": shape,
                "columns": columns,
                "dtypes": dtypes,
                "missing_values": missing_values,
                "sample_data": sample_data,
                "duplicated_rows": duplicated_rows,
            }
            
            # Define available cleaning operations with their capabilities and constraints
            available_operations = {
                "duplicates": {
                    "description": "Identify and remove duplicate rows",
                    "options": ["remove_all", "keep_first", "keep_last"],
                    "constraints": ["Operates on entire rows, not individual columns"]
                },
                "missing_values": {
                    "description": "Handle missing values in the dataset",
                    "options": ["drop_rows", "drop_columns", "fill_mean", "fill_median", "fill_mode", "fill_constant", "fill_interpolate"],
                    "constraints": ["Numerical methods only apply to numeric columns", "Categorical methods require sufficient non-null values"]
                },
                "outliers": {
                    "description": "Detect and handle statistical outliers",
                    "options": ["z_score", "iqr", "isolation_forest", "drop", "cap", "replace_mean", "replace_median"],
                    "constraints": ["Most methods only work on numeric columns", "Statistical methods assume normal distribution"]
                },
                "type_conversion": {
                    "description": "Convert column data types",
                    "options": ["to_numeric", "to_datetime", "to_category", "to_string", "auto_numeric", "auto_dates", "optimize_types"],
                    "constraints": ["Conversion must be compatible with data values", "May cause data loss if incompatible"]
                },
                "column_operations": {
                    "description": "Transform or create columns",
                    "options": ["rename", "drop", "normalize", "standardize", "log_transform", "bin", "one_hot_encode", "create_derived"],
                    "constraints": ["Some transformations only work on numeric data", "One-hot encoding increases dimensionality"]
                },
                "text_cleaning": {
                    "description": "Clean and standardize text data",
                    "options": ["lowercase", "remove_punctuation", "remove_digits", "remove_whitespace", "standardize_pattern"],
                    "constraints": ["Only applies to text/string columns", "May alter original meaning if not carefully applied"]
                }
            }
            
            # Define the expected output format with examples
            expected_output_format = {
                "summary": "Brief overview of the dataset quality and issues",
                "cleaning_plan": [
                    {
                        "operation": "duplicates",  # Must be one of the available_operations keys
                        "method": "remove_all",     # Must be one of the options for this operation
                        "parameters": {             # Parameters specific to the operation
                            "subset": ["column1", "column2"]  # Optional, depends on operation
                        },
                        "justification": "Reason for this recommendation"
                    }
                ]
            }
            
            # Prepare prompt with all information
            prompt = f"""
            You are a data cleaning assistant. Analyze the provided dataset information and recommend the most appropriate 
            cleaning steps to prepare this data for analysis.

            Dataset Information:
            {json.dumps(data_info, indent=2)}

            Available Cleaning Operations:
            {json.dumps(available_operations, indent=2)}

            Your response MUST be valid JSON following this exact structure:
            {json.dumps(expected_output_format, indent=2)}

            Important requirements:
            1. The "operation" field must be exactly one of: {", ".join(available_operations.keys())}
            2. The "method" field must be an option listed for that operation
            3. The "parameters" must be valid for the operation and method
            4. Steps should be ordered by priority
            5. Each step must include a clear justification
            6. Your response must be valid JSON with no additional text before or after
            """
            
            try:
                # Call the LLM API with the prepared prompt
                result = await call_llm_async_json(prompt)
                
                # Check if result is already a dictionary (parsed JSON)
                if isinstance(result, dict):
                    cleaning_plan = result
                else:
                    # Only try to parse as string if it's not already a dictionary
                    try:
                        # Try to extract JSON if the response has additional text
                        json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
                        if json_match:
                            result = json_match.group(1)
                        
                        # Try to find JSON directly
                        json_start = result.find('{')
                        json_end = result.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            result = result[json_start:json_end]
                        
                        # Parse the JSON
                        cleaning_plan = json.loads(result)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
                
                # Validate the cleaning plan structure
                if "cleaning_plan" not in cleaning_plan or not isinstance(cleaning_plan["cleaning_plan"], list):
                    raise ValueError("Missing or invalid 'cleaning_plan' in response")
                
                # Validate each step in the plan
                for step_index, step in enumerate(cleaning_plan["cleaning_plan"]):
                    # Validate operation
                    if "operation" not in step or step["operation"] not in available_operations:
                        raise ValueError(f"Step {step_index+1} has invalid operation: {step.get('operation')}")
                    
                    # Validate method
                    if "method" not in step or step["method"] not in available_operations[step["operation"]]["options"]:
                        raise ValueError(f"Step {step_index+1} has invalid method: {step.get('method')}")
                    
                    # Add default parameters if missing
                    if "parameters" not in step:
                        step["parameters"] = {}
                
                # Store the validated cleaning plan
                self.current_cleaning_plan = cleaning_plan
                self.current_step = 0
                self.total_steps = len(cleaning_plan.get("cleaning_plan", []))
                
                # Display the plan in a readable format
                plan_display = f"# Dataset Analysis Summary\n\n{cleaning_plan.get('summary', 'No summary provided.')}\n\n"
                plan_display += f"# Recommended Cleaning Plan ({self.total_steps} steps)\n\n"
                
                for i, step in enumerate(cleaning_plan.get("cleaning_plan", [])):
                    op_name = step.get("operation", "unknown").title()
                    method = step.get("method", "unknown")
                    parameters = step.get("parameters", {})
                    
                    plan_display += f"## Step {i+1}: {op_name} - {method}\n"
                    
                    # Format parameters for display
                    if parameters:
                        plan_display += "Parameters:\n"
                        
                        # Add proper type checking for parameters
                        if isinstance(parameters, dict):
                            for param, value in parameters.items():
                                if isinstance(value, list):
                                    if len(value) > 5:  # Truncate long lists
                                        value_display = f"[{', '.join(str(v) for v in value[:5])}...]"
                                    else:
                                        value_display = f"[{', '.join(str(v) for v in value)}]"
                                else:
                                    value_display = str(value)
                                plan_display += f"- {param}: {value_display}\n"
                        elif isinstance(parameters, list):
                            # If parameters is a list, display each item
                            for j, item in enumerate(parameters):
                                if isinstance(item, dict):
                                    # If list items are dictionaries
                                    for param, value in item.items():
                                        plan_display += f"- {param}: {value}\n"
                                else:
                                    # Simple list item
                                    plan_display += f"- Item {j+1}: {item}\n"
                        else:
                            # For any other type
                            plan_display += f"- {str(parameters)}\n"
                        
                        plan_display += "\n"
                    else:
                        plan_display += "\n"
                    
                    if "justification" in step:
                        plan_display += f"_{step['justification']}_\n\n"
                    else:
                        plan_display += "\n"
                
                self.ai_response_text.setPlainText(plan_display)
                
                # Highlight recommended tabs
                self._highlight_recommended_tabs(cleaning_plan.get("cleaning_plan", []))
                
                # Enable the execute plan button if we have steps
                self.execute_step_button.setEnabled(self.total_steps > 0)
                self.progress_label.setText(f"Ready to execute: 0/{self.total_steps} steps completed")
                
            except Exception as e:
                error_message = f"Error processing cleaning plan: {str(e)}"
                
                # Add detailed debugging info
                error_message += "\n\nDebugging information:"
                if 'cleaning_plan' in locals():
                    error_message += f"\n- cleaning_plan type: {type(cleaning_plan)}"
                    if isinstance(cleaning_plan, dict) and 'cleaning_plan' in cleaning_plan:
                        error_message += f"\n- cleaning_plan['cleaning_plan'] type: {type(cleaning_plan['cleaning_plan'])}"
                        
                        for i, step in enumerate(cleaning_plan['cleaning_plan']):
                            error_message += f"\n- Step {i+1} type: {type(step)}"
                            if isinstance(step, dict) and 'parameters' in step:
                                error_message += f"\n  - parameters type: {type(step['parameters'])}"
                                error_message += f"\n  - parameters value: {repr(step['parameters'])}"
                
                if not isinstance(result, dict):
                    error_message += f"\n\nRaw response:\n{result}"
                self.ai_response_text.setPlainText(error_message)
                self.current_cleaning_plan = None
        
        except Exception as e:
            self.ai_response_text.setPlainText(f"Error preparing dataset information: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    
    def execute_current_step(self):
        if not self.current_cleaning_plan or not self.current_cleaning_plan.get("cleaning_plan"):
            self.progress_label.setText("No cleaning plan available")
            return
        
        # Get the current step
        step = self.current_cleaning_plan["cleaning_plan"][self.current_step]
        
        # Add logging for debugging
        print(f"Executing step {self.current_step+1}: {step.get('operation')} - {step.get('method')}")
        print(f"Parameters: {step.get('parameters', {})}")
        
        # Update step indicator in AI tab
        self.ai_step_indicator.setText(f"Processing Step {self.current_step+1}/{len(self.current_cleaning_plan['cleaning_plan'])}: {step.get('operation')} - {step.get('method')}")
        
        # Check if we're in AI tab mode (staying in AI tab)
        ai_tab_mode = hasattr(self, 'stay_in_ai_tab_checkbox') and self.stay_in_ai_tab_checkbox.isChecked()
        
        # Store original tab index if in AI tab mode
        original_tab_index = self.cleaning_tabs.currentIndex() if ai_tab_mode else None
        
        # Navigate to the appropriate tab (if not in AI tab mode)
        tab_mapping = {
            "duplicates": 0,
            "missing_values": 1,
            "outliers": 2,
            "type_conversion": 3,
            "column_operations": 4,
            "text_cleaning": 5
        }
        
        # Check if the operation exists in our tab mapping
        if step.get("operation") not in tab_mapping:
            error_message = f"Unknown operation: {step.get('operation')}"
            self.progress_label.setText(f"Error in step {self.current_step+1}: {error_message}")
            return
            
        # Set the active tab (if not in AI tab mode)
        tab_index = tab_mapping[step["operation"]]
        if not ai_tab_mode:
            self.cleaning_tabs.setCurrentIndex(tab_index)
        else:
            # Temporarily switch to required tab without user seeing it
            self.cleaning_tabs.blockSignals(True)
            self.cleaning_tabs.setCurrentIndex(tab_index)
        
        try:
            # Apply the parameters from the step to the UI
            success = self.apply_step_parameters(step)
            if not success:
                if ai_tab_mode:
                    self.cleaning_tabs.setCurrentIndex(original_tab_index)
                    self.cleaning_tabs.blockSignals(False)
                return
            
            # Execute the operation based on the step
            success = self.execute_step_operation(step)
            if not success:
                if ai_tab_mode:
                    self.cleaning_tabs.setCurrentIndex(original_tab_index)
                    self.cleaning_tabs.blockSignals(False)
                return
            
            # Add to cleaning history
            operation_str = f"{step.get('operation')} - {step.get('method')}"
            param_str = ", ".join([f"{k}: {v}" for k, v in step.get('parameters', {}).items()])
            history_entry = f"{operation_str} [{param_str}]"
            self.add_to_history(history_entry)
            
            # Return to AI tab if in AI tab mode
            if ai_tab_mode:
                self.cleaning_tabs.setCurrentIndex(original_tab_index)
                self.cleaning_tabs.blockSignals(False)
            
            # Update progress
            self.current_step += 1
            if self.current_step < len(self.current_cleaning_plan["cleaning_plan"]):
                next_step = self.current_cleaning_plan["cleaning_plan"][self.current_step]
                self.progress_label.setText(f"Step {self.current_step}/{len(self.current_cleaning_plan['cleaning_plan'])} completed. Next: {next_step.get('operation')} - {next_step.get('method')}")
                # Update next step button state
                if hasattr(self, 'next_step_button'):
                    self.next_step_button.setEnabled(True)
            else:
                self.progress_label.setText("All steps completed!")
                if hasattr(self, 'next_step_button'):
                    self.next_step_button.setEnabled(False)
            
            # Update previous step button state
            if hasattr(self, 'prev_step_button'):
                self.prev_step_button.setEnabled(self.current_step > 0)
                
        except Exception as e:
            error_message = str(e)
            self.progress_label.setText(f"Error in step {self.current_step+1}: {error_message}")
            if ai_tab_mode:
                self.cleaning_tabs.setCurrentIndex(original_tab_index)
                self.cleaning_tabs.blockSignals(False)
            return False

        return True
    
    def apply_step_parameters(self, step):
        """Apply parameters from a cleaning step to the UI"""
        try:
            operation = step.get("operation")
            method = step.get("method")
            parameters = step.get("parameters", {})
            
            # Ensure parameters is a dictionary
            if not isinstance(parameters, dict):
                print(f"Warning: parameters is not a dictionary: {type(parameters)}")
                if isinstance(parameters, list):
                    # For list parameters, create a synthetic dictionary
                    parameters = {"items": parameters}
                else:
                    # For other types, convert to string
                    parameters = {"value": str(parameters)}
            
            # Switch to the appropriate tab
            tab_mapping = {
                "duplicates": 0,
                "missing_values": 1,
                "outliers": 2,
                "type_conversion": 3,
                "column_operations": 4,
                "text_cleaning": 5
            }
            
            if operation in tab_mapping:
                self.cleaning_tabs.setCurrentIndex(tab_mapping[operation])
            else:
                print(f"Unknown operation: {operation}")
                return False
                
            # Apply parameters based on the operation
            if operation == "duplicates":
                # Set method in the combo box
                if method == "remove_all":
                    self.duplicate_keep_combo.setCurrentText("None")
                elif method == "keep_first":
                    self.duplicate_keep_combo.setCurrentText("First")
                elif method == "keep_last":
                    self.duplicate_keep_combo.setCurrentText("Last")
                    
                # Set subset if provided
                if "subset" in parameters and isinstance(parameters["subset"], list):
                    for i in range(self.duplicate_subset_list.count()):
                        item = self.duplicate_subset_list.item(i)
                        if item.text() in parameters["subset"]:
                            item.setCheckState(Qt.Checked)
                        else:
                            item.setCheckState(Qt.Unchecked)
                        
            elif operation == "missing_values":
                # Set missing values method
                if method == "drop_rows":
                    self.missing_method_combo.setCurrentText("Drop Rows")
                elif method == "drop_columns":
                    self.missing_method_combo.setCurrentText("Drop Columns")
                elif method == "fill_mean":
                    self.missing_method_combo.setCurrentText("Fill with Mean")
                elif method == "fill_median":
                    self.missing_method_combo.setCurrentText("Fill with Median")
                elif method == "fill_mode":
                    self.missing_method_combo.setCurrentText("Fill with Mode")
                elif method == "fill_constant":
                    self.missing_method_combo.setCurrentText("Fill with Constant")
                    if "value" in parameters:
                        self.missing_fill_value.setText(str(parameters["value"]))
                elif method == "fill_interpolate":
                    self.missing_method_combo.setCurrentText("Interpolate")
                    
                # Set columns if provided
                if "columns" in parameters and isinstance(parameters["columns"], list):
                    for i in range(self.missing_columns_list.count()):
                        item = self.missing_columns_list.item(i)
                        if item.text() in parameters["columns"]:
                            item.setCheckState(Qt.Checked)
                        else:
                            item.setCheckState(Qt.Unchecked)
                
            elif operation == "outliers":
                # Set outlier detection method
                if method == "z_score":
                    self.outlier_method_combo.setCurrentText("Z-Score")
                    if "threshold" in parameters:
                        self.outlier_z_threshold.setValue(float(parameters["threshold"]))
                elif method == "iqr":
                    self.outlier_method_combo.setCurrentText("IQR")
                    if "threshold" in parameters:
                        self.outlier_iqr_factor.setValue(float(parameters["threshold"]))
                elif method == "isolation_forest":
                    self.outlier_method_combo.setCurrentText("Isolation Forest")
                    if "contamination" in parameters:
                        self.outlier_if_contamination.setValue(float(parameters["contamination"]))
                
                # Set handling method
                if "strategy" in parameters:
                    strategy = parameters["strategy"]
                    if strategy == "drop":
                        self.outlier_handling_combo.setCurrentText("Drop Rows")
                    elif strategy == "cap":
                        self.outlier_handling_combo.setCurrentText("Cap Values")
                    elif strategy == "replace_mean":
                        self.outlier_handling_combo.setCurrentText("Replace with Mean")
                    elif strategy == "replace_median":
                        self.outlier_handling_combo.setCurrentText("Replace with Median")
                
                # Set columns if provided
                if "columns" in parameters and isinstance(parameters["columns"], list):
                    # First select the first column to trigger column change event
                    if parameters["columns"]:
                        first_col = parameters["columns"][0]
                        index = self.outlier_column_combo.findText(first_col)
                        if index >= 0:
                            self.outlier_column_combo.setCurrentIndex(index)
                            
                    # If there are multiple columns, store them for later handling
                    if len(parameters["columns"]) > 1:
                        self.pending_outlier_columns = parameters["columns"][1:]
                    else:
                        self.pending_outlier_columns = []
                    
            elif operation == "type_conversion":
                # Set conversion method based on the method parameter
                if method == "to_numeric":
                    self.target_type_combo.setCurrentText("Integer (int)")
                elif method == "to_datetime":
                    self.target_type_combo.setCurrentText("Date/Time")
                    # Check if we need to clear the format for timestamp conversion
                    if "is_timestamp" in parameters and parameters["is_timestamp"]:
                        self.date_format_input.setText("")
                elif method == "to_category":
                    self.target_type_combo.setCurrentText("Categorical")
                elif method == "to_string":
                    self.target_type_combo.setCurrentText("String (text)")
                elif method == "auto_numeric":
                    # Auto convert is a button click, will be handled in execute_step_operation
                    pass
                elif method == "auto_dates":
                    # Auto convert is a button click, will be handled in execute_step_operation
                    pass
                elif method == "optimize_types":
                    # Optimize is a button click, will be handled in execute_step_operation
                    pass
                
                # Set columns if provided
                if "columns" in parameters and isinstance(parameters["columns"], list):
                    if parameters["columns"]:
                        first_col = parameters["columns"][0]
                        index = self.conversion_column_combo.findText(first_col)
                        if index >= 0:
                            self.conversion_column_combo.setCurrentIndex(index)
                            
                    # If there are multiple columns, store them for later handling
                    if len(parameters["columns"]) > 1:
                        self.pending_conversion_columns = parameters["columns"][1:]
                    else:
                        self.pending_conversion_columns = []
                        
                # Set format if provided (for date conversion)
                if "format" in parameters:
                    self.date_format_input.setText(parameters["format"])
                    
            elif operation == "column_operations":
                # Set transformation method based on the method parameter
                if method == "rename":
                    self.transform_type_combo.setCurrentText("Rename")
                    if "new_name" in parameters:
                        self.transform_new_name.setText(parameters["new_name"])
                elif method == "drop":
                    self.transform_type_combo.setCurrentText("Drop")
                elif method == "normalize":
                    self.transform_type_combo.setCurrentText("Normalize")
                elif method == "standardize":
                    self.transform_type_combo.setCurrentText("Standardize")
                elif method == "log_transform":
                    self.transform_type_combo.setCurrentText("Log Transform")
                elif method == "bin":
                    self.transform_type_combo.setCurrentText("Bin Values")
                    if "bins" in parameters:
                        self.transform_bins.setValue(int(parameters["bins"]))
                elif method == "one_hot_encode":
                    self.transform_type_combo.setCurrentText("One-Hot Encode")
                elif method == "create_derived":
                    self.transform_type_combo.setCurrentText("Create Derived Column")
                    if "expression" in parameters:
                        self.transform_expression.setText(parameters["expression"])
                    if "new_column" in parameters:
                        self.transform_derived_name.setText(parameters["new_column"])
                        
                # Set output type if provided
                if "output_type" in parameters:
                    output_type = parameters["output_type"]
                    index = self.transform_output_combo.findText(output_type, Qt.MatchContains)
                    if index >= 0:
                        self.transform_output_combo.setCurrentIndex(index)
                        
                # Set columns if provided
                if "columns" in parameters and isinstance(parameters["columns"], list):
                    if parameters["columns"]:
                        first_col = parameters["columns"][0]
                        index = self.transform_column_combo.findText(first_col)
                        if index >= 0:
                            self.transform_column_combo.setCurrentIndex(index)
                            
                    # If there are multiple columns, store them for later handling
                    if len(parameters["columns"]) > 1:
                        self.pending_transform_columns = parameters["columns"][1:]
                    else:
                        self.pending_transform_columns = []
                        
            elif operation == "text_cleaning":
                # Set text cleaning method based on the method parameter
                if method == "lowercase":
                    self.text_function_combo.setCurrentText("Convert to Lowercase")
                elif method == "remove_punctuation":
                    self.text_function_combo.setCurrentText("Remove Punctuation")
                elif method == "remove_digits":
                    self.text_function_combo.setCurrentText("Remove Digits")
                elif method == "remove_whitespace":
                    self.text_function_combo.setCurrentText("Remove Extra Whitespace")
                elif method == "standardize_pattern":
                    self.text_function_combo.setCurrentText("Standardize Pattern")
                    if "pattern" in parameters:
                        pattern = parameters["pattern"]
                        index = self.text_pattern_combo.findText(pattern, Qt.MatchContains)
                        if index >= 0:
                            self.text_pattern_combo.setCurrentIndex(index)
                            
                # Set output option
                if "output_type" in parameters:
                    output_type = parameters["output_type"]
                    if output_type == "inplace":
                        self.text_output_combo.setCurrentText("Replace Original")
                    elif output_type == "new_column":
                        self.text_output_combo.setCurrentText("Create New Column")
                        if "new_column" in parameters:
                            self.text_new_column_name.setText(parameters["new_column"])
                            
                # Set columns if provided
                if "columns" in parameters and isinstance(parameters["columns"], list):
                    if parameters["columns"]:
                        first_col = parameters["columns"][0]
                        index = self.text_column_combo.findText(first_col)
                        if index >= 0:
                            self.text_column_combo.setCurrentIndex(index)
                            
                    # If there are multiple columns, store them for later handling
                    if len(parameters["columns"]) > 1:
                        self.pending_text_columns = parameters["columns"][1:]
                    else:
                        self.pending_text_columns = []
                        
                # Set custom pattern if provided
                if "custom_pattern" in parameters:
                    self.text_custom_pattern.setText(parameters["custom_pattern"])
                    
                # Set replacement if provided
                if "replacement" in parameters:
                    self.text_replacement.setText(parameters["replacement"])
                    
            # Return success
            return True
            
        except Exception as e:
            error_message = f"Error applying parameters: {str(e)}"
            self.progress_label.setText(error_message)
            print(error_message)
            import traceback
            traceback.print_exc()
            return False
    
    def execute_step_operation(self, step):
        """Execute an operation based on a cleaning step"""
        try:
            operation = step.get("operation")
            method = step.get("method")
            
            # Get parameters and ensure it's a dictionary
            parameters = step.get("parameters", {})
            if not isinstance(parameters, dict):
                print(f"Warning: parameters is not a dictionary: {type(parameters)}")
                if isinstance(parameters, list):
                    # For list parameters, create a synthetic dictionary
                    parameters = {"items": parameters}
                else:
                    # For other types, convert to string
                    parameters = {"value": str(parameters)}
            
            # Execute the appropriate operation
            if operation == "duplicates":
                self.find_duplicates()  # First identify duplicates
                self.remove_duplicates()  # Then remove them
                return True
                
            elif operation == "missing_values":
                self.find_missing_values()  # First identify missing values
                self.handle_missing_values()  # Then handle them
                return True
                
            elif operation == "outliers":
                self.detect_outliers()  # First detect outliers
                self.handle_outliers()  # Then handle them
                return True
                
            elif operation == "type_conversion":
                if method in ["auto_numeric", "to_numeric"]:
                    self.auto_convert_numeric()
                elif method in ["auto_dates", "to_datetime"]:
                    # Check if we're dealing with timestamp values
                    column = None
                    if "columns" in parameters and isinstance(parameters["columns"], list) and parameters["columns"]:
                        column = parameters["columns"][0]
                    
                    if column and self.current_dataframe is not None:
                        # Check if column contains large numeric values (likely timestamps)
                        if pd.api.types.is_numeric_dtype(self.current_dataframe[column]):
                            sample_val = self.current_dataframe[column].iloc[0] if len(self.current_dataframe) > 0 else 0
                            if sample_val > 1e10:  # Likely a timestamp value
                                # Clear the date format to trigger timestamp handling
                                self.date_format_input.setText("")
                
                    # Now apply the conversion
                    self.apply_type_conversion()
                elif method == "optimize_types":
                    self.optimize_data_types()
                else:
                    self.apply_type_conversion()
                return True
                
            elif operation == "column_operations":
                if method == "rename":
                    self.rename_column()
                else:
                    self.apply_column_transformation()
                return True
                
            elif operation == "text_cleaning":
                if "standardize" in method:
                    self.apply_text_standardization()
                else:
                    self.apply_text_cleaning()
                return True
                
            else:
                print(f"Unknown operation: {operation}")
                return False
                
        except Exception as e:
            error_message = f"Error executing operation: {str(e)}"
            self.progress_label.setText(error_message)
            print(error_message)
            import traceback
            traceback.print_exc()
            return False
    
    def create_ai_assistance_tab(self):
        """Create the AI assistance tab with improved step-by-step execution"""
        ai_tab = QWidget()
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("This tab uses AI to analyze your dataset and create a custom cleaning plan.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Create area for AI response
        self.ai_response_text = QTextEdit()
        self.ai_response_text.setReadOnly(True)
        self.ai_response_text.setMinimumHeight(300)
        layout.addWidget(self.ai_response_text)
        
        # Progress indicator
        progress_widget = QWidget()
        progress_layout = QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_label = QLabel("No cleaning plan generated yet")
        progress_layout.addWidget(self.progress_label)
        
        # Execution controls
        button_layout = QHBoxLayout()
        
        generate_button = QPushButton("Analyze Dataset & Generate Plan")
        generate_button.setIcon(load_bootstrap_icon("robot"))
        generate_button.clicked.connect(self.determine_applicable_processes)
        button_layout.addWidget(generate_button)
        
        self.execute_step_button = QPushButton("Execute Next Step")
        self.execute_step_button.setIcon(load_bootstrap_icon("play"))
        self.execute_step_button.clicked.connect(self.execute_current_step)
        self.execute_step_button.setEnabled(False)
        button_layout.addWidget(self.execute_step_button)
        
        progress_layout.addLayout(button_layout)
        layout.addWidget(progress_widget)
        
        # Store the current cleaning plan and step tracking
        self.current_cleaning_plan = None
        self.current_step = 0
        self.total_steps = 0
        
        # Add step navigation section
        step_nav_group = QGroupBox("Step Navigation")
        step_nav_layout = QVBoxLayout()
        
        # Step indicator
        self.ai_step_indicator = QLabel("No steps in progress")
        step_nav_layout.addWidget(self.ai_step_indicator)
        
        # Stay in AI tab checkbox
        self.stay_in_ai_tab_checkbox = QCheckBox("Stay in AI tab while executing steps")
        self.stay_in_ai_tab_checkbox.setChecked(True)
        step_nav_layout.addWidget(self.stay_in_ai_tab_checkbox)
        
        # Navigation buttons
        nav_buttons_layout = QHBoxLayout()
        self.prev_step_button = QPushButton("Previous Step")
        self.prev_step_button.setIcon(load_bootstrap_icon("arrow-left"))
        self.prev_step_button.setEnabled(False)
        self.prev_step_button.clicked.connect(self.go_to_previous_step)
        
        self.next_step_button = QPushButton("Next Step")
        self.next_step_button.setIcon(load_bootstrap_icon("arrow-right"))
        self.next_step_button.setEnabled(False)
        self.next_step_button.clicked.connect(self.execute_current_step)
        
        self.apply_all_button = QPushButton("Apply All Steps")
        self.apply_all_button.setIcon(load_bootstrap_icon("lightning"))
        self.apply_all_button.clicked.connect(self.apply_all_steps)
        
        nav_buttons_layout.addWidget(self.prev_step_button)
        nav_buttons_layout.addWidget(self.next_step_button)
        nav_buttons_layout.addWidget(self.apply_all_button)
        
        step_nav_layout.addLayout(nav_buttons_layout)
        step_nav_group.setLayout(step_nav_layout)
        
        # Add to main layout
        layout.addWidget(step_nav_group)
        
        ai_tab.setLayout(layout)
        return ai_tab
    
    # ---------------------------------------------------------------
    # Methods for cleaning history tab
    # ---------------------------------------------------------------
    def add_to_history(self, operation):
        """Add an operation to the cleaning history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cleaning_history.append(f"[{timestamp}] {operation}")
        self.update_history()
    
    def update_history(self):
        """Update the history display"""
        if not self.cleaning_history:
            self.history_text.setPlainText("No cleaning operations performed yet.")
        else:
            self.history_text.setPlainText("\n".join(self.cleaning_history))
    
    def export_cleaning_report(self):
        """Export the cleaning history to a text file"""
        if not self.cleaning_history:
            QMessageBox.information(self, "Information", "No cleaning operations to export")
            return
        
        # Get file name for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cleaning Report", f"{self.current_name}_cleaning_report.txt", "Text Files (*.txt)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "w") as f:
                f.write(f"Data Cleaning Report for: {self.current_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Cleaning Operations:\n")
                for operation in self.cleaning_history:
                    f.write(f"{operation}\n")
                
                # Add dataset summary
                f.write("\n\nDataset Summary:\n")
                rows, cols = self._preview_df.shape
                f.write(f"Rows: {rows}, Columns: {cols}\n")
                f.write("Column Types:\n")
                for col, dtype in self._preview_df.dtypes.items():
                    f.write(f"  {col}: {dtype}\n")
                
                # Add missing values summary
                f.write("\nMissing Values:\n")
                for col, count in self._preview_df.isna().sum().items():
                    if count > 0:
                        pct = (count / rows) * 100
                        f.write(f"  {col}: {count} ({pct:.1f}%)\n")
            
            QMessageBox.information(self, "Success", f"Cleaning report saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")
    
    def reset_dataset(self):
        """Reset the preview dataset to the original"""
        if self.current_dataframe is None:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Reset", 
            "Are you sure you want to reset all cleaning operations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._preview_df = self.current_dataframe.copy()
            self.preview_display.display_dataframe(self._preview_df)
            self.cleaning_history = []
            self.update_history()
            self.status_bar.showMessage("Dataset reset to original state")
    
    # ---------------------------------------------------------------
    # Save the cleaned dataset
    # ---------------------------------------------------------------
    def save_cleaned_dataset(self):
        """Save the cleaned dataset"""
        new_name = self.save_name_input.text()
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the cleaned dataset")
            return
        if self._preview_df is None:
            QMessageBox.warning(self, "Error", "No dataset to save")
            return
        
        # Add the cleaned dataset as a source
        self.add_source(new_name, self._preview_df)
        
        # Add the dataset to the studies manager if available
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            main_window.studies_manager.add_dataset_to_active_study(new_name, self._preview_df)
        
        # Add final cleaning operation to history
        self.add_to_history(f"Saved cleaned dataset as '{new_name}'")
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.status_bar.showMessage(f"Cleaned dataset saved as '{new_name}'")
    
    def _highlight_recommended_tabs(self, recommendations):
        """Highlight tabs that are recommended based on the AI analysis"""
        try:
            # Reset all tab colors first
            for i in range(self.cleaning_tabs.count()):
                self.cleaning_tabs.tabBar().setTabTextColor(i, QColor("black"))
                
            tab_mapping = {
                "duplicates": 0,
                "missing_values": 1,
                "outliers": 2,
                "type_conversion": 3,
                "column_operations": 4,
                "text_cleaning": 5
            }
            
            # Ensure recommendations is iterable (list or similar)
            if not isinstance(recommendations, (list, tuple)):
                # If it's a dictionary, try to extract operations
                if isinstance(recommendations, dict):
                    if "cleaning_plan" in recommendations and isinstance(recommendations["cleaning_plan"], list):
                        recommendations = recommendations["cleaning_plan"]
                    else:
                        # Try to get operations directly from the dict
                        recommendations = [{"operation": op} for op in recommendations.keys()]
                else:
                    # If it's neither a list nor a dict, return
                    return
            
            # Highlight tabs for each recommended operation
            for rec in recommendations:
                # Check if rec is a dictionary with 'operation' key
                if isinstance(rec, dict) and "operation" in rec:
                    operation = rec["operation"]
                    if operation in tab_mapping:
                        tab_index = tab_mapping[operation]
                        self.cleaning_tabs.tabBar().setTabTextColor(tab_index, QColor("blue"))
                # If rec is a string (operation name directly)
                elif isinstance(rec, str) and rec in tab_mapping:
                    tab_index = tab_mapping[rec]
                    self.cleaning_tabs.tabBar().setTabTextColor(tab_index, QColor("blue"))
                    
        except Exception as e:
            print(f"Error highlighting recommended tabs: {str(e)}")
    
    def apply_ai_recommendations(self):
        """Apply AI recommendations to the dataset (simplified implementation)"""
        QMessageBox.information(self, "Information", 
                              "This feature would extract and apply the code snippets from the AI recommendations.\n\n"
                              "For safety, implementation requires human review first. Please copy and apply any "
                              "recommendations manually from the appropriate tabs.")
    
    def go_to_previous_step(self):
        """Navigate to the previous step in the cleaning plan"""
        if self.current_step > 0:
            self.current_step -= 1
            step = self.current_cleaning_plan["cleaning_plan"][self.current_step]
            self.ai_step_indicator.setText(f"Ready for Step {self.current_step+1}/{len(self.current_cleaning_plan['cleaning_plan'])}: {step.get('operation')} - {step.get('method')}")
            self.progress_label.setText(f"Moved to step {self.current_step+1}")
            
            # Update button states
            self.prev_step_button.setEnabled(self.current_step > 0)
            self.next_step_button.setEnabled(True)

    def apply_all_steps(self):
        """Apply all remaining steps in the cleaning plan"""
        if not self.current_cleaning_plan or not self.current_cleaning_plan.get("cleaning_plan"):
            self.progress_label.setText("No cleaning plan available")
            return
            
        total_steps = len(self.current_cleaning_plan["cleaning_plan"])
        start_step = self.current_step
        
        self.progress_label.setText(f"Applying all remaining steps ({start_step+1}-{total_steps})...")
        
        # Apply each step one by one
        success = True
        while self.current_step < total_steps and success:
            success = self.execute_current_step()
            
        if success:
            self.progress_label.setText(f"Successfully applied all steps!")
        else:
            self.progress_label.setText(f"Error occurred at step {self.current_step+1}")


# ---------------------------------------------------------------------
# Simple class to hold source connection details (deprecated)
# ---------------------------------------------------------------------
class SourceConnection:
    def __init__(self, source_type, connection_params, name):
        self.source_type = source_type
        self.connection_params = connection_params
        self.name = name
