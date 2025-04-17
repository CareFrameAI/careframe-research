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
    QTabWidget, QStatusBar, QDoubleSpinBox, QCheckBox, QFileDialog, QSlider, QInputDialog
)
from PyQt6.QtGui import QIcon, QFont
import re
import asyncio
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
# Main Data Filtering Widget
# ---------------------------------------------------------------------
class DataFilteringWidget(QWidget):
    # Signal emitted when a source (or filtered dataset) is selected
    source_selected = pyqtSignal(str, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Filtering")
        
        # Internal storage for data sources and datasets
        self.dataframes = {}
        self.current_name = ""
        self.current_dataframe = None
        self._preview_df = None
        self.slicing_history = []  # Track slicing operations for reporting
        
        # Build the UI
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Main splitter dividing left (dataset selection) and right (slicing operations)
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
        refresh_button.setIcon(load_bootstrap_icon("arrow-repeat"))
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
        # Right section: Slicing operations and preview
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
        
        # Slicing operations tabs
        self.slicing_tabs = QTabWidget()
        
        # Tab 1: Row Filtering
        row_filtering_tab = self.create_row_filtering_tab()
        self.slicing_tabs.addTab(row_filtering_tab, "Row Filtering")
        
        # Tab 2: Column Management
        column_management_tab = self.create_column_management_tab()
        self.slicing_tabs.addTab(column_management_tab, "Column Management")
        
        # Tab 3: Dataset Slicing
        dataset_slicing_tab = self.create_dataset_slicing_tab()
        self.slicing_tabs.addTab(dataset_slicing_tab, "Dataset Slicing")
        
        # Tab 4: Study Criteria
        criteria_tab = self.create_study_criteria_tab()
        self.slicing_tabs.addTab(criteria_tab, "Study Criteria")
        
        # Tab 5: Ordinal Encoding
        ordinal_encoding_tab = self.create_ordinal_encoding_tab()
        self.slicing_tabs.addTab(ordinal_encoding_tab, "Ordinal Encoding")
        
        # Tab 6: Slicing History
        history_tab = self.create_history_tab()
        self.slicing_tabs.addTab(history_tab, "Slicing History")
        
        right_layout.addWidget(self.slicing_tabs)
        
        # Preview and save section
        preview_group = QGroupBox("Preview & Save")
        preview_layout = QVBoxLayout(preview_group)
        
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("Transformation Preview"))
        preview_header.addStretch()
        preview_header.addWidget(QLabel("Save As:"))
        self.save_name_input = QLineEdit()
        self.save_name_input.setPlaceholderText("Enter name for sliced dataset")
        preview_header.addWidget(self.save_name_input)
        save_button = QPushButton("Save")
        save_button.setIcon(load_bootstrap_icon("save"))
        save_button.clicked.connect(self.save_sliced_dataset)
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
    # Create tab for row filtering
    # ---------------------------------------------------------------
    def create_row_filtering_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Filter rows based on conditions or random sampling.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Condition-based filtering
        condition_group = QGroupBox("Filter by Condition")
        condition_layout = QGridLayout(condition_group)
        
        condition_layout.addWidget(QLabel("Column:"), 0, 0)
        self.filter_column_combo = QComboBox()
        self.filter_column_combo.currentTextChanged.connect(self.on_filter_column_changed)
        condition_layout.addWidget(self.filter_column_combo, 0, 1, 1, 3)
        
        condition_layout.addWidget(QLabel("Condition:"), 1, 0)
        self.filter_condition_combo = QComboBox()
        self.filter_condition_combo.addItems([
            "Equal to", 
            "Not equal to", 
            "Greater than", 
            "Less than", 
            "Greater than or equal to", 
            "Less than or equal to", 
            "Contains", 
            "Does not contain", 
            "Starts with", 
            "Ends with", 
            "Is missing", 
            "Is not missing"
        ])
        condition_layout.addWidget(self.filter_condition_combo, 1, 1, 1, 3)
        
        condition_layout.addWidget(QLabel("Value:"), 2, 0)
        self.filter_value_input = QLineEdit()
        condition_layout.addWidget(self.filter_value_input, 2, 1, 1, 3)
        
        filter_button = QPushButton("Apply Filter")
        filter_button.setIcon(load_bootstrap_icon("funnel"))
        filter_button.clicked.connect(self.apply_row_filter)
        condition_layout.addWidget(filter_button, 3, 3)
        
        layout.addWidget(condition_group)
        
        # Sampling
        sampling_group = QGroupBox("Sample Rows")
        sampling_layout = QGridLayout(sampling_group)
        
        sampling_layout.addWidget(QLabel("Sampling method:"), 0, 0)
        self.sampling_method_combo = QComboBox()
        self.sampling_method_combo.addItems([
            "Random sample (fixed number)",
            "Random sample (percentage)",
            "Systematic sample",
            "Stratified sample"
        ])
        self.sampling_method_combo.currentTextChanged.connect(self.on_sampling_method_changed)
        sampling_layout.addWidget(self.sampling_method_combo, 0, 1, 1, 2)
        
        self.sample_n_label = QLabel("Number of rows:")
        sampling_layout.addWidget(self.sample_n_label, 1, 0)
        self.sample_n_spin = QSpinBox()
        self.sample_n_spin.setRange(1, 1000000)
        self.sample_n_spin.setValue(100)
        sampling_layout.addWidget(self.sample_n_spin, 1, 1, 1, 2)
        
        self.sample_frac_label = QLabel("Percentage:")
        sampling_layout.addWidget(self.sample_frac_label, 1, 0)
        self.sample_frac_spin = QDoubleSpinBox()
        self.sample_frac_spin.setRange(0.1, 100.0)
        self.sample_frac_spin.setValue(10.0)
        self.sample_frac_spin.setSuffix("%")
        sampling_layout.addWidget(self.sample_frac_spin, 1, 1, 1, 2)
        
        self.stratify_label = QLabel("Stratify by column:")
        sampling_layout.addWidget(self.stratify_label, 2, 0)
        self.stratify_column_combo = QComboBox()
        sampling_layout.addWidget(self.stratify_column_combo, 2, 1, 1, 2)
        
        self.random_seed_check = QCheckBox("Use random seed for reproducibility")
        self.random_seed_check.setChecked(True)
        sampling_layout.addWidget(self.random_seed_check, 3, 0, 1, 3)
        
        sample_button = QPushButton("Apply Sampling")
        sample_button.setIcon(load_bootstrap_icon("clipboard-data"))
        sample_button.clicked.connect(self.apply_sampling)
        sampling_layout.addWidget(sample_button, 4, 2)
        
        layout.addWidget(sampling_group)
        
        # Complex filtering
        complex_group = QGroupBox("Advanced Filtering")
        complex_layout = QVBoxLayout(complex_group)
        
        complex_layout.addWidget(QLabel("Custom filter expression (pandas syntax):"))
        self.complex_filter_expr = QPlainTextEdit()
        self.complex_filter_expr.setPlaceholderText("Examples:\n"
                                                   "df['Age'] > 18 & df['Sex'] == 'M'  # Males over 18\n"
                                                   "df['Value'].between(10, 100)  # Values between 10 and 100\n"
                                                   "df['Name'].str.contains('Smith')  # Names containing 'Smith'")
        self.complex_filter_expr.setMaximumHeight(80)
        complex_layout.addWidget(self.complex_filter_expr)
        
        complex_button = QPushButton("Apply Custom Filter")
        complex_button.setIcon(load_bootstrap_icon("code-slash"))
        complex_button.clicked.connect(self.apply_complex_filter)
        complex_layout.addWidget(complex_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(complex_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.filter_result_label = QLabel("No filtering applied yet")
        results_layout.addWidget(self.filter_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        # Initial state
        self.sample_n_label.setVisible(True)
        self.sample_n_spin.setVisible(True)
        self.sample_frac_label.setVisible(False)
        self.sample_frac_spin.setVisible(False)
        self.stratify_label.setVisible(False)
        self.stratify_column_combo.setVisible(False)
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for column management
    # ---------------------------------------------------------------
    def create_column_management_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Manage dataset columns: select, drop, or reorder columns.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Column selection
        selection_group = QGroupBox("Column Selection")
        selection_layout = QVBoxLayout(selection_group)
        
        selection_layout.addWidget(QLabel("Select columns to keep in the dataset:"))
        
        # List widget for column selection
        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        selection_layout.addWidget(self.columns_list)
        
        # Buttons for column selection
        buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.setIcon(load_bootstrap_icon("check-all"))
        select_all_button.clicked.connect(self.select_all_columns)
        buttons_layout.addWidget(select_all_button)
        
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.setIcon(load_bootstrap_icon("x-square"))
        deselect_all_button.clicked.connect(self.deselect_all_columns)
        buttons_layout.addWidget(deselect_all_button)
        
        invert_selection_button = QPushButton("Invert Selection")
        invert_selection_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        invert_selection_button.clicked.connect(self.invert_column_selection)
        buttons_layout.addWidget(invert_selection_button)
        
        selection_layout.addLayout(buttons_layout)
        
        # Apply column selection button
        apply_selection_button = QPushButton("Keep Selected Columns")
        apply_selection_button.setIcon(load_bootstrap_icon("check2-circle"))
        apply_selection_button.clicked.connect(self.keep_selected_columns)
        selection_layout.addWidget(apply_selection_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(selection_group)
        
        # Drop columns by pattern
        pattern_group = QGroupBox("Drop Columns by Pattern")
        pattern_layout = QVBoxLayout(pattern_group)
        
        pattern_layout.addWidget(QLabel("Drop columns matching pattern:"))
        
        pattern_input_layout = QHBoxLayout()
        self.column_pattern_input = QLineEdit()
        self.column_pattern_input.setPlaceholderText("e.g., temp_, _id, .*unused.*")
        pattern_input_layout.addWidget(self.column_pattern_input)
        
        self.pattern_is_regex_check = QCheckBox("Use regex")
        self.pattern_is_regex_check.setChecked(True)
        pattern_input_layout.addWidget(self.pattern_is_regex_check)
        
        pattern_layout.addLayout(pattern_input_layout)
        
        drop_pattern_button = QPushButton("Drop Matching Columns")
        drop_pattern_button.setIcon(load_bootstrap_icon("trash"))
        drop_pattern_button.clicked.connect(self.drop_columns_by_pattern)
        pattern_layout.addWidget(drop_pattern_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(pattern_group)
        
        # Column reordering
        reorder_group = QGroupBox("Column Reordering")
        reorder_layout = QVBoxLayout(reorder_group)
        
        reorder_layout.addWidget(QLabel("Move columns to beginning:"))
        
        # List widget for columns to move to beginning
        self.reorder_columns_list = QListWidget()
        self.reorder_columns_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        reorder_layout.addWidget(self.reorder_columns_list)
        
        reorder_button = QPushButton("Move Selected to Beginning")
        reorder_button.setIcon(load_bootstrap_icon("arrow-left"))
        reorder_button.clicked.connect(self.reorder_columns)
        reorder_layout.addWidget(reorder_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(reorder_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.column_ops_result_label = QLabel("No column operations performed yet")
        results_layout.addWidget(self.column_ops_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for dataset slicing
    # ---------------------------------------------------------------
    def create_dataset_slicing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Slice the dataset by selecting specific rows, ranges, or indices.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Row range selection
        range_group = QGroupBox("Row Range Selection")
        range_layout = QGridLayout(range_group)
        
        range_layout.addWidget(QLabel("From row:"), 0, 0)
        self.from_row_spin = QSpinBox()
        self.from_row_spin.setRange(0, 1000000)
        self.from_row_spin.setValue(0)
        range_layout.addWidget(self.from_row_spin, 0, 1)
        
        range_layout.addWidget(QLabel("To row:"), 0, 2)
        self.to_row_spin = QSpinBox()
        self.to_row_spin.setRange(1, 1000000)
        self.to_row_spin.setValue(100)
        range_layout.addWidget(self.to_row_spin, 0, 3)
        
        range_layout.addWidget(QLabel("Step:"), 1, 0)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(1)
        range_layout.addWidget(self.step_spin, 1, 1)
        
        apply_range_button = QPushButton("Apply Range Selection")
        apply_range_button.setIcon(load_bootstrap_icon("rulers"))
        apply_range_button.clicked.connect(self.apply_row_range)
        range_layout.addWidget(apply_range_button, 1, 3)
        
        layout.addWidget(range_group)
        
        # Index-based slicing
        index_group = QGroupBox("Index-Based Selection")
        index_layout = QVBoxLayout(index_group)
        
        index_layout.addWidget(QLabel("Select rows by index (comma-separated, ranges supported):"))
        self.index_selection_input = QLineEdit()
        self.index_selection_input.setPlaceholderText("e.g., 0, 2, 5-10, 15, 20-30")
        index_layout.addWidget(self.index_selection_input)
        
        apply_index_button = QPushButton("Apply Index Selection")
        apply_index_button.setIcon(load_bootstrap_icon("list-ol"))
        apply_index_button.clicked.connect(self.apply_index_selection)
        index_layout.addWidget(apply_index_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(index_group)
        
        # Head/Tail selection
        head_tail_group = QGroupBox("Get First/Last Rows")
        head_tail_layout = QGridLayout(head_tail_group)
        
        head_tail_layout.addWidget(QLabel("Get first:"), 0, 0)
        self.head_spin = QSpinBox()
        self.head_spin.setRange(1, 10000)
        self.head_spin.setValue(5)
        head_tail_layout.addWidget(self.head_spin, 0, 1)
        
        head_button = QPushButton("Get First Rows")
        head_button.setIcon(load_bootstrap_icon("chevron-double-up"))
        head_button.clicked.connect(self.get_head)
        head_tail_layout.addWidget(head_button, 0, 2)
        
        head_tail_layout.addWidget(QLabel("Get last:"), 1, 0)
        self.tail_spin = QSpinBox()
        self.tail_spin.setRange(1, 10000)
        self.tail_spin.setValue(5)
        head_tail_layout.addWidget(self.tail_spin, 1, 1)
        
        tail_button = QPushButton("Get Last Rows")
        tail_button.setIcon(load_bootstrap_icon("chevron-double-down"))
        tail_button.clicked.connect(self.get_tail)
        head_tail_layout.addWidget(tail_button, 1, 2)
        
        layout.addWidget(head_tail_group)
        
        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.slice_result_label = QLabel("No slicing operations performed yet")
        results_layout.addWidget(self.slice_result_label)
        
        layout.addWidget(results_group)
        layout.addStretch()
        
        return tab
    
    # ---------------------------------------------------------------
    # Create tab for study criteria management
    # ---------------------------------------------------------------
    def create_study_criteria_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Apply eligibility, enrollment, and attrition criteria from the study design to filter participants.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # Criteria source section
        source_group = QGroupBox("Criteria Source")
        source_layout = QVBoxLayout(source_group)
        
        # Dataset selection section
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Target Dataset:"))
        self.study_datasets_combo = QComboBox()
        self.study_datasets_combo.currentIndexChanged.connect(self.on_study_dataset_changed)
        dataset_layout.addWidget(self.study_datasets_combo)
        source_layout.addLayout(dataset_layout)
        
        # Refresh button to load criteria from study
        refresh_criteria_button = QPushButton("Load Criteria from Study Design")
        refresh_criteria_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_criteria_button.clicked.connect(self.load_study_criteria)
        source_layout.addWidget(refresh_criteria_button)
        
        # Status label for criteria loading
        self.criteria_status_label = QLabel("No criteria loaded. Click the button above to load.")
        self.criteria_status_label.setWordWrap(True)
        source_layout.addWidget(self.criteria_status_label)
        
        layout.addWidget(source_group)
        
        # Create tab widget for different criteria types
        self.criteria_tabs = QTabWidget()
        
        # Tab 1: Eligibility Criteria
        self.eligibility_tab = self.create_eligibility_criteria_tab()
        self.criteria_tabs.addTab(self.eligibility_tab, "Eligibility")
        
        # Tab 2: Enrollment Criteria
        self.enrollment_tab = self.create_enrollment_criteria_tab()
        self.criteria_tabs.addTab(self.enrollment_tab, "Enrollment")
        
        # Tab 3: Attrition/LTF Criteria
        self.attrition_tab = self.create_attrition_criteria_tab()
        self.criteria_tabs.addTab(self.attrition_tab, "Attrition")
        
        layout.addWidget(self.criteria_tabs)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        apply_all_button = QPushButton("Apply All Criteria")
        apply_all_button.setIcon(load_bootstrap_icon("check-circle"))
        apply_all_button.clicked.connect(self.apply_all_criteria)
        buttons_layout.addWidget(apply_all_button)
        
        apply_to_study_button = QPushButton("Apply and Save to Study")
        apply_to_study_button.setIcon(load_bootstrap_icon("save"))
        apply_to_study_button.clicked.connect(self.apply_and_save_to_study)
        buttons_layout.addWidget(apply_to_study_button)
        
        # Result label
        self.criteria_result_label = QLabel("No criteria applied yet")
        self.criteria_result_label.setStyleSheet("color: #666;")
        buttons_layout.addWidget(self.criteria_result_label)
        
        layout.addLayout(buttons_layout)
        
        return tab
    
    def create_eligibility_criteria_tab(self):
        """Create the tab for eligibility criteria"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Inclusion criteria
        inclusion_group = QGroupBox("Inclusion Criteria")
        inclusion_layout = QVBoxLayout(inclusion_group)
        
        # Table for inclusion criteria
        self.inclusion_criteria_table = QTableWidget()
        self.inclusion_criteria_table.setColumnCount(4)
        self.inclusion_criteria_table.setHorizontalHeaderLabels([
            "Criterion", "Study Column", "Dataset Column", "Apply"
        ])
        self.inclusion_criteria_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.inclusion_criteria_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.inclusion_criteria_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.inclusion_criteria_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        inclusion_layout.addWidget(self.inclusion_criteria_table)
        
        apply_inclusion_button = QPushButton("Apply Inclusion Criteria")
        apply_inclusion_button.setIcon(load_bootstrap_icon("check-circle"))
        apply_inclusion_button.clicked.connect(lambda: self.apply_criteria("inclusion"))
        inclusion_layout.addWidget(apply_inclusion_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(inclusion_group)
        
        # Exclusion criteria
        exclusion_group = QGroupBox("Exclusion Criteria")
        exclusion_layout = QVBoxLayout(exclusion_group)
        
        # Table for exclusion criteria
        self.exclusion_criteria_table = QTableWidget()
        self.exclusion_criteria_table.setColumnCount(4)
        self.exclusion_criteria_table.setHorizontalHeaderLabels([
            "Criterion", "Study Column", "Dataset Column", "Apply"
        ])
        self.exclusion_criteria_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.exclusion_criteria_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.exclusion_criteria_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.exclusion_criteria_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        exclusion_layout.addWidget(self.exclusion_criteria_table)
        
        apply_exclusion_button = QPushButton("Apply Exclusion Criteria")
        apply_exclusion_button.setIcon(load_bootstrap_icon("x-circle"))
        apply_exclusion_button.clicked.connect(lambda: self.apply_criteria("exclusion"))
        exclusion_layout.addWidget(apply_exclusion_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(exclusion_group)
        
        return tab
    
    def create_enrollment_criteria_tab(self):
        """Create the tab for enrollment criteria"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Enrollment status section
        enrollment_group = QGroupBox("Enrollment Status")
        enrollment_layout = QVBoxLayout(enrollment_group)
        
        # Table for enrollment criteria
        self.enrollment_criteria_table = QTableWidget()
        self.enrollment_criteria_table.setColumnCount(4)
        self.enrollment_criteria_table.setHorizontalHeaderLabels([
            "Criterion", "Study Column", "Dataset Column", "Apply"
        ])
        self.enrollment_criteria_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.enrollment_criteria_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.enrollment_criteria_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.enrollment_criteria_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        enrollment_layout.addWidget(self.enrollment_criteria_table)
        
        apply_enrollment_button = QPushButton("Apply Enrollment Criteria")
        apply_enrollment_button.setIcon(load_bootstrap_icon("person-check"))
        apply_enrollment_button.clicked.connect(lambda: self.apply_criteria("enrollment"))
        enrollment_layout.addWidget(apply_enrollment_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(enrollment_group)
        
        return tab
    
    def create_attrition_criteria_tab(self):
        """Create the tab for attrition/lost to follow-up criteria"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Attrition status section
        attrition_group = QGroupBox("Attrition/Lost to Follow-up Status")
        attrition_layout = QVBoxLayout(attrition_group)
        
        # Table for attrition criteria
        self.attrition_criteria_table = QTableWidget()
        self.attrition_criteria_table.setColumnCount(4)
        self.attrition_criteria_table.setHorizontalHeaderLabels([
            "Criterion", "Study Column", "Dataset Column", "Apply"
        ])
        self.attrition_criteria_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.attrition_criteria_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.attrition_criteria_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.attrition_criteria_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        attrition_layout.addWidget(self.attrition_criteria_table)
        
        apply_attrition_button = QPushButton("Apply Attrition Criteria")
        apply_attrition_button.setIcon(load_bootstrap_icon("person-dash"))
        apply_attrition_button.clicked.connect(lambda: self.apply_criteria("attrition"))
        attrition_layout.addWidget(apply_attrition_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(attrition_group)
        
        return tab
    
    # ---------------------------------------------------------------
    # Methods for handling dataset selection and display
    # ---------------------------------------------------------------
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
                self.add_source(name, df)
                count += 1
        
        self.status_bar.showMessage(f"Refreshed {count} datasets from active study")
    
    def add_source(self, name, dataframe):
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
            self.slicing_history = []
            self.update_history()
            
            # Update the study datasets combo if it exists
            if hasattr(self, 'study_datasets_combo'):
                self.refresh_study_datasets_combo()
            
            # Specifically select this dataset in the ordinal encoding combo
            if hasattr(self, 'ordinal_dataset_combo'):
                index = self.ordinal_dataset_combo.findText(name)
                if index >= 0:
                    self.ordinal_dataset_combo.setCurrentIndex(index)
    
    def display_dataset(self, name, dataframe):
        self.current_dataset_label.setText(f"Dataset: {name}")
        rows, cols = dataframe.shape
        self.dataset_info_label.setText(f"Rows: {rows} | Columns: {cols}")
        self.preview_display.display_dataframe(dataframe)
        self.save_name_input.setText(f"{name}_sliced")
    
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
        
        # Column list
        summary += "\nColumns:\n"
        for col in df.columns:
            summary += f"  {col}\n"
        
        self.summary_text.setPlainText(summary)
    
    def update_column_selections(self):
        """Update all combo boxes and lists with column names"""
        if self.current_dataframe is None:
            return
            
        columns = list(self.current_dataframe.columns)
        
        # Update all combo boxes with column names
        for combo in [
            self.filter_column_combo, self.stratify_column_combo
        ]:
            if isinstance(combo, QComboBox):
                combo.clear()
                combo.addItems(columns)
                
        # Also update column combo boxes in the criteria tables
        for table in [
            self.inclusion_criteria_table, 
            self.exclusion_criteria_table,
            self.enrollment_criteria_table,
            self.attrition_criteria_table
        ]:
            for row in range(table.rowCount()):
                column_combo = table.cellWidget(row, 2)
                if column_combo:
                    current_text = column_combo.currentText()
                    column_combo.clear()
                    column_combo.addItem("-- Select Column --")
                    column_combo.addItems(columns)
                    
                    # Try to restore previous selection if it exists
                    if current_text in columns:
                        column_combo.setCurrentText(current_text)
        
        # Update column selection lists
        self.columns_list.clear()
        self.reorder_columns_list.clear()
        for col in columns:
            self.columns_list.addItem(col)
            self.reorder_columns_list.addItem(col)
            
        # Select all columns by default
        for i in range(self.columns_list.count()):
            self.columns_list.item(i).setSelected(True)
        
        # Update row spinners based on dataset size
        rows = len(self.current_dataframe)
        self.from_row_spin.setMaximum(max(0, rows - 1))
        self.to_row_spin.setMaximum(rows)
        self.to_row_spin.setValue(min(rows, 100))
    
    # ---------------------------------------------------------------
    # Methods for row filtering tab
    # ---------------------------------------------------------------
    def on_filter_column_changed(self, column):
        """Adjust filter conditions based on column data type"""
        if not column or self.current_dataframe is None:
            return
        
        df = self.current_dataframe
        
        # Adjust available conditions based on data type
        self.filter_condition_combo.clear()
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # Numeric column
            self.filter_condition_combo.addItems([
                "Equal to", 
                "Not equal to", 
                "Greater than", 
                "Less than", 
                "Greater than or equal to", 
                "Less than or equal to",
                "Is missing", 
                "Is not missing"
            ])
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            # Date column
            self.filter_condition_combo.addItems([
                "Equal to", 
                "Not equal to", 
                "Greater than (later than)", 
                "Less than (earlier than)", 
                "Greater than or equal to", 
                "Less than or equal to",
                "Is missing", 
                "Is not missing"
            ])
        else:
            # Text or categorical column
            self.filter_condition_combo.addItems([
                "Equal to", 
                "Not equal to", 
                "Contains", 
                "Does not contain", 
                "Starts with", 
                "Ends with", 
                "Is missing", 
                "Is not missing"
            ])
    
    def on_sampling_method_changed(self, method):
        """Show/hide relevant inputs based on sampling method"""
        # Hide all inputs first
        self.sample_n_label.setVisible(False)
        self.sample_n_spin.setVisible(False)
        self.sample_frac_label.setVisible(False)
        self.sample_frac_spin.setVisible(False)
        self.stratify_label.setVisible(False)
        self.stratify_column_combo.setVisible(False)
        
        # Show relevant inputs based on method
        if "fixed number" in method:
            self.sample_n_label.setVisible(True)
            self.sample_n_spin.setVisible(True)
        elif "percentage" in method:
            self.sample_frac_label.setVisible(True)
            self.sample_frac_spin.setVisible(True)
        elif "systematic" in method:
            self.sample_n_label.setVisible(True)
            self.sample_n_spin.setVisible(True)
        elif "stratified" in method:
            self.sample_frac_label.setVisible(True)
            self.sample_frac_spin.setVisible(True)
            self.stratify_label.setVisible(True)
            self.stratify_column_combo.setVisible(True)
    
    def apply_row_filter(self):
        """Apply filter to rows based on condition"""
        if self.current_dataframe is None:
            return
        
        column = self.filter_column_combo.currentText()
        condition = self.filter_condition_combo.currentText()
        value_text = self.filter_value_input.text()
        
        if not column:
            QMessageBox.warning(self, "Warning", "Please select a column")
            return
        
        if condition not in ["Is missing", "Is not missing"] and not value_text:
            QMessageBox.warning(self, "Warning", "Please enter a filter value")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Apply filter based on condition
            if condition == "Equal to":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] == value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] == value
                else:
                    mask = df[column] == value_text
                
                filter_desc = f"'{column}' equal to '{value_text}'"
                
            elif condition == "Not equal to":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] != value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] != value
                else:
                    mask = df[column] != value_text
                
                filter_desc = f"'{column}' not equal to '{value_text}'"
                
            elif condition == "Greater than" or condition == "Greater than (later than)":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] > value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] > value
                else:
                    mask = df[column] > value_text
                
                filter_desc = f"'{column}' greater than '{value_text}'"
                
            elif condition == "Less than" or condition == "Less than (earlier than)":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] < value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] < value
                else:
                    mask = df[column] < value_text
                
                filter_desc = f"'{column}' less than '{value_text}'"
                
            elif condition == "Greater than or equal to":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] >= value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] >= value
                else:
                    mask = df[column] >= value_text
                
                filter_desc = f"'{column}' greater than or equal to '{value_text}'"
                
            elif condition == "Less than or equal to":
                # Convert value to appropriate type
                if pd.api.types.is_numeric_dtype(df[column]):
                    value = float(value_text)
                    mask = df[column] <= value
                elif pd.api.types.is_datetime64_any_dtype(df[column]):
                    value = pd.to_datetime(value_text)
                    mask = df[column] <= value
                else:
                    mask = df[column] <= value_text
                
                filter_desc = f"'{column}' less than or equal to '{value_text}'"
                
            elif condition == "Contains":
                # Convert column to string and check for containment
                mask = df[column].astype(str).str.contains(value_text, na=False)
                filter_desc = f"'{column}' contains '{value_text}'"
                
            elif condition == "Does not contain":
                # Convert column to string and check for non-containment
                mask = ~df[column].astype(str).str.contains(value_text, na=False)
                filter_desc = f"'{column}' does not contain '{value_text}'"
                
            elif condition == "Starts with":
                # Convert column to string and check for start with
                mask = df[column].astype(str).str.startswith(value_text, na=False)
                filter_desc = f"'{column}' starts with '{value_text}'"
                
            elif condition == "Ends with":
                # Convert column to string and check for end with
                mask = df[column].astype(str).str.endswith(value_text, na=False)
                filter_desc = f"'{column}' ends with '{value_text}'"
                
            elif condition == "Is missing":
                mask = df[column].isna()
                filter_desc = f"'{column}' is missing"
                
            elif condition == "Is not missing":
                mask = df[column].notna()
                filter_desc = f"'{column}' is not missing"
            
            # Apply filter
            filtered_df = df[mask].reset_index(drop=True)
            
            # Update preview
            self._preview_df = filtered_df
            self.preview_display.display_dataframe(filtered_df)
            
            # Add to history
            removed_count = len(df) - len(filtered_df)
            self.add_to_history(f"Filtered rows where {filter_desc} (kept {len(filtered_df)}, removed {removed_count})")
            
            # Update result label
            self.filter_result_label.setText(
                f"Filtered rows where {filter_desc}\n"
                f"Results: {len(filtered_df)} rows kept, {removed_count} rows removed")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Filter failed: {str(e)}")
            self.filter_result_label.setText(f"Error: {str(e)}")
    
    def apply_sampling(self):
        """Apply sampling to the dataset"""
        if self.current_dataframe is None:
            return
        
        # Get sampling method
        method = self.sampling_method_combo.currentText()
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Set random seed if requested
            random_state = 42 if self.random_seed_check.isChecked() else None
            
            # Apply sampling based on method
            if "fixed number" in method:
                n = min(self.sample_n_spin.value(), len(df))
                sampled_df = df.sample(n=n, random_state=random_state)
                sample_desc = f"random sample of {n} rows"
                
            elif "percentage" in method:
                frac = self.sample_frac_spin.value() / 100
                sampled_df = df.sample(frac=frac, random_state=random_state)
                sample_desc = f"random sample of {frac:.1%} ({len(sampled_df)} rows)"
                
            elif "systematic" in method:
                n = min(self.sample_n_spin.value(), len(df))
                step = max(len(df) // n, 1)
                indices = list(range(0, len(df), step))[:n]
                sampled_df = df.iloc[indices].reset_index(drop=True)
                sample_desc = f"systematic sample of {len(sampled_df)} rows (every {step} rows)"
                
            elif "stratified" in method:
                strat_col = self.stratify_column_combo.currentText()
                frac = self.sample_frac_spin.value() / 100
                
                if not strat_col:
                    QMessageBox.warning(self, "Warning", "Please select a column for stratification")
                    return
                
                # Handle missing values in stratification column
                if df[strat_col].isna().any():
                    df_clean = df.copy()
                    df_clean[strat_col] = df_clean[strat_col].fillna('_MISSING_')
                else:
                    df_clean = df
                
                # Group by stratification column and sample from each group
                groups = []
                for name, group in df_clean.groupby(strat_col):
                    sampled_group = group.sample(frac=frac, random_state=random_state)
                    groups.append(sampled_group)
                
                # Combine sampled groups
                sampled_df = pd.concat(groups).reset_index(drop=True)
                
                # If we used a temporary '_MISSING_' value, remove it from the results
                if '_MISSING_' in sampled_df[strat_col].values:
                    sampled_df.loc[sampled_df[strat_col] == '_MISSING_', strat_col] = np.nan
                
                sample_desc = f"stratified sample of {frac:.1%} ({len(sampled_df)} rows) by '{strat_col}'"
            
            # Update preview
            self._preview_df = sampled_df
            self.preview_display.display_dataframe(sampled_df)
            
            # Add to history
            self.add_to_history(f"Created {sample_desc}")
            
            # Update result label
            self.filter_result_label.setText(
                f"Created {sample_desc}\n"
                f"Original size: {len(df)} rows, Sample size: {len(sampled_df)} rows")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Sampling failed: {str(e)}")
            self.filter_result_label.setText(f"Error: {str(e)}")
    
    def apply_complex_filter(self):
        """Apply a custom filter expression to the dataset"""
        if self.current_dataframe is None:
            return
        
        expression = self.complex_filter_expr.toPlainText()
        
        if not expression:
            QMessageBox.warning(self, "Warning", "Please provide a filter expression")
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
            
            # Evaluate the expression to get a boolean mask
            mask = eval(expression, namespace)
            
            # Apply filter
            filtered_df = df[mask].reset_index(drop=True)
            
            # Update preview
            self._preview_df = filtered_df
            self.preview_display.display_dataframe(filtered_df)
            
            # Add to history
            removed_count = len(df) - len(filtered_df)
            self.add_to_history(f"Applied custom filter (kept {len(filtered_df)}, removed {removed_count})")
            
            # Update result label
            self.filter_result_label.setText(
                f"Applied custom filter: {expression}\n"
                f"Results: {len(filtered_df)} rows kept, {removed_count} rows removed")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Filter expression failed: {str(e)}")
            self.filter_result_label.setText(f"Error: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods for column management tab
    # ---------------------------------------------------------------
    def select_all_columns(self):
        """Select all columns in the list"""
        for i in range(self.columns_list.count()):
            self.columns_list.item(i).setSelected(True)
    
    def deselect_all_columns(self):
        """Deselect all columns in the list"""
        for i in range(self.columns_list.count()):
            self.columns_list.item(i).setSelected(False)
    
    def invert_column_selection(self):
        """Invert the current column selection"""
        for i in range(self.columns_list.count()):
            item = self.columns_list.item(i)
            item.setSelected(not item.isSelected())
    
    def keep_selected_columns(self):
        """Keep only the selected columns in the dataset"""
        if self.current_dataframe is None:
            return
        
        # Get selected columns
        selected_items = self.columns_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one column to keep")
            return
        
        selected_columns = [item.text() for item in selected_items]
        
        # Create a copy of the dataset with only selected columns
        df = self._preview_df.copy()
        
        try:
            # Check if all selected columns exist
            missing_cols = [col for col in selected_columns if col not in df.columns]
            if missing_cols:
                QMessageBox.warning(self, "Warning", f"Some selected columns don't exist: {', '.join(missing_cols)}")
                return
            
            # Keep only selected columns
            dropped_columns = [col for col in df.columns if col not in selected_columns]
            sliced_df = df[selected_columns]
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Kept {len(selected_columns)} columns, dropped {len(dropped_columns)} columns")
            
            # Update result label
            self.column_ops_result_label.setText(
                f"Kept {len(selected_columns)} columns, dropped {len(dropped_columns)} columns\n"
                f"Kept columns: {', '.join(selected_columns[:5])}{' and more...' if len(selected_columns) > 5 else ''}")
            
            # Update column selections
            self.update_column_selections()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Column selection failed: {str(e)}")
            self.column_ops_result_label.setText(f"Error: {str(e)}")
    
    def drop_columns_by_pattern(self):
        """Drop columns that match a specific pattern"""
        if self.current_dataframe is None:
            return
        
        pattern = self.column_pattern_input.text()
        is_regex = self.pattern_is_regex_check.isChecked()
        
        if not pattern:
            QMessageBox.warning(self, "Warning", "Please enter a pattern")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Find columns matching the pattern
            if is_regex:
                matching_columns = [col for col in df.columns if re.search(pattern, col)]
            else:
                matching_columns = [col for col in df.columns if pattern in col]
            
            if not matching_columns:
                QMessageBox.information(self, "Information", "No columns match the pattern")
                return
            
            # Drop matching columns
            sliced_df = df.drop(columns=matching_columns)
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Dropped {len(matching_columns)} columns matching pattern '{pattern}'")
            
            # Update result label
            self.column_ops_result_label.setText(
                f"Dropped {len(matching_columns)} columns matching '{pattern}':\n"
                f"{', '.join(matching_columns[:5])}{' and more...' if len(matching_columns) > 5 else ''}")
            
            # Update column selections
            self.update_column_selections()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Pattern-based column dropping failed: {str(e)}")
            self.column_ops_result_label.setText(f"Error: {str(e)}")
    
    def reorder_columns(self):
        """Move selected columns to the beginning of the dataset"""
        if self.current_dataframe is None:
            return
        
        # Get selected columns
        selected_items = self.reorder_columns_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one column to move")
            return
        
        move_columns = [item.text() for item in selected_items]
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Check if all selected columns exist
            missing_cols = [col for col in move_columns if col not in df.columns]
            if missing_cols:
                QMessageBox.warning(self, "Warning", f"Some selected columns don't exist: {', '.join(missing_cols)}")
                return
            
            # Create new column order with selected columns first, then the rest
            other_columns = [col for col in df.columns if col not in move_columns]
            new_order = move_columns + other_columns
            
            # Reorder columns
            sliced_df = df[new_order]
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Moved {len(move_columns)} columns to beginning of dataset")
            
            # Update result label
            self.column_ops_result_label.setText(
                f"Moved {len(move_columns)} columns to beginning:\n"
                f"{', '.join(move_columns[:5])}{' and more...' if len(move_columns) > 5 else ''}")
            
            # Update column selections
            self.update_column_selections()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Column reordering failed: {str(e)}")
            self.column_ops_result_label.setText(f"Error: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods for dataset slicing tab
    # ---------------------------------------------------------------
    def apply_row_range(self):
        """Slice dataset by row range"""
        if self.current_dataframe is None:
            return
        
        from_row = self.from_row_spin.value()
        to_row = self.to_row_spin.value()
        step = self.step_spin.value()
        
        if from_row >= to_row:
            QMessageBox.warning(self, "Warning", "From row must be less than To row")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Apply range slice
            sliced_df = df.iloc[from_row:to_row:step].copy().reset_index(drop=True)
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Sliced rows from {from_row} to {to_row} with step {step} (kept {len(sliced_df)} rows)")
            
            # Update result label
            self.slice_result_label.setText(
                f"Sliced rows from {from_row} to {to_row} with step {step}\n"
                f"Result: {len(sliced_df)} rows")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Row range slicing failed: {str(e)}")
            self.slice_result_label.setText(f"Error: {str(e)}")
    
    def apply_index_selection(self):
        """Select rows by index"""
        if self.current_dataframe is None:
            return
        
        index_text = self.index_selection_input.text()
        
        if not index_text:
            QMessageBox.warning(self, "Warning", "Please enter indices to select")
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Parse index selections
            indices = []
            for part in index_text.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-')
                    indices.extend(range(int(start), int(end) + 1))
                else:
                    indices.append(int(part))
            
            # Filter indices that are in range
            valid_indices = [idx for idx in indices if 0 <= idx < len(df)]
            
            if not valid_indices:
                QMessageBox.warning(self, "Warning", "No valid indices provided")
                return
            
            # Apply index selection
            sliced_df = df.iloc[valid_indices].copy().reset_index(drop=True)
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Selected {len(valid_indices)} rows by index")
            
            # Update result label
            self.slice_result_label.setText(
                f"Selected {len(valid_indices)} rows by index\n"
                f"Result: {len(sliced_df)} rows")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Index selection failed: {str(e)}")
            self.slice_result_label.setText(f"Error: {str(e)}")
    
    def get_head(self):
        """Get the first N rows of the dataset"""
        if self.current_dataframe is None:
            return
        
        n_rows = self.head_spin.value()
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Get head
            sliced_df = df.head(n_rows).copy()
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Selected first {n_rows} rows (head)")
            
            # Update result label
            self.slice_result_label.setText(f"Selected first {n_rows} rows (head)")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Head selection failed: {str(e)}")
            self.slice_result_label.setText(f"Error: {str(e)}")
    
    def get_tail(self):
        """Get the last N rows of the dataset"""
        if self.current_dataframe is None:
            return
        
        n_rows = self.tail_spin.value()
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        try:
            # Get tail
            sliced_df = df.tail(n_rows).copy()
            
            # Update preview
            self._preview_df = sliced_df
            self.preview_display.display_dataframe(sliced_df)
            
            # Add to history
            self.add_to_history(f"Selected last {n_rows} rows (tail)")
            
            # Update result label
            self.slice_result_label.setText(f"Selected last {n_rows} rows (tail)")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Tail selection failed: {str(e)}")
            self.slice_result_label.setText(f"Error: {str(e)}")
    
    # ---------------------------------------------------------------
    # Methods for slicing history tab
    # ---------------------------------------------------------------
    def add_to_history(self, operation):
        """Add an operation to the slicing history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.slicing_history.append(f"[{timestamp}] {operation}")
        self.update_history()
    
    def update_history(self):
        """Update the history display"""
        if not self.slicing_history:
            self.history_text.setPlainText("No slicing operations performed yet.")
        else:
            self.history_text.setPlainText("\n".join(self.slicing_history))
    
    def export_slicing_report(self):
        """Export the slicing history to a text file"""
        if not self.slicing_history:
            QMessageBox.information(self, "Information", "No slicing operations to export")
            return
        
        # Get file name for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Slicing Report", f"{self.current_name}_slicing_report.txt", "Text Files (*.txt)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "w") as f:
                f.write(f"Data Slicing Report for: {self.current_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Slicing Operations:\n")
                for operation in self.slicing_history:
                    f.write(f"{operation}\n")
                
                # Add dataset summary
                f.write("\n\nDataset Summary:\n")
                rows, cols = self._preview_df.shape
                f.write(f"Rows: {rows}, Columns: {cols}\n")
                f.write("Columns:\n")
                for col in self._preview_df.columns:
                    f.write(f"  {col}\n")
            
            QMessageBox.information(self, "Success", f"Slicing report saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")
    
    def reset_dataset(self):
        """Reset the preview dataset to the original"""
        if self.current_dataframe is None:
            return
        
        reply = QMessageBox.question(
            self, "Confirm Reset", 
            "Are you sure you want to reset all slicing operations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._preview_df = self.current_dataframe.copy()
            self.preview_display.display_dataframe(self._preview_df)
            self.slicing_history = []
            self.update_history()
            self.status_bar.showMessage("Dataset reset to original state")
    
    # ---------------------------------------------------------------
    # Save the sliced dataset
    # ---------------------------------------------------------------
    def save_sliced_dataset(self):
        """Save the sliced dataset"""
        new_name = self.save_name_input.text()
        if not new_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the sliced dataset")
            return
        if self._preview_df is None:
            QMessageBox.warning(self, "Error", "No dataset to save")
            return
        
        # Add the sliced dataset as a source
        self.add_source(new_name, self._preview_df)
        
        # Add the dataset to the studies manager if available
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            main_window.studies_manager.add_dataset_to_active_study(new_name, self._preview_df)
        
        # Add final slicing operation to history
        self.add_to_history(f"Saved sliced dataset as '{new_name}'")
        
        QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved successfully")
        self.status_bar.showMessage(f"Sliced dataset saved as '{new_name}'")

    def refresh_study_datasets_combo(self):
        """Refresh the list of available datasets from the active study"""
        if not hasattr(self, 'study_datasets_combo'):
            return
            
        # Save current selection
        current_text = self.study_datasets_combo.currentText()
        
        self.study_datasets_combo.clear()
        self.ordinal_dataset_combo.clear()  # Clear the ordinal dataset combo
        
        # Add current dataset
        if self.current_name:
            self.study_datasets_combo.addItem(self.current_name)
            self.ordinal_dataset_combo.addItem(self.current_name)  # Add to ordinal combo
        
        # Get reference to the main app window
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            # Get datasets from active study
            datasets = main_window.studies_manager.get_datasets_from_active_study()
            
            # Add each dataset to the combo box if not already there
            for name, _ in datasets:
                # Skip the current dataset which was already added
                if name != self.current_name:
                    self.study_datasets_combo.addItem(name)
                    self.ordinal_dataset_combo.addItem(name)  # Add to ordinal combo
        
        # Restore selection if possible
        index = self.study_datasets_combo.findText(current_text)
        if index >= 0:
            self.study_datasets_combo.setCurrentIndex(index)
        elif self.study_datasets_combo.count() > 0:
            self.study_datasets_combo.setCurrentIndex(0)
        
        # Set the current dataset in the ordinal dropdown and trigger column update
        if self.ordinal_dataset_combo.count() > 0:
            self.ordinal_dataset_combo.setCurrentIndex(0)
            # This will trigger on_ordinal_dataset_changed to populate the columns

    def on_study_dataset_changed(self, index):
        """Handle selection of a different dataset in the study datasets combo"""
        if index < 0 or not hasattr(self, 'study_datasets_combo'):
            return
            
        selected_dataset = self.study_datasets_combo.currentText()
        
        # If the selected dataset is already loaded, just update the UI
        if selected_dataset == self.current_name:
            return
            
        # Check if we need to load this dataset from the study manager
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            datasets = main_window.studies_manager.get_datasets_from_active_study()
            
            for name, df in datasets:
                if name == selected_dataset:
                    # Add to our internal dictionary if not already there
                    if name not in self.dataframes:
                        self.add_source(name, df)
                    
                    # Select this source
                    items = self.sources_list.findItems(name, Qt.MatchFlag.MatchExactly)
                    if items:
                        self.sources_list.setCurrentItem(items[0])
                        self.on_source_selected(items[0])
                    break

    def load_study_criteria(self):
        """Load criteria from the study design section"""
        # Get reference to the main app window
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        # Get active study
        active_study = main_window.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'study_design_section'):
            QMessageBox.warning(self, "Error", "No active study with a study design section")
            return
        
        # Refresh study datasets combo
        self.refresh_study_datasets_combo()
        
        study_design = active_study.study_design_section
        
        # Load eligibility criteria
        try:
            # Get inclusion criteria
            inclusion_criteria = []
            inclusion_table = getattr(study_design, 'inclusion_criteria_table', None)
            if inclusion_table:
                inclusion_df = study_design.table_to_dataframe(inclusion_table)
                for _, row in inclusion_df.iterrows():
                    criterion = row.get('Criterion', '')
                    if criterion:
                        inclusion_criteria.append({
                            'criterion': criterion,
                            'column': row.get('Variable', ''),
                            'value': row.get('Value', ''),
                            'condition': row.get('Condition', '=='),
                        })
            
            # Get exclusion criteria
            exclusion_criteria = []
            exclusion_table = getattr(study_design, 'exclusion_criteria_table', None)
            if exclusion_table:
                exclusion_df = study_design.table_to_dataframe(exclusion_table)
                for _, row in exclusion_df.iterrows():
                    criterion = row.get('Criterion', '')
                    if criterion:
                        exclusion_criteria.append({
                            'criterion': criterion,
                            'column': row.get('Variable', ''),
                            'value': row.get('Value', ''),
                            'condition': row.get('Condition', '=='),
                        })
            
            # Populate inclusion criteria table
            self.inclusion_criteria_table.setRowCount(len(inclusion_criteria))
            for i, criterion in enumerate(inclusion_criteria):
                self.inclusion_criteria_table.setItem(i, 0, QTableWidgetItem(criterion['criterion']))
                self.inclusion_criteria_table.setItem(i, 1, QTableWidgetItem(criterion['column']))
                
                # Add combobox for dataset column selection
                column_combo = QComboBox()
                column_combo.addItem("-- Select Column --")
                if self.current_dataframe is not None:
                    column_combo.addItems(self.current_dataframe.columns)
                    
                    # Try to auto-select matching column based on variable name
                    variable_name = criterion['column'].lower()
                    for col_idx in range(column_combo.count()):
                        col_name = column_combo.itemText(col_idx).lower()
                        if col_name == variable_name or variable_name in col_name or col_name in variable_name:
                            column_combo.setCurrentIndex(col_idx)
                            break
                            
                self.inclusion_criteria_table.setCellWidget(i, 2, column_combo)
                
                # Add checkbox for applying criterion
                apply_check = QCheckBox()
                apply_check.setChecked(True)
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(apply_check)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                cell_widget.setLayout(layout)
                self.inclusion_criteria_table.setCellWidget(i, 3, cell_widget)
            
            # Populate exclusion criteria table
            self.exclusion_criteria_table.setRowCount(len(exclusion_criteria))
            for i, criterion in enumerate(exclusion_criteria):
                self.exclusion_criteria_table.setItem(i, 0, QTableWidgetItem(criterion['criterion']))
                self.exclusion_criteria_table.setItem(i, 1, QTableWidgetItem(criterion['column']))
                
                # Add combobox for dataset column selection
                column_combo = QComboBox()
                column_combo.addItem("-- Select Column --")
                if self.current_dataframe is not None:
                    column_combo.addItems(self.current_dataframe.columns)
                    
                    # Try to auto-select matching column based on variable name
                    variable_name = criterion['column'].lower()
                    for col_idx in range(column_combo.count()):
                        col_name = column_combo.itemText(col_idx).lower()
                        if col_name == variable_name or variable_name in col_name or col_name in variable_name:
                            column_combo.setCurrentIndex(col_idx)
                            break
                            
                self.exclusion_criteria_table.setCellWidget(i, 2, column_combo)
                
                # Add checkbox for applying criterion
                apply_check = QCheckBox()
                apply_check.setChecked(True)
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(apply_check)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                cell_widget.setLayout(layout)
                self.exclusion_criteria_table.setCellWidget(i, 3, cell_widget)
            
            # Load enrollment criteria
            enrollment_criteria = []
            enrollment_status = getattr(study_design, 'enrollment_status_table', None)
            if enrollment_status:
                enrollment_df = study_design.table_to_dataframe(enrollment_status)
                for _, row in enrollment_df.iterrows():
                    status = row.get('Status', '')
                    if status:
                        enrollment_criteria.append({
                            'criterion': status,
                            'column': 'enrollment_status',
                            'value': status,
                            'condition': '==',
                        })
            
            # Populate enrollment criteria table
            self.enrollment_criteria_table.setRowCount(len(enrollment_criteria))
            for i, criterion in enumerate(enrollment_criteria):
                self.enrollment_criteria_table.setItem(i, 0, QTableWidgetItem(criterion['criterion']))
                self.enrollment_criteria_table.setItem(i, 1, QTableWidgetItem(criterion['column']))
                
                # Add combobox for dataset column selection
                column_combo = QComboBox()
                column_combo.addItem("-- Select Column --")
                if self.current_dataframe is not None:
                    column_combo.addItems(self.current_dataframe.columns)
                    
                    # Try to auto-select matching column for enrollment status
                    for col_idx in range(column_combo.count()):
                        col_name = column_combo.itemText(col_idx).lower()
                        if 'enroll' in col_name or 'status' in col_name:
                            column_combo.setCurrentIndex(col_idx)
                            break
                            
                self.enrollment_criteria_table.setCellWidget(i, 2, column_combo)
                
                # Add checkbox for applying criterion
                apply_check = QCheckBox()
                apply_check.setChecked(True)
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(apply_check)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                cell_widget.setLayout(layout)
                self.enrollment_criteria_table.setCellWidget(i, 3, cell_widget)
            
            # Load attrition criteria
            attrition_criteria = []
            ltf_status = getattr(study_design, 'ltf_status_table', None)
            if ltf_status:
                ltf_df = study_design.table_to_dataframe(ltf_status)
                for _, row in ltf_df.iterrows():
                    status = row.get('Status', '')
                    if status:
                        attrition_criteria.append({
                            'criterion': status,
                            'column': 'ltf_status',
                            'value': status,
                            'condition': '==',
                        })
            
            # Populate attrition criteria table
            self.attrition_criteria_table.setRowCount(len(attrition_criteria))
            for i, criterion in enumerate(attrition_criteria):
                self.attrition_criteria_table.setItem(i, 0, QTableWidgetItem(criterion['criterion']))
                self.attrition_criteria_table.setItem(i, 1, QTableWidgetItem(criterion['column']))
                
                # Add combobox for dataset column selection
                column_combo = QComboBox()
                column_combo.addItem("-- Select Column --")
                if self.current_dataframe is not None:
                    column_combo.addItems(self.current_dataframe.columns)
                    
                    # Try to auto-select matching column for LTF/attrition status
                    for col_idx in range(column_combo.count()):
                        col_name = column_combo.itemText(col_idx).lower()
                        if 'ltf' in col_name or 'attrition' in col_name or 'follow' in col_name:
                            column_combo.setCurrentIndex(col_idx)
                            break
                            
                self.attrition_criteria_table.setCellWidget(i, 2, column_combo)
                
                # Add checkbox for applying criterion
                apply_check = QCheckBox()
                apply_check.setChecked(True)
                cell_widget = QWidget()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(apply_check)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                cell_widget.setLayout(layout)
                self.attrition_criteria_table.setCellWidget(i, 3, cell_widget)
            
            # Update status
            total_criteria = (len(inclusion_criteria) + len(exclusion_criteria) + 
                            len(enrollment_criteria) + len(attrition_criteria))
            
            if total_criteria > 0:
                self.criteria_status_label.setText(
                    f"Loaded {total_criteria} criteria from the study design: "
                    f"{len(inclusion_criteria)} inclusion, {len(exclusion_criteria)} exclusion, "
                    f"{len(enrollment_criteria)} enrollment, {len(attrition_criteria)} attrition."
                )
            else:
                self.criteria_status_label.setText(
                    "No criteria found in the study design. Please define criteria in the Study Design section first."
                )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load criteria: {str(e)}")
            self.criteria_status_label.setText(f"Error loading criteria: {str(e)}")

    def apply_criteria(self, criteria_type):
        """Apply specific criteria type"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Warning", "No dataset selected")
            return
        
        # Get the appropriate table based on criteria type
        table = None
        criteria_desc = ""
        if criteria_type == "inclusion":
            table = self.inclusion_criteria_table
            criteria_desc = "inclusion"
        elif criteria_type == "exclusion":
            table = self.exclusion_criteria_table
            criteria_desc = "exclusion"
        elif criteria_type == "enrollment":
            table = self.enrollment_criteria_table
            criteria_desc = "enrollment"
        elif criteria_type == "attrition":
            table = self.attrition_criteria_table
            criteria_desc = "attrition"
        
        if not table:
            return
        
        # Create a copy of the dataset for preview
        df = self._preview_df.copy()
        
        # Collect all active criteria
        active_criteria = []
        for row in range(table.rowCount()):
            # Get checkbox in the apply column
            cell_widget = table.cellWidget(row, 3)
            if cell_widget:
                checkbox = cell_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    criterion = table.item(row, 0).text()
                    study_column = table.item(row, 1).text()
                    
                    # Get selected dataset column
                    column_combo = table.cellWidget(row, 2)
                    if column_combo and column_combo.currentText() != "-- Select Column --":
                        dataset_column = column_combo.currentText()
                        active_criteria.append({
                            'criterion': criterion,
                            'study_column': study_column,
                            'dataset_column': dataset_column
                        })
        
        if not active_criteria:
            QMessageBox.information(self, "Information", f"No {criteria_desc} criteria to apply")
            return
        
        try:
            # Apply criteria
            rows_before = len(df)
            
            if criteria_type == "inclusion":
                # For inclusion, keep rows that match ALL criteria
                for criterion in active_criteria:
                    # Simple equality match for now - can be extended with more complex conditions
                    mask = df[criterion['dataset_column']].astype(str).str.contains(criterion['criterion'], na=False)
                    df = df[mask].reset_index(drop=True)
            
            elif criteria_type == "exclusion":
                # For exclusion, remove rows that match ANY criteria
                for criterion in active_criteria:
                    # Simple equality match for now - can be extended with more complex conditions
                    mask = ~df[criterion['dataset_column']].astype(str).str.contains(criterion['criterion'], na=False)
                    df = df[mask].reset_index(drop=True)
            
            elif criteria_type == "enrollment" or criteria_type == "attrition":
                # For these types, keep rows matching ANY of the selected status values
                combined_mask = pd.Series(False, index=df.index)
                
                for criterion in active_criteria:
                    mask = df[criterion['dataset_column']].astype(str).str.contains(criterion['criterion'], na=False)
                    combined_mask = combined_mask | mask
                
                df = df[combined_mask].reset_index(drop=True)
            
            # Update preview
            self._preview_df = df
            self.preview_display.display_dataframe(df)
            
            # Add to history
            rows_after = len(df)
            rows_removed = rows_before - rows_after
            criteria_summary = ', '.join([c['criterion'] for c in active_criteria])
            self.add_to_history(f"Applied {len(active_criteria)} {criteria_desc} criteria: {criteria_summary} (kept {rows_after}, removed {rows_removed})")
            
            # Update result label
            self.criteria_result_label.setText(
                f"Applied {len(active_criteria)} {criteria_desc} criteria: "
                f"{rows_after} rows kept, {rows_removed} rows removed"
            )
            
            # Return the criteria details for metadata
            return {
                'type': criteria_type,
                'criteria': active_criteria,
                'rows_before': rows_before,
                'rows_after': rows_after,
                'rows_removed': rows_removed,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to apply criteria: {str(e)}")
            self.criteria_result_label.setText(f"Error: {str(e)}")
            return None
    
    def apply_all_criteria(self):
        """Apply all active criteria from all tabs"""
        # Apply criteria in this order: inclusion, exclusion, enrollment, attrition
        inclusion_results = self.apply_criteria("inclusion")
        exclusion_results = self.apply_criteria("exclusion")
        enrollment_results = self.apply_criteria("enrollment")
        attrition_results = self.apply_criteria("attrition")
        
        # Add a summary entry to history
        self.add_to_history("Applied all study criteria")
        
        # Return all results for metadata
        return {
            'inclusion': inclusion_results,
            'exclusion': exclusion_results,
            'enrollment': enrollment_results,
            'attrition': attrition_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def apply_and_save_to_study(self):
        """Apply all criteria and save the resulting dataset to the study"""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Warning", "No dataset selected")
            return
            
        # Apply all criteria and get results for metadata
        criteria_results = self.apply_all_criteria()
        
        # Get a name for the new dataset
        default_name = f"{self.current_name}_filtered"
        new_name, ok = QInputDialog.getText(self, 
                                           "Save Filtered Dataset", 
                                           "Enter name for filtered dataset:", 
                                           QLineEdit.EchoMode.Normal, 
                                           default_name)
        
        if not ok or not new_name:
            return
            
        # Create metadata for the filtered dataset
        metadata = {
            'source_dataset': self.current_name,
            'filter_criteria': criteria_results,
            'created_at': datetime.now().isoformat(),
            'description': f"Dataset filtered using study criteria from {self.current_name}",
            'rows_original': len(self.current_dataframe),
            'rows_filtered': len(self._preview_df)
        }
        
        # Save the dataset to the study
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            # Add to the studies manager
            success = main_window.studies_manager.add_dataset_to_active_study(
                new_name, 
                self._preview_df.copy(), 
                metadata
            )
            
            if success:
                # Add to our internal sources
                self.add_source(new_name, self._preview_df.copy())
                
                # Update the dataset selection combo
                self.refresh_study_datasets_combo()
                
                QMessageBox.information(self, "Success", f"Dataset '{new_name}' saved to study")
                self.status_bar.showMessage(f"Dataset saved as '{new_name}' in active study")
            else:
                QMessageBox.warning(self, "Error", "Failed to save dataset to study")
        else:
            QMessageBox.warning(self, "Error", "Could not access studies manager")
    
    # ---------------------------------------------------------------
    # Create tab for slicing history
    # ---------------------------------------------------------------
    def create_history_tab(self):
        """Create the history tab to track slice operations"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel("Track all slicing operations performed on your dataset.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)
        
        # History display
        self.history_text = QPlainTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText("No slicing operations performed yet.")
        layout.addWidget(self.history_text)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        export_button = QPushButton("Export Slicing Report")
        export_button.setIcon(load_bootstrap_icon("file-earmark-arrow-down"))
        export_button.clicked.connect(self.export_slicing_report)
        buttons_layout.addWidget(export_button)
        
        reset_button = QPushButton("Reset Dataset")
        reset_button.setIcon(load_bootstrap_icon("arrow-counterclockwise"))
        reset_button.clicked.connect(self.reset_dataset)
        buttons_layout.addWidget(reset_button)
        
        layout.addLayout(buttons_layout)
        
        return tab
    
    # ---------------------------------------------------------------
    # Methods for slicing history
    # ---------------------------------------------------------------
    def add_to_history(self, operation):
        """Add an operation to the slicing history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.slicing_history.append(f"[{timestamp}] {operation}")
        self.update_history()
    
    def update_history(self):
        """Update the history display"""
        if not self.slicing_history:
            self.history_text.setPlainText("No slicing operations performed yet.")
        else:
            self.history_text.setPlainText("\n".join(self.slicing_history))
    
    def export_slicing_report(self):
        """Export the slicing history to a text file"""
        if not self.slicing_history:
            QMessageBox.information(self, "Information", "No slicing operations to export")
            return
        
        # Get file name for saving
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Slicing Report", f"{self.current_name}_slicing_report.txt", "Text Files (*.txt)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, "w") as f:
                f.write(f"Data Slicing Report for: {self.current_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Slicing Operations:\n")
                for operation in self.slicing_history:
                    f.write(f"{operation}\n")
                
                # Add dataset summary
                f.write("\n\nDataset Summary:\n")
                rows, cols = self._preview_df.shape
                f.write(f"Rows: {rows}, Columns: {cols}\n")
                f.write("Columns:\n")
                for col in self._preview_df.columns:
                    f.write(f"  {col}\n")
            
            QMessageBox.information(self, "Success", f"Slicing report saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report: {str(e)}")

    def create_ordinal_encoding_tab(self):
        """Create the tab for ordinal encoding"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Instructions
        instructions = QLabel("Encode ordinal columns by mapping values to integers.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #666;")
        layout.addWidget(instructions)

        # Dataset and column selection
        selection_group = QGroupBox("Dataset & Column Selection")
        selection_layout = QGridLayout(selection_group)

        selection_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.ordinal_dataset_combo = QComboBox()
        self.ordinal_dataset_combo.currentIndexChanged.connect(self.on_ordinal_dataset_changed)
        selection_layout.addWidget(self.ordinal_dataset_combo, 0, 1, 1, 2)

        selection_layout.addWidget(QLabel("Column:"), 1, 0)
        self.ordinal_column_combo = QComboBox()
        self.ordinal_column_combo.currentIndexChanged.connect(self.on_ordinal_column_changed)
        selection_layout.addWidget(self.ordinal_column_combo, 1, 1, 1, 2)

        layout.addWidget(selection_group)

        # Mapping table
        mapping_group = QGroupBox("Ordinal Mapping")
        mapping_layout = QVBoxLayout(mapping_group)

        self.ordinal_mapping_table = QTableWidget()
        self.ordinal_mapping_table.setColumnCount(2)
        self.ordinal_mapping_table.setHorizontalHeaderLabels(["Value", "Integer"])
        self.ordinal_mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ordinal_mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        mapping_layout.addWidget(self.ordinal_mapping_table)

        # Buttons
        buttons_layout = QHBoxLayout()
        auto_generate_button = QPushButton("Auto Generate Mapping")
        auto_generate_button.setIcon(load_bootstrap_icon("magic"))
        auto_generate_button.clicked.connect(self.auto_generate_ordinal_mapping)
        buttons_layout.addWidget(auto_generate_button)

        apply_encoding_button = QPushButton("Apply Encoding")
        apply_encoding_button.setIcon(load_bootstrap_icon("123"))
        apply_encoding_button.clicked.connect(self.apply_ordinal_encoding)
        buttons_layout.addWidget(apply_encoding_button)

        reset_mapping_button = QPushButton("Reset Mapping")
        reset_mapping_button.setIcon(load_bootstrap_icon("arrow-counterclockwise"))
        reset_mapping_button.clicked.connect(self.reset_ordinal_mapping)
        buttons_layout.addWidget(reset_mapping_button)

        mapping_layout.addLayout(buttons_layout)
        layout.addWidget(mapping_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        self.ordinal_result_label = QLabel("No encoding applied yet")
        results_layout.addWidget(self.ordinal_result_label)
        layout.addWidget(results_group)

        layout.addStretch()

        return tab

    def on_ordinal_dataset_changed(self, index):
        """Update column combo box when dataset changes"""
        self.ordinal_column_combo.clear()
        if index < 0:
            return

        dataset_name = self.ordinal_dataset_combo.currentText()
        if dataset_name in self.dataframes:
            df = self.dataframes[dataset_name]
            self.ordinal_column_combo.addItems(df.columns)

    def on_ordinal_column_changed(self, index):
        """Populate mapping table when column changes"""
        self.ordinal_mapping_table.clearContents()
        self.ordinal_mapping_table.setRowCount(0)

        if index < 0:
            return

        dataset_name = self.ordinal_dataset_combo.currentText()
        column_name = self.ordinal_column_combo.currentText()

        if dataset_name in self.dataframes:
            df = self.dataframes[dataset_name]
            if column_name in df.columns:
                # Check if there's an existing mapping
                main_window = self.window()
                if hasattr(main_window, 'studies_manager'):
                    mapping = main_window.studies_manager.get_ordinal_encoding(dataset_name, column_name)
                    if mapping:
                        # Populate table with existing mapping
                        value_to_int = mapping['value_to_int']
                        self.ordinal_mapping_table.setRowCount(len(value_to_int))
                        for row, (value, integer) in enumerate(value_to_int.items()):
                            value_item = QTableWidgetItem(str(value))
                            integer_item = QTableWidgetItem(str(integer))
                            self.ordinal_mapping_table.setItem(row, 0, value_item)
                            self.ordinal_mapping_table.setItem(row, 1, integer_item)
                    else:
                        # Populate table with unique values from the column
                        unique_values = df[column_name].unique()
                        self.ordinal_mapping_table.setRowCount(len(unique_values))
                        for row, value in enumerate(unique_values):
                            value_item = QTableWidgetItem(str(value))
                            self.ordinal_mapping_table.setItem(row, 0, value_item)

    def auto_generate_ordinal_mapping(self):
        """Auto-generate mapping based on sorted unique values"""
        dataset_name = self.ordinal_dataset_combo.currentText()
        column_name = self.ordinal_column_combo.currentText()

        if dataset_name in self.dataframes and column_name in self.dataframes[dataset_name].columns:
            df = self.dataframes[dataset_name]
            unique_values = sorted(df[column_name].unique())
            self.ordinal_mapping_table.setRowCount(len(unique_values))
            for row, value in enumerate(unique_values):
                value_item = QTableWidgetItem(str(value))
                integer_item = QTableWidgetItem(str(row))  # Auto-generated integer
                self.ordinal_mapping_table.setItem(row, 0, value_item)
                self.ordinal_mapping_table.setItem(row, 1, integer_item)

    def apply_ordinal_encoding(self):
        """Apply the ordinal encoding to the selected column"""
        dataset_name = self.ordinal_dataset_combo.currentText()
        column_name = self.ordinal_column_combo.currentText()

        if dataset_name not in self.dataframes or column_name not in self.dataframes[dataset_name].columns:
            QMessageBox.warning(self, "Warning", "Please select a valid dataset and column.")
            return

        # Collect mapping from table
        custom_mapping = {}
        for row in range(self.ordinal_mapping_table.rowCount()):
            value_item = self.ordinal_mapping_table.item(row, 0)
            integer_item = self.ordinal_mapping_table.item(row, 1)
            if value_item and integer_item:
                try:
                    value = value_item.text()
                    integer = int(integer_item.text())
                    custom_mapping[value] = integer
                except ValueError:
                    QMessageBox.warning(self, "Error", f"Invalid integer value in row {row + 1}.")
                    return

        if not custom_mapping:
            QMessageBox.warning(self, "Warning", "Please define a mapping.")
            return

        # Apply encoding using StudiesManager
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            encoded_series, mapping = main_window.studies_manager.encode_ordinal_column(
                dataset_name, column_name, custom_mapping
            )
            if encoded_series is not None:
                # Update the preview dataframe
                if dataset_name == self.current_name:
                    self._preview_df[column_name] = encoded_series
                    self.preview_display.display_dataframe(self._preview_df)

                    # Add to history
                    self.add_to_history(f"Applied ordinal encoding to '{column_name}' in '{dataset_name}'")
                    self.ordinal_result_label.setText(f"Ordinal encoding applied to '{column_name}'.")
                else:
                    # If encoding a different dataset, update the stored dataframe directly
                    self.dataframes[dataset_name][column_name] = encoded_series
                    # and add it to the studies manager
                    main_window.studies_manager.update_dataset_in_active_study(dataset_name, self.dataframes[dataset_name])
                    self.ordinal_result_label.setText(f"Ordinal encoding applied to '{column_name}' in dataset {dataset_name}.")
            else:
                QMessageBox.critical(self, "Error", "Failed to apply ordinal encoding.")
                self.ordinal_result_label.setText("Failed to apply ordinal encoding.")
        else:
            QMessageBox.warning(self, "Error", "Could not access Studies Manager.")

    def reset_ordinal_mapping(self):
        """Reset the ordinal mapping for the selected column"""
        dataset_name = self.ordinal_dataset_combo.currentText()
        column_name = self.ordinal_column_combo.currentText()

        if dataset_name not in self.dataframes or column_name not in self.dataframes[dataset_name].columns:
            QMessageBox.warning(self, "Warning", "Please select a valid dataset and column.")
            return

        # Clear mapping from StudiesManager metadata
        main_window = self.window()
        if hasattr(main_window, 'studies_manager'):
            metadata = main_window.studies_manager.get_dataset_metadata(dataset_name)
            if metadata and 'encodings' in metadata and column_name in metadata['encodings']:
                del metadata['encodings'][column_name]
                # Update the metadata in studies manager
                main_window.studies_manager.update_dataset_in_active_study(dataset_name, self.dataframes[dataset_name], metadata)

                # Reset the column in the preview dataframe to the original values
                if dataset_name == self.current_name:
                    self._preview_df[column_name] = self.current_dataframe[column_name]
                    self.preview_display.display_dataframe(self._preview_df)

                    # Add to history
                    self.add_to_history(f"Reset ordinal encoding for '{column_name}' in '{dataset_name}'")
                    self.ordinal_result_label.setText(f"Ordinal encoding reset for '{column_name}'.")
                else:
                    # If resetting a different dataset, update the stored dataframe directly
                    self.dataframes[dataset_name][column_name] = self.dataframes[dataset_name][column_name].copy()
                    # and add it to the studies manager
                    main_window.studies_manager.update_dataset_in_active_study(dataset_name, self.dataframes[dataset_name])
                    self.ordinal_result_label.setText(f"Ordinal encoding reset for '{column_name}' in dataset {dataset_name}.")

                # Clear the mapping table
                self.on_ordinal_column_changed(self.ordinal_column_combo.currentIndex())
            else:
                self.ordinal_result_label.setText("No mapping to reset.")
        else:
            QMessageBox.warning(self, "Error", "Could not access Studies Manager.")

    def on_ordinal_column_changed(self, index):
        """Populate mapping table when column changes"""
        self.ordinal_mapping_table.clearContents()
        self.ordinal_mapping_table.setRowCount(0)

        if index < 0:
            return

        dataset_name = self.ordinal_dataset_combo.currentText()
        column_name = self.ordinal_column_combo.currentText()

        if dataset_name in self.dataframes:
            df = self.dataframes[dataset_name]
            if column_name in df.columns:
                # Check if there's an existing mapping
                main_window = self.window()
                if hasattr(main_window, 'studies_manager'):
                    mapping = main_window.studies_manager.get_ordinal_encoding(dataset_name, column_name)
                    if mapping:
                        # Populate table with existing mapping
                        value_to_int = mapping['value_to_int']
                        self.ordinal_mapping_table.setRowCount(len(value_to_int))
                        for row, (value, integer) in enumerate(value_to_int.items()):
                            value_item = QTableWidgetItem(str(value))
                            integer_item = QTableWidgetItem(str(integer))
                            self.ordinal_mapping_table.setItem(row, 0, value_item)
                            self.ordinal_mapping_table.setItem(row, 1, integer_item)
                    else:
                        # Populate table with unique values from the column
                        unique_values = df[column_name].unique()
                        self.ordinal_mapping_table.setRowCount(len(unique_values))
                        for row, value in enumerate(unique_values):
                            value_item = QTableWidgetItem(str(value))
                            self.ordinal_mapping_table.setItem(row, 0, value_item)
