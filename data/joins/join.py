import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QComboBox, QFormLayout, QGridLayout, QGroupBox, QMessageBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSplitter,
    QDialog, QTabWidget, QListWidget, QPlainTextEdit, QDialogButtonBox,
    QFileDialog, QApplication, QMenu, QSizePolicy, QScrollArea, QToolButton
)
from PyQt6.QtGui import QIcon, QAction
import json
import re
from qasync import asyncSlot

from data.selection.masking_utils import get_column_mapping
from llms.client import call_llm_async_json
from helpers.load_icon import load_bootstrap_icon

class DataFrameDisplay(QTableWidget):
    """Table widget to display pandas DataFrames"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSectionsMovable(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
    def display_dataframe(self, df: pd.DataFrame):
        self.clear()
        if df is None or df.empty:
            print("Warning: Attempting to display None or empty DataFrame")
            return
            
        try:
            # Set up table dimensions
            rows_to_display = min(1000, len(df))
            self.setRowCount(rows_to_display)  # Limit to 1000 rows for performance
            self.setColumnCount(len(df.columns))
            
            # Set headers
            self.setHorizontalHeaderLabels(df.columns)
            
            # Add data
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= rows_to_display:  # Limit to 1000 rows
                    break
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    self.setItem(i, j, item)
            
            # Resize columns to content
            self.resizeColumnsToContents()
            
            print(f"Successfully displayed DataFrame with {rows_to_display} rows and {len(df.columns)} columns")
        except Exception as e:
            print(f"Error in display_dataframe: {str(e)}")
            # Clear the table on error
            self.clear()
            self.setRowCount(0)
            self.setColumnCount(0)

class DataJoinWidget(QWidget):
    """
    Enhanced widget for joining datasets together with various options.
    Integrates with studies manager to get available datasets.
    """
    
    join_completed = pyqtSignal(str, object)  # Signal emitted when join is completed (name, dataframe)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Join Operations")
        
        # Set size policies to prevent excessive stretching
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Dictionary to store available dataframes
        self.available_dataframes = {}  # {name: dataframe}
        
        # Store the result of the join operation
        self.result_df = None
        
        # Store state for stepwise relationship application
        self.applied_relationships = []  # List of applied relationships
        self.intermediate_datasets = {}  # Datasets created during stepwise application
        self.original_datasets = {}  # Original datasets before any relationships were applied
        self.relationship_history = []  # History of applied relationships for undo
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for left and right sides
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel with datasets viewer and join configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)
        
        left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Title
        title_label = QLabel("Join Operations")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(title_label)
        
        # Datasets panel
        datasets_group = QGroupBox("Available Datasets")
        datasets_layout = QVBoxLayout(datasets_group)
        datasets_layout.setSpacing(5)
        
        # Dataset list with search
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.dataset_search = QLineEdit()
        self.dataset_search.textChanged.connect(self.filter_datasets)
        search_layout.addWidget(self.dataset_search)
        datasets_layout.addLayout(search_layout)
        
        # Dataset list widget
        self.datasets_list = QListWidget()
        self.datasets_list.itemClicked.connect(self.on_dataset_clicked)
        self.datasets_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.datasets_list.customContextMenuRequested.connect(self.show_datasets_context_menu)
        self.datasets_list.setMinimumHeight(100)
        self.datasets_list.setMaximumHeight(200)
        datasets_layout.addWidget(self.datasets_list)
        
        # Add refresh button at the top of the sources list
        refresh_button = QPushButton("Refresh Datasets")
        refresh_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_button.clicked.connect(self.refresh_datasets)
        datasets_layout.addWidget(refresh_button)
        
        left_layout.addWidget(datasets_group)
        
        # Create a scrollable area for the configuration panels
        config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        config_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)
        config_layout.setSpacing(5)
        config_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        
        # Join configuration
        join_config_group = QGroupBox("Join Configuration")
        self.setup_join_config(join_config_group)
        config_layout.addWidget(join_config_group)
        
        # Join options
        join_options_group = QGroupBox("Join Options")
        self.setup_join_options(join_options_group)
        config_layout.addWidget(join_options_group)
        
        # Result options
        result_group = QGroupBox("Result Options")
        self.setup_result_options(result_group)
        config_layout.addWidget(result_group)
        
        # Execute button
        execute_button = QPushButton("Execute Join")
        execute_button.setIcon(load_bootstrap_icon("play-fill"))
        execute_button.clicked.connect(self.execute_join)
        config_layout.addWidget(execute_button)
        
        # Removed extra stretch to avoid an empty bottom section
        
        # Set the scrollable widget
        config_scroll.setWidget(config_container)
        left_layout.addWidget(config_scroll)
        
        # Add Left panel to splitter
        main_splitter.addWidget(left_panel)
        
        # Right panel for preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Preview title
        preview_title = QLabel("Join Preview")
        preview_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(preview_title)
        
        # Preview controls
        preview_controls = QHBoxLayout()
        self.preview_label = QLabel("No preview available")
        preview_controls.addWidget(self.preview_label)
        preview_controls.addStretch()
        
        preview_controls.addWidget(QLabel("Preview:"))
        self.preview_selector = QComboBox()
        self.preview_selector.addItems(["Result", "Left Table", "Right Table"])
        self.preview_selector.currentTextChanged.connect(self.update_preview)
        preview_controls.addWidget(self.preview_selector)
        
        right_layout.addLayout(preview_controls)
        
        # Preview table
        self.preview_table = DataFrameDisplay()
        self.preview_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.preview_table)
        
        # Add right panel to splitter
        main_splitter.addWidget(right_panel)
        
        layout.addWidget(main_splitter)
        main_splitter.setSizes([400, 600])
        
        # Status bar with fixed height to avoid extra space
        self.status_label = QLabel("Ready")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.status_label)
        
        self.setMinimumSize(900, 600)
        self.refresh_datasets()
        
    # Override sizeHint to give a reasonable default size
    def sizeHint(self):
        return QSize(900, 600)
    
    def setup_join_config(self, parent):
        """Setup the join configuration group"""
        layout = QGridLayout(parent)
        
        # Row 0: Left table and keys
        layout.addWidget(QLabel("Left Table:"), 0, 0)
        self.left_table_combo = QComboBox()
        self.left_table_combo.currentTextChanged.connect(self.update_left_keys)
        layout.addWidget(self.left_table_combo, 0, 1)
        
        layout.addWidget(QLabel("Left Key:"), 0, 2)
        self.left_key_combo = QComboBox()
        layout.addWidget(self.left_key_combo, 0, 3)
        
        self.left_multi_key_button = QPushButton("+ Add Key")
        self.left_multi_key_button.clicked.connect(lambda: self.add_join_key('left'))
        layout.addWidget(self.left_multi_key_button, 0, 4)
        
        # Row 1: Right table and keys
        layout.addWidget(QLabel("Right Table:"), 1, 0)
        self.right_table_combo = QComboBox()
        self.right_table_combo.currentTextChanged.connect(self.update_right_keys)
        layout.addWidget(self.right_table_combo, 1, 1)
        
        layout.addWidget(QLabel("Right Key:"), 1, 2)
        self.right_key_combo = QComboBox()
        layout.addWidget(self.right_key_combo, 1, 3)
        
        self.right_multi_key_button = QPushButton("+ Add Key")
        self.right_multi_key_button.clicked.connect(lambda: self.add_join_key('right'))
        layout.addWidget(self.right_multi_key_button, 1, 4)
        
        # Row 2: Join type
        layout.addWidget(QLabel("Join Type:"), 2, 0)
        self.join_type_combo = QComboBox()
        self.join_type_combo.addItems(["Inner", "Left", "Right", "Outer", "Cross"])
        self.join_type_combo.setCurrentIndex(1)  # Default to Left join
        layout.addWidget(self.join_type_combo, 2, 1)
        
        analyze_button = QPushButton("Analyze Relationships")
        analyze_button.setIcon(load_bootstrap_icon("search"))
        analyze_button.clicked.connect(self.identify_and_group_datasets)
        layout.addWidget(analyze_button, 2, 3, 1, 2)
        
        join_explanation = QLabel(
            "Inner: Only rows with matching keys in both tables\n"
            "Left: All rows from left table, matching rows from right\n"
            "Right: All rows from right table, matching rows from left\n"
            "Outer: All rows from both tables\n"
            "Cross: Cartesian product of rows from both tables"
        )
        join_explanation.setWordWrap(True)
        layout.addWidget(join_explanation, 3, 0, 1, 5)
        
        self.left_keys = []
        self.right_keys = []
    
    def setup_join_options(self, parent):
        """Setup additional join options"""
        layout = QFormLayout(parent)
        
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Left Suffix:"))
        self.left_suffix_input = QLineEdit("_x")
        suffix_layout.addWidget(self.left_suffix_input)
        
        suffix_layout.addWidget(QLabel("Right Suffix:"))
        self.right_suffix_input = QLineEdit("_y")
        suffix_layout.addWidget(self.right_suffix_input)
        
        layout.addRow("Column Suffixes:", suffix_layout)
        
        self.indicator_checkbox = QCheckBox("Add indicator column")
        self.indicator_checkbox.setChecked(False)
        layout.addRow("Merge Indicator:", self.indicator_checkbox)
        
        validation_layout = QHBoxLayout()
        self.validate_checkbox = QCheckBox("Validate relationship:")
        self.validate_checkbox.setChecked(True)
        validation_layout.addWidget(self.validate_checkbox)
        
        self.validate_type_combo = QComboBox()
        self.validate_type_combo.addItems(["One-to-one (1:1)", "One-to-many (1:m)", "Many-to-one (m:1)", "Many-to-many (m:m)"])
        validation_layout.addWidget(self.validate_type_combo)
        
        self.validate_checkbox.stateChanged.connect(
            lambda state: self.validate_type_combo.setEnabled(state == Qt.CheckState.Checked)
        )
        
        layout.addRow("Validation:", validation_layout)
        
        self.sort_checkbox = QCheckBox("Sort by join keys")
        self.sort_checkbox.setChecked(True)
        layout.addRow("Sorting:", self.sort_checkbox)
    
    def setup_result_options(self, parent):
        """Setup result options"""
        layout = QFormLayout(parent)
        
        self.result_name_input = QLineEdit()
        self.result_name_input.setPlaceholderText("Enter name for result dataset")
        layout.addRow("Result Name:", self.result_name_input)
        
        self.auto_name_checkbox = QCheckBox("Auto-generate name based on tables")
        self.auto_name_checkbox.setChecked(True)
        self.auto_name_checkbox.stateChanged.connect(self.update_auto_name)
        layout.addRow("Auto Name:", self.auto_name_checkbox)
        
        column_layout = QHBoxLayout()
        self.include_all_checkbox = QCheckBox("Include all columns")
        self.include_all_checkbox.setChecked(True)
        column_layout.addWidget(self.include_all_checkbox)
        
        self.column_select_button = QPushButton("Select Columns")
        self.column_select_button.setIcon(load_bootstrap_icon("list-check"))
        self.column_select_button.setEnabled(False)
        self.column_select_button.clicked.connect(self.show_column_selection)
        column_layout.addWidget(self.column_select_button)
        
        self.include_all_checkbox.stateChanged.connect(
            lambda state: self.column_select_button.setEnabled(not state)
        )
        
        layout.addRow("Columns:", column_layout)
        
        self.auto_save_checkbox = QCheckBox("Add to available datasets")
        self.auto_save_checkbox.setChecked(True)
        layout.addRow("Save Result:", self.auto_save_checkbox)
        
        self.add_to_study_checkbox = QCheckBox("Add to active study")
        self.add_to_study_checkbox.setChecked(True)
        layout.addRow("Study Integration:", self.add_to_study_checkbox)
    
    def update_hierarchical_options(self, state):
        is_hierarchical = state == Qt.CheckState.Checked
        
        self.auto_save_checkbox.setEnabled(not is_hierarchical)
        self.include_all_checkbox.setEnabled(not is_hierarchical and not self.include_all_checkbox.isChecked())
        self.column_select_button.setEnabled(not is_hierarchical and not self.include_all_checkbox.isChecked())
        
        if is_hierarchical:
            self.add_to_study_checkbox.setChecked(True)
        
        self.update_preview()
    
    def filter_datasets(self, search_text):
        for i in range(self.datasets_list.count()):
            item = self.datasets_list.item(i)
            item.setHidden(search_text.lower() not in item.text().lower())
    
    def on_dataset_clicked(self, item):
        dataset_name = item.text()
        if dataset_name in self.available_dataframes:
            # Preview the dataset immediately when clicked
            self.preview_dataset(dataset_name)
            self.preview_selector.setCurrentText("Result" if self.result_df is not None else "Left Table")
            
            # Existing functionality to set as left/right table
            left_selected = self.left_table_combo.currentText()
            right_selected = self.right_table_combo.currentText()
            
            if not left_selected:
                self.left_table_combo.setCurrentText(dataset_name)
            elif not right_selected:
                self.right_table_combo.setCurrentText(dataset_name)
            else:
                if self.preview_selector.currentText() == "Left Table":
                    self.left_table_combo.setCurrentText(dataset_name)
                else:
                    self.right_table_combo.setCurrentText(dataset_name)
            self.update_preview()
    
    def show_datasets_context_menu(self, position):
        menu = QMenu()
        item = self.datasets_list.itemAt(position)
        if item:
            dataset_name = item.text()
            set_left_action = menu.addAction(load_bootstrap_icon("arrow-left"), "Set as Left Table")
            set_left_action.triggered.connect(lambda: self.left_table_combo.setCurrentText(dataset_name))
            
            set_right_action = menu.addAction(load_bootstrap_icon("arrow-right"), "Set as Right Table")
            set_right_action.triggered.connect(lambda: self.right_table_combo.setCurrentText(dataset_name))
            
            menu.addSeparator()
            
            preview_action = menu.addAction(load_bootstrap_icon("eye"), "Preview Dataset")
            preview_action.triggered.connect(lambda: self.preview_dataset(dataset_name))
            
            menu.addSeparator()
            
            info_action = menu.addAction(load_bootstrap_icon("info-circle"), "Dataset Information")
            info_action.triggered.connect(lambda: self.show_dataset_info(dataset_name))
            
            menu.exec(self.datasets_list.mapToGlobal(position))
    
    def preview_dataset(self, dataset_name):
        if dataset_name in self.available_dataframes:
            df = self.available_dataframes[dataset_name]
            self.preview_table.display_dataframe(df)
            rows, cols = df.shape
            self.preview_label.setText(f"Preview: {dataset_name} - {rows} rows, {cols} columns")
            
    def show_dataset_info(self, dataset_name):
        if dataset_name in self.available_dataframes:
            df = self.available_dataframes[dataset_name]
            rows, cols = df.shape
            info = f"Dataset: {dataset_name}\nRows: {rows}, Columns: {cols}\n\nColumns:\n"
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isna().sum()
                null_percent = (null_count / rows) * 100
                unique_count = df[col].nunique()
                info += f"- {col} ({dtype})\n"
                info += f"  * Unique values: {unique_count}\n"
                info += f"  * Null values: {null_count} ({null_percent:.1f}%)\n"
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle(f"Dataset Information: {dataset_name}")
            msg_box.setText(info)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.exec()
    
    def update_left_keys(self, table_name):
        self.left_key_combo.clear()
        self.left_keys = []
        if not table_name or table_name not in self.available_dataframes:
            return
        df = self.available_dataframes[table_name]
        for column in df.columns:
            self.left_key_combo.addItem(column)
        self.update_auto_name()
        self.update_preview()
    
    def update_right_keys(self, table_name):
        self.right_key_combo.clear()
        self.right_keys = []
        if not table_name or table_name not in self.available_dataframes:
            return
        df = self.available_dataframes[table_name]
        for column in df.columns:
            self.right_key_combo.addItem(column)
        self.update_auto_name()
        self.update_preview()
    
    def add_join_key(self, side):
        if side == 'left':
            key = self.left_key_combo.currentText()
            if key and key not in self.left_keys:
                self.left_keys.append(key)
                self.update_status(f"Added left key: {key}")
        else:
            key = self.right_key_combo.currentText()
            if key and key not in self.right_keys:
                self.right_keys.append(key)
                self.update_status(f"Added right key: {key}")
        self.update_preview()
    
    def update_auto_name(self):
        if not self.auto_name_checkbox.isChecked():
            return
        left_table = self.left_table_combo.currentText()
        right_table = self.right_table_combo.currentText()
        join_type = self.join_type_combo.currentText().lower()
        if left_table and right_table:
            self.result_name_input.setText(f"{left_table}_{right_table}_{join_type}_join")
    
    def show_column_selection(self):
        left_table = self.left_table_combo.currentText()
        right_table = self.right_table_combo.currentText()
        
        if not left_table or not right_table:
            QMessageBox.warning(self, "Error", "Please select both tables first")
            return
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Columns")
        dialog.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(dialog)
        column_list = QListWidget()
        column_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        left_df = self.available_dataframes[left_table]
        right_df = self.available_dataframes[right_table]
        
        for col in left_df.columns:
            item = QListWidgetItem(f"{left_table}: {col}")
            item.setData(Qt.ItemDataRole.UserRole, (left_table, col))
            column_list.addItem(item)
            item.setSelected(True)
        
        for col in right_df.columns:
            if col not in self.right_keys:
                item = QListWidgetItem(f"{right_table}: {col}")
                item.setData(Qt.ItemDataRole.UserRole, (right_table, col))
                column_list.addItem(item)
                item.setSelected(True)
        
        layout.addWidget(column_list)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_columns = []
            for i in range(column_list.count()):
                item = column_list.item(i)
                if item.isSelected():
                    table, col = item.data(Qt.ItemDataRole.UserRole)
                    self.selected_columns.append((table, col))
            self.update_status(f"Selected {len(self.selected_columns)} columns for result")
        else:
            self.selected_columns = None
    
    def execute_join(self):
        left_table = self.left_table_combo.currentText()
        right_table = self.right_table_combo.currentText()
        
        if not left_table or not right_table:
            QMessageBox.warning(self, "Error", "Please select both tables")
            return
        
        if left_table not in self.available_dataframes or right_table not in self.available_dataframes:
            QMessageBox.warning(self, "Error", "One or both of the selected tables are not available")
            return
        
        left_key = self.left_key_combo.currentText() if not self.left_keys else self.left_keys
        right_key = self.right_key_combo.currentText() if not self.right_keys else self.right_keys
        
        if not left_key or not right_key:
            QMessageBox.warning(self, "Error", "Please select keys for both tables")
            return
        
        join_type = self.join_type_combo.currentText().lower()
        left_suffix = self.left_suffix_input.text()
        right_suffix = self.right_suffix_input.text()
        indicator = self.indicator_checkbox.isChecked()
        validate = self.validate_checkbox.isChecked()
        
        validate_type = None
        if validate:
            validate_type_text = self.validate_type_combo.currentText()
            if "1:1" in validate_type_text:
                validate_type = "1:1"
            elif "1:m" in validate_type_text:
                validate_type = "1:m"
            elif "m:1" in validate_type_text:
                validate_type = "m:1"
        
        sort = self.sort_checkbox.isChecked()
        result_name = self.result_name_input.text()
        
        if not result_name:
            QMessageBox.warning(self, "Error", "Please enter a name for the result dataset")
            return
        
        left_df = self.available_dataframes[left_table]
        right_df = self.available_dataframes[right_table]

        try:
            self.update_status(f"Executing {join_type} join...")
            print(f"Join inputs: {left_table}[{left_key}] {join_type} join {right_table}[{right_key}]")
            
            result_df = pd.merge(
                left_df, right_df,
                left_on=left_key, right_on=right_key,
                how=join_type,
                suffixes=(left_suffix, right_suffix),
                indicator=indicator,
                validate=validate_type if validate else None,
                sort=sort
            )
            
            if not self.include_all_checkbox.isChecked() and hasattr(self, 'selected_columns') and self.selected_columns:
                # Future implementation: filter columns based on self.selected_columns
                pass
            
            self.result_df = result_df.copy()
            print(f"Join result: {len(self.result_df)} rows, {len(self.result_df.columns)} columns")
            
            self.preview_selector.blockSignals(True)
            self.preview_selector.setCurrentText("Result")
            self.preview_selector.blockSignals(False)
            self.update_preview()
            
            if self.auto_save_checkbox.isChecked():
                self.join_completed.emit(result_name, self.result_df)
                self.available_dataframes[result_name] = self.result_df
                current_left = self.left_table_combo.currentText()
                current_right = self.right_table_combo.currentText()
                self.left_table_combo.addItem(current_left)
                self.right_table_combo.addItem(current_right)
                self.left_table_combo.setCurrentText(current_left)
                self.right_table_combo.setCurrentText(current_right)
                self.refresh_datasets_list()
            
            if self.add_to_study_checkbox.isChecked():
                self.add_to_active_study(result_name, self.result_df)
            
            rows, cols = self.result_df.shape
            QMessageBox.information(self, "Join Complete", f"Join operation successful.\nResult: {rows} rows, {cols} columns")
            self.update_status(f"Join completed - {len(self.result_df)} rows")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Join operation failed: {str(e)}")
            self.update_status(f"Join failed: {str(e)}")
            
            if "not a one-to-one merge" in str(e):
                QMessageBox.information(
                    self, 
                    "Suggestion",
                    "The keys are not unique for a one-to-one merge.\n\n"
                    "Try changing the validation type to 'One-to-many' or 'Many-to-one' "
                    "depending on your data relationship."
                )
    
    def add_to_active_study(self, dataset_name, dataframe, metadata=None):
        """Add a dataset to the active study via the studies manager"""
        try:
            main_window = self.window()
            if hasattr(main_window, 'studies_manager'):
                if metadata is None:
                    metadata = {
                        'source': 'join_operation',
                        'join_type': self.join_type_combo.currentText().lower(),
                        'left_table': self.left_table_combo.currentText(),
                        'right_table': self.right_table_combo.currentText(),
                        'left_key': self.left_key_combo.currentText() if not self.left_keys else self.left_keys,
                        'right_key': self.right_key_combo.currentText() if not self.right_keys else self.right_keys,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Ensure we have an active study
                active_study = main_window.studies_manager.get_active_study()
                if not active_study:
                    QMessageBox.warning(self, "No Active Study", 
                                      "No active study found. Please create or select a study first.")
                    self.update_status("Failed to add dataset: No active study")
                    return False
                
                # Add the dataset to the study
                success = main_window.studies_manager.add_dataset_to_active_study(
                    dataset_name, dataframe, metadata
                )
                
                if success:
                    self.update_status(f"Added {dataset_name} to active study")
                    # Force a refresh of the studies manager UI if possible
                    if hasattr(main_window, 'refresh_studies_ui'):
                        main_window.refresh_studies_ui()
                    elif hasattr(main_window.studies_manager, 'refresh_ui'):
                        main_window.studies_manager.refresh_ui()
                    return True
                else:
                    QMessageBox.warning(self, "Failed to Add Dataset", 
                                      f"Failed to add dataset '{dataset_name}' to the active study.")
                    self.update_status("Failed to add dataset to active study")
                    return False
            else:
                QMessageBox.warning(self, "No Studies Manager", 
                                  "Studies manager not available. Dataset was saved locally only.")
                self.update_status("No studies manager found, dataset not added to study")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error adding dataset to study: {str(e)}")
            self.update_status(f"Error adding to study: {str(e)}")
            return False
    
    def update_preview(self, force_update=False):
        preview_type = self.preview_selector.currentText()
        
        if preview_type == "Result" and self.result_df is not None:
            try:
                self.preview_table.display_dataframe(self.result_df)
                rows, cols = self.result_df.shape
                self.preview_label.setText(f"Join Result: {rows} rows, {cols} columns")
                print(f"Displaying join result with {rows} rows and {cols} columns")
            except Exception as e:
                print(f"Error displaying join result: {str(e)}")
                self.update_status(f"Error displaying join result: {str(e)}")
                self.preview_table.clear()
                self.preview_label.setText("Error displaying result")
        elif preview_type == "Left Table":
            left_table = self.left_table_combo.currentText()
            if left_table and left_table in self.available_dataframes:
                df = self.available_dataframes[left_table]
                self.preview_table.display_dataframe(df)
                rows, cols = df.shape
                self.preview_label.setText(f"Left Table: {left_table} - {rows} rows, {cols} columns")
            else:
                self.preview_table.clear()
                self.preview_label.setText("Left Table: Not selected")
        elif preview_type == "Right Table":
            right_table = self.right_table_combo.currentText()
            if right_table and right_table in self.available_dataframes:
                df = self.available_dataframes[right_table]
                self.preview_table.display_dataframe(df)
                rows, cols = df.shape
                self.preview_label.setText(f"Right Table: {right_table} - {rows} rows, {cols} columns")
            else:
                self.preview_table.clear()
                self.preview_label.setText("Right Table: Not selected")
    
    def update_status(self, message):
        self.status_label.setText(message)
        print(f"Join status: {message}")

    async def generate_dataset_metadata(self, name):
        if name not in self.available_dataframes:
            QMessageBox.warning(self, "Error", f"Dataset '{name}' not found")
            return
        
        dataframe = self.available_dataframes[name]
        masked_mapping = get_column_mapping(dataframe)
        self.update_status(f"Generated metadata for dataset: {name}")
        return masked_mapping

    @asyncSlot()
    async def identify_and_group_datasets(self):
        if not self.available_dataframes:
            QMessageBox.warning(self, "No Datasets", "No datasets available to analyze")
            return

        waiting_msg = QMessageBox(self)
        waiting_msg.setWindowTitle("Analyzing Datasets")
        waiting_msg.setText("Analyzing datasets to identify relationships...")
        waiting_msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        waiting_msg.show()

        try:
            datasets_info = {}
            for name, df in self.available_dataframes.items():
                masked_mapping = get_column_mapping(df)
                datasets_info[name] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "column_mapping": masked_mapping
                }
            prompt = self.create_dataset_analysis_prompt(datasets_info)
            response = await call_llm_async_json(prompt)
            
            if isinstance(response, dict):
                grouping_result = response
                waiting_msg.accept()
                self.show_dataset_grouping_results(grouping_result)
            else:
                try:
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    grouping_result = json.loads(clean_response)
                    waiting_msg.accept()
                    self.show_dataset_grouping_results(grouping_result)
                except json.JSONDecodeError:
                    waiting_msg.accept()
                    json_match = re.search(r'({[\s\S]*})', clean_response)
                    if json_match:
                        try:
                            grouping_result = json.loads(json_match.group(1))
                            self.show_dataset_grouping_results(grouping_result)
                            return
                        except:
                            pass
                    self.show_raw_grouping_response(response)
            
        except Exception as e:
            waiting_msg.accept()
            QMessageBox.critical(self, "Error", f"Failed to analyze datasets: {str(e)}")
            print(f"Error analyzing datasets: {str(e)}")
    
    def create_dataset_analysis_prompt(self, datasets_info):
        prompt = """
        I have multiple datasets that may be related in a hierarchical or relational manner. Please analyze them to:
        1. Identify datasets that likely belong together in a relational structure
        2. Suggest logical groupings based on column relationships
        3. Identify potential join keys between datasets
        4. Determine if datasets should be merged directly or kept separate with relationship metadata
        
        Important: Not all datasets need to be merged into a single flat table. Some relationships may be hierarchical 
        (e.g., one-to-many or many-to-many) where merging would cause data duplication or loss of structure.
        
        Here are the datasets with masked column information to protect sensitive data:
        """
        for name, info in datasets_info.items():
            prompt += f"\n\nDATASET: {name}\n"
            prompt += f"Rows: {info['row_count']}, Columns: {info['column_count']}\n"
            prompt += "Columns:\n"
            for column, col_info in info["column_mapping"]["column_mappings"].items():
                prompt += f"- {column} ({col_info['type']})\n"
                prompt += f"  * Unique values: {col_info['unique_count']}\n"
                prompt += f"  * Null values: {col_info['null_count']} ({col_info['null_percentage']:.1f}%)\n"
                if "value_distribution" in col_info:
                    if isinstance(col_info["value_distribution"], list):
                        prompt += f"  * Example masked values: {', '.join(col_info['value_distribution'])}\n"
                    else:
                        dist = col_info["value_distribution"]
                        if isinstance(dist, dict):
                            if "first_values" in dist:
                                prompt += f"  * First examples (masked): {', '.join(dist['first_values'])}\n"
                            if "last_values" in dist:
                                prompt += f"  * Last examples (masked): {', '.join(dist['last_values'])}\n"
                            if "min" in dist and dist["min"] is not None:
                                prompt += f"  * Range: {dist.get('min')} to {dist.get('max')}, Mean: {dist.get('mean')}\n"
        prompt += """
        
        Analyze the datasets and suggest:
        
        1. GROUPINGS: Which datasets appear to belong together and why?
           - Use naming patterns, column structures, and potential join keys
           - Consider normalized data patterns (e.g., "patients" table, "visits" table)
           
        2. JOIN RELATIONSHIPS: For each grouping, identify:
           - Primary/foreign key relationships between tables
           - Which columns should be used to join the datasets
           - The cardinality of relationships (one-to-one, one-to-many, many-to-many)
           - Whether datasets should be merged directly or kept separate with relationship metadata
           
        3. HIERARCHICAL STRUCTURE: If applicable, suggest:
           - Which datasets represent higher level entities
           - Which datasets represent child/detail records
           - How the hierarchy should be represented (e.g., parent-child relationships)
           
        4. DATA RESOLUTION: For each relationship:
           - Are the datasets at the same resolution/granularity?
           - Would merging cause data duplication or loss?
           - Recommend whether to merge or create a logical relationship
           
        5. CRITERIA: What criteria did you use to determine relationships?
           - Column name patterns matching across tables
           - ID columns with matching masked patterns
           - Cardinality of values suggesting relationships
           - Semantic relationships based on column names
           
        6. CONFIDENCE: For each suggested grouping, indicate your confidence level (high, medium, low)
        
        Return your response in JSON format:
        {
            "groups": [
                {
                    "name": "Group name (e.g., 'Patient Clinical Data')",
                    "datasets": ["dataset1", "dataset2"],
                    "relationships": [
                        {
                            "from_dataset": "dataset1",
                            "to_dataset": "dataset2",
                            "join_columns": ["dataset1.column_a", "dataset2.column_b"],
                            "cardinality": "one-to-many",
                            "merge_recommendation": "keep_separate", 
                            "reasoning": "Explanation of the relationship and merge recommendation"
                        }
                    ],
                    "hierarchical_structure": {
                        "parent_datasets": ["dataset1"],
                        "child_datasets": ["dataset2"],
                        "description": "Explanation of the hierarchical structure"
                    },
                    "confidence": "high/medium/low",
                    "reasoning": "Explanation of why these datasets are grouped"
                }
            ],
            "ungrouped_datasets": ["dataset3"],
            "criteria_used": ["List of criteria used to determine relationships"]
        }
        """
        return prompt

    def show_dataset_grouping_results(self, grouping_result):
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Relationship Analysis")
        dialog.setMinimumSize(900, 700)
        
        # Save the original state before any relationships are applied
        self.original_datasets = self.available_dataframes.copy()
        self.intermediate_datasets = {}
        self.applied_relationships = []
        self.relationship_history = []
        
        layout = QVBoxLayout(dialog)
        
        # Create a tab for the stepwise workflow view
        stepwise_widget = QWidget()
        stepwise_layout = QVBoxLayout(stepwise_widget)
        
        # Add section for relationship selection 
        rel_selection_group = QGroupBox("Available Relationships")
        rel_selection_layout = QVBoxLayout(rel_selection_group)
        
        # Table for all relationships from all groups
        all_relationships_table = QTableWidget()
        all_relationships_table.setColumnCount(6)
        all_relationships_table.setHorizontalHeaderLabels([
            "Group", "From Dataset", "To Dataset", "Join Columns", 
            "Cardinality", "Recommendation"
        ])
        
        # Collect all relationships across all groups
        row_count = 0
        for group in grouping_result.get("groups", []):
            for rel in group.get("relationships", []):
                row_count += 1
        
        all_relationships_table.setRowCount(row_count)
        
        row_idx = 0
        for group in grouping_result.get("groups", []):
            group_name = group.get("name", "Unnamed Group")
            for rel in group.get("relationships", []):
                all_relationships_table.setItem(row_idx, 0, QTableWidgetItem(group_name))
                all_relationships_table.setItem(row_idx, 1, QTableWidgetItem(rel.get("from_dataset", "")))
                all_relationships_table.setItem(row_idx, 2, QTableWidgetItem(rel.get("to_dataset", "")))
                all_relationships_table.setItem(row_idx, 3, QTableWidgetItem(", ".join(rel.get("join_columns", []))))
                all_relationships_table.setItem(row_idx, 4, QTableWidgetItem(rel.get("cardinality", "")))
                all_relationships_table.setItem(row_idx, 5, QTableWidgetItem(
                    rel.get("merge_recommendation", "").replace("_", " ").title()
                ))
                row_idx += 1
        
        all_relationships_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        all_relationships_table.setMinimumHeight(200)
        rel_selection_layout.addWidget(all_relationships_table)
        
        # Add button to apply relationship
        apply_button = QPushButton("Apply Selected Relationship")
        apply_button.setIcon(load_bootstrap_icon("link"))
        apply_button.clicked.connect(lambda: self.apply_stepwise_relationship(all_relationships_table, grouping_result, stepwise_result_view, applied_list, dialog))
        rel_selection_layout.addWidget(apply_button)
        
        stepwise_layout.addWidget(rel_selection_group)
        
        # Section for applied relationships and workflow
        workflow_group = QGroupBox("Applied Relationships (Workflow)")
        workflow_layout = QVBoxLayout(workflow_group)
        
        applied_list = QListWidget()
        applied_list.setMinimumHeight(100)
        workflow_layout.addWidget(applied_list)
        
        workflow_controls = QHBoxLayout()
        
        undo_button = QPushButton("Undo Last Step")
        undo_button.setIcon(load_bootstrap_icon("arrow-counterclockwise"))
        undo_button.clicked.connect(lambda: self.undo_last_relationship(applied_list, stepwise_result_view, dialog))
        undo_button.setEnabled(False)
        workflow_controls.addWidget(undo_button)
        
        reset_button = QPushButton("Reset to Original")
        reset_button.setIcon(load_bootstrap_icon("x-circle"))
        reset_button.clicked.connect(lambda: self.reset_relationships(applied_list, stepwise_result_view, dialog))
        reset_button.setEnabled(False)
        workflow_controls.addWidget(reset_button)
        
        save_final_button = QPushButton("Save Final Dataset")
        save_final_button.setIcon(load_bootstrap_icon("save"))
        save_final_button.clicked.connect(lambda: self.save_final_dataset(dialog))
        save_final_button.setEnabled(False)
        workflow_controls.addWidget(save_final_button)
        
        workflow_layout.addLayout(workflow_controls)
        stepwise_layout.addWidget(workflow_group)
        
        # Preview of the current dataset
        preview_group = QGroupBox("Current Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        stepwise_result_view = DataFrameDisplay()
        stepwise_result_view.setMinimumHeight(300)
        preview_layout.addWidget(stepwise_result_view)
        
        stepwise_layout.addWidget(preview_group)
        
        # Add to main layout directly without tabs
        layout.addWidget(stepwise_widget)
        
        # Display criteria used
        criteria_group = QGroupBox("Analysis Criteria")
        criteria_layout = QVBoxLayout(criteria_group)
        
        criteria_list = QListWidget()
        for criterion in grouping_result.get("criteria_used", []):
            criteria_list.addItem(criterion)
        criteria_layout.addWidget(criteria_list)
        
        layout.addWidget(criteria_group)
        
        button_layout = QHBoxLayout()
        
        create_views_button = QPushButton("Create Study Views")
        create_views_button.setIcon(load_bootstrap_icon("layout-text-window"))
        create_views_button.clicked.connect(lambda: self.create_study_views_from_groups(grouping_result))
        button_layout.addWidget(create_views_button)
        
        export_button = QPushButton("Export Analysis")
        export_button.setIcon(load_bootstrap_icon("file-earmark-arrow-down"))
        export_button.clicked.connect(lambda: self.export_grouping_analysis(grouping_result))
        button_layout.addWidget(export_button)
        
        close_button = QPushButton("Close")
        close_button.setIcon(load_bootstrap_icon("x-circle"))
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Set relationships between UI elements for enabling/disabling
        applied_list.itemSelectionChanged.connect(
            lambda: save_final_button.setEnabled(len(self.applied_relationships) > 0)
        )
        
        # Set references for callbacks
        dialog.undo_button = undo_button
        dialog.reset_button = reset_button
        dialog.save_final_button = save_final_button
        
        dialog.exec()
    
    def apply_stepwise_relationship(self, table, grouping_result, result_view, applied_list, dialog):
        """Apply a relationship in the stepwise workflow"""
        selected_rows = table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a relationship to apply")
            return
        
        row = selected_rows[0].row()
        group_name = table.item(row, 0).text()
        from_dataset = table.item(row, 1).text()
        to_dataset = table.item(row, 2).text()
        join_columns_text = table.item(row, 3).text()
        cardinality = table.item(row, 4).text()
        recommendation = table.item(row, 5).text().lower()
        
        join_columns = [col.strip() for col in join_columns_text.split(",")]
        left_key = []
        right_key = []
        
        for join_col in join_columns:
            if "." in join_col:
                parts = join_col.split(".")
                if parts[0] == from_dataset:
                    left_key.append(parts[1])
                elif parts[0] == to_dataset:
                    right_key.append(parts[1])
        
        if not left_key or not right_key:
            QMessageBox.warning(self, "Invalid Join", "Could not parse join columns properly")
            return
        
        # Determine if we need to use original or intermediate datasets
        left_df = None
        right_df = None
        
        # Find the most recent version of datasets
        if from_dataset in self.intermediate_datasets:
            left_df = self.intermediate_datasets[from_dataset]
        elif from_dataset in self.available_dataframes:
            left_df = self.available_dataframes[from_dataset]
        
        if to_dataset in self.intermediate_datasets:
            right_df = self.intermediate_datasets[to_dataset]
        elif to_dataset in self.available_dataframes:
            right_df = self.available_dataframes[to_dataset]
        
        if left_df is None or right_df is None:
            QMessageBox.warning(self, "Missing Dataset", 
                                f"Could not find datasets: {from_dataset if left_df is None else ''} "
                                f"{to_dataset if right_df is None else ''}")
            return
        
        # Determine appropriate join type based on cardinality
        join_type = "inner"
        if "one-to-many" in cardinality.lower():
            join_type = "left"
        elif "many-to-one" in cardinality.lower():
            join_type = "right"
        elif "many-to-many" in cardinality.lower():
            join_type = "outer"
        
        # Execute the join
        try:
            result_df = pd.merge(
                left_df, right_df,
                left_on=left_key, right_on=right_key,
                how=join_type,
                suffixes=('_x', '_y'),
                indicator=True
            )
            # Define result_name before using it
            result_name = f"{from_dataset}_{to_dataset}_{join_type}_join_{len(self.intermediate_datasets)}"
            self.intermediate_datasets[result_name] = result_df
            
            # Store the relationship for history
            relationship_info = {
                "result_name": result_name,
                "from_dataset": from_dataset,
                "to_dataset": to_dataset,
                "left_key": left_key,
                "right_key": right_key,
                "join_type": join_type,
                "cardinality": cardinality,
                "recommendation": recommendation
            }
            self.applied_relationships.append(relationship_info)
            self.relationship_history.append(relationship_info)
            
            # Update the applied relationships list
            applied_list.addItem(f"{len(self.applied_relationships)}. {from_dataset} {join_type} join {to_dataset} on {join_columns_text}")
            
            # Display the result in the preview
            result_view.display_dataframe(result_df)
            
            # Enable buttons
            dialog.undo_button.setEnabled(True)
            dialog.reset_button.setEnabled(True)
            dialog.save_final_button.setEnabled(True)
            
            QMessageBox.information(
                self, 
                "Relationship Applied", 
                f"Join between {from_dataset} and {to_dataset} applied successfully.\n\n"
                f"Result dataset '{result_name}' has {len(result_df)} rows."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply relationship: {str(e)}")
    
    def apply_suggested_relationship(self, table, group):
        """Apply a suggested relationship to the join configuration UI"""
        selected_rows = table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a relationship to apply")
            return
        
        row = selected_rows[0].row()
        from_dataset = table.item(row, 0).text()
        to_dataset = table.item(row, 1).text()
        join_columns_text = table.item(row, 2).text()
        cardinality = table.item(row, 3).text()
        recommendation = table.item(row, 4).text().lower()
        
        join_columns = [col.strip() for col in join_columns_text.split(",")]
        left_key = []
        right_key = []
        
        for join_col in join_columns:
            if "." in join_col:
                parts = join_col.split(".")
                if parts[0] == from_dataset:
                    left_key.append(parts[1])
                elif parts[0] == to_dataset:
                    right_key.append(parts[1])
        
        if not left_key or not right_key:
            QMessageBox.warning(self, "Invalid Join", "Could not parse join columns properly")
            return
        
        self.left_table_combo.setCurrentText(from_dataset)
        self.right_table_combo.setCurrentText(to_dataset)
        
        if left_key:
            self.left_key_combo.setCurrentText(left_key[0])
            self.left_keys = left_key[1:]
            
        if right_key:
            self.right_key_combo.setCurrentText(right_key[0])
            self.right_keys = right_key[1:]
        
        # Determine join type based on uniqueness of keys
        left_df = self.available_dataframes.get(from_dataset)
        right_df = self.available_dataframes.get(to_dataset)
        if left_df is not None and right_df is not None:
            left_key_val = left_key[0]
            right_key_val = right_key[0]
            left_unique = left_df[left_key_val].nunique() == len(left_df)
            right_unique = right_df[right_key_val].nunique() == len(right_df)
            
            if left_unique and right_unique:
                self.join_type_combo.setCurrentText("Inner")
            elif left_unique and not right_unique:
                self.join_type_combo.setCurrentText("Left")
            elif right_unique and not left_unique:
                self.join_type_combo.setCurrentText("Right")
            else:
                self.join_type_combo.setCurrentText("Outer")
        else:
            self.join_type_combo.setCurrentText("Inner")
        
        # Set hierarchical checkbox based on recommendation
        self.hierarchical_checkbox.setChecked("keep separate" in recommendation)
        
        result_name = f"{from_dataset}_{to_dataset}_join"
        self.result_name_input.setText(result_name)
        
        QMessageBox.information(
            self, 
            "Relationship Applied", 
            f"Relationship between {from_dataset} and {to_dataset} has been applied.\n\n"
            f"You can now review and execute the join operation."
        )
    
    def show_raw_grouping_response(self, response):
        """Show the raw LLM response when it can't be parsed as JSON"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Dataset Analysis Results (Raw)")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel("The analysis results could not be parsed as JSON. Here is the raw response:")
        layout.addWidget(label)
        
        text_view = QPlainTextEdit()
        text_view.setPlainText(response)
        text_view.setReadOnly(True)
        layout.addWidget(text_view)
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(response))
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(copy_button)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def undo_last_relationship(self, applied_list, result_view, dialog):
        """Undo the last applied relationship"""
        if not self.applied_relationships:
            return
        
        # Remove the last relationship
        self.applied_relationships.pop()
        
        # Clear the intermediate datasets and rebuild them
        self.intermediate_datasets = {}
        
        # Rebuild the intermediate datasets
        for rel in self.applied_relationships:
            from_dataset = rel["from_dataset"]
            to_dataset = rel["to_dataset"]
            join_type = rel["join_type"]
            left_key = rel["left_key"] if isinstance(rel["left_key"], list) else [rel["left_key"]]
            right_key = rel["right_key"] if isinstance(rel["right_key"], list) else [rel["right_key"]]
            join_columns = [f"{from_dataset}.{k}" for k in left_key] + [f"{to_dataset}.{k}" for k in right_key]
            
            # Get the most recent version of datasets
            left_df = None
            right_df = None
            
            if from_dataset in self.intermediate_datasets:
                left_df = self.intermediate_datasets[from_dataset]
            elif from_dataset in self.available_dataframes:
                left_df = self.available_dataframes[from_dataset]
            
            if to_dataset in self.intermediate_datasets:
                right_df = self.intermediate_datasets[to_dataset]
            elif to_dataset in self.available_dataframes:
                right_df = self.available_dataframes[to_dataset]
            
            if left_df is not None and right_df is not None:
                try:
                    result_df = pd.merge(
                        left_df, right_df,
                        left_on=left_key, right_on=right_key,
                        how=join_type,
                        suffixes=('_x', '_y'),
                        indicator=True
                    )
                    # Define result_name before using it
                    result_name = f"{from_dataset}_{to_dataset}_{join_type}_join_{len(self.intermediate_datasets)}"
                    self.intermediate_datasets[result_name] = result_df
                except Exception:
                    # If error, just skip this relationship
                    pass
        
        # Update the applied list
        applied_list.clear()
        for i, rel in enumerate(self.applied_relationships, 1):
            from_dataset = rel["from_dataset"]
            to_dataset = rel["to_dataset"]
            join_type = rel["join_type"]
            left_key = rel["left_key"] if isinstance(rel["left_key"], list) else [rel["left_key"]]
            right_key = rel["right_key"] if isinstance(rel["right_key"], list) else [rel["right_key"]]
            join_columns = [f"{from_dataset}.{k}" for k in left_key] + [f"{to_dataset}.{k}" for k in right_key]
            applied_list.addItem(f"{i}. {from_dataset} {join_type} join {to_dataset} on {', '.join(join_columns)}")
        
        # Update the preview
        if self.applied_relationships:
            last_result = self.applied_relationships[-1]["result_name"]
            if last_result in self.intermediate_datasets:
                result_view.display_dataframe(self.intermediate_datasets[last_result])
        else:
            result_view.clear()
            dialog.undo_button.setEnabled(False)
            dialog.reset_button.setEnabled(False)
            dialog.save_final_button.setEnabled(False)
    
    def reset_relationships(self, applied_list, result_view, dialog):
        """Reset all applied relationships"""
        # Clear all applied relationships
        self.applied_relationships = []
        self.intermediate_datasets = {}
        
        # Clear the UI
        applied_list.clear()
        result_view.clear()
        
        # Disable buttons
        dialog.undo_button.setEnabled(False)
        dialog.reset_button.setEnabled(False)
        dialog.save_final_button.setEnabled(False)
        
        QMessageBox.information(self, "Reset", "All applied relationships have been reset.")
    
    def save_final_dataset(self, dialog):
        """Save the final dataset after applying all relationships"""
        if not self.applied_relationships:
            QMessageBox.warning(self, "No Dataset", "No relationships have been applied.")
            return
        
        # Get the last result dataset
        last_result = self.applied_relationships[-1]["result_name"]
        if last_result not in self.intermediate_datasets:
            QMessageBox.warning(self, "Error", "Final dataset not found.")
            return
        
        final_df = self.intermediate_datasets[last_result]
        
        # Show dialog to name the dataset
        name_dialog = QDialog(self)
        name_dialog.setWindowTitle("Save Final Dataset")
        name_layout = QVBoxLayout(name_dialog)
        
        form_layout = QFormLayout()
        dataset_name_input = QLineEdit()
        dataset_name_input.setText(last_result)
        form_layout.addRow("Dataset Name:", dataset_name_input)
        
        # Options for saving
        save_options_group = QGroupBox("Save Options")
        save_options_layout = QVBoxLayout(save_options_group)
        
        add_to_study_checkbox = QCheckBox("Add to active study")
        add_to_study_checkbox.setChecked(True)
        save_options_layout.addWidget(add_to_study_checkbox)
        
        add_to_local_checkbox = QCheckBox("Add to local datasets")
        add_to_local_checkbox.setChecked(True)
        save_options_layout.addWidget(add_to_local_checkbox)
        
        # Warning about save options
        save_options_note = QLabel(
            "Note: At least one save option must be selected.\n"
            "Datasets saved only locally will not persist between sessions."
        )
        save_options_note.setWordWrap(True)
        save_options_note.setStyleSheet("color: #666; font-style: italic;")
        save_options_layout.addWidget(save_options_note)
        
        form_layout.addRow(save_options_group)
        
        # Add metadata fields
        metadata_group = QGroupBox("Dataset Metadata (Optional)")
        metadata_layout = QFormLayout(metadata_group)
        
        description_input = QPlainTextEdit()
        description_input.setMaximumHeight(80)
        description_input.setPlaceholderText("Optional description for the dataset")
        metadata_layout.addRow("Description:", description_input)
        
        tags_input = QLineEdit()
        tags_input.setPlaceholderText("comma,separated,tags")
        metadata_layout.addRow("Tags:", tags_input)
        
        form_layout.addRow(metadata_group)
        
        name_layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        # Add OK icon to buttons
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button:
            ok_button.setIcon(load_bootstrap_icon("check-circle"))
        
        # Add Cancel icon to buttons
        cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        if cancel_button:
            cancel_button.setIcon(load_bootstrap_icon("x-circle"))
            
        buttons.accepted.connect(name_dialog.accept)
        buttons.rejected.connect(name_dialog.reject)
        name_layout.addWidget(buttons)
        
        # Function to validate that at least one save option is selected
        def validate_save_options():
            if not add_to_study_checkbox.isChecked() and not add_to_local_checkbox.isChecked():
                QMessageBox.warning(name_dialog, "Invalid Options", 
                                  "At least one save option must be selected.")
                return False
            return True
        
        # Connect the accept signal to validate first
        name_dialog.accepted = lambda: validate_save_options()
        
        if name_dialog.exec() == QDialog.DialogCode.Accepted:
            final_name = dataset_name_input.text().strip()
            if not final_name:
                QMessageBox.warning(self, "Invalid Name", "Please enter a valid name for the dataset.")
                return
            
            # Create metadata about the stepwise process
            steps_metadata = []
            for i, rel in enumerate(self.applied_relationships, 1):
                steps_metadata.append({
                    "step": i,
                    "from_dataset": rel["from_dataset"],
                    "to_dataset": rel["to_dataset"],
                    "join_type": rel["join_type"],
                    "left_key": rel["left_key"],
                    "right_key": rel["right_key"],
                    "cardinality": rel["cardinality"]
                })
            
            # Add custom metadata
            description = description_input.toPlainText().strip()
            tags = [tag.strip() for tag in tags_input.text().split(',') if tag.strip()]
            
            metadata = {
                'source': 'stepwise_join_operation',
                'steps': steps_metadata,
                'timestamp': datetime.now().isoformat(),
                'description': description if description else None,
                'tags': tags if tags else []
            }
            
            # Track if at least one save operation succeeds
            save_success = False
            
            # Add to local datasets
            if add_to_local_checkbox.isChecked():
                self.available_dataframes[final_name] = final_df
                
                # Add to the available datasets in the main interface
                self.left_table_combo.addItem(final_name)
                self.right_table_combo.addItem(final_name)
                self.refresh_datasets_list()
                
                # Emit the signal that a join was completed
                self.join_completed.emit(final_name, final_df)
                save_success = True
            
            # Add to study if requested
            study_save_success = False
            if add_to_study_checkbox.isChecked():
                study_save_success = self.add_to_active_study(final_name, final_df, metadata)
                save_success = save_success or study_save_success
            
            if save_success:
                # Prepare success message
                if add_to_study_checkbox.isChecked() and add_to_local_checkbox.isChecked():
                    save_location = "both the active study and local datasets"
                elif add_to_study_checkbox.isChecked():
                    save_location = "the active study"
                else:
                    save_location = "local datasets"
                
                QMessageBox.information(
                    self, 
                    "Dataset Saved", 
                    f"Final dataset '{final_name}' has been saved to {save_location} with {len(final_df)} rows."
                )
                
                # Close the analysis dialog if requested
                close_dialog = QMessageBox.question(
                    self,
                    "Close Analysis",
                    "Would you like to close the analysis dialog?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if close_dialog == QMessageBox.StandardButton.Yes:
                    dialog.accept()
            else:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    "Failed to save the dataset. Please check the error messages."
                )
    
    def create_study_views_from_groups(self, grouping_result):
        try:
            main_window = self.window()
            if hasattr(main_window, 'studies_manager'):
                active_study = main_window.studies_manager.get_active_study()
                if active_study:
                    for group in grouping_result.get("groups", []):
                        group_name = group.get("name", "Unnamed Group")
                        datasets = group.get("datasets", [])
                        relationships = group.get("relationships", [])
                        
                        group_metadata = {
                            'type': 'dataset_group',
                            'name': group_name,
                            'datasets': datasets,
                            'relationships': relationships,
                            'hierarchical_structure': group.get("hierarchical_structure"),
                            'reasoning': group.get("reasoning"),
                            'confidence': group.get("confidence"),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        if not hasattr(active_study, 'datasets_metadata') or active_study.datasets_metadata is None:
                            active_study.datasets_metadata = {}
                        
                        if 'groups' not in active_study.datasets_metadata:
                            active_study.datasets_metadata['groups'] = []
                        
                        active_study.datasets_metadata['groups'].append(group_metadata)
                    
                    active_study.updated_at = datetime.now().isoformat()
                    
                    QMessageBox.information(
                        self, 
                        "Views Created", 
                        f"Dataset groups and relationships have been saved to the active study."
                    )
                else:
                    QMessageBox.warning(self, "Error", "No active study found")
            else:
                QMessageBox.information(
                    self,
                    "Feature Not Implemented",
                    "Creating study views from groupings is not yet fully implemented in this version."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create study views: {str(e)}")
    
    def export_grouping_analysis(self, grouping_result):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Analysis Results",
            "dataset_grouping_analysis.json",
            "JSON Files (*.json);;All Files (*)",
            options=options
        )
        if file_name:
            try:
                with open(file_name, 'w') as f:
                    json.dump(grouping_result, f, indent=2)
                QMessageBox.information(self, "Success", "Analysis results saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save analysis results: {str(e)}")
    
    def refresh_datasets(self):
        self.update_status("Refreshing datasets...")
        try:
            main_window = self.window()
            if hasattr(main_window, 'studies_manager') and hasattr(main_window.studies_manager, 'get_datasets_from_active_study'):
                datasets = main_window.studies_manager.get_datasets_from_active_study()
                self.available_dataframes = {name: df for name, df in datasets}
            else:
                if hasattr(main_window, 'data_collection_widget') and hasattr(main_window.data_collection_widget, 'dataframes'):
                    self.available_dataframes = main_window.data_collection_widget.dataframes
                else:
                    self.available_dataframes = {}
            
            self.left_table_combo.clear()
            self.right_table_combo.clear()
            
            for name in self.available_dataframes:
                self.left_table_combo.addItem(name)
                self.right_table_combo.addItem(name)
            
            if len(self.available_dataframes) >= 2:
                self.left_table_combo.setCurrentIndex(0)
                self.right_table_combo.setCurrentIndex(1)
                self.update_auto_name()
            
            self.refresh_datasets_list()
            self.update_status(f"Refreshed datasets - {len(self.available_dataframes)} available")
        except Exception as e:
            self.update_status(f"Error refreshing datasets: {str(e)}")
            print(f"Error refreshing datasets: {str(e)}")
    
    def refresh_datasets_list(self):
        self.datasets_list.clear()
        for name in sorted(self.available_dataframes.keys()):
            self.datasets_list.addItem(name)
        if self.dataset_search.text():
            self.filter_datasets(self.dataset_search.text())
    
    # Removed the forced resizeEvent override to avoid potential UI freeze issues.
    
    def showMaximized(self):
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        super().showMaximized()
