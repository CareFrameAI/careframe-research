import os
import json
import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal, QSize, QByteArray
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QLabel, QComboBox, QPlainTextEdit, QFormLayout, QSplitter, 
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QStatusBar, QRadioButton, QButtonGroup, QScrollArea, QApplication,
    QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QSizePolicy, QDialog, QFrame
)
from PyQt6.QtGui import QIcon, QColor, QBrush, QFont
from PyQt6.QtSvgWidgets import QSvgWidget

import re
import asyncio
from qasync import asyncSlot

# Add statistical analysis packages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# Add these imports for new models
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from data.selection.detailed_tests.formatting import PASTEL_COLORS, fig_to_svg
from llms.client import call_llm_async
from qt_sections.llm_manager import llm_config
from helpers.load_icon import load_bootstrap_icon

import io


class SensitivityAnalysisType(Enum):
    """Types of sensitivity analyses that can be performed."""
    MISSING_DATA = "Missing Data"
    OUTLIER_DETECTION = "Outlier Detection"
    ALTERNATIVE_MODEL = "Alternative Model"
    PARAMETER_VARIATION = "Parameter Variation"


class SensitivityResults:
    """Class to store and format sensitivity analysis results."""
    
    def __init__(self, analysis_type):
        self.analysis_type = analysis_type
        self.baseline_model = None
        self.sensitivity_models = []
        self.baseline_results = {}
        self.sensitivity_results = []
        self.summary = ""
        self.recommendation = ""
        self.robustness_assessment = ""
        self.visualization_data = {}
        
    def add_baseline_result(self, method, p_value, effect_size, interpretation):
        """Add baseline analysis results."""
        self.baseline_results = {
            "method": method,
            "p_value": p_value,
            "effect_size": effect_size,
            "interpretation": interpretation
        }
        
    def add_sensitivity_result(self, scenario, p_value, effect_size, change, significant_change):
        """Add a sensitivity analysis result."""
        self.sensitivity_results.append({
            "scenario": scenario,
            "p_value": p_value,
            "effect_size": effect_size,
            "change": change,
            "significant_change": significant_change
        })
        
    def set_summary(self, summary):
        """Set the summary text."""
        self.summary = summary
        
    def set_recommendation(self, recommendation):
        """Set the recommendation text."""
        self.recommendation = recommendation
        
    def set_robustness_assessment(self, assessment):
        """Set the robustness assessment text."""
        self.robustness_assessment = assessment
        
    def add_visualization_data(self, key, data):
        """Add data for visualization."""
        self.visualization_data[key] = data
    
    def format_summary(self):
        """Format a complete summary of sensitivity analysis results."""
        summary = f"# {self.analysis_type.value} Sensitivity Analysis Results\n\n"
        
        # Add baseline results
        summary += "## Baseline Results\n\n"
        summary += f"Method: {self.baseline_results.get('method', 'N/A')}\n"
        summary += f"p-value: {self.baseline_results.get('p_value', 'N/A')}\n"
        summary += f"Effect Size: {self.baseline_results.get('effect_size', 'N/A')}\n"
        summary += f"Interpretation: {self.baseline_results.get('interpretation', 'N/A')}\n\n"
        
        # Add sensitivity results
        summary += "## Sensitivity Analysis Results\n\n"
        summary += "| Scenario | p-value | Effect Size | Change | Significant Change |\n"
        summary += "|----------|---------|-------------|--------|-------------------|\n"
        
        for result in self.sensitivity_results:
            p_value = f"{result.get('p_value', 'N/A')}"
            effect_size = f"{result.get('effect_size', 'N/A')}"
            scenario = result.get('scenario', 'N/A')
            change = result.get('change', 'N/A')
            significant = "Yes" if result.get('significant_change', False) else "No"
            
            summary += f"| {scenario} | {p_value} | {effect_size} | {change} | {significant} |\n"
        
        summary += "\n"
        
        # Add robustness assessment
        if self.robustness_assessment:
            summary += f"## Robustness Assessment\n\n{self.robustness_assessment}\n\n"
        
        # Add summary
        if self.summary:
            summary += f"## Summary\n\n{self.summary}\n\n"
        
        # Add recommendation
        if self.recommendation:
            summary += f"## Recommendation\n\n{self.recommendation}\n\n"
        
        return summary

class SensitivitySettingsDialog(QDialog):
    """Dialog for sensitivity analysis settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensitivity Analysis Settings")
        
        layout = QVBoxLayout(self)
        
        # Create settings groups
        variable_group = QGroupBox("Variable Selection")
        variable_layout = QVBoxLayout(variable_group)
        
        self.manual_outcome_check = QCheckBox("Manually select outcome variable")
        self.manual_outcome_check.setToolTip("If checked, AI will suggest an outcome but you will have final choice")
        variable_layout.addWidget(self.manual_outcome_check)
        
        analysis_group = QGroupBox("Analysis Configuration")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.manual_analysis_type_check = QCheckBox("Manually select analysis type")
        self.manual_analysis_type_check.setToolTip("If checked, AI will suggest an analysis type but you will have final choice")
        analysis_layout.addWidget(self.manual_analysis_type_check)
        
        # Add groups to main layout
        layout.addWidget(variable_group)
        layout.addWidget(analysis_group)
        
        # Button row
        button_row = QHBoxLayout()
        button_row.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_row.addWidget(ok_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        
        layout.addLayout(button_row)
        
        # Set default size
        self.resize(400, 250)

class SensitivityAnalysisWidget(QWidget):
    """Widget for performing sensitivity analysis on statistical models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensitivity Analysis")
        
        # Internal state
        self.current_dataframe = None
        self.current_name = ""
        self.analysis_results = {}
        
        # Settings - simplified
        self.manual_outcome_selection = False
        self.manual_analysis_type_selection = False
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top section container
        top_section = QWidget()
        top_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(5, 5, 5, 5)
        top_layout.setSpacing(10)
        
        # Header row with dataset selection AND action buttons
        header_row = QHBoxLayout()
        header_row.setSpacing(10)
        
        # Left side: Load button and dataset selection
        load_button = QPushButton("Load")
        load_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        load_button.clicked.connect(self.load_dataset_from_study)
        header_row.addWidget(load_button)
        
        self.dataset_selector = QComboBox()
        self.dataset_selector.setMinimumWidth(250)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_changed)
        header_row.addWidget(self.dataset_selector)
        
        # Action buttons - simplified text
        self.select_vars_button = QPushButton("Select")
        self.select_vars_button.setIcon(load_bootstrap_icon("magic"))
        self.select_vars_button.setToolTip("Select Variables (AI)")
        self.select_vars_button.clicked.connect(self.auto_select_variables_clicked)
        header_row.addWidget(self.select_vars_button)
        
        self.config_analysis_button = QPushButton("Configure")
        self.config_analysis_button.setIcon(load_bootstrap_icon("sliders"))
        self.config_analysis_button.setToolTip("Configure Analysis (AI)")
        self.config_analysis_button.clicked.connect(self.auto_configure_analysis_options_clicked)
        header_row.addWidget(self.config_analysis_button)
        
        # Settings button
        self.settings_button = QPushButton("Settings")
        self.settings_button.setIcon(load_bootstrap_icon("gear"))
        self.settings_button.setToolTip("Analysis Settings")
        self.settings_button.clicked.connect(self.show_settings_dialog)
        header_row.addWidget(self.settings_button)
        
        # Clear button
        self.clear_all_button = QPushButton("Clear")
        self.clear_all_button.setIcon(load_bootstrap_icon("x"))
        self.clear_all_button.setToolTip("Clear All")
        self.clear_all_button.clicked.connect(self.clear_all)
        header_row.addWidget(self.clear_all_button)
        
        # add stretch to push remaining buttons to the right
        header_row.addStretch()
        
        # Run and Save buttons on the right
        self.run_analysis_button = QPushButton("Run")
        self.run_analysis_button.setIcon(load_bootstrap_icon("play-fill"))
        self.run_analysis_button.setToolTip("Run Sensitivity Analysis")
        self.run_analysis_button.clicked.connect(self.run_sensitivity_analysis)
        header_row.addWidget(self.run_analysis_button)
        
        self.save_results_button = QPushButton("Save")
        self.save_results_button.setIcon(load_bootstrap_icon("save"))
        self.save_results_button.setToolTip("Save Results to Study")
        self.save_results_button.clicked.connect(self.save_results_to_study)
        header_row.addWidget(self.save_results_button)
        
        # Add a new button to the header_row in init_ui method
        self.interpret_button = QPushButton("Interpret")
        self.interpret_button.setIcon(load_bootstrap_icon("chat"))
        self.interpret_button.setToolTip("Get AI Interpretation")
        self.interpret_button.clicked.connect(lambda: asyncio.create_task(self.share_and_display_interpretation()))
        header_row.addWidget(self.interpret_button)
        
        top_layout.addLayout(header_row)
        
        # Combined analysis settings group
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(8, 12, 8, 12)
        settings_layout.setSpacing(10)
        
        # Analysis type and variable selection in the same row
        analysis_vars_row = QHBoxLayout()
        analysis_vars_row.setSpacing(15)
        
        # Left side: Analysis type selection
        analysis_type_container = QWidget()
        analysis_type_layout = QVBoxLayout(analysis_type_container)
        analysis_type_layout.setContentsMargins(0, 0, 0, 0)
        analysis_type_layout.setSpacing(5)
        
        font = QFont()
        font.setBold(True)
        type_label = QLabel("Analysis Type:")
        type_label.setFont(font)  # Use the same bold font
        analysis_type_layout.addWidget(type_label)
        
        self.analysis_type_combo = QComboBox()
        # Make analysis type dropdown stand out
        large_font = QFont()
        large_font.setBold(True)
        large_font.setPointSize(font.pointSize() + 2)
        self.analysis_type_combo.setFont(large_font)
        self.analysis_type_combo.setMinimumHeight(30)
        
        for analysis_type in SensitivityAnalysisType:
            self.analysis_type_combo.addItem(analysis_type.value, analysis_type)
        self.analysis_type_combo.currentIndexChanged.connect(self.on_analysis_type_changed)
        analysis_type_layout.addWidget(self.analysis_type_combo)
        
        # Description of selected analysis
        self.analysis_description = QLabel()
        self.analysis_description.setWordWrap(True)
        self.analysis_description.setStyleSheet("font-style: italic; font-size: 11px;")
        self.analysis_description.setMaximumHeight(60)
        analysis_type_layout.addWidget(self.analysis_description)
        
        analysis_vars_row.addWidget(analysis_type_container, 2)
        
        # Right side: Variable selection
        variable_container = QWidget()
        variable_layout = QGridLayout(variable_container)
        variable_layout.setContentsMargins(0, 0, 0, 0)
        variable_layout.setSpacing(8)
        
        # Variables header
        variables_label = QLabel("Variable Selection:")
        variables_label.setFont(font)  # Use the same bold font
        variable_layout.addWidget(variables_label, 0, 0, 1, 4)
        
        # Set fixed label width for alignment
        label_width = 80
        
        # Row 1: Outcome and Predictor variables
        outcome_label = QLabel("Outcome:")
        outcome_label.setMinimumWidth(label_width)
        self.outcome_variable_combo = QComboBox()
        # Make outcome dropdown stand out
        self.outcome_variable_combo.setFont(large_font)
        self.outcome_variable_combo.setMinimumHeight(30)
        
        predictor_label = QLabel("Predictor:")
        predictor_label.setMinimumWidth(label_width)
        self.predictor_variable_combo = QComboBox()
        
        # Row 2: Available variables and Additional variables
        available_label = QLabel("Available:")
        available_label.setMinimumWidth(label_width)
        
        available_container = QWidget()
        available_layout = QHBoxLayout(available_container)
        available_layout.setContentsMargins(0, 0, 0, 0)
        available_layout.setSpacing(5)
        
        self.available_variables_combo = QComboBox()
        available_layout.addWidget(self.available_variables_combo, 1)
        
        add_variable_button = QPushButton("Add")
        add_variable_button.setIcon(load_bootstrap_icon("plus"))
        add_variable_button.clicked.connect(self.add_variable)
        available_layout.addWidget(add_variable_button)
        
        additional_label = QLabel("Additional:")
        additional_label.setMinimumWidth(label_width)
        self.additional_variables_list = QPlainTextEdit()
        self.additional_variables_list.setMaximumHeight(60)
        self.additional_variables_list.setReadOnly(True)
        
        # Add all elements to the grid
        variable_layout.addWidget(outcome_label, 1, 0)
        variable_layout.addWidget(self.outcome_variable_combo, 1, 1)
        variable_layout.addWidget(predictor_label, 1, 2)
        variable_layout.addWidget(self.predictor_variable_combo, 1, 3)
        variable_layout.addWidget(available_label, 2, 0)
        variable_layout.addWidget(available_container, 2, 1)
        variable_layout.addWidget(additional_label, 2, 2)
        variable_layout.addWidget(self.additional_variables_list, 2, 3)
        
        analysis_vars_row.addWidget(variable_container, 3)
        
        # Add the analysis type and variable selection row to the settings layout
        settings_layout.addLayout(analysis_vars_row)
        
        # Configuration section - below analysis type and variables
        config_label = QLabel("Configuration Options:")
        config_label.setFont(font)  # Use the same bold font
        settings_layout.addWidget(config_label)
        
        self.config_group = QWidget()  # Changed to QWidget from QGroupBox
        self.config_layout = QVBoxLayout(self.config_group)
        self.config_layout.setContentsMargins(0, 0, 0, 0)  # No margins since it's inside a group box
        self.config_layout.setSpacing(8)
        
        settings_layout.addWidget(self.config_group)
        
        # Add the settings group to the top layout
        top_layout.addWidget(settings_group)
        
        main_layout.addWidget(top_section)
        
        # Create horizontal splitter for main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Left side - Multiple tabs for data, results, etc.
        left_tabs = QTabWidget()
        left_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Tab 1: Dataset display
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        dataset_layout.setContentsMargins(5, 5, 5, 5)
        
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        dataset_layout.addWidget(self.data_table)
        left_tabs.addTab(dataset_tab, "Dataset")
        
        # Tab 2: Summary Results
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(5, 5, 5, 5)
        
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        summary_layout.addWidget(self.results_text)
        
        left_tabs.addTab(summary_tab, "Summary")
        
        # Tab 3: Detailed Results
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(5, 5, 5, 5)
        
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(3)
        self.details_table.setHorizontalHeaderLabels(["Parameter", "Original", "Sensitivity Result"])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        details_layout.addWidget(self.details_table)
        left_tabs.addTab(details_tab, "Detailed Results")
        
        # Tab 4: Interpretation Results
        interpretation_tab = QWidget()
        interpretation_layout = QVBoxLayout(interpretation_tab)
        interpretation_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create a scroll area for the content
        interpretation_scroll = QScrollArea()
        interpretation_scroll.setWidgetResizable(True)
        interpretation_content = QWidget()
        self.interpretation_layout = QVBoxLayout(interpretation_content)
        self.interpretation_layout.setSpacing(10)
        interpretation_scroll.setWidget(interpretation_content)
        
        # Add placeholder text for when no interpretation is available
        self.interpretation_placeholder = QLabel("No AI interpretation available yet. Click the 'Interpret' button to generate one.")
        self.interpretation_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.interpretation_placeholder.setStyleSheet("font-style: italic; color: gray;")
        self.interpretation_layout.addWidget(self.interpretation_placeholder)
        
        interpretation_layout.addWidget(interpretation_scroll)
        left_tabs.addTab(interpretation_tab, "Interpretation")
        
        # Right side - Visualization (modified)
        visualization_group = QGroupBox("Visualization")
        self.visualization_layout = QVBoxLayout(visualization_group)
        self.visualization_layout.setContentsMargins(5, 5, 5, 5)
        
        self.visualization_placeholder = QLabel("Visualizations will be shown here")
        self.visualization_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_placeholder.setStyleSheet("font-style: italic; color: gray;")
        self.visualization_layout.addWidget(self.visualization_placeholder)
        
        # Save results button at the bottom of visualization panel
        save_layout = QHBoxLayout()
        
        self.visualization_layout.addLayout(save_layout)
        
        # Add panels to the splitter
        content_splitter.addWidget(left_tabs)
        content_splitter.addWidget(visualization_group)
        
        # Set sizes for better visibility (60% left, 40% right)
        content_splitter.setSizes([600, 400])
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize the UI based on the selected analysis type
        self.on_analysis_type_changed(0)
        
        # Set a reasonable minimum size
        self.setMinimumSize(1000, 800)
    
    def load_dataset_from_study(self):
        """Load available datasets from the studies manager."""
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        # Get datasets from active study
        datasets = main_window.studies_manager.get_datasets_from_active_study()
        
        if not datasets:
            QMessageBox.information(self, "Info", "No datasets available in the active study")
            return
        
        # Clear and repopulate the dataset selector
        self.dataset_selector.clear()
        for name, df in datasets:
            # Store the dataframe as user data
            self.dataset_selector.addItem(name, df)
        
        self.status_bar.showMessage(f"Loaded {len(datasets)} datasets from active study")
    
    def on_dataset_changed(self, index):
        """Handle selection of a dataset from the dropdown."""
        if index < 0:
            return
        
        # Get the selected dataset
        name = self.dataset_selector.currentText()
        df = self.dataset_selector.currentData()
        
        if df is None:
            return
        
        self.current_name = name
        self.current_dataframe = df
        
        # Update the UI
        self.display_dataset(df)
        self.populate_variable_selectors(df)
        
        self.status_bar.showMessage(f"Selected dataset: {name} with {len(df)} rows and {len(df.columns)} columns")
    
    def display_dataset(self, df):
        """Display a dataset in the table."""
        if df is None or df.empty:
            return
        
        self.data_table.clear()
        
        # Set up the table dimensions
        self.data_table.setRowCount(min(1000, len(df)))
        self.data_table.setColumnCount(len(df.columns))
        
        # Set up the headers
        self.data_table.setHorizontalHeaderLabels(df.columns)
        
        # Fill the table with data
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 1000:  # Limit to 1000 rows for performance
                break
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.data_table.setItem(i, j, item)
        
        # Resize columns for better display
        self.data_table.resizeColumnsToContents()
    
    def populate_variable_selectors(self, df):
        """Populate the variable selection dropdowns."""
        if df is None or df.empty:
            return
        
        # Clear existing items
        self.outcome_variable_combo.clear()
        self.predictor_variable_combo.clear()
        self.available_variables_combo.clear()
        
        # Add placeholder item
        self.outcome_variable_combo.addItem("Select...", None)
        self.predictor_variable_combo.addItem("Select...", None)
        self.available_variables_combo.addItem("Select...", None)
        
        # Add each column to the dropdowns
        for col in df.columns:
            self.outcome_variable_combo.addItem(col, col)
            self.predictor_variable_combo.addItem(col, col)
            self.available_variables_combo.addItem(col, col)
    
    def add_variable(self):
        """Add a variable to the additional variables list."""
        selected_var = self.available_variables_combo.currentText()
        if selected_var and selected_var != "Select...":
            # Check if the variable is already in the list
            current_vars = self.additional_variables_list.toPlainText().strip().split('\n')
            if selected_var not in current_vars:
                # Add the variable to the list
                if current_vars == [''] or not current_vars:
                    self.additional_variables_list.setPlainText(selected_var)
                else:
                    self.additional_variables_list.setPlainText('\n'.join(current_vars + [selected_var]))
                self.status_bar.showMessage(f"Added variable: {selected_var}")
    
    def on_analysis_type_changed(self, index):
        """Update UI based on the selected analysis type."""
        if index < 0:
            return
        
        analysis_type = self.analysis_type_combo.currentData()
        if not analysis_type:
            return
        
        # Clear the existing configuration layout
        self.clear_layout(self.config_layout)
        
        # Update description
        self.analysis_description.setText(self.get_analysis_description(analysis_type))
        
        # Add configuration options based on analysis type
        if analysis_type == SensitivityAnalysisType.MISSING_DATA:
            self.setup_missing_data_config()
        elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
            self.setup_outlier_detection_config()
        elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
            self.setup_alternative_model_config()
        elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
            self.setup_parameter_variation_config()
    
    def clear_layout(self, layout):
        """Clear all widgets from a layout."""
        if layout is None:
            return
        
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clear_layout(item.layout())
    
    def get_analysis_description(self, analysis_type):
        """Get description for the selected analysis type."""
        if analysis_type == SensitivityAnalysisType.MISSING_DATA:
            return ("Evaluates how missing data assumptions and imputation methods affect your results. "
                   "Helps identify if your findings are robust to different missing data handling techniques.")
        elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
            return ("Identifies and evaluates the impact of outliers on your statistical model. "
                   "Determines if your results are sensitive to extreme values in your dataset.")
        elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
            return ("Tests how your results change when using alternative statistical models or specifications. "
                   "Helps validate that your findings aren't dependent on a specific model choice.")
        elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
            return ("Varies key parameters in your analysis to understand how sensitive your results are "
                   "to small changes in assumptions or algorithm settings.")
        else:
            return ""
    
    def setup_missing_data_config(self):
        """Setup configuration options for missing data analysis."""
        # Create a single-row layout for all options
        config_row = QHBoxLayout()
        
        # Missing data methods
        method_group = QGroupBox("Missing Data Method")
        method_layout = QVBoxLayout(method_group)
        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "Complete Case Analysis", 
            "Mean Imputation", 
            "Median Imputation",
            "Mode Imputation",
            "Multiple Imputation",
            "MICE (Multiple Imputation by Chained Equations)",
            "K-Nearest Neighbors"
        ])
        method_layout.addWidget(self.missing_method_combo)
        config_row.addWidget(method_group)
        
        # Missing threshold
        threshold_group = QGroupBox("Missing Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        self.missing_threshold_spinner = QSpinBox()
        self.missing_threshold_spinner.setRange(1, 50)
        self.missing_threshold_spinner.setValue(10)
        self.missing_threshold_spinner.setSuffix("%")
        threshold_layout.addWidget(self.missing_threshold_spinner)
        config_row.addWidget(threshold_group)
        
        # Simulate missing data
        simulate_group = QGroupBox("Missing Data Simulation")
        simulate_layout = QVBoxLayout(simulate_group)
        self.simulate_missing_check = QCheckBox("Simulate additional missing data")
        simulate_layout.addWidget(self.simulate_missing_check)
        
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Pattern:"))
        self.missing_pattern_combo = QComboBox()
        self.missing_pattern_combo.addItems(["MCAR (Missing Completely at Random)", 
                                          "MAR (Missing at Random)", 
                                          "MNAR (Missing Not at Random)"])
        pattern_layout.addWidget(self.missing_pattern_combo)
        simulate_layout.addLayout(pattern_layout)
        
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Rate:"))
        self.missing_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.missing_rate_slider.setRange(1, 50)
        self.missing_rate_slider.setValue(10)
        self.missing_rate_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.missing_rate_slider.setTickInterval(5)
        rate_layout.addWidget(self.missing_rate_slider)
        
        self.missing_rate_label = QLabel("10%")
        rate_layout.addWidget(self.missing_rate_label)
        simulate_layout.addLayout(rate_layout)
        
        # Connect slider to label
        self.missing_rate_slider.valueChanged.connect(
            lambda value: self.missing_rate_label.setText(f"{value}%")
        )
        
        config_row.addWidget(simulate_group)
        
        # Add the row to the config layout
        self.config_layout.addLayout(config_row)
    
    def setup_outlier_detection_config(self):
        """Setup configuration options for outlier detection."""
        # Create a single-row layout for all options
        config_row = QHBoxLayout()
        
        # Outlier detection method
        method_group = QGroupBox("Detection Method")
        method_layout = QVBoxLayout(method_group)
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems([
            "Z-Score",
            "IQR (Interquartile Range)",
            "Isolation Forest",
            "Local Outlier Factor",
            "DBSCAN"
        ])
        method_layout.addWidget(self.outlier_method_combo)
        config_row.addWidget(method_group)
        
        # Threshold for outlier detection
        threshold_group = QGroupBox("Outlier Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        self.outlier_threshold_spinner = QDoubleSpinBox()
        self.outlier_threshold_spinner.setRange(1.5, 10.0)
        self.outlier_threshold_spinner.setValue(2.5)
        self.outlier_threshold_spinner.setSingleStep(0.1)
        threshold_layout.addWidget(self.outlier_threshold_spinner)
        config_row.addWidget(threshold_group)
        
        # Handling method
        handling_group = QGroupBox("Outlier Handling")
        handling_layout = QVBoxLayout(handling_group)
        self.outlier_handling_combo = QComboBox()
        self.outlier_handling_combo.addItems([
            "Remove",
            "Winsorize",
            "Trim",
            "Transform",
            "Robust Analysis"
        ])
        handling_layout.addWidget(self.outlier_handling_combo)
        
        # Compare results checkbox
        self.compare_outliers_check = QCheckBox("Compare with/without outliers")
        self.compare_outliers_check.setChecked(True)
        handling_layout.addWidget(self.compare_outliers_check)
        
        config_row.addWidget(handling_group)
        
        # Add the row to the config layout
        self.config_layout.addLayout(config_row)
    
    def setup_alternative_model_config(self):
        """Setup configuration options for alternative model analysis."""
        # Create a single-row layout for all options
        config_row = QHBoxLayout()
        
        # Current model
        current_model_group = QGroupBox("Current Model")
        current_model_layout = QVBoxLayout(current_model_group)
        self.current_model_combo = QComboBox()
        self.current_model_combo.addItems([
            "Linear Regression",
            "Logistic Regression",
            "ANOVA",
            "t-test",
            "Non-parametric Test",
            "Custom"
        ])
        current_model_layout.addWidget(self.current_model_combo)
        config_row.addWidget(current_model_group)
        
        # Alternative models to test
        alt_models_group = QGroupBox("Alternative Models to Test")
        alt_models_layout = QVBoxLayout(alt_models_group)
        
        # Scrollable area for checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(150)  # Set minimum height to ensure visibility
        
        scroll_widget = QWidget()
        checkbox_layout = QGridLayout(scroll_widget)
        checkbox_layout.setContentsMargins(5, 5, 5, 5)  # Add some margins for better appearance
        
        self.alt_model_checks = {}
        
        # Updated model list - removed Linear & Logistic Regression and added new models
        models = [
            "Polynomial Regression", "Ridge Regression", 
            "Lasso Regression", "ElasticNet", "Robust Regression", 
            "Probit Regression", 
            "Decision Tree", "Random Forest",
            "Gradient Boosting Regression", "Gradient Boosting Classification",
            "Neural Network", "Gaussian Process Regression"
        ]
        
        # Arrange checkboxes in a grid
        for i, model in enumerate(models):
            row, col = divmod(i, 3)
            checkbox = QCheckBox(model)
            checkbox_layout.addWidget(checkbox, row, col)
            self.alt_model_checks[model] = checkbox
        
        scroll_widget.setLayout(checkbox_layout)  # Make sure the layout is set
        scroll_area.setWidget(scroll_widget)
        alt_models_layout.addWidget(scroll_area)
        
        # Parameter variations checkbox
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Vary Parameters:"))
        self.vary_params_check = QCheckBox("Yes")
        params_layout.addWidget(self.vary_params_check)
        alt_models_layout.addLayout(params_layout)
        
        # Ensure the layout is set on the group
        alt_models_group.setLayout(alt_models_layout)
        
        config_row.addWidget(alt_models_group, 2)  # Give more space to alternatives
        
        # Add the row to the config layout
        self.config_layout.addLayout(config_row)
    
    def setup_parameter_variation_config(self):
        """Setup configuration options for parameter variation analysis."""
        # Create a single-row layout for all options
        config_row = QHBoxLayout()
        
        # Parameter selection group
        param_group = QGroupBox("Parameter to Vary")
        param_layout = QVBoxLayout(param_group)
        self.parameter_combo = QComboBox()
        self.parameter_combo.addItems([
            "Alpha Level",
            "Effect Size Threshold",
            "Sample Size",
            "Missing Data Threshold",
            "Outlier Definition",
            "Model Hyperparameters"
        ])
        param_layout.addWidget(self.parameter_combo)
        config_row.addWidget(param_group)
        
        # Range settings group
        range_group = QGroupBox("Parameter Range")
        range_layout = QGridLayout()
        
        range_layout.addWidget(QLabel("Start:"), 0, 0)
        self.start_value_spinner = QDoubleSpinBox()
        self.start_value_spinner.setRange(0.001, 1.0)
        self.start_value_spinner.setValue(0.01)
        self.start_value_spinner.setSingleStep(0.01)
        range_layout.addWidget(self.start_value_spinner, 0, 1)
        
        range_layout.addWidget(QLabel("End:"), 0, 2)
        self.end_value_spinner = QDoubleSpinBox()
        self.end_value_spinner.setRange(0.001, 1.0)
        self.end_value_spinner.setValue(0.10)
        self.end_value_spinner.setSingleStep(0.01)
        range_layout.addWidget(self.end_value_spinner, 0, 3)
        
        range_layout.addWidget(QLabel("Steps:"), 1, 0)
        self.num_steps_spinner = QSpinBox()
        self.num_steps_spinner.setRange(3, 50)
        self.num_steps_spinner.setValue(10)
        range_layout.addWidget(self.num_steps_spinner, 1, 1)
        
        range_group.setLayout(range_layout)
        config_row.addWidget(range_group)
        
        # Options group
        options_group = QGroupBox("Analysis Options")
        options_layout = QVBoxLayout(options_group)
        
        # Multiple testing correction
        correction_layout = QHBoxLayout()
        correction_layout.addWidget(QLabel("Correction:"))
        self.correction_combo = QComboBox()
        self.correction_combo.addItems([
            "None",
            "Bonferroni",
            "Holm-Bonferroni",
            "Benjamini-Hochberg",
            "Benjamini-Yekutieli"
        ])
        correction_layout.addWidget(self.correction_combo)
        options_layout.addLayout(correction_layout)
        
        # Visualization options
        self.show_visualization_check = QCheckBox("Generate visualization")
        self.show_visualization_check.setChecked(True)
        options_layout.addWidget(self.show_visualization_check)
        
        # Threshold for significance change
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Significance Threshold:"))
        self.significance_threshold_spinner = QDoubleSpinBox()
        self.significance_threshold_spinner.setRange(0.001, 0.1)
        self.significance_threshold_spinner.setValue(0.01)
        self.significance_threshold_spinner.setSingleStep(0.001)
        threshold_layout.addWidget(self.significance_threshold_spinner)
        options_layout.addLayout(threshold_layout)
        
        config_row.addWidget(options_group)
        
        # Add the row to the config layout
        self.config_layout.addLayout(config_row)
    
    def show_settings_dialog(self):
        """Show settings dialog for sensitivity analysis."""
        dialog = SensitivitySettingsDialog(self)
        
        # Set current settings
        dialog.manual_outcome_check.setChecked(self.manual_outcome_selection)
        dialog.manual_analysis_type_check.setChecked(self.manual_analysis_type_selection)
        
        # Show dialog
        if dialog.exec():
            # Update settings if OK was pressed
            self.manual_outcome_selection = dialog.manual_outcome_check.isChecked()
            self.manual_analysis_type_selection = dialog.manual_analysis_type_check.isChecked()
            
            self.status_bar.showMessage("Settings updated")

    @asyncSlot()
    async def auto_select_variables_clicked(self):
        """Handle click of the Select Variables button."""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "Please select a dataset first")
            return
            
        # Show loading message
        self.status_bar.showMessage("Using AI to analyze your dataset and select appropriate variables...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Run variable selection
            success = await self.auto_select_variables()
            if success:
                self.status_bar.showMessage("Variables selected successfully")
            else:
                self.status_bar.showMessage("Variable selection failed or was cancelled")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Variable selection failed: {str(e)}")
            self.status_bar.showMessage("Variable selection failed with error")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    @asyncSlot()
    async def auto_configure_analysis_options_clicked(self):
        """Handle click of the Configure Analysis button."""
        # Check if variables are selected
        outcome = self.outcome_variable_combo.currentText()
        predictor = self.predictor_variable_combo.currentText()
        
        if outcome == "Select..." or predictor == "Select...":
            QMessageBox.warning(self, "Incomplete Selection", 
                               "Please select both outcome and predictor variables before configuring the analysis.")
            return
        
        # Check if we should suggest analysis type and let user select
        if self.manual_analysis_type_selection:
            # Get AI suggestion for analysis type and let user choose
            analysis_type = await self.suggest_analysis_type_and_prompt()
            if analysis_type is None:  # User cancelled
                return
            
            # Set the combobox to this analysis type
            for i in range(self.analysis_type_combo.count()):
                if self.analysis_type_combo.itemData(i) == analysis_type:
                    self.analysis_type_combo.setCurrentIndex(i)
                    break
        else:
            # Just auto-suggest and apply
            analysis_type = self.suggest_analysis_type()
            # Set the combobox to this analysis type
            for i in range(self.analysis_type_combo.count()):
                if self.analysis_type_combo.itemData(i) == analysis_type:
                    self.analysis_type_combo.setCurrentIndex(i)
                    break
        
        # Show loading message
        self.status_bar.showMessage("Using AI to configure sensitivity analysis...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Run analysis configuration
            success = await self.auto_configure_analysis_options()
            if success:
                self.status_bar.showMessage("Analysis configured successfully")
            else:
                self.status_bar.showMessage("Analysis configuration failed or was cancelled")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis configuration failed: {str(e)}")
            self.status_bar.showMessage("Analysis configuration failed with error")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    @asyncSlot()
    async def auto_configure_analysis(self):
        """Use AI to automatically configure the sensitivity analysis (both variables and options)."""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "Please select a dataset first")
            return
            
        # Show loading message
        self.status_bar.showMessage("Using AI to analyze your dataset and configure sensitivity analysis...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # First phase: Select variables
            vars_success = await self.auto_select_variables()
            
            if vars_success:
                # Analysis type selection phase
                if self.manual_analysis_type_selection:
                    # Get AI suggestion for analysis type and let user choose
                    analysis_type = await self.suggest_analysis_type_and_prompt()
                    if analysis_type is None:  # User cancelled
                        self.status_bar.showMessage("Variable selection successful, but analysis configuration cancelled")
                        QApplication.restoreOverrideCursor()
                        return
                else:
                    # Just auto-suggest and apply
                    analysis_type = self.suggest_analysis_type()
                
                # Set the combobox to this analysis type
                for i in range(self.analysis_type_combo.count()):
                    if self.analysis_type_combo.itemData(i) == analysis_type:
                        self.analysis_type_combo.setCurrentIndex(i)
                        break
                
                # Second phase: Configure analysis options based on selected analysis type
                config_success = await self.auto_configure_analysis_options()
                
                if config_success:
                    self.status_bar.showMessage("Full AI configuration completed successfully")
                else:
                    self.status_bar.showMessage("Variable selection successful, but analysis configuration failed")
            else:
                self.status_bar.showMessage("AI configuration failed - could not select variables")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto-configuration failed: {str(e)}")
            self.status_bar.showMessage("AI configuration failed with error")
            import traceback
            traceback.print_exc()
            
        finally:
            QApplication.restoreOverrideCursor()

    def suggest_analysis_type(self):
        """Suggest an appropriate analysis type based on the data."""
        df = self.current_dataframe
        outcome = self.outcome_variable_combo.currentText()
        predictor = self.predictor_variable_combo.currentText()
        
        # Check for missing data
        if df is not None and outcome in df.columns and predictor in df.columns:
            missing_pct = (df[outcome].isna().mean() + df[predictor].isna().mean()) / 2 * 100
            if missing_pct > 5:
                return SensitivityAnalysisType.MISSING_DATA
        
        # Check for potential outliers
        try:
            if df is not None and outcome in df.columns and predictor in df.columns:
                if df[outcome].dtype.kind in 'ifc' and df[predictor].dtype.kind in 'ifc':
                    z_scores_outcome = np.abs((df[outcome] - df[outcome].mean()) / df[outcome].std())
                    z_scores_predictor = np.abs((df[predictor] - df[predictor].mean()) / df[predictor].std())
                    if (z_scores_outcome > 3).any() or (z_scores_predictor > 3).any():
                        return SensitivityAnalysisType.OUTLIER_DETECTION
        except:
            pass
        
        # If no other condition is met, default to alternative model analysis
        return SensitivityAnalysisType.ALTERNATIVE_MODEL

    async def auto_configure_analysis_options(self):
        """Configure analysis options based on the selected analysis type."""
        # Get the current selections
        df = self.current_dataframe
        outcome = self.outcome_variable_combo.currentText()
        predictor = self.predictor_variable_combo.currentText()
        additional_vars = self.additional_variables_list.toPlainText().strip().split('\n')
        if additional_vars == ['']:
            additional_vars = []
        
        # Skip the rest if we don't have valid variables selected
        if outcome == "Select..." or predictor == "Select...":
            QMessageBox.warning(self, "Incomplete Selection", 
                              "Please select both outcome and predictor variables before configuring the analysis.")
            self.status_bar.showMessage("Auto-configuration cancelled - incomplete variable selection")
            return False
        
        # Get the current analysis type
        analysis_type = self.analysis_type_combo.currentData()
        
        # Prepare data statistics for the prompt
        data_info = self.prepare_data_statistics(outcome, predictor, additional_vars)
        
        # Prepare prompt based on the specific analysis type
        config_prompt = self.get_analysis_config_prompt(analysis_type, outcome, predictor, additional_vars, data_info)
        
        # Second LLM call for configuration recommendations
        config_response = await call_llm_async(config_prompt, model=llm_config.default_text_model)
        
        # Parse configuration recommendations based on analysis type
        try:
            success = self.parse_and_apply_config(config_response, analysis_type)
            
            if success:
                self.status_bar.showMessage(f"AI configuration for {analysis_type.value} sensitivity analysis applied successfully")
                return True
            else:
                self.status_bar.showMessage("AI configuration failed - could not apply settings")
                return False
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not configure analysis options: {str(e)}")
            self.status_bar.showMessage("AI configuration failed with error")
            import traceback
            traceback.print_exc()
            return False

    def prepare_data_statistics(self, outcome, predictor, additional_vars):
        """Prepare dataset statistics for the AI configuration prompt."""
        df = self.current_dataframe
        
        # Collect dataset information
        data_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": {col: int(df[col].isna().sum()) for col in [outcome, predictor] + additional_vars if col in df.columns},
            "variable_types": {col: str(df[col].dtype) for col in [outcome, predictor] + additional_vars if col in df.columns}
        }
        
        # Add descriptive statistics for numerical variables
        data_info["statistics"] = {}
        for col in [outcome, predictor] + additional_vars:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                data_info["statistics"][col] = {
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "missing_percent": float(df[col].isna().mean() * 100)
                }
            elif col in df.columns:
                # For categorical variables, provide value counts
                value_counts = df[col].value_counts().to_dict()
                data_info["statistics"][col] = {
                    "unique_values": len(value_counts),
                    "top_values": {str(k): int(v) for k, v in list(value_counts.items())[:5]},
                    "missing_percent": float(df[col].isna().mean() * 100)
                }
        
        return data_info

    def get_analysis_config_prompt(self, analysis_type, outcome, predictor, additional_vars, data_info):
        """Get the appropriate prompt for the selected analysis type configuration."""
        df = self.current_dataframe
        
        # Base prompt structure
        base_prompt = f"""
        Based on a detailed analysis of the selected variables, recommend the optimal sensitivity analysis configuration.
        
        Selected variables:
        - Outcome: {outcome} ({df[outcome].dtype if outcome in df.columns else 'unknown'})
        - Predictor: {predictor} ({df[predictor].dtype if predictor in df.columns else 'unknown'})
        - Additional variables: {', '.join(additional_vars) if additional_vars else 'None'}
        
        Analysis type: {analysis_type.value}
        
        Variable statistics:
        {json.dumps(data_info["statistics"], indent=2)}
        
        Missing data summary:
        {json.dumps(data_info["missing_values"], indent=2)}
        """
        
        # Add analysis-specific instructions
        if analysis_type == SensitivityAnalysisType.MISSING_DATA:
            prompt = base_prompt + """
            Please provide a detailed configuration for Missing Data sensitivity analysis.
            
            Your response should be a JSON object with the following structure:
            {
                "config": {
                    "method": "One of: Complete Case Analysis, Mean Imputation, Median Imputation, Mode Imputation, Multiple Imputation, MICE, K-Nearest Neighbors",
                    "threshold": "An integer between 1-50 representing missing data percentage threshold",
                    "simulate_missing": "Boolean indicating whether to simulate additional missing data",
                    "pattern": "One of: MCAR (Missing Completely at Random), MAR (Missing at Random), MNAR (Missing Not at Random)",
                    "rate": "An integer between 1-50 representing the simulated missing data rate"
                },
                "explanation": "Explanation of the configuration choices"
            }
            """
        
        elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
            prompt = base_prompt + """
            Please provide a detailed configuration for Outlier Detection sensitivity analysis.
            
            Your response should be a JSON object with the following structure:
            {
                "config": {
                    "method": "One of: Z-Score, IQR (Interquartile Range), Isolation Forest, Local Outlier Factor, DBSCAN",
                    "threshold": "A number between 1.5-10.0 representing the outlier detection threshold",
                    "handling": "One of: Remove, Winsorize, Trim, Transform, Robust Analysis",
                    "compare": "Boolean indicating whether to compare results with and without outliers"
                },
                "explanation": "Explanation of the configuration choices"
            }
            """
        
        elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
            prompt = base_prompt + """
            Please provide a detailed configuration for Alternative Model sensitivity analysis.
            
            Your response should be a JSON object with the following structure:
            {
                "config": {
                    "current_model": "One of: Linear Regression, Logistic Regression, ANOVA, t-test, Non-parametric Test, Custom",
                    "alternative_models": ["Array of model names to test, such as: Polynomial Regression, Ridge Regression, Lasso Regression, ElasticNet, Robust Regression, Probit Regression, Decision Tree, Random Forest, Gradient Boosting Regression, Gradient Boosting Classification, Neural Network, Gaussian Process Regression"],
                    "vary_params": "Boolean indicating whether to vary parameters"
                },
                "explanation": "Explanation of the configuration choices"
            }
            """
        
        elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
            prompt = base_prompt + """
            Please provide a detailed configuration for Parameter Variation sensitivity analysis.
            
            Your response should be a JSON object with the following structure:
            {
                "config": {
                    "parameter": "One of: Alpha Level, Effect Size Threshold, Sample Size, Missing Data Threshold, Outlier Definition, Model Hyperparameters",
                    "start_value": "A number between 0.001-1.0 for the start of the parameter range",
                    "end_value": "A number between 0.001-1.0 for the end of the parameter range",
                    "num_steps": "An integer between 3-50 for the number of steps in the parameter range",
                    "correction": "One of: None, Bonferroni, Holm-Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli",
                    "show_visualization": "Boolean indicating whether to generate visualizations",
                    "significance_threshold": "A number between 0.001-0.1 representing the significance threshold"
                },
                "explanation": "Explanation of the configuration choices"
            }
            """
        
        return prompt

    def parse_and_apply_config(self, config_response, analysis_type):
        """Parse and apply configuration based on analysis type."""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', config_response, re.DOTALL)
            if json_match:
                config_json = json.loads(json_match.group(1))
            else:
                config_json = json.loads(config_response)
            
            # Apply the configuration based on analysis type
            if analysis_type == SensitivityAnalysisType.MISSING_DATA:
                self.apply_missing_data_config(config_json.get("config", {}))
            elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
                self.apply_outlier_detection_config(config_json.get("config", {}))
            elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
                self.apply_alternative_model_config(config_json.get("config", {}))
            elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
                self.apply_parameter_variation_config(config_json.get("config", {}))
            else:
                return False
            
            # Show explanation
            explanation = config_json.get("explanation", "Configuration applied based on dataset characteristics.")
            QMessageBox.information(self, "AI Configuration Applied", 
                                  f"The sensitivity analysis has been configured automatically.\n\n{explanation}")
            
            return True
            
        except json.JSONDecodeError as e:
            # Add detailed debugging information
            print(f"JSON parse error: {str(e)}")
            print(f"Response content: {config_response[:200]}...")  # Print first 200 chars
            
            # Try to extract any JSON-like content and show it to debug
            try:
                # Find anything that looks like JSON
                potential_json = re.search(r'\{.*\}', config_response, re.DOTALL)
                if potential_json:
                    print(f"Potential JSON found: {potential_json.group(0)[:200]}...")
            except:
                pass
                
            QMessageBox.warning(self, "Error", f"Could not parse AI configuration recommendations: {str(e)}")
            return False
        except Exception as e:
            print(f"Error applying configuration: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to apply configuration: {str(e)}")
            return False
    
    def apply_missing_data_config(self, config):
        """Apply AI-recommended configuration for missing data analysis."""
        if "method" in config:
            index = self.missing_method_combo.findText(config["method"])
            if index >= 0:
                self.missing_method_combo.setCurrentIndex(index)
        
        if "threshold" in config:
            self.missing_threshold_spinner.setValue(int(config["threshold"]))
        
        if "simulate_missing" in config:
            self.simulate_missing_check.setChecked(config["simulate_missing"])
        
        if "pattern" in config:
            # Fix: Use Qt.MatchFlag.MatchContains instead of Qt.MatchContains
            index = self.missing_pattern_combo.findText(config["pattern"], Qt.MatchFlag.MatchContains)
            if index >= 0:
                self.missing_pattern_combo.setCurrentIndex(index)
        
        if "rate" in config:
            self.missing_rate_slider.setValue(int(config["rate"]))
    
    def apply_outlier_detection_config(self, config):
        """Apply AI-recommended configuration for outlier detection analysis."""
        if "method" in config:
            index = self.outlier_method_combo.findText(config["method"])
            if index >= 0:
                self.outlier_method_combo.setCurrentIndex(index)
        
        if "threshold" in config:
            self.outlier_threshold_spinner.setValue(float(config["threshold"]))
        
        if "handling" in config:
            index = self.outlier_handling_combo.findText(config["handling"])
            if index >= 0:
                self.outlier_handling_combo.setCurrentIndex(index)
        
        if "compare" in config:
            self.compare_outliers_check.setChecked(config["compare"])
    
    def apply_alternative_model_config(self, config):
        """Apply AI-recommended configuration for alternative model analysis."""
        if "current_model" in config:
            index = self.current_model_combo.findText(config["current_model"])
            if index >= 0:
                self.current_model_combo.setCurrentIndex(index)
        
        if "alternative_models" in config and isinstance(config["alternative_models"], list):
            # Reset all checkboxes first
            for checkbox in self.alt_model_checks.values():
                checkbox.setChecked(False)
            
            # Check the recommended models
            for model in config["alternative_models"]:
                if model in self.alt_model_checks:
                    self.alt_model_checks[model].setChecked(True)
        
        if "vary_params" in config:
            self.vary_params_check.setChecked(config["vary_params"])
    
    def apply_parameter_variation_config(self, config):
        """Apply AI-recommended configuration for parameter variation analysis."""
        if "parameter" in config:
            index = self.parameter_combo.findText(config["parameter"])
            if index >= 0:
                self.parameter_combo.setCurrentIndex(index)
        
        if "start_value" in config:
            self.start_value_spinner.setValue(float(config["start_value"]))
        
        if "end_value" in config:
            self.end_value_spinner.setValue(float(config["end_value"]))
        
        if "num_steps" in config:
            self.num_steps_spinner.setValue(int(config["num_steps"]))
        
        if "correction" in config:
            index = self.correction_combo.findText(config["correction"])
            if index >= 0:
                self.correction_combo.setCurrentIndex(index)
        
        if "show_visualization" in config:
            self.show_visualization_check.setChecked(config["show_visualization"])
        
        if "significance_threshold" in config:
            self.significance_threshold_spinner.setValue(float(config["significance_threshold"]))
    
    def run_sensitivity_analysis(self):
        """Run the selected sensitivity analysis."""
        if self.current_dataframe is None:
            QMessageBox.warning(self, "Error", "Please select a dataset first")
            return
        
        # Get the current analysis type
        analysis_type = self.analysis_type_combo.currentData()
        if not analysis_type:
            return
        
        # Get the selected variables
        outcome = self.outcome_variable_combo.currentText()
        if outcome == "Select...":
            QMessageBox.warning(self, "Error", "Please select an outcome variable")
            return
        
        predictor = self.predictor_variable_combo.currentText()
        if predictor == "Select...":
            QMessageBox.warning(self, "Error", "Please select a predictor/group variable")
            return
        
        additional_vars = self.additional_variables_list.toPlainText().strip().split('\n')
        if additional_vars == ['']:
            additional_vars = []
        
        # Show loading message
        self.status_bar.showMessage(f"Running {analysis_type.value} sensitivity analysis...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Perform the appropriate analysis based on analysis type
            if analysis_type == SensitivityAnalysisType.MISSING_DATA:
                results = self.perform_missing_data_analysis(outcome, predictor, additional_vars)
            elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
                results = self.perform_outlier_analysis(outcome, predictor, additional_vars)
            elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
                results = self.perform_alternative_model_analysis(outcome, predictor, additional_vars)
            elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
                results = self.perform_parameter_variation_analysis(outcome, predictor, additional_vars)
            else:
                results = None
                
            QApplication.restoreOverrideCursor()
            
            if results:
                # Store results
                self.analysis_results = {
                    'type': 'sensitivity_analysis',
                    'analysis_type': analysis_type.value,
                    'dataset': self.current_name,
                    'outcome': outcome,
                    'predictor': predictor,
                    'additional_vars': additional_vars,
                    'results': results.__dict__,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Display results
                self.display_sensitivity_results(results)
                self.create_visualizations(results)
                
                # Clear the interpretation tab since this is a new analysis
                self.clear_layout(self.interpretation_layout)
                
                # Create a new placeholder label instead of reusing the old one that was deleted
                new_placeholder = QLabel("No AI interpretation available yet. Click the 'Interpret' button to generate one.")
                new_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                new_placeholder.setStyleSheet("font-style: italic; color: gray;")
                self.interpretation_layout.addWidget(new_placeholder)
                self.interpretation_placeholder = new_placeholder  # Update the reference
                
                # Ask user if they want LLM interpretation
                reply = QMessageBox.question(
                    self, 
                    "LLM Interpretation",
                    "Would you like to get an AI-generated interpretation of these results?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    asyncio.create_task(self.share_results_with_llm())
                
                self.status_bar.showMessage(f"Completed {analysis_type.value} sensitivity analysis")
            else:
                QMessageBox.warning(self, "Error", "Analysis failed to produce results")
                self.status_bar.showMessage("Analysis failed")
        
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Sensitivity analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage("Analysis failed with error")
    
    def perform_missing_data_analysis(self, outcome, predictor, additional_vars):
        """Perform missing data sensitivity analysis."""
        df = self.current_dataframe
        method = self.missing_method_combo.currentText()
        threshold = self.missing_threshold_spinner.value() / 100
        simulate = self.simulate_missing_check.isChecked()
        pattern = self.missing_pattern_combo.currentText()
        rate = self.missing_rate_slider.value() / 100
        
        # Initialize results
        results = SensitivityResults(SensitivityAnalysisType.MISSING_DATA)
        
        # Create formula for regression
        covariates = ""
        if additional_vars and len(additional_vars) > 0:
            covariates = " + " + " + ".join(additional_vars)
            
        formula = f"{outcome} ~ {predictor}{covariates}"
        
        # Run baseline analysis with complete data
        complete_df = df.dropna(subset=[outcome, predictor] + additional_vars)
        if len(complete_df) < 10:
            raise ValueError("Too few complete cases for analysis")
            
        # Baseline model
        baseline_model = smf.ols(formula, data=complete_df).fit()
        
        # Extract key statistics
        p_value = baseline_model.pvalues[predictor]
        effect_size = baseline_model.params[predictor]
        r_squared = baseline_model.rsquared
        
        # Add baseline results
        baseline_interpretation = "Statistically significant" if p_value < 0.05 else "Not statistically significant"
        results.add_baseline_result("OLS Regression", p_value, effect_size, 
                                   f"{baseline_interpretation} (p={p_value:.4f}, effect={effect_size:.4f}, R²={r_squared:.4f})")
        
        # Sensitivity scenarios
        scenarios = []
        
        # 1. Different imputation methods
        if method == "Mean Imputation":
            # Create a copy of the dataframe to modify
            imputed_data = df.copy()
            
            # Apply mean imputation only to numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if numeric_cols.any():
                imputer = SimpleImputer(strategy='mean')
                imputed_data[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # For categorical columns, use mode imputation
            cat_cols = df.select_dtypes(exclude=['number']).columns
            if cat_cols.any():
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                
            scenarios.append(("Mean Imputation", imputed_data))
            
        elif method == "Median Imputation":
            # Create a copy of the dataframe to modify
            imputed_data = df.copy()
            
            # Apply median imputation only to numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if numeric_cols.any():
                imputer = SimpleImputer(strategy='median')
                imputed_data[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # For categorical columns, use mode imputation
            cat_cols = df.select_dtypes(exclude=['number']).columns
            if cat_cols.any():
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                
            scenarios.append(("Median Imputation", imputed_data))
            
        elif method == "Mode Imputation":
            imputer = SimpleImputer(strategy='most_frequent')
            imputed_data = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
            scenarios.append(("Mode Imputation", imputed_data))
            
        elif method == "K-Nearest Neighbors":
            # Create a copy of the dataframe to modify
            imputed_data = df.copy()
            
            # Apply KNN imputation only to numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if numeric_cols.any():
                imputer = KNNImputer(n_neighbors=5)
                imputed_data[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # For categorical columns, use mode imputation
            cat_cols = df.select_dtypes(exclude=['number']).columns
            if cat_cols.any():
                cat_imputer = SimpleImputer(strategy='most_frequent')
                imputed_data[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
                
            scenarios.append(("KNN Imputation", imputed_data))
        
        # 2. If simulate missing data is checked, create scenarios with more missing data
        if simulate:
            # MCAR - randomly remove data
            if "MCAR" in pattern:
                mcar_df = df.copy()
                for col in [outcome, predictor] + additional_vars:
                    if col in mcar_df.columns:
                        mask = np.random.random(size=len(mcar_df)) < rate
                        mcar_df.loc[mask, col] = np.nan
                
                # Impute the MCAR data - handle numeric and categorical separately
                imputed_data = mcar_df.copy()
                
                # Handle numeric columns
                numeric_cols = mcar_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    num_imputer = SimpleImputer(strategy='mean')
                    imputed_data[numeric_cols] = num_imputer.fit_transform(mcar_df[numeric_cols])
                
                # Handle categorical columns
                cat_cols = mcar_df.select_dtypes(exclude=['number']).columns
                if len(cat_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    imputed_data[cat_cols] = cat_imputer.fit_transform(mcar_df[cat_cols])
                    
                scenarios.append((f"MCAR ({rate*100:.0f}%) + Mean Imputation", imputed_data))
            
            # MAR - missing depending on other variables
            if "MAR" in pattern:
                mar_df = df.copy()
                # Make missingness depend on predictor value
                median_value = mar_df[predictor].median()
                mask = (mar_df[predictor] > median_value) & (np.random.random(size=len(mar_df)) < rate*1.5)
                mar_df.loc[mask, outcome] = np.nan
                
                # Impute the MAR data - handle numeric and categorical separately
                imputed_data = mar_df.copy()
                
                # Handle numeric columns
                numeric_cols = mar_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    num_imputer = SimpleImputer(strategy='mean')
                    imputed_data[numeric_cols] = num_imputer.fit_transform(mar_df[numeric_cols])
                
                # Handle categorical columns
                cat_cols = mar_df.select_dtypes(exclude=['number']).columns
                if len(cat_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    imputed_data[cat_cols] = cat_imputer.fit_transform(mar_df[cat_cols])
                    
                scenarios.append((f"MAR ({rate*100:.0f}%) + Mean Imputation", imputed_data))
        
        # Run models for each scenario
        for scenario_name, scenario_df in scenarios:
            try:
                model = smf.ols(formula, data=scenario_df).fit()
                
                # Extract key statistics
                scenario_p = model.pvalues[predictor]
                scenario_effect = model.params[predictor]
                
                # Calculate change
                p_change = scenario_p - p_value
                effect_change = scenario_effect - effect_size
                
                # Determine if change is significant
                significant_change = abs(p_change) > 0.05 or abs(effect_change/effect_size) > 0.1
                
                # Format the change description
                change_desc = f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}, " + \
                             f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}"
                
                # Add to results
                results.add_sensitivity_result(
                    scenario_name, 
                    scenario_p, 
                    scenario_effect,
                    change_desc,
                    significant_change
                )
                
            except Exception as e:
                print(f"Error in scenario {scenario_name}: {str(e)}")
        
        # Generate visualizations - store data for plots
        p_values = [result['p_value'] for result in results.sensitivity_results]
        effects = [result['effect_size'] for result in results.sensitivity_results]
        scenarios = [result['scenario'] for result in results.sensitivity_results]
        
        results.add_visualization_data('p_values', {'baseline': p_value, 'sensitivity': p_values, 'labels': scenarios})
        results.add_visualization_data('effects', {'baseline': effect_size, 'sensitivity': effects, 'labels': scenarios})
        
        # Generate summary and assessment
        if any(result['significant_change'] for result in results.sensitivity_results):
            results.set_robustness_assessment(
                "Results show sensitivity to missing data handling methods. " +
                "This suggests that missing data mechanisms may be influencing your findings."
            )
            results.set_summary(
                "The analysis reveals that different approaches to handling missing data " +
                "lead to notable changes in the results. This indicates potential bias " +
                "due to missing data patterns."
            )
            results.set_recommendation(
                "Consider using multiple imputation techniques and reporting ranges of " +
                "estimates. Collecting more complete data or understanding the missing " +
                "data mechanism would also strengthen your analysis."
            )
        else:
            results.set_robustness_assessment(
                "Results are robust to different missing data handling methods. " +
                "This suggests that missing data is unlikely to be biasing your findings substantially."
            )
            results.set_summary(
                "The analysis shows consistent results across different approaches to " +
                "handling missing data. This suggests the findings are not highly " +
                "sensitive to missing data patterns."
            )
            results.set_recommendation(
                "The analysis appears robust to missing data considerations. You can " +
                "proceed with your chosen method for handling missing values with " +
                "reasonable confidence."
            )
        
        return results
    
    def perform_outlier_analysis(self, outcome, predictor, additional_vars):
        """Perform outlier detection sensitivity analysis."""
        df = self.current_dataframe
        method = self.outlier_method_combo.currentText()
        threshold = self.outlier_threshold_spinner.value()
        handling = self.outlier_handling_combo.currentText()
        compare = self.compare_outliers_check.isChecked()
        
        # Initialize results
        results = SensitivityResults(SensitivityAnalysisType.OUTLIER_DETECTION)
        
        # Create formula for regression
        covariates = ""
        if additional_vars and len(additional_vars) > 0:
            covariates = " + " + " + ".join(additional_vars)
            
        formula = f"{outcome} ~ {predictor}{covariates}"
        
        # Run baseline analysis with all data
        baseline_model = smf.ols(formula, data=df).fit()
        
        # Extract key statistics
        p_value = baseline_model.pvalues[predictor]
        effect_size = baseline_model.params[predictor]
        r_squared = baseline_model.rsquared
        
        # Add baseline results
        baseline_interpretation = "Statistically significant" if p_value < 0.05 else "Not statistically significant"
        results.add_baseline_result("OLS Regression", p_value, effect_size, 
                                  f"{baseline_interpretation} (p={p_value:.4f}, effect={effect_size:.4f}, R²={r_squared:.4f})")
        
        # Detect outliers based on the selected method
        outliers_mask = np.zeros(len(df), dtype=bool)
        
        # Variables to consider for outlier detection
        analysis_vars = [outcome, predictor] + additional_vars
        analysis_df = df[analysis_vars].copy()
        
        if method == "Z-Score":
            # Z-score method
            for col in analysis_vars:
                if pd.api.types.is_numeric_dtype(df[col]):
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers_mask = outliers_mask | (z_scores > threshold)
        
        elif method == "IQR (Interquartile Range)":
            # IQR method
            for col in analysis_vars:
                if pd.api.types.is_numeric_dtype(df[col]):
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers_mask = outliers_mask | ((df[col] < lower_bound) | (df[col] > upper_bound))
        
        elif method == "Isolation Forest":
            # Isolation Forest
            numeric_vars = [col for col in analysis_vars if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_vars:
                # Drop rows with NaN values for isolation forest
                temp_df = df[numeric_vars].dropna()
                indices = temp_df.index
                
                # Standardize the data
                scaler = StandardScaler()
                X = scaler.fit_transform(temp_df)
                
                # Fit isolation forest
                clf = IsolationForest(contamination=0.1, random_state=42)
                preds = clf.fit_predict(X)
                
                # Create mask for the original dataframe
                outlier_series = pd.Series(False, index=df.index)
                outlier_series.loc[indices[preds == -1]] = True
                outliers_mask = outlier_series.values
        
        elif method == "Local Outlier Factor":
            # Local Outlier Factor
            numeric_vars = [col for col in analysis_vars if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_vars:
                # Drop rows with NaN values
                temp_df = df[numeric_vars].dropna()
                indices = temp_df.index
                
                # Standardize the data
                scaler = StandardScaler()
                X = scaler.fit_transform(temp_df)
                
                # Fit LOF
                clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                preds = clf.fit_predict(X)
                
                # Create mask for the original dataframe
                outlier_series = pd.Series(False, index=df.index)
                outlier_series.loc[indices[preds == -1]] = True
                outliers_mask = outlier_series.values
        
        elif method == "DBSCAN":
            # DBSCAN
            numeric_vars = [col for col in analysis_vars if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_vars:
                # Drop rows with NaN values
                temp_df = df[numeric_vars].dropna()
                indices = temp_df.index
                
                # Standardize the data
                scaler = StandardScaler()
                X = scaler.fit_transform(temp_df)
                
                # Fit DBSCAN
                dbscan = DBSCAN(eps=threshold/2, min_samples=5)
                clusters = dbscan.fit_predict(X)
                
                # Points with cluster label -1 are outliers
                outlier_series = pd.Series(False, index=df.index)
                outlier_series.loc[indices[clusters == -1]] = True
                outliers_mask = outlier_series.values
        
        # Count outliers
        outlier_count = outliers_mask.sum()
        outlier_percent = (outlier_count / len(df)) * 100
        
        # Store outlier info for visualization
        results.add_visualization_data('outliers', {
            'count': int(outlier_count),
            'percent': float(outlier_percent),
            'mask': outliers_mask.tolist()
        })
        
        # Handle outliers based on the selected method
        scenarios = []
        
        if handling == "Remove":
            # Remove outliers
            no_outliers_df = df[~outliers_mask].copy()
            scenarios.append((f"Remove Outliers ({outlier_count} points, {outlier_percent:.1f}%)", no_outliers_df))
        
        elif handling == "Winsorize":
            # Winsorize outliers (cap at threshold)
            winsorized_df = df.copy()
            for col in analysis_vars:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if method == "Z-Score":
                        # Cap based on z-score threshold
                        mean = df[col].mean()
                        std = df[col].std()
                        lower_cap = mean - threshold * std
                        upper_cap = mean + threshold * std
                    elif method == "IQR (Interquartile Range)":
                        # Cap based on IQR threshold
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_cap = Q1 - threshold * IQR
                        upper_cap = Q3 + threshold * IQR
                    else:
                        # Default winsorization at 5th and 95th percentiles
                        lower_cap = df[col].quantile(0.05)
                        upper_cap = df[col].quantile(0.95)
                    
                    # Apply caps
                    winsorized_df[col] = np.where(winsorized_df[col] < lower_cap, lower_cap, winsorized_df[col])
                    winsorized_df[col] = np.where(winsorized_df[col] > upper_cap, upper_cap, winsorized_df[col])
            
            scenarios.append((f"Winsorize Outliers ({outlier_count} points, {outlier_percent:.1f}%)", winsorized_df))
        
        elif handling == "Trim":
            # Trim (remove) the top 5% and bottom 5% of data
            trimmed_df = df.copy()
            for col in analysis_vars:
                if pd.api.types.is_numeric_dtype(df[col]):
                    lower_pct = df[col].quantile(0.05)
                    upper_pct = df[col].quantile(0.95)
                    trimmed_mask = (trimmed_df[col] < lower_pct) | (trimmed_df[col] > upper_pct)
                    trimmed_df = trimmed_df[~trimmed_mask]
            
            trim_count = len(df) - len(trimmed_df)
            trim_percent = (trim_count / len(df)) * 100
            scenarios.append((f"Trim 5%-95% ({trim_count} points, {trim_percent:.1f}%)", trimmed_df))
        
        elif handling == "Transform":
            # Apply log transformation to reduce impact of outliers
            transformed_df = df.copy()
            for col in analysis_vars:
                if pd.api.types.is_numeric_dtype(df[col]) and (df[col] > 0).all():
                    transformed_df[col] = np.log1p(df[col])
            
            scenarios.append(("Log Transform", transformed_df))
        
        elif handling == "Robust Analysis":
            # Use robust regression instead of modifying data
            try:
                robust_model = sm.RLM(df[outcome], sm.add_constant(df[[predictor] + additional_vars]), 
                                    M=sm.robust.norms.HuberT()).fit()
                
                # Get robust results directly
                robust_p = np.nan  # RLM doesn't provide p-values directly
                robust_effect = robust_model.params[1]  # Assuming predictor is the first column after constant
                
                # Add to results directly
                results.add_sensitivity_result(
                    "Robust Regression (Huber)", 
                    robust_p, 
                    robust_effect,
                    "Used robust method instead of modifying data",
                    abs(robust_effect - effect_size) / abs(effect_size) > 0.1
                )
            except Exception as e:
                print(f"Error in robust regression: {str(e)}")
        
        # Run models for each scenario
        for scenario_name, scenario_df in scenarios:
            try:
                model = smf.ols(formula, data=scenario_df).fit()
                
                # Extract key statistics
                scenario_p = model.pvalues[predictor]
                scenario_effect = model.params[predictor]
                
                # Calculate change
                p_change = scenario_p - p_value
                effect_change = scenario_effect - effect_size
                
                # Determine if change is significant
                significant_change = abs(p_change) > 0.05 or abs(effect_change/effect_size) > 0.1 if effect_size != 0 else abs(effect_change) > 0.1
                
                # Format the change description
                change_desc = f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}, " + \
                             f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}"
                
                # Add to results
                results.add_sensitivity_result(
                    scenario_name, 
                    scenario_p, 
                    scenario_effect,
                    change_desc,
                    significant_change
                )
                
            except Exception as e:
                print(f"Error in scenario {scenario_name}: {str(e)}")
        
        # Generate summary and assessment
        if any(result['significant_change'] for result in results.sensitivity_results):
            results.set_robustness_assessment(
                "Results show sensitivity to outliers. This suggests that extreme values " +
                "are substantially influencing your findings."
            )
            results.set_summary(
                "The analysis reveals that different approaches to handling outliers " +
                "lead to notable changes in the results. Your findings may be driven " +
                "by a small number of influential data points."
            )
            results.set_recommendation(
                "Consider using robust statistical methods that are less sensitive to outliers. " +
                "Carefully examine the identified outliers to determine if they represent " +
                "valid observations or potential data issues."
            )
        else:
            results.set_robustness_assessment(
                "Results are robust to different outlier handling methods. " +
                "This suggests that outliers are not significantly biasing your findings."
            )
            results.set_summary(
                "The analysis shows consistent results regardless of how outliers are handled. " +
                "This suggests your findings are not driven by extreme values in the data."
            )
            results.set_recommendation(
                "The analysis appears robust to outliers. You can proceed with standard " +
                "methods, though it's always good practice to check for influential observations."
            )
        
        return results
    
    def perform_alternative_model_analysis(self, outcome, predictor, additional_vars):
        """Perform alternative model sensitivity analysis."""
        df = self.current_dataframe
        current_model = self.current_model_combo.currentText()
        
        # Get selected alternative models to test
        alternative_models = [model for model, checkbox in self.alt_model_checks.items() if checkbox.isChecked()]
        
        # If no alternative models selected, choose a few reasonable ones
        if not alternative_models:
            if current_model == "Linear Regression":
                alternative_models = ["Robust Regression", "Ridge Regression"]
            elif current_model == "Logistic Regression":
                alternative_models = ["Probit Regression"]
            else:
                alternative_models = ["Linear Regression"]
        
        # Initialize results
        results = SensitivityResults(SensitivityAnalysisType.ALTERNATIVE_MODEL)
        
        # Create formula for regression
        covariates = ""
        if additional_vars and len(additional_vars) > 0:
            covariates = " + " + " + ".join(additional_vars)
            
        formula = f"{outcome} ~ {predictor}{covariates}"
        
        # Determine if outcome is binary (for classification models)
        is_binary_outcome = False
        if df[outcome].nunique() == 2:
            is_binary_outcome = True
        
        # Run baseline model
        if current_model == "Linear Regression" or current_model == "Custom":
            baseline_model = smf.ols(formula, data=df).fit()
            baseline_method = "OLS Regression"
            p_value = baseline_model.pvalues[predictor]
            effect_size = baseline_model.params[predictor]
            interpretation = f"Coefficient: {effect_size:.4f}, p-value: {p_value:.4f}"
            
        elif current_model == "Logistic Regression":
            if not is_binary_outcome:
                raise ValueError("Logistic regression requires a binary outcome variable")
            baseline_model = smf.logit(formula, data=df).fit(disp=False)
            baseline_method = "Logistic Regression"
            p_value = baseline_model.pvalues[predictor]
            effect_size = np.exp(baseline_model.params[predictor])  # Odds ratio
            interpretation = f"Odds Ratio: {effect_size:.4f}, p-value: {p_value:.4f}"
            
        elif current_model == "ANOVA":
            # Simple one-way ANOVA
            groups = df.groupby(predictor)[outcome].apply(list)
            anova_result = stats.f_oneway(*groups)
            baseline_model = anova_result
            baseline_method = "One-way ANOVA"
            p_value = anova_result.pvalue
            effect_size = anova_result.statistic  # F-statistic
            interpretation = f"F-statistic: {effect_size:.4f}, p-value: {p_value:.4f}"
            
        elif current_model == "t-test":
            # Independent samples t-test
            if df[predictor].nunique() != 2:
                raise ValueError("T-test requires a binary predictor variable")
            group1 = df[df[predictor] == df[predictor].unique()[0]][outcome]
            group2 = df[df[predictor] == df[predictor].unique()[1]][outcome]
            t_result = stats.ttest_ind(group1, group2, equal_var=False)
            baseline_model = t_result
            baseline_method = "Independent Samples t-test"
            p_value = t_result.pvalue
            effect_size = t_result.statistic  # t-statistic
            interpretation = f"t-statistic: {effect_size:.4f}, p-value: {p_value:.4f}"
        
        elif current_model == "Non-parametric Test":
            # Mann-Whitney U test
            if df[predictor].nunique() != 2:
                raise ValueError("Mann-Whitney test requires a binary predictor variable")
            group1 = df[df[predictor] == df[predictor].unique()[0]][outcome]
            group2 = df[df[predictor] == df[predictor].unique()[1]][outcome]
            u_result = stats.mannwhitneyu(group1, group2)
            baseline_model = u_result
            baseline_method = "Mann-Whitney U Test"
            p_value = u_result.pvalue
            effect_size = u_result.statistic  # U statistic
            interpretation = f"U-statistic: {effect_size:.4f}, p-value: {p_value:.4f}"
        
        else:
            raise ValueError(f"Unsupported model type: {current_model}")
        
        # Add baseline results
        results.add_baseline_result(baseline_method, p_value, effect_size, interpretation)
        
        # Run alternative models
        for alt_model in alternative_models:
            try:
                if alt_model == "Linear Regression":
                    model = smf.ols(formula, data=df).fit()
                    alt_method = "OLS Regression"
                    alt_p = model.pvalues[predictor]
                    alt_effect = model.params[predictor]
                    alt_desc = f"Coefficient: {alt_effect:.4f}, p-value: {alt_p:.4f}"
                
                elif alt_model == "Robust Regression":
                    model = sm.RLM(df[outcome], sm.add_constant(df[[predictor] + additional_vars]), 
                                M=sm.robust.norms.HuberT()).fit()
                    alt_method = "Robust Regression"
                    alt_p = np.nan  # Not directly available
                    alt_effect = model.params[1]  # Assuming predictor is the first column after constant
                    alt_desc = f"Coefficient: {alt_effect:.4f}"
                
                elif alt_model == "Ridge Regression":
                    X = sm.add_constant(df[[predictor] + additional_vars])
                    y = df[outcome]
                    model = sm.OLS(y, X).fit_regularized(alpha=0.1, L1_wt=0)
                    alt_method = "Ridge Regression"
                    alt_p = np.nan  # Not available for regularized models
                    alt_effect = model.params[1]  # Assuming predictor is the first column after constant
                    alt_desc = f"Coefficient: {alt_effect:.4f}"
                
                elif alt_model == "Lasso Regression":
                    X = sm.add_constant(df[[predictor] + additional_vars])
                    y = df[outcome]
                    model = sm.OLS(y, X).fit_regularized(alpha=0.1, L1_wt=1)
                    alt_method = "Lasso Regression"
                    alt_p = np.nan  # Not available for regularized models
                    alt_effect = model.params[1]  # Assuming predictor is the first column after constant
                    alt_desc = f"Coefficient: {alt_effect:.4f}"
                
                elif alt_model == "Polynomial Regression":
                    # Add squared term
                    df_poly = df.copy()
                    df_poly[f"{predictor}_squared"] = df_poly[predictor] ** 2
                    poly_formula = f"{outcome} ~ {predictor} + {predictor}_squared{covariates}"
                    model = smf.ols(poly_formula, data=df_poly).fit()
                    alt_method = "Polynomial Regression"
                    alt_p = model.pvalues[predictor]
                    alt_effect = model.params[predictor]
                    alt_desc = (f"Linear term: {alt_effect:.4f}, p-value: {alt_p:.4f}, " +
                               f"Squared term: {model.params[f'{predictor}_squared']:.4f}, " +
                               f"p-value: {model.pvalues[f'{predictor}_squared']:.4f}")
                
                elif alt_model == "Logistic Regression" and is_binary_outcome:
                    model = smf.logit(formula, data=df).fit(disp=False)
                    alt_method = "Logistic Regression"
                    alt_p = model.pvalues[predictor]
                    alt_effect = np.exp(model.params[predictor])  # Odds ratio
                    alt_desc = f"Odds Ratio: {alt_effect:.4f}, p-value: {alt_p:.4f}"
                
                elif alt_model == "Probit Regression" and is_binary_outcome:
                    model = smf.probit(formula, data=df).fit(disp=False)
                    alt_method = "Probit Regression"
                    alt_p = model.pvalues[predictor]
                    alt_effect = model.params[predictor]
                    alt_desc = f"Coefficient: {alt_effect:.4f}, p-value: {alt_p:.4f}"
                
                elif alt_model == "Decision Tree":
                    # Prepare X and y for sklearn models
                    X = df[[predictor] + additional_vars].copy()
                    y = df[outcome].copy()
                    
                    # Handle categorical variables with one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # Choose classifier or regressor based on outcome type
                    if is_binary_outcome:
                        tree = DecisionTreeClassifier(max_depth=5, random_state=42)
                    else:
                        tree = DecisionTreeRegressor(max_depth=5, random_state=42)
                    
                    # Fit the model
                    tree.fit(X, y)
                    
                    # Get feature importance for the predictor
                    # Find all columns related to the predictor (in case it was one-hot encoded)
                    predictor_cols = [col for col in X.columns if predictor in col]
                    if predictor_cols:
                        # Sum importance of all related columns
                        importance = sum(tree.feature_importances_[
                            [list(X.columns).index(col) for col in predictor_cols]
                        ])
                    else:
                        importance = 0
                    
                    alt_method = "Decision Tree"
                    alt_p = np.nan  # p-values not available for tree models
                    alt_effect = importance  # Use feature importance as effect size
                    alt_desc = f"Feature importance: {alt_effect:.4f}"
                
                elif alt_model == "Random Forest":
                    # Prepare X and y for sklearn models
                    X = df[[predictor] + additional_vars].copy()
                    y = df[outcome].copy()
                    
                    # Handle categorical variables with one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # Choose classifier or regressor based on outcome type
                    if is_binary_outcome:
                        forest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    else:
                        forest = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                    
                    # Fit the model
                    forest.fit(X, y)
                    
                    # Get feature importance for the predictor
                    # Find all columns related to the predictor (in case it was one-hot encoded)
                    predictor_cols = [col for col in X.columns if predictor in col]
                    if predictor_cols:
                        # Sum importance of all related columns
                        importance = sum(forest.feature_importances_[
                            [list(X.columns).index(col) for col in predictor_cols]
                        ])
                    else:
                        importance = 0
                    
                    alt_method = "Random Forest"
                    alt_p = np.nan  # p-values not available for random forest
                    alt_effect = importance  # Use feature importance as effect size
                    alt_desc = f"Feature importance: {alt_effect:.4f}"
                
                elif alt_model == "ElasticNet":
                    # Prepare X and y 
                    X = df[[predictor] + additional_vars].copy()
                    y = df[outcome].copy()
                    
                    # Handle categorical variables with one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # ElasticNet for regression (only available for regression)
                    elastic = SKElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                    elastic.fit(X, y)
                    
                    # Find the predictor columns
                    predictor_cols = [col for col in X.columns if predictor in col]
                    if predictor_cols:
                        # Sum coefficients of all related columns
                        coef = sum(elastic.coef_[
                            [list(X.columns).index(col) for col in predictor_cols]
                        ])
                    else:
                        coef = 0
                    
                    alt_method = "ElasticNet"
                    alt_p = np.nan  # p-values not available
                    alt_effect = coef
                    alt_desc = f"Coefficient: {alt_effect:.4f}"
                
                elif alt_model == "Gradient Boosting Regression" or alt_model == "Gradient Boosting Classification":
                    # Prepare X and y
                    X = df[[predictor] + additional_vars].copy()
                    y = df[outcome].copy()
                    
                    # Handle categorical variables with one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # Choose classifier or regressor based on outcome type
                    if is_binary_outcome and alt_model == "Gradient Boosting Classification":
                        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                        model_type = "Classification"
                    else:
                        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                        model_type = "Regression"
                    
                    # Fit the model
                    gb.fit(X, y)
                    
                    # Get feature importance for the predictor
                    predictor_cols = [col for col in X.columns if predictor in col]
                    if predictor_cols:
                        # Sum importance of all related columns
                        importance = sum(gb.feature_importances_[
                            [list(X.columns).index(col) for col in predictor_cols]
                        ])
                    else:
                        importance = 0
                    
                    alt_method = f"Gradient Boosting {model_type}"
                    alt_p = np.nan  # p-values not available
                    alt_effect = importance
                    alt_desc = f"Feature importance: {alt_effect:.4f}"
                
                elif alt_model == "Neural Network":
                    # Prepare X and y
                    X = df[[predictor] + additional_vars].copy()
                    y = df[outcome].copy()
                    
                    # Handle categorical variables with one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # Standardize features for better neural network performance
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Choose classifier or regressor based on outcome type
                    if is_binary_outcome:
                        nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
                        nn.fit(X_scaled, y)
                    else:
                        nn = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
                        nn.fit(X_scaled, y)
                    
                    # Since neural networks don't have direct feature importance, use permutation importance
                    # Here we use a simpler approach - check if the model predicts differently when predictor is modified
                    X_test = X.copy()
                    predictor_cols = [col for col in X.columns if predictor in col]
                    
                    if predictor_cols and len(predictor_cols) > 0:
                        # Make a copy of the scaled data for prediction
                        X_scaled_modified = X_scaled.copy()
                        
                        # Get the index of the first predictor column
                        idx = list(X.columns).index(predictor_cols[0])
                        
                        # Modify the predictor column values (shift by 1 std)
                        X_scaled_modified[:, idx] += 1.0
                        
                        # Get predictions for original and modified data
                        if is_binary_outcome:
                            orig_pred = nn.predict_proba(X_scaled)[:, 1]
                            mod_pred = nn.predict_proba(X_scaled_modified)[:, 1]
                        else:
                            orig_pred = nn.predict(X_scaled)
                            mod_pred = nn.predict(X_scaled_modified)
                        
                        # Calculate effect as mean absolute difference in predictions
                        effect = np.mean(np.abs(mod_pred - orig_pred))
                    else:
                        effect = 0
                    
                    alt_method = "Neural Network"
                    alt_p = np.nan  # p-values not available
                    alt_effect = effect
                    alt_desc = f"Predictive impact: {alt_effect:.4f}"
                
                elif alt_model == "Gaussian Process Regression":
                    # Only for regression problems
                    if not is_binary_outcome:
                        # Prepare X and y
                        X = df[[predictor] + additional_vars].copy()
                        y = df[outcome].copy()
                        
                        # Handle categorical variables with one-hot encoding
                        X = pd.get_dummies(X, drop_first=True)
                        
                        # Standardize features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Define kernel
                        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
                        
                        # Fit Gaussian Process Regressor
                        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42)
                        gpr.fit(X_scaled, y)
                        
                        # For GP, we can use the kernel's lengthscales as indicators of feature relevance
                        # Smaller lengthscale = more relevant feature
                        if hasattr(gpr.kernel_, 'k2') and hasattr(gpr.kernel_.k2, 'length_scale'):
                            if isinstance(gpr.kernel_.k2.length_scale, np.ndarray):
                                # Find predictor columns
                                predictor_cols = [col for col in X.columns if predictor in col]
                                indices = [list(X.columns).index(col) for col in predictor_cols]
                                
                                if indices:
                                    # Smaller lengthscale means more important, so inverse it
                                    importance = np.mean(1.0 / gpr.kernel_.k2.length_scale[indices])
                                else:
                                    importance = 0
                            else:
                                # Single lengthscale for all features
                                importance = 1.0 / gpr.kernel_.k2.length_scale
                        else:
                            # Fallback if kernel structure is different
                            importance = 0.5  # Default value
                        
                        alt_method = "Gaussian Process Regression"
                        alt_p = np.nan  # p-values not available
                        alt_effect = importance
                        alt_desc = f"Feature relevance: {alt_effect:.4f}"
                    else:
                        # Skip for binary outcomes
                        continue
                
                # Determine if change is significant
                # For p-values, we check if significance conclusion changes
                # For effect sizes, we check if the relative change is substantial
                if not np.isnan(p_value) and not np.isnan(alt_p):
                    p_significant_change = (p_value < 0.05) != (alt_p < 0.05)
                else:
                    p_significant_change = False
                
                if effect_size != 0 and alt_effect is not None:
                    effect_relative_change = abs((alt_effect - effect_size) / effect_size)
                    effect_significant_change = effect_relative_change > 0.2  # 20% change
                else:
                    effect_significant_change = False
                    if alt_effect is not None:
                        effect_significant_change = abs(alt_effect - effect_size) > 0.1
                
                significant_change = p_significant_change or effect_significant_change
                
                # Format the change description
                if not np.isnan(p_value) and not np.isnan(alt_p):
                    p_change = alt_p - p_value
                    p_desc = f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}"
                else:
                    p_desc = "p-value not comparable"
                
                if alt_effect is not None:
                    effect_change = alt_effect - effect_size
                    effect_desc = f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}"
                else:
                    effect_desc = "effect not comparable"
                
                change_desc = f"{p_desc}, {effect_desc}"
                
                # Add to results
                results.add_sensitivity_result(
                    f"Alternative Model: {alt_model}", 
                    alt_p, 
                    alt_effect,
                    change_desc,
                    significant_change
                )
                
            except Exception as e:
                print(f"Error with alternative model {alt_model}: {str(e)}")
        
        # Generate summary and assessment
        # Count how many models showed significant changes and identify patterns
        significant_count = sum(1 for result in results.sensitivity_results if result.get('significant_change', False))
        total_models = len(results.sensitivity_results)
        significance_percentage = significant_count / total_models * 100 if total_models > 0 else 0
        
        # Check if there are patterns in which types of models show sensitivity
        tree_based_models = ["Random Forest", "Decision Tree", "Gradient Boosting"]
        linear_models = ["Ridge Regression", "Lasso Regression", "ElasticNet", "Robust Regression"]
        nonlinear_models = ["Polynomial Regression", "Neural Network", "Gaussian Process Regression"]
        
        # Calculate sensitivity rates by model families
        tree_significant = sum(1 for r in results.sensitivity_results if any(model in r['scenario'] for model in tree_based_models) and r.get('significant_change', False))
        linear_significant = sum(1 for r in results.sensitivity_results if any(model in r['scenario'] for model in linear_models) and r.get('significant_change', False))
        nonlinear_significant = sum(1 for r in results.sensitivity_results if any(model in r['scenario'] for model in nonlinear_models) and r.get('significant_change', False))
        
        # Did statistical significance conclusions change or just effect sizes?
        sig_conclusion_changes = sum(1 for r in results.sensitivity_results 
                                     if not np.isnan(p_value) and not np.isnan(r.get('p_value', np.nan))
                                     and ((p_value < 0.05) != (r.get('p_value', 1) < 0.05)))
        
        if significance_percentage > 50:
            # High sensitivity
            results.set_robustness_assessment(
                f"Results show substantial sensitivity to model specification ({significant_count} out of {total_models} "
                f"alternative models, {significance_percentage:.1f}%). The choice of statistical model significantly "
                f"influences your findings. {sig_conclusion_changes} models yielded different conclusions about statistical "
                f"significance compared to the baseline model."
            )
            
            # Identify which model families show the most sensitivity
            family_assessment = "The sensitivity is most pronounced in "
            if nonlinear_significant > linear_significant and nonlinear_significant > tree_significant:
                family_assessment += "nonlinear models (like Neural Networks and Polynomial Regression), "
                family_assessment += "suggesting possible nonlinear relationships in your data."
            elif tree_significant > linear_significant:
                family_assessment += "tree-based models (like Random Forest and Decision Tree), "
                family_assessment += "suggesting possible complex interactions or nonlinear thresholds in your data."
            else:
                family_assessment += "various model types, indicating fundamental differences in how they handle your data's structure."
            
            results.set_summary(
                f"The analysis reveals that different model specifications lead to notably different results. "
                f"This suggests that the relationship between your variables may be more complex than captured "
                f"by a single model. {family_assessment} Your findings may be sensitive to underlying model "
                f"assumptions such as linearity, independence, or normality of residuals."
            )
            
            results.set_recommendation(
                "Consider the following steps to address this sensitivity:\n\n"
                "1. Use multiple model specifications and report results from several models to provide a more complete picture\n"
                "2. Examine why different models yield different results by checking model assumptions and fit diagnostics\n"
                "3. Explore your data for nonlinearities, interactions, or heterogeneity that might explain these differences\n"
                "4. Consider methods like model averaging, ensemble methods, or Bayesian model averaging to incorporate uncertainty\n"
                "5. Be transparent about sensitivity to model specification in your reporting and discussion\n"
                "6. Focus on the practical significance of effects rather than just statistical significance"
            )
        elif significance_percentage > 20:
            # Moderate sensitivity
            results.set_robustness_assessment(
                f"Results show moderate sensitivity to model specification ({significant_count} out of {total_models} "
                f"alternative models, {significance_percentage:.1f}%). While most model choices lead to similar conclusions, "
                f"some alternative specifications yield different results. {sig_conclusion_changes} models yielded different "
                f"conclusions about statistical significance."
            )
            
            # More nuanced analysis of which models differ
            different_models = [r['scenario'].replace("Alternative Model: ", "") for r in results.sensitivity_results if r.get('significant_change', False)]
            
            results.set_summary(
                f"The analysis indicates that while your core findings are somewhat robust, they are not entirely "
                f"insensitive to modeling choices. Models showing different results include: {', '.join(different_models)}. "
                f"This suggests potential complexity in the data that may be captured differently by different model specifications."
            )
            
            results.set_recommendation(
                "To address this moderate sensitivity:\n\n"
                "1. Report results from both your primary model and alternative specifications that yielded different results\n"
                "2. Check model diagnostics to ensure assumptions are satisfied for your primary model\n"
                "3. Consider whether the differences between models are practically significant for your research question\n"
                "4. Examine specific features of models that produce different results to understand what aspects of your data might be driving these differences\n"
                "5. Discuss the range of possible effects indicated across models in your conclusions"
            )
        else:
            # Low/no sensitivity
            results.set_robustness_assessment(
                f"Results are robust to model specification. {significant_count} out of {total_models} alternative models "
                f"({significance_percentage:.1f}%) showed meaningful differences from the baseline model. Different statistical "
                f"models yield consistent findings, indicating your results are not dependent on specific modeling choices."
            )
            
            # Slightly different framing based on whether there are any differences at all
            if significant_count == 0:
                consistency_text = "remarkably consistent"
                implication_text = "This strongly supports the validity of your findings"
            else:
                changed_models = [r['scenario'].replace("Alternative Model: ", "") for r in results.sensitivity_results if r.get('significant_change', False)]
                consistency_text = "generally consistent with only minor differences"
                implication_text = f"The few models showing some differences ({', '.join(changed_models)}) do not substantively change your conclusions"
            
            results.set_summary(
                f"The analysis shows {consistency_text} results across different model specifications. "
                f"{implication_text} and suggests that the relationship you've identified is stable "
                f"across different analytical approaches. This strengthens confidence in the robustness of your results."
            )
            
            results.set_recommendation(
                "With such robust results across model specifications:\n\n"
                "1. You can proceed with your preferred approach, noting that results are consistent across multiple specifications\n"
                "2. Consider including a brief sensitivity analysis section mentioning the alternative models tested\n"
                "3. Focus on the interpretation and implications of your findings since the effect appears stable\n"
                "4. If publication is a goal, highlight this robustness as it strengthens the credibility of your findings\n"
                "5. Ensure you select the most appropriate model for your research question and theoretical framework despite the consistency"
            )
        
        return results
    
    def perform_parameter_variation_analysis(self, outcome, predictor, additional_vars):
        """Perform parameter variation sensitivity analysis."""
        df = self.current_dataframe
        parameter = self.parameter_combo.currentText()
        start_value = self.start_value_spinner.value()
        end_value = self.end_value_spinner.value()
        num_steps = self.num_steps_spinner.value()
        correction = self.correction_combo.currentText()
        threshold = self.significance_threshold_spinner.value()
        
        # Initialize results
        results = SensitivityResults(SensitivityAnalysisType.PARAMETER_VARIATION)
        
        # Create formula for regression
        covariates = ""
        if additional_vars and len(additional_vars) > 0:
            covariates = " + " + " + ".join(additional_vars)
            
        formula = f"{outcome} ~ {predictor}{covariates}"
        
        try:
            # Run baseline model
            baseline_model = smf.ols(formula, data=df).fit()
            
            # Check if predictor is in the model parameters to avoid KeyError
            if predictor not in baseline_model.params:
                available_params = baseline_model.params.index.tolist()
                raise ValueError(f"Predictor variable '{predictor}' not found in model parameters. "
                               f"Available parameters: {', '.join(available_params)}")
            
            # Extract key statistics
            p_value = baseline_model.pvalues[predictor]
            effect_size = baseline_model.params[predictor]
            r_squared = baseline_model.rsquared
            
            # Add baseline results
            baseline_interpretation = "Statistically significant" if p_value < 0.05 else "Not statistically significant"
            results.add_baseline_result("OLS Regression", p_value, effect_size, 
                                       f"{baseline_interpretation} (p={p_value:.4f}, effect={effect_size:.4f}, R²={r_squared:.4f})")
            
            # Generate parameter values to test
            parameter_values = np.linspace(start_value, end_value, num_steps)
            
            # Store values for visualization
            p_values = []
            effects = []
            r_squareds = []
            
            # Run analysis for each parameter value
            for param_val in parameter_values:
                try:
                    # Different parameters require different approaches
                    if parameter == "Alpha Level":
                        # Just change the interpretation threshold
                        significant = p_value < param_val
                        param_interpretation = f"{'Significant' if significant else 'Not significant'} at α={param_val:.4f}"
                        
                        # No need to refit the model, just interpret differently
                        results.add_sensitivity_result(
                            f"Alpha = {param_val:.4f}",
                            p_value,
                            effect_size,
                            f"Changed significance threshold: {param_interpretation}",
                            (p_value < 0.05) != significant
                        )
                        
                        # Store the same values
                        p_values.append(p_value)
                        effects.append(effect_size)
                        r_squareds.append(r_squared)
                        
                    elif parameter == "Sample Size":
                        # Subsample the data to simulate smaller sample size
                        subsample_pct = param_val
                        subsample_size = int(len(df) * subsample_pct)
                        if subsample_size < 10:
                            continue  # Skip if sample too small
                        
                        # Generate 5 random subsamples and take the average results
                        subsample_p_values = []
                        subsample_effects = []
                        subsample_r2s = []
                        
                        for _ in range(5):
                            try:
                                # Sample without replacement
                                subsample = df.sample(n=subsample_size)
                                subsample_model = smf.ols(formula, data=subsample).fit()
                                
                                # Verify predictor is in this model too
                                if predictor in subsample_model.params:
                                    subsample_p_values.append(subsample_model.pvalues[predictor])
                                    subsample_effects.append(subsample_model.params[predictor])
                                    subsample_r2s.append(subsample_model.rsquared)
                            except Exception as subsample_err:
                                print(f"Error in subsample: {str(subsample_err)}")
                                continue
                        
                        if not subsample_p_values:
                            continue  # Skip if all subsample models failed
                        
                        # Average results
                        subsample_p = np.mean(subsample_p_values)
                        subsample_effect = np.mean(subsample_effects)
                        subsample_r2 = np.mean(subsample_r2s)
                        
                        # Calculate change
                        p_change = subsample_p - p_value
                        effect_change = subsample_effect - effect_size
                        
                        # Determine if change is significant
                        significant_change = abs(p_change) > threshold or abs(effect_change/effect_size) > 0.1
                        
                        # Add to results
                        results.add_sensitivity_result(
                            f"Sample Size = {subsample_size} ({subsample_pct:.2f} of data)",
                            subsample_p,
                            subsample_effect,
                            f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}, " +
                            f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}",
                            significant_change
                        )
                        
                        # Store values for visualization
                        p_values.append(subsample_p)
                        effects.append(subsample_effect)
                        r_squareds.append(subsample_r2)
                        
                    elif parameter == "Outlier Definition":
                        # Change outlier definition threshold and rerun without outliers
                        z_threshold = param_val
                        
                        # Identify outliers using z-score with varying threshold
                        outliers_mask = np.zeros(len(df), dtype=bool)
                        for col in [outcome, predictor] + additional_vars:
                            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                                outliers_mask = outliers_mask | (z_scores > z_threshold)
                        
                        outlier_count = outliers_mask.sum()
                        if outlier_count > len(df) * 0.5:
                            continue  # Skip if too many outliers are removed
                        
                        if outlier_count == 0:
                            # No outliers found, results same as baseline
                            results.add_sensitivity_result(
                                f"Z-threshold = {z_threshold:.4f}",
                                p_value,
                                effect_size,
                                "No outliers detected, same as baseline",
                                False
                            )
                            
                            # Store the same values
                            p_values.append(p_value)
                            effects.append(effect_size)
                            r_squareds.append(r_squared)
                            
                        else:
                            try:
                                # Fit model without outliers
                                no_outliers_df = df[~outliers_mask].copy()
                                outlier_model = smf.ols(formula, data=no_outliers_df).fit()
                                
                                if predictor in outlier_model.params:
                                    outlier_p = outlier_model.pvalues[predictor]
                                    outlier_effect = outlier_model.params[predictor]
                                    outlier_r2 = outlier_model.rsquared
                                    
                                    # Calculate change
                                    p_change = outlier_p - p_value
                                    effect_change = outlier_effect - effect_size
                                    
                                    # Determine if change is significant
                                    significant_change = abs(p_change) > threshold or abs(effect_change/effect_size) > 0.1
                                    
                                    # Add to results
                                    results.add_sensitivity_result(
                                        f"Z-threshold = {z_threshold:.4f} ({outlier_count} outliers removed)",
                                        outlier_p,
                                        outlier_effect,
                                        f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}, " +
                                        f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}",
                                        significant_change
                                    )
                                    
                                    # Store values for visualization
                                    p_values.append(outlier_p)
                                    effects.append(outlier_effect)
                                    r_squareds.append(outlier_r2)
                                else:
                                    # Predictor not in model
                                    print(f"Predictor {predictor} not in outlier model parameters")
                            except Exception as outlier_err:
                                print(f"Error fitting model without outliers: {str(outlier_err)}")
                                continue
                    
                    elif parameter == "Missing Data Threshold":
                        # Simulate missing data at different rates and perform imputation
                        missing_rate = param_val
                        
                        try:
                            # Simulate MCAR missing data
                            mcar_df = df.copy()
                            for col in [outcome, predictor] + additional_vars:
                                if col in mcar_df.columns:
                                    mask = np.random.random(size=len(mcar_df)) < missing_rate
                                    mcar_df.loc[mask, col] = np.nan
                            
                            # Skip if too many missing values
                            if mcar_df[outcome].isna().sum() > len(df) * 0.5:
                                continue
                            
                            # Impute missing values with mean
                            imputer = SimpleImputer(strategy='mean')
                            imputed_data = pd.DataFrame(
                                imputer.fit_transform(mcar_df),
                                columns=mcar_df.columns,
                                index=mcar_df.index
                            )
                            
                            # Fit model on imputed data
                            imputed_model = smf.ols(formula, data=imputed_data).fit()
                            
                            if predictor in imputed_model.params:
                                imputed_p = imputed_model.pvalues[predictor]
                                imputed_effect = imputed_model.params[predictor]
                                imputed_r2 = imputed_model.rsquared
                                
                                # Calculate change
                                p_change = imputed_p - p_value
                                effect_change = imputed_effect - effect_size
                                
                                # Determine if change is significant
                                significant_change = abs(p_change) > threshold or abs(effect_change/effect_size) > 0.1
                                
                                # Add to results
                                results.add_sensitivity_result(
                                    f"Missing data rate = {missing_rate:.4f}",
                                    imputed_p,
                                    imputed_effect,
                                    f"p-value {'increased' if p_change > 0 else 'decreased'} by {abs(p_change):.4f}, " +
                                    f"effect {'increased' if effect_change > 0 else 'decreased'} by {abs(effect_change):.4f}",
                                    significant_change
                                )
                                
                                # Store values for visualization
                                p_values.append(imputed_p)
                                effects.append(imputed_effect)
                                r_squareds.append(imputed_r2)
                            else:
                                print(f"Predictor {predictor} not in imputed model parameters")
                        except Exception as missing_err:
                            print(f"Error in missing data analysis: {str(missing_err)}")
                            continue
                    
                    elif parameter == "Effect Size Threshold":
                        # Just change the interpretation threshold for practical significance
                        practical_significance = abs(effect_size) > param_val
                        param_interpretation = (f"{'Practically significant' if practical_significance else 'Not practically significant'} " +
                                              f"at threshold={param_val:.4f}")
                        
                        # No need to refit the model, just interpret differently
                        results.add_sensitivity_result(
                            f"Effect size threshold = {param_val:.4f}",
                            p_value,
                            effect_size,
                            f"Changed practical significance threshold: {param_interpretation}",
                            (abs(effect_size) > 0.1) != practical_significance  # Using 0.1 as default practical significance
                        )
                        
                        # Store the same values
                        p_values.append(p_value)
                        effects.append(effect_size)
                        r_squareds.append(r_squared)
                
                except Exception as param_err:
                    print(f"Error processing parameter value {param_val}: {str(param_err)}")
                    continue
            
            # Add visualization data if we have results
            if len(p_values) > 0:
                results.add_visualization_data('parameter_values', parameter_values.tolist())
                results.add_visualization_data('p_values', p_values)
                results.add_visualization_data('effects', effects)
                results.add_visualization_data('r_squared', r_squareds)
                results.add_visualization_data('parameter_name', parameter)
                
                # Generate summary and assessment
                if any(result['significant_change'] for result in results.sensitivity_results):
                    results.set_robustness_assessment(
                        f"Results are sensitive to variations in the {parameter.lower()} parameter. " +
                        "This suggests that your findings may change depending on analytical choices."
                    )
                    results.set_summary(
                        f"The analysis reveals that different values of {parameter.lower()} " +
                        "lead to notable changes in the results. This indicates that your " +
                        "findings are not robust to this parameter."
                    )
                    results.set_recommendation(
                        f"Consider reporting results across a range of {parameter.lower()} values " +
                        "rather than a single value. Transparently discuss how the choice of " +
                        f"{parameter.lower()} affects your conclusions."
                    )
                else:
                    results.set_robustness_assessment(
                        f"Results are robust to variations in the {parameter.lower()} parameter. " +
                        "This suggests that your findings are not sensitive to analytical choices."
                    )
                    results.set_summary(
                        f"The analysis shows consistent results across different values of {parameter.lower()}. " +
                        "This strengthens confidence in the robustness of your findings."
                    )
                    results.set_recommendation(
                        f"The choice of {parameter.lower()} does not appear to be critical for your analysis. " +
                        "You can proceed with standard values, noting that the results are consistent."
                    )
            else:
                # No parameter variations were successful
                results.set_summary("No successful parameter variations were computed. Please check your data and parameters.")
                results.set_recommendation("Try different parameter ranges or a different sensitivity analysis approach.")
        
        except Exception as e:
            # Handle errors in the main analysis
            import traceback
            traceback.print_exc()
            print(f"Error in parameter variation analysis: {str(e)}")
            
            # Add minimal information to results
            results.set_summary(f"Analysis failed with error: {str(e)}")
            results.set_recommendation("Please check your data and variable selections.")
            
        # Always return results object, even if empty
        return results

    def display_sensitivity_results(self, results):
        """Display the sensitivity analysis results."""
        # Display summary in the results text area
        if hasattr(results, 'format_summary'):
            summary_text = results.format_summary()
        else:
            summary_text = "No formatted results available."
        
        self.results_text.setPlainText(summary_text)
        
        # Populate the details table with sensitivity results
        self.details_table.setRowCount(0)
        
        sensitivity_results = results.sensitivity_results
        for i, result in enumerate(sensitivity_results):
            self.details_table.insertRow(i)
            
            # Scenario
            scenario_item = QTableWidgetItem(result.get('scenario', ''))
            self.details_table.setItem(i, 0, scenario_item)
            
            # Original result (baseline)
            if i == 0 and results.baseline_results:
                original_text = f"p={results.baseline_results.get('p_value', 'N/A')}, effect={results.baseline_results.get('effect_size', 'N/A')}"
            else:
                original_text = "See baseline"
            original_item = QTableWidgetItem(original_text)
            self.details_table.setItem(i, 1, original_item)
            
            # Sensitivity result
            result_text = f"p={result.get('p_value', 'N/A')}, effect={result.get('effect_size', 'N/A')}"
            result_item = QTableWidgetItem(result_text)
            
            # Color code based on significant change
            if result.get('significant_change', False):
                result_item.setBackground(QBrush(QColor(255, 200, 200)))  # Light red
            
            self.details_table.setItem(i, 2, result_item)
        
        # Resize rows and columns
        self.details_table.resizeColumnsToContents()
        self.details_table.resizeRowsToContents()
    
    def create_visualizations(self, results):
        """Create visualizations for the sensitivity analysis results."""
        # Clear the visualization layout
        for i in reversed(range(self.visualization_layout.count())): 
            widget = self.visualization_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Create a scroll area for visualizations
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)
        self.visualization_layout.addWidget(scroll_area)
        
        analysis_type = results.analysis_type
        
        if analysis_type == SensitivityAnalysisType.MISSING_DATA:
            self.create_missing_data_visualizations(results, scroll_layout)
            
        elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION:
            self.create_outlier_visualizations(results, scroll_layout)
            
        elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL:
            self.create_alternative_model_visualizations(results, scroll_layout)
            
        elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION:
            self.create_parameter_variation_visualizations(results, scroll_layout)
    
    def create_missing_data_visualizations(self, results, parent_layout):
        """Create visualizations for missing data sensitivity analysis."""
        if 'p_values' not in results.visualization_data or 'effects' not in results.visualization_data:
            parent_layout.addWidget(QLabel("No visualization data available"))
            return
        
        # Create figure with two subplots
        fig = Figure(figsize=(10, 6))
        # Set transparent background
        fig.patch.set_alpha(0.0)
        
        # Use consistent color palette from formatting module
        colors = PASTEL_COLORS
        
        # p-value comparison
        ax1 = fig.add_subplot(121)
        ax1.patch.set_alpha(0.0)  # Make axis background transparent
        
        p_value_data = results.visualization_data['p_values']
        baseline_p = p_value_data['baseline']
        sensitivity_p = p_value_data['sensitivity']
        labels = p_value_data['labels']
        
        # Bar chart for p-values
        x = np.arange(len(sensitivity_p)+1)
        bars = ax1.bar(x, [baseline_p] + sensitivity_p, color=[colors[0]] + [colors[1]]*len(sensitivity_p))
        ax1.axhline(0.05, color=colors[5], linestyle='--', label='α = 0.05')
        
        # Add labels
        ax1.set_xlabel('Analysis')
        ax1.set_ylabel('p-value')
        ax1.set_title('p-value Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Baseline'] + labels, rotation=45, ha='right')
        ax1.legend()
        
        # Effect size comparison
        ax2 = fig.add_subplot(122)
        ax2.patch.set_alpha(0.0)  # Make axis background transparent
        
        effect_data = results.visualization_data['effects']
        baseline_effect = effect_data['baseline']
        sensitivity_effect = effect_data['sensitivity']
        
        # Bar chart for effect sizes
        bars = ax2.bar(x, [baseline_effect] + sensitivity_effect, color=[colors[0]] + [colors[2]]*len(sensitivity_effect))
        
        # Add labels
        ax2.set_xlabel('Analysis')
        ax2.set_ylabel('Effect Size')
        ax2.set_title('Effect Size Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Baseline'] + labels, rotation=45, ha='right')
        
        fig.tight_layout()
        
        # Create SVG visualization
        svg_string = fig_to_svg(fig)
        
        # Create container with maximize button
        self.add_visualization_with_maximize(parent_layout, svg_string, "Missing Data Sensitivity Analysis", fig)
    
    def create_outlier_visualizations(self, results, parent_layout):
        """Create visualizations for outlier sensitivity analysis."""
        if 'outliers' not in results.visualization_data:
            parent_layout.addWidget(QLabel("No visualization data available"))
            return
        
        # Create figure with appropriate subplots
        fig = Figure(figsize=(10, 6))
        fig.patch.set_alpha(0.0)  # Set transparent background
        
        # First subplot: Compare effect sizes
        ax1 = fig.add_subplot(121)
        ax1.patch.set_alpha(0.0)  # Make axis background transparent
        
        # Extract data
        scenarios = [result['scenario'] for result in results.sensitivity_results]
        effects = [result['effect_size'] for result in results.sensitivity_results]
        baseline_effect = results.baseline_results['effect_size']
        significant_changes = [result['significant_change'] for result in results.sensitivity_results]
        
        # Bar chart for effect sizes
        x = np.arange(len(scenarios)+1)
        bars = ax1.bar(x, [baseline_effect] + effects)
        
        # Color the bars by significance using the color palette
        bars[0].set_color(PASTEL_COLORS[0])  # Baseline
        for i, significant in enumerate(significant_changes):
            bars[i+1].set_color(PASTEL_COLORS[4] if significant else PASTEL_COLORS[2])
        
        # Add labels
        ax1.set_xlabel('Analysis')
        ax1.set_ylabel('Effect Size')
        ax1.set_title('Effect Size Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Baseline'] + scenarios, rotation=45, ha='right')
        
        # Second subplot: Scatter plot of data with outliers highlighted
        ax2 = fig.add_subplot(122)
        ax2.patch.set_alpha(0.0)  # Make axis background transparent
        
        # Check if we have the original data
        if hasattr(self, 'current_dataframe') and self.current_dataframe is not None:
            df = self.current_dataframe
            outcome = self.outcome_variable_combo.currentText()
            predictor = self.predictor_variable_combo.currentText()
            
            if outcome in df.columns and predictor in df.columns:
                # Get outlier mask if available
                outlier_data = results.visualization_data['outliers']
                outlier_mask = outlier_data.get('mask', [False] * len(df))
                
                # Scatter plot with pastel colors
                ax2.scatter(df[predictor], df[outcome], 
                           c=[PASTEL_COLORS[4] if x else PASTEL_COLORS[2] for x in outlier_mask], 
                           alpha=0.6)
                ax2.set_xlabel(predictor)
                ax2.set_ylabel(outcome)
                ax2.set_title('Data with Outliers (red)')
                
                # Add regression line
                try:
                    # Get the baseline model coefficients
                    intercept = results.baseline_model.params['Intercept']
                    slope = results.baseline_model.params[predictor]
                    
                    # Create a line using these coefficients
                    x_range = np.linspace(df[predictor].min(), df[predictor].max(), 100)
                    y_range = intercept + slope * x_range
                    ax2.plot(x_range, y_range, '--', color=PASTEL_COLORS[5], label='Baseline regression')
                    
                    # Add legend
                    ax2.legend()
                except Exception as e:
                    print(f"Error creating regression line: {str(e)}")
        
        fig.tight_layout()
        
        # Convert to SVG
        svg_string = fig_to_svg(fig)
        
        # Add visualization with maximize button
        self.add_visualization_with_maximize(parent_layout, svg_string, "Outlier Detection Analysis", fig)
    
    def create_alternative_model_visualizations(self, results, parent_layout):
        """Create visualizations for alternative model sensitivity analysis."""
        if not results.sensitivity_results:
            parent_layout.addWidget(QLabel("No visualization data available"))
            return
        
        # Create figure
        fig = Figure(figsize=(10, 6))
        fig.patch.set_alpha(0.0)  # Set transparent background
        
        # Extract data
        models = ["Baseline"] + [result['scenario'] for result in results.sensitivity_results]
        p_values = [results.baseline_results['p_value']] + [result['p_value'] for result in results.sensitivity_results]
        effect_sizes = [results.baseline_results['effect_size']] + [result['effect_size'] for result in results.sensitivity_results]
        
        # Replace NaN values with None for better display
        p_values = [None if np.isnan(p) else p for p in p_values]
        
        # First subplot: p-values
        ax1 = fig.add_subplot(121)
        ax1.patch.set_alpha(0.0)  # Make axis background transparent
        
        # Bar chart for p-values if they exist
        if not all(p is None for p in p_values):
            # Filter out None values for plotting
            valid_indices = [i for i, p in enumerate(p_values) if p is not None]
            valid_models = [models[i] for i in valid_indices]
            valid_p_values = [p_values[i] for i in valid_indices]
            
            x = np.arange(len(valid_models))
            bars = ax1.bar(x, valid_p_values)
            
            # Color bars with pastel palette
            for i, bar in enumerate(bars):
                original_idx = valid_indices[i]
                if original_idx == 0:
                    bar.set_color(PASTEL_COLORS[0])  # Baseline
                elif valid_p_values[i] < 0.05:
                    bar.set_color(PASTEL_COLORS[1])  # Significant
                else:
                    bar.set_color(PASTEL_COLORS[4])  # Not significant
            
            # Add significance line
            ax1.axhline(0.05, color=PASTEL_COLORS[5], linestyle='--', label='α = 0.05')
            
            # Add labels
            ax1.set_xlabel('Model')
            ax1.set_ylabel('p-value')
            ax1.set_title('p-value Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(valid_models, rotation=45, ha='right')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "p-values not available for comparison", 
                   ha='center', va='center', transform=ax1.transAxes)
        
        # Second subplot: effect sizes
        ax2 = fig.add_subplot(122)
        ax2.patch.set_alpha(0.0)  # Make axis background transparent
        
        # Filter out None values for effect sizes as well
        valid_effect_indices = [i for i, e in enumerate(effect_sizes) if e is not None]
        valid_effect_models = [models[i] for i in valid_effect_indices]
        valid_effects = [effect_sizes[i] for i in valid_effect_indices]
        
        if valid_effects:
            # Bar chart for effect sizes
            x = np.arange(len(valid_effect_models))
            bars = ax2.bar(x, valid_effects)
            
            # Color by model using pastel colors
            for i, bar in enumerate(bars):
                original_idx = valid_effect_indices[i]
                if original_idx == 0:
                    bar.set_color(PASTEL_COLORS[0])  # Baseline
                else:
                    # Check if it's significantly different
                    if results.sensitivity_results[original_idx-1].get('significant_change', False):
                        bar.set_color(PASTEL_COLORS[3])  # Significant change
                    else:
                        bar.set_color(PASTEL_COLORS[2])  # Not significant change
            
            # Add labels
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Effect Size')
            ax2.set_title('Effect Size Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_effect_models, rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, "No valid effect sizes to display", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        fig.tight_layout()
        
        # Convert to SVG
        svg_string = fig_to_svg(fig)
        
        # Add visualization with maximize button
        self.add_visualization_with_maximize(parent_layout, svg_string, "Alternative Model Analysis", fig)
    
    def create_parameter_variation_visualizations(self, results, parent_layout):
        """Create visualizations for parameter variation sensitivity analysis."""
        if 'parameter_values' not in results.visualization_data:
            parent_layout.addWidget(QLabel("No visualization data available"))
            return
        
        # Create figure
        fig = Figure(figsize=(10, 8))  # Slightly taller for 4 subplots
        fig.patch.set_alpha(0.0)  # Set transparent background
        
        # Extract data
        param_values = results.visualization_data['parameter_values']
        p_values = results.visualization_data.get('p_values', [])
        effects = results.visualization_data.get('effects', [])
        r_squared = results.visualization_data.get('r_squared', [])
        param_name = results.visualization_data.get('parameter_name', 'Parameter')
        
        # First subplot: p-values
        ax1 = fig.add_subplot(221)
        ax1.patch.set_alpha(0.0)  # Make axis background transparent
        
        if p_values:
            # Line plot for p-values
            ax1.plot(param_values, p_values, 'o-', color=PASTEL_COLORS[0])
            
            # Add significance line
            ax1.axhline(0.05, color=PASTEL_COLORS[4], linestyle='--', label='α = 0.05')
            
            # Add labels
            ax1.set_xlabel(param_name)
            ax1.set_ylabel('p-value')
            ax1.set_title('p-value vs. Parameter')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "p-values not available", 
                   ha='center', va='center', transform=ax1.transAxes)
        
        # Second subplot: effect sizes
        ax2 = fig.add_subplot(222)
        ax2.patch.set_alpha(0.0)  # Make axis background transparent
        
        if effects:
            # Line plot for effect sizes
            ax2.plot(param_values, effects, 'o-', color=PASTEL_COLORS[1])
            
            # Add labels
            ax2.set_xlabel(param_name)
            ax2.set_ylabel('Effect Size')
            ax2.set_title('Effect Size vs. Parameter')
        else:
            ax2.text(0.5, 0.5, "Effect sizes not available", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        # Third subplot: R-squared
        ax3 = fig.add_subplot(223)
        ax3.patch.set_alpha(0.0)  # Make axis background transparent
        
        if r_squared:
            # Line plot for R-squared
            ax3.plot(param_values, r_squared, 'o-', color=PASTEL_COLORS[2])
            
            # Add labels
            ax3.set_xlabel(param_name)
            ax3.set_ylabel('R²')
            ax3.set_title('R² vs. Parameter')
        else:
            ax3.text(0.5, 0.5, "R² values not available", 
                   ha='center', va='center', transform=ax3.transAxes)
        
        # Fourth subplot: Combined visualization
        ax4 = fig.add_subplot(224)
        ax4.patch.set_alpha(0.0)  # Make axis background transparent
        
        # Plot all metrics on a normalized scale
        if p_values and effects:
            # Normalize to 0-1 scale
            norm_p = [(p - min(p_values)) / (max(p_values) - min(p_values)) if max(p_values) > min(p_values) else 0.5 for p in p_values]
            norm_effects = [(e - min(effects)) / (max(effects) - min(effects)) if max(effects) > min(effects) else 0.5 for e in effects]
            
            # Plot normalized values
            ax4.plot(param_values, norm_p, 'o-', color=PASTEL_COLORS[0], label='p-value (norm)')
            ax4.plot(param_values, norm_effects, 'o-', color=PASTEL_COLORS[1], label='Effect (norm)')
            
            if r_squared:
                norm_r2 = [(r - min(r_squared)) / (max(r_squared) - min(r_squared)) if max(r_squared) > min(r_squared) else 0.5 for r in r_squared]
                ax4.plot(param_values, norm_r2, 'o-', color=PASTEL_COLORS[2], label='R² (norm)')
            
            # Add labels
            ax4.set_xlabel(param_name)
            ax4.set_ylabel('Normalized Value')
            ax4.set_title('Normalized Metrics')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "Insufficient data for comparison", 
                   ha='center', va='center', transform=ax4.transAxes)
        
        fig.tight_layout()
        
        # Convert to SVG
        svg_string = fig_to_svg(fig)
        
        # Add visualization with maximize button
        self.add_visualization_with_maximize(parent_layout, svg_string, "Parameter Variation Analysis", fig)
    
    def add_visualization_with_maximize(self, parent_layout, svg_string, title, fig=None):
        """Add an SVG visualization with a maximize button."""
        # Create container
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header with title and maximize button
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 5)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        # Spacer
        header_layout.addStretch()
        
        # Maximize button
        maximize_button = QPushButton()
        maximize_button.setIcon(load_bootstrap_icon("arrows-fullscreen"))
        maximize_button.setToolTip("Maximize Visualization")
        maximize_button.setFixedSize(30, 30)
        maximize_button.clicked.connect(lambda: self.show_plot_modal(fig if fig else svg_string, title))
        header_layout.addWidget(maximize_button)
        
        container_layout.addWidget(header)
        
        # SVG widget
        svg_widget = QSvgWidget()
        svg_widget.renderer().load(QByteArray(svg_string.encode('utf-8')))
        svg_widget.setMinimumHeight(300)
        container_layout.addWidget(svg_widget)
        
        # Add container to parent layout
        parent_layout.addWidget(container)
    
    def save_results_to_study(self):
        """Save the sensitivity analysis results to the active study."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results to save")
            return
        
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        active_study = main_window.studies_manager.get_active_study()
        if not active_study:
            QMessageBox.warning(self, "Error", "No active study")
            return
        
        try:
            # Add the sensitivity analysis to the study
            if hasattr(main_window.studies_manager, 'add_sensitivity_analysis_to_study'):
                main_window.studies_manager.add_sensitivity_analysis_to_study(
                    sensitivity_result=self.analysis_results
                )
                QMessageBox.information(self, "Success", "Sensitivity analysis results saved to study")
                self.status_bar.showMessage("Results saved to study")
            else:
                # If the method doesn't exist, add as a generic result
                main_window.studies_manager.add_generic_results_to_study(
                    result_type='sensitivity_analysis',
                    result_details=self.analysis_results
                )
                QMessageBox.information(self, "Success", "Sensitivity analysis results saved as generic results")
                self.status_bar.showMessage("Results saved to study")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save results: {str(e)}")
    
    def fig_to_svg(self, fig):
        """Convert matplotlib figure to SVG string."""
        # Set figure and axes backgrounds to transparent
        fig.patch.set_alpha(0.0)
        for ax in fig.get_axes():
            ax.patch.set_alpha(0.0)
            # Make legend background transparent if it exists
            if ax.get_legend() is not None:
                ax.get_legend().get_frame().set_alpha(0.0)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
        buf.seek(0)
        svg_string = buf.getvalue().decode('utf-8')
        buf.close()
        plt.close(fig)
        return svg_string
    
    def show_plot_modal(self, figure, title="Plot"):
        """
        Open the given figure in a modal dialog that maximizes and preserves aspect ratio.
        
        Args:
            figure: A matplotlib Figure or SVG content (string).
            title (str): Title of the modal dialog.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        layout = QVBoxLayout(dialog)
        
        # Handle SVG figures
        if isinstance(figure, str):
            try:
                svg_widget = QSvgWidget()
                svg_widget.renderer().load(QByteArray(figure.encode('utf-8')))
                svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                aspect_widget = SVGAspectRatioWidget(svg_widget)
                layout.addWidget(aspect_widget)
            except Exception as e:
                error_label = QLabel(f"Error processing SVG: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        
        # Handle matplotlib figures
        elif isinstance(figure, Figure):
            # Ensure transparent background
            figure.patch.set_alpha(0.0)
            for ax in figure.get_axes():
                ax.patch.set_alpha(0.0)
            
            # Convert to SVG and display
            svg_string = fig_to_svg(figure)
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_string.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            layout.addWidget(aspect_widget)
        
        elif hasattr(figure, 'figure') and isinstance(figure.figure, Figure):
            # Ensure transparent background
            figure.figure.patch.set_alpha(0.0)
            for ax in figure.figure.get_axes():
                ax.patch.set_alpha(0.0)
            
            # Convert to SVG and display
            svg_string = fig_to_svg(figure.figure)
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_string.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            layout.addWidget(aspect_widget)
        
        else:
            placeholder = QLabel(f"Unsupported plot format: {type(figure).__name__}")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(placeholder)
        
        dialog.setLayout(layout)
        dialog.showMaximized()
        dialog.exec()

    async def share_results_with_llm(self):
        """Share results with LLM to get an interpretation."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results to interpret")
            return
        
        # Create a prompt from the results
        prompt = self.format_results_for_prompt()
        if not prompt:
            QMessageBox.warning(self, "Error", "Could not format results for interpretation")
            return
        
        # Show loading message
        self.status_bar.showMessage("Getting AI interpretation...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Get interpretation from LLM
            llm_interpretation = await call_llm_async(prompt, model=llm_config.default_text_model)
            
            # Save interpretation
            self.analysis_results['interpretation'] = {
                'prompt': prompt,
                'response': llm_interpretation,
                'timestamp': datetime.now().isoformat()
            }
            
            # Display interpretation
            self.display_llm_interpretation()
            
            self.status_bar.showMessage("AI interpretation complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get AI interpretation: {str(e)}")
            self.status_bar.showMessage("Interpretation failed with error")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def format_results_for_prompt(self):
        """Format sensitivity analysis results for inclusion in the LLM prompt."""
        
        # Get analysis-specific details
        analysis_type = self.analysis_results['analysis_type']
        results = self.analysis_results['results']
        
        # Common information across all analysis types
        summary = []
        summary.append("BASELINE RESULTS:")
        summary.append(f"Method: {results.get('baseline_results', {}).get('method', 'N/A')}")
        summary.append(f"p-value: {results.get('baseline_results', {}).get('p_value', 'N/A')}")
        summary.append(f"Effect Size: {results.get('baseline_results', {}).get('effect_size', 'N/A')}")
        summary.append(f"Interpretation: {results.get('baseline_results', {}).get('interpretation', 'N/A')}")
        summary.append("")
        
        summary.append("SENSITIVITY RESULTS:")
        sensitivity_results = results.get('sensitivity_results', [])
        for i, result in enumerate(sensitivity_results):
            summary.append(f"Scenario {i+1}: {result.get('scenario', 'N/A')}")
            summary.append(f"  p-value: {result.get('p_value', 'N/A')}")
            summary.append(f"  Effect Size: {result.get('effect_size', 'N/A')}")
            summary.append(f"  Change: {result.get('change', 'N/A')}")
            summary.append(f"  Significant Change: {'Yes' if result.get('significant_change', False) else 'No'}")
            summary.append("")
        
        # Add analysis-specific information
        if analysis_type == SensitivityAnalysisType.MISSING_DATA.value:
            summary.append("MISSING DATA ANALYSIS SPECIFICS:")
            summary.append(f"Missing Data Method: {self.missing_method_combo.currentText()}")
            summary.append(f"Missing Threshold: {self.missing_threshold_spinner.value()}%")
            summary.append(f"Simulated Missing Data: {'Yes' if self.simulate_missing_check.isChecked() else 'No'}")
            if self.simulate_missing_check.isChecked():
                summary.append(f"Missing Pattern: {self.missing_pattern_combo.currentText()}")
                summary.append(f"Missing Rate: {self.missing_rate_slider.value()}%")
        
        elif analysis_type == SensitivityAnalysisType.OUTLIER_DETECTION.value:
            summary.append("OUTLIER DETECTION ANALYSIS SPECIFICS:")
            summary.append(f"Detection Method: {self.outlier_method_combo.currentText()}")
            summary.append(f"Outlier Threshold: {self.outlier_threshold_spinner.value()}")
            summary.append(f"Outlier Handling: {self.outlier_handling_combo.currentText()}")
            summary.append(f"Compared With/Without Outliers: {'Yes' if self.compare_outliers_check.isChecked() else 'No'}")
            
            # Add outlier count if available
            if 'outliers' in results.get('visualization_data', {}):
                outlier_data = results['visualization_data']['outliers']
                summary.append(f"Outliers Detected: {outlier_data.get('count', 'N/A')} ({outlier_data.get('percent', 'N/A'):.1f}%)")
        
        elif analysis_type == SensitivityAnalysisType.ALTERNATIVE_MODEL.value:
            summary.append("ALTERNATIVE MODEL ANALYSIS SPECIFICS:")
            summary.append(f"Current Model: {self.current_model_combo.currentText()}")
            
            # Get selected alternative models
            alt_models = [model for model, checkbox in self.alt_model_checks.items() if checkbox.isChecked()]
            summary.append(f"Alternative Models Tested: {', '.join(alt_models) if alt_models else 'None'}")
            
            # Count models showing significant changes
            significant_count = sum(1 for result in sensitivity_results if result.get('significant_change', False))
            total_models = len(sensitivity_results)
            summary.append(f"Models Showing Significant Changes: {significant_count}/{total_models}")
        
        elif analysis_type == SensitivityAnalysisType.PARAMETER_VARIATION.value:
            summary.append("PARAMETER VARIATION ANALYSIS SPECIFICS:")
            summary.append(f"Parameter Varied: {self.parameter_combo.currentText()}")
            summary.append(f"Parameter Range: {self.start_value_spinner.value()} to {self.end_value_spinner.value()}")
            summary.append(f"Number of Steps: {self.num_steps_spinner.value()}")
            summary.append(f"Multiple Testing Correction: {self.correction_combo.currentText()}")
            summary.append(f"Significance Threshold: {self.significance_threshold_spinner.value()}")
        
        # Add the assessment, summary and recommendation
        summary.append("\nASSESSMENT:")
        summary.append(results.get('robustness_assessment', 'No assessment available.'))
        
        summary.append("\nSUMMARY:")
        summary.append(results.get('summary', 'No summary available.'))
        
        summary.append("\nRECOMMENDATION:")
        summary.append(results.get('recommendation', 'No recommendation available.'))
        
        return "\n".join(summary)

    def export_results_as_json(self):
        """Export sensitivity analysis results as a JSON file."""
        if not self.analysis_results:
            QMessageBox.warning(self, "Error", "No analysis results to export")
            return
        
        try:
            import json
            from datetime import datetime
            import os
            
            # Get timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = self.analysis_results['analysis_type'].replace(" ", "_").lower()
            default_filename = f"sensitivity_{analysis_type}_{timestamp}.json"
            
            # Get save path
            main_window = self.window()
            if hasattr(main_window, 'get_export_path'):
                file_path = main_window.get_export_path(default_filename, file_filter="JSON Files (*.json)")
            else:
                from PyQt6.QtWidgets import QFileDialog
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Save JSON Results", default_filename, "JSON Files (*.json)"
                )
            
            if not file_path:
                return  # User cancelled
            
            # Format the results in a more readable structure
            export_data = {
                "meta": {
                    "analysis_type": self.analysis_results['analysis_type'],
                    "dataset": self.analysis_results['dataset'],
                    "timestamp": self.analysis_results['timestamp'],
                    "export_date": datetime.now().isoformat()
                },
                "variables": {
                    "outcome": self.analysis_results['outcome'],
                    "predictor": self.analysis_results['predictor'],
                    "additional": self.analysis_results['additional_vars']
                },
                "results": self.analysis_results['results'],
            }
            
            # Add LLM interpretation if available
            if 'llm_interpretation' in self.analysis_results:
                export_data["llm_interpretation"] = self.analysis_results['llm_interpretation']
            
            # Save to file with pretty formatting
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.status_bar.showMessage(f"Results exported to {file_path}")
            QMessageBox.information(self, "Success", f"Results successfully exported to {os.path.basename(file_path)}")
            
            return file_path
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def display_llm_interpretation(self):
        """Display the LLM interpretation in a formatted dialog."""
        if not self.analysis_results or 'llm_interpretation' not in self.analysis_results:
            QMessageBox.warning(self, "Error", "No LLM interpretation available")
            return
        
        interpretation = self.analysis_results['llm_interpretation']
        
        # Create a dialog to display the interpretation
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Interpretation of Sensitivity Analysis")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Create tabs for different parts of the interpretation
        tabs = QTabWidget()
        
        # Summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # Add robustness assessment with appropriate styling based on level
        robustness = interpretation.get('robustness', 'Unknown')
        robustness_label = QLabel(f"Robustness Assessment: {robustness}")
        robustness_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            background-color: {
                '#d4edda' if 'high' in robustness.lower() else
                '#fff3cd' if 'moderate' in robustness.lower() else
                '#f8d7da' if 'low' in robustness.lower() else
                '#e2e3e5'
            };
            border-radius: 4px;
        """)
        summary_layout.addWidget(robustness_label)
        
        # Add short summary
        short_summary = QLabel(interpretation.get('short_summary', 'No summary available'))
        short_summary.setWordWrap(True)
        short_summary.setStyleSheet("font-size: 14px; margin: 10px 0;")
        summary_layout.addWidget(short_summary)
        
        # Add detailed interpretation
        interpretation_text = QPlainTextEdit()
        interpretation_text.setPlainText(interpretation.get('interpretation', 'No interpretation available'))
        interpretation_text.setReadOnly(True)
        summary_layout.addWidget(interpretation_text)
        
        tabs.addTab(summary_tab, "Summary")
        
        # Findings tab
        findings_tab = QWidget()
        findings_layout = QVBoxLayout(findings_tab)
        
        key_findings_label = QLabel("Key Findings")
        key_findings_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        findings_layout.addWidget(key_findings_label)
        
        key_findings = interpretation.get('key_findings', [])
        findings_list = QVBoxLayout()
        for i, finding in enumerate(key_findings):
            finding_label = QLabel(f"{i+1}. {finding}")
            finding_label.setWordWrap(True)
            finding_label.setStyleSheet("margin-bottom: 8px;")
            findings_list.addWidget(finding_label)
        
        findings_layout.addLayout(findings_list)
        findings_layout.addStretch()
        
        tabs.addTab(findings_tab, "Key Findings")
        
        # Recommendations tab
        recommendations_tab = QWidget()
        recommendations_layout = QVBoxLayout(recommendations_tab)
        
        recommendations_label = QLabel("Recommendations")
        recommendations_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        recommendations_layout.addWidget(recommendations_label)
        
        recommendations = interpretation.get('recommendations', [])
        recommendations_list = QVBoxLayout()
        for i, recommendation in enumerate(recommendations):
            rec_label = QLabel(f"{i+1}. {recommendation}")
            rec_label.setWordWrap(True)
            rec_label.setStyleSheet("margin-bottom: 8px;")
            recommendations_list.addWidget(rec_label)
        
        recommendations_layout.addLayout(recommendations_list)
        recommendations_layout.addStretch()
        
        tabs.addTab(recommendations_tab, "Recommendations")
        
        # Limitations tab
        limitations_tab = QWidget()
        limitations_layout = QVBoxLayout(limitations_tab)
        
        limitations_label = QLabel("Limitations")
        limitations_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        limitations_layout.addWidget(limitations_label)
        
        limitations = interpretation.get('limitations', [])
        limitations_list = QVBoxLayout()
        for i, limitation in enumerate(limitations):
            limit_label = QLabel(f"{i+1}. {limitation}")
            limit_label.setWordWrap(True)
            limit_label.setStyleSheet("margin-bottom: 8px;")
            limitations_list.addWidget(limit_label)
        
        limitations_layout.addLayout(limitations_list)
        limitations_layout.addStretch()
        
        tabs.addTab(limitations_tab, "Limitations")
        
        # Visualization notes tab
        if 'visualization_notes' in interpretation:
            viz_tab = QWidget()
            viz_layout = QVBoxLayout(viz_tab)
            
            viz_label = QLabel("Visualization Notes")
            viz_label.setStyleSheet("font-size: 14px; font-weight: bold;")
            viz_layout.addWidget(viz_label)
            
            viz_notes = QPlainTextEdit()
            viz_notes.setPlainText(interpretation.get('visualization_notes', ''))
            viz_notes.setReadOnly(True)
            viz_layout.addWidget(viz_notes)
            
            tabs.addTab(viz_tab, "Visualization Notes")
        
        layout.addWidget(tabs)
        
        # Add button row
        button_row = QHBoxLayout()
        button_row.addStretch()
        
        export_button = QPushButton("Export as JSON")
        export_button.clicked.connect(self.export_results_as_json)
        button_row.addWidget(export_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_row.addWidget(close_button)
        
        layout.addLayout(button_row)
        
        dialog.setLayout(layout)
        dialog.exec()

    async def share_and_display_interpretation(self):
        """Share results with LLM and display interpretation."""
        results = await self.share_results_with_llm()
        if results:
            # Instead of showing a dialog, just make sure the interpretation tab is visible
            left_tabs = self.findChild(QTabWidget)
            if left_tabs:
                # Find the index of the interpretation tab
                for i in range(left_tabs.count()):
                    if left_tabs.tabText(i) == "Interpretation":
                        left_tabs.setCurrentIndex(i)
                        break

    def update_interpretation_tab(self):
        """Update the interpretation tab with the LLM-generated interpretation."""
        if not self.analysis_results or 'llm_interpretation' not in self.analysis_results:
            return
        
        # Clear the existing layout content
        self.clear_layout(self.interpretation_layout)
        
        interpretation = self.analysis_results['llm_interpretation']
        
        # Add a title and short summary section
        title_label = QLabel("AI Interpretation of Sensitivity Analysis")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.interpretation_layout.addWidget(title_label)
        
        # Add robustness assessment
        robustness = interpretation.get('robustness', 'Unknown')
        robustness_label = QLabel(f"Robustness Assessment: {robustness}")
        robustness_label.setStyleSheet("font-weight: bold;")
        self.interpretation_layout.addWidget(robustness_label)
        
        # Add short summary
        short_summary = QLabel(interpretation.get('short_summary', 'No summary available'))
        short_summary.setWordWrap(True)
        self.interpretation_layout.addWidget(short_summary)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.interpretation_layout.addWidget(separator)
        
        # Add detailed interpretation section
        interp_title = QLabel("Detailed Interpretation")
        interp_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.interpretation_layout.addWidget(interp_title)
        
        interpretation_text = QPlainTextEdit()
        interpretation_text.setPlainText(interpretation.get('interpretation', 'No interpretation available'))
        interpretation_text.setReadOnly(True)
        interpretation_text.setMaximumHeight(150)
        self.interpretation_layout.addWidget(interpretation_text)
        
        # Add key findings section
        findings_title = QLabel("Key Findings")
        findings_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.interpretation_layout.addWidget(findings_title)
        
        key_findings = interpretation.get('key_findings', [])
        for i, finding in enumerate(key_findings):
            finding_label = QLabel(f"{i+1}. {finding}")
            finding_label.setWordWrap(True)
            self.interpretation_layout.addWidget(finding_label)
        
        # Add recommendations section
        recommendations_title = QLabel("Recommendations")
        recommendations_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.interpretation_layout.addWidget(recommendations_title)
        
        recommendations = interpretation.get('recommendations', [])
        for i, recommendation in enumerate(recommendations):
            rec_label = QLabel(f"{i+1}. {recommendation}")
            rec_label.setWordWrap(True)
            self.interpretation_layout.addWidget(rec_label)
        
        # Add limitations section
        limitations_title = QLabel("Limitations")
        limitations_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.interpretation_layout.addWidget(limitations_title)
        
        limitations = interpretation.get('limitations', [])
        for i, limitation in enumerate(limitations):
            limit_label = QLabel(f"{i+1}. {limitation}")
            limit_label.setWordWrap(True)
            self.interpretation_layout.addWidget(limit_label)
        
        # Add visualization notes if available
        if 'visualization_notes' in interpretation:
            viz_title = QLabel("Visualization Notes")
            viz_title.setStyleSheet("font-size: 14px; font-weight: bold;")
            self.interpretation_layout.addWidget(viz_title)
            
            viz_notes = QPlainTextEdit()
            viz_notes.setPlainText(interpretation.get('visualization_notes', ''))
            viz_notes.setReadOnly(True)
            viz_notes.setMaximumHeight(100)
            self.interpretation_layout.addWidget(viz_notes)
        
        # Add spacer at the bottom
        self.interpretation_layout.addStretch()

    def prompt_for_outcome_variable(self, suggested_outcome=None):
        """Prompt the user to manually select an outcome variable."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Outcome Variable")
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Select the outcome (dependent) variable for your analysis:")
        layout.addWidget(label)
        
        combo = QComboBox()
        for i in range(self.outcome_variable_combo.count()):
            combo.addItem(self.outcome_variable_combo.itemText(i))
        
        # Set suggested outcome if provided
        if suggested_outcome:
            index = combo.findText(suggested_outcome)
            if index >= 0:
                combo.setCurrentIndex(index)
                
            # Add label showing AI suggestion
            suggestion_label = QLabel(f"AI suggestion: {suggested_outcome}")
            suggestion_label.setStyleSheet("color: blue; font-style: italic;")
            layout.addWidget(suggestion_label)
        
        layout.addWidget(combo)
        
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)
        
        if dialog.exec():
            return combo.currentText()
        
        return None

    async def auto_select_variables(self):
        """AI-based selection of appropriate variables for analysis."""
        df = self.current_dataframe
        
        # Check if user has already selected an outcome
        current_outcome = self.outcome_variable_combo.currentText()
        has_valid_outcome = current_outcome != "Select..." and current_outcome in df.columns
        
        # If manual outcome selection is enabled, or user already has a valid selection
        if self.manual_outcome_selection or has_valid_outcome:
            if has_valid_outcome:
                # Use the existing outcome
                outcome = current_outcome
            else:
                # Just get a suggestion for the outcome variable and prompt user
                outcome = await self.suggest_outcome_variable()
                if not outcome:  # User cancelled or error occurred
                    return False
        else:
            # Do full automatic variable selection
            variables = await self.get_all_variable_suggestions()
            if not variables:  # Error occurred
                return False
                
            outcome = variables["outcome"]
            predictor = variables["predictor"]
            additional_vars = variables["additional_vars"]
            explanation = variables["explanation"]
            
            # Validate that selected variables exist in the dataset
            error_message = self.validate_variable_suggestions(outcome, predictor, additional_vars)
            
            if error_message:
                QMessageBox.warning(self, "Variable Selection Error", error_message + "\nPlease select variables manually.")
                return False
            
            # Set predictor variable
            predictor_index = self.predictor_variable_combo.findText(predictor)
            if predictor_index >= 0:
                self.predictor_variable_combo.setCurrentIndex(predictor_index)
            
            # Set additional variables
            if additional_vars:
                self.additional_variables_list.setPlainText('\n'.join(additional_vars))
            
            # Show variable selection explanation
            if explanation:
                QMessageBox.information(self, "AI Variable Selection", 
                                      f"The following variables have been selected for your sensitivity analysis:\n\n"
                                      f"Outcome: {outcome}\n"
                                      f"Predictor: {predictor}\n"
                                      f"Controls: {', '.join(additional_vars) if additional_vars else 'None'}\n\n"
                                      f"Explanation: {explanation}")
        
        # Set outcome variable if one was determined (either manually or automatically)
        if outcome and outcome != current_outcome:
            outcome_index = self.outcome_variable_combo.findText(outcome)
            if outcome_index >= 0:
                self.outcome_variable_combo.setCurrentIndex(outcome_index)
        
        self.status_bar.showMessage("Variable selection completed")
        return True

    async def suggest_outcome_variable(self):
        """Use LLM to suggest the most appropriate outcome variable."""
        if not self.current_dataframe is not None:
            return None
            
        # Get dataframe info
        df = self.current_dataframe
        df_info = df.describe().to_string()
        
        # Create prompt for LLM
        variables_prompt = f"""
        Analyze the following dataset and identify the most appropriate outcome variable for a sensitivity analysis.
        
        Dataset columns:
        {', '.join(df.columns.tolist())}
        
        Dataset statistics:
        {df_info}
        
        Based on this data, what is the most appropriate outcome variable?
        Return just the name of the column that would be most appropriate as an outcome variable.
        """
        
        try:
            # Call LLM to get suggestions
            variable_response = await call_llm_async(variables_prompt, model=llm_config.default_text_model)
            
            # Extract variable name - look for exact column matches
            suggested_outcome = None
            for line in variable_response.split('\n'):
                line = line.strip()
                if line in df.columns:
                    suggested_outcome = line
                    break
            
            # If no exact match found, try to extract any mention of column names
            if not suggested_outcome:
                for col in df.columns:
                    if col in variable_response:
                        suggested_outcome = col
                        break
            
            return suggested_outcome
            
        except Exception as e:
            print(f"Error getting outcome variable suggestion: {str(e)}")
            return None

    async def get_all_variable_suggestions(self):
        """Use LLM to suggest outcome, predictor, and additional variables."""
        if self.current_dataframe is None:
            return None, None, []
            
        # Create prompt for LLM
        df = self.current_dataframe
        columns_info = {col: str(df[col].dtype) for col in df.columns}
        
        variables_prompt = f"""
        Please analyze this dataset and suggest appropriate variables for a sensitivity analysis.
        
        Dataset columns and types:
        {json.dumps(columns_info, indent=2)}
        
        Based on this data, suggest:
        1. The most appropriate outcome variable (dependent variable)
        2. The most important predictor variable (independent variable)
        3. Up to 3 additional variables that might be relevant
        
        Format your response as a JSON object with keys: outcome, predictor, additional_vars
        """
        
        try:
            # Call LLM
            variable_response = await call_llm_async(variables_prompt, model=llm_config.default_text_model)
            
            # Try to parse the response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', variable_response, re.DOTALL)
            if json_match:
                variables_json = json.loads(json_match.group(1))
            else:
                # Try to extract JSON without markdown
                json_match = re.search(r'\{.*\}', variable_response, re.DOTALL)
                if json_match:
                    variables_json = json.loads(json_match.group(0))
                else:
                    # Simple parsing as last resort
                    variables_json = {}
                    if "outcome" in variable_response.lower():
                        # Extract line with "outcome" and try to get variable name
                        for line in variable_response.split('\n'):
                            if "outcome" in line.lower() or "dependent" in line.lower():
                                for col in df.columns:
                                    if col in line:
                                        variables_json["outcome"] = col
                                        break
                    
                    if "predictor" in variable_response.lower():
                        # Extract line with "predictor" and try to get variable name
                        for line in variable_response.split('\n'):
                            if "predictor" in line.lower() or "independent" in line.lower():
                                for col in df.columns:
                                    if col in line and col != variables_json.get("outcome", ""):
                                        variables_json["predictor"] = col
                                        break
                    
                    variables_json["additional_vars"] = []
                    for col in df.columns:
                        if col != variables_json.get("outcome", "") and col != variables_json.get("predictor", ""):
                            if len(variables_json["additional_vars"]) < 3:
                                variables_json["additional_vars"].append(col)
            
            # Extract the variables
            outcome = variables_json.get("outcome")
            predictor = variables_json.get("predictor")
            additional_vars = variables_json.get("additional_vars", [])
            
            # Validate they exist in the dataframe
            if outcome and outcome not in df.columns:
                outcome = None
            if predictor and predictor not in df.columns:
                predictor = None
            additional_vars = [var for var in additional_vars if var in df.columns]
            
            return outcome, predictor, additional_vars
            
        except Exception as e:
            print(f"Error getting variable suggestions: {str(e)}")
            return None, None, []

    def validate_variable_suggestions(self, outcome, predictor, additional_vars):
        """Validate that suggested variables exist in the dataset."""
        df = self.current_dataframe
        error_message = ""
        
        if outcome not in df.columns:
            error_message += f"Recommended outcome variable '{outcome}' not found in dataset. "
        if predictor not in df.columns:
            error_message += f"Recommended predictor variable '{predictor}' not found in dataset. "
        for var in additional_vars:
            if var not in df.columns:
                error_message += f"Recommended control variable '{var}' not found in dataset. "
        
        return error_message

    def clear_all(self):
        """Clear all selections and results."""
        # Reset variable selections
        self.outcome_variable_combo.setCurrentIndex(0)
        self.predictor_variable_combo.setCurrentIndex(0)
        self.additional_variables_list.clear()
        
        # Reset analysis type to default
        self.analysis_type_combo.setCurrentIndex(0)
        
        # Clear results
        self.results_text.clear()
        self.details_table.setRowCount(0)
        
        # Clear interpretation
        self.clear_layout(self.interpretation_layout)
        new_placeholder = QLabel("No AI interpretation available yet. Click the 'Interpret' button to generate one.")
        new_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        new_placeholder.setStyleSheet("font-style: italic; color: gray;")
        self.interpretation_layout.addWidget(new_placeholder)
        self.interpretation_placeholder = new_placeholder
        
        # Clear visualizations
        for i in reversed(range(self.visualization_layout.count())):
            widget = self.visualization_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Add visualization placeholder
        self.visualization_placeholder = QLabel("Visualizations will be shown here")
        self.visualization_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_placeholder.setStyleSheet("font-style: italic; color: gray;")
        self.visualization_layout.addWidget(self.visualization_placeholder)
        
        # Reset internal state
        self.analysis_results = {}
        
        self.status_bar.showMessage("All selections and results cleared")

    async def suggest_analysis_type_and_prompt(self):
        """Get AI suggestion for analysis type and let user choose."""
        # Get AI suggestion
        suggested_type = self.suggest_analysis_type()
        
        # Create dialog for user selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Analysis Type")
        layout = QVBoxLayout(dialog)
        
        suggestion_label = QLabel(f"AI suggests: <b>{suggested_type.value}</b> analysis")
        suggestion_label.setStyleSheet("color: blue;")
        layout.addWidget(suggestion_label)
        
        info_label = QLabel(self.get_analysis_description(suggested_type))
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(info_label)
        
        layout.addWidget(QLabel("Select the type of sensitivity analysis to perform:"))
        
        # Create radio buttons for all analysis types
        radio_group = QButtonGroup(dialog)
        analysis_types = [(i, self.analysis_type_combo.itemText(i), self.analysis_type_combo.itemData(i)) 
                         for i in range(self.analysis_type_combo.count())]
        
        # Find index of suggested type
        suggested_index = -1
        for i, (_, _, analysis_type) in enumerate(analysis_types):
            if analysis_type == suggested_type:
                suggested_index = i
                break
        
        # Add radio buttons
        for i, (_, text, data) in enumerate(analysis_types):
            radio = QRadioButton(text)
            radio.setProperty("data", data)
            if i == suggested_index:  # Select the suggested type by default
                radio.setChecked(True)
            radio_group.addButton(radio)
            layout.addWidget(radio)
        
        description_label = QLabel()
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-style: italic; font-size: 11px;")
        layout.addWidget(description_label)
        
        # Update description when selection changes
        def update_description():
            for button in radio_group.buttons():
                if button.isChecked():
                    analysis_type = button.property("data")
                    description_label.setText(self.get_analysis_description(analysis_type))
                    break
        
        for button in radio_group.buttons():
            button.toggled.connect(update_description)
        
        # Initialize description
        update_description()
        
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_box.addStretch()
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)
        
        if dialog.exec():
            # Find the selected analysis type
            for button in radio_group.buttons():
                if button.isChecked():
                    return button.property("data")
        
        return None

    async def auto_configure_analysis_options(self):
        """Configure analysis options based on the selected analysis type."""
        # Get the current selections
        df = self.current_dataframe
        outcome = self.outcome_variable_combo.currentText()
        predictor = self.predictor_variable_combo.currentText()
        additional_vars = self.additional_variables_list.toPlainText().strip().split('\n')
        if additional_vars == ['']:
            additional_vars = []
        
        # Skip the rest if we don't have valid variables selected
        if outcome == "Select..." or predictor == "Select...":
            QMessageBox.warning(self, "Incomplete Selection", 
                              "Please select both outcome and predictor variables before configuring the analysis.")
            self.status_bar.showMessage("Auto-configuration cancelled - incomplete variable selection")
            return False
        
        # Get the current analysis type
        analysis_type = self.analysis_type_combo.currentData()
        
        # Prepare data statistics for the prompt
        data_info = self.prepare_data_statistics(outcome, predictor, additional_vars)
        
        # Prepare prompt based on the specific analysis type
        config_prompt = self.get_analysis_config_prompt(analysis_type, outcome, predictor, additional_vars, data_info)
        
        # Second LLM call for configuration recommendations
        config_response = await call_llm_async(config_prompt, model=llm_config.default_text_model)
        
        # Parse configuration recommendations based on analysis type
        try:
            success = self.parse_and_apply_config(config_response, analysis_type)
            
            if success:
                self.status_bar.showMessage(f"AI configuration for {analysis_type.value} sensitivity analysis applied successfully")
                return True
            else:
                self.status_bar.showMessage("AI configuration failed - could not apply settings")
                return False
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not configure analysis options: {str(e)}")
            self.status_bar.showMessage("AI configuration failed with error")
            import traceback
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
        default_size = svg_widget.renderer().defaultSize()
        self.aspect_ratio = default_size.width() / default_size.height() if default_size.height() > 0 else 1
    
    def resizeEvent(self, event):
        """Maintain aspect ratio during resize."""
        super().resizeEvent(event)
        width = event.size().width()
        height = event.size().height()
        new_height = width / self.aspect_ratio
        if new_height > height:
            new_width = height * self.aspect_ratio
            new_height = height
        else:
            new_width = width
        x = (width - new_width) / 2
        y = (height - new_height) / 2
        self.svg_widget.setGeometry(int(x), int(y), int(new_width), int(new_height))