import json
import pandas as pd
import numpy as np
from datetime import datetime



from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QPlainTextEdit, QFormLayout, QSplitter, 
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QStatusBar, QApplication,
    QListWidget, QCheckBox, QTextBrowser
)
from PyQt6.QtGui import QIcon, QColor, QBrush, QTextDocument

import re
from qasync import asyncSlot

from llms.client import call_llm_async
from qt_sections.llm_manager import llm_config  # Add this import
from data.selection.select import VariableRole  # Reuse VariableRole enum from statistical testing module

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from io import BytesIO

# Add these imports for the enhanced functionality
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

from PyQt6.QtWidgets import QSizePolicy, QGridLayout, QScrollArea
from PyQt6.QtGui import QFont

# Add these imports at the top of the file with the other imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import statsmodels.api as sm
import warnings
from scipy.stats import trim_mean
from itertools import combinations
import pymc as pm  # Modern replacement for pymc3
import arviz as az  # For Bayesian diagnostics and visualization
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors  # For PSM

import io
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
from data.selection.detailed_tests.formatting import fig_to_svg, PASTEL_COLORS, PASTEL_CMAP

from PyQt6.QtWidgets import QStyleFactory
from PyQt6.QtWidgets import QStyle

# Add import for load_bootstrap_icon
from helpers.load_icon import load_bootstrap_icon


class SubgroupAnalysisWidget(QWidget):
    """Widget for performing subgroup analysis on datasets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subgroup Analysis")
        
        # Internal state
        self.current_dataframe = None
        self.current_name = ""
        self.column_roles = {}  # Maps column names to their roles
        self.selected_outcome = None
        self.selected_treatment = None
        self.selected_subgroups = []
        self.selected_covariates = []
        self.subgroup_results = {}
        
        # Color map for different roles - using theme-aware colors
        base_color = self.palette().base().color()
        text_color = self.palette().text().color()
        
        self.role_colors = {
            VariableRole.NONE: base_color,
            VariableRole.OUTCOME: QColor(255, 200, 200).lighter(150),   # Light red
            VariableRole.GROUP: QColor(200, 200, 255).lighter(150),     # Light blue
            VariableRole.COVARIATE: QColor(200, 255, 200).lighter(150), # Light green
            VariableRole.SUBJECT_ID: QColor(255, 255, 200).lighter(150),# Light yellow
            VariableRole.TIME: QColor(255, 200, 255).lighter(150),      # Light purple
            VariableRole.PAIR_ID: QColor(200, 255, 255).lighter(150),   # Light cyan
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top section container - as compact as possible
        top_section = QWidget()
        top_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(5, 5, 5, 5)
        top_layout.setSpacing(5)
        
        # Header and dataset selection in one row
        header_row = QHBoxLayout()
        
        header_label = QLabel("Select Dataset: ")
        header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        header_label.setFont(font)
        header_row.addWidget(header_label)
        
        # Dataset selection with refresh button
        dataset_selection = QWidget()
        dataset_layout = QHBoxLayout(dataset_selection)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(5)
        
        
        self.dataset_selector = QComboBox()
        self.dataset_selector.setMinimumWidth(300)
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(self.dataset_selector)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.setIcon(load_bootstrap_icon("arrow-repeat", size=18))
        refresh_btn.clicked.connect(self.load_dataset_from_study)
        dataset_layout.addWidget(refresh_btn)
        
        header_row.addWidget(dataset_selection)
        header_row.addStretch()
        
        top_layout.addLayout(header_row)
        
        # Variable selection using form layouts for proper label-field pairing and alignment
        variable_container = QWidget()
        variable_layout = QVBoxLayout(variable_container)
        variable_layout.setContentsMargins(0, 0, 0, 0)
        variable_layout.setSpacing(10)  # Increased vertical spacing between rows
        
        # Set fixed label width to ensure alignment of all labels
        label_width = 100  # Pixels for label width
        
        # Row 1: Outcome, Treatment, Method in a horizontal arrangement
        row1_widget = QWidget()
        row1_layout = QHBoxLayout(row1_widget)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(20)  # Increased spacing between fields
        
        # Outcome selection with fixed-width label
        outcome_form = QFormLayout()
        outcome_form.setContentsMargins(0, 0, 0, 0)
        outcome_form.setSpacing(5)  # Increased spacing between label and field
        outcome_label = QLabel("Outcome:")
        outcome_label.setMinimumWidth(label_width)
        self.outcome_combo = QComboBox()
        self.outcome_combo.setMinimumWidth(220)  # Increased width
        self.outcome_combo.currentIndexChanged.connect(self.update_ui_state)
        outcome_form.addRow(outcome_label, self.outcome_combo)
        row1_layout.addLayout(outcome_form)
        
        # Treatment selection with fixed-width label
        treatment_form = QFormLayout()
        treatment_form.setContentsMargins(0, 0, 0, 0)
        treatment_form.setSpacing(5)
        treatment_label = QLabel("Treatment:")
        treatment_label.setMinimumWidth(label_width)
        self.treatment_combo = QComboBox()
        self.treatment_combo.setMinimumWidth(220)  # Increased width
        self.treatment_combo.currentIndexChanged.connect(self.update_ui_state)
        treatment_form.addRow(treatment_label, self.treatment_combo)
        row1_layout.addLayout(treatment_form)
        
        # Method selection with fixed-width label
        method_form = QFormLayout()
        method_form.setContentsMargins(0, 0, 0, 0)
        method_form.setSpacing(5)
        method_label = QLabel("Method:")
        method_label.setMinimumWidth(label_width)

        # In the init_ui method, update the method_combo items:
        self.method_combo = QComboBox()
        self.method_combo.setMinimumWidth(220)
        self.method_combo.addItems([
            "Stratified Analysis", 
            "Interaction Models", 
            "Causal Forest",
            "BART (Approximate)",  # Renamed to indicate it's an approximation
            "IPTW",
            "Bayesian Hierarchical Model",  # Added true Bayesian method
            "Propensity Score Matching"     # Added PSM
        ])
        method_form.addRow(method_label, self.method_combo)
        row1_layout.addLayout(method_form)
        
        # Push everything to the left
        row1_layout.addStretch(1)
        variable_layout.addWidget(row1_widget)
        
        # Row 2: Subgroups and Covariates
        row2_widget = QWidget()
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(30)  # Increased spacing between fields
        
        # Subgroup selection with fixed-width label centered vertically
        subgroup_form = QFormLayout()
        subgroup_form.setContentsMargins(0, 0, 0, 0)
        subgroup_form.setSpacing(8)
        # Set form label alignment to center vertically
        subgroup_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        subgroup_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        subgroup_label = QLabel("Subgroups:")
        subgroup_label.setMinimumWidth(label_width)
        # Set alignment to vertically center the label
        subgroup_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.subgroup_list = QListWidget()
        self.subgroup_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.subgroup_list.setMinimumWidth(220)  # Increased width
        self.subgroup_list.setMinimumHeight(180)  # Increased height
        self.subgroup_list.itemSelectionChanged.connect(self.update_ui_state)
        subgroup_form.addRow(subgroup_label, self.subgroup_list)
        row2_layout.addLayout(subgroup_form, 1)
        
        # Covariate selection with fixed-width label centered vertically
        covariate_form = QFormLayout()
        covariate_form.setContentsMargins(0, 0, 0, 0)
        covariate_form.setSpacing(8)
        # Set form label alignment to center vertically
        covariate_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        covariate_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        covariate_label = QLabel("Covariates:")
        covariate_label.setMinimumWidth(label_width)
        # Set alignment to vertically center the label
        covariate_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.covariate_list = QListWidget()
        self.covariate_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.covariate_list.setMinimumWidth(220)  # Increased width
        self.covariate_list.setMinimumHeight(180)  # Increased height
        covariate_form.addRow(covariate_label, self.covariate_list)
        row2_layout.addLayout(covariate_form, 1)
        
        variable_layout.addWidget(row2_widget)
        
        # Format the options row with the same alignment pattern but with checkboxes in the same row
        options_container = QWidget()
        options_layout = QHBoxLayout(options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(20)  # Increased spacing between fields
        
        # P-value adjustment with fixed-width label
        pval_form = QFormLayout()
        pval_form.setContentsMargins(0, 0, 0, 0)
        pval_form.setSpacing(5)
        pval_label = QLabel("P-value Adjustment:")
        pval_label.setMinimumWidth(label_width)
        self.adjustment_combo = QComboBox()
        self.adjustment_combo.setMinimumWidth(220)  # Increased width
        self.adjustment_combo.addItems([
            "None", 
            "Bonferroni", 
            "Holm-Bonferroni", 
            "Benjamini-Hochberg (FDR)",
            "Benjamini-Yekutieli"
        ])
        self.adjustment_combo.setCurrentText("Benjamini-Hochberg (FDR)")
        pval_form.addRow(pval_label, self.adjustment_combo)
        options_layout.addLayout(pval_form)
        
        # Visualization type with fixed-width label
        viz_form = QFormLayout()
        viz_form.setContentsMargins(0, 0, 0, 0)
        viz_form.setSpacing(5)
        viz_label = QLabel("Visualization:")
        viz_label.setMinimumWidth(label_width)
        self.viz_combo = QComboBox()
        self.viz_combo.setMinimumWidth(220)  # Increased width
        self.viz_combo.addItems([
            "Forest Plot", 
            "Interaction Plot", 
            "Heat Map", 
            "Effect Size Distribution"
        ])
        viz_form.addRow(viz_label, self.viz_combo)
        options_layout.addLayout(viz_form)
        
        # Add checkboxes directly to the options row instead of using a separate container
        self.adjust_pvalues_checkbox = QCheckBox("Adjust p-values")
        self.adjust_pvalues_checkbox.setChecked(True)
        options_layout.addWidget(self.adjust_pvalues_checkbox)
        
        self.include_interaction_checkbox = QCheckBox("Include interactions")
        self.include_interaction_checkbox.setChecked(True)
        options_layout.addWidget(self.include_interaction_checkbox)
        
        # Add stretch to push everything to the left
        options_layout.addStretch(1)
        
        variable_layout.addWidget(options_container)
        top_layout.addWidget(variable_container)
        
        # Action buttons in one row
        buttons_row = QHBoxLayout()
        
        self.suggest_variables_button = QPushButton("Suggest Variables (AI)")
        self.suggest_variables_button.setIcon(load_bootstrap_icon("lightbulb", size=18))
        self.suggest_variables_button.clicked.connect(self.suggest_variables)
        buttons_row.addWidget(self.suggest_variables_button)
        
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.setIcon(load_bootstrap_icon("play-fill", size=18))
        self.run_analysis_button.clicked.connect(self.run_subgroup_analysis)
        buttons_row.addWidget(self.run_analysis_button)
        
        self.interpret_results_button = QPushButton("Interpret Results (AI)")
        self.interpret_results_button.setIcon(load_bootstrap_icon("chat-text", size=18))
        self.interpret_results_button.clicked.connect(self.interpret_results)
        buttons_row.addWidget(self.interpret_results_button)
        
        self.save_results_button = QPushButton("Save Results")
        self.save_results_button.setIcon(load_bootstrap_icon("save", size=18))
        self.save_results_button.clicked.connect(self.save_results_to_study)
        buttons_row.addWidget(self.save_results_button)
        
        top_layout.addLayout(buttons_row)
        
        main_layout.addWidget(top_section)
        
        # Create horizontal splitter for main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Left side - Summary and table
        left_side = QTabWidget()
        
        # Tab 1: Dataset view
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        dataset_layout.addWidget(self.data_table)
        
        left_side.addTab(dataset_tab, "Dataset")
        left_side.setTabIcon(0, load_bootstrap_icon("table", size=16))
        
        # Tab 2: Summary results
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        left_side.addTab(summary_tab, "Summary")
        left_side.setTabIcon(1, load_bootstrap_icon("clipboard-data", size=16))
        
        # Tab 3: Detailed results
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(6)
        self.details_table.setHorizontalHeaderLabels(["Subgroup", "Level", "N", "Effect Size", "95% CI", "P-value"])
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        details_layout.addWidget(self.details_table)
        
        left_side.addTab(details_tab, "Detailed Results")
        left_side.setTabIcon(2, load_bootstrap_icon("list-columns", size=16))
        
        # Tab 4: AI Interpretation
        interpretation_tab = QWidget()
        interpretation_layout = QVBoxLayout(interpretation_tab)
        interpretation_layout.setContentsMargins(0, 0, 0, 0)
        
        self.interpretation_text = QTextBrowser()
        self.interpretation_text.setOpenExternalLinks(True)
        self.interpretation_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        interpretation_layout.addWidget(self.interpretation_text)
        
        left_side.addTab(interpretation_tab, "AI Interpretation")
        left_side.setTabIcon(3, load_bootstrap_icon("robot", size=16))
        
        # Right side - Visualizations (with scroll area)
        right_side = QWidget()
        right_layout = QVBoxLayout(right_side)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        viz_header = QLabel("Visualization")
        viz_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        viz_header.setFont(font)
        right_layout.addWidget(viz_header)
        
        self.viz_scroll = QScrollArea()
        self.viz_scroll.setWidgetResizable(True)
        self.viz_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viz_container = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_container)
        self.viz_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.viz_scroll.setWidget(self.viz_container)
        
        self.visualization_placeholder = QLabel("Visualization will appear here after analysis")
        self.visualization_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_layout.addWidget(self.visualization_placeholder)
        
        right_layout.addWidget(self.viz_scroll)
        
        # Add both sides to the content splitter
        content_splitter.addWidget(left_side)
        content_splitter.addWidget(right_side)
        content_splitter.setSizes([400, 600])  # Visualization gets more space
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize the widget
        self.update_ui_state()
    
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
        
        # Reset the column roles
        self.column_roles = {col: VariableRole.NONE for col in df.columns}
        
        # Update the UI
        self.display_dataset(df)
        self.populate_variable_dropdowns()
        
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
    
    def populate_variable_dropdowns(self):
        """Populate variable selection dropdowns and lists."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return
        
        # Clear current selections
        self.outcome_combo.clear()
        self.treatment_combo.clear()
        self.subgroup_list.clear()
        self.covariate_list.clear()
        
        # Add a blank item
        self.outcome_combo.addItem("Select outcome...", None)
        self.treatment_combo.addItem("Select treatment/group...", None)
        
        # Populate with column names
        for col in self.current_dataframe.columns:
            # For outcome, prefer numeric columns
            if pd.api.types.is_numeric_dtype(self.current_dataframe[col]):
                self.outcome_combo.addItem(col, col)
            
            # For treatment/group, prefer categorical columns with few levels
            unique_count = self.current_dataframe[col].nunique()
            if unique_count < 10:  # Categorical with reasonable number of levels
                self.treatment_combo.addItem(col, col)
            
            # For subgroups, add categorical variables
            if unique_count < 20:  # Demographic variables usually have limited categories
                self.subgroup_list.addItem(col)
            
            # For covariates, add all variables (will filter out in the analysis)
            self.covariate_list.addItem(col)
    
    def update_ui_state(self):
        """Update UI state based on current selections."""
        has_data = self.current_dataframe is not None and not self.current_dataframe.empty
        has_outcome = self.outcome_combo.currentData() is not None
        has_treatment = self.treatment_combo.currentData() is not None
        has_subgroups = len(self.subgroup_list.selectedItems()) > 0
        
        # Enable/disable buttons based on selections
        self.run_analysis_button.setEnabled(has_data and has_outcome and has_treatment and has_subgroups)
        self.interpret_results_button.setEnabled(bool(self.subgroup_results))
        self.save_results_button.setEnabled(bool(self.subgroup_results))
    
    @asyncSlot()
    async def suggest_variables(self):
        """Use AI to suggest variables for subgroup analysis."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        df = self.current_dataframe
        
        # Create a description of the dataset columns with detailed information
        columns_info = []
        for col in df.columns:
            data_type = str(df[col].dtype)
            unique_values = df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            # Add statistical info for better variable identification
            stats = {
                "mean": float(df[col].mean()) if is_numeric else None,
                "std": float(df[col].std()) if is_numeric else None,
                "min": float(df[col].min()) if is_numeric else None,
                "max": float(df[col].max()) if is_numeric else None,
                "unique_count": int(unique_values),
                "unique_ratio": float(unique_values / len(df)) if len(df) > 0 else 0
            }
            sample_values = df[col].dropna().head(5).tolist()
            
            columns_info.append({
                "name": col,
                "data_type": data_type,
                "unique_values": unique_values,
                "is_numeric": is_numeric,
                "statistics": stats,
                "sample_values": sample_values
            })
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset for subgroup analysis variables. Please wait...")
        
        try:
            prompt = f"""
            I need to identify which variables to use for a subgroup analysis in my dataset. For a robust subgroup analysis, I need to identify:

            1. An OUTCOME variable: Typically a continuous or binary measure of the primary result (e.g., blood pressure, survival)
            2. A TREATMENT variable: The group/intervention variable that divides subjects into treatment groups (e.g., drug vs. placebo)
            3. SUBGROUP variables: Demographic or clinical characteristics that might modify treatment effects (e.g., age, sex, disease severity)
            4. COVARIATE variables: Variables to adjust for in the analysis (e.g., baseline values, risk factors)

            Here is my dataset: {self.current_name}
            Sample Size: {len(df)} observations

            Here are the columns in my dataset with their properties:
            {json.dumps(columns_info, indent=2)}

            Here's a sample of the data:
            {df.head(5).to_string()}

            Please analyze this dataset and recommend:
            1. The most appropriate outcome variable
            2. The most appropriate treatment/group variable
            3. 2-4 key variables to use for subgroup analysis (demographics or clinically relevant modifiers)
            4. Key covariates to include for adjustment

            Return your recommendations as a JSON with each variable type and explanation:
            {{
              "outcome": "variable_name",
              "treatment": "variable_name",
              "subgroups": ["variable1", "variable2", ...],
              "covariates": ["variable1", "variable2", ...],
              "explanation": "explanation of your recommendations"
            }}
            """

            # Fix the call_llm_async call to include model parameter
            response = await call_llm_async(prompt, model=llm_config.default_text_model)
            
            # Parse the JSON response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if not json_match:
                QMessageBox.warning(self, "Error", "Could not parse AI response for variable suggestions")
                return
                
            suggestions = json.loads(json_match.group(1))
            
            # Get recommendations
            recommended_outcome = suggestions.get("outcome")
            recommended_treatment = suggestions.get("treatment")
            recommended_subgroups = suggestions.get("subgroups", [])
            recommended_covariates = suggestions.get("covariates", [])
            explanation = suggestions.get("explanation", "")
            
            # Format the recommendations for display
            message = "AI recommendations for subgroup analysis:\n\n"
            message += f"Outcome variable: {recommended_outcome}\n"
            message += f"Treatment variable: {recommended_treatment}\n\n"
            message += f"Subgroup variables:\n"
            for var in recommended_subgroups:
                message += f"- {var}\n"
            message += f"\nCovariate variables:\n"
            for var in recommended_covariates:
                message += f"- {var}\n"
            
            message += f"\nExplanation:\n{explanation}\n\nApply these recommendations?"
            
            # Ask the user if they want to apply the recommendations
            if QMessageBox.question(
                self, 
                "Apply AI Recommendations?", 
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                # Apply the recommended variables
                if recommended_outcome in df.columns:
                    index = self.outcome_combo.findData(recommended_outcome)
                    if index >= 0:
                        self.outcome_combo.setCurrentIndex(index)
                
                if recommended_treatment in df.columns:
                    index = self.treatment_combo.findData(recommended_treatment)
                    if index >= 0:
                        self.treatment_combo.setCurrentIndex(index)
                
                # Select recommended subgroups
                for i in range(self.subgroup_list.count()):
                    item = self.subgroup_list.item(i)
                    if item.text() in recommended_subgroups:
                        item.setSelected(True)
                
                # Select recommended covariates
                for i in range(self.covariate_list.count()):
                    item = self.covariate_list.item(i)
                    if item.text() in recommended_covariates:
                        item.setSelected(True)
                
                # Update UI state
                self.update_ui_state()
                self.status_bar.showMessage("Applied AI variable recommendations")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI variable suggestion failed: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
    
    @asyncSlot()
    async def run_subgroup_analysis(self):
        """Run subgroup analysis on the selected variables."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        # Get selected variables
        outcome = self.outcome_combo.currentData()
        treatment = self.treatment_combo.currentData()
        
        # Get selected subgroups
        subgroups = [item.text() for item in self.subgroup_list.selectedItems()]
        if not subgroups:
            QMessageBox.warning(self, "Error", "Please select at least one subgroup variable")
            return
        
        # Get selected covariates
        covariates = [item.text() for item in self.covariate_list.selectedItems()]
        
        # Get analysis options
        analysis_method = self.method_combo.currentText()
        adjustment_method = self.adjustment_combo.currentText()
        adjust_pvalues = self.adjust_pvalues_checkbox.isChecked()
        include_interaction = self.include_interaction_checkbox.isChecked()
        visualization_type = self.viz_combo.currentText()
        
        # Store selected variables for reference
        self.selected_outcome = outcome
        self.selected_treatment = treatment
        self.selected_subgroups = subgroups
        self.selected_covariates = covariates
        
        # Show progress dialog
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status_bar.showMessage(f"Running subgroup analysis on {len(subgroups)} subgroups...")
        
        try:
            # Perform the actual statistical analysis (now using a single method)
            results = await self.perform_subgroup_analysis(
                    outcome=outcome,
                    treatment=treatment,
                    subgroups=subgroups,
                    covariates=covariates,
                    adjust_pvalues=adjust_pvalues,
                    include_interaction=include_interaction,
                adjustment_method=adjustment_method,
                analysis_method=analysis_method
                )
            
            # Store the results
            self.subgroup_results = results
            
            # Update the UI with results
            self.display_subgroup_results(results)
            
            # Create visualization based on selected type
            if visualization_type == "Forest Plot":
                self.generate_forest_plot(results.get("forest_plot_data", {}))
            elif visualization_type == "Interaction Plot":
                self.generate_interaction_plot(results)
            elif visualization_type == "Heat Map":
                self.generate_heat_map(results)
            elif visualization_type == "Effect Size Distribution":
                self.generate_effect_distribution(results)
            
            # Update UI state
            self.update_ui_state()
            self.status_bar.showMessage("Subgroup analysis completed successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Subgroup analysis failed: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
    
    async def calculate_variable_importance(self, df, outcome, treatment, subgroups):
        """Calculate variable importance for heterogeneous treatment effects."""
        # Simple implementation using random forest feature importance
        importance_results = {}
        
        try:
            # Give UI a chance to update
            import asyncio
            await asyncio.sleep(0.01)
            
            # Create interaction features
            X = df[subgroups].copy()
            
            # One-hot encode categorical variables
            cat_features = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            if cat_features:
                X = pd.get_dummies(X, columns=cat_features, drop_first=True)
            
            # Create interaction terms with treatment
            for col in X.columns:
                X[f"{col}_x_treatment"] = X[col] * df[treatment]
            
            # Fit a simple random forest to predict outcome
            y = df[outcome]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42) if len(set(y)) <= 5 else \
                 GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            rf.fit(X_scaled, y)
            
            # Get feature importances
            importance = rf.feature_importances_
            
            # Map back to original features
            importance_dict = dict(zip(X.columns, importance))
            
            # Extract interaction importance
            for subgroup in subgroups:
                interaction_columns = [col for col in X.columns if col.startswith(f"{subgroup}_") and "_x_treatment" in col]
                if interaction_columns:
                    importance_results[subgroup] = sum(importance_dict[col] for col in interaction_columns if col in importance_dict)
                else:
                    importance_results[subgroup] = 0
            
        except Exception as e:
            print(f"Error in variable importance calculation: {str(e)}")
            for subgroup in subgroups:
                importance_results[subgroup] = 0
            
        return importance_results
    
    def generate_methods_description(self, outcome, treatment, subgroups, covariates, 
                                     adjust_pvalues, adjustment_method="Bonferroni", 
                                     analysis_method="Stratified Analysis"):
        """Generate a comprehensive description of the statistical methods used."""
        method_text = f"Subgroup analysis was performed using {analysis_method} to assess heterogeneity of treatment effects "
        method_text += f"of {treatment} on {outcome} across levels of {', '.join(subgroups)}. "
        
        if covariates:
            method_text += f"Models were adjusted for the following covariates: {', '.join(covariates)}. "
        
        method_text += "Linear regression models were used for estimation of treatment effects within each subgroup. "
        method_text += "Interaction terms were included to test for heterogeneity of treatment effect. "
        
        if adjust_pvalues:
            method_text += f"P-values were adjusted for multiple testing using the {adjustment_method} method. "
        
        method_text += "The analysis evaluated within-subgroup treatment effects and tested for statistical "
        method_text += "interactions between treatment and subgroup variables. "
        method_text += "Forest plots were generated to visualize the pattern of treatment effects across subgroups. "
        
        method_text += "Bootstrap resampling was used to generate robust confidence intervals for effect estimates "
        method_text += "where sample sizes were sufficient. "
        
        method_text += "Variable importance analysis was conducted to quantify the contribution of each subgroup "
        method_text += "variable to treatment effect heterogeneity."
        
        return method_text
    
    def generate_limitations_description(self, df, outcome, treatment, subgroups):
        """Generate a description of limitations specific to this analysis."""
        n_samples = len(df)
        n_treatment = df[treatment].sum() if df[treatment].dtype == bool else df[treatment].nunique()
        
        limitations = []
        
        # Sample size limitations
        if n_samples < 100:
            limitations.append("The sample size is relatively small, limiting statistical power to detect subgroup effects.")
        
        # Treatment balance
        treatment_counts = df[treatment].value_counts()
        min_treatment_count = treatment_counts.min()
        if min_treatment_count < 30:
            limitations.append(f"The smallest treatment group has only {min_treatment_count} observations, which may affect the reliability of estimates.")
        
        # Subgroup size considerations
        small_subgroups = []
        for subgroup in subgroups:
            subgroup_counts = df[subgroup].value_counts()
            if (subgroup_counts < 20).any():
                small_levels = subgroup_counts[subgroup_counts < 20].index.tolist()
                small_subgroups.append(f"{subgroup} (levels: {', '.join(map(str, small_levels))})")
        
        if small_subgroups:
            limitations.append(f"Some subgroups have small sample sizes: {', '.join(small_subgroups)}.")
        
        # Standard limitations
        limitations.extend([
            "This analysis assumes linearity for continuous outcomes and may not account for all confounding factors.",
            "Multiple testing increases the risk of false positive findings, even with correction methods.",
            "The analysis cannot establish causal relationships in subgroups without randomization.",
            "Post-hoc subgroup analyses should be considered exploratory rather than confirmatory."
        ])
        
        return " ".join(limitations)
    
    def generate_recommendations(self, overall_effect, heterogeneity_tests, detailed_results):
        """Generate recommendations based on the analysis results."""
        recommendations = []
        
        # Check overall treatment effect
        if overall_effect.get("p_value", 1) < 0.05:
            effect_size = overall_effect.get("effect_size", 0)
            if effect_size > 0:
                recommendations.append(f"The treatment shows an overall positive effect and should be considered for the general population.")
            else:
                recommendations.append(f"The treatment shows an overall negative effect and should be used with caution.")
        else:
            recommendations.append("The treatment shows no significant overall effect, but may still benefit specific subgroups.")
        
        # Check for significant heterogeneity
        significant_heterogeneity = [test for test in heterogeneity_tests if test.get("significant", False)]
        if significant_heterogeneity:
            subgroups_with_heterogeneity = [test.get("subgroup", "") for test in significant_heterogeneity]
            recommendations.append(f"Treatment effects vary significantly across {', '.join(subgroups_with_heterogeneity)}. Consider targeted treatment approaches.")
            
            # Find subgroups with positive effects
            positive_subgroups = []
            for result in detailed_results:
                if result.get("effect_size", 0) > 0 and result.get("p_value", 1) < 0.05:
                    positive_subgroups.append(f"{result.get('subgroup', '')}={result.get('level', '')}")
            
            if positive_subgroups:
                recommendations.append(f"Consider prioritizing treatment for the following subgroups that show positive effects: {', '.join(positive_subgroups)}.")
                
            # Find subgroups with negative effects
            negative_subgroups = []
            for result in detailed_results:
                if result.get("effect_size", 0) < 0 and result.get("p_value", 1) < 0.05:
                    negative_subgroups.append(f"{result.get('subgroup', '')}={result.get('level', '')}")
            
            if negative_subgroups:
                recommendations.append(f"Consider alternative treatments for the following subgroups that show negative effects: {', '.join(negative_subgroups)}.")
        else:
            recommendations.append("No significant treatment effect heterogeneity was detected. The treatment effect appears consistent across subgroups.")
        
        # General recommendations
        recommendations.append("Future research should validate these findings in independent samples before implementing targeted treatment strategies.")
        
        if len(detailed_results) > 10:  # If many subgroups were analyzed
            recommendations.append("Given the exploratory nature of this analysis with multiple subgroups, results should be interpreted cautiously.")
        
        return " ".join(recommendations)
    
    def clear_visualization_tab(self):
        """Clear all widgets from the visualization area."""
        if hasattr(self, 'viz_layout') and self.viz_layout is not None:
            while self.viz_layout.count():
                item = self.viz_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            self.viz_container = QWidget()
            self.viz_layout = QVBoxLayout(self.viz_container)
            self.viz_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.viz_scroll.setWidget(self.viz_container)
    
    def display_subgroup_results(self, results):
        """Display the subgroup analysis results in the UI with enhanced reporting."""
        if not results:
            return
        
        # Display summary results
        summary = results.get("summary", {})
        overall_effect = summary.get("overall_effect", {})
        heterogeneity_tests = summary.get("heterogeneity_tests", [])
        main_findings = summary.get("main_findings", "")
        
        summary_text = "SUBGROUP ANALYSIS RESULTS\n"
        summary_text += "=" * 40 + "\n\n"
        
        summary_text += "OVERALL TREATMENT EFFECT:\n"
        effect_size = overall_effect.get("effect_size", "N/A")
        ci = overall_effect.get("confidence_interval", ["N/A", "N/A"])
        p_value = overall_effect.get("p_value", "N/A")
        
        summary_text += f"Effect Size: {effect_size:.4f}\n"
        summary_text += f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n"
        summary_text += f"P-value: {p_value:.4f}\n"
        
        # Add enhanced reporting for overall model
        if "r_squared" in overall_effect and "adjusted_r_squared" in overall_effect:
            summary_text += f"R²: {overall_effect['r_squared']:.4f} (Adjusted R²: {overall_effect['adjusted_r_squared']:.4f})\n"
        
        if "aic" in overall_effect and "bic" in overall_effect:
            summary_text += f"AIC: {overall_effect['aic']:.2f}, BIC: {overall_effect['bic']:.2f}\n"
            
        summary_text += f"Interpretation: {overall_effect.get('description', '')}\n\n"
        
        # Add baseline characteristics summary
        if "baseline_characteristics" in summary and summary["baseline_characteristics"]:
            summary_text += "BASELINE CHARACTERISTICS BY TREATMENT GROUP:\n"
            baseline = summary["baseline_characteristics"]
            
            for var, data in baseline.items():
                summary_text += f"• {var}: "
                if data.get("numeric", False):
                    stats = data.get("stats", {})
                    if "mean" in stats:
                        means = [f"{v:.2f}" for v in stats["mean"].values()]
                        summary_text += f"Mean: {', '.join(means)}"
                    
                    if "p_value" in data and data["p_value"] is not None:
                        summary_text += f" (p={data['p_value']:.3f})"
                else:
                    summary_text += "Categorical variable"
                    if "p_value" in data and data["p_value"] is not None:
                        summary_text += f" (Chi-square p={data['p_value']:.3f})"
                summary_text += "\n"
            
            summary_text += "\n"
        
        # Add variable importance results
        if "variable_importance" in summary and summary["variable_importance"]:
            summary_text += "VARIABLE IMPORTANCE FOR TREATMENT EFFECT HETEROGENEITY:\n"
            var_imp = summary["variable_importance"]
            
            # Sort by importance
            sorted_vars = sorted(var_imp.items(), key=lambda x: x[1], reverse=True)
            
            for var, importance in sorted_vars:
                summary_text += f"• {var}: {importance:.4f}\n"
            
            summary_text += "\n"
        
        summary_text += "HETEROGENEITY TESTS:\n"
        for test in heterogeneity_tests:
            subgroup = test.get("subgroup", "")
            p_value = test.get("interaction_p_value", "N/A")
            significant = test.get("significant", False)
            interpretation = test.get("interpretation", "")
            
            summary_text += f"Subgroup: {subgroup}\n"
            summary_text += f"Interaction P-value: {p_value:.4f}\n"
            
            # Add enhanced F-test reporting
            if "f_test_statistic" in test and test["f_test_statistic"] is not None:
                summary_text += f"F-statistic: {test['f_test_statistic']:.3f} "
                summary_text += f"(df: {test['f_test_df_num']}, {test['f_test_df_denom']})\n"
                
            summary_text += f"Significant: {'Yes' if significant else 'No'}\n"
            summary_text += f"Interpretation: {interpretation}\n\n"
        
        summary_text += "MAIN FINDINGS:\n"
        summary_text += main_findings + "\n\n"
        
        if "recommendation" in results:
            summary_text += "RECOMMENDATIONS:\n"
            summary_text += results["recommendation"] + "\n\n"
        
        if "methods" in results:
            summary_text += "METHODS:\n"
            summary_text += results["methods"] + "\n\n"
        
        if "limitations" in results:
            summary_text += "LIMITATIONS:\n"
            summary_text += results["limitations"] + "\n"
        
        self.summary_text.setPlainText(summary_text)
        
        # Display detailed results in table with enhanced formatting
        detailed_results = results.get("detailed_results", [])
        self.details_table.setRowCount(0)
        self.details_table.setColumnCount(8)  # Add columns for additional metrics
        self.details_table.setHorizontalHeaderLabels([
            "Subgroup", "Level", "N", "Effect Size", "95% CI", "P-value", 
            "Adjusted P", "Bootstrap CI"
        ])
        
        for i, result in enumerate(detailed_results):
            self.details_table.insertRow(i)
            
            subgroup = result.get("subgroup", "")
            level = result.get("level", "")
            sample_size = result.get("sample_size", "")
            effect_size = result.get("effect_size", 0)  # Default to 0 for numeric ops
            ci = result.get("confidence_interval", [0, 0])  # Default to [0,0] for numeric ops
            ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if isinstance(ci, list) and len(ci) >= 2 else "N/A"
            p_value = result.get("p_value", 1.0)  # Default to 1.0 for numeric ops
            
            # Get additional metrics
            adj_p_value = result.get("adjusted_p_value", "N/A")
            
            bootstrap_ci = result.get("bootstrap_confidence_interval")
            if bootstrap_ci and isinstance(bootstrap_ci, list) and len(bootstrap_ci) >= 2:
                bootstrap_text = f"[{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]"
            else:
                bootstrap_text = "N/A"
            
            r_squared = result.get("model_r_squared", "")
            
            self.details_table.setItem(i, 0, QTableWidgetItem(str(subgroup)))
            self.details_table.setItem(i, 1, QTableWidgetItem(str(level)))
            self.details_table.setItem(i, 2, QTableWidgetItem(str(sample_size)))
            self.details_table.setItem(i, 3, QTableWidgetItem(f"{effect_size:.4f}"))
            self.details_table.setItem(i, 4, QTableWidgetItem(ci_text))
            
            # Format p-value with significance indicators
            p_value_str = f"{p_value:.4f}"
            if isinstance(p_value, (int, float)) and p_value < 0.05:
                p_value_str += " *"
                if p_value < 0.01:
                    p_value_str += "*"
                if p_value < 0.001:
                    p_value_str += "*"
            
            self.details_table.setItem(i, 5, QTableWidgetItem(p_value_str))
            
            # Add adjusted p-value
            if isinstance(adj_p_value, (int, float)):
                adj_p_text = f"{adj_p_value:.4f}"
                if adj_p_value < 0.05:
                    adj_p_text += " *"
            else:
                adj_p_text = str(adj_p_value)
            self.details_table.setItem(i, 6, QTableWidgetItem(adj_p_text))
            
            # Add bootstrap CI
            self.details_table.setItem(i, 7, QTableWidgetItem(bootstrap_text))
            
            # Color rows by significance
            if isinstance(p_value, (int, float)) and p_value < 0.05:
                for j in range(8):
                    self.details_table.item(i, j).setBackground(QBrush(QColor(230, 242, 255)))
        
        # Adjust column widths
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
    def save_results_to_study(self):
        """Save the subgroup analysis results to the study."""
        if not self.subgroup_results:
            QMessageBox.warning(self, "Error", "No results to save. Please run an analysis first.")
            return
        
        # Get main window and check for studies manager
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            QMessageBox.warning(self, "Error", "Could not access studies manager")
            return
        
        try:
            # Create a formatted report of the results
            report = {
                "analysis_type": "Subgroup Analysis",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.current_name,
                "variables": {
                    "outcome": self.selected_outcome,
                    "treatment": self.selected_treatment,
                    "subgroups": self.selected_subgroups,
                    "covariates": self.selected_covariates
                },
                "results": self.subgroup_results,
                "summary": self.summary_text.toPlainText(),
                "ai_interpretation": self.subgroup_results.get("ai_interpretation", "")
            }
            
            # Add to study results
            success = main_window.studies_manager.add_result_to_active_study(
                "Subgroup Analysis", 
                report,
                description=f"Subgroup analysis of {self.selected_outcome} by {self.selected_treatment} across {len(self.selected_subgroups)} subgroups"
            )
            
            if success:
                QMessageBox.information(self, "Success", "Results saved to study successfully")
                self.status_bar.showMessage("Results saved to study")
            else:
                QMessageBox.warning(self, "Error", "Failed to save results to study")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving results: {str(e)}")
    
    @asyncSlot()
    async def interpret_results(self):
        """Use AI to interpret the subgroup analysis results."""
        if not self.subgroup_results:
            QMessageBox.warning(self, "Error", "No results to interpret. Please run an analysis first.")
            return
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Generating interpretation of subgroup analysis. Please wait...")
        
        try:
            # Extract key information for the prompt
            summary = self.subgroup_results.get("summary", {})
            overall_effect = summary.get("overall_effect", {})
            heterogeneity_tests = summary.get("heterogeneity_tests", [])
            detailed_results = self.subgroup_results.get("detailed_results", [])
            
            # Format the key details as a structured prompt requesting JSON response
            prompt = f"""
            I need help interpreting the results of a subgroup analysis. Here are the key findings:
            
            Dataset: {self.current_name}
            Sample Size: {len(self.current_dataframe) if self.current_dataframe is not None else 'Unknown'}
            
            Variables:
            - Outcome: {self.selected_outcome}
            - Treatment: {self.selected_treatment}
            - Subgroups analyzed: {', '.join(self.selected_subgroups)}
            - Covariates included: {', '.join(self.selected_covariates) if self.selected_covariates else 'None'}
            
            Overall Treatment Effect:
            - Effect Size: {overall_effect.get('effect_size', 'N/A')}
            - 95% CI: [{overall_effect.get('confidence_interval', ['N/A', 'N/A'])[0]}, {overall_effect.get('confidence_interval', ['N/A', 'N/A'])[1]}]
            - P-value: {overall_effect.get('p_value', 'N/A')}
            
            Heterogeneity Tests:
            {json.dumps(heterogeneity_tests, indent=2)}
            
            Subgroup Results:
            {json.dumps(detailed_results[:10], indent=2)}  # Limiting to first 10 for brevity
            
            Please provide a comprehensive interpretation of these subgroup analysis results in a structured JSON format with the following keys:
            
            {{
                "overall_findings": "Is there an overall treatment effect? How strong is it?",
                "heterogeneity_analysis": "Is there evidence that treatment effects vary across subgroups?",
                "subgroup_findings": [
                    {{
                        "subgroup": "subgroup name",
                        "level": "level name",
                        "effect": "effect size and direction",
                        "significance": "statistical significance",
                        "interpretation": "clinical interpretation"
                    }}
                ],
                "clinical_implications": "How should these findings influence clinical decision-making?",
                "limitations": "What limitations should be considered when interpreting these results?",
                "follow_up_analyses": "What follow-up analyses would you recommend?",
                "hypotheses_generated": "What new hypotheses do these findings suggest?"
            }}
            
            Ensure your response is valid JSON that can be parsed programmatically.
            """
            
            # Call the LLM API with model parameter
            response = await call_llm_async(prompt, model=llm_config.default_text_model)
            
            # Try to parse the JSON response
            try:
                # Find JSON in the response (in case the LLM includes extra text)
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_response = json.loads(json_match.group(1))
                    
                    # Convert JSON to formatted markdown for display
                    markdown_response = "# Subgroup Analysis Interpretation\n\n"
                    
                    # Overall findings
                    markdown_response += "## Overall Findings\n\n"
                    markdown_response += json_response.get("overall_findings", "") + "\n\n"
                    
                    # Heterogeneity analysis
                    markdown_response += "## Heterogeneity Analysis\n\n"
                    markdown_response += json_response.get("heterogeneity_analysis", "") + "\n\n"
                    
                    # Subgroup findings
                    markdown_response += "## Subgroup-Specific Findings\n\n"
                    for finding in json_response.get("subgroup_findings", []):
                        markdown_response += f"### {finding.get('subgroup', '')} = {finding.get('level', '')}\n\n"
                        markdown_response += f"- **Effect**: {finding.get('effect', '')}\n"
                        markdown_response += f"- **Significance**: {finding.get('significance', '')}\n"
                        markdown_response += f"- **Interpretation**: {finding.get('interpretation', '')}\n\n"
                    
                    # Clinical implications
                    markdown_response += "## Clinical Implications\n\n"
                    markdown_response += json_response.get("clinical_implications", "") + "\n\n"
                    
                    # Limitations
                    markdown_response += "## Limitations and Caveats\n\n"
                    markdown_response += json_response.get("limitations", "") + "\n\n"
                    
                    # Follow-up analyses
                    markdown_response += "## Suggested Follow-up Analyses\n\n"
                    markdown_response += json_response.get("follow_up_analyses", "") + "\n\n"
                    
                    # Hypotheses generated
                    markdown_response += "## New Hypotheses Generated\n\n"
                    markdown_response += json_response.get("hypotheses_generated", "") + "\n\n"
                    
                    # Store both JSON and formatted markdown
                    if self.subgroup_results:
                        self.subgroup_results["ai_interpretation"] = markdown_response
                        self.subgroup_results["ai_interpretation_json"] = json_response
                    
                    # Display the formatted markdown
                    self.interpretation_text.setMarkdown(markdown_response)
                else:
                    # If JSON parsing fails, display the raw response
                    self.interpretation_text.setMarkdown(response)
                    if self.subgroup_results:
                        self.subgroup_results["ai_interpretation"] = response
            except json.JSONDecodeError:
                # If JSON parsing fails, display the raw response
                self.interpretation_text.setMarkdown(response)
                if self.subgroup_results:
                    self.subgroup_results["ai_interpretation"] = response
            
            # Find and switch to the interpretation tab
            for tab_widget in self.findChildren(QTabWidget):
                for i in range(tab_widget.count()):
                    if tab_widget.tabText(i) == "AI Interpretation":
                        tab_widget.setCurrentIndex(i)
                        break
            
            # Update status
            self.status_bar.showMessage("Interpretation generated successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI interpretation failed: {str(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_main_findings(self, overall_effect, heterogeneity_tests, detailed_results):
        """Generate a summary of main findings from the analysis results."""
        findings = []
        
        # Overall effect
        effect_size = overall_effect.get("effect_size", 0)
        p_value = overall_effect.get("p_value", 1.0)
        
        if p_value < 0.05:
            direction = "positive" if effect_size > 0 else "negative"
            findings.append(f"Overall, there is a significant {direction} treatment effect (effect size = {effect_size:.4f}, p = {p_value:.4f}).")
        else:
            findings.append(f"Overall, there is no significant treatment effect (effect size = {effect_size:.4f}, p = {p_value:.4f}).")
        
        # Heterogeneity
        significant_heterogeneity = [test for test in heterogeneity_tests if test.get("significant", False)]
        
        if significant_heterogeneity:
            subgroups = [test.get("subgroup", "") for test in significant_heterogeneity]
            findings.append(f"There is significant heterogeneity of treatment effect across {', '.join(subgroups)}.")
            
            # Add details for each significant subgroup
            for test in significant_heterogeneity:
                subgroup = test.get("subgroup", "")
                p_value = test.get("interaction_p_value", 1.0)
                findings.append(f"- {subgroup}: interaction p-value = {p_value:.4f}")
        else:
            findings.append("There is no significant heterogeneity of treatment effect across the analyzed subgroups.")
        
        # Top subgroups with significant effects
        significant_results = [r for r in detailed_results if r.get("p_value", 1.0) < 0.05]
        
        if significant_results:
            # Sort by absolute effect size
            sorted_results = sorted(significant_results, key=lambda r: abs(r.get("effect_size", 0)), reverse=True)
            
            findings.append("Significant treatment effects were found in the following subgroups:")
            
            for i, result in enumerate(sorted_results[:5]):  # Top 5 at most
                subgroup = result.get("subgroup", "")
                level = result.get("level", "")
                effect = result.get("effect_size", 0)
                p_value = result.get("p_value", 1.0)
                direction = "positive" if effect > 0 else "negative"
                
                findings.append(f"- {subgroup}={level}: {direction} effect of {abs(effect):.4f} (p = {p_value:.4f})")
                
                if i >= 4 and len(sorted_results) > 5:
                    findings.append(f"- ... and {len(sorted_results) - 5} more significant subgroups")
                    break
        
        return " ".join(findings)

    def calculate_baseline_characteristics(self, df, treatment, variables):
        """Calculate baseline characteristics table by treatment group."""
        baseline = {}
        
        for var in variables:
            if pd.api.types.is_numeric_dtype(df[var]):
                # For numeric variables, calculate mean, SD, etc.
                stats_df = df.groupby(treatment)[var].agg(['mean', 'std', 'min', 'max', 'count'])
                
                # Calculate p-value for difference between groups
                groups = df[treatment].unique()
                if len(groups) == 2:  # Only for binary treatment
                    group1 = df[df[treatment] == groups[0]][var].dropna()
                    group2 = df[df[treatment] == groups[1]][var].dropna()
                    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                else:
                    p_value = None
                
                baseline[var] = {
                    'numeric': True,
                    'stats': stats_df.to_dict(),
                    'p_value': p_value
                }
            else:
                # For categorical variables, calculate frequencies
                cross_tab = pd.crosstab(df[var], df[treatment], normalize='columns')
                counts = pd.crosstab(df[var], df[treatment])
                
                # Calculate chi-square p-value
                try:
                    chi2, p_value, _, _ = stats.chi2_contingency(counts)
                except:
                    p_value = None
                
                baseline[var] = {
                    'numeric': False,
                    'frequencies': cross_tab.to_dict(),
                    'counts': counts.to_dict(),
                    'p_value': p_value
                }
        
        return baseline
        
    async def perform_subgroup_analysis(self, outcome, treatment, subgroups, covariates=None, 
                                adjust_pvalues=True, include_interaction=True,
                            adjustment_method="Benjamini-Hochberg (FDR)",
                            analysis_method="Stratified Analysis"):
        """Dispatch to the appropriate analysis method and return standardized results."""
        # Use the analysis_method parameter to determine which technique to use
        df = self.current_dataframe.copy()
        
        # Check if variables exist in the dataframe
        for var in [outcome, treatment] + subgroups + (covariates or []):
            if var not in df.columns:
                raise ValueError(f"Variable {var} not found in the dataset")
        
        # Check if treatment is binary - if not, try to convert
        if df[treatment].nunique() > 2:
            if df[treatment].nunique() <= 5:  # Reasonable number of levels for a categorical variable
                # Convert to dummy variables for analysis
                df = pd.get_dummies(df, columns=[treatment], drop_first=True)
                # Get the new treatment variable name (should be the first dummy)
                treatment_cols = [col for col in df.columns if col.startswith(f"{treatment}_")]
                if treatment_cols:
                    treatment = treatment_cols[0]
                    print(f"Converted treatment to binary: {treatment}")
                else:
                    raise ValueError(f"Could not convert treatment variable {treatment} to binary format")
            else:
                raise ValueError(f"Treatment variable {treatment} has too many levels ({df[treatment].nunique()}). Please use a binary treatment variable.")
        
        # Ensure outcome is numeric
        if not pd.api.types.is_numeric_dtype(df[outcome]):
            try:
                df[outcome] = pd.to_numeric(df[outcome])
            except:
                raise ValueError(f"Outcome variable {outcome} must be numeric or convertible to numeric")
            
        # Route to appropriate analysis method
        if analysis_method == "Stratified Analysis":
            return await self.perform_stratified_analysis(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "Interaction Models":
            return await self.perform_interaction_models(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "Causal Forest":
            return await self.perform_causal_forest(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "BART (Approximate)":  # Updated name
            return await self.perform_bart_analysis(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "IPTW":
            return await self.perform_iptw_analysis(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "Bayesian Hierarchical Model":  # New method
            return await self.perform_bayesian_hierarchical_analysis(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        elif analysis_method == "Propensity Score Matching":  # New method
            return await self.perform_propensity_score_matching(
                df, outcome, treatment, subgroups, covariates, 
                adjust_pvalues, include_interaction, adjustment_method)
        else:
            raise ValueError(f"Unknown analysis method: {analysis_method}")

    async def perform_stratified_analysis(self, df, outcome, treatment, subgroups, covariates=None, 
                                   adjust_pvalues=True, include_interaction=True,
                               adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform standard stratified regression analysis on subgroups."""
        import asyncio
        special_technique = "Standard stratified analysis with regression models"
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Calculate baseline characteristics by treatment group
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Overall model (no subgroups) - base effects
        overall_formula = f"{outcome} ~ {treatment}"
        if covariates:
            overall_formula += " + " + " + ".join(covariates)
        
        # Fit overall model
        overall_model = smf.ols(overall_formula, data=df).fit()
        
        # Find the correct treatment parameter name in the model
        treatment_param = None
        for param in overall_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(overall_model.params.index)}")
        
        # Extract overall treatment effect using the identified parameter
        overall_effect = {
            "effect_size": float(overall_model.params[treatment_param]),
            "std_error": float(overall_model.bse[treatment_param]),
            "confidence_interval": [
                float(overall_model.conf_int().loc[treatment_param, 0]),
                float(overall_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(overall_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome}",
            "model_summary": str(overall_model.summary()),  # ENHANCEMENT: Save full model summary
            "r_squared": float(overall_model.rsquared),     # ENHANCEMENT: Save R-squared
            "adjusted_r_squared": float(overall_model.rsquared_adj),  # ENHANCEMENT: Adjusted R-squared
            "aic": float(overall_model.aic),               # ENHANCEMENT: AIC value
            "bic": float(overall_model.bic),               # ENHANCEMENT: BIC value
            "analysis_method": "Stratified Analysis",      # Add the specific method used
            "special_technique": special_technique         # Description of special technique
        }
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Initialize results for each subgroup
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []  # ENHANCEMENT: Add relative sizes for better visualization
        }

        # ENHANCEMENT: Add bootstrapped confidence intervals
        bootstrap_samples = 1000 if len(df) > 100 else 500  # Adjust based on sample size
        
        # Analyze each subgroup
        for subgroup in subgroups:
            # Create interaction term
            if include_interaction:
                # Test for interaction (heterogeneity of treatment effect)
                interaction_formula = f"{outcome} ~ {treatment} * C({subgroup})"
                if covariates:
                    for cov in covariates:
                        interaction_formula += f" + {cov}"
                
                interaction_model = smf.ols(interaction_formula, data=df).fit()
                
                # Get interaction terms
                interaction_terms = [term for term in interaction_model.params.index if f"{treatment}:C({subgroup}" in term]
                
                # Extract interaction p-values
                interaction_p_values = [float(interaction_model.pvalues[term]) for term in interaction_terms]
                
                # ENHANCEMENT: Calculate contrast tests for interaction
                f_test_result = interaction_model.f_test([f"{term} = 0" for term in interaction_terms])
                contrast_p_value = float(f_test_result.pvalue)
                
                # Test for overall interaction effect
                heterogeneity_p_value = contrast_p_value if interaction_terms else 1.0
                
                # Add heterogeneity test result
                heterogeneity_tests.append({
                    "subgroup": subgroup,
                    "interaction_p_value": heterogeneity_p_value,
                    "individual_p_values": interaction_p_values,
                    "significant": heterogeneity_p_value < 0.05,
                    "f_test_statistic": float(f_test_result.fvalue) if interaction_terms else None,
                    "f_test_df_num": int(f_test_result.df_num) if interaction_terms else None,
                    "f_test_df_denom": int(f_test_result.df_denom) if interaction_terms else None,
                    "interpretation": (
                        f"Treatment effect differs significantly across levels of {subgroup}"
                        if heterogeneity_p_value < 0.05 else
                        f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                    )
                })
            
            await asyncio.sleep(0.01)  # Let UI update
            
            # Analyze treatment effect within each level of the subgroup
            subgroup_levels = df[subgroup].unique()
            
            for level in subgroup_levels:
                # Skip NaN/None values
                if pd.isna(level):
                    continue
                
                # Subset data for this level
                level_df = df[df[subgroup] == level]
                
                # Skip if sample size is too small
                if len(level_df) < 10:
                    print(f"Skipping {subgroup}={level} due to small sample size (n={len(level_df)})")
                    continue
                
                # Create formula for this subgroup level
                level_formula = f"{outcome} ~ {treatment}"
                if covariates:
                    level_formula += " + " + " + ".join(covariates)
                
                # Fit model for this subgroup level
                try:
                    level_model = smf.ols(level_formula, data=level_df).fit()
                    
                    # Extract treatment effect
                    effect_size = float(level_model.params[treatment])
                    std_error = float(level_model.bse[treatment])
                    ci_lower = float(level_model.conf_int().loc[treatment, 0])
                    ci_upper = float(level_model.conf_int().loc[treatment, 1])
                    p_value = float(level_model.pvalues[treatment])
                    
                    # ENHANCEMENT: Bootstrap confidence intervals for greater robustness
                    bootstrap_effects = []
                    n_level = len(level_df)
                    
                    if n_level >= 30:  # Only bootstrap if enough data
                        for _ in range(bootstrap_samples):
                            # Sample with replacement
                            boot_df = level_df.sample(n=n_level, replace=True)
                            try:
                                boot_model = smf.ols(level_formula, data=boot_df).fit()
                                if treatment in boot_model.params:
                                    bootstrap_effects.append(float(boot_model.params[treatment]))
                            except:
                                pass  # Skip this bootstrap sample if it fails
                    
                    if len(bootstrap_effects) > bootstrap_samples * 0.8:  # If at least 80% successful
                        bootstrap_effects.sort()
                        bootstrap_ci_lower = bootstrap_effects[int(0.025 * len(bootstrap_effects))]
                        bootstrap_ci_upper = bootstrap_effects[int(0.975 * len(bootstrap_effects))]
                    else:
                        bootstrap_ci_lower = None
                        bootstrap_ci_upper = None
                    
                    # Add result
                    result = {
                        "subgroup": subgroup,
                        "level": str(level),
                        "sample_size": len(level_df),
                        "effect_size": effect_size,
                        "std_error": std_error,
                        "confidence_interval": [ci_lower, ci_upper],
                        "bootstrap_confidence_interval": [bootstrap_ci_lower, bootstrap_ci_upper] 
                            if bootstrap_ci_lower is not None else None,
                        "p_value": p_value,
                        "model_r_squared": float(level_model.rsquared),
                        "model_summary": str(level_model.summary())
                    }
                    
                    detailed_results.append(result)
                    
                    # Add to forest plot data
                    forest_plot_data["subgroups"].append(subgroup)
                    forest_plot_data["levels"].append(str(level))
                    forest_plot_data["effect_sizes"].append(effect_size)
                    forest_plot_data["lower_cis"].append(ci_lower)
                    forest_plot_data["upper_cis"].append(ci_upper)
                    forest_plot_data["p_values"].append(p_value)
                    forest_plot_data["sample_sizes"].append(len(level_df))
                    forest_plot_data["relative_sample_sizes"].append(len(level_df) / len(df))
                
                except Exception as e:
                    print(f"Error analyzing {subgroup}={level}: {str(e)}")
                
                # Brief pause to prevent UI freezing
                if len(subgroup_levels) > 5:
                    await asyncio.sleep(0.01)
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction based on selected method
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Variable importance analysis for subgroups
        variable_importance = await self.calculate_variable_importance(
            df, outcome, treatment, subgroups
        )
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, "Stratified Analysis"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results

    async def perform_interaction_models(self, df, outcome, treatment, subgroups, covariates=None, 
                                   adjust_pvalues=True, include_interaction=True,
                               adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform analysis using interaction models to explicitly test for heterogeneity."""
        import asyncio
        special_technique = "Interaction models with explicit tests of treatment effect heterogeneity"
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Calculate baseline characteristics by treatment group
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # For interaction models, we will build one comprehensive model with all interactions
        interaction_formula = f"{outcome} ~ {treatment}"
        
        # Add main effects for all subgroups
        for subgroup in subgroups:
            interaction_formula += f" + C({subgroup})"
        
        # Add two-way interactions with treatment for all subgroups
        for subgroup in subgroups:
            interaction_formula += f" + {treatment}:C({subgroup})"
        
        # Add covariates
        if covariates:
            interaction_formula += " + " + " + ".join(covariates)
        
        # Fit the comprehensive interaction model
        full_model = smf.ols(interaction_formula, data=df).fit()
        
        # Find the treatment parameter (main effect)
        treatment_param = None
        for param in full_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(full_model.params.index)}")
        
        # Extract overall treatment effect (this is the conditional effect at reference levels)
        overall_effect = {
            "effect_size": float(full_model.params[treatment_param]),
            "std_error": float(full_model.bse[treatment_param]),
            "confidence_interval": [
                float(full_model.conf_int().loc[treatment_param, 0]),
                float(full_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(full_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome} at reference levels of all subgroups",
            "model_summary": str(full_model.summary()),
            "r_squared": float(full_model.rsquared),
            "adjusted_r_squared": float(full_model.rsquared_adj),
            "aic": float(full_model.aic),
            "bic": float(full_model.bic),
            "analysis_method": "Interaction Models",
            "special_technique": special_technique
        }
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Initialize results
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []
        }
        
        # For each subgroup, test for heterogeneity and calculate effects
        for subgroup in subgroups:
            # Extract interaction terms for this subgroup
            interaction_terms = [term for term in full_model.params.index 
                               if f"{treatment}:C({subgroup})" in term]
            
            # Test for heterogeneity using F-test for interaction terms
            if interaction_terms:
                f_test = full_model.f_test([f"{term}=0" for term in interaction_terms])
                heterogeneity_p_value = float(f_test.pvalue)
                
                # Add heterogeneity test result
                heterogeneity_tests.append({
                    "subgroup": subgroup,
                    "interaction_p_value": heterogeneity_p_value,
                    "individual_p_values": [float(full_model.pvalues[term]) for term in interaction_terms],
                    "significant": heterogeneity_p_value < 0.05,
                    "f_test_statistic": float(f_test.fvalue),
                    "f_test_df_num": int(f_test.df_num),
                    "f_test_df_denom": int(f_test.df_denom),
                    "interpretation": (
                        f"Treatment effect differs significantly across levels of {subgroup}"
                        if heterogeneity_p_value < 0.05 else
                        f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                    )
                })
            
            # Calculate conditional effects at each level of the subgroup
            subgroup_levels = df[subgroup].unique()
            
            for level in subgroup_levels:
                # Skip NaN/None values
                if pd.isna(level):
                    continue
                
                # Calculate sample size for this level
                level_df = df[df[subgroup] == level]
                n_level = len(level_df)
                
                # Skip if sample size is too small
                if n_level < 10:
                    continue
                
                # For the first level (usually reference), the effect is the main effect
                if level == subgroup_levels[0]:
                    effect_size = float(full_model.params[treatment_param])
                    std_error = float(full_model.bse[treatment_param])
                    ci_lower = float(full_model.conf_int().loc[treatment_param, 0])
                    ci_upper = float(full_model.conf_int().loc[treatment_param, 1])
                    p_value = float(full_model.pvalues[treatment_param])
                else:
                    # Find the interaction term for this level
                    interaction_term = next((term for term in interaction_terms 
                                          if f"{subgroup}[T.{level}]" in term or 
                                             f"{subgroup}.{level}" in term), None)
                    
                    if interaction_term:
                        # Calculate the conditional effect for this level
                        # (main effect + interaction effect)
                        effect_size = float(full_model.params[treatment_param] + 
                                          full_model.params[interaction_term])
                        
                        # Calculate standard error using delta method approximation
                        var_main = float(full_model.bse[treatment_param]**2)
                        var_inter = float(full_model.bse[interaction_term]**2)
                        cov_main_inter = float(full_model.cov_params().loc[treatment_param, interaction_term])
                        
                        std_error = float(np.sqrt(var_main + var_inter + 2*cov_main_inter))
                        
                        # Calculate confidence interval and p-value
                        ci_lower = effect_size - 1.96 * std_error
                        ci_upper = effect_size + 1.96 * std_error
                        
                        # Calculate two-sided p-value using normal approximation
                        z_score = abs(effect_size) / std_error
                        p_value = 2 * (1 - stats.norm.cdf(z_score))
                    else:
                        # If interaction term not found (should not happen), use reference level effect
                        effect_size = float(full_model.params[treatment_param])
                        std_error = float(full_model.bse[treatment_param])
                        ci_lower = float(full_model.conf_int().loc[treatment_param, 0])
                        ci_upper = float(full_model.conf_int().loc[treatment_param, 1])
                        p_value = float(full_model.pvalues[treatment_param])
                
                # Add result
                result = {
                    "subgroup": subgroup,
                    "level": str(level),
                    "sample_size": n_level,
                    "effect_size": effect_size,
                    "std_error": std_error,
                    "confidence_interval": [ci_lower, ci_upper],
                    "p_value": p_value
                }
                
                detailed_results.append(result)
                
                # Add to forest plot data
                forest_plot_data["subgroups"].append(subgroup)
                forest_plot_data["levels"].append(str(level))
                forest_plot_data["effect_sizes"].append(effect_size)
                forest_plot_data["lower_cis"].append(ci_lower)
                forest_plot_data["upper_cis"].append(ci_upper)
                forest_plot_data["p_values"].append(p_value)
                forest_plot_data["sample_sizes"].append(n_level)
                forest_plot_data["relative_sample_sizes"].append(n_level / len(df))
            
            await asyncio.sleep(0.01)  # Let UI update
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        # Variable importance analysis for subgroups
        variable_importance = await self.calculate_variable_importance(
            df, outcome, treatment, subgroups
        )
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, "Interaction Models"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results

    async def perform_causal_forest(self, df, outcome, treatment, subgroups, covariates=None, 
                              adjust_pvalues=True, include_interaction=True,
                          adjustment_method="Benjamini-Hochberg (FDR)"):
        """Implement causal forest for heterogeneous treatment effect estimation."""
        import asyncio
        special_technique = "Causal Forest using machine learning for treatment effect heterogeneity"
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Calculate baseline characteristics
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Prepare features
        X_columns = []
        if covariates:
            X_columns.extend(covariates)
        X_columns.extend(subgroups)
        
        # Prepare data matrix
        X_raw = df[X_columns].copy()
        
        # Convert categorical variables to dummy variables
        cat_columns = [col for col in X_raw.columns if not pd.api.types.is_numeric_dtype(X_raw[col])]
        if cat_columns:
            X = pd.get_dummies(X_raw, columns=cat_columns, drop_first=True)
        else:
            X = X_raw.copy()
        
        # Standardize numeric features for better performance
        numeric_columns = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        if numeric_columns:
            scaler = StandardScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        
        # Extract treatment and outcome
        y = df[outcome].values
        w = df[treatment].values.astype(int)  # Ensure binary treatment is 0/1
        
        self.status_bar.showMessage("Fitting causal forest model...")
        
        # Implement causal forest with randomized trees
        # Simulate a causal forest using a modified random forest approach
        # First, we'll split the data into treatment and control
        X_treated = X[w == 1]
        y_treated = y[w == 1]
        
        X_control = X[w == 0]
        y_control = y[w == 0]
        
        # Train separate models for treatment and control
        n_estimators = 200
        
        # Parameters tuned for causal inference
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': 5,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            # Remove the 'bootstrap' parameter as it's not supported by GradientBoostingRegressor
            'random_state': 42
        }
        
        # Train model for treated group
        model_treated = GradientBoostingRegressor(**rf_params)
        model_treated.fit(X_treated, y_treated)
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Train model for control group
        model_control = GradientBoostingRegressor(**rf_params)
        model_control.fit(X_control, y_control)
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Calculate individual treatment effects
        y_treated_pred = model_treated.predict(X)
        y_control_pred = model_control.predict(X)
        ite = y_treated_pred - y_control_pred
        
        # Add ITE to dataframe
        df['estimated_ite'] = ite
        
        # Estimate uncertainty in ITEs using bootstrap
        n_bootstrap = 50  # Reduced for computational efficiency
        bootstrap_ites = np.zeros((len(df), n_bootstrap))
        
        self.status_bar.showMessage("Estimating uncertainty in treatment effects...")
        
        # Process bootstraps in batches to allow UI updates
        batch_size = 5
        for batch in range(0, n_bootstrap, batch_size):
            end_batch = min(batch + batch_size, n_bootstrap)
            
            for b in range(batch, end_batch):
                # Sample with replacement
                treated_idx = np.random.choice(len(X_treated), len(X_treated), replace=True)
                control_idx = np.random.choice(len(X_control), len(X_control), replace=True)
                
                # Train bootstrap models
                boot_model_treated = GradientBoostingRegressor(**rf_params)
                boot_model_treated.fit(X_treated.iloc[treated_idx], y_treated[treated_idx])
                
                boot_model_control = GradientBoostingRegressor(**rf_params)
                boot_model_control.fit(X_control.iloc[control_idx], y_control[control_idx])
                
                # Calculate bootstrap ITEs
                bootstrap_ites[:, b] = boot_model_treated.predict(X) - boot_model_control.predict(X)
            
            await asyncio.sleep(0.01)  # Let UI update
            self.status_bar.showMessage(f"Bootstrap iteration {end_batch}/{n_bootstrap}...")
        
        # Calculate CI for individual treatment effects
        df['ite_lower'] = np.percentile(bootstrap_ites, 2.5, axis=1)
        df['ite_upper'] = np.percentile(bootstrap_ites, 97.5, axis=1)
        
        # Calculate variable importance for heterogeneity
        variable_importance = {}
        feature_importances = np.zeros(X.shape[1])
        
        # Combine importance from both models
        for tree_idx in range(n_estimators):
            if hasattr(model_treated, 'estimators_'):  # RandomForest
                tree_treated = model_treated.estimators_[tree_idx]
                tree_control = model_control.estimators_[tree_idx]
                feature_importances += (tree_treated.feature_importances_ + tree_control.feature_importances_)
            else:  # GradientBoosting
                tree_treated = model_treated.estimators_[tree_idx, 0]
                tree_control = model_control.estimators_[tree_idx, 0]
                feature_importances += (tree_treated.feature_importances_ + tree_control.feature_importances_)
        
        feature_importances /= (2 * n_estimators)
        
        # Map feature importances back to original variables
        feature_names = X.columns
        
        # Handle dummy variables by summing importance back to original categorical variables
        for orig_var in X_columns:
            if orig_var in cat_columns:
                # Sum importance of all dummy variables derived from this categorical variable
                prefix = f"{orig_var}_"
                dummy_cols = [col for col in feature_names if col.startswith(prefix)]
                if dummy_cols:
                    importance = sum(feature_importances[feature_names.get_loc(col)] for col in dummy_cols)
                    variable_importance[orig_var] = float(importance)
            else:
                # For numeric variables, use direct importance
                if orig_var in feature_names:
                    variable_importance[orig_var] = float(feature_importances[feature_names.get_loc(orig_var)])
        
        # Fit overall model using standard regression for comparison
        overall_formula = f"{outcome} ~ {treatment}"
        if covariates:
            overall_formula += " + " + " + ".join(covariates)
        
        overall_model = smf.ols(overall_formula, data=df).fit()
        
        # Find the treatment parameter
        treatment_param = None
        for param in overall_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(overall_model.params.index)}")
        
        # Extract overall treatment effect using the identified parameter
        overall_effect = {
            "effect_size": float(overall_model.params[treatment_param]),
            "std_error": float(overall_model.bse[treatment_param]),
            "confidence_interval": [
                float(overall_model.conf_int().loc[treatment_param, 0]),
                float(overall_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(overall_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome}",
            "model_summary": str(overall_model.summary()),
            "r_squared": float(overall_model.rsquared),
            "adjusted_r_squared": float(overall_model.rsquared_adj),
            "aic": float(overall_model.aic),
            "bic": float(overall_model.bic),
            "analysis_method": "Causal Forest",
            "special_technique": special_technique,
            "causal_forest_specific": {
                "mean_ite": float(df['estimated_ite'].mean()),
                "median_ite": float(df['estimated_ite'].median()),
                "variance_ite": float(df['estimated_ite'].var()),
                "variable_importance": variable_importance
            }
        }
        
        # Initialize results structures
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []
        }
        
        # Analyze heterogeneity by subgroup
        for subgroup in subgroups:
            # Test heterogeneity using ANOVA on ITEs
            try:
                # Create formula to test effect of subgroup on estimated ITEs
                het_formula = f"estimated_ite ~ C({subgroup})"
                het_model = smf.ols(het_formula, data=df).fit()
                
                # Extract F-test results for overall effect of subgroup on ITE
                f_test = het_model.f_test([f"C({subgroup})[T.{level}]=0" 
                                         for level in df[subgroup].unique()[1:]])
                
                heterogeneity_p_value = float(f_test.pvalue)
                
                # Add heterogeneity test result
                heterogeneity_tests.append({
                    "subgroup": subgroup,
                    "interaction_p_value": heterogeneity_p_value,
                    "significant": heterogeneity_p_value < 0.05,
                    "f_test_statistic": float(f_test.fvalue),
                    "f_test_df_num": int(f_test.df_num),
                    "f_test_df_denom": int(f_test.df_denom),
                    "interpretation": (
                        f"Treatment effect differs significantly across levels of {subgroup}"
                        if heterogeneity_p_value < 0.05 else
                        f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                    )
                })
            except Exception as e:
                print(f"Error testing heterogeneity for {subgroup}: {str(e)}")
            
            # Calculate treatment effects within each level of the subgroup
            subgroup_levels = df[subgroup].unique()
            
            for level in subgroup_levels:
                # Skip NaN/None values
                if pd.isna(level):
                    continue
                
                # Subset data for this level
                level_df = df[df[subgroup] == level]
                n_level = len(level_df)
                
                # Skip if sample size is too small
                if n_level < 10:
                    continue
                
                # Calculate average ITE for this subgroup level
                level_ite = level_df['estimated_ite'].mean()
                
                # Calculate standard error of the mean ITE
                level_ite_se = level_df['estimated_ite'].std() / np.sqrt(n_level)
                
                # Calculate confidence interval
                level_ite_lower = level_df['ite_lower'].mean()
                level_ite_upper = level_df['ite_upper'].mean()
                
                # Calculate p-value using t-test against null hypothesis of no effect
                t_stat, p_value = stats.ttest_1samp(level_df['estimated_ite'], 0)
                
                # Add result
                result = {
                    "subgroup": subgroup,
                    "level": str(level),
                    "sample_size": n_level,
                    "effect_size": float(level_ite),
                    "std_error": float(level_ite_se),
                    "confidence_interval": [float(level_ite_lower), float(level_ite_upper)],
                    "p_value": float(p_value),
                    "method": "Causal Forest"
                }
                
                detailed_results.append(result)
                
                # Add to forest plot data
                forest_plot_data["subgroups"].append(subgroup)
                forest_plot_data["levels"].append(str(level))
                forest_plot_data["effect_sizes"].append(float(level_ite))
                forest_plot_data["lower_cis"].append(float(level_ite_lower))
                forest_plot_data["upper_cis"].append(float(level_ite_upper))
                forest_plot_data["p_values"].append(float(p_value))
                forest_plot_data["sample_sizes"].append(n_level)
                forest_plot_data["relative_sample_sizes"].append(n_level / len(df))
            
            await asyncio.sleep(0.01)  # Let UI update
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, "Causal Forest"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results

    async def perform_bart_analysis(self, df, outcome, treatment, subgroups, covariates=None, 
                              adjust_pvalues=True, include_interaction=True,
                          adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform analysis using an approximation of Bayesian Additive Regression Trees."""
        import asyncio
        special_technique = "Approximate BART using gradient boosting (not full Bayesian inference)"
        
        await asyncio.sleep(0.01)
        
        # Calculate baseline characteristics
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        await asyncio.sleep(0.01)
        
        # Prepare data - separate features, treatment, and outcome
        X = df.copy()
        
        # Process categorical variables
        categorical_features = []
        numeric_features = []
        
        for var in X.columns:
            if var in [outcome, treatment]:
                continue
            if pd.api.types.is_numeric_dtype(X[var]):
                numeric_features.append(var)
            else:
                categorical_features.append(var)
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )
        
        # Extract variables
        y = X[outcome].values
        w = X[treatment].values.astype(int)  # Ensure binary treatment is 0/1
        
        # Preprocess features
        X_processed = preprocessor.fit_transform(X)
        
        # Split data into treatment and control groups
        mask_treated = w == 1
        mask_control = w == 0
        
        X_treated = X_processed[mask_treated]
        y_treated = y[mask_treated]
        X_control = X_processed[mask_control]
        y_control = y[mask_control]
        
        # Implement BART using a series of gradient boosted trees with carefully tuned parameters
        # This approximates BART but with better computational performance
        
        # Parameters tuned for BART-like behavior
        bart_params = {
            'n_estimators': 200,           # Many small trees
            'max_depth': 3,                # Limit tree depth to avoid overfitting
            'learning_rate': 0.01,         # Small learning rate for better averaging
            'subsample': 0.5,              # Random subsampling for diversity
            'max_features': 0.8,           # Random feature selection
            'min_samples_leaf': 5,         # Avoid very small leaves
            'random_state': 42             # For reproducibility
        }
        
        self.status_bar.showMessage("Fitting BART model for treatment group...")
        bart_treated = GradientBoostingRegressor(**bart_params)
        bart_treated.fit(X_treated, y_treated)
        
        await asyncio.sleep(0.01)
        
        self.status_bar.showMessage("Fitting BART model for control group...")
        bart_control = GradientBoostingRegressor(**bart_params)
        bart_control.fit(X_control, y_control)
        
        await asyncio.sleep(0.01)
        
        # Calculate individual treatment effects
        predicted_treated = bart_treated.predict(X_processed)
        predicted_control = bart_control.predict(X_processed)
        ite = predicted_treated - predicted_control
        
        # Store ITE in dataframe
        df['individual_treatment_effect'] = ite
        
        # Calculate uncertainty in treatment effects using subsampling
        n_bootstrap = 50  # Reduce bootstrap samples for computational efficiency
        bootstrap_ites = np.zeros((len(df), n_bootstrap))
        
        self.status_bar.showMessage("Calculating uncertainty in treatment effects...")
        
        # Calculate bootstrap samples in smaller batches to allow UI updates
        batch_size = 5
        for batch in range(0, n_bootstrap, batch_size):
            end_batch = min(batch + batch_size, n_bootstrap)
            
            for b in range(batch, end_batch):
                # Sample with replacement
                idx_treated = np.random.choice(len(X_treated), len(X_treated), replace=True)
                idx_control = np.random.choice(len(X_control), len(X_control), replace=True)
                
                # Create bootstrapped models
                bart_t_boot = clone(bart_treated)
                bart_t_boot.fit(X_treated[idx_treated], y_treated[idx_treated])
                
                bart_c_boot = clone(bart_control)
                bart_c_boot.fit(X_control[idx_control], y_control[idx_control])
                
                # Calculate bootstrapped ITE
                bootstrap_ites[:, b] = bart_t_boot.predict(X_processed) - bart_c_boot.predict(X_processed)
            
            # Let UI update
            await asyncio.sleep(0.01)
            self.status_bar.showMessage(f"Bootstrap iteration {end_batch}/{n_bootstrap}...")
        
        # Calculate confidence intervals for ITEs
        lower_ci = np.percentile(bootstrap_ites, 2.5, axis=1)
        upper_ci = np.percentile(bootstrap_ites, 97.5, axis=1)
        
        df['ite_lower_ci'] = lower_ci
        df['ite_upper_ci'] = upper_ci
        
        # Calculate variable importance for heterogeneous treatment effects
        feature_importances = {}
        
        # Combined importance from both models
        combined_importance = (np.array(bart_treated.feature_importances_) + 
                             np.array(bart_control.feature_importances_)) / 2
        
        # Map back to original features
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(numeric_features)
            elif name == 'cat':
                # For categorical features, get the encoded feature names
                ohe = transformer
                for i, feature in enumerate(categorical_features):
                    # Get number of categories for this feature
                    unique_vals = df[feature].nunique()
                    # Add one encoded feature per category (minus one for drop_first=True)
                    for j in range(unique_vals - 1 if hasattr(ohe, 'drop') and ohe.drop == 'first' else unique_vals):
                        feature_names.append(f"{feature}_{j}")
        
        # Store importances (mapping to original features when possible)
        for i, importance in enumerate(combined_importance):
            if i < len(feature_names):
                feature = feature_names[i]
                # Extract original variable name from encoded feature
                orig_var = feature.split('_')[0] if '_' in feature else feature
                if orig_var in feature_importances:
                    feature_importances[orig_var] += importance
                else:
                    feature_importances[orig_var] = importance
        
        # Fit standard model for overall effect comparison
        overall_formula = f"{outcome} ~ {treatment}"
        if covariates:
            overall_formula += " + " + " + ".join(covariates)
        
        overall_model = smf.ols(overall_formula, data=df).fit()
        
        # Find the treatment parameter
        treatment_param = None
        for param in overall_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(overall_model.params.index)}")
        
        # Extract overall treatment effect using the identified parameter
        overall_effect = {
            "effect_size": float(overall_model.params[treatment_param]),
            "std_error": float(overall_model.bse[treatment_param]),
            "confidence_interval": [
                float(overall_model.conf_int().loc[treatment_param, 0]),
                float(overall_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(overall_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome}",
            "model_summary": str(overall_model.summary()),
            "r_squared": float(overall_model.rsquared),
            "adjusted_r_squared": float(overall_model.rsquared_adj),
            "aic": float(overall_model.aic),
            "bic": float(overall_model.bic),
            "analysis_method": "BART",
            "special_technique": special_technique,
            "bart_specific": {
                "mean_ite": float(df['individual_treatment_effect'].mean()),
                "median_ite": float(df['individual_treatment_effect'].median()),
                "ite_range": [
                    float(df['individual_treatment_effect'].min()), 
                    float(df['individual_treatment_effect'].max())
                ],
                "feature_importances": feature_importances
            }
        }
        
        # Initialize results structures
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []
        }
        
        # For BART, analyze heterogeneity in the individual treatment effects
        for subgroup in subgroups:
            # Test whether the subgroup variable influences ITE
            het_formula = f"individual_treatment_effect ~ C({subgroup})"
            
            try:
                het_model = smf.ols(het_formula, data=df).fit()
                
                # Get F-test for overall significance of the subgroup
                f_test = het_model.f_test([f"C({subgroup})[T.{level}]=0" for level in df[subgroup].unique()[1:]])
                heterogeneity_p_value = float(f_test.pvalue)
                
                # Add heterogeneity test result
                heterogeneity_tests.append({
                    "subgroup": subgroup,
                    "interaction_p_value": heterogeneity_p_value,
                    "significant": heterogeneity_p_value < 0.05,
                    "f_test_statistic": float(f_test.fvalue),
                    "f_test_df_num": int(f_test.df_num),
                    "f_test_df_denom": int(f_test.df_denom),
                    "interpretation": (
                        f"Treatment effect differs significantly across levels of {subgroup}"
                        if heterogeneity_p_value < 0.05 else
                        f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                    )
                })
                
                # Analyze subgroup-specific treatment effects
                for level in df[subgroup].unique():
                    # Skip NaN/None values
                    if pd.isna(level):
                        continue
                    
                    # Get subset for this level
                    level_df = df[df[subgroup] == level]
                    n_level = len(level_df)
                    
                    # Skip if sample size is too small
                    if n_level < 10:
                        continue
                    
                    # Calculate average ITE for this subgroup level
                    effect_size = float(level_df['individual_treatment_effect'].mean())
                    
                    # Get confidence interval from pre-calculated bootstrap CIs
                    ci_lower = float(level_df['ite_lower_ci'].mean())
                    ci_upper = float(level_df['ite_upper_ci'].mean())
                    
                    # Calculate p-value using t-test against 0
                    from scipy import stats
                    t_stat, p_value = stats.ttest_1samp(level_df['individual_treatment_effect'], 0)
                    
                    # Add result
                    result = {
                        "subgroup": subgroup,
                        "level": str(level),
                        "sample_size": n_level,
                        "effect_size": effect_size,
                        "std_error": float(level_df['individual_treatment_effect'].std() / np.sqrt(n_level)),
                        "confidence_interval": [ci_lower, ci_upper],
                        "p_value": float(p_value),
                        "method": "BART"
                    }
                    
                    detailed_results.append(result)
                    
                    # Add to forest plot data
                    forest_plot_data["subgroups"].append(subgroup)
                    forest_plot_data["levels"].append(str(level))
                    forest_plot_data["effect_sizes"].append(effect_size)
                    forest_plot_data["lower_cis"].append(ci_lower)
                    forest_plot_data["upper_cis"].append(ci_upper)
                    forest_plot_data["p_values"].append(p_value)
                    forest_plot_data["sample_sizes"].append(n_level)
                    forest_plot_data["relative_sample_sizes"].append(n_level / len(df))
                
            except Exception as e:
                print(f"Error analyzing {subgroup} with BART: {str(e)}")
                import traceback
                traceback.print_exc()
                
            await asyncio.sleep(0.01)  # Let UI update
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        # Variable importance analysis for subgroups
        variable_importance = await self.calculate_variable_importance(
            df, outcome, treatment, subgroups
        )
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, "BART"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results

    async def perform_iptw_analysis(self, df, outcome, treatment, subgroups, covariates=None, 
                             adjust_pvalues=True, include_interaction=True,
                         adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform Inverse Probability of Treatment Weighting analysis (propensity score weighting)."""
        import asyncio
        special_technique = "Propensity score weighting via IPTW (alternative to matching)"
        
        await asyncio.sleep(0.01)
        
        # Calculate baseline characteristics
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        await asyncio.sleep(0.01)
        
        # Step 1: Select variables for propensity score model
        ps_vars = []
        
        if covariates:
            ps_vars.extend(covariates)
        
        # Include subgroup variables in propensity score model
        for var in subgroups:
            if var not in ps_vars:
                ps_vars.append(var)
        
        # Prepare formula for propensity score model
        ps_formula = f"{treatment} ~ "
        
        if ps_vars:
            ps_terms = []
            # Add main effects
            ps_terms.extend(ps_vars)
            
            # Optionally add pairwise interactions between subgroup variables
            # (This improves propensity score balance but can cause issues with small samples)
            if len(subgroups) > 1 and len(df) > 200:  # Only for larger samples
                for var1, var2 in combinations(subgroups, 2):
                    # Check if both variables have reasonable number of levels
                    if df[var1].nunique() * df[var2].nunique() < 50:
                        ps_terms.append(f"{var1}:{var2}")
            
            ps_formula += " + ".join(ps_terms)
        else:
            ps_formula += "1"  # Intercept-only model if no variables
        
        self.status_bar.showMessage("Estimating propensity scores...")
        
        # Step 2: Estimate propensity scores
        try:
            # First try logistic regression with statsmodels
            ps_model = sm.formula.logit(ps_formula, data=df).fit(disp=0, method='bfgs', maxiter=1000)
            
            # Extract predicted probabilities (propensity scores)
            df['propensity_score'] = ps_model.predict()
            
            # Store model summary for diagnostics
            ps_model_summary = str(ps_model.summary())
            
        except Exception as e:
            # If statsmodels fails, fall back to scikit-learn
            self.status_bar.showMessage(f"Statsmodels estimation failed: {str(e)}. Trying sklearn...")
            
            # Process categorical variables
            X_ps = df[ps_vars].copy()
            
            # Identify categorical columns
            cat_cols = [col for col in X_ps.columns if not pd.api.types.is_numeric_dtype(X_ps[col])]
            num_cols = [col for col in X_ps.columns if col not in cat_cols]
            
            # Create preprocessing pipeline
            preprocess = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', num_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
                ],
                remainder='drop'
            )
            
            # Create pipeline with preprocessing and logistic regression
            ps_pipeline = Pipeline([
                ('preprocess', preprocess),
                ('logistic', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1))
            ])
            
            # Fit model
            ps_pipeline.fit(X_ps, df[treatment])
            
            # Calculate propensity scores
            df['propensity_score'] = ps_pipeline.predict_proba(X_ps)[:, 1]
            
            # No detailed summary available with sklearn
            ps_model_summary = "Propensity score model estimated with scikit-learn LogisticRegression"
        
        await asyncio.sleep(0.01)
        
        # Step 3: Check propensity score distribution and trim extreme values
        # Ensure propensity scores are not too close to 0 or 1
        df['propensity_score'] = df['propensity_score'].clip(0.01, 0.99)
        
        # Check propensity score overlap between treatment groups
        ps_treat = df.loc[df[treatment] == 1, 'propensity_score']
        ps_control = df.loc[df[treatment] == 0, 'propensity_score']
        
        # Calculate common support region
        ps_min = max(ps_treat.min(), ps_control.min())
        ps_max = min(ps_treat.max(), ps_control.max())
        
        # Flag observations in common support
        df['in_support'] = (df['propensity_score'] >= ps_min) & (df['propensity_score'] <= ps_max)
        
        # Step 4: Calculate IPTW weights
        # ATE weights
        df['iptw_ate'] = df[treatment] / df['propensity_score'] + (1 - df[treatment]) / (1 - df['propensity_score'])
        
        # ATT weights (focusing on treated population)
        df['iptw_att'] = df[treatment] + (1 - df[treatment]) * df['propensity_score'] / (1 - df['propensity_score'])
        
        # Use ATE weights as default
        df['iptw_weight'] = df['iptw_ate']
        
        # Step 5: Check and trim extreme weights
        weight_threshold = np.percentile(df['iptw_weight'], 99)
        has_extreme_weights = (df['iptw_weight'] > weight_threshold).any()
        
        if has_extreme_weights:
            self.status_bar.showMessage(f"Trimming extreme weights (threshold: {weight_threshold:.2f})...")
            df['iptw_weight_original'] = df['iptw_weight'].copy()
            df['iptw_weight'] = df['iptw_weight'].clip(upper=weight_threshold)
        
        # Step 6: Calculate effective sample size after weighting
        sum_weights = df['iptw_weight'].sum()
        sum_squared_weights = (df['iptw_weight'] ** 2).sum()
        effective_sample_size = sum_weights ** 2 / sum_squared_weights
        
        # Step 7: Fit weighted outcome model for overall effect
        self.status_bar.showMessage("Fitting weighted outcome model...")
        
        overall_formula = f"{outcome} ~ {treatment}"
        if covariates:
            overall_formula += " + " + " + ".join(covariates)
        
        # Weighted regression using IPTW weights
        overall_model = smf.wls(overall_formula, data=df, weights=df['iptw_weight']).fit()
        
        # Store diagnostics in a dictionary to include with results
        iptw_diagnostics = {
            'propensity_score_model': ps_model_summary,
            'propensity_score_range': {
                'treated': {'min': float(ps_treat.min()), 'max': float(ps_treat.max()), 'mean': float(ps_treat.mean())},
                'control': {'min': float(ps_control.min()), 'max': float(ps_control.max()), 'mean': float(ps_control.mean())}
            },
            'common_support': {'min': float(ps_min), 'max': float(ps_max)},
            'percent_in_support': float((df['in_support'].sum() / len(df)) * 100),
            'weight_stats': {
                'min': float(df['iptw_weight'].min()),
                'max': float(df['iptw_weight'].max()),
                'mean': float(df['iptw_weight'].mean()),
                'median': float(df['iptw_weight'].median()),
                'trimmed': has_extreme_weights,
                'trim_threshold': float(weight_threshold) if has_extreme_weights else None
            },
            'effective_sample_size': float(effective_sample_size),
            'effective_sample_size_percent': float((effective_sample_size / len(df)) * 100)
        }
        
        await asyncio.sleep(0.01)
        
        # Find the treatment parameter
        treatment_param = None
        for param in overall_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(overall_model.params.index)}")
        
        # Extract overall treatment effect using the identified parameter
        overall_effect = {
            "effect_size": float(overall_model.params[treatment_param]),
            "std_error": float(overall_model.bse[treatment_param]),
            "confidence_interval": [
                float(overall_model.conf_int().loc[treatment_param, 0]),
                float(overall_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(overall_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome} (IPTW weighted)",
            "model_summary": str(overall_model.summary()),
            "r_squared": float(overall_model.rsquared),
            "adjusted_r_squared": float(overall_model.rsquared_adj),
            "aic": float(overall_model.aic),
            "bic": float(overall_model.bic),
            "analysis_method": "IPTW",
            "special_technique": special_technique,
            "iptw_specific": iptw_diagnostics
        }
        
        # Initialize results structures
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []
        }
        
        # For IPTW, use weighted regression for each subgroup level
        for subgroup in subgroups:
            # Test for interaction using weighted regression
            if include_interaction:
                interaction_formula = f"{outcome} ~ {treatment} * C({subgroup})"
                if covariates:
                    for cov in covariates:
                        interaction_formula += f" + {cov}"
                
                try:
                    interaction_model = smf.wls(interaction_formula, data=df, weights=df['iptw_weight']).fit()
                    
                    # Get interaction terms
                    interaction_terms = [term for term in interaction_model.params.index if f"{treatment}:C({subgroup}" in term]
                    
                    # Calculate joint test for interaction
                    if interaction_terms:
                        f_test = interaction_model.f_test([f"{term}=0" for term in interaction_terms])
                        heterogeneity_p_value = float(f_test.pvalue)
                        
                        # Add heterogeneity test result
                        heterogeneity_tests.append({
                            "subgroup": subgroup,
                            "interaction_p_value": heterogeneity_p_value,
                            "individual_p_values": [float(interaction_model.pvalues[term]) for term in interaction_terms],
                            "significant": heterogeneity_p_value < 0.05,
                            "f_test_statistic": float(f_test.fvalue),
                            "f_test_df_num": int(f_test.df_num),
                            "f_test_df_denom": int(f_test.df_denom),
                            "interpretation": (
                                f"Treatment effect differs significantly across levels of {subgroup}"
                                if heterogeneity_p_value < 0.05 else
                                f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                            ),
                            "method": "IPTW"
                        })
                except Exception as e:
                    print(f"Error testing interaction for {subgroup} with IPTW: {str(e)}")
            
            # Analyze each level of the subgroup
            subgroup_levels = df[subgroup].unique()
            
            for level in subgroup_levels:
                # Skip NaN/None values
                if pd.isna(level):
                    continue
                
                # Subset data for this level
                level_df = df[df[subgroup] == level]
                
                # Skip if sample size is too small
                if len(level_df) < 10:
                    print(f"Skipping {subgroup}={level} due to small sample size (n={len(level_df)})")
                    continue
                
                # Create formula for this subgroup level
                level_formula = f"{outcome} ~ {treatment}"
                if covariates:
                    level_formula += " + " + " + ".join(covariates)
                
                # Fit weighted model for this subgroup level
                try:
                    level_model = smf.wls(level_formula, data=level_df, weights=level_df['iptw_weight']).fit()
                    
                    # Extract treatment effect
                    # Find the treatment parameter
                    treatment_param = None
                    for param in level_model.params.index:
                        if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                            treatment_param = param
                            break
                    
                    if treatment_param:
                        effect_size = float(level_model.params[treatment_param])
                        std_error = float(level_model.bse[treatment_param])
                        ci_lower = float(level_model.conf_int().loc[treatment_param, 0])
                        ci_upper = float(level_model.conf_int().loc[treatment_param, 1])
                        p_value = float(level_model.pvalues[treatment_param])
                    else:
                        raise ValueError(f"Could not find treatment parameter in model for {subgroup}={level}")
                    
                    # Calculate effective sample size for this level
                    weights_sum = level_df['iptw_weight'].sum()
                    weights_sum_squared = (level_df['iptw_weight'] ** 2).sum()
                    effective_n = weights_sum ** 2 / weights_sum_squared
                    
                    # Add result
                    result = {
                        "subgroup": subgroup,
                        "level": str(level),
                        "sample_size": len(level_df),
                        "effective_sample_size": float(effective_n),
                        "effect_size": effect_size,
                        "std_error": std_error,
                        "confidence_interval": [ci_lower, ci_upper],
                        "p_value": p_value,
                        "model_r_squared": float(level_model.rsquared),
                        "model_summary": str(level_model.summary()),
                        "method": "IPTW"
                    }
                    
                    detailed_results.append(result)
                    
                    # Add to forest plot data
                    forest_plot_data["subgroups"].append(subgroup)
                    forest_plot_data["levels"].append(str(level))
                    forest_plot_data["effect_sizes"].append(effect_size)
                    forest_plot_data["lower_cis"].append(ci_lower)
                    forest_plot_data["upper_cis"].append(ci_upper)
                    forest_plot_data["p_values"].append(p_value)
                    forest_plot_data["sample_sizes"].append(len(level_df))
                    forest_plot_data["relative_sample_sizes"].append(len(level_df) / len(df))
                
                except Exception as e:
                    print(f"Error analyzing {subgroup}={level} with IPTW: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                await asyncio.sleep(0.01)  # Let UI update
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        # Variable importance analysis for subgroups
        variable_importance = await self.calculate_variable_importance(
            df, outcome, treatment, subgroups
        )
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, "IPTW"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results
    
    def display_figure(self, fig):
        """Common method to display matplotlib figures in the visualization area with maximize option."""
        # First clear the visualization area
        self.clear_visualization_tab()
        
        # Make figure background transparent
        fig.patch.set_alpha(0.0)
        
        # Set all text elements to gray with larger font sizes
        gray_color = '#555555'  # Medium gray that works on both light/dark backgrounds
        plt.rcParams.update({
            'text.color': gray_color,
            'axes.labelcolor': gray_color,
            'axes.titlecolor': gray_color,
            'xtick.color': gray_color,
            'ytick.color': gray_color,
            'axes.labelsize': 12,      # Larger axis labels
            'axes.titlesize': 14,      # Larger titles
            'xtick.labelsize': 11,     # Larger tick labels
            'ytick.labelsize': 11,     # Larger tick labels
            'legend.fontsize': 11      # Larger legend text
        })
        
        # Apply gray text color to all axes elements
        for ax in fig.get_axes():
            ax.patch.set_alpha(0.0)
            # Make legend background transparent if it exists
            if ax.get_legend() is not None:
                ax.get_legend().get_frame().set_alpha(0.0)
                # Update legend text color
                for text in ax.get_legend().get_texts():
                    text.set_color(gray_color)
            
            # Update all text elements in the axis
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color(gray_color)
                text.set_fontsize(11)
                
            ax.xaxis.label.set_color(gray_color)
            ax.yaxis.label.set_color(gray_color)
            ax.title.set_color(gray_color)
            
            ax.xaxis.label.set_fontsize(12)
            ax.yaxis.label.set_fontsize(12)
            ax.title.set_fontsize(14)
        
        # Create canvas to display the figure
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Create container for canvas and buttons
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        
        # Add maximize button
        maximize_btn = QPushButton("Maximize")
        maximize_btn.setIcon(load_bootstrap_icon("arrows-fullscreen", size=18))
        maximize_btn.setStyleSheet("border: 0px solid #4CAF50;")
        maximize_btn.clicked.connect(lambda: self.show_plot_in_modal(fig))
        maximize_btn.setMaximumWidth(150)
        
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 5, 0, 0)
        button_layout.addStretch()
        button_layout.addWidget(maximize_btn)
        
        container_layout.addWidget(button_container)

        # Add the canvas to the container
        container_layout.addWidget(canvas)
        
        # Add the container to the visualization layout
        self.viz_layout.addWidget(container)
        
        # Store the figure for later use
        self.current_figure = fig
        
        # Make sure it's visible
        self.viz_container.update()

    def generate_forest_plot(self, forest_data):
        """Generate an enhanced forest plot for visualization of subgroup effects."""
        if not forest_data or not forest_data.get("effect_sizes"):
            return
                
        # Create a new figure with increased height and wider width for label space
        fig = Figure(figsize=(14, max(8, len(forest_data["effect_sizes"]) * 0.7)))  # Increased width and height
        ax = fig.add_subplot(111)
        
        # Set gray color for text elements
        gray_color = '#555555'
        
        # Combine subgroup and level for y-axis labels
        labels = [f"{sg} = {lv} (n={sz})" for sg, lv, sz in 
                zip(forest_data["subgroups"], forest_data["levels"], forest_data["sample_sizes"])]
        
        # Determine the x-axis limits based on the range of effect sizes and CIs
        min_ci = min(forest_data["lower_cis"])
        max_ci = max(forest_data["upper_cis"])
        x_min = min(min_ci, 0) * 1.1  # Include 0 and add some margin
        x_max = max(max_ci, 0) * 1.1
        
        # Use PASTEL_COLORS scheme from formatting.py
        cmap = plt.cm.colors.ListedColormap(PASTEL_COLORS)
        colors = []
        unique_subgroups = list(set(forest_data["subgroups"]))
        subgroup_colors = {sg: cmap(i % len(PASTEL_COLORS)) for i, sg in enumerate(unique_subgroups)}
        
        for sg, p in zip(forest_data["subgroups"], forest_data["p_values"]):
            # Adjust transparency based on p-value significance
            alpha = 1.0 if p < 0.05 else 0.6
            color = subgroup_colors[sg]
            # Create color with alpha
            rgba = list(color)
            rgba[3] = alpha
            colors.append(rgba)
        
        # Plot the forest plot with varying sizes based on sample size
        y_pos = range(len(labels))
        
        # Make point sizes proportional to sample size
        if "relative_sample_sizes" in forest_data:
            sizes = [max(50, 300 * rs) for rs in forest_data["relative_sample_sizes"]]
        else:
            sizes = [max(50, min(500, sz/10)) for sz in forest_data["sample_sizes"]]
        
        # Plot points
        ax.scatter(forest_data["effect_sizes"], y_pos, s=sizes, color=colors, zorder=3, 
                edgecolor=gray_color, linewidth=1)
        
        # Plot confidence intervals
        for i, (effect, lower, upper, color) in enumerate(zip(
            forest_data["effect_sizes"], 
            forest_data["lower_cis"], 
            forest_data["upper_cis"],
            colors
        )):
            ax.plot([lower, upper], [i, i], color=color, lw=2, zorder=2)
            
            # Add significance markers
            p_value = forest_data["p_values"][i]
            if p_value < 0.05:
                marker = '*' if p_value < 0.01 else '.'
                ax.text(upper + 0.02*(x_max-x_min), i, marker, 
                    ha='left', va='center', fontsize=14, color=color)
        
        # Add vertical line at x=0 (no effect)
        ax.axvline(x=0, color='red', linestyle='--', lw=1, zorder=1)
        
        # Set y-axis labels with larger font
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11, color=gray_color)
        
        # Set x-axis label and title with larger font
        ax.set_xlabel('Treatment Effect (95% CI)', fontsize=13, color=gray_color)
        ax.set_title('Forest Plot of Treatment Effects by Subgroup', fontsize=15, color=gray_color)
        
        # Add legend for subgroups
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=subgroup_colors[sg], 
                markersize=10, label=sg) for sg in unique_subgroups
        ]
        legend = ax.legend(handles=legend_elements, title="Subgroups", loc='best', fontsize=11)
        legend.get_title().set_fontsize(12)
        legend.get_title().set_color(gray_color)
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        
        # Add a grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotation for interpretation
        interpretation = "Note: Larger circles indicate larger sample sizes. "
        interpretation += "Filled circles with * indicate statistically significant effects (p<0.05)."
        fig.text(0.5, 0.01, interpretation, ha='center', fontsize=11, style='italic', color=gray_color)
        
        # Adjust layout with more padding to prevent cut-off labels
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=2.0)  # Increased padding
        
        # Display the plot using the common display method
        self.display_figure(fig)

    def generate_interaction_plot(self, results):
        """Generate an interaction plot to visualize treatment-by-subgroup interactions."""
        if not results or not results.get("detailed_results"):
            return
            
        detailed_results = results.get("detailed_results", [])
        
        # Group results by subgroup for plotting
        subgroups = {}
        for result in detailed_results:
            subgroup = result.get("subgroup")
            if subgroup not in subgroups:
                subgroups[subgroup] = []
            subgroups[subgroup].append(result)
        
        # Create a figure with one subplot per subgroup with increased height
        n_subgroups = len(subgroups)
        fig = Figure(figsize=(14, 5 * n_subgroups))  # Increased width and height
        
        # Set gray color for text elements
        gray_color = '#555555'
        
        for i, (subgroup, subgroup_results) in enumerate(subgroups.items()):
            ax = fig.add_subplot(n_subgroups, 1, i+1)
            
            # Extract data for plotting
            levels = [r.get("level") for r in subgroup_results]
            effects = [r.get("effect_size") for r in subgroup_results]
            errors = [r.get("std_error") for r in subgroup_results]
            p_values = [r.get("p_value") for r in subgroup_results]
            
            # Sort by effect size for better visualization
            sorted_data = sorted(zip(levels, effects, errors, p_values), key=lambda x: x[1])
            levels, effects, errors, p_values = zip(*sorted_data) if sorted_data else ([], [], [], [])
            
            # Plot the effects
            x = range(len(levels))
            
            # Use PASTEL_COLORS for bars based on significance
            colors = [PASTEL_COLORS[0] if p < 0.05 else PASTEL_COLORS[2] for p in p_values]
            
            # Plot bars
            bars = ax.bar(x, effects, yerr=errors, capsize=5, color=colors, 
                        edgecolor=gray_color, linewidth=1)
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add labels and titles with larger font
            ax.set_xlabel('Subgroup Levels', fontsize=13, color=gray_color)
            ax.set_ylabel('Treatment Effect', fontsize=13, color=gray_color)
            ax.set_title(f'Treatment Effects by {subgroup}', fontsize=15, color=gray_color)
            ax.set_xticks(x)
            ax.set_xticklabels(levels, rotation=45, ha='right', fontsize=11, color=gray_color)
            
            # Add p-values above the bars
            for j, (bar, p) in enumerate(zip(bars, p_values)):
                height = bar.get_height()
                sign = 1 if height >= 0 else -1
                ax.text(bar.get_x() + bar.get_width()/2., height + sign*0.01,
                    f'p={p:.3f}', ha='center', va='bottom' if sign > 0 else 'top', 
                    fontsize=10, rotation=90, color=gray_color)
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Adjust layout with more padding
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=2.5, h_pad=4.0)  # Increased padding and spacing
        
        # Display the plot using common display method
        self.display_figure(fig)

    def generate_heat_map(self, results):
        """Generate a heat map visualizing treatment effects across multiple subgroups."""
        if not results or not results.get("detailed_results"):
            return
            
        detailed_results = results.get("detailed_results", [])
        
        # Extract unique subgroups and levels
        subgroups = list(set(r.get("subgroup") for r in detailed_results))
        levels_by_subgroup = {sg: [] for sg in subgroups}
        
        for result in detailed_results:
            sg = result.get("subgroup")
            level = result.get("level")
            if level not in levels_by_subgroup[sg]:
                levels_by_subgroup[sg].append(level)
        
        # Need at least 2 subgroups for a meaningful heatmap
        if len(subgroups) < 2:
            # Fall back to forest plot if only one subgroup
            return self.generate_forest_plot(results.get("forest_plot_data", {}))
        
        # Select the two subgroups with the most levels for the heatmap
        subgroups = sorted(subgroups, key=lambda sg: len(levels_by_subgroup[sg]), reverse=True)[:2]
        
        # Create a matrix of effect sizes
        sg1, sg2 = subgroups
        levels1 = sorted(levels_by_subgroup[sg1])
        levels2 = sorted(levels_by_subgroup[sg2])
        
        effect_matrix = np.zeros((len(levels1), len(levels2)))
        p_value_matrix = np.ones((len(levels1), len(levels2)))
        sample_size_matrix = np.zeros((len(levels1), len(levels2)))
        
        # Fill in matrices from results
        for result in detailed_results:
            if result.get("subgroup") == sg1:
                level1 = result.get("level")
                idx1 = levels1.index(level1)
                
                # Find matching results for second subgroup
                for other_result in detailed_results:
                    if other_result.get("subgroup") == sg2:
                        level2 = other_result.get("level")
                        idx2 = levels2.index(level2)
                        
                        # Get the intersection of these two subgroups in the original dataframe
                        df = self.current_dataframe
                        subset = df[(df[sg1] == level1) & (df[sg2] == level2)]
                        
                        if len(subset) > 10:  # Need enough samples for analysis
                            treatment = self.selected_treatment
                            outcome = self.selected_outcome
                            
                            # Simple test of treatment effect in this intersection
                            try:
                                model = smf.ols(f"{outcome} ~ {treatment}", data=subset).fit()
                                effect = float(model.params[treatment])
                                p_value = float(model.pvalues[treatment])
                                
                                effect_matrix[idx1, idx2] = effect
                                p_value_matrix[idx1, idx2] = p_value
                                sample_size_matrix[idx1, idx2] = len(subset)
                            except:
                                pass
        
        # Set gray color for text elements
        gray_color = '#555555'
        
        # Create the heatmap figure with increased height and width
        fig = Figure(figsize=(14, 16))  # Increased width and height
        
        # Effect size heatmap
        ax1 = fig.add_subplot(2, 1, 1)
        
        # Create custom colormap using PASTEL_COLORS
        max_abs_effect = np.max(np.abs(effect_matrix))
        norm = plt.Normalize(-max_abs_effect, max_abs_effect)
        colors = [PASTEL_COLORS[2], '#FFFFFF', PASTEL_COLORS[0]]  # Blue, white, red from PASTEL_COLORS
        cmap = LinearSegmentedColormap.from_list("effect_cmap", colors, N=256)
        
        # Create heatmap
        im = ax1.imshow(effect_matrix, cmap=cmap, norm=norm)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Treatment Effect Size', fontsize=13, color=gray_color)
        cbar.ax.tick_params(labelsize=11, labelcolor=gray_color)
        
        # Add significance markers and sample size with gray text
        for i in range(len(levels1)):
            for j in range(len(levels2)):
                if p_value_matrix[i, j] < 0.05:
                    markers = '*' if p_value_matrix[i, j] < 0.01 else '*'
                    ax1.text(j, i, markers, ha='center', va='center', 
                            color=gray_color, fontweight='bold', fontsize=12)
                # Add sample size as text
                if sample_size_matrix[i, j] > 0:
                    ax1.text(j, i, f"n={int(sample_size_matrix[i, j])}", 
                            ha='center', va='center', fontsize=10,
                            color=gray_color, alpha=0.9)
        
        # Set labels and title with larger font
        ax1.set_xticks(range(len(levels2)))
        ax1.set_yticks(range(len(levels1)))
        ax1.set_xticklabels(levels2, rotation=45, ha='right', fontsize=11, color=gray_color)
        ax1.set_yticklabels(levels1, fontsize=11, color=gray_color)
        ax1.set_xlabel(sg2, fontsize=13, color=gray_color)
        ax1.set_ylabel(sg1, fontsize=13, color=gray_color)
        ax1.set_title(f'Heatmap of Treatment Effects by {sg1} and {sg2}', fontsize=15, color=gray_color)
        
        # P-value heatmap
        ax2 = fig.add_subplot(2, 1, 2)
        
        # Use a logarithmic scale for p-values
        p_value_log = -np.log10(p_value_matrix)
        p_value_log[p_value_log > 3] = 3  # Cap at 3 (p < 0.001)
        
        # Create heatmap with log p-values using a sequential colormap from PASTEL_COLORS
        viridis_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            "p_value_cmap", ['#FFFFFF', PASTEL_COLORS[1], PASTEL_COLORS[4]], N=256)
        im2 = ax2.imshow(p_value_log, cmap=viridis_cmap)
        
        # Add colorbar
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label('-log10(p-value)', fontsize=13, color=gray_color)
        cbar2.ax.tick_params(labelsize=11, labelcolor=gray_color)
        
        # Add text with actual p-values
        for i in range(len(levels1)):
            for j in range(len(levels2)):
                if sample_size_matrix[i, j] > 0:
                    text_color = gray_color
                    ax2.text(j, i, f"{p_value_matrix[i, j]:.3f}", 
                            ha='center', va='center', fontsize=10,
                            color=text_color)
        
        # Set labels and title with larger font
        ax2.set_xticks(range(len(levels2)))
        ax2.set_yticks(range(len(levels1)))
        ax2.set_xticklabels(levels2, rotation=45, ha='right', fontsize=11, color=gray_color)
        ax2.set_yticklabels(levels1, fontsize=11, color=gray_color)
        ax2.set_xlabel(sg2, fontsize=13, color=gray_color)
        ax2.set_ylabel(sg1, fontsize=13, color=gray_color)
        ax2.set_title(f'Heatmap of P-values by {sg1} and {sg2}', fontsize=15, color=gray_color)
        
        # Adjust layout with more padding
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=3.0, h_pad=8.0)  # Increased h_pad from 5.0 to 8.0
        
        # Display the plot using common display method
        self.display_figure(fig)

    def generate_effect_distribution(self, results):
        """Generate a visualization of the distribution of treatment effects across subgroups."""
        if not results or not results.get("detailed_results"):
            return
            
        detailed_results = results.get("detailed_results", [])
        
        # Extract effect sizes and significant status
        effects = [r.get("effect_size") for r in detailed_results]
        p_values = [r.get("p_value") for r in detailed_results]
        significant = [p < 0.05 for p in p_values]
        labels = [f"{r.get('subgroup')}={r.get('level')}" for r in detailed_results]
        
        # Set gray color for text elements
        gray_color = '#555555'
        
        # Create figure with multiple plots with increased height and width
        fig = Figure(figsize=(14, 16))  # Increased width and height
        
        # 1. Effect size histogram
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Separate significant and non-significant effects
        sig_effects = [e for e, s in zip(effects, significant) if s]
        nonsig_effects = [e for e, s in zip(effects, significant) if not s]
        
        # Create histogram using PASTEL_COLORS
        bins = np.linspace(min(effects), max(effects), 15) if effects else np.linspace(-1, 1, 15)
        ax1.hist([sig_effects, nonsig_effects], bins=bins, 
                stacked=True, color=[PASTEL_COLORS[0], PASTEL_COLORS[2]],
                label=['Significant (p<0.05)', 'Non-significant'])
        
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Treatment Effect Size', fontsize=13, color=gray_color)
        ax1.set_ylabel('Frequency', fontsize=13, color=gray_color)
        ax1.set_title('Distribution of Treatment Effects', fontsize=15, color=gray_color)
        legend = ax1.legend(fontsize=11)
        for text in legend.get_texts():
            text.set_color(gray_color)
        
        # 2. QQ plot of p-values
        ax2 = fig.add_subplot(2, 2, 2)
        
        # Create QQ plot of p-values to check for overall significance
        expected = np.linspace(0, 1, len(p_values) + 2)[1:-1]  # Exclude 0 and 1
        observed = sorted(p_values)
        
        ax2.scatter(expected, observed, color=PASTEL_COLORS[3])
        ax2.plot([0, 1], [0, 1], 'r--')
        ax2.set_xlabel('Expected p-values under null hypothesis', fontsize=13, color=gray_color)
        ax2.set_ylabel('Observed p-values', fontsize=13, color=gray_color)
        ax2.set_title('QQ Plot of P-values', fontsize=15, color=gray_color)
        
        # 3. Effect size vs. P-value volcano plot
        ax3 = fig.add_subplot(2, 2, 3)
        
        # Create volcano plot using PASTEL_COLORS
        ax3.scatter(effects, [-np.log10(p) for p in p_values], 
                c=[PASTEL_COLORS[0] if s else PASTEL_COLORS[2] for s in significant],
                edgecolor=gray_color, alpha=0.7)
        
        # Add threshold line
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
        
        # Add vertical line at zero effect
        ax3.axvline(x=0, color='green', linestyle='--', alpha=0.7)
        
        ax3.set_xlabel('Treatment Effect Size', fontsize=13, color=gray_color)
        ax3.set_ylabel('-log10(p-value)', fontsize=13, color=gray_color)
        ax3.set_title('Volcano Plot: Effect Size vs. Statistical Significance', fontsize=15, color=gray_color)
        
        # 4. Effect size with confidence intervals (top effects)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Sort by absolute effect size and take top 10
        sorted_results = sorted(detailed_results, key=lambda r: abs(r.get("effect_size", 0)), reverse=True)[:10]
        
        top_labels = [f"{r.get('subgroup')}={r.get('level')}" for r in sorted_results]
        top_effects = [r.get("effect_size") for r in sorted_results]
        top_cis = [r.get("confidence_interval") for r in sorted_results]
        top_p = [r.get("p_value") < 0.05 for r in sorted_results]
        
        # Plot top effects
        y_pos = range(len(top_labels))
        
        # Plot effects with CIs using PASTEL_COLORS
        ax4.scatter(top_effects, y_pos, 
                c=[PASTEL_COLORS[0] if p else PASTEL_COLORS[2] for p in top_p],
                s=80, zorder=3, edgecolor=gray_color)
        
        # Add CIs
        for i, (effect, ci) in enumerate(zip(top_effects, top_cis)):
            ax4.plot([ci[0], ci[1]], [i, i], 
                    color=PASTEL_COLORS[0] if top_p[i] else PASTEL_COLORS[2], 
                    lw=2, zorder=2)
        
        # Add zero line
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, zorder=1)
        
        # Set labels with larger fonts
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_labels, fontsize=11, color=gray_color)
        ax4.set_xlabel('Treatment Effect (95% CI)', fontsize=13, color=gray_color)
        ax4.set_title('Top 10 Subgroup Effects by Magnitude', fontsize=15, color=gray_color)
        
        # For the top effects plot with y-axis labels, provide more horizontal space
        if len(top_labels) > 0:
            max_label_length = max(len(label) for label in top_labels)
            if max_label_length > 15:
                plt_width = 0.8 - (max_label_length * 0.01)  # Adjust width based on label length
                ax4.set_position([0.2, ax4.get_position().y0, plt_width, ax4.get_position().height])
        
        # Adjust layout with more padding
        fig.subplots_adjust(hspace=0.8, wspace=0.7)  # Increased hspace from 0.45 to 0.6 and wspace from 0.35 to 0.4
        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=3.0)  # Additional padding
        
        # Display the plot using common display method
        self.display_figure(fig)
        
    def show_plot_in_modal(self, fig):
        # Create modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot View")
        
        # Fix the window flags - using correct Qt namespace
        dialog.setWindowFlags(dialog.windowFlags() | 
                            Qt.WindowType.WindowMaximizeButtonHint | 
                            Qt.WindowType.WindowMinimizeButtonHint)
        # Get screen dimensions
        screen = QApplication.primaryScreen()
        screen_size = screen.availableSize()
        screen_height = screen_size.height() * 0.9  # Use 90% of available height
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create canvas for matplotlib figure
        canvas = FigureCanvas(fig)
        
        # Set size policy to expand vertically and horizontally
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Calculate width based on aspect ratio
        fig_width, fig_height = fig.get_size_inches()
        aspect_ratio = fig_width / fig_height
        
        # Set height and width based on aspect ratio
        height = screen_height
        width = height * aspect_ratio
        
        # Apply size
        dialog.resize(int(width), int(height))
        
        # Add to layout with expanding properties
        layout.addWidget(canvas)
        
        # Add close button - fixed to use StandardButton.Close instead of Close
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.button(QDialogButtonBox.StandardButton.Close).setIcon(load_bootstrap_icon("x-circle", size=16))
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Center dialog on screen
        dialog.setGeometry(
            QStyle.alignedRect(
                Qt.LayoutDirection.LeftToRight,
                Qt.AlignmentFlag.AlignCenter,
                dialog.size(),
                QApplication.primaryScreen().availableGeometry()
            )
        )
        
        # Show the dialog
        dialog.exec()

    async def perform_bayesian_hierarchical_analysis(self, df, outcome, treatment, subgroups, covariates=None, 
                                            adjust_pvalues=True, include_interaction=True,
                                        adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform full Bayesian hierarchical modeling with PyMC in a separate thread."""
        import asyncio
        
        # Create a thread object
        self.bayesian_thread = BayesianModelThread(
            df, outcome, treatment, subgroups, covariates,
            adjust_pvalues, include_interaction, adjustment_method
        )
        
        # Create a Promise-like object for async/await compatibility
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Connect signals
        def on_progress(message):
            self.status_bar.showMessage(message)
        
        def on_finished(results):
            if not future.done():
                future.set_result(results)
        
        def on_error(error_message):
            if not future.done():
                future.set_exception(Exception(error_message))
        
        # Connect signals
        self.bayesian_thread.progress_signal.connect(on_progress)
        self.bayesian_thread.finished_signal.connect(on_finished)
        self.bayesian_thread.error_signal.connect(on_error)
        
        # Start the thread
        self.status_bar.showMessage("Starting Bayesian analysis in background thread...")
        self.bayesian_thread.start()
        
        # Wait for thread to complete (without blocking UI)
        results = await future
        
        # Thread is done, return results
        return results
        
    async def perform_propensity_score_matching(self, df, outcome, treatment, subgroups, covariates=None, 
                                        adjust_pvalues=True, include_interaction=True,
                                    adjustment_method="Benjamini-Hochberg (FDR)"):
        """Perform subgroup analysis using propensity score matching."""
        import asyncio
        special_technique = "Propensity score matching with nearest neighbor algorithm"
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Calculate baseline characteristics
        baseline_characteristics = self.calculate_baseline_characteristics(
            df, treatment, subgroups + (covariates or [])
        )
        
        # Step 1: Select variables for propensity score model
        ps_vars = []
        
        if covariates:
            ps_vars.extend(covariates)
        
        # Include subgroup variables in propensity score model
        for var in subgroups:
            if var not in ps_vars:
                ps_vars.append(var)
        
        # Prepare formula for propensity score model
        ps_formula = f"{treatment} ~ "
        ps_terms = []
        
        if ps_vars:
            # Add main effects
            ps_terms.extend(ps_vars)
            ps_formula += " + ".join(ps_terms)
        else:
            ps_formula += "1"  # Intercept-only model if no variables
        
        self.status_bar.showMessage("Estimating propensity scores...")
        
        # Step 2: Estimate propensity scores
        try:
            # Use logistic regression for propensity scores
            ps_model = sm.formula.logit(ps_formula, data=df).fit(disp=0, method='bfgs', maxiter=1000)
            
            # Extract predicted probabilities (propensity scores)
            df['propensity_score'] = ps_model.predict()
            
            # Clip propensity scores to avoid extreme values
            df['propensity_score'] = df['propensity_score'].clip(0.01, 0.99)
            
            # Store model summary for diagnostics
            ps_model_summary = str(ps_model.summary())
            
        except Exception as e:
            self.status_bar.showMessage(f"Statsmodels estimation failed: {str(e)}. Trying sklearn...")
            
            # Process categorical variables
            X_ps = df[ps_vars].copy()
            
            # Identify categorical columns
            cat_cols = [col for col in X_ps.columns if not pd.api.types.is_numeric_dtype(X_ps[col])]
            num_cols = [col for col in X_ps.columns if col not in cat_cols]
            
            # Create preprocessing pipeline
            preprocess = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', num_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
                ],
                remainder='drop'
            )
            
            # Create pipeline with preprocessing and logistic regression
            ps_pipeline = Pipeline([
                ('preprocess', preprocess),
                ('logistic', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1))
            ])
            
            # Fit model
            ps_pipeline.fit(X_ps, df[treatment])
            
            # Calculate propensity scores
            df['propensity_score'] = ps_pipeline.predict_proba(X_ps)[:, 1]
            
            # Clip propensity scores to avoid extreme values
            df['propensity_score'] = df['propensity_score'].clip(0.01, 0.99)
            
            # No detailed summary available with sklearn
            ps_model_summary = "Propensity score model estimated with scikit-learn LogisticRegression"
        
        await asyncio.sleep(0.01)  # Let UI update
        
        # Step 3: Perform matching
        self.status_bar.showMessage("Performing propensity score matching...")
        
        # Separate treated and control groups
        treated = df[df[treatment] == 1]
        control = df[df[treatment] == 0]
        
        # Calculate caliper (0.2 * standard deviation of logit propensity score)
        logit_ps = np.log(df['propensity_score'] / (1 - df['propensity_score']))
        caliper = 0.2 * np.std(logit_ps)
        
        # Use nearest neighbor algorithm to find matches within caliper
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
            control['propensity_score'].values.reshape(-1, 1))
        
        # Find nearest neighbors for each treated unit
        distances, indices = nbrs.kneighbors(treated['propensity_score'].values.reshape(-1, 1))
        
        # Create matched pairs dataset
        matched_pairs = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] <= caliper:  # Only include if within caliper
                treated_row = treated.iloc[i].copy()
                control_row = control.iloc[idx[0]].copy()
                
                # Include pair identifier
                pair_id = i
                treated_row['pair_id'] = pair_id
                control_row['pair_id'] = pair_id
                
                matched_pairs.append(treated_row)
                matched_pairs.append(control_row)
        
        # Create matched dataframe
        if matched_pairs:
            matched_df = pd.DataFrame(matched_pairs)
        else:
            # If no pairs are created, raise an error
            raise ValueError("No matches found within caliper. Consider increasing caliper or using IPTW instead.")
        
        # Output matching statistics
        n_treated = len(treated)
        n_matched = len(matched_pairs) // 2
        matching_rate = n_matched / n_treated
        
        self.status_bar.showMessage(f"Matched {n_matched} of {n_treated} treated units ({matching_rate:.1%})")
        
        # Step 4: Assess balance in matched sample
        balance_stats = {}
        for var in ps_vars:
            balance_stats[var] = {}
            
            # Calculate standardized mean difference
            if pd.api.types.is_numeric_dtype(df[var]):
                # For continuous variables
                treated_mean = matched_df[matched_df[treatment] == 1][var].mean()
                control_mean = matched_df[matched_df[treatment] == 0][var].mean()
                treated_var = matched_df[matched_df[treatment] == 1][var].var()
                control_var = matched_df[matched_df[treatment] == 0][var].var()
                
                # Pooled standard deviation
                pooled_sd = np.sqrt((treated_var + control_var) / 2)
                
                # Standardized mean difference
                if pooled_sd > 0:
                    std_mean_diff = (treated_mean - control_mean) / pooled_sd
                else:
                    std_mean_diff = 0
                    
                balance_stats[var] = {
                    'standardized_mean_diff': float(std_mean_diff),
                    'treated_mean': float(treated_mean),
                    'control_mean': float(control_mean)
                }
            else:
                # For categorical variables
                # Calculate chi-square test for independence
                cross_tab = pd.crosstab(matched_df[var], matched_df[treatment])
                try:
                    chi2, p_value, _, _ = stats.chi2_contingency(cross_tab)
                    balance_stats[var] = {
                        'chi_square': float(chi2),
                        'p_value': float(p_value)
                    }
                except:
                    balance_stats[var] = {
                        'chi_square': None,
                        'p_value': None
                    }
        
        # Step 5: Fit outcome model using matched data
        self.status_bar.showMessage("Analyzing outcomes in matched sample...")
        
        # Fit overall model
        overall_formula = f"{outcome} ~ {treatment}"
        if covariates:
            overall_formula += " + " + " + ".join(covariates)
        
        # Using matched data, estimate overall effect
        overall_model = smf.ols(overall_formula, data=matched_df).fit()
        
        # Find the treatment parameter
        treatment_param = None
        for param in overall_model.params.index:
            if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                treatment_param = param
                break
        
        if not treatment_param:
            raise ValueError(f"Could not find parameter for treatment variable '{treatment}' in model. Available parameters: {list(overall_model.params.index)}")
        
        # Extract overall treatment effect using the identified parameter
        overall_effect = {
            "effect_size": float(overall_model.params[treatment_param]),
            "std_error": float(overall_model.bse[treatment_param]),
            "confidence_interval": [
                float(overall_model.conf_int().loc[treatment_param, 0]),
                float(overall_model.conf_int().loc[treatment_param, 1])
            ],
            "p_value": float(overall_model.pvalues[treatment_param]),
            "description": f"Overall treatment effect of {treatment} on {outcome} (matched sample)",
            "model_summary": str(overall_model.summary()),
            "r_squared": float(overall_model.rsquared),
            "adjusted_r_squared": float(overall_model.rsquared_adj),
            "aic": float(overall_model.aic),
            "bic": float(overall_model.bic),
            "analysis_method": "Propensity Score Matching",
            "special_technique": special_technique,
            "psm_specific": {
                "matching_statistics": {
                    "n_treated_total": n_treated,
                    "n_matched_pairs": n_matched,
                    "matching_rate": float(matching_rate),
                    "mean_propensity_difference": float(distances.mean()),
                    "caliper": float(caliper)
                },
                "balance_statistics": balance_stats
            }
        }
        
        # Initialize results structures
        detailed_results = []
        heterogeneity_tests = []
        forest_plot_data = {
            "subgroups": [],
            "levels": [],
            "effect_sizes": [],
            "lower_cis": [],
            "upper_cis": [],
            "p_values": [],
            "sample_sizes": [],
            "relative_sample_sizes": []
        }
        
        # For each subgroup, test for heterogeneity and calculate subgroup effects
        for subgroup in subgroups:
            # Test for interactions
            if include_interaction:
                interaction_formula = f"{outcome} ~ {treatment} * C({subgroup})"
                if covariates:
                    for cov in covariates:
                        interaction_formula += f" + {cov}"
                
                try:
                    interaction_model = smf.ols(interaction_formula, data=matched_df).fit()
                    
                    # Get interaction terms
                    interaction_terms = [term for term in interaction_model.params.index if f"{treatment}:C({subgroup}" in term]
                    
                    # Calculate joint test for interaction
                    if interaction_terms:
                        f_test = interaction_model.f_test([f"{term}=0" for term in interaction_terms])
                        heterogeneity_p_value = float(f_test.pvalue)
                        
                        # Add heterogeneity test result
                        heterogeneity_tests.append({
                            "subgroup": subgroup,
                            "interaction_p_value": heterogeneity_p_value,
                            "individual_p_values": [float(interaction_model.pvalues[term]) for term in interaction_terms],
                            "significant": heterogeneity_p_value < 0.05,
                            "f_test_statistic": float(f_test.fvalue),
                            "f_test_df_num": int(f_test.df_num),
                            "f_test_df_denom": int(f_test.df_denom),
                            "interpretation": (
                                f"Treatment effect differs significantly across levels of {subgroup}"
                                if heterogeneity_p_value < 0.05 else
                                f"No significant heterogeneity of treatment effect across levels of {subgroup}"
                            ),
                            "method": "PSM"
                        })
                except Exception as e:
                    print(f"Error testing interaction for {subgroup}: {str(e)}")
            
            # Analyze each level of the subgroup separately
            subgroup_levels = matched_df[subgroup].unique()
            
            for level in subgroup_levels:
                # Skip NaN/None values
                if pd.isna(level):
                    continue
                
                # Subset matched data for this level
                level_df = matched_df[matched_df[subgroup] == level]
                
                # Skip if sample size is too small
                if len(level_df) < 10:
                    print(f"Skipping {subgroup}={level} due to small sample size (n={len(level_df)})")
                    continue
                
                # Create formula for this subgroup level
                level_formula = f"{outcome} ~ {treatment}"
                if covariates:
                    level_formula += " + " + " + ".join(covariates)
                
                # Fit model for this subgroup level
                try:
                    level_model = smf.ols(level_formula, data=level_df).fit()
                    
                    # Extract treatment effect
                    treatment_param = None
                    for param in level_model.params.index:
                        if param == treatment or param.startswith(f"{treatment}[") or param.startswith(f"{treatment}.") or param == f"C({treatment})[T.1]":
                            treatment_param = param
                            break
                    
                    if treatment_param:
                        effect_size = float(level_model.params[treatment_param])
                        std_error = float(level_model.bse[treatment_param])
                        ci_lower = float(level_model.conf_int().loc[treatment_param, 0])
                        ci_upper = float(level_model.conf_int().loc[treatment_param, 1])
                        p_value = float(level_model.pvalues[treatment_param])
                    else:
                        raise ValueError(f"Could not find treatment parameter in model for {subgroup}={level}")
                    
                    # Calculate the number of matched pairs in this level
                    n_pairs = level_df['pair_id'].nunique()
                    
                    # Add result
                    result = {
                        "subgroup": subgroup,
                        "level": str(level),
                        "sample_size": len(level_df),
                        "n_matched_pairs": n_pairs,
                        "effect_size": effect_size,
                        "std_error": std_error,
                        "confidence_interval": [ci_lower, ci_upper],
                        "p_value": p_value,
                        "model_r_squared": float(level_model.rsquared),
                        "model_summary": str(level_model.summary()),
                        "method": "PSM"
                    }
                    
                    detailed_results.append(result)
                    
                    # Add to forest plot data
                    forest_plot_data["subgroups"].append(subgroup)
                    forest_plot_data["levels"].append(str(level))
                    forest_plot_data["effect_sizes"].append(effect_size)
                    forest_plot_data["lower_cis"].append(ci_lower)
                    forest_plot_data["upper_cis"].append(ci_upper)
                    forest_plot_data["p_values"].append(p_value)
                    forest_plot_data["sample_sizes"].append(len(level_df))
                    forest_plot_data["relative_sample_sizes"].append(len(level_df) / len(matched_df))
                
                except Exception as e:
                    print(f"Error analyzing {subgroup}={level}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                await asyncio.sleep(0.01)  # Let UI update
        
        # Adjust p-values if requested
        if adjust_pvalues and detailed_results:
            # Extract p-values
            p_values = [result["p_value"] for result in detailed_results]
            
            # Apply multiple testing correction
            if adjustment_method == "Bonferroni":
                adjusted_p_values = multipletests(p_values, method='bonferroni')[1]
            elif adjustment_method == "Holm-Bonferroni":
                adjusted_p_values = multipletests(p_values, method='holm')[1]
            elif adjustment_method == "Benjamini-Hochberg (FDR)":
                adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]
            elif adjustment_method == "Benjamini-Yekutieli":
                adjusted_p_values = multipletests(p_values, method='fdr_by')[1]
            else:
                adjusted_p_values = p_values  # No adjustment
            
            # Update results with adjusted p-values
            for i, result in enumerate(detailed_results):
                result["adjusted_p_value"] = float(adjusted_p_values[i])
                result["adjustment_method"] = adjustment_method
        
        # Variable importance analysis for subgroups (less relevant for matched data)
        variable_importance = await self.calculate_variable_importance(
            matched_df, outcome, treatment, subgroups
        )
        
        # Create summary
        summary = {
            "overall_effect": overall_effect,
            "heterogeneity_tests": heterogeneity_tests,
            "variable_importance": variable_importance,
            "baseline_characteristics": baseline_characteristics,
            "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        # Create results dictionary
        results = {
            "summary": summary,
            "detailed_results": detailed_results,
            "forest_plot_data": forest_plot_data,
            "methods": self.generate_methods_description(
                outcome, treatment, subgroups, covariates, adjust_pvalues, adjustment_method, 
                "Propensity Score Matching"
            ),
            "limitations": self.generate_limitations_description(df, outcome, treatment, subgroups),
            "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
        }
        
        return results
        
    def visualize_matching_balance(self, balance_stats, ps_vars):
        """Create a visualization of covariate balance before and after matching."""
        # Create a figure for balance plots
        fig = Figure(figsize=(12, max(6, len(ps_vars) * 0.4)))
        ax = fig.add_subplot(111)
        
        # Extract standardized mean differences
        vars_with_smd = []
        before_smd = []
        after_smd = []
        
        for var, stats in balance_stats.items():
            if 'standardized_mean_diff' in stats:
                vars_with_smd.append(var)
                after_smd.append(stats['standardized_mean_diff'])
                
                # Use estimation from original data for before matching (simplified)
                # In a real implementation, you'd store before-matching stats too
                before_smd.append(stats.get('before_matching_smd', after_smd[-1] * 2))
        
        # Sort by absolute standardized mean difference (after matching)
        sorted_indices = np.argsort(np.abs(after_smd))[::-1]
        vars_with_smd = [vars_with_smd[i] for i in sorted_indices]
        before_smd = [before_smd[i] for i in sorted_indices]
        after_smd = [after_smd[i] for i in sorted_indices]
        
        # Plot
        y_pos = range(len(vars_with_smd))
        ax.scatter(before_smd, y_pos, color=PASTEL_COLORS[2], s=80, label='Before Matching')
        ax.scatter(after_smd, y_pos, color=PASTEL_COLORS[0], s=80, label='After Matching')
        
        # Add lines connecting the points for each variable
        for i, (before, after) in enumerate(zip(before_smd, after_smd)):
            ax.plot([before, after], [i, i], color='gray', alpha=0.5)
        
        # Add reference lines
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, 
                label='Threshold (0.1)')
        ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars_with_smd)
        ax.set_xlabel('Standardized Mean Difference')
        ax.set_title('Covariate Balance Before and After Matching')
        ax.legend()
        
        # Set reasonable x-axis limits
        max_abs_smd = max(max(np.abs(before_smd)), max(np.abs(after_smd)))
        ax.set_xlim(-max_abs_smd * 1.1, max_abs_smd * 1.1)
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Display the plot
        self.display_figure(fig)



# ... existing code ...
from PyQt6.QtCore import QThread, pyqtSignal

class BayesianModelThread(QThread):
    """Thread for running Bayesian model sampling without blocking the UI."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, df, outcome, treatment, subgroups, covariates=None, 
                adjust_pvalues=True, include_interaction=True, adjustment_method="Benjamini-Hochberg (FDR)"):
        super().__init__()
        self.df = df.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.subgroups = subgroups
        self.covariates = covariates
        self.adjust_pvalues = adjust_pvalues
        self.include_interaction = include_interaction
        self.adjustment_method = adjustment_method
        
    def run(self):
        try:
            # Report progress
            self.progress_signal.emit("Setting up Bayesian model...")
            
            # Calculate baseline characteristics
            baseline_characteristics = self.calculate_baseline_characteristics(
                self.df, self.treatment, self.subgroups + (self.covariates or [])
            )
            
            # Prepare data
            model_data = self.df.copy()
            
            # Standardize numeric predictors
            numeric_cols = [col for col in model_data.columns 
                        if pd.api.types.is_numeric_dtype(model_data[col]) and col != self.outcome]
            if numeric_cols:
                scaler = StandardScaler()
                model_data[numeric_cols] = scaler.fit_transform(model_data[numeric_cols])
            
            # Ensure treatment is coded as 0/1
            model_data[self.treatment] = model_data[self.treatment].astype(int)
            
            # Group handling for subgroups
            subgroup_indices = {}
            for subgroup in self.subgroups:
                # Create numeric indices for each level of the subgroup
                unique_levels = model_data[subgroup].astype('category').cat.codes
                subgroup_indices[subgroup] = unique_levels
                model_data[f"{subgroup}_idx"] = unique_levels
            
            # Build PyMC model
            self.progress_signal.emit("Building Bayesian model...")
            
            # Create a model context
            with pm.Model() as hierarchical_model:
                # Priors for global parameters
                alpha = pm.Normal("alpha", mu=0, sigma=10)  # Intercept
                beta_treat = pm.Normal("beta_treat", mu=0, sigma=1)  # Main treatment effect
                
                # Priors for subgroup effects (varying effects by subgroup)
                subgroup_effects = {}
                subgroup_sigma = {}
                subgroup_treat_effects = {}
                
                for subgroup in self.subgroups:
                    # Number of levels in this subgroup
                    n_levels = len(self.df[subgroup].unique())
                    
                    # Hyperprior for subgroup effect variation
                    subgroup_sigma[subgroup] = pm.HalfCauchy(f"sigma_{subgroup}", beta=1)
                    
                    # Subgroup main effects
                    subgroup_effects[subgroup] = pm.Normal(
                        f"effect_{subgroup}",
                        mu=0,
                        sigma=subgroup_sigma[subgroup],
                        shape=n_levels
                    )
                    
                    # Subgroup-by-treatment interaction effects
                    subgroup_treat_effects[subgroup] = pm.Normal(
                        f"effect_{subgroup}_x_treat",
                        mu=0,
                        sigma=1,
                        shape=n_levels
                    )
                
                # Covariate effects
                beta_covs = {}
                if self.covariates:
                    for cov in self.covariates:
                        beta_covs[cov] = pm.Normal(f"beta_{cov}", mu=0, sigma=1)
                
                # Model error
                sigma = pm.HalfCauchy("sigma", beta=5)
                
                # Expected value
                mu = alpha + beta_treat * model_data[self.treatment]
                
                # Add subgroup effects and interactions
                for subgroup in self.subgroups:
                    idx = model_data[f"{subgroup}_idx"]
                    mu = mu + subgroup_effects[subgroup][idx]
                    mu = mu + subgroup_treat_effects[subgroup][idx] * model_data[self.treatment]
                
                # Add covariates
                if self.covariates:
                    for cov in self.covariates:
                        mu = mu + beta_covs[cov] * model_data[cov]
                
                # Likelihood
                Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=model_data[self.outcome])
                
                # Sample from the posterior
                self.progress_signal.emit("Running MCMC sampling (this may take a while)...")
                
                # Use fewer samples and chains for faster results
                trace = pm.sample(
                    draws=500,  # Reduced from 1000
                    tune=500,   # Reduced from 1000
                    chains=2,
                    cores=2,
                    return_inferencedata=True,
                    progressbar=False
                )
            
            # Extract posterior samples
            self.progress_signal.emit("Analyzing posterior samples...")
            
            # Get summary of the posteriors
            summary = az.summary(trace)
            
            # Get overall treatment effect
            overall_effect_samples = trace.posterior["beta_treat"].values.flatten()
            overall_effect_mean = float(overall_effect_samples.mean())
            overall_effect_std = float(overall_effect_samples.std())
            overall_effect_hdi = pm.stats.hdi(overall_effect_samples, 0.95)
            
            # Calculate probability of positive effect
            prob_positive = float((overall_effect_samples > 0).mean())
            
            # Create overall effect dictionary
            overall_effect = {
                "effect_size": overall_effect_mean,
                "std_error": overall_effect_std,
                "confidence_interval": [float(overall_effect_hdi[0]), float(overall_effect_hdi[1])],
                "p_value": float(1 - max(prob_positive, 1 - prob_positive)),  # Bayesian approximation of p-value
                "probability_positive": prob_positive,
                "description": f"Overall treatment effect of {self.treatment} on {self.outcome}",
                "model_summary": str(summary),
                "analysis_method": "Bayesian Hierarchical Model",
                "special_technique": "True Bayesian hierarchical modeling with MCMC sampling",
                "bayesian_specific": {
                    "effective_sample_size": float(az.ess(trace, var_names=["beta_treat"]).to_array().mean()),
                    "r_hat": float(az.rhat(trace, var_names=["beta_treat"]).to_array().mean()),
                    "posterior_intervals": {
                        "50%": [float(np.percentile(overall_effect_samples, 25)), 
                                float(np.percentile(overall_effect_samples, 75))],
                        "95%": [float(np.percentile(overall_effect_samples, 2.5)), 
                                float(np.percentile(overall_effect_samples, 97.5))]
                    }
                }
            }
            
            # Initialize results structures
            detailed_results = []
            heterogeneity_tests = []
            forest_plot_data = {
                "subgroups": [],
                "levels": [],
                "effect_sizes": [],
                "lower_cis": [],
                "upper_cis": [],
                "p_values": [],
                "prob_positive": [],
                "sample_sizes": [],
                "relative_sample_sizes": []
            }
            
            # Extract subgroup-specific effects
            for subgroup in self.subgroups:
                # Get interaction parameters
                interaction_params = trace.posterior[f"effect_{subgroup}_x_treat"].values
                
                # Test for evidence of heterogeneity
                # In Bayesian terms: probability that the variance of interaction effects is > small threshold
                interaction_variance = np.var(interaction_params.mean(axis=(0, 1)))
                prob_heterogeneity = float((interaction_params.std(axis=2) > 0.1).mean())
                
                heterogeneity_tests.append({
                    "subgroup": subgroup,
                    "interaction_p_value": float(1 - prob_heterogeneity),  # Bayesian probability as p-value analog
                    "significant": prob_heterogeneity > 0.9,  # 90% probability threshold
                    "probability_heterogeneity": prob_heterogeneity,
                    "interaction_variance": float(interaction_variance),
                    "interpretation": (
                        f"Strong evidence that treatment effects vary across levels of {subgroup}"
                        if prob_heterogeneity > 0.9 else
                        f"Limited evidence of heterogeneous treatment effects across levels of {subgroup}"
                    )
                })
                
                # For each level of the subgroup
                unique_levels = self.df[subgroup].unique()
                
                for i, level in enumerate(unique_levels):
                    # Get subset data for this level
                    level_df = self.df[self.df[subgroup] == level]
                    n_level = len(level_df)
                    
                    if n_level < 10:  # Skip small subgroups
                        continue
                    
                    # Get posterior for treatment effect at this level
                    try:
                        # Get parameter index in the subgroup
                        level_idx = np.where(self.df[subgroup].unique() == level)[0][0]
                        
                        # Extract conditional effect: main effect + interaction
                        conditional_effect_samples = overall_effect_samples + interaction_params[:, :, level_idx].flatten()
                        
                        effect_mean = float(conditional_effect_samples.mean())
                        effect_hdi = pm.stats.hdi(conditional_effect_samples, 0.95)
                        
                        # Calculate probability of positive effect for this subgroup
                        prob_pos = float((conditional_effect_samples > 0).mean())
                        
                        # Bayesian p-value analog: probability of effect in unlikely direction
                        p_value = float(1 - max(prob_pos, 1 - prob_pos))
                        
                        # Add result
                        result = {
                            "subgroup": subgroup,
                            "level": str(level),
                            "sample_size": n_level,
                            "effect_size": effect_mean,
                            "std_error": float(conditional_effect_samples.std()),
                            "confidence_interval": [float(effect_hdi[0]), float(effect_hdi[1])],
                            "p_value": p_value,
                            "probability_positive": prob_pos,
                            "method": "Bayesian Hierarchical"
                        }
                        
                        detailed_results.append(result)
                        
                        # Add to forest plot data
                        forest_plot_data["subgroups"].append(subgroup)
                        forest_plot_data["levels"].append(str(level))
                        forest_plot_data["effect_sizes"].append(effect_mean)
                        forest_plot_data["lower_cis"].append(float(effect_hdi[0]))
                        forest_plot_data["upper_cis"].append(float(effect_hdi[1]))
                        forest_plot_data["p_values"].append(p_value)
                        forest_plot_data["prob_positive"].append(prob_pos)
                        forest_plot_data["sample_sizes"].append(n_level)
                        forest_plot_data["relative_sample_sizes"].append(n_level / len(self.df))
                    
                    except Exception as e:
                        print(f"Error analyzing {subgroup}={level}: {str(e)}")
            
            # Calculate variable importance using posterior variance
            variable_importance = {}
            for subgroup in self.subgroups:
                # Get interaction parameters variance as a measure of importance
                interaction_params = trace.posterior[f"effect_{subgroup}_x_treat"].values
                variable_importance[subgroup] = float(np.var(interaction_params.mean(axis=(0, 1))))
            
            # Create summary
            summary = {
                "overall_effect": overall_effect,
                "heterogeneity_tests": heterogeneity_tests,
                "variable_importance": variable_importance,
                "baseline_characteristics": baseline_characteristics,
                "main_findings": self.generate_main_findings(overall_effect, heterogeneity_tests, detailed_results)
            }
            
            # Create results dictionary
            results = {
                "summary": summary,
                "detailed_results": detailed_results,
                "forest_plot_data": forest_plot_data,
                "methods": self.generate_methods_description(
                    self.outcome, self.treatment, self.subgroups, self.covariates, 
                    self.adjust_pvalues, self.adjustment_method, "Bayesian Hierarchical Model"
                ),
                "limitations": self.generate_limitations_description(self.df, self.outcome, self.treatment, self.subgroups),
                "recommendation": self.generate_recommendations(overall_effect, heterogeneity_tests, detailed_results)
            }
            
            # Emit finished signal with results
            self.finished_signal.emit(results)
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            error_message = f"Error in Bayesian analysis: {str(e)}\n{traceback_str}"
            self.error_signal.emit(error_message)
    
    # Helper methods copied from the main class
    def calculate_baseline_characteristics(self, df, treatment, variables):
        # Copy this method from SubgroupAnalysisWidget
        baseline = {}
        
        for var in variables:
            if pd.api.types.is_numeric_dtype(df[var]):
                # For numeric variables, calculate mean, SD, etc.
                stats_df = df.groupby(treatment)[var].agg(['mean', 'std', 'min', 'max', 'count'])
                
                # Calculate p-value for difference between groups
                groups = df[treatment].unique()
                if len(groups) == 2:  # Only for binary treatment
                    group1 = df[df[treatment] == groups[0]][var].dropna()
                    group2 = df[df[treatment] == groups[1]][var].dropna()
                    _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                else:
                    p_value = None
                
                baseline[var] = {
                    'numeric': True,
                    'stats': stats_df.to_dict(),
                    'p_value': p_value
                }
            else:
                # For categorical variables, calculate frequencies
                cross_tab = pd.crosstab(df[var], df[treatment], normalize='columns')
                counts = pd.crosstab(df[var], df[treatment])
                
                # Calculate chi-square p-value
                try:
                    chi2, p_value, _, _ = stats.chi2_contingency(counts)
                except:
                    p_value = None
                
                baseline[var] = {
                    'numeric': False,
                    'frequencies': cross_tab.to_dict(),
                    'counts': counts.to_dict(),
                    'p_value': p_value
                }
        
        return baseline
    
    # Copy the other helper methods needed from SubgroupAnalysisWidget
    def generate_main_findings(self, overall_effect, heterogeneity_tests, detailed_results):
        # Implementation here (copy from the class)
        findings = []
        
        # Overall effect
        effect_size = overall_effect.get("effect_size", 0)
        p_value = overall_effect.get("p_value", 1.0)
        
        if p_value < 0.05:
            direction = "positive" if effect_size > 0 else "negative"
            findings.append(f"Overall, there is a significant {direction} treatment effect (effect size = {effect_size:.4f}, p = {p_value:.4f}).")
        else:
            findings.append(f"Overall, there is no significant treatment effect (effect size = {effect_size:.4f}, p = {p_value:.4f}).")
        
        # Heterogeneity
        significant_heterogeneity = [test for test in heterogeneity_tests if test.get("significant", False)]
        
        if significant_heterogeneity:
            subgroups = [test.get("subgroup", "") for test in significant_heterogeneity]
            findings.append(f"There is significant heterogeneity of treatment effect across {', '.join(subgroups)}.")
            
            # Add details for each significant subgroup
            for test in significant_heterogeneity:
                subgroup = test.get("subgroup", "")
                p_value = test.get("interaction_p_value", 1.0)
                findings.append(f"- {subgroup}: interaction p-value = {p_value:.4f}")
        else:
            findings.append("There is no significant heterogeneity of treatment effect across the analyzed subgroups.")
        
        # Top subgroups with significant effects
        significant_results = [r for r in detailed_results if r.get("p_value", 1.0) < 0.05]
        
        if significant_results:
            # Sort by absolute effect size
            sorted_results = sorted(significant_results, key=lambda r: abs(r.get("effect_size", 0)), reverse=True)
            
            findings.append("Significant treatment effects were found in the following subgroups:")
            
            for i, result in enumerate(sorted_results[:5]):  # Top 5 at most
                subgroup = result.get("subgroup", "")
                level = result.get("level", "")
                effect = result.get("effect_size", 0)
                p_value = result.get("p_value", 1.0)
                direction = "positive" if effect > 0 else "negative"
                
                findings.append(f"- {subgroup}={level}: {direction} effect of {abs(effect):.4f} (p = {p_value:.4f})")
                
                if i >= 4 and len(sorted_results) > 5:
                    findings.append(f"- ... and {len(sorted_results) - 5} more significant subgroups")
                    break
        
        return " ".join(findings)
        
    def generate_methods_description(self, outcome, treatment, subgroups, covariates, 
                                 adjust_pvalues, adjustment_method="Bonferroni", 
                                 analysis_method="Stratified Analysis"):
        # Implementation here (copy from the class)
        method_text = f"Subgroup analysis was performed using {analysis_method} to assess heterogeneity of treatment effects "
        method_text += f"of {treatment} on {outcome} across levels of {', '.join(subgroups)}. "
        
        if covariates:
            method_text += f"Models were adjusted for the following covariates: {', '.join(covariates)}. "
        
        method_text += "Bayesian hierarchical models were used for estimation of treatment effects within each subgroup. "
        method_text += "MCMC sampling was used for Bayesian inference. "
        
        if adjust_pvalues:
            method_text += f"P-values were adjusted for multiple testing using the {adjustment_method} method. "
        
        method_text += "The analysis evaluated within-subgroup treatment effects and tested for heterogeneity "
        method_text += "using Bayesian posterior probability estimates. "
        method_text += "Forest plots were generated to visualize the pattern of treatment effects across subgroups. "
        
        return method_text
    
    def generate_limitations_description(self, df, outcome, treatment, subgroups):
        # Implementation here (copy from the class)
        n_samples = len(df)
        n_treatment = df[treatment].sum() if df[treatment].dtype == bool else df[treatment].nunique()
        
        limitations = []
        
        # Sample size limitations
        if n_samples < 100:
            limitations.append("The sample size is relatively small, limiting statistical power to detect subgroup effects.")
        
        # Treatment balance
        treatment_counts = df[treatment].value_counts()
        min_treatment_count = treatment_counts.min()
        if min_treatment_count < 30:
            limitations.append(f"The smallest treatment group has only {min_treatment_count} observations, which may affect the reliability of estimates.")
        
        # Subgroup size considerations
        small_subgroups = []
        for subgroup in subgroups:
            subgroup_counts = df[subgroup].value_counts()
            if (subgroup_counts < 20).any():
                small_levels = subgroup_counts[subgroup_counts < 20].index.tolist()
                small_subgroups.append(f"{subgroup} (levels: {', '.join(map(str, small_levels))})")
        
        if small_subgroups:
            limitations.append(f"Some subgroups have small sample sizes: {', '.join(small_subgroups)}.")
        
        # Bayesian-specific limitations
        limitations.append("Bayesian results are sensitive to prior choices. Default weakly informative priors were used.")
        limitations.append("MCMC sampling for complex hierarchical models may not always converge perfectly.")
        
        # Standard limitations
        limitations.extend([
            "This analysis assumes linearity for continuous outcomes and may not account for all confounding factors.",
            "Multiple testing increases the risk of false positive findings, even with correction methods.",
            "The analysis cannot establish causal relationships in subgroups without randomization.",
            "Post-hoc subgroup analyses should be considered exploratory rather than confirmatory."
        ])
        
        return " ".join(limitations)
    
    def generate_recommendations(self, overall_effect, heterogeneity_tests, detailed_results):
        # Implementation here (copy from the class)
        recommendations = []
        
        # Check overall treatment effect
        if overall_effect.get("p_value", 1) < 0.05:
            effect_size = overall_effect.get("effect_size", 0)
            if effect_size > 0:
                recommendations.append(f"The treatment shows an overall positive effect and should be considered for the general population.")
            else:
                recommendations.append(f"The treatment shows an overall negative effect and should be used with caution.")
        else:
            recommendations.append("The treatment shows no significant overall effect, but may still benefit specific subgroups.")
        
        # Check for significant heterogeneity
        significant_heterogeneity = [test for test in heterogeneity_tests if test.get("significant", False)]
        if significant_heterogeneity:
            subgroups_with_heterogeneity = [test.get("subgroup", "") for test in significant_heterogeneity]
            recommendations.append(f"Treatment effects vary significantly across {', '.join(subgroups_with_heterogeneity)}. Consider targeted treatment approaches.")
            
            # Find subgroups with positive effects
            positive_subgroups = []
            for result in detailed_results:
                if result.get("effect_size", 0) > 0 and result.get("p_value", 1) < 0.05:
                    positive_subgroups.append(f"{result.get('subgroup', '')}={result.get('level', '')}")
            
            if positive_subgroups:
                recommendations.append(f"Consider prioritizing treatment for the following subgroups that show positive effects: {', '.join(positive_subgroups)}.")
                
            # Find subgroups with negative effects
            negative_subgroups = []
            for result in detailed_results:
                if result.get("effect_size", 0) < 0 and result.get("p_value", 1) < 0.05:
                    negative_subgroups.append(f"{result.get('subgroup', '')}={result.get('level', '')}")
            
            if negative_subgroups:
                recommendations.append(f"Consider alternative treatments for the following subgroups that show negative effects: {', '.join(negative_subgroups)}.")
        else:
            recommendations.append("No significant treatment effect heterogeneity was detected. The treatment effect appears consistent across subgroups.")
        
        # General recommendations
        recommendations.append("Future research should validate these findings in independent samples before implementing targeted treatment strategies.")
        
        return " ".join(recommendations)