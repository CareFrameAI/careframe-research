import json
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QPlainTextEdit, QFormLayout, QSplitter, 
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QTabWidget, QStatusBar, QApplication, QTextBrowser
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QSizePolicy
import re
from qasync import asyncSlot

# Add statistical analysis packages
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

from data.selection.detailed_tests.formatting import fig_to_svg
from llms.client import call_llm_async
from helpers.load_icon import load_bootstrap_icon
from qt_sections.llm_manager import llm_config  # Add this import

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, 
                              QTableWidgetItem, QPushButton, QLabel, QGroupBox, 
                              QSplitter, QTabWidget, QComboBox,
                              QFormLayout, QSizePolicy, QDialog)
from PyQt6.QtCore import Qt, QByteArray
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtSvgWidgets import QSvgWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VariableRole(Enum):
    """Defines the roles that variables can play in mediation analysis."""
    NONE = "none"
    INDEPENDENT = "independent"  # X variable (predictor)
    DEPENDENT = "dependent"      # Y variable (outcome)
    MEDIATOR = "mediator"        # M variable (mediator)
    MODERATOR = "moderator"      # W variable (moderator for moderated mediation)
    COVARIATE = "covariate"      # Control variables


class MediationType(Enum):
    """Types of mediation analysis."""
    SIMPLE = "Simple Mediation"             # X -> M -> Y
    MULTIPLE_MEDIATORS = "Multiple Mediators"  # X -> M1, M2, ... -> Y
    MULTIPLE_PREDICTORS = "Multiple Predictors"  # X1, X2, ... -> M -> Y
    MODERATED = "Moderated Mediation"       # Includes moderator variable(s)
    SERIAL = "Serial Mediation"             # X -> M1 -> M2 -> Y

# Add helper class for mediation analysis results
class MediationResults:
    """Class to store and format mediation analysis results."""
    
    def __init__(self, mediation_type):
        self.mediation_type = mediation_type
        self.models = {}  # Store regression models
        self.paths = {}   # Store path coefficients
        self.effects = {} # Store direct, indirect effects
        self.bootstrap_results = {} # Store bootstrap results
        self.statistics = {} # Store general statistics
        self.significance = {} # Store significance tests
        self.model_summary = "" # Store formatted model summary
    
    def add_model(self, name, model):
        """Add a regression model."""
        self.models[name] = model
    
    def add_path(self, name, value, se=None, p_value=None):
        """Add a path coefficient."""
        self.paths[name] = {"value": value, "se": se, "p_value": p_value}
    
    def add_effect(self, name, value, se=None, p_value=None, ci_lower=None, ci_upper=None):
        """Add an effect."""
        self.effects[name] = {
            "value": value, 
            "se": se, 
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    
    def add_bootstrap_result(self, name, values, ci_lower=None, ci_upper=None):
        """Add bootstrap results."""
        self.bootstrap_results[name] = {
            "values": values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    
    def format_summary(self):
        """Format a complete summary of mediation analysis results."""
        summary = f"# {self.mediation_type.value} Analysis Results\n\n"
        
        # Add regression models summaries
        summary += "## Regression Models\n\n"
        for name, model in self.models.items():
            summary += f"### {name} Model\n"
            summary += f"```\n{model.summary().as_text()}\n```\n\n"
        
        # Add path coefficients
        summary += "## Path Coefficients\n\n"
        summary += "| Path | Coefficient | Std. Error | P-value |\n"
        summary += "|------|-------------|------------|--------|\n"
        for name, path in self.paths.items():
            summary += f"| {name} | {path['value']:.4f} | {path['se']:.4f} | {path['p_value']:.4f} |\n"
        summary += "\n"
        
        # Add effects
        summary += "## Mediation Effects\n\n"
        summary += "| Effect | Estimate | Std. Error | P-value | 95% CI Lower | 95% CI Upper |\n"
        summary += "|--------|----------|------------|---------|--------------|-------------|\n"
        for name, effect in self.effects.items():
            # Handle each value separately with proper formatting
            value = f"{effect['value']:.4f}" if effect['value'] is not None else "N/A"
            se = f"{effect['se']:.4f}" if effect['se'] is not None else "N/A"
            p_value = f"{effect['p_value']:.4f}" if effect['p_value'] is not None else "N/A"
            ci_lower = f"{effect['ci_lower']:.4f}" if effect['ci_lower'] is not None else "N/A"
            ci_upper = f"{effect['ci_upper']:.4f}" if effect['ci_upper'] is not None else "N/A"
            
            summary += f"| {name} | {value} | {se} | {p_value} | {ci_lower} | {ci_upper} |\n"
        summary += "\n"
        
        # Add interpretation
        summary += "## Interpretation\n\n"
        
        # Check if we have both indirect effect and its confidence interval
        if 'indirect_effect' in self.effects and self.effects['indirect_effect'].get('ci_lower') is not None:
            indirect = self.effects['indirect_effect']
            if indirect['ci_lower'] * indirect['ci_upper'] > 0:  # same sign means CI doesn't include 0
                summary += "- **Significant mediation**: The indirect effect confidence interval does not include zero.\n"
            else:
                summary += "- **Non-significant mediation**: The indirect effect confidence interval includes zero.\n"
        
        # Add more interpretation based on specific effects
        for name, effect in self.effects.items():
            if name == 'direct_effect' and effect.get('p_value') is not None:
                p_val = effect['p_value']
                if p_val < 0.05:
                    summary += f"- **Direct effect** is statistically significant (p={p_val:.4f}).\n"
                else:
                    summary += f"- **Direct effect** is not statistically significant (p={p_val:.4f}).\n"
        
        # Add mediation type
        if 'direct_effect' in self.effects and 'indirect_effect' in self.effects:
            direct = self.effects['direct_effect']
            indirect = self.effects['indirect_effect']
            direct_sig = direct.get('p_value', 1) < 0.05
            indirect_sig = indirect.get('ci_lower', 0) * indirect.get('ci_upper', 0) > 0
            
            if direct_sig and indirect_sig:
                summary += "- **Partial mediation**: Both direct and indirect effects are significant.\n"
            elif not direct_sig and indirect_sig:
                summary += "- **Full mediation**: Indirect effect is significant, but direct effect is not.\n"
            elif direct_sig and not indirect_sig:
                summary += "- **No mediation**: Direct effect is significant, but indirect effect is not.\n"
            else:
                summary += "- **No effect**: Neither direct nor indirect effects are significant.\n"
        
        # Store and return the formatted summary
        self.model_summary = summary
        return summary

class MediationAnalysisWidget(QWidget):
    """Widget for performing mediation analysis with direct column selection."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mediation Analysis")
        
        # Internal state
        self.current_dataframe = None
        self.current_name = ""
        self.column_roles = {}  # Maps column names to their roles
        self.selected_mediation_type = MediationType.SIMPLE
        self.last_analysis_result = None
        
        # Color map for different roles
        self.role_colors = {
            VariableRole.NONE: QColor(255, 255, 255),
            VariableRole.INDEPENDENT: QColor(200, 200, 255),  # Light blue (X)
            VariableRole.DEPENDENT: QColor(255, 200, 200),    # Light red (Y)
            VariableRole.MEDIATOR: QColor(200, 255, 200),     # Light green (M)
            VariableRole.MODERATOR: QColor(255, 255, 200),    # Light yellow (W)
            VariableRole.COVARIATE: QColor(255, 200, 255),    # Light purple
        }
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Top section container - as compact as possible
        top_section = QWidget()
        top_section.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(5, 5, 5, 5)
        top_layout.setSpacing(5)
        
        # Header and dataset selection in one row
        header_row = QHBoxLayout()
        
        header_label = QLabel("Select Dataset:")
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
        
        refresh_button = QPushButton("Refresh")
        refresh_button.setFixedWidth(100)
        refresh_button.clicked.connect(self.load_dataset_from_study)
        dataset_layout.addWidget(refresh_button)
        
        header_row.addWidget(dataset_selection)
        header_row.addStretch()
        
        top_layout.addLayout(header_row)
        
        # Set fixed label width to ensure alignment
        label_width = 120
        
        # Create a horizontal layout for variables and mediation model
        variables_mediation_layout = QHBoxLayout()
        variables_mediation_layout.setSpacing(20)  # Add more spacing between the sections
        
        # Variable selection section - using form layouts
        variable_container = QWidget()
        variable_layout = QVBoxLayout(variable_container)
        variable_layout.setContentsMargins(0, 0, 0, 0)
        variable_layout.setSpacing(10)
        
        # Set size policy to allow variable container to expand horizontally
        variable_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Row 1: Independent, Dependent variables
        row1_widget = QWidget()
        row1_layout = QHBoxLayout(row1_widget)
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(20)
        
        # Independent variable selection
        independent_form = QFormLayout()
        independent_form.setContentsMargins(0, 0, 0, 0)
        independent_form.setSpacing(5)
        independent_label = QLabel("Independent (X):")
        independent_label.setMinimumWidth(label_width)
        self.independent_combo = QComboBox()
        self.independent_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.independent_combo.setMinimumWidth(180)
        self.independent_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.INDEPENDENT))
        independent_form.addRow(independent_label, self.independent_combo)
        row1_layout.addLayout(independent_form, 1)  # Add stretch factor
        
        # Dependent variable selection
        dependent_form = QFormLayout()
        dependent_form.setContentsMargins(0, 0, 0, 0)
        dependent_form.setSpacing(5)
        dependent_label = QLabel("Dependent (Y):")
        dependent_label.setMinimumWidth(label_width)
        self.dependent_combo = QComboBox()
        self.dependent_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.dependent_combo.setMinimumWidth(180)
        self.dependent_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.DEPENDENT))
        dependent_form.addRow(dependent_label, self.dependent_combo)
        row1_layout.addLayout(dependent_form, 1)  # Add stretch factor
        
        variable_layout.addWidget(row1_widget)
        
        # Row 2: Mediator, Moderator variables
        row2_widget = QWidget()
        row2_layout = QHBoxLayout(row2_widget)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(20)
        
        # Mediator variable selection
        mediator_form = QFormLayout()
        mediator_form.setContentsMargins(0, 0, 0, 0)
        mediator_form.setSpacing(5)
        mediator_label = QLabel("Mediator (M):")
        mediator_label.setMinimumWidth(label_width)
        self.mediator_combo = QComboBox()
        self.mediator_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mediator_combo.setMinimumWidth(180)
        self.mediator_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.MEDIATOR))
        mediator_form.addRow(mediator_label, self.mediator_combo)
        row2_layout.addLayout(mediator_form, 1)  # Add stretch factor
        
        # Moderator variable selection
        moderator_form = QFormLayout()
        moderator_form.setContentsMargins(0, 0, 0, 0)
        moderator_form.setSpacing(5)
        moderator_label = QLabel("Moderator (W):")
        moderator_label.setMinimumWidth(label_width)
        self.moderator_combo = QComboBox()
        self.moderator_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.moderator_combo.setMinimumWidth(180)
        self.moderator_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.MODERATOR))
        moderator_form.addRow(moderator_label, self.moderator_combo)
        row2_layout.addLayout(moderator_form, 1)  # Add stretch factor
        
        variable_layout.addWidget(row2_widget)
        
        # Row 3: Covariates
        row3_widget = QWidget()
        row3_layout = QHBoxLayout(row3_widget)
        row3_layout.setContentsMargins(0, 0, 0, 0)
        row3_layout.setSpacing(20)
        
        # Covariates selection
        covariate_form = QFormLayout()
        covariate_form.setContentsMargins(0, 0, 0, 0)
        covariate_form.setSpacing(5)
        covariate_label = QLabel("Add Covariate:")
        covariate_label.setMinimumWidth(label_width)
        
        covariate_widget = QWidget()
        covariate_layout = QHBoxLayout(covariate_widget)
        covariate_layout.setContentsMargins(0, 0, 0, 0)
        covariate_layout.setSpacing(5)
        
        self.covariates_combo = QComboBox()
        self.covariates_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.covariates_combo.setMinimumWidth(180)
        covariate_layout.addWidget(self.covariates_combo, 1)
        
        self.add_covariate_button = QPushButton("Add")
        self.add_covariate_button.setIcon(load_bootstrap_icon("plus-circle", size=18))  # Add icon
        self.add_covariate_button.clicked.connect(self.add_covariate)
        covariate_layout.addWidget(self.add_covariate_button)
        
        covariate_form.addRow(covariate_label, covariate_widget)
        row3_layout.addLayout(covariate_form, 1)  # Add stretch factor
        
        # Container for selected covariates
        selected_covariates_form = QFormLayout()
        selected_covariates_form.setContentsMargins(0, 0, 0, 0)
        selected_covariates_form.setSpacing(5)
        selected_covariates_label = QLabel("Selected Covariates:")
        selected_covariates_label.setMinimumWidth(label_width)
        
        self.covariates_list = QPlainTextEdit()
        self.covariates_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.covariates_list.setReadOnly(True)
        self.covariates_list.setMinimumHeight(90)
        self.covariates_list.setMaximumHeight(90)
        self.covariates_list.setPlaceholderText("No covariates selected")
        
        selected_covariates_form.addRow(selected_covariates_label, self.covariates_list)
        row3_layout.addLayout(selected_covariates_form, 1)  # Add stretch factor
        
        variable_layout.addWidget(row3_widget)
        
        # Clear all button in a separate row
        clear_row = QHBoxLayout()
        clear_button = QPushButton("Clear All Assignments")
        clear_button.setIcon(load_bootstrap_icon("trash", size=18))  # Add icon
        clear_button.clicked.connect(self.clear_all_assignments)
        clear_row.addWidget(clear_button)
        clear_row.addStretch()
        variable_layout.addLayout(clear_row)
        
        # Add variable container to the horizontal layout
        variables_mediation_layout.addWidget(variable_container, 3)  # Give the variable section more space
        
        # Mediation type selection
        mediation_group = QGroupBox("Mediation Model")
        mediation_layout = QVBoxLayout(mediation_group)
        mediation_layout.setContentsMargins(10, 15, 10, 15)  # Add more padding
        mediation_layout.setSpacing(15)  # Increase spacing for readability
        
        # Set size policy to allow mediation group to expand horizontally
        mediation_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Mediation type selection in a row
        mediation_row = QHBoxLayout()
        type_label = QLabel("Select Mediation Type:")
        type_label.setMinimumWidth(label_width)
        
        self.mediation_type_combo = QComboBox()
        self.mediation_type_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.mediation_type_combo.setMinimumWidth(250)
        for mediation_type in MediationType:
            self.mediation_type_combo.addItem(mediation_type.value, mediation_type)
        
        mediation_help_button = QPushButton("Model Info")
        mediation_help_button.setIcon(load_bootstrap_icon("info-circle", size=18))  # Add icon
        mediation_help_button.clicked.connect(self.show_mediation_info)
        
        mediation_row.addWidget(type_label)
        mediation_row.addWidget(self.mediation_type_combo, 1)
        mediation_row.addWidget(mediation_help_button)
        
        mediation_layout.addLayout(mediation_row)
        
        # Description label for the selected mediation type
        self.mediation_description = QLabel()
        self.mediation_description.setWordWrap(True)
        self.mediation_description.setStyleSheet("font-style: italic;")
        self.mediation_description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mediation_layout.addWidget(self.mediation_description)
        
        # Connect the combobox to update the description
        self.mediation_type_combo.currentIndexChanged.connect(self.update_mediation_description)
        
        # Add mediation group to the horizontal layout
        variables_mediation_layout.addWidget(mediation_group, 2)  # Give appropriate space
        
        top_layout.addLayout(variables_mediation_layout)
        
        # Action buttons in one row - arranged in sequence of execution
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(20, 10, 20, 10)  # Add margins around buttons
        buttons_row.setSpacing(30)  # Increase spacing between buttons
        
        # Add a spacer at the beginning
        buttons_row.addStretch(1)
        
        self.auto_assign_button = QPushButton("Build Model (AI)")
        self.auto_assign_button.setIcon(load_bootstrap_icon("search", size=18))
        self.auto_assign_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.auto_assign_button.setMinimumWidth(150)  # Set minimum width for button
        self.auto_assign_button.clicked.connect(self.build_model)
        buttons_row.addWidget(self.auto_assign_button)
        
        buttons_row.addStretch(2)  # Add more spacing between button groups
        
        self.run_analysis_button = QPushButton("Run Mediation Analysis")
        self.run_analysis_button.setIcon(load_bootstrap_icon("play-fill", size=18))
        self.run_analysis_button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.run_analysis_button.setMinimumWidth(200)  # Set minimum width for button
        self.run_analysis_button.clicked.connect(self.run_mediation_analysis)
        buttons_row.addWidget(self.run_analysis_button)
        
        # Add a spacer at the end
        buttons_row.addStretch(1)
        
        top_layout.addLayout(buttons_row)
        
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
        self.data_table.horizontalHeader().setSectionsClickable(True)
        
        dataset_layout.addWidget(self.data_table)
        left_tabs.addTab(dataset_tab, "Dataset")
        
        # Tab 2: Summary Results
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(5, 5, 5, 5)
        
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        left_tabs.addTab(summary_tab, "Summary")
        
        # Tab 3: Detailed Results
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(5, 5, 5, 5)
        
        self.results_text = QPlainTextEdit()
        self.results_text.setReadOnly(True)
        
        details_layout.addWidget(self.results_text)
        left_tabs.addTab(details_tab, "Detailed Results")
        
        # Tab 4: AI Interpretation
        interpretation_tab = QWidget()
        interpretation_layout = QVBoxLayout(interpretation_tab)
        interpretation_layout.setContentsMargins(5, 5, 5, 5)
        
        self.interpretation_text = QTextBrowser()
        self.interpretation_text.setOpenExternalLinks(True)
        self.interpretation_text.setPlaceholderText("Click 'Interpret Results' button to generate AI interpretation")
        
        # Add interpret button
        interpret_button = QPushButton("Interpret Results (AI)")
        interpret_button.setIcon(load_bootstrap_icon("cpu", size=18))  # Update icon
        interpret_button.clicked.connect(self.interpret_results)
        
        interpretation_layout.addWidget(interpret_button)
        interpretation_layout.addWidget(self.interpretation_text)
        
        left_tabs.addTab(interpretation_tab, "AI Interpretation")
        
        # Right side - Visualization
        visualization_group = QGroupBox("Visualization")
        visualization_layout = QVBoxLayout(visualization_group)
        visualization_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create a tab widget for different visualizations
        self.viz_tabs = QTabWidget()
        
        # Placeholder for visualization
        self.visualization_placeholder = QLabel("Mediation path visualization will be shown here after analysis.")
        self.visualization_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visualization_placeholder.setStyleSheet("font-style: italic; color: gray;")
        
        placeholder_tab = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_tab)
        placeholder_layout.addWidget(self.visualization_placeholder)
        self.viz_tabs.addTab(placeholder_tab, "Path Diagram")
        
        visualization_layout.addWidget(self.viz_tabs)
        
        # Add the tabs and visualization to the splitter
        content_splitter.addWidget(left_tabs)
        content_splitter.addWidget(visualization_group)
        
        # Set the sizes for better visibility (60% left, 40% right)
        content_splitter.setSizes([600, 400])
        
        main_layout.addWidget(content_splitter, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize the widget
        self.update_mediation_description()
        
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
        
        # Apply role-based coloring to headers
        for col_idx, col_name in enumerate(df.columns):
            header_item = self.data_table.horizontalHeaderItem(col_idx)
            if header_item:
                role = self.column_roles.get(col_name, VariableRole.NONE)
                header_item.setBackground(self.role_colors[role])
                
                # Make the text bold if the column has a role
                if role != VariableRole.NONE:
                    font = header_item.font()
                    font.setBold(True)
                    header_item.setFont(font)
        
        # Fill the table with data
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 1000:  # Limit to 1000 rows for performance
                break
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.data_table.setItem(i, j, item)
        
        # Resize columns for better display
        self.data_table.resizeColumnsToContents()
    
    def update_mediation_description(self):
        """Update the mediation description based on the selected type."""
        mediation_type = self.mediation_type_combo.currentData()
        if not mediation_type:
            return
        
        descriptions = {
            MediationType.SIMPLE: 
                "Tests how an independent variable (X) affects a dependent variable (Y) through a mediator (M).",
            MediationType.MULTIPLE_MEDIATORS: 
                "Tests mediation through multiple parallel mediators (X → M1, M2, etc. → Y).",
            MediationType.MULTIPLE_PREDICTORS: 
                "Tests multiple independent variables (X1, X2, etc.) affecting Y through a mediator (M).",
            MediationType.MODERATED: 
                "Tests if the mediation effect varies across levels of a moderator variable (W).",
            MediationType.SERIAL: 
                "Tests a causal chain with mediators in sequence (X → M1 → M2 → Y)."
        }
        
        self.mediation_description.setText(descriptions.get(mediation_type, ""))
        
        # Update required fields highlighting based on mediation type
        self.update_required_fields(mediation_type)
    
    def update_required_fields(self, mediation_type):
        """Update which fields are required based on the selected mediation type."""
        # Reset all styles
        self.independent_combo.setStyleSheet("")
        self.dependent_combo.setStyleSheet("")
        self.mediator_combo.setStyleSheet("")
        self.moderator_combo.setStyleSheet("")
        
        # Set tooltips
        self.independent_combo.setToolTip("Required")
        self.dependent_combo.setToolTip("Required")
        self.mediator_combo.setToolTip("Required")
        
        # Bold for required fields
        self.independent_combo.setStyleSheet("font-weight: bold;")
        self.dependent_combo.setStyleSheet("font-weight: bold;")
        self.mediator_combo.setStyleSheet("font-weight: bold;")
        
        # Special case for moderated mediation
        if mediation_type == MediationType.MODERATED:
            self.moderator_combo.setStyleSheet("font-weight: bold;")
            self.moderator_combo.setToolTip("Required for moderated mediation")
        else:
            self.moderator_combo.setStyleSheet("color: gray;")
            self.moderator_combo.setToolTip("Not used in this mediation model")
    
    def show_mediation_info(self):
        """Show detailed information about the selected mediation type."""
        mediation_type = self.mediation_type_combo.currentData()
        if not mediation_type:
            return
        
        # Common info structure
        info = {
            MediationType.SIMPLE: {
                "title": "Simple Mediation",
                "description": "Tests how an independent variable (X) affects a dependent variable (Y) through a mediator (M).",
                "model": "Y = c'X + bM + e\nM = aX + e",
                "effects": [
                    "Direct effect: The effect of X on Y controlling for M (c')",
                    "Indirect effect: The product of paths a and b (a×b)",
                    "Total effect: The sum of direct and indirect effects (c = c' + ab)"
                ],
                "requirements": ["One independent variable (X)", "One dependent variable (Y)", "One mediator (M)"],
                "diagram": "X → M → Y (X also directly affects Y)"
            },
            MediationType.MULTIPLE_MEDIATORS: {
                "title": "Multiple Mediators",
                "description": "Tests mediation through multiple parallel mediators.",
                "model": "Y = c'X + b₁M₁ + b₂M₂ + ... + e\nM₁ = a₁X + e\nM₂ = a₂X + e\n...",
                "effects": [
                    "Direct effect: The effect of X on Y controlling for all mediators (c')",
                    "Specific indirect effects: The product of paths through each mediator (a₁×b₁, a₂×b₂, etc.)",
                    "Total indirect effect: The sum of all specific indirect effects",
                    "Total effect: c = c' + Σ(aᵢ×bᵢ)"
                ],
                "requirements": ["One independent variable (X)", "One dependent variable (Y)", "Multiple mediators (M₁, M₂, etc.)"],
                "diagram": "X → M₁ → Y\nX → M₂ → Y\n(X also directly affects Y)"
            },
            MediationType.MULTIPLE_PREDICTORS: {
                "title": "Multiple Predictors",
                "description": "Tests multiple independent variables affecting Y through a mediator.",
                "model": "Y = c'₁X₁ + c'₂X₂ + ... + bM + e\nM = a₁X₁ + a₂X₂ + ... + e",
                "effects": [
                    "Direct effects: The effects of each X on Y controlling for M (c'₁, c'₂, etc.)",
                    "Indirect effects: The products of paths for each predictor (a₁×b, a₂×b, etc.)",
                    "Total effects: c₁ = c'₁ + a₁b, c₂ = c'₂ + a₂b, etc."
                ],
                "requirements": ["Multiple independent variables (X₁, X₂, etc.)", "One dependent variable (Y)", "One mediator (M)"],
                "diagram": "X₁ → M → Y\nX₂ → M → Y\n(Each X also directly affects Y)"
            },
            MediationType.MODERATED: {
                "title": "Moderated Mediation",
                "description": "Tests if the mediation effect varies across levels of a moderator variable.",
                "model": "Y = c'X + bM + e\nM = a₀ + a₁X + a₂W + a₃XW + e",
                "effects": [
                    "Conditional indirect effect: The indirect effect at different levels of the moderator",
                    "Index of moderated mediation: Tests if the indirect effect differs significantly across moderator levels"
                ],
                "requirements": ["One independent variable (X)", "One dependent variable (Y)", "One mediator (M)", "One moderator (W)"],
                "diagram": "X → M → Y, with W moderating the X → M path"
            },
            MediationType.SERIAL: {
                "title": "Serial Mediation",
                "description": "Tests a causal chain with mediators in sequence.",
                "model": "Y = c'X + b₁M₁ + b₂M₂ + e\nM₂ = a₂X + d₂₁M₁ + e\nM₁ = a₁X + e",
                "effects": [
                    "Direct effect: The effect of X on Y controlling for all mediators (c')",
                    "Specific indirect effects through M₁ only: a₁×b₁",
                    "Specific indirect effects through M₂ only: a₂×b₂",
                    "Specific indirect effect through both M₁ and M₂ in sequence: a₁×d₂₁×b₂",
                    "Total indirect effect: Sum of all specific indirect effects",
                    "Total effect: c = c' + a₁b₁ + a₂b₂ + a₁d₂₁b₂"
                ],
                "requirements": ["One independent variable (X)", "One dependent variable (Y)", "Multiple mediators in sequence (M₁, M₂, etc.)"],
                "diagram": "X → M₁ → M₂ → Y (X also directly affects M₂ and Y, M₁ affects Y)"
            }
        }
        
        model_info = info.get(mediation_type, {})
        if not model_info:
            return
        
        # Format for QMessageBox
        message = f"<h3>{model_info.get('title', '')}</h3>"
        message += f"<p><b>Description:</b> {model_info.get('description', '')}</p>"
        
        message += "<p><b>Model:</b></p><pre>" + model_info.get('model', '') + "</pre>"
        
        message += "<p><b>Effects:</b></p><ul>"
        for effect in model_info.get('effects', []):
            message += f"<li>{effect}</li>"
        message += "</ul>"
        
        message += "<p><b>Requirements:</b></p><ul>"
        for req in model_info.get('requirements', []):
            message += f"<li>{req}</li>"
        message += "</ul>"
        
        message += f"<p><b>Diagram:</b></p><pre>{model_info.get('diagram', '')}</pre>"
        
        # Show the message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(f"Mediation Model: {model_info.get('title', '')}")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()
    
    @asyncSlot()
    async def build_model(self):
        """Use AI to build a mediation model by assigning variable roles."""
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
            # Add more statistical info for better variable role identification
            stats = {
                "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "std": float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "unique_count": int(df[col].nunique()),
                "unique_ratio": float(df[col].nunique() / len(df)) if len(df) > 0 else 0
            }
            sample_values = df[col].dropna().head(3).tolist()
            
            columns_info.append({
                "name": col,
                "data_type": data_type,
                "unique_values": unique_values,
                "is_numeric": is_numeric,
                "statistics": stats,
                "sample_values": sample_values
            })
        
        # Show loading message
        QMessageBox.information(self, "Processing", "Analyzing dataset for mediation analysis with AI. Please wait...")
        
        try:
            # Create a prompt for the LLM
            prompt = f"""
            I need to analyze this dataset to identify variables for a mediation analysis model.

            Dataset: {self.current_name}
            Sample Size: {len(df)} observations

            Here are the columns in my dataset with their properties:
            {json.dumps(columns_info, indent=2)}

            Here's a sample of the data:
            {df.head(5).to_string()}

            For mediation analysis, I need to identify:
            1. INDEPENDENT (X): Predictor/independent variable that may affect the outcome directly and indirectly
            2. MEDIATOR (M): Variable that mediates the relationship between X and Y
            3. DEPENDENT (Y): Outcome/response variable 
            4. MODERATOR (W): Variable that might modify the relationship between X and M or M and Y (if applicable)
            5. COVARIATE: Control variables that should be included in the analysis
            6. NONE: Variables that should not be used in the analysis

            Based on variable names, correlations, and theoretical relationships, please identify the most likely role for each variable.

            Also recommend the most appropriate mediation model from these options:
            - Simple Mediation: X affects Y through a single mediator M
            - Multiple Mediators: X affects Y through multiple parallel mediators
            - Multiple Predictors: Multiple X variables affect Y through a mediator M
            - Moderated Mediation: A moderator W affects the strength of the mediation relationship
            - Serial Mediation: X affects Y through a chain of mediators (M1 → M2 → etc.)

            Return your analysis as a JSON with:
            {{
              "column_roles": {{
                "column_name1": "INDEPENDENT",
                "column_name2": "MEDIATOR",
                "column_name3": "DEPENDENT", 
                "etc": "..."
              }},
              "recommended_model": "Simple Mediation",
              "explanation": "brief explanation of your reasoning"
            }}
            """

            # Call LLM to identify variable roles - add model parameter
            response = await call_llm_async(prompt, model=llm_config.default_text_model)
            
            # Parse the JSON response
            json_match = re.search(r'({.*})', response, re.DOTALL)
            if not json_match:
                QMessageBox.warning(self, "Error", "Could not parse AI response for variable roles")
                return
                
            result = json.loads(json_match.group(1))
            column_roles = result.get("column_roles", {})
            recommended_model = result.get("recommended_model", "Simple Mediation")
            explanation = result.get("explanation", "")
            
            # Combine results for display to user
            message = "AI recommended the following mediation model:\n\n"
            message += f"Model Type: {recommended_model}\n\n"
            message += "Variable Roles:\n"
            for col, role in column_roles.items():
                message += f"{col}: {role}\n"
            
            message += f"\nExplanation:\n{explanation}\n\nApply this model?"
            
            if QMessageBox.question(
                self, 
                "Apply AI Model?", 
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                # Apply the recommended model type
                for i in range(self.mediation_type_combo.count()):
                    model_type = self.mediation_type_combo.itemData(i)
                    if model_type and model_type.value == recommended_model:
                        self.mediation_type_combo.setCurrentIndex(i)
                        break
                
                # Reset all roles first
                for col in self.column_roles:
                    self.column_roles[col] = VariableRole.NONE
                
                # Apply recommended roles
                for col, role_str in column_roles.items():
                    if col in df.columns:
                        try:
                            # Convert string role to enum
                            role = VariableRole[role_str]
                            self.column_roles[col] = role
                        except (KeyError, ValueError):
                            # If the role string doesn't match an enum value, set to NONE
                            self.column_roles[col] = VariableRole.NONE
                
                # Update the UI
                self.populate_variable_dropdowns()
                
                self.status_bar.showMessage(f"Model built: {recommended_model} with {len(column_roles)} assigned variables")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI model building failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def manual_role_changed(self, role):
        """Handle selection changes in the manual variable role dropdowns."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return
        
        combo_map = {
            VariableRole.INDEPENDENT: self.independent_combo,
            VariableRole.DEPENDENT: self.dependent_combo,
            VariableRole.MEDIATOR: self.mediator_combo,
            VariableRole.MODERATOR: self.moderator_combo
        }
        
        combo = combo_map.get(role)
        if not combo:
            return
            
        new_var = combo.currentText()
        if new_var and new_var != "Select...":
            # Clear any existing variable with this role
            for col, existing_role in self.column_roles.items():
                if existing_role == role:
                    self.column_roles[col] = VariableRole.NONE
            # Set the new variable with this role
            self.column_roles[new_var] = role
    
    def add_covariate(self):
        """Add a covariate to the list of covariates."""
        new_cov = self.covariates_combo.currentText()
        if new_cov and new_cov != "Select...":
            # Set the new variable as a covariate
            self.column_roles[new_cov] = VariableRole.COVARIATE
            # Update the UI
            self.update_covariates_display()
    
    def update_covariates_display(self):
        """Update the display of selected covariates."""
        covariates = [col for col, role in self.column_roles.items() 
                     if role == VariableRole.COVARIATE]
        
        if covariates:
            self.covariates_list.setPlainText("\n".join(covariates))
        else:
            self.covariates_list.setPlainText("")
    
    def clear_all_assignments(self):
        """Clear all variable role assignments."""
        if QMessageBox.question(
            self, 
            "Clear Assignments", 
            "Are you sure you want to clear all variable assignments?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            # Reset all roles
            for col in self.column_roles:
                self.column_roles[col] = VariableRole.NONE
            
            # Reset all dropdowns
            self.populate_variable_dropdowns()
            
            # Update UI
            self.update_covariates_display()
    
    def populate_variable_dropdowns(self):
        """Populate the variable selection dropdowns based on current column roles."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return
        
        # Temporarily block signals to prevent recursive calls
        self.independent_combo.blockSignals(True)
        self.dependent_combo.blockSignals(True)
        self.mediator_combo.blockSignals(True)
        self.moderator_combo.blockSignals(True)
        self.covariates_combo.blockSignals(True)
        
        # Clear and repopulate all dropdowns
        self.independent_combo.clear()
        self.dependent_combo.clear()
        self.mediator_combo.clear()
        self.moderator_combo.clear()
        self.covariates_combo.clear()
        
        # Add placeholder items
        self.independent_combo.addItem("Select...")
        self.dependent_combo.addItem("Select...")
        self.mediator_combo.addItem("Select...")
        self.moderator_combo.addItem("Select...")
        self.covariates_combo.addItem("Select...")
        
        # Add all columns
        for col in self.current_dataframe.columns:
            self.independent_combo.addItem(col)
            self.dependent_combo.addItem(col)
            self.mediator_combo.addItem(col)
            self.moderator_combo.addItem(col)
            self.covariates_combo.addItem(col)
        
        # Set dropdowns based on current role assignments
        independent = next((col for col, role in self.column_roles.items() if role == VariableRole.INDEPENDENT), None)
        dependent = next((col for col, role in self.column_roles.items() if role == VariableRole.DEPENDENT), None)
        mediator = next((col for col, role in self.column_roles.items() if role == VariableRole.MEDIATOR), None)
        moderator = next((col for col, role in self.column_roles.items() if role == VariableRole.MODERATOR), None)
        
        if independent:
            self.independent_combo.setCurrentText(independent)
        if dependent:
            self.dependent_combo.setCurrentText(dependent)
        if mediator:
            self.mediator_combo.setCurrentText(mediator)
        if moderator:
            self.moderator_combo.setCurrentText(moderator)
        
        # Re-enable signals
        self.independent_combo.blockSignals(False)
        self.dependent_combo.blockSignals(False)
        self.mediator_combo.blockSignals(False)
        self.moderator_combo.blockSignals(False)
        self.covariates_combo.blockSignals(False)
        
        # Update covariates display
        self.update_covariates_display()
        
    def perform_simple_mediation(self, df, x, m, y, covariates=None):
        """Perform simple mediation analysis (X → M → Y)."""
        results = MediationResults(MediationType.SIMPLE)
        
        # Create formula strings
        x_term = x
        m_term = m
        y_term = y
        cov_terms = ""
        
        if covariates and len(covariates) > 0:
            cov_terms = " + " + " + ".join(covariates)
        
        # Path a: X → M
        a_formula = f"{m_term} ~ {x_term}{cov_terms}"
        model_a = smf.ols(a_formula, data=df).fit()
        results.add_model("a (X → M)", model_a)
        
        # Extract path a coefficient
        a_coef = model_a.params[x_term]
        a_se = model_a.bse[x_term]
        a_p = model_a.pvalues[x_term]
        results.add_path("a (X → M)", a_coef, a_se, a_p)
        
        # Path b and c': Y ~ X + M
        b_formula = f"{y_term} ~ {x_term} + {m_term}{cov_terms}"
        model_b = smf.ols(b_formula, data=df).fit()
        results.add_model("b and c' (Y ~ X + M)", model_b)
        
        # Extract path b coefficient
        b_coef = model_b.params[m_term]
        b_se = model_b.bse[m_term]
        b_p = model_b.pvalues[m_term]
        results.add_path("b (M → Y)", b_coef, b_se, b_p)
        
        # Extract path c' coefficient (direct effect)
        c_prime_coef = model_b.params[x_term]
        c_prime_se = model_b.bse[x_term]
        c_prime_p = model_b.pvalues[x_term]
        results.add_path("c' (X → Y, direct)", c_prime_coef, c_prime_se, c_prime_p)
        
        # Path c: Total effect, Y ~ X
        c_formula = f"{y_term} ~ {x_term}{cov_terms}"
        model_c = smf.ols(c_formula, data=df).fit()
        results.add_model("c (X → Y, total)", model_c)
        
        # Extract path c coefficient (total effect)
        c_coef = model_c.params[x_term]
        c_se = model_c.bse[x_term]
        c_p = model_c.pvalues[x_term]
        results.add_path("c (X → Y, total)", c_coef, c_se, c_p)
        
        # Calculate indirect effect (a*b)
        indirect_effect = a_coef * b_coef
        
        # Calculate standard error for indirect effect using Sobel test
        se_indirect = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
        z_value = indirect_effect / se_indirect
        p_indirect = 2 * (1 - stats.norm.cdf(abs(z_value)))
        
        results.add_effect("indirect_effect", indirect_effect, se_indirect, p_indirect)
        results.add_effect("direct_effect", c_prime_coef, c_prime_se, c_prime_p)
        results.add_effect("total_effect", c_coef, c_se, c_p)
        
        # Bootstrap confidence intervals for indirect effect
        n_bootstrap = 5000
        bootstrap_samples = []
        
        # Bootstrap resampling
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[indices]
            
            # Estimate models on the resampled data
            model_a_boot = smf.ols(a_formula, data=boot_df).fit()
            model_b_boot = smf.ols(b_formula, data=boot_df).fit()
            
            a_boot = model_a_boot.params[x_term]
            b_boot = model_b_boot.params[m_term]
            
            # Calculate indirect effect
            indirect_boot = a_boot * b_boot
            bootstrap_samples.append(indirect_boot)
        
        # Calculate percentile confidence intervals
        bootstrap_samples = np.array(bootstrap_samples)
        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)
        
        # Update the indirect effect with bootstrap CIs
        results.add_effect("indirect_effect", indirect_effect, se_indirect, p_indirect, ci_lower, ci_upper)
        results.add_bootstrap_result("indirect_effect", bootstrap_samples, ci_lower, ci_upper)
        
        return results
            
    def perform_moderated_mediation(self, df, x, m, y, w, covariates=None):
        """Perform moderated mediation analysis."""
        results = MediationResults(MediationType.MODERATED)
        
        x_term = x
        m_term = m
        y_term = y
        w_term = w
        
        cov_terms = ""
        if covariates and len(covariates) > 0:
            cov_terms = " + " + " + ".join(covariates)
        
        # Create a copy of the dataframe to avoid modifying the original
        df_analysis = df.copy()
        df_analysis['XW_interaction'] = df_analysis[x] * df_analysis[w]
        
        # Path a: X → M, moderated by W (includes X*W interaction)
        a_formula = f"{m_term} ~ {x_term} + {w_term} + XW_interaction{cov_terms}"
        model_a = smf.ols(a_formula, data=df_analysis).fit()
        results.add_model("a (X → M, moderated by W)", model_a)
        
        # Extract coefficients for X, W, and X*W
        a_x_coef = model_a.params[x_term]
        a_w_coef = model_a.params[w_term]
        a_xw_coef = model_a.params['XW_interaction']
        
        results.add_path("a_x (X → M)", a_x_coef, model_a.bse[x_term], model_a.pvalues[x_term])
        results.add_path("a_w (W → M)", a_w_coef, model_a.bse[w_term], model_a.pvalues[w_term])
        results.add_path("a_xw (X*W → M)", a_xw_coef, model_a.bse['XW_interaction'], 
                    model_a.pvalues['XW_interaction'])
        
        # Path b and c': Y ~ X + M
        b_formula = f"{y_term} ~ {x_term} + {m_term}{cov_terms}"
        model_b = smf.ols(b_formula, data=df_analysis).fit()
        results.add_model("b and c' (Y ~ X + M)", model_b)
        
        # Extract path b coefficient
        b_coef = model_b.params[m_term]
        b_se = model_b.bse[m_term]
        b_p = model_b.pvalues[m_term]
        results.add_path("b (M → Y)", b_coef, b_se, b_p)
        
        # Extract path c' coefficient (direct effect)
        c_prime_coef = model_b.params[x_term]
        c_prime_se = model_b.bse[x_term]
        c_prime_p = model_b.pvalues[x_term]
        results.add_path("c' (X → Y, direct)", c_prime_coef, c_prime_se, c_prime_p)
        results.add_effect("direct_effect", c_prime_coef, c_prime_se, c_prime_p)
        
        # Calculate conditional indirect effects at different levels of W
        w_mean = df[w].mean()
        w_sd = df[w].std()
        
        w_levels = {
            "mean-1sd": w_mean - w_sd,
            "mean": w_mean,
            "mean+1sd": w_mean + w_sd
        }
        
        bootstrap_conditional = {level: [] for level in w_levels}
        
        # Conditional indirect effects
        for level_name, w_value in w_levels.items():
            # Conditional a path at this level of W
            a_conditional = a_x_coef + a_xw_coef * w_value
            
            # Conditional indirect effect
            indirect_conditional = a_conditional * b_coef
            
            results.add_effect(f"conditional_indirect_at_{level_name}", indirect_conditional)
        
        # Bootstrap for conditional indirect effects
        n_bootstrap = 5000
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[indices]
            boot_df['XW_interaction'] = boot_df[x] * boot_df[w]
            
            # Estimate models on the resampled data
            model_a_boot = smf.ols(a_formula, data=boot_df).fit()
            model_b_boot = smf.ols(b_formula, data=boot_df).fit()
            
            a_x_boot = model_a_boot.params[x_term] 
            a_xw_boot = model_a_boot.params['XW_interaction']
            b_boot = model_b_boot.params[m_term]
            
            # Calculate conditional indirect effects for each W level
            for level_name, w_value in w_levels.items():
                a_conditional = a_x_boot + a_xw_boot * w_value
                indirect_conditional = a_conditional * b_boot
                bootstrap_conditional[level_name].append(indirect_conditional)
        
        # Calculate confidence intervals for conditional indirect effects
        for level_name in w_levels.keys():
            samples = np.array(bootstrap_conditional[level_name])
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            
            effect_key = f"conditional_indirect_at_{level_name}"
            results.add_bootstrap_result(effect_key, samples, ci_lower, ci_upper)
            
            # Update the effect with CIs
            effect = results.effects[effect_key]
            effect['ci_lower'] = ci_lower
            effect['ci_upper'] = ci_upper
        
        # Calculate index of moderated mediation (a_xw * b)
        index_moderated_mediation = a_xw_coef * b_coef
        results.add_effect("index_of_moderated_mediation", index_moderated_mediation)
        
        # Bootstrap for index of moderated mediation
        index_bootstrap = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[indices]
            boot_df['XW_interaction'] = boot_df[x] * boot_df[w]
            
            model_a_boot = smf.ols(a_formula, data=boot_df).fit()
            model_b_boot = smf.ols(b_formula, data=boot_df).fit()
            
            a_xw_boot = model_a_boot.params['XW_interaction']
            b_boot = model_b_boot.params[m_term]
            
            index_boot = a_xw_boot * b_boot
            index_bootstrap.append(index_boot)
        
        # Calculate confidence interval for index of moderated mediation
        index_samples = np.array(index_bootstrap)
        index_ci_lower = np.percentile(index_samples, 2.5)
        index_ci_upper = np.percentile(index_samples, 97.5)
        
        results.add_bootstrap_result("index_of_moderated_mediation", index_samples, 
                                index_ci_lower, index_ci_upper)
        
        # Update the effect with CIs
        results.effects["index_of_moderated_mediation"]["ci_lower"] = index_ci_lower
        results.effects["index_of_moderated_mediation"]["ci_upper"] = index_ci_upper
        
        return results
        
    def perform_multiple_mediators(self, df, x, mediators, y, covariates=None):
        """Perform multiple mediator analysis (X → M1, M2, ... → Y)."""
        results = MediationResults(MediationType.MULTIPLE_MEDIATORS)
        
        # Validate we have at least one mediator
        if not mediators or len(mediators) == 0:
            raise ValueError("Multiple mediator analysis requires at least one mediator variable.")
        
        x_term = x
        y_term = y
        cov_terms = ""
        
        if covariates and len(covariates) > 0:
            cov_terms = " + " + " + ".join(covariates)
        
        # Total effect: Y ~ X
        c_formula = f"{y_term} ~ {x_term}{cov_terms}"
        model_c = smf.ols(c_formula, data=df).fit()
        results.add_model("c (X → Y, total)", model_c)
        
        # Extract path c coefficient (total effect)
        c_coef = model_c.params[x_term]
        c_se = model_c.bse[x_term]
        c_p = model_c.pvalues[x_term]
        results.add_path("c (X → Y, total)", c_coef, c_se, c_p)
        results.add_effect("total_effect", c_coef, c_se, c_p)
        
        # Paths a: X → M for each mediator
        a_paths = {}
        for m in mediators:
            a_formula = f"{m} ~ {x_term}{cov_terms}"
            model_a = smf.ols(a_formula, data=df).fit()
            results.add_model(f"a (X → {m})", model_a)
            
            a_coef = model_a.params[x_term]
            a_se = model_a.bse[x_term]
            a_p = model_a.pvalues[x_term]
            
            path_key = f"a (X → {m})"
            results.add_path(path_key, a_coef, a_se, a_p)
            a_paths[m] = (a_coef, a_se, a_p)
        
        # Paths b and c': Y ~ X + M1 + M2 + ...
        mediators_term = " + ".join(mediators)
        b_formula = f"{y_term} ~ {x_term} + {mediators_term}{cov_terms}"
        model_b = smf.ols(b_formula, data=df).fit()
        results.add_model("b and c' (Y ~ X + M1 + M2 + ...)", model_b)
        
        # Direct effect
        c_prime_coef = model_b.params[x_term]
        c_prime_se = model_b.bse[x_term]
        c_prime_p = model_b.pvalues[x_term]
        results.add_path("c' (X → Y, direct)", c_prime_coef, c_prime_se, c_prime_p)
        results.add_effect("direct_effect", c_prime_coef, c_prime_se, c_prime_p)
        
        # Extract b paths for each mediator
        b_paths = {}
        for m in mediators:
            b_coef = model_b.params[m]
            b_se = model_b.bse[m]
            b_p = model_b.pvalues[m]
            
            path_key = f"b ({m} → Y)"
            results.add_path(path_key, b_coef, b_se, b_p)
            b_paths[m] = (b_coef, b_se, b_p)
        
        # Calculate indirect effects for each mediator
        total_indirect_effect = 0
        indirect_effects = {}
        
        for m in mediators:
            a_coef, a_se, _ = a_paths[m]
            b_coef, b_se, _ = b_paths[m]
            
            # Check for invalid values
            if np.isnan(a_coef) or np.isnan(b_coef) or np.isinf(a_coef) or np.isinf(b_coef):
                continue
                
            indirect = a_coef * b_coef
            # Sobel test for standard error
            se_indirect = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
            
            # Check for invalid standard error
            if np.isnan(se_indirect) or np.isinf(se_indirect) or se_indirect <= 0:
                z_value = np.nan
                p_indirect = np.nan
            else:
                z_value = indirect / se_indirect
                p_indirect = 2 * (1 - stats.norm.cdf(abs(z_value)))
            
            effect_key = f"indirect_effect_through_{m}"
            results.add_effect(effect_key, indirect, se_indirect, p_indirect)
            indirect_effects[m] = indirect
            total_indirect_effect += indirect
        
        results.add_effect("total_indirect_effect", total_indirect_effect)
        
        # Bootstrap confidence intervals
        n_bootstrap = 5000
        bootstrap_samples = {m: [] for m in mediators}
        bootstrap_total_indirect = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[indices]
            
            boot_indirect_total = 0
            
            for m in mediators:
                a_formula = f"{m} ~ {x_term}{cov_terms}"
                model_a_boot = smf.ols(a_formula, data=boot_df).fit()
                a_boot = model_a_boot.params[x_term]
                
                # Joint model with all mediators
                model_b_boot = smf.ols(b_formula, data=boot_df).fit()
                b_boot = model_b_boot.params[m]
                
                # Check for invalid values
                if np.isnan(a_boot) or np.isnan(b_boot) or np.isinf(a_boot) or np.isinf(b_boot):
                    continue
                    
                indirect_boot = a_boot * b_boot
                bootstrap_samples[m].append(indirect_boot)
                boot_indirect_total += indirect_boot
            
            bootstrap_total_indirect.append(boot_indirect_total)
        
        # Calculate percentile confidence intervals for each mediator
        for m in mediators:
            samples = np.array(bootstrap_samples[m])
            if len(samples) > 0:  # Ensure we have bootstrap samples
                ci_lower = np.percentile(samples, 2.5)
                ci_upper = np.percentile(samples, 97.5)
                
                effect_key = f"indirect_effect_through_{m}"
                results.add_bootstrap_result(effect_key, samples, ci_lower, ci_upper)
                
                # Update effect with CIs
                effect = results.effects[effect_key]
                effect['ci_lower'] = ci_lower
                effect['ci_upper'] = ci_upper
        
        # Total indirect effect CI
        total_samples = np.array(bootstrap_total_indirect)
        if len(total_samples) > 0:  # Ensure we have bootstrap samples
            total_ci_lower = np.percentile(total_samples, 2.5)
            total_ci_upper = np.percentile(total_samples, 97.5)
            
            results.add_effect("total_indirect_effect", total_indirect_effect, 
                            ci_lower=total_ci_lower, ci_upper=total_ci_upper)
            results.add_bootstrap_result("total_indirect_effect", total_samples, 
                                    total_ci_lower, total_ci_upper)
        
        return results
    
    def perform_serial_mediation(self, df, x, mediators, y, covariates=None):
        """Perform serial mediation analysis (X → M1 → M2 → Y)."""
        results = MediationResults(MediationType.SERIAL)
        
        if len(mediators) < 2:
            raise ValueError("Serial mediation requires at least two mediators.")
        
        x_term = x
        y_term = y
        m_terms = mediators
        
        cov_terms = ""
        if covariates and len(covariates) > 0:
            cov_terms = " + " + " + ".join(covariates)
        
        # Total effect: Y ~ X
        c_formula = f"{y_term} ~ {x_term}{cov_terms}"
        model_c = smf.ols(c_formula, data=df).fit()
        results.add_model("c (X → Y, total)", model_c)
        
        c_coef = model_c.params[x_term]
        results.add_path("c (X → Y, total)", c_coef, model_c.bse[x_term], model_c.pvalues[x_term])
        results.add_effect("total_effect", c_coef, model_c.bse[x_term], model_c.pvalues[x_term])
        
        # M1 ~ X (first mediator)
        m1 = m_terms[0]
        a1_formula = f"{m1} ~ {x_term}{cov_terms}"
        model_a1 = smf.ols(a1_formula, data=df).fit()
        results.add_model(f"a1 (X → {m1})", model_a1)
        
        a1_coef = model_a1.params[x_term]
        results.add_path(f"a1 (X → {m1})", a1_coef, model_a1.bse[x_term], model_a1.pvalues[x_term])
        
        # Middle path models for serial mediators
        for i in range(1, len(m_terms)):
            current_m = m_terms[i]
            prev_terms = " + ".join(m_terms[:i])
            
            m_formula = f"{current_m} ~ {x_term} + {prev_terms}{cov_terms}"
            model_m = smf.ols(m_formula, data=df).fit()
            results.add_model(f"Path to {current_m}", model_m)
            
            # Path from X to current M
            results.add_path(f"X → {current_m}", model_m.params[x_term], 
                          model_m.bse[x_term], model_m.pvalues[x_term])
            
            # Paths from previous mediators to current M
            for j in range(i):
                prev_m = m_terms[j]
                results.add_path(f"{prev_m} → {current_m}", model_m.params[prev_m], 
                              model_m.bse[prev_m], model_m.pvalues[prev_m])
        
        # Final model: Y ~ X + M1 + M2 + ...
        all_mediators = " + ".join(m_terms)
        y_formula = f"{y_term} ~ {x_term} + {all_mediators}{cov_terms}"
        model_y = smf.ols(y_formula, data=df).fit()
        results.add_model("Final model (Y ~ X + all mediators)", model_y)
        
        # Direct effect (c')
        c_prime_coef = model_y.params[x_term]
        results.add_path("c' (X → Y, direct)", c_prime_coef, model_y.bse[x_term], model_y.pvalues[x_term])
        results.add_effect("direct_effect", c_prime_coef, model_y.bse[x_term], model_y.pvalues[x_term])
        
        # Paths from each mediator to Y
        for m in m_terms:
            results.add_path(f"{m} → Y", model_y.params[m], model_y.bse[m], model_y.pvalues[m])
        
        # Calculate specific indirect effects (using bootstrap)
        n_bootstrap = 5000
        
        # Potential paths in serial mediation
        paths = []
        
        # X → M1 → Y
        paths.append(("X→M1→Y", [0]))
        
        # X → M2 → Y
        if len(m_terms) >= 2:
            paths.append(("X→M2→Y", [1]))
        
        # X → M1 → M2 → Y
        if len(m_terms) >= 2:
            paths.append(("X→M1→M2→Y", [0, 1]))
        
        # More complex paths for more mediators
        if len(m_terms) >= 3:
            for i in range(2, len(m_terms)):
                # Direct path X → Mi → Y
                paths.append((f"X→{m_terms[i]}→Y", [i]))
                
                # Path through all mediators
                paths.append((f"X→M1→...→{m_terms[i]}→Y", list(range(i+1))))
        
        # Setup bootstrap containers
        bootstrap_paths = {path[0]: [] for path in paths}
        bootstrap_total = []
        
        # Perform bootstrap
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(df), size=len(df), replace=True)
            boot_df = df.iloc[indices]
            
            # First path model (X → M1)
            model_a1_boot = smf.ols(a1_formula, data=boot_df).fit()
            a1_boot = model_a1_boot.params[x_term]
            
            # Middle models
            middle_coeffs = []
            for i in range(1, len(m_terms)):
                current_m = m_terms[i]
                prev_terms = " + ".join(m_terms[:i])
                
                m_formula = f"{current_m} ~ {x_term} + {prev_terms}{cov_terms}"
                model_m_boot = smf.ols(m_formula, data=boot_df).fit()
                
                # Store coefficients from X and previous mediators
                coefs = {
                    'X': model_m_boot.params[x_term]
                }
                
                for j in range(i):
                    prev_m = m_terms[j]
                    coefs[prev_m] = model_m_boot.params[prev_m]
                
                middle_coeffs.append(coefs)
            
            # Final model
            model_y_boot = smf.ols(y_formula, data=boot_df).fit()
            
            # Direct effect
            c_prime_boot = model_y_boot.params[x_term]
            
            # Paths from mediators to Y
            b_boots = {}
            for m in m_terms:
                b_boots[m] = model_y_boot.params[m]
            
            # Calculate specific indirect effects
            boot_total_indirect = 0
            
            # X → M1 → Y
            indirect_xm1y = a1_boot * b_boots[m_terms[0]]
            bootstrap_paths["X→M1→Y"].append(indirect_xm1y)
            boot_total_indirect += indirect_xm1y
            
            # Additional specific paths (if applicable)
            if len(m_terms) >= 2:
                # X → M2 → Y
                indirect_xm2y = middle_coeffs[0]['X'] * b_boots[m_terms[1]]
                bootstrap_paths["X→M2→Y"].append(indirect_xm2y)
                boot_total_indirect += indirect_xm2y
                
                # X → M1 → M2 → Y
                indirect_xm1m2y = a1_boot * middle_coeffs[0][m_terms[0]] * b_boots[m_terms[1]]
                bootstrap_paths["X→M1→M2→Y"].append(indirect_xm1m2y)
                boot_total_indirect += indirect_xm1m2y
            
            # More complex paths for more mediators would be calculated here
            
            bootstrap_total.append(boot_total_indirect)
        
        # Calculate confidence intervals for each path
        for path_name, samples in bootstrap_paths.items():
            samples_array = np.array(samples)
            path_mean = np.mean(samples_array)
            ci_lower = np.percentile(samples_array, 2.5)
            ci_upper = np.percentile(samples_array, 97.5)
            
            results.add_effect(f"indirect_effect_{path_name}", path_mean, ci_lower=ci_lower, ci_upper=ci_upper)
            results.add_bootstrap_result(f"indirect_effect_{path_name}", samples_array, ci_lower, ci_upper)
        
        # Total indirect effect
        total_samples = np.array(bootstrap_total)
        total_indirect = np.mean(total_samples)
        total_ci_lower = np.percentile(total_samples, 2.5)
        total_ci_upper = np.percentile(total_samples, 97.5)
        
        results.add_effect("total_indirect_effect", total_indirect, 
                         ci_lower=total_ci_lower, ci_upper=total_ci_upper)
        results.add_bootstrap_result("total_indirect_effect", total_samples, 
                                  total_ci_lower, total_ci_upper)
        
        return results
    
    def create_mediation_path_diagram(self, results, x, m, y, w=None):
        """Create a path diagram visualization for mediation analysis."""
        fig = Figure(figsize=(8, 5))  # Slightly taller figure
        # Set transparent background for initial display
        fig.patch.set_alpha(0.0)
        
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Turn off axis and make background transparent
        ax.axis('off')
        ax.patch.set_alpha(0.0)
        
        # Define colors
        node_color = '#E0F0FF'  # Light blue for nodes
        node_edge_color = '#3080CF'  # Darker blue for node edges
        a_path_color = '#4CAF50'  # Green for a path
        b_path_color = '#2196F3'  # Blue for b path
        c_path_color = '#F44336'  # Red for c' path
        moderator_color = '#FF9800'  # Orange for moderator
        text_color = '#555555'  # Gray text color
        
        # Font settings - remove color from dictionaries since we specify it directly
        node_font = {'fontsize': 12, 'fontweight': 'bold'}
        coef_font = {'fontsize': 10, 'fontweight': 'normal', 'backgroundcolor': 'white', 'alpha': 0.7}
        
        # Set up coordinates
        if w is None:  # Simple mediation or no moderator
            # Position nodes in better layout
            node_positions = {
                'X': (0.1, 0.5),
                'M': (0.5, 0.7),
                'Y': (0.9, 0.5)
            }
            
            # Draw nodes with better styling
            for name, (x_pos, y_pos) in node_positions.items():
                label = {'X': x, 'M': m, 'Y': y}[name]
                # Create more attractive node
                circle = plt.Circle((x_pos, y_pos), 0.12, fill=True, color=node_color, 
                                   ec=node_edge_color, lw=2, zorder=10)
                ax.add_patch(circle)
                
                # Clean up variable names by removing underscores and capitalizing
                display_label = label.replace('_', ' ').title()
                if len(display_label) > 15:  # Truncate long names
                    display_label = display_label[:12] + '...'
                    
                ax.text(x_pos, y_pos, display_label, ha='center', va='center', color=text_color, **node_font, zorder=11)
            
            # Get path coefficients with proper formatting
            a_coef = results.paths.get('a (X → M)', {}).get('value', 0)
            b_coef = results.paths.get('b (M → Y)', {}).get('value', 0)
            c_prime_path = "c' (X → Y, direct)"
            c_prime_coef = results.paths.get(c_prime_path, {}).get('value', 0)
            
            # Get p-values for significance
            a_p = results.paths.get('a (X → M)', {}).get('p_value', 1)
            b_p = results.paths.get('b (M → Y)', {}).get('p_value', 1)
            c_prime_p = results.paths.get(c_prime_path, {}).get('p_value', 1)
            
            # Determine significance for styling
            a_sig = '**' if a_p < 0.01 else ('*' if a_p < 0.05 else '')
            b_sig = '**' if b_p < 0.01 else ('*' if b_p < 0.05 else '')
            c_sig = '**' if c_prime_p < 0.01 else ('*' if c_prime_p < 0.05 else '')
            
            # Draw curved arrows with better styling
            # X → M (path a)
            ax.annotate("", xy=node_positions['M'], xytext=node_positions['X'],
                       arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", 
                                     color=a_path_color, lw=2))
            
            # M → Y (path b)
            ax.annotate("", xy=node_positions['Y'], xytext=node_positions['M'],
                       arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", 
                                     color=b_path_color, lw=2))
            
            # X → Y (path c')
            ax.annotate("", xy=node_positions['Y'], xytext=node_positions['X'],
                       arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.2", 
                                     color=c_path_color, lw=2))
            
            # Add path labels with significance indicators in better positions
            ax.text(0.3, 0.68, f"a = {a_coef:.3f}{a_sig}", color=a_path_color, **coef_font)
            ax.text(0.7, 0.68, f"b = {b_coef:.3f}{b_sig}", color=b_path_color, **coef_font)
            ax.text(0.5, 0.35, f"c' = {c_prime_coef:.3f}{c_sig}", color=c_path_color, **coef_font)
            
        else:  # Moderated mediation
            # Position nodes in better layout
            node_positions = {
                'X': (0.1, 0.5),
                'M': (0.55, 0.5),
                'Y': (0.9, 0.5),
                'W': (0.3, 0.85)
            }
            
            # Draw nodes with better styling
            for name, (x_pos, y_pos) in node_positions.items():
                label = {'X': x, 'M': m, 'Y': y, 'W': w}[name]
                # Choose colors - moderator gets special color
                color = moderator_color if name == 'W' else node_color
                edge_color = moderator_color if name == 'W' else node_edge_color
                
                # Create more attractive node
                circle = plt.Circle((x_pos, y_pos), 0.12, fill=True, color=color, 
                                   ec=edge_color, lw=2, alpha=0.7, zorder=10)
                ax.add_patch(circle)
                
                # Clean up variable names
                display_label = label.replace('_', ' ').title()
                if len(display_label) > 15:  # Truncate long names
                    display_label = display_label[:12] + '...'
                    
                ax.text(x_pos, y_pos, display_label, ha='center', va='center', color=text_color, **node_font, zorder=11)
            
            # Get coefficients with proper formatting
            a_x_coef = results.paths.get('a_x (X → M)', {}).get('value', 0)
            a_w_coef = results.paths.get('a_w (W → M)', {}).get('value', 0)
            a_xw_coef = results.paths.get('a_xw (X*W → M)', {}).get('value', 0)
            b_coef = results.paths.get('b (M → Y)', {}).get('value', 0)
            c_prime_path = "c' (X → Y, direct)"
            c_prime_coef = results.paths.get(c_prime_path, {}).get('value', 0)
            
            # Get p-values for significance
            a_x_p = results.paths.get('a_x (X → M)', {}).get('p_value', 1)
            a_w_p = results.paths.get('a_w (W → M)', {}).get('p_value', 1)
            a_xw_p = results.paths.get('a_xw (X*W → M)', {}).get('p_value', 1)
            b_p = results.paths.get('b (M → Y)', {}).get('p_value', 1)
            c_prime_p = results.paths.get(c_prime_path, {}).get('p_value', 1)
            
            # Determine significance for styling
            a_x_sig = '**' if a_x_p < 0.01 else ('*' if a_x_p < 0.05 else '')
            a_w_sig = '**' if a_w_p < 0.01 else ('*' if a_w_p < 0.05 else '')
            a_xw_sig = '**' if a_xw_p < 0.01 else ('*' if a_xw_p < 0.05 else '')
            b_sig = '**' if b_p < 0.01 else ('*' if b_p < 0.05 else '')
            c_sig = '**' if c_prime_p < 0.01 else ('*' if c_prime_p < 0.05 else '')
            
            # Draw arrows with better styling
            # X → M (path a)
            ax.annotate("", xy=node_positions['M'], xytext=node_positions['X'],
                       arrowprops=dict(arrowstyle="-|>", color=a_path_color, lw=2))
            
            # M → Y (path b)
            ax.annotate("", xy=node_positions['Y'], xytext=node_positions['M'],
                       arrowprops=dict(arrowstyle="-|>", color=b_path_color, lw=2))
            
            # X → Y (path c')
            ax.annotate("", xy=node_positions['Y'], xytext=node_positions['X'],
                       arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-0.3", 
                                     color=c_path_color, lw=2))
            
            # W → M (moderator)
            ax.annotate("", xy=node_positions['M'], xytext=node_positions['W'],
                       arrowprops=dict(arrowstyle="-|>", color=moderator_color, lw=2))
            
            # Moderation effect (X*W interaction) - create specific point for intersection
            interaction_point = (0.33, 0.6)
            
            # Draw dotted line to interaction point from W
            ax.annotate("", xy=interaction_point, xytext=node_positions['W'],
                       arrowprops=dict(arrowstyle="-", linestyle='dotted', 
                                     color=moderator_color, lw=1.5))
            
            # Draw dotted line to interaction point from X
            ax.annotate("", xy=interaction_point, xytext=(0.2, 0.5),  # Point on X path
                       arrowprops=dict(arrowstyle="-", linestyle='dotted', 
                                     color=moderator_color, lw=1.5))
            
            # Draw arrow from interaction to M path
            ax.annotate("", xy=(0.4, 0.5), xytext=interaction_point,  # Point on path to M
                       arrowprops=dict(arrowstyle="-|>", linestyle='dotted', 
                                     color=moderator_color, lw=1.5))
            
            # Add path labels with significance indicators
            ax.text(0.33, 0.45, f"a_x = {a_x_coef:.3f}{a_x_sig}", color=a_path_color, **coef_font)
            ax.text(0.4, 0.7, f"a_w = {a_w_coef:.3f}{a_w_sig}", color=moderator_color, **coef_font)
            ax.text(0.33, 0.6, f"a_xw = {a_xw_coef:.3f}{a_xw_sig}", color=moderator_color, 
                   fontsize=9, ha='right', va='bottom')
            ax.text(0.73, 0.55, f"b = {b_coef:.3f}{b_sig}", color=b_path_color, **coef_font)
            ax.text(0.45, 0.3, f"c' = {c_prime_coef:.3f}{c_sig}", color=c_path_color, **coef_font)
        
        # Add legend for significance
        legend_x = 0.05
        legend_y = 0.05
        ax.text(legend_x, legend_y, "* p < 0.05, ** p < 0.01", fontsize=8, ha='left', va='bottom', color=text_color)
        
        # Add title with better formatting
        fig.suptitle(f"Mediation Path Diagram: {results.mediation_type.value}", 
                    fontsize=14, fontweight='bold', y=0.98, color=text_color)
        
        # Add indirect effect if available
        if 'indirect_effect' in results.effects:
            ie = results.effects['indirect_effect']['value']
            ie_ci_lower = results.effects['indirect_effect'].get('ci_lower')
            ie_ci_upper = results.effects['indirect_effect'].get('ci_upper')
            
            if ie_ci_lower is not None and ie_ci_upper is not None:
                # Check if significant (CI doesn't include zero)
                is_sig = (ie_ci_lower * ie_ci_upper) > 0
                sig_text = "significant" if is_sig else "not significant"
                ci_text = f"95% CI [{ie_ci_lower:.3f}, {ie_ci_upper:.3f}]"
                
                effect_color = 'green' if is_sig else text_color
                fig.text(0.5, 0.02, 
                        f"Indirect effect = {ie:.3f} ({sig_text})\n{ci_text}", 
                        ha='center', fontsize=10, 
                        fontweight='bold' if is_sig else 'normal',
                        color=effect_color)
        
        return canvas
    
    def create_bootstrap_distribution_plot(self, results, effect_name):
        """Create a visualization of bootstrap distribution for a specific effect."""
        if effect_name not in results.bootstrap_results:
            return None
            
        bootstrap_data = results.bootstrap_results[effect_name]
        values = bootstrap_data['values']
        ci_lower = bootstrap_data['ci_lower']
        ci_upper = bootstrap_data['ci_upper']
        
        fig = Figure(figsize=(8, 4))
        # Set transparent background for initial display
        fig.patch.set_alpha(0.0)
        
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)
        
        # Set text color to gray
        text_color = '#555555'
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        
        # Plot histogram with KDE
        sns.histplot(values, kde=True, ax=ax)
        
        # Add vertical lines for confidence intervals
        ax.axvline(ci_lower, color='red', linestyle='--', label=f'95% CI Lower: {ci_lower:.4f}')
        ax.axvline(ci_upper, color='red', linestyle='--', label=f'95% CI Upper: {ci_upper:.4f}')
        
        # Add vertical line for 0 (for hypothesis testing context)
        ax.axvline(0, color=text_color, linestyle='-', alpha=0.5, label='Zero Effect')
        
        # Add effect mean
        mean_value = np.mean(values)
        ax.axvline(mean_value, color='blue', linestyle='-', label=f'Mean: {mean_value:.4f}')
        
        ax.set_title(f"Bootstrap Distribution for {effect_name}", color=text_color)
        ax.set_xlabel("Effect Size", color=text_color)
        ax.set_ylabel("Frequency", color=text_color)
        
        # Set legend text color to gray
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color(text_color)
        
        return canvas
    
    def generate_moderation_interaction_plot(self, results, df, x, m, w):
        """Generate an interaction plot showing the moderating effect of W on X->M path."""
        # Get different levels of the moderator
        w_mean = df[w].mean()
        w_sd = df[w].std()
        
        w_levels = {
            "low": w_mean - w_sd,
            "moderate": w_mean,
            "high": w_mean + w_sd
        }
        
        # Get coefficients
        a_x = results.paths.get('a_x (X → M)', {}).get('value', 0)
        a_w = results.paths.get('a_w (W → M)', {}).get('value', 0)
        a_xw = results.paths.get('a_xw (X*W → M)', {}).get('value', 0)
        
        # Create X values for plotting
        x_min = df[x].min()
        x_max = df[x].max()
        x_values = np.linspace(x_min, x_max, 100)
        
        fig = Figure(figsize=(8, 4))
        # Set transparent background for initial display
        fig.patch.set_alpha(0.0)
        
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)
        
        # Set text color to gray
        text_color = '#555555'
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(axis='x', colors=text_color)
        ax.tick_params(axis='y', colors=text_color)
        
        # Plot a line for each moderator level
        for level_name, w_value in w_levels.items():
            # Calculate predicted M values
            y_values = []
            for x_val in x_values:
                # Predicted M = a_x*X + a_w*W + a_xw*X*W
                m_pred = a_x*x_val + a_w*w_value + a_xw*x_val*w_value
                y_values.append(m_pred)
            
            ax.plot(x_values, y_values, label=f"{w} = {level_name} ({w_value:.2f})")
        
        ax.set_xlabel(f"Independent Variable ({x})", color=text_color)
        ax.set_ylabel(f"Mediator ({m})", color=text_color)
        ax.set_title(f"Moderation Effect of {w} on the {x} → {m} Relationship", color=text_color)
        
        # Set legend text color to gray
        legend = ax.legend()
        for text in legend.get_texts():
            text.set_color(text_color)
        
        return canvas

    @asyncSlot()
    async def run_mediation_analysis(self):
        """Run the mediation analysis with the selected variables."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        # Get the selected mediation type
        mediation_type = self.mediation_type_combo.currentData()
        if not mediation_type:
            QMessageBox.warning(self, "Error", "No mediation type selected")
            return
        
        # Get the selected variables
        independent = next((col for col, role in self.column_roles.items() 
                    if role == VariableRole.INDEPENDENT), None)
        dependent = next((col for col, role in self.column_roles.items() 
                    if role == VariableRole.DEPENDENT), None)
        mediator = next((col for col, role in self.column_roles.items() 
                    if role == VariableRole.MEDIATOR), None)
        moderator = next((col for col, role in self.column_roles.items() 
                    if role == VariableRole.MODERATOR), None)
        covariates = [col for col, role in self.column_roles.items() 
                    if role == VariableRole.COVARIATE]
        
        # Validate required variables
        missing_vars = []
        if not independent:
            missing_vars.append("Independent (X)")
        if not dependent:
            missing_vars.append("Dependent (Y)")
        if not mediator:
            missing_vars.append("Mediator (M)")
        if mediation_type == MediationType.MODERATED and not moderator:
            missing_vars.append("Moderator (W)")
        
        if missing_vars:
            QMessageBox.warning(
                self,
                "Missing Required Variables",
                "The following required variables are not assigned:\n\n" +
                "\n".join(f"- {var}" for var in missing_vars)
            )
            return
        
        # Show loading message
        self.status_bar.showMessage(f"Running {mediation_type.value} analysis...")
        
        try:
            # Perform the appropriate analysis based on mediation type
            df = self.current_dataframe
            results = None
            
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            if mediation_type == MediationType.SIMPLE:
                results = self.perform_simple_mediation(df, independent, mediator, dependent, covariates)
            
            elif mediation_type == MediationType.MULTIPLE_MEDIATORS:
                # Get all mediators (could be more than one)
                mediators = [col for col, role in self.column_roles.items() 
                        if role == VariableRole.MEDIATOR]
                
                # Validate we have at least one mediator
                if not mediators:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Error", "Multiple mediator analysis requires at least one mediator.")
                    return
                    
                results = self.perform_multiple_mediators(df, independent, mediators, dependent, covariates)
            
            elif mediation_type == MediationType.MULTIPLE_PREDICTORS:
                # Get all independent variables (could be more than one)
                independents = [col for col, role in self.column_roles.items() 
                        if role == VariableRole.INDEPENDENT]
                
                # Validate we have at least one independent variable
                if not independents:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Error", "Multiple predictors analysis requires at least one independent variable.")
                    return
                    
                # Implementation for multiple predictors would go here
                # Since this isn't implemented yet, show a message
                QApplication.restoreOverrideCursor()
                QMessageBox.information(self, "Not Implemented", 
                                    "Multiple predictors analysis is not yet implemented.")
                return
            
            elif mediation_type == MediationType.MODERATED:
                if not moderator:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Error", "Moderated mediation requires a moderator variable.")
                    return
                    
                results = self.perform_moderated_mediation(df, independent, mediator, dependent, moderator, covariates)
            
            elif mediation_type == MediationType.SERIAL:
                # Get all mediators for serial mediation (need at least 2)
                mediators = [col for col, role in self.column_roles.items() 
                        if role == VariableRole.MEDIATOR]
                if len(mediators) < 2:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Error", "Serial mediation requires at least two mediators.")
                    return
                results = self.perform_serial_mediation(df, independent, mediators, dependent, covariates)
            
            QApplication.restoreOverrideCursor()
            
            if results:
                # Display the full analysis results in the detailed results tab
                self.results_text.setPlainText(results.format_summary())
                
                # Create a summary for the summary tab
                summary = self.create_results_summary(results, independent, mediator, dependent, moderator)
                self.summary_text.setPlainText(summary)
                
                # Clear previous interpretation
                self.interpretation_text.clear()
                
                # Create visualization tab
                self.create_visualizations(results, df, independent, mediator, dependent, moderator)
                
                # Store the results for later use
                self.last_analysis_result = {
                    'mediation_type': mediation_type.value,
                    'independent': independent,
                    'mediator': mediator,
                    'dependent': dependent,
                    'moderator': moderator,
                    'covariates': covariates,
                    'response': results.format_summary(),
                    'dataset': self.current_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update the table to reflect variable roles
                self.display_dataset(df)
                
                # Show the summary tab
                left_tabs = self.findChild(QTabWidget)
                if left_tabs:
                    for i in range(left_tabs.count()):
                        if left_tabs.tabText(i) == "Summary":
                            left_tabs.setCurrentIndex(i)
                            break
                
                self.status_bar.showMessage(f"Completed {mediation_type.value} analysis")
            else:
                QMessageBox.warning(self, "Error", "Analysis did not produce valid results")
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Mediation analysis failed: {str(e)}")
            self.status_bar.showMessage("Analysis failed with error")
            import traceback
            traceback.print_exc()
    
    def create_results_summary(self, results, x, m, y, w=None):
        """Create a summary of the mediation analysis results."""
        summary = f"# {results.mediation_type.value} Analysis Summary\n\n"
        
        # Variables used
        summary += "## Variables\n"
        summary += f"- Independent (X): {x}\n"
        summary += f"- Mediator (M): {m}\n"
        summary += f"- Dependent (Y): {y}\n"
        if w:
            summary += f"- Moderator (W): {w}\n"
            
        # Key effects
        summary += "\n## Key Effects\n"
        
        # Direct effect (c')
        if 'direct_effect' in results.effects:
            effect = results.effects['direct_effect']
            value = effect['value']
            p_value = effect.get('p_value')
            ci_lower = effect.get('ci_lower')
            ci_upper = effect.get('ci_upper')
            
            summary += f"- Direct effect (X → Y): {value:.4f}"
            if p_value is not None:
                summary += f", p = {p_value:.4f}"
            if ci_lower is not None and ci_upper is not None:
                summary += f", 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]"
            summary += "\n"
        
        # Indirect effect
        if 'indirect_effect' in results.effects:
            effect = results.effects['indirect_effect']
            value = effect['value']
            p_value = effect.get('p_value')
            ci_lower = effect.get('ci_lower')
            ci_upper = effect.get('ci_upper')
            
            summary += f"- Indirect effect (X → M → Y): {value:.4f}"
            if p_value is not None:
                summary += f", p = {p_value:.4f}"
            if ci_lower is not None and ci_upper is not None:
                summary += f", 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]"
            summary += "\n"
        
        # Total effect
        if 'total_effect' in results.effects:
            effect = results.effects['total_effect']
            value = effect['value']
            p_value = effect.get('p_value')
            
            summary += f"- Total effect (X → Y): {value:.4f}"
            if p_value is not None:
                summary += f", p = {p_value:.4f}"
            summary += "\n"
            
        # Add basic interpretation
        summary += "\n## Quick Interpretation\n"
        
        # Check for significant mediation
        if 'indirect_effect' in results.effects and results.effects['indirect_effect'].get('ci_lower') is not None:
            indirect = results.effects['indirect_effect']
            if indirect['ci_lower'] * indirect['ci_upper'] > 0:  # same sign means CI doesn't include 0
                summary += "- ✓ **Significant mediation**: The indirect effect confidence interval does not include zero.\n"
            else:
                summary += "- ✗ **Non-significant mediation**: The indirect effect confidence interval includes zero.\n"
        
        # Check for significant direct effect
        if 'direct_effect' in results.effects and results.effects['direct_effect'].get('p_value') is not None:
            direct_p = results.effects['direct_effect']['p_value']
            if direct_p < 0.05:
                summary += f"- ✓ **Direct effect** is statistically significant (p={direct_p:.4f}).\n"
            else:
                summary += f"- ✗ **Direct effect** is not statistically significant (p={direct_p:.4f}).\n"
        
        # Determine mediation type
        if 'direct_effect' in results.effects and 'indirect_effect' in results.effects:
            direct = results.effects['direct_effect']
            indirect = results.effects['indirect_effect']
            direct_sig = direct.get('p_value', 1) < 0.05
            indirect_sig = indirect.get('ci_lower', 0) * indirect.get('ci_upper', 0) > 0
            
            if direct_sig and indirect_sig:
                summary += "- **Partial mediation**: Both direct and indirect effects are significant.\n"
            elif not direct_sig and indirect_sig:
                summary += "- **Full mediation**: Indirect effect is significant, but direct effect is not.\n"
            elif direct_sig and not indirect_sig:
                summary += "- **No mediation**: Direct effect is significant, but indirect effect is not.\n"
            else:
                summary += "- **No effect**: Neither direct nor indirect effects are significant.\n"
        
        return summary
    
    def create_visualizations(self, results, df, x, m, y, w=None):
        """Create visualizations for the mediation analysis results."""
        # Clear previous visualizations
        self.viz_tabs.clear()
        # Initialize bootstrap plot variable
        self.current_bootstrap_plot = None
        
        # Tab 1: Path Diagram
        path_diagram_tab = QWidget()
        path_diagram_layout = QVBoxLayout(path_diagram_tab)
        
        # Create toolbar with maximize button
        toolbar = QHBoxLayout()
        maximize_btn = QPushButton()
        maximize_btn.setIcon(load_bootstrap_icon("arrows-fullscreen", size=18))
        maximize_btn.setToolTip("Maximize")
        maximize_btn.setFixedSize(30, 30)
        toolbar.addStretch()
        toolbar.addWidget(maximize_btn)
        path_diagram_layout.addLayout(toolbar)
        
        # Create the path diagram
        path_diagram = self.create_mediation_path_diagram(results, x, m, y, w)
        path_diagram_layout.addWidget(path_diagram)
        
        # Connect maximize button to use fig_to_svg
        maximize_btn.clicked.connect(lambda: self.show_plot_modal(fig_to_svg(path_diagram.figure), "Path Diagram"))
        
        self.viz_tabs.addTab(path_diagram_tab, "Path Diagram")
        
        # Tab 2: Bootstrap Distributions
        bootstrap_tab = QWidget()
        bootstrap_layout = QVBoxLayout(bootstrap_tab)
        
        # Add dropdown to select which effect to visualize
        effect_selector = QComboBox()
        for effect_name in results.bootstrap_results.keys():
            effect_selector.addItem(effect_name)
        
        bootstrap_layout.addWidget(QLabel("Select effect to visualize:"))
        bootstrap_layout.addWidget(effect_selector)
        
        # Add toolbar with maximize button
        bootstrap_toolbar = QHBoxLayout()
        bootstrap_maximize_btn = QPushButton()
        bootstrap_maximize_btn.setIcon(load_bootstrap_icon("arrows-fullscreen", size=18))
        bootstrap_maximize_btn.setToolTip("Maximize")
        bootstrap_maximize_btn.setFixedSize(30, 30)
        bootstrap_toolbar.addStretch()
        bootstrap_toolbar.addWidget(bootstrap_maximize_btn)
        bootstrap_layout.addLayout(bootstrap_toolbar)
        
        # Plot container
        bootstrap_plot_container = QWidget()
        bootstrap_plot_layout = QVBoxLayout(bootstrap_plot_container)
        
        # Initial plot
        if results.bootstrap_results:
            initial_effect = next(iter(results.bootstrap_results))
            initial_plot = self.create_bootstrap_distribution_plot(results, initial_effect)
            if initial_plot:
                self.current_bootstrap_plot = initial_plot.figure
                bootstrap_plot_layout.addWidget(initial_plot)
        
        bootstrap_layout.addWidget(bootstrap_plot_container)
        
        # Connect maximize button - use fig_to_svg
        bootstrap_maximize_btn.clicked.connect(lambda: self.show_plot_modal(
            fig_to_svg(self.current_bootstrap_plot) if self.current_bootstrap_plot else "No plot available", 
            f"Bootstrap Distribution: {effect_selector.currentText()}"
        ))
        
        # Connect dropdown to update plot
        def update_bootstrap_plot():
            # Clear current plot
            for i in reversed(range(bootstrap_plot_layout.count())): 
                bootstrap_plot_layout.itemAt(i).widget().setParent(None)
            
            # Add new plot
            effect_name = effect_selector.currentText()
            new_plot = self.create_bootstrap_distribution_plot(results, effect_name)
            if new_plot:
                self.current_bootstrap_plot = new_plot.figure
                bootstrap_plot_layout.addWidget(new_plot)
        
        effect_selector.currentIndexChanged.connect(update_bootstrap_plot)
        
        self.viz_tabs.addTab(bootstrap_tab, "Bootstrap Distributions")
        
        # Tab 3: Interaction Plot (for moderated mediation)
        if results.mediation_type == MediationType.MODERATED and w is not None:
            interaction_tab = QWidget()
            interaction_layout = QVBoxLayout(interaction_tab)
            
            # Add toolbar with maximize button
            interaction_toolbar = QHBoxLayout()
            interaction_maximize_btn = QPushButton()
            interaction_maximize_btn.setIcon(load_bootstrap_icon("arrows-fullscreen", size=18))
            interaction_maximize_btn.setToolTip("Maximize")
            interaction_maximize_btn.setFixedSize(30, 30)
            interaction_toolbar.addStretch()
            interaction_toolbar.addWidget(interaction_maximize_btn)
            interaction_layout.addLayout(interaction_toolbar)
            
            # Create the plot
            interaction_plot = self.generate_moderation_interaction_plot(results, df, x, m, w) 
            interaction_layout.addWidget(interaction_plot)
            
            # Connect maximize button to use fig_to_svg
            interaction_maximize_btn.clicked.connect(lambda: self.show_plot_modal(
                fig_to_svg(interaction_plot.figure), 
                "Moderation Effect"
            ))
            
            self.viz_tabs.addTab(interaction_tab, "Moderation Effect")
    
    @asyncSlot()
    async def interpret_results(self):
        """Use LLM to help interpret the mediation analysis results."""
        if not self.last_analysis_result:
            QMessageBox.warning(self, "Error", "No analysis results to interpret")
            return
        
        self.interpretation_text.setHtml("<p><i>Generating interpretation... Please wait.</i></p>")
        QApplication.processEvents()  # Update the UI
        
        try:
            # Get the analysis results
            results_text = self.results_text.toPlainText()
            
            # Create a prompt for deeper interpretation with JSON format
            prompt = f"""
            I've conducted a {self.last_analysis_result['mediation_type']} analysis and need help interpreting the results more thoroughly.
            
            Variables:
            - Independent (X): {self.last_analysis_result['independent']}
            - Mediator (M): {self.last_analysis_result['mediator']}
            - Dependent (Y): {self.last_analysis_result['dependent']}
            """
            
            if self.last_analysis_result.get('moderator'):
                prompt += f"- Moderator (W): {self.last_analysis_result['moderator']}\n"
            
            prompt += f"""
            Here are the statistical results:
            
            {results_text}
            
            Please provide your interpretation in a JSON format with the following structure:
            {{
                "summary": "Brief 1-2 sentence summary of findings",
                "detailed_interpretation": "Detailed interpretation of the results in plain language",
                "mechanisms": "What these findings suggest about the underlying mechanisms",
                "alternative_explanations": "Potential alternative explanations for these results",
                "methodological_notes": "Strengths and limitations of this analysis",
                "recommendations": "Recommendations for future research based on these findings"
            }}
            
            Make sure the JSON is valid. Within each JSON field, you can use markdown formatting.
            Focus particularly on the practical significance of these results, not just statistical significance.
            """
            
            # Call LLM for interpretation - add model parameter
            response = await call_llm_async(prompt, model=llm_config.default_text_model)
            
            # Extract JSON portion from response
            import re
            import json
            json_match = re.search(r'({[\s\S]*})', response)
            
            if json_match:
                json_str = json_match.group(1)
                interpretation_json = json.loads(json_str)
                
                # Store the JSON in the last_analysis_result for future use
                self.last_analysis_result['interpretation'] = interpretation_json
                
                # Convert JSON to HTML for display
                html_content = "<h2>Mediation Analysis Interpretation</h2>"
                html_content += "<h3>Summary</h3>"
                html_content += f"<p>{interpretation_json.get('summary', '')}</p>"
                
                html_content += "<h3>Detailed Interpretation</h3>"
                html_content += f"<p>{interpretation_json.get('detailed_interpretation', '').replace(chr(10), '<br>')}</p>"
                
                html_content += "<h3>Underlying Mechanisms</h3>"
                html_content += f"<p>{interpretation_json.get('mechanisms', '').replace(chr(10), '<br>')}</p>"
                
                html_content += "<h3>Alternative Explanations</h3>"
                html_content += f"<p>{interpretation_json.get('alternative_explanations', '').replace(chr(10), '<br>')}</p>"
                
                html_content += "<h3>Methodological Strengths and Limitations</h3>"
                html_content += f"<p>{interpretation_json.get('methodological_notes', '').replace(chr(10), '<br>')}</p>"
                
                html_content += "<h3>Recommendations for Future Research</h3>"
                html_content += f"<p>{interpretation_json.get('recommendations', '').replace(chr(10), '<br>')}</p>"
                
                # Apply markdown formatting to HTML content
                for section in html_content.split('<p>'):
                    if '</p>' in section:
                        content = section.split('</p>')[0]
                        formatted_content = content.replace('**', '<b>').replace('**', '</b>').replace('*', '<i>').replace('*', '</i>')
                        html_content = html_content.replace(content, formatted_content)
                
                # Display the formatted interpretation
                self.interpretation_text.setHtml(html_content)
            else:
                # If JSON parsing fails, display raw text
                self.interpretation_text.setHtml(f"<p>Could not parse response as JSON. Raw response:</p><pre>{response}</pre>")
            
            # Switch to the interpretation tab
            left_tabs = self.findChild(QTabWidget)
            if left_tabs:
                for i in range(left_tabs.count()):
                    if left_tabs.tabText(i) == "AI Interpretation":
                        left_tabs.setCurrentIndex(i)
                        break
            
        except Exception as e:
            self.interpretation_text.setHtml(f"<p style='color:red'>Error generating interpretation: {str(e)}</p>")
            import traceback
            traceback.print_exc()

    def show_plot_modal(self, figure, title="Plot"):
        """
        Open the given figure in a modal dialog that maximizes and preserves aspect ratio.
        
        Args:
            figure: A matplotlib Figure, a wrapper with a .figure attribute, or SVG content (string).
            title (str): Title of the modal dialog.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        layout = QVBoxLayout(dialog)
        
        # Handle SVG figures
        if isinstance(figure, str):
            try:
                # Check if it's SVG content
                if figure.startswith('<?xml') or figure.startswith('<svg'):
                    svg_widget = QSvgWidget()
                    svg_widget.renderer().load(QByteArray(figure.encode('utf-8')))
                    svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                    aspect_widget = SVGAspectRatioWidget(svg_widget)
                    layout.addWidget(aspect_widget)
                else:
                    placeholder = QLabel("Invalid SVG content")
                    placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(placeholder)
            except Exception as e:
                error_label = QLabel(f"Error processing SVG: {str(e)}")
                error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(error_label)
        
        # Handle matplotlib figures
        elif isinstance(figure, Figure):
            # Convert to SVG for consistent display
            svg_content = fig_to_svg(figure)
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_content.encode('utf-8')))
            svg_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            aspect_widget = SVGAspectRatioWidget(svg_widget)
            layout.addWidget(aspect_widget)
        
        elif hasattr(figure, 'figure') and isinstance(figure.figure, Figure):
            # Convert to SVG for consistent display
            svg_content = fig_to_svg(figure.figure)
            svg_widget = QSvgWidget()
            svg_widget.renderer().load(QByteArray(svg_content.encode('utf-8')))
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
