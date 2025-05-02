import logging
import os
import json
import uuid
import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer, QUrl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGridLayout,
    QLabel, QComboBox, QPlainTextEdit, QFormLayout, QSplitter, 
    QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QStatusBar, QRadioButton, QButtonGroup, QScrollArea, QApplication, QCheckBox,
    QDialog, QDialogButtonBox, QTextBrowser, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem, QListWidget, QFrame, QTextEdit
)
from PyQt6.QtGui import QIcon, QColor, QBrush, QFont, QTextDocument, QTransform

# Import the load_bootstrap_icon function
from data.selection.helpers import analyze_dataset_structure
from helpers.load_icon import load_bootstrap_icon

import re
import asyncio
from qasync import asyncSlot

from data.selection.test_executors import TestExecutorFactory
from study_model.data_type_registry import get_compatible_tests_for_data_type, infer_data_type
from study_model.study_design_registry import STUDY_DESIGN_REGISTRY, VariableRequirement
from study_model.study_model import (
    StatisticalTest, StudyType, CFDataType, TimePoint, 
    OutcomeMeasure, CovariateDefinition, AnalysisPlan,
    OutcomeCategory, DataCollectionMethod  # Add these imports
)
from data.selection.stat_tests import TEST_REGISTRY
from llms.client import call_llm_async
from data.selection.masking_utils import get_column_mapping
import scipy.stats as stats  # Add this near other imports at the top

# At the top of the file, add a helper function to convert icons to pixmaps
def get_indicator_pixmap(icon_name, color, size=16):
    """Convert a bootstrap icon to a pixmap for use in status indicators."""
    icon = load_bootstrap_icon(icon_name, color, size)
    # Convert the icon to a pixmap
    return icon.pixmap(QSize(size, size))

class VariableRole(Enum):
    """Defines the roles that variables can play in statistical tests."""
    NONE = "none"
    OUTCOME = "outcome"
    GROUP = "group"
    COVARIATE = "covariate"
    SUBJECT_ID = "subject_id"
    TIME = "time"
    PAIR_ID = "pair_id"
    EVENT = "event"

class DataTestingWidget(QWidget):
    """Widget for selecting variables and statistical tests."""
    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Data Testing")
        
        # Internal state
        self.current_dataframe = None
        self.current_name = ""
        self.column_roles = {}  # Maps column names to their roles
        self.covariate_order = []  # Track the order of covariates
        self.selected_test = None
        self.last_test_result = None
        self.test_results = {}  # Dictionary to store results for multiple outcomes
        self.studies_manager = None

        self.variable_roles = {}

        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI."""
        # Main layout
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)
        
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Main splitter for the UI
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top section: Dataset selection and variable assignment
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)
        top_layout.setContentsMargins(5, 5, 5, 5)
        
        # Dataset selector layout
        dataset_selector_layout = QHBoxLayout()
        dataset_selector_layout.setContentsMargins(0, 0, 0, 0)
        dataset_selector_layout.setSpacing(8)  # Increased spacing
        
        # Dataset selector button
        refresh_button = QPushButton()
        refresh_button.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_button.setToolTip("Select Dataset")
        refresh_button.setFixedSize(32, 32)  # Slightly larger
        refresh_button.clicked.connect(self.load_dataset_from_study)
        dataset_selector_layout.addWidget(refresh_button)
        
        # Dataset dropdown - balanced width
        self.dataset_selector = QComboBox()
        self.dataset_selector.setMinimumWidth(400)  # Increased from 350
        self.dataset_selector.setMaximumWidth(600)  # Added maximum constraint
        self.dataset_selector.currentIndexChanged.connect(self.on_dataset_changed)
        dataset_selector_layout.addWidget(self.dataset_selector)
        
        # Spacer to separate dropdown from utility buttons
        dataset_selector_layout.addSpacing(12)
        
        # Add some commonly used buttons next to the dropdown
        clear_all_small = QPushButton()
        clear_all_small.setIcon(load_bootstrap_icon("x-circle"))
        clear_all_small.setToolTip("Reset all selections and results")
        clear_all_small.setFixedSize(32, 32)  # Slightly larger
        clear_all_small.clicked.connect(self.clear_all)
        dataset_selector_layout.addWidget(clear_all_small)
        
        # Add stretch to push everything to the left
        dataset_selector_layout.addStretch(1)
        
        top_layout.addLayout(dataset_selector_layout)

        # Data source controls - now using grid layout for two rows of buttons
        buttons_section = QWidget()
        buttons_layout = QGridLayout(buttons_section)
        buttons_layout.setContentsMargins(5, 8, 5, 8)  # Increased vertical margins
        buttons_layout.setHorizontalSpacing(10)  # Increased spacing
        buttons_layout.setVerticalSpacing(12)  # Increased spacing

        # Set consistent style for all buttons - more balanced
        button_style = """
            QPushButton {
                padding: 5px 10px;
                border-radius: 4px;
                text-align: center;
                min-width: 110px;
                border: none;
            }
        """

        # Row 1: Test-related buttons
        self.auto_assign_button = QPushButton(" Map Variables")
        self.auto_assign_button.setIcon(load_bootstrap_icon("graph-up"))
        self.auto_assign_button.setToolTip("Automatically map variables to roles")
        self.auto_assign_button.setStyleSheet(button_style)
        self.auto_assign_button.clicked.connect(self.build_model)
        buttons_layout.addWidget(self.auto_assign_button, 0, 0)
        
        self.auto_select_button = QPushButton(" Identify Test")
        self.auto_select_button.setIcon(load_bootstrap_icon("search"))
        self.auto_select_button.setToolTip("Automatically identify appropriate statistical test")
        self.auto_select_button.setStyleSheet(button_style)
        self.auto_select_button.clicked.connect(self.auto_select_test)
        buttons_layout.addWidget(self.auto_select_button, 0, 1)
        
        self.run_test_button = QPushButton(" Run Test")
        self.run_test_button.setIcon(load_bootstrap_icon("play"))
        self.run_test_button.setToolTip("Run the selected statistical test")
        self.run_test_button.setStyleSheet(button_style)
        self.run_test_button.clicked.connect(self.run_statistical_test)
        buttons_layout.addWidget(self.run_test_button, 0, 2)
        
        self.analyze_all_outcomes_button = QPushButton(" Run All")
        self.analyze_all_outcomes_button.setIcon(load_bootstrap_icon("play-fill"))
        self.analyze_all_outcomes_button.setToolTip("Run tests for all outcomes")
        self.analyze_all_outcomes_button.setStyleSheet(button_style)
        self.analyze_all_outcomes_button.clicked.connect(self.analyze_all_outcomes)
        buttons_layout.addWidget(self.analyze_all_outcomes_button, 0, 3)
        
        # Add vertical divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        buttons_layout.addWidget(divider, 0, 4, 2, 1)  # Span 2 rows
        
        # Add the second set of buttons on the same row
        self.update_design_button = QPushButton(" Update study")
        self.update_design_button.setIcon(load_bootstrap_icon("file-earmark-text"))
        self.update_design_button.setToolTip("Update study documentation")
        self.update_design_button.setStyleSheet(button_style)
        self.update_design_button.clicked.connect(self.update_study_design)
        buttons_layout.addWidget(self.update_design_button, 0, 5)
        
        self.save_results_button = QPushButton(" Save Session")
        self.save_results_button.setIcon(load_bootstrap_icon("save"))
        self.save_results_button.setToolTip("Save test results to the study")
        self.save_results_button.setStyleSheet(button_style)
        self.save_results_button.clicked.connect(self.save_results_to_study)
        buttons_layout.addWidget(self.save_results_button, 0, 6)
        
        self.manual_interpretation_button = QPushButton(" Interpret Results")
        self.manual_interpretation_button.setIcon(load_bootstrap_icon("chat-text"))
        self.manual_interpretation_button.setToolTip("Interpret the results of the selected test")
        self.manual_interpretation_button.setStyleSheet(button_style)
        self.manual_interpretation_button.clicked.connect(self.show_manual_interpretation_dialog)
        buttons_layout.addWidget(self.manual_interpretation_button, 0, 7)

        self.manual_hypothesis_button = QPushButton(" Add Hypothesis")
        self.manual_hypothesis_button.setIcon(load_bootstrap_icon("journal-text"))
        self.manual_hypothesis_button.setToolTip("Add hypothesis for the selected outcome")
        self.manual_hypothesis_button.setStyleSheet(button_style)
        self.manual_hypothesis_button.clicked.connect(self.show_manual_hypothesis_dialog)
        buttons_layout.addWidget(self.manual_hypothesis_button, 0, 8)
        
        # Add stretch column for better layout
        buttons_layout.setColumnStretch(9, 1)
        
        top_layout.addWidget(buttons_section)
        
        
        # Enhanced Selected Variables section with manual mapping - using grid layout
        selected_vars_group = QGroupBox("Variable Selection")
        selected_vars_layout = QHBoxLayout(selected_vars_group)  # Change to horizontal layout
        
        # Create dropdowns for each role
        self.outcome_combo = QComboBox()
        self.outcome_combo.setMinimumWidth(180)
        self.outcome_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.OUTCOME))
        self.outcome_combo.setEnabled(False)
        
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(180)
        self.group_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.GROUP))
        self.group_combo.setEnabled(False)
        
        self.subject_id_combo = QComboBox()
        self.subject_id_combo.setMinimumWidth(180)
        self.subject_id_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.SUBJECT_ID))
        self.subject_id_combo.setEnabled(False)
        
        self.time_combo = QComboBox()
        self.time_combo.setMinimumWidth(180)
        self.time_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.TIME))
        self.time_combo.setEnabled(False)
        
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(180)
        self.event_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.EVENT))
        self.event_combo.setEnabled(False)
        
        self.pair_id_combo = QComboBox()
        self.pair_id_combo.setMinimumWidth(180)
        self.pair_id_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.PAIR_ID))
        self.pair_id_combo.setEnabled(False)
        
        # Setup for covariates selection
        self.covariates_combo = QComboBox()
        self.covariates_combo.setMinimumWidth(180)
        self.covariates_combo.setEnabled(False)
        
        self.add_covariate_button = QPushButton("Add")
        self.add_covariate_button.setIcon(load_bootstrap_icon("plus"))
        self.add_covariate_button.clicked.connect(self.add_covariate)
        self.add_covariate_button.setEnabled(False)
        
        # Left side: Role selectors
        role_selectors_widget = QWidget()
        role_selectors_layout = QGridLayout(role_selectors_widget)
        role_selectors_layout.setContentsMargins(0, 0, 0, 0)
        
        # Arrange in a balanced grid
        role_selectors_layout.addWidget(QLabel("Outcome:"), 0, 0)
        role_selectors_layout.addWidget(self.outcome_combo, 0, 1)
        role_selectors_layout.addWidget(QLabel("Group:"), 1, 0)
        role_selectors_layout.addWidget(self.group_combo, 1, 1)
        role_selectors_layout.addWidget(QLabel("Subject ID:"), 2, 0)
        role_selectors_layout.addWidget(self.subject_id_combo, 2, 1)
        role_selectors_layout.addWidget(QLabel("Time:"), 3, 0)
        role_selectors_layout.addWidget(self.time_combo, 3, 1)
        role_selectors_layout.addWidget(QLabel("Event:"), 4, 0)
        role_selectors_layout.addWidget(self.event_combo, 4, 1)
        role_selectors_layout.addWidget(QLabel("Pair ID:"), 5, 0)
        role_selectors_layout.addWidget(self.pair_id_combo, 5, 1)
        
        # Add clear button below role selectors
        clear_button = QPushButton("Clear All Assignments")
        clear_button.setIcon(load_bootstrap_icon("x-circle"))
        clear_button.clicked.connect(self.clear_all_assignments)
        role_selectors_layout.addWidget(clear_button, 6, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        
        # Add role selectors to the main layout
        selected_vars_layout.addWidget(role_selectors_widget, 1)  # Equal stretch
        
        # Create a separate section for covariates management
        covariates_group = QGroupBox("Covariates")
        covariates_group_layout = QVBoxLayout(covariates_group)
        
        # Add covariate controls in horizontal layout
        add_covariate_container = QWidget()
        add_covariate_layout = QHBoxLayout(add_covariate_container)
        add_covariate_layout.setContentsMargins(0, 0, 0, 0)
        add_covariate_layout.addWidget(QLabel("Add:"))
        add_covariate_layout.addWidget(self.covariates_combo, 1)  # Give it stretch factor
        add_covariate_layout.addWidget(self.add_covariate_button)
        covariates_group_layout.addWidget(add_covariate_container)
        
        # Replace QPlainTextEdit with QListWidget for selected covariates
        covariates_group_layout.addWidget(QLabel("Selected Covariates:"))
        self.covariates_list = QListWidget()
        self.covariates_list.setMinimumHeight(200)  # Increased height for covariates
        self.covariates_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.covariates_list.itemSelectionChanged.connect(self._update_covariate_button_states)
        covariates_group_layout.addWidget(self.covariates_list)
        
        # Add buttons to move covariates up/down
        covariate_buttons_container = QWidget()
        covariate_buttons_layout = QHBoxLayout(covariate_buttons_container)
        covariate_buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.setIcon(load_bootstrap_icon("arrow-up"))
        self.move_up_button.clicked.connect(self.move_covariate_up)
        
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.setIcon(load_bootstrap_icon("arrow-down"))
        self.move_down_button.clicked.connect(self.move_covariate_down)
        
        self.remove_covariate_button = QPushButton("Remove")
        self.remove_covariate_button.setIcon(load_bootstrap_icon("trash"))
        self.remove_covariate_button.clicked.connect(self.remove_covariate)
        
        covariate_buttons_layout.addWidget(self.move_up_button)
        covariate_buttons_layout.addWidget(self.move_down_button)
        covariate_buttons_layout.addWidget(self.remove_covariate_button)
        
        covariates_group_layout.addWidget(covariate_buttons_container)
        
        # Add the covariates group to the main layout 
        selected_vars_layout.addWidget(covariates_group, 1)  # Equal stretch
        
        top_layout.addWidget(selected_vars_group)
        
        # Create a horizontal layout for Study Design and Test Selection
        design_and_test_container = QWidget()
        design_and_test_layout = QHBoxLayout(design_and_test_container)
        design_and_test_layout.setContentsMargins(0, 0, 0, 0)
        
        # Study design type selection
        design_group = QGroupBox("Study Design")
        design_layout = QVBoxLayout(design_group)
        
        # Add combobox for study design types
        design_selector_layout = QHBoxLayout()
        design_selector_layout.addWidget(QLabel("Type:"))
        self.design_type_combo = QComboBox()
        # We will populate this in update_design_type_combo, not here
        design_selector_layout.addWidget(self.design_type_combo, 1)
        
        # Design help button
        design_help_button = QPushButton("Info")
        design_help_button.setIcon(load_bootstrap_icon("info-circle"))
        design_help_button.clicked.connect(self.show_design_info)
        design_selector_layout.addWidget(design_help_button)
        
        design_layout.addLayout(design_selector_layout)
        
        # Add a description label
        self.design_description = QLabel()
        self.design_description.setWordWrap(True)
        self.design_description.setStyleSheet("font-style: italic;")
        design_layout.addWidget(self.design_description)
        
        # Connect the combobox to update the description and required fields
        self.design_type_combo.currentIndexChanged.connect(self.update_design_description)
        self.design_type_combo.currentIndexChanged.connect(self.update_required_fields)
        
        # Add design group to the horizontal layout
        design_and_test_layout.addWidget(design_group, 1)  # Equal stretch
        
        # Test selection
        test_selection_group = QGroupBox("Test Selection")
        test_selection_layout = QVBoxLayout(test_selection_group)
        test_selection_group.setMinimumWidth(350)  # More balanced width
        
        # Add hypothesis dropdown for existing hypotheses
        hypothesis_layout = QHBoxLayout()
        hypothesis_layout.addWidget(QLabel("Hypothesis:"))
        self.hypothesis_combo = QComboBox()
        self.hypothesis_combo.setToolTip("Select from existing hypotheses for this study")
        hypothesis_layout.addWidget(self.hypothesis_combo, 1)
        
        # Refresh hypothesis button
        refresh_hyp_btn = QPushButton()
        refresh_hyp_btn.setIcon(load_bootstrap_icon("arrow-repeat"))
        refresh_hyp_btn.setToolTip("Refresh hypotheses list")
        refresh_hyp_btn.setFixedSize(24, 24)
        refresh_hyp_btn.clicked.connect(self.refresh_hypotheses)
        hypothesis_layout.addWidget(refresh_hyp_btn)
        
        test_selection_layout.addLayout(hypothesis_layout)
        
        # Add test selection dropdown
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Statistical Test:"))
        self.test_combo = QComboBox()
        
        # Add tests from the registry
        for test_key in TEST_REGISTRY:
            test_data = TEST_REGISTRY[test_key]
            self.test_combo.addItem(test_data.name, test_key)
        
        # Connect test combo changes to update mu input visibility
        self.test_combo.currentIndexChanged.connect(self.on_test_changed)
        
        test_layout.addWidget(self.test_combo, 1)
        test_selection_layout.addLayout(test_layout)
        
        # NEW: Add assumption status indicators
        assumptions_status_layout = QHBoxLayout()
        assumptions_status_layout.setSpacing(10)
        
        # Sample size/CLT indicator
        self.clt_status_label = QLabel("Sample Size:")
        self.clt_status_icon = QLabel()
        self.clt_status_icon.setToolTip("Sample size sufficient for Central Limit Theorem")
        # Default to unknown state (gray)
        self.clt_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
        assumptions_status_layout.addWidget(self.clt_status_label)
        assumptions_status_layout.addWidget(self.clt_status_icon)
        
        # Normality indicator
        self.normality_status_label = QLabel("Normality:")
        self.normality_status_icon = QLabel()
        self.normality_status_icon.setToolTip("Data normality status")
        # Default to unknown state (gray)
        self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
        assumptions_status_layout.addWidget(self.normality_status_label)
        assumptions_status_layout.addWidget(self.normality_status_icon)
        
        # Group balance indicator
        self.balance_status_label = QLabel("Group Balance:")
        self.balance_status_icon = QLabel()
        self.balance_status_icon.setToolTip("Group size balance status")
        # Default to unknown state (gray)
        self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
        assumptions_status_layout.addWidget(self.balance_status_label)
        assumptions_status_layout.addWidget(self.balance_status_icon)
        
        # Add stretcher to keep indicators left-aligned
        assumptions_status_layout.addStretch(1)
        
        # Add clickable info button that explains the indicators
        indicators_help_btn = QPushButton()
        indicators_help_btn.setIcon(load_bootstrap_icon("info-circle", size=16))
        indicators_help_btn.setToolTip("Click for information about these indicators")
        indicators_help_btn.setFixedSize(24, 24)
        indicators_help_btn.setStyleSheet("background-color: transparent; border: none;")
        indicators_help_btn.clicked.connect(self.show_assumption_indicators_help)
        assumptions_status_layout.addWidget(indicators_help_btn)
        
        test_selection_layout.addLayout(assumptions_status_layout)
        
        # Add outcome selector
        outcome_selector_layout = QHBoxLayout()
        outcome_selector_layout.addWidget(QLabel("Evaluate"))
        self.outcome_selector = QComboBox()
        self.outcome_selector.currentIndexChanged.connect(self.on_outcome_changed)
        outcome_selector_layout.addWidget(self.outcome_selector, 1)
        
        test_selection_layout.addLayout(outcome_selector_layout)
        
        # Add test selection group to the horizontal layout
        design_and_test_layout.addWidget(test_selection_group, 1)  # Equal stretch
        
        # Add the combined container to the top layout
        top_layout.addWidget(design_and_test_container)
        
        main_splitter.addWidget(top_section)
        
        # Bottom section: Data display and test results
        bottom_section = QSplitter(Qt.Orientation.Horizontal)
        
        # Dataset display with clickable headers
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # Add color strip legend above table
        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setSpacing(0)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a strip for each column
        self.legend_strips = {}  # Store references to update later
        
        dataset_layout.addWidget(legend_widget)
        
        self.data_table = QTableWidget()
        self.data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.data_table.horizontalHeader().setSectionsClickable(True)
        
        dataset_layout.addWidget(self.data_table)
        
        # Results display with tabs for results and assumptions
        results_group = QGroupBox("Analysis")
        results_layout = QVBoxLayout(results_group)
        
        # Create a tab widget for results
        results_tabs = QTabWidget()
        
        # Tab for Assumptions
        assumptions_tab = QWidget()
        assumptions_tab_layout = QVBoxLayout(assumptions_tab)
        
        # Replace QTextBrowser with QTreeWidget for assumptions
        self.assumptions_tree = QTreeWidget()
        self.assumptions_tree.setHeaderLabels(["Assumption", "Status"])
        self.assumptions_tree.setColumnWidth(0, 300)
        self.assumptions_tree.setColumnWidth(1, 100)
        # Remove alternating row colors
        
        assumptions_tab_layout.addWidget(self.assumptions_tree)
        
        # Tab for JSON Interpretation
        json_tab = QWidget()
        json_tab_layout = QVBoxLayout(json_tab)
        
        # Replace QTextBrowser with QTreeWidget for JSON interpretation
        self.json_tree = QTreeWidget()
        self.json_tree.setHeaderLabels(["Key", "Value"])
        self.json_tree.setColumnWidth(0, 200)
        # Remove alternating row colors
        
        json_tab_layout.addWidget(self.json_tree)
        
        # Add the tabs to the tab widget - remove the Test Results tab
        results_tabs.addTab(assumptions_tab, "Assumptions")
        results_tabs.addTab(json_tab, "Interpretation")
        
        results_layout.addWidget(results_tabs)
        
                
        # Add the dataset and results to the bottom section
        bottom_section.addWidget(dataset_group)
        bottom_section.addWidget(results_group)
        
        # Set the sizes for better visibility
        bottom_section.setSizes([600, 400])
        
        main_splitter.addWidget(bottom_section)
        
        # Adjust the main splitter sizes
        main_splitter.setSizes([400, 600])
        
        main_layout.addWidget(main_splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize the widget
        self.update_required_fields()
        
        # Set a reasonable minimum size
        self.setMinimumSize(1000, 800)
        
        # Find where the test parameters are defined, likely near the test_combo definition
        # Look for a group box or layout for test settings
        
        # Create a layout for test parameters if it doesn't exist
        self.test_params_group = QGroupBox("One-sample")
        self.test_params_layout = QFormLayout()
        self.test_params_group.setLayout(self.test_params_layout)
        
        # Add mu input field for one-sample tests
        self.mu_label = QLabel("μ ")
        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(-999999, 999999)
        self.mu_input.setValue(0)
        self.mu_input.setDecimals(1)
        self.mu_input.setVisible(False)
        self.mu_label.setVisible(False)
        
        # Connect value changed signal to track changes
        self.mu_input.valueChanged.connect(self.on_mu_value_changed)
        
        # Add to test parameters layout
        self.test_params_layout.addRow(self.mu_label, self.mu_input)
        
        outcome_selector_layout.addWidget(self.test_params_group)
        
        # Initialize test settings with default values
        self.test_settings = {
            'update_documentation': False,
            'update_hypothesis': False,
            'generate_interpretation': False
        }

        # Create a panel for correlation tests
        self.correlation_panel = QGroupBox("Correlation Variables")
        correlation_layout = QVBoxLayout(self.correlation_panel)

        # Create a visual representation of the correlation relationship
        correlation_display = QWidget()
        correlation_display_layout = QHBoxLayout(correlation_display)
        self.correlation_x_label = QLabel("X Variable: Not set")
        correlation_display_layout.addWidget(self.correlation_x_label)
        correlation_display_layout.addWidget(QLabel(" ↔ "))  # Bidirectional arrow
        self.correlation_y_label = QLabel("Y Variable: Not set")
        correlation_display_layout.addWidget(self.correlation_y_label)
        correlation_layout.addWidget(correlation_display)

        # Add a note explaining the selection mechanism
        note_label = QLabel("For correlation tests, X is your selected outcome, and Y is your first covariate.")
        note_label.setStyleSheet("color: gray; font-style: italic;")
        correlation_layout.addWidget(note_label)

        # Add a button to swap X and Y variables
        swap_button = QPushButton("Swap X and Y")
        swap_button.setIcon(load_bootstrap_icon("arrow-left-right"))
        swap_button.clicked.connect(self.swap_correlation_variables)
        correlation_layout.addWidget(swap_button)

        # Hide by default
        self.correlation_panel.setVisible(False)

        # Add to layout near the test parameters
        outcome_selector_layout.addWidget(self.correlation_panel)
    
    def set_studies_manager(self, studies_manager):
        """Set the studies manager for integration."""
        self.studies_manager = studies_manager

    def update_design_type_combo(self):
        """
        Updates the study design dropdown based on current variable assignments.
        Shows all designs but sorts valid ones (with checkmarks) to the top.
        """
        # Store current selection if possible
        current_selection = None
        if self.design_type_combo.count() > 0:
            current_data = self.design_type_combo.currentData()
            if current_data:
                current_selection = current_data
        
        # Clear the dropdown
        self.design_type_combo.blockSignals(True)
        self.design_type_combo.clear()
        
        # Get current variable assignments
        role_to_var = {}
        for role in VariableRole:
            if role != VariableRole.NONE:
                vars_with_role = [col for col, assigned_role in self.column_roles.items() if assigned_role == role]
                if vars_with_role:
                    role_to_var[role.value] = vars_with_role
        
        # Check which study designs are valid
        valid_designs = []
        invalid_designs = {}
        
        # Load icons for status indicators
        check_icon = load_bootstrap_icon("check-circle-fill", color="#43A047", size=14)
        warning_icon = load_bootstrap_icon("question-circle-fill", color="#FB8C00", size=14)
        error_icon = load_bootstrap_icon("x-circle-fill", color="#E53935", size=14)
        
        for study_type in StudyType:
            spec = STUDY_DESIGN_REGISTRY.get(study_type)
            if not spec:
                continue
            
            # Check if all required variables are assigned
            all_required_assigned = True
            missing_vars = []
            
            for var_name, requirement in spec.variable_requirements.items():
                if requirement == VariableRequirement.REQUIRED and var_name not in role_to_var:
                    all_required_assigned = False
                    missing_vars.append(var_name)
            
            # Calculate match score
            match_score = 0
            for role, requirement in spec.variable_requirements.items():
                if role in role_to_var and requirement != VariableRequirement.NOT_USED:
                    match_score += 2
                if role in role_to_var and requirement == VariableRequirement.REQUIRED:
                    match_score += 3
            
            design_entry = (study_type, spec.description, match_score, missing_vars)
            if all_required_assigned:
                valid_designs.append(design_entry)
            else:
                # Add invalid designs with a warning and reason
                error_message = f"Missing required variables: {', '.join(missing_vars)}" if missing_vars else "Not all required variables are assigned"
                invalid_designs[study_type] = (error_message, missing_vars)
        
        # Sort valid designs by match score
        valid_designs.sort(key=lambda x: x[2], reverse=True)
        
        # Add valid designs first with checkmarks
        for study_type, desc, _, _ in valid_designs:
            self.design_type_combo.addItem(check_icon, f" {study_type.display_name}", study_type)
            self.design_type_combo.setItemData(
                self.design_type_combo.count() - 1, 
                f"Compatible with current variable assignments: {desc}", 
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Add invalid designs with warning icons and error messages
        for study_type, (error_message, missing_vars) in invalid_designs.items():
            self.design_type_combo.addItem(warning_icon, f" {study_type.display_name}", study_type)
            self.design_type_combo.setItemData(
                self.design_type_combo.count() - 1,
                f"Incompatible: {error_message}",
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Restore previous selection if possible
        if current_selection:
            for i in range(self.design_type_combo.count()):
                if self.design_type_combo.itemData(i, Qt.ItemDataRole.UserRole) == current_selection:
                    self.design_type_combo.setCurrentIndex(i)
                    break
        
        self.design_type_combo.blockSignals(False)
        self.design_type_combo.update()
    
    def load_dataset_from_study(self):
        """Load available datasets from the studies manager."""
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager'):
            self.status_bar.showMessage("Could not access studies manager")
            return
        
        # Get datasets from active study
        datasets = main_window.studies_manager.get_datasets_from_active_study()
        
        if not datasets:
            self.status_bar.showMessage("No datasets available in the active study")
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
        # Clear the covariate order list
        self.covariate_order = []
        
        # Update the UI
        self.display_dataset(df)
        self.populate_variable_dropdowns()  # This replaces update_selected_variables_display()
        
        # Enable the dropdown selectors now that we have a dataset
        self.outcome_combo.setEnabled(True)
        self.group_combo.setEnabled(True)
        self.subject_id_combo.setEnabled(True)
        self.time_combo.setEnabled(True)
        self.event_combo.setEnabled(True)
        self.pair_id_combo.setEnabled(True)
        self.covariates_combo.setEnabled(True)
        self.add_covariate_button.setEnabled(True)
        
        # Connect signals for manual role changes if not already connected
        self.outcome_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.OUTCOME))
        self.group_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.GROUP))
        self.subject_id_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.SUBJECT_ID))
        self.time_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.TIME))
        self.event_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.EVENT))
        self.pair_id_combo.currentIndexChanged.connect(lambda: self.manual_role_changed(VariableRole.PAIR_ID))
        
        # Update the design dropdown to reflect available options
        self.update_design_type_combo()
        self.update_test_dropdown()
        
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
        self.update_design_type_combo()

    def update_required_fields(self):
        """Update which fields are required based on the selected design type."""
        study_type = self.design_type_combo.currentData()
        if not study_type:
            return
        
        # Get the design specification from the registry
        spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not spec:
            return
        
        # NEW: Check if current dataset is compatible with this design
        if self.current_dataframe is not None:
            dataset_analysis = analyze_dataset_structure(self.current_dataframe)
            design_is_compatible = study_type in dataset_analysis["compatible_designs"]
            
            if not design_is_compatible:
                self.status_bar.showMessage(
                    f"Warning: {study_type.display_name} design may not be compatible with your dataset structure",
                    5000  # Show for 5 seconds
                )
        
        # Update UI based on variable requirements
        requirements = spec.variable_requirements
        
        # Update styles and tooltips for each variable
        widgets = {
            "outcome": self.outcome_combo,
            "group": self.group_combo,
            "subject_id": self.subject_id_combo,
            "time": self.time_combo,
            "pair_id": self.pair_id_combo
        }
        
        for var_name, widget in widgets.items():
            req = requirements.get(var_name, VariableRequirement.NOT_USED)
            
            if req == VariableRequirement.REQUIRED:
                widget.setStyleSheet("font-weight: bold;")
                widget.setToolTip(f"Required for {study_type.display_name} design")
            elif req == VariableRequirement.OPTIONAL:
                widget.setStyleSheet("")
                widget.setToolTip(f"Optional for {study_type.display_name} design")
            else:  # NOT_USED
                widget.setStyleSheet("color: gray;")
                widget.setToolTip(f"Not typically used in {study_type.display_name} design")
        
        # Update the test dropdown with compatible tests
        self.update_test_dropdown()

        # Get the currently selected test
        current_test = self.test_combo.currentData()
        
        if current_test:
            # Get required parameters for this test
            required_params = TestExecutorFactory.get_required_parameters(current_test)
            
            # Create input fields for each required parameter
            for param_name, param_description in required_params.items():
                if param_name == "mu":
                    # Update the existing mu_input tooltip without creating a new widget
                    if hasattr(self, 'mu_input') and hasattr(self, 'mu_label'):
                        self.mu_input.setToolTip(param_description)
     
    
    def update_test_dropdown(self):
        """Filter and update the test dropdown based on current design and data type."""
        # Store current selection to restore later
        current_test = self.test_combo.currentData()
        self.test_combo.clear()
        
        # Get the current study type
        study_type = self.design_type_combo.currentData()
        if not study_type:
            return
            
        # Get compatible tests for this design
        spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not spec:
            return
            
        # Get the variables from the current roles
        outcome = next((col for col, role in self.column_roles.items() if role == VariableRole.OUTCOME), None)
        group = next((col for col, role in self.column_roles.items() if role == VariableRole.GROUP), None)
        subject_id = next((col for col, role in self.column_roles.items() if role == VariableRole.SUBJECT_ID), None)
        time = next((col for col, role in self.column_roles.items() if role == VariableRole.TIME), None)
        pair_id = next((col for col, role in self.column_roles.items() if role == VariableRole.PAIR_ID), None)
        covariates = [col for col, role in self.column_roles.items() if role == VariableRole.COVARIATE]
        
        # Get design-compatible tests
        design_compatible_tests = [test.value for test in spec.compatible_tests]
        
        # Determine outcome data type if available
        outcome_data_type = None
        data_compatible_tests = []
        if outcome and self.current_dataframe is not None:
            outcome_data_type = infer_data_type(self.current_dataframe, outcome)
            if outcome_data_type:
                data_compatible_tests = [test.value for test in get_compatible_tests_for_data_type(outcome_data_type)]
        
        # Function to check if a test is valid based on required variables
        def is_test_valid(test_key):
            """Check if the test has all required variables assigned."""
            if test_key == StatisticalTest.ONE_SAMPLE_T_TEST.value:
                return outcome is not None
            
            elif test_key in [StatisticalTest.INDEPENDENT_T_TEST.value, StatisticalTest.MANN_WHITNEY_U_TEST.value,
                        StatisticalTest.CHI_SQUARE_TEST.value, StatisticalTest.FISHERS_EXACT_TEST.value]:
                return outcome is not None and group is not None
            
            elif test_key in [StatisticalTest.PAIRED_T_TEST.value, StatisticalTest.WILCOXON_SIGNED_RANK_TEST.value]:
                return outcome is not None and subject_id is not None and time is not None
            
            elif test_key in [StatisticalTest.ONE_WAY_ANOVA.value, StatisticalTest.KRUSKAL_WALLIS_TEST.value]:
                return outcome is not None and group is not None
            
            elif test_key == StatisticalTest.REPEATED_MEASURES_ANOVA.value:
                return outcome is not None and subject_id is not None and time is not None
            
            elif test_key == StatisticalTest.MIXED_ANOVA.value:
                return outcome is not None and group is not None and subject_id is not None and time is not None
            
            elif test_key in [StatisticalTest.LINEAR_REGRESSION.value, StatisticalTest.LOGISTIC_REGRESSION.value, 
                        StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION.value, StatisticalTest.POISSON_REGRESSION.value,
                        StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION.value, StatisticalTest.ORDINAL_REGRESSION.value]:
                return outcome is not None and (group is not None or len(covariates) > 0)
            
            elif test_key == StatisticalTest.ANCOVA.value:
                return outcome is not None and group is not None and len(covariates) > 0
            
            elif test_key == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL.value:
                return outcome is not None and subject_id is not None
            
            elif test_key in [StatisticalTest.PEARSON_CORRELATION.value, StatisticalTest.SPEARMAN_CORRELATION.value, 
                        StatisticalTest.KENDALL_TAU_CORRELATION.value]:
                return outcome is not None and (group is not None or len(covariates) > 0)
            
            elif test_key == StatisticalTest.SURVIVAL_ANALYSIS.value:
                return outcome is not None and time is not None
            
            elif test_key == StatisticalTest.POINT_BISERIAL_CORRELATION.value:
                # Point-biserial specifically needs a binary variable, not just any second variable
                return outcome is not None and group is not None and (
                    # Ensure group is binary if we have a dataframe to check
                    (self.current_dataframe is not None and self.current_dataframe[group].nunique() == 2) or
                    # Or allow using binary_treatment from the test dataset
                    'binary_treatment' in self.current_dataframe.columns
                )
            # Default case - require at least an outcome
            return outcome is not None
        
        # Get missing variable reason
        def get_missing_variables(test_key):
            """Get list of missing variables for this test."""
            missing = []
            
            if not outcome:
                missing.append("outcome")
            
            if test_key in [StatisticalTest.INDEPENDENT_T_TEST.value, StatisticalTest.MANN_WHITNEY_U_TEST.value,
                        StatisticalTest.CHI_SQUARE_TEST.value, StatisticalTest.FISHERS_EXACT_TEST.value,
                        StatisticalTest.ONE_WAY_ANOVA.value, StatisticalTest.KRUSKAL_WALLIS_TEST.value]:
                if not group:
                    missing.append("group")
            
            elif test_key in [StatisticalTest.PAIRED_T_TEST.value, StatisticalTest.WILCOXON_SIGNED_RANK_TEST.value]:
                if not subject_id:
                    missing.append("subject_id")
                if not time:
                    missing.append("time")
            
            elif test_key == StatisticalTest.REPEATED_MEASURES_ANOVA.value:
                if not subject_id:
                    missing.append("subject_id")
                if not time:
                    missing.append("time")
            
            elif test_key == StatisticalTest.MIXED_ANOVA.value:
                if not group:
                    missing.append("group")
                if not subject_id:
                    missing.append("subject_id")
                if not time:
                    missing.append("time")
            
            elif test_key in [StatisticalTest.LINEAR_REGRESSION.value, StatisticalTest.LOGISTIC_REGRESSION.value, 
                        StatisticalTest.MULTINOMIAL_LOGISTIC_REGRESSION.value, StatisticalTest.POISSON_REGRESSION.value,
                        StatisticalTest.NEGATIVE_BINOMIAL_REGRESSION.value, StatisticalTest.ORDINAL_REGRESSION.value]:
                if not group and not covariates:
                    missing.append("group or covariates")
            
            elif test_key == StatisticalTest.ANCOVA.value:
                if not group:
                    missing.append("group")
                if not covariates:
                    missing.append("covariates")
            
            elif test_key == StatisticalTest.LINEAR_MIXED_EFFECTS_MODEL.value:
                if not subject_id:
                    missing.append("subject_id")
            
            elif test_key in [StatisticalTest.PEARSON_CORRELATION.value, StatisticalTest.SPEARMAN_CORRELATION.value, 
                        StatisticalTest.KENDALL_TAU_CORRELATION.value]:
                if not group and not covariates:
                    missing.append("second variable")

            elif test_key == StatisticalTest.POINT_BISERIAL_CORRELATION.value:
                if not group:
                    missing.append("binary group variable")
                elif self.current_dataframe is not None and self.current_dataframe[group].nunique() != 2:
                    missing.append("binary group variable (must have exactly 2 unique values)")
            
            elif test_key == StatisticalTest.SURVIVAL_ANALYSIS.value:
                if not time:
                    missing.append("time")
            
            return missing
        
        # Categorize tests
        fully_compatible_tests = []
        design_compatible_only = []
        data_compatible_only = []
        incompatible_tests = []
        
        # Check all tests in registry
        for test_key in TEST_REGISTRY.keys():
            test_data = TEST_REGISTRY[test_key]
            
            # Check design compatibility
            is_design_compatible = test_key in design_compatible_tests
            
            # Check data type compatibility
            is_data_compatible = True
            if outcome_data_type and data_compatible_tests:
                is_data_compatible = test_key in data_compatible_tests
            
            # Check variable compatibility
            has_required_variables = is_test_valid(test_key)
            missing_vars = get_missing_variables(test_key)
            
            # Categorize the test
            if is_design_compatible and is_data_compatible and has_required_variables:
                fully_compatible_tests.append((test_key, test_data))
            elif is_design_compatible and not is_data_compatible:
                # Design compatible but data type incompatible
                incompatible_tests.append((test_key, test_data, f"Incompatible with {outcome_data_type.value if outcome_data_type else 'unknown'} outcome"))
            elif is_design_compatible and missing_vars:
                # Design compatible but missing variables
                design_compatible_only.append((test_key, test_data, f"Missing: {', '.join(missing_vars)}"))
            elif is_data_compatible and not is_design_compatible:
                # Data compatible but not design compatible
                data_compatible_only.append((test_key, test_data, f"Not suitable for {study_type.value} design"))
            else:
                # Incompatible on multiple criteria
                incompatible_tests.append((test_key, test_data, "Incompatible with current setup"))
        
        # Use bootstrap icons for better visibility
        check_icon = load_bootstrap_icon("check-circle-fill", color="#43A047", size=14)
        question_icon = load_bootstrap_icon("question-circle-fill", color="#FB8C00", size=14)
        x_icon = load_bootstrap_icon("x-circle-fill", color="#E53935", size=14)
        
        # Add fully compatible tests first (with checkmark)
        for test_key, test_data in fully_compatible_tests:
            self.test_combo.addItem(check_icon, f" {test_data.name}", test_key)
            self.test_combo.setItemData(
                self.test_combo.count() - 1,
                f"Compatible with current study design and variables",
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Add design compatible but missing variables (with question mark)
        for test_key, test_data, reason in design_compatible_only:
            self.test_combo.addItem(question_icon, f" {test_data.name}", test_key)
            self.test_combo.setItemData(
                self.test_combo.count() - 1,
                f"Study design compatible but {reason}",
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Add data compatible but not design compatible (with question mark)
        for test_key, test_data, reason in data_compatible_only:
            self.test_combo.addItem(question_icon, f" {test_data.name}", test_key)
            self.test_combo.setItemData(
                self.test_combo.count() - 1,
                f"Data type compatible but {reason}",
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Add incompatible tests last (with X)
        for test_key, test_data, reason in incompatible_tests:
            self.test_combo.addItem(x_icon, f" {test_data.name}", test_key)
            self.test_combo.setItemData(
                self.test_combo.count() - 1,
                reason,
                Qt.ItemDataRole.ToolTipRole
            )
        
        # Restore previous selection if possible
        if current_test:
            for i in range(self.test_combo.count()):
                if self.test_combo.itemData(i) == current_test:
                    self.test_combo.setCurrentIndex(i)
                    break
        
        # Update visibility of one-sample parameters
        current_test_text = self.test_combo.currentText().lower()
        is_one_sample_test = "one_sample" in current_test_text or "one sample" in current_test_text
        
        # Make sure these attributes exist before using them
        if hasattr(self, 'mu_input') and hasattr(self, 'mu_label') and hasattr(self, 'test_params_group'):
            self.mu_input.setVisible(is_one_sample_test)
            self.mu_label.setVisible(is_one_sample_test)
            self.test_params_group.setVisible(is_one_sample_test)
    
    def validate_within_subjects_format(self, df, subject_id_column):
        """
        Verify that the data is in long format with multiple observations per subject.
        This is essential for within-subjects designs and repeated measures.
        
        Args:
            df: The DataFrame to check
            subject_id_column: The column name that might contain subject IDs
            
        Returns:
            bool: True if the data has valid within-subjects structure, False otherwise
        """
        if subject_id_column not in df.columns:
            return False
        
        # Basic validation - subject ID should have sufficient unique values
        n_unique_subjects = df[subject_id_column].nunique()
        if n_unique_subjects < 5:  # Too few subjects is suspicious
            logging.info(f"Rejecting {subject_id_column} as subject_id - too few unique values ({n_unique_subjects})")
            return False
        
        # Subject ID should have high cardinality but not be unique for every row
        cardinality_ratio = n_unique_subjects / len(df)
        if cardinality_ratio > 0.95:  # Almost every row has a unique value - likely not a subject ID
            logging.info(f"Rejecting {subject_id_column} as subject_id - cardinality too high ({cardinality_ratio:.2f})")
            return False
        if cardinality_ratio < 0.05:  # Too few unique values - likely not a subject ID
            logging.info(f"Rejecting {subject_id_column} as subject_id - cardinality too low ({cardinality_ratio:.2f})")
            return False
            
        # Count observations per subject
        counts = df[subject_id_column].value_counts()
        
        # Calculate statistics about repeated measures
        multiple_obs_subjects = (counts > 1).sum()  # Subjects with > 1 observation
        mean_obs_per_subject = counts.mean()
        max_obs_per_subject = counts.max()
        
        # For a true within-subjects design:
        # 1. A significant proportion of subjects should have multiple observations
        # 2. The average observations per subject should be > 1.5
        # 3. At least some subjects should have many observations (for time series data)
        
        # Check if enough subjects have multiple observations
        if multiple_obs_subjects < max(3, n_unique_subjects * 0.2):  # At least 20% of subjects or 3 subjects
            logging.info(f"Rejecting {subject_id_column} as subject_id - too few subjects with multiple observations ({multiple_obs_subjects})")
            return False
            
        # Check if average observations per subject is sufficient
        if mean_obs_per_subject < 1.5:  # On average, subjects should have more than 1.5 observations
            logging.info(f"Rejecting {subject_id_column} as subject_id - too few observations per subject ({mean_obs_per_subject:.2f})")
            return False
            
        # Look for evidence of time variation (if at least some columns are likely to be outcomes)
        likely_outcome_columns = []
        for col in df.columns:
            # Look for numeric columns with sufficient variance
            if (col != subject_id_column and 
                pd.api.types.is_numeric_dtype(df[col]) and
                df[col].nunique() > 5 and
                df[col].std() > 0):
                likely_outcome_columns.append(col)
        
        # If we found potential outcome columns, check for within-subject variation
        if likely_outcome_columns:
            # Sample a few potential outcome columns and check if they vary within subjects
            has_within_subject_variation = False
            
            for col in likely_outcome_columns[:3]:  # Check up to 3 columns
                # Group by subject and check if values vary within each subject
                within_subject_var = df.groupby(subject_id_column)[col].nunique()
                subjects_with_varying_values = (within_subject_var > 1).sum()
                
                # If a good number of subjects have varying values, this is likely time-varying data
                if subjects_with_varying_values >= max(3, n_unique_subjects * 0.1):
                    has_within_subject_variation = True
                    break
            
            # If we didn't find any within-subject variation in potential outcomes,
            # this might not be a true repeated measures design
            if not has_within_subject_variation and n_unique_subjects > 10:
                logging.info(f"Rejecting {subject_id_column} as subject_id - no within-subject variation in outcome variables")
                return False
        
        logging.info(f"Accepting {subject_id_column} as valid subject_id")
        # If we passed all tests, this is likely a valid within-subjects format
        return True

    def set_button_working_state(self, button, is_working=True, message=None):
        """Set a button to a working state with animation or reset it."""
        if is_working:
            # Store original text and style
            if not hasattr(button, '_original_text'):
                button._original_text = button.text()
                
            if not hasattr(button, '_original_icon'):
                button._original_icon = button.icon()
            
            # Store original style
            if not hasattr(button, '_original_style'):
                button._original_style = button.styleSheet()
            
            # Set spinner icon
            spinner_icon = load_bootstrap_icon("arrow-clockwise")
            button.setIcon(spinner_icon)
            
            # Create spinning animation with a timer
            if not hasattr(button, '_pulse_timer'):
                button._pulse_timer = QTimer(button)
                button._pulse_state = 0
                
                def update_opacity():
                    button._pulse_state = (button._pulse_state + 1) % 12
                    
                    # Rotate the spinner icon - smoother with 12 steps
                    rotation = button._pulse_state * 30  # 12 steps for 360 degree rotation (30 degrees each)
                    transform = QTransform().rotate(rotation)
                    icon = load_bootstrap_icon("arrow-clockwise")
                    pixmap = icon.pixmap(16, 16)
                    rotated_pixmap = pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
                    button.setIcon(QIcon(rotated_pixmap))
                
                button._pulse_timer.timeout.connect(update_opacity)
                button._pulse_timer.start(60)  # Update every 60ms for smoother rotation
            
            # Disable the button but keep it visible
            button.setEnabled(False)
            QApplication.processEvents()  # Update UI
        else:
            # Stop the pulse timer if it exists
            if hasattr(button, '_pulse_timer'):
                button._pulse_timer.stop()
                delattr(button, '_pulse_timer')
                button._pulse_state = 0
            
            # Restore original style
            if hasattr(button, '_original_style'):
                button.setStyleSheet(button._original_style)
            
            # Restore original icon
            if hasattr(button, '_original_icon'):
                button.setIcon(button._original_icon)
                
            # Always restore original text if we stored it
            if hasattr(button, '_original_text'):
                button.setText(button._original_text)
                
            # Re-enable the button
            button.setEnabled(True)
            QApplication.processEvents()  # Update UI

    # Add this helper method for JSON serialization
    def _sanitize_for_json(self, obj):
        """Helper method to sanitize dictionary values for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(item) for item in obj)
        elif isinstance(obj, pd.DataFrame):
            return f"DataFrame with shape {obj.shape}"
        elif hasattr(obj, '__class__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return f"Object of type {obj.__class__.__name__}"
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)

    def show_hypothesis_input_dialog(self):
        """Show dialog to optionally input a study hypothesis for better variable mapping."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Input Hypothesis (Optional)")
        
        layout = QVBoxLayout(dialog)
        
        # Explanation label
        explanation = QLabel("Enter your study hypothesis to help AI better identify variable roles and study design:")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Text input for hypothesis
        hypothesis_text = QTextEdit()
        hypothesis_text.setPlaceholderText("Example: Patients discharged against medical advice have significantly higher 30-day readmission rates compared to planned discharges, independent of comorbidity burden.")
        hypothesis_text.setMinimumHeight(100)
        layout.addWidget(hypothesis_text)
        
        # Look for existing hypotheses in the study
        existing_hypotheses = []
        
        main_window = self.window()
        if hasattr(main_window, 'studies_manager') and main_window.studies_manager:
            existing_hypotheses = main_window.studies_manager.get_study_hypotheses()
        
        # Add existing hypotheses dropdown if available
        if existing_hypotheses:
            existing_label = QLabel("Or select an existing hypothesis:")
            layout.addWidget(existing_label)
            
            hypotheses_combo = QComboBox()
            hypotheses_combo.addItem("-- Select Existing Hypothesis --", None)
            for hypothesis in existing_hypotheses:
                title = hypothesis.get('title', '')
                if title:
                    hypotheses_combo.addItem(title, hypothesis)
            
            # Connect selection to text field
            def on_hypothesis_selected(index):
                if index > 0:  # Skip the first item (placeholder)
                    selected = hypotheses_combo.itemData(index)
                    if selected:
                        alt_text = selected.get('alternative_hypothesis', '')
                        null_text = selected.get('null_hypothesis', '')
                        full_text = f"{selected.get('title', '')}\n\nH1: {alt_text}\nH0: {null_text}"
                        hypothesis_text.setText(full_text)
            
            hypotheses_combo.currentIndexChanged.connect(on_hypothesis_selected)
            layout.addWidget(hypotheses_combo)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Add a "Skip" button that sets an empty result and accepts
        skip_button = button_box.addButton("Skip (Auto-detect)", QDialogButtonBox.ButtonRole.ActionRole)
        skip_button.clicked.connect(lambda: dialog.done(2))  # Custom return code for skip
        
        layout.addWidget(button_box)
        
        # Set a reasonable size
        dialog.setMinimumWidth(500)
        
        # Run the dialog
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            return hypothesis_text.toPlainText().strip()
        elif result == 2:  # Skip code
            return ""
        else:
            return None
    
    @asyncSlot()
    async def build_model(self, direct_hypothesis=None):
        """
        Automatically build a statistical model based on the data.
        
        Args:
            direct_hypothesis: Optional hypothesis text to use directly without showing dialog
        """
        # Set button to working state
        self.set_button_working_state(self.auto_assign_button, True)
        
        # Use direct hypothesis if provided, otherwise show dialog
        hypothesis = None
        if direct_hypothesis is not None:
            hypothesis = direct_hypothesis
        else:
            hypothesis = self.show_hypothesis_input_dialog()
        
        # If user cancelled, abort
        if hypothesis is None:
            self.set_button_working_state(self.auto_assign_button, False)
            return
        
        try:
            if self.current_dataframe is None or self.current_dataframe.empty:
                self.set_button_working_state(self.auto_assign_button, False)
                self.status_bar.showMessage("No dataset selected")
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
                    "unique_ratio": float(df[col].nunique() / len(df)) if len(df) > 0 else 0,
                    "missing_ratio": float(df[col].isna().mean()),
                    "is_categorical": pd.api.types.is_categorical_dtype(df[col]) or 
                                     (not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < min(30, len(df) * 0.2)),
                    "is_binary": df[col].nunique() == 2
                }
                
                # Enhanced sample values with more context
                sample_values = df[col].dropna().head(5).tolist()
                # If numeric, also provide distribution characteristics
                if pd.api.types.is_numeric_dtype(df[col]):
                    q1 = float(df[col].quantile(0.25))
                    q3 = float(df[col].quantile(0.75))
                    stats["median"] = float(df[col].median())
                    stats["q1"] = q1
                    stats["q3"] = q3
                    stats["iqr"] = q3 - q1
                    stats["is_likely_continuous"] = unique_values > min(20, len(df) * 0.1)
                    stats["is_count_data"] = (df[col] >= 0).all() and all(float(x).is_integer() for x in df[col].dropna() if pd.notna(x))
                
                columns_info.append({
                    "name": col,
                    "data_type": data_type,
                    "unique_values": unique_values,
                    "is_numeric": is_numeric,
                    "statistics": stats,
                    "sample_values": sample_values
                })
            
            # Update status bar
            self.status_bar.showMessage("Analyzing dataset structure with AI. Please wait...")
            
            # STEP 1: Enhanced validation of subject_id and time variables
            # Identify potential subject_id columns and validate if they are used for repeated measures
            
            # First, identify potential subject ID columns based on statistical properties only
            potential_subject_ids = []
            
            # Look for columns that have properties typical of ID variables
            for col in df.columns:
                # Only consider columns with reasonable cardinality for subject IDs
                # (not too low, not too high relative to dataset size)
                unique_count = df[col].nunique()
                cardinality_ratio = unique_count / len(df)
                
                # Statistical criteria only - no term-based heuristics
                if (0.05 < cardinality_ratio < 0.95 and  # Not too few, not too many unique values
                    unique_count >= 5 and                # At least 5 unique values
                    df[col].duplicated().any()):         # Has some duplicates (potential for repeated measures)
                    potential_subject_ids.append(col)
            
            # Log potential subject IDs for debugging
            logging.info(f"Potential subject IDs identified by statistical criteria: {potential_subject_ids}")
            
            # Improved validation checking if any potential subject_id has repeated observations
            valid_subject_id = None
            validation_results = {}
            
            for subject_id_col in potential_subject_ids:
                validation_result = self.validate_within_subjects_format(df, subject_id_col)
                validation_results[subject_id_col] = validation_result
                
                if validation_result and valid_subject_id is None:
                    valid_subject_id = subject_id_col
            
            valid_subject_id_exists = valid_subject_id is not None
            
            # Enhanced time variable detection based solely on statistical properties
            potential_time_vars = []
            
            # Look only for columns with statistical properties typical of time variables
            for col in df.columns:
                # Time variables typically have few unique values and are orderable
                unique_count = df[col].nunique()
                
                # Statistical criteria for potential time variables
                if (unique_count <= 10 and unique_count >= 2 and  # Few unique values (2-10 is typical for time points)
                    (pd.api.types.is_numeric_dtype(df[col]) or     # Numeric or
                     pd.api.types.is_datetime64_dtype(df[col]))):  # Date/time type
                    potential_time_vars.append(col)
                        
            # Log potential time variables for debugging
            logging.info(f"Potential time variables identified by statistical criteria: {potential_time_vars}")
            
            # Further validate potential time variables
            valid_time_var = None
            if valid_subject_id_exists and potential_time_vars:
                for time_col in potential_time_vars:
                    # Check if each subject has multiple time points
                    if df.groupby(valid_subject_id)[time_col].nunique().mean() > 1:
                        valid_time_var = time_col
                        logging.info(f"Validated time variable: {time_col}")
                    break
            
            # Get the column mapping
            column_mapping = get_column_mapping(df, include_actual=False)

            # ============ FIRST LLM CALL: MAP VARIABLES TO ROLES ============
            
            # Add hypothesis context if provided
            hypothesis_context = ""
            if hypothesis:
                hypothesis_context = f"""
                Study Hypothesis:
                {hypothesis}
                
                Given this hypothesis, identify the relevant variables that would be needed to test it. 
                Be sure to correctly identify outcome, group/treatment, and other variables based on their role in the hypothesis.
                """
            
            # Improved prompt for variable role identification
            variable_roles_prompt = f"""
            I need to analyze this dataset to understand the roles of each variable in a scientific or statistical context.

            {hypothesis_context}

            Dataset Information:
            - Sample Size: {len(df)} observations
            - Potential Subject ID candidates based on statistical properties: {potential_subject_ids if potential_subject_ids else "None identified"}
            - Potential Time variable candidates based on statistical properties: {potential_time_vars if potential_time_vars else "None identified"}
            
            Column Encoding:
            - 'A' represents uppercase letters
            - 'a' represents lowercase letters
            - 'N' represents numeric digits
            - Special patterns are used for dates, phone numbers, etc.
            - NULL represents missing values            

            Detailed column statistics:
            {json.dumps(self._sanitize_for_json(columns_info), indent=2)}

            Available columns in the dataset:
            {', '.join(df.columns.tolist())}
            
            Possible roles for variables:
            - OUTCOME: The primary outcome or dependent variable (e.g., blood pressure, depression score)
            - GROUP: The grouping or independent variable (e.g., treatment/placebo, gender, disease status)
            - COVARIATE: A confounding/control variable to adjust for (e.g., age, weight, baseline measurements)
            - SUBJECT_ID: A unique identifier for individual participants
            - TIME: A time variable for repeated measures (e.g., visit number, days since baseline)
            - PAIR_ID: An identifier for matched pairs in paired designs
            - EVENT: For survival/time-to-event analysis, indicates whether the event occurred (usually binary 0/1)
            - NONE: Variable not relevant for the current analysis
            
            For survival analysis, please assign:
            - Time-to-event variable as OUTCOME
            - Binary event indicator (0=censored, 1=event occurred) as EVENT
            - Treatment/comparison group as GROUP

            In your response, please give each column one of the roles above. 

            Prioritize the provided 'Study Hypothesis' (if available) when assigning roles. For example, if the hypothesis mentions a specific outcome (e.g., 'outcome at month 9') or time point, ensure the corresponding variable(s) are assigned the OUTCOME and TIME roles correctly, even if other variables look statistically similar. Use statistical properties and patterns (like repeated measures, uniqueness, distributions) as secondary confirmation. 

            Be objective and focus on statistical properties, NOT on variable names. Consider patterns of repeated measurements, uniqueness, and data distributions.

            Respond with a JSON object in this format:
            {{
              "column_roles": {{
                "patient_id": "SUBJECT_ID",
                "treatment_group": "GROUP",
                "blood_pressure": "OUTCOME",
                "visit_number": "TIME",
                "age": "COVARIATE",
                "gender": "COVARIATE",
                "event_status": "EVENT",
                "notes": "NONE"
              }},
              "explanation": "Brief explanation of your reasoning"
            }}

            Respond with a valid JSON object that includes all columns from the dataset. Do not include any explanation text outside the JSON object.
            """
            
            # Use the LLM to determine variable roles
            response = await call_llm_async(variable_roles_prompt)
            # Log the raw LLM response for debugging
            logging.info("LLM RAW RESPONSE:")
            for line in response.split('\n'):
                logging.info(line)
            
            # Get the column roles from the LLM's response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            try:
                response_json = json.loads(json_str)
                column_roles = response_json.get("column_roles", {})
                explanation = response_json.get("explanation", "No explanation provided")
                
                # Log the parsed column roles
                logging.info("PARSED COLUMN ROLES:")
                for col, role in column_roles.items():
                    logging.info(f"  {col}: {role}")
                    
                # Store the LLM-assigned subject_id for later verification
                llm_subject_id = next((col for col, role in column_roles.items() if role == "SUBJECT_ID"), None)
                logging.info(f"LLM assigned subject_id: {llm_subject_id}")
                
                # Check for mixed-up assignments - if the LLM identified a column as OUTCOME but our statistics
                # suggest it might be a SUBJECT_ID, prioritize the LLM's OUTCOME assignment
                for col, role in column_roles.items():
                    if role == "OUTCOME" and col in potential_subject_ids:
                        logging.info(f"Column {col} was statistically identified as potential subject_id but LLM assigned as OUTCOME. Respecting LLM decision.")
                        # Remove from potential_subject_ids to prevent reassignment
                        if col in potential_subject_ids:
                            potential_subject_ids.remove(col)
                
                # IMPORTANT: Check and correct specific variable confusions 
                # If a column contains "outcome" in its name, it should probably be an OUTCOME
                for col in df.columns:
                    if "outcome" in col.lower() and col in column_roles and column_roles[col] != "OUTCOME":
                        logging.info(f"Column {col} has 'outcome' in its name but was assigned as {column_roles[col]}. Fixing to OUTCOME.")
                        column_roles[col] = "OUTCOME"
                
                # If the LLM correctly identified a subject_id, validate that assignment
                # instead of potentially overriding it with statistical validation
                if llm_subject_id:
                    logging.info(f"Validating LLM-assigned subject_id: {llm_subject_id}")
                    if self.validate_within_subjects_format(df, llm_subject_id):
                        logging.info(f"LLM subject_id {llm_subject_id} validated successfully.")
                        # Remove from potential_subject_ids to prevent reassignment
                        if llm_subject_id in potential_subject_ids:
                            potential_subject_ids.remove(llm_subject_id)
                    else:
                        logging.info(f"LLM subject_id {llm_subject_id} validation failed. Will use statistical approach.")
                        llm_subject_id = None
                
            except json.JSONDecodeError:
                # If JSON parsing fails, use an empty dict and log the error
                column_roles = {}
                explanation = "Error parsing LLM response."
                logging.error(f"Failed to parse JSON from LLM response: {response}")
            
            # ============ SECOND LLM CALL: IDENTIFY STUDY DESIGN ============
            
            # Extract the assigned roles for the second LLM call
            outcomes = [col for col, role in column_roles.items() if role == "OUTCOME"]
            groups = [col for col, role in column_roles.items() if role == "GROUP"]
            subject_ids = [col for col, role in column_roles.items() if role == "SUBJECT_ID"]
            times = [col for col, role in column_roles.items() if role == "TIME"]
            events = [col for col, role in column_roles.items() if role == "EVENT"]
            covariates = [col for col, role in column_roles.items() if role == "COVARIATE"]
            
            # Add hypothesis to the study design prompt if provided
            study_design_hypothesis = ""
            if hypothesis:
                study_design_hypothesis = f"""
                Study Hypothesis:
                {hypothesis}
                
                Based on this hypothesis, identify the appropriate study design that would be needed to test it.
                """
                
            # Create a prompt for study design identification
            study_design_prompt = f"""
            Based on the variable role assignments in a dataset, I need you to identify the most appropriate study design. 
            
            {study_design_hypothesis}
            
            Dataset Information:
            - Total observations: {len(df)}
            - Number of unique subjects: {df[subject_ids[0]].nunique() if subject_ids else "N/A"}
            
            Variable Role Assignments:
            - Outcome variables: {outcomes if outcomes else "None"}
            - Group variables: {groups if groups else "None"}
            - Subject ID variables: {subject_ids if subject_ids else "None"}
            - Time variables: {times if times else "None"}
            - Event variables: {events if events else "None"}
            - Covariates: {covariates if covariates else "None"}
            
            Additional context:
            {f"- Group variable values: {sorted(df[groups[0]].unique().tolist())}" if groups else ""}
            {f"- Time variable values: {sorted(df[times[0]].unique().tolist())}" if times else ""}
            {f"- Multiple observations per subject: {df.groupby(subject_ids[0]).size().mean() > 1 if subject_ids else False}" if subject_ids else ""}
            
            Please analyze these variable roles and identify the most appropriate study design. Consider each design type equally and do not bias toward any particular design. Be objective based solely on the statistical properties and role assignments.
            
            Possible study designs to consider:
            - one_sample: Single group, no comparison
            - between_subjects: Different subjects in each group
            - within_subjects: Same subjects measured at different times
            - mixed: Both between and within factors
            - cross_over: Subjects receive multiple treatments sequentially
            - survival_analysis: Time-to-event analysis
            
            Respond with a JSON object in this format:
            {{
              "design": "one of the designs listed above",
              "recommendation": "Brief name of the recommendation",
              "explanation": "Detailed explanation of why this design is appropriate",
              "confidence": "high/medium/low"
            }}
            
            Important: Your decision should be based ONLY on the statistical properties and variable assignments, not on any assumed research questions. Focus on what the data structure indicates.
            """
            
            # Call LLM to identify study design
            design_response = await call_llm_async(study_design_prompt)
            
            # Parse the JSON response for study design
            design_json_match = re.search(r'({.*})', design_response, re.DOTALL)
            if design_json_match:
                try:
                    design_info = json.loads(design_json_match.group(1))
                    design_recommendation = design_info.get("recommendation", "Unknown design")
                    design_explanation = design_info.get("explanation", "")
                    design_confidence = design_info.get("confidence", "low")
                    recommended_design = design_info.get("design", "one_sample")
                    
                    logging.info(f"LLM identified study design: {recommended_design} (confidence: {design_confidence})")
                except json.JSONDecodeError:
                    # Fallback to system determination if parsing fails
                    logging.error("Failed to parse study design from LLM response, using system determination")
                    design_info = self.determine_possible_study_designs(df, column_roles)
                    design_recommendation = design_info["recommendation"]
                    design_explanation = design_info["explanation"]
                    design_confidence = design_info["confidence"]
                    recommended_design = design_info["design"]
            else:
                # Fallback to system determination if no JSON found
                logging.error("No JSON found in LLM study design response, using system determination")
                design_info = self.determine_possible_study_designs(df, column_roles)
                design_recommendation = design_info["recommendation"]
                design_explanation = design_info["explanation"]
                design_confidence = design_info["confidence"]
                recommended_design = design_info["design"]
            
            # Use design_confidence for confidence_level
            confidence_level = design_confidence
            
            # Map string design names directly to StudyType enum values from the registry
            design_type_map = {study_type.value: study_type for study_type in STUDY_DESIGN_REGISTRY}
            
            # Apply the model directly without showing a dialog
            
            # Determine the StudyType from the string
            selected_study_type = None
            if recommended_design in design_type_map:
                selected_study_type = design_type_map[recommended_design]
            
            # Apply the model
            if selected_study_type:
                # Find and select the appropriate study type in the combo box
                for i in range(self.design_type_combo.count()):
                    study_type = self.design_type_combo.itemData(i)
                    if study_type == selected_study_type:
                        self.design_type_combo.setCurrentIndex(i)
                        break
            
            # Reset all roles first
            for col in self.column_roles:
                self.column_roles[col] = VariableRole.NONE
            
            # Apply the LLM's assignments directly
            # Log all final assignments for debugging
            logging.info("APPLYING LLM VARIABLE ASSIGNMENTS:")
            for col, role_str in column_roles.items():
                if col in df.columns:
                    try:
                        # Convert string role to enum
                        role = VariableRole[role_str.upper()]
                        self.column_roles[col] = role
                        logging.info(f"  Applied {col}: {role_str}")
                    except KeyError:
                        # If the role string doesn't match an enum value, set to NONE
                        self.column_roles[col] = VariableRole.NONE
                        logging.info(f"  Invalid role {role_str} for {col}, set to NONE")
            
            # Update the UI
            self.populate_variable_dropdowns()
            self.update_test_dropdown()
            
            # Update the outcome selector with all assigned outcome variables
            self.outcome_selector.clear()
            outcomes = [col for col, role in self.column_roles.items() 
                      if role == VariableRole.OUTCOME]
            for outcome in outcomes:
                self.outcome_selector.addItem(outcome)
            
            # Select the first outcome if available
            if self.outcome_selector.count() > 0:
                self.outcome_selector.setCurrentIndex(0)
            
            # Show whether a hypothesis was used
            hypothesis_used = " (with hypothesis)" if hypothesis else ""
            self.status_bar.showMessage(f"Model built: {recommended_design.replace('_', ' ').title()} design with {len(outcomes) + len(groups) + len(subject_ids) + len(times) + len(events) + len(covariates)} assigned variables{hypothesis_used}")
        
        except Exception as e:
            # Handle exceptions
            self.set_button_working_state(self.auto_assign_button, False)
            self.status_bar.showMessage(f"Error building model: {str(e)}")
            logging.error(f"Error in build_model: {str(e)}", exc_info=True)
        
        # Reset button state
        self.set_button_working_state(self.auto_assign_button, False)
    
    def update_design_description(self):
        """Update the design description based on the selected study type."""
        study_type = self.design_type_combo.currentData()
        if study_type:
            # Get the specification from the registry
            spec = STUDY_DESIGN_REGISTRY.get(study_type)
            if spec:
                self.design_description.setText(spec.description)
            else:
                self.design_description.setText("")

    def show_design_info(self):
        """Show information about the selected study design in the status bar."""
        study_type = self.design_type_combo.currentData()
        if study_type:
            spec = STUDY_DESIGN_REGISTRY.get(study_type)
            if spec:
                # Just show basic info in status bar instead of dialog
                required_vars = ", ".join(spec.get_required_variables())
                self.status_bar.showMessage(
                    f"{study_type.display_name}: {spec.description} | Required: {required_vars}"
                )

    def check_statistical_assumptions(self, df, outcome, group, subject_id, time, covariates):
        """
        Perform comprehensive statistical assumption checks for better test selection.
        Returns a dictionary of diagnostic results.
        """
        diagnostics = {
            "sample_size": {
                "overall": len(df),
                "after_na_drop": 0,
                "min_recommended": 25,
                "sufficient": True,
                "message": ""
            },
            "normality": {
                "outcome": {
                    "test_name": "Shapiro-Wilk",
                    "test_statistic": None,
                    "p_value": None,
                    "normal": None,
                    "message": ""
                },
                "by_group": {}
            },
            "group_balance": {
                "balanced": True,
                "min_group_size": 5,
                "group_counts": {},
                "smallest_group": None,
                "largest_group": None,
                "ratio": None,
                "message": ""
            },
            "variance_homogeneity": {
                "test_name": "Levene's Test",
                "test_statistic": None,
                "p_value": None,
                "equal_variance": None,
                "message": ""
            },
            "data_type": {
                "outcome": None,
                "is_binary": False,
                "is_count": False,
                "is_categorical": False,
                "is_ordinal": False,
                "is_continuous": False,
                "message": ""
            },
            "outliers": {
                "detected": False,
                "count": 0,
                "indices": [],
                "message": ""
            },
            "multicollinearity": {
                "detected": False,
                "vif_values": {},
                "message": ""
            },
            "warnings": []
        }
        
        # Create filtered dataset (drop NA in analysis columns)
        analysis_cols = [col for col in [outcome, group, subject_id, time] if col is not None]
        analysis_cols.extend(covariates)
        
        if analysis_cols:
            df_filtered = df.dropna(subset=analysis_cols)
        else:
            df_filtered = df
        
        # 1. Sample Size Checks
        clean_sample_size = len(df_filtered)
        diagnostics["sample_size"]["after_na_drop"] = clean_sample_size
        
        if clean_sample_size < diagnostics["sample_size"]["min_recommended"]:
            diagnostics["sample_size"]["sufficient"] = False
            diagnostics["sample_size"]["message"] = (
                f"Sample size ({clean_sample_size}) is below recommended minimum of "
                f"{diagnostics['sample_size']['min_recommended']}. Central Limit Theorem may not apply, "
                f"making parametric test results less reliable."
            )
            diagnostics["warnings"].append(diagnostics["sample_size"]["message"])
        
        # 2. Determine data type of outcome variable
        if outcome and outcome in df_filtered.columns:
            # Detect data type
            if pd.api.types.is_numeric_dtype(df_filtered[outcome]):
                # Check if binary (only 0-1 or two unique values)
                unique_vals = df_filtered[outcome].nunique()
                if unique_vals == 2:
                    diagnostics["data_type"]["is_binary"] = True
                    diagnostics["data_type"]["outcome"] = "binary"
                # Check if count data (only non-negative integers)
                elif df_filtered[outcome].dropna().apply(lambda x: x >= 0 and float(x).is_integer()).all():
                    diagnostics["data_type"]["is_count"] = True
                    diagnostics["data_type"]["outcome"] = "count"
                else:
                    # Check if potentially ordinal (small number of unique values)
                    if unique_vals <= 7:
                        diagnostics["data_type"]["is_ordinal"] = True
                        diagnostics["data_type"]["outcome"] = "ordinal"
                    else:
                        diagnostics["data_type"]["is_continuous"] = True
                        diagnostics["data_type"]["outcome"] = "continuous"
            else:
                # Categorical data
                diagnostics["data_type"]["is_categorical"] = True
                diagnostics["data_type"]["outcome"] = "categorical"
        
        # 3. Normality checks (for outcome)
        if outcome and outcome in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[outcome]):
            # Only perform normality test for continuous data
            if diagnostics["data_type"]["is_continuous"]:
                # Remove any remaining NaNs specifically in outcome
                outcome_data = df_filtered[outcome].dropna()
                
                if len(outcome_data) >= 3:  # Shapiro-Wilk requires at least 3 observations
                    try:
                        stat, p_val = stats.shapiro(outcome_data)
                        is_normal = p_val > 0.05  # Standard threshold, p > 0.05 means we can't reject normality
                        
                        diagnostics["normality"]["outcome"]["test_statistic"] = stat
                        diagnostics["normality"]["outcome"]["p_value"] = p_val
                        diagnostics["normality"]["outcome"]["normal"] = is_normal
                        
                        if not is_normal:
                            msg = (
                                f"Outcome variable '{outcome}' fails normality test (Shapiro-Wilk p={p_val:.4f}). "
                                f"Consider a non-parametric test or data transformation."
                            )
                            diagnostics["normality"]["outcome"]["message"] = msg
                            
                            # Add to warnings only if sample size is small (when normality matters more)
                            if clean_sample_size < 30:
                                diagnostics["warnings"].append(msg + " Since sample size is small, this is particularly important.")
                            else:
                                diagnostics["warnings"].append(
                                    f"Outcome variable '{outcome}' is not normally distributed. However, with sample size > 30, "
                                    f"the Central Limit Theorem suggests that parametric tests can still be used."
                                )
                    except Exception as e:
                        diagnostics["normality"]["outcome"]["message"] = f"Could not perform normality test: {str(e)}"
            else:
                # Skip normality testing for non-continuous data
                diagnostics["normality"]["outcome"]["message"] = f"Normality test not applicable for {diagnostics['data_type']['outcome']} data"
        
        # 4. Outlier detection (for continuous outcome)
        if outcome and outcome in df_filtered.columns and diagnostics["data_type"]["is_continuous"]:
            try:
                # Use IQR method for outlier detection
                outcome_data = df_filtered[outcome].dropna()
                q1 = outcome_data.quantile(0.25)
                q3 = outcome_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = outcome_data[(outcome_data < lower_bound) | (outcome_data > upper_bound)]
                
                if not outliers.empty:
                    diagnostics["outliers"]["detected"] = True
                    diagnostics["outliers"]["count"] = len(outliers)
                    diagnostics["outliers"]["indices"] = outliers.index.tolist()
                    
                    # Only add warning if significant number of outliers
                    if len(outliers) / len(outcome_data) > 0.05:  # If more than 5% are outliers
                        msg = f"Detected {len(outliers)} outliers ({100*len(outliers)/len(outcome_data):.1f}% of data) in outcome variable '{outcome}', which may affect test validity."
                        diagnostics["outliers"]["message"] = msg
                        diagnostics["warnings"].append(msg)
            except Exception as e:
                diagnostics["outliers"]["message"] = f"Could not perform outlier detection: {str(e)}"
        
        # 5. Group balance (if we have a group variable)
        if group and group in df_filtered.columns:
            group_counts = df_filtered[group].value_counts().to_dict()
            diagnostics["group_balance"]["group_counts"] = group_counts
            
            if group_counts:
                min_group = min(group_counts.items(), key=lambda x: x[1])
                max_group = max(group_counts.items(), key=lambda x: x[1])
                
                diagnostics["group_balance"]["smallest_group"] = {
                    "name": str(min_group[0]),
                    "count": min_group[1]
                }
                diagnostics["group_balance"]["largest_group"] = {
                    "name": str(max_group[0]),
                    "count": max_group[1]
                }
                
                # Check if any group is too small
                if min_group[1] < diagnostics["group_balance"]["min_group_size"]:
                    diagnostics["group_balance"]["balanced"] = False
                    msg = (
                        f"Group '{min_group[0]}' has only {min_group[1]} observations, "
                        f"below the recommended minimum of {diagnostics['group_balance']['min_group_size']}."
                    )
                    diagnostics["group_balance"]["message"] = msg
                    diagnostics["warnings"].append(msg)
                
                # Check group size ratio
                if min_group[1] > 0:  # Avoid division by zero
                    ratio = max_group[1] / min_group[1]
                    diagnostics["group_balance"]["ratio"] = ratio
                    
                    if ratio > 3:
                        msg = f"Groups are highly imbalanced (ratio {ratio:.1f}:1), which may affect test power and interpretation."
                        if not diagnostics["group_balance"]["message"]:
                            diagnostics["group_balance"]["message"] = msg
                        else:
                            diagnostics["group_balance"]["message"] += " " + msg
                        diagnostics["warnings"].append(msg)
                
                # Check for normality within each group (important for parametric tests)
                if outcome and outcome in df_filtered.columns and diagnostics["data_type"]["is_continuous"]:
                    diagnostics["normality"]["by_group"] = {}
                    
                    for group_val in group_counts.keys():
                        group_data = df_filtered[df_filtered[group] == group_val][outcome].dropna()
                        
                        if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
                            try:
                                stat, p_val = stats.shapiro(group_data)
                                is_normal = p_val > 0.05
                                
                                group_key = str(group_val)
                                diagnostics["normality"]["by_group"][group_key] = {
                                    "test_statistic": stat,
                                    "p_value": p_val,
                                    "normal": is_normal,
                                    "n": len(group_data),
                                    "message": ""
                                }
                                
                                if not is_normal:
                                    msg = f"Group '{group_val}' fails normality test (p={p_val:.4f})"
                                    diagnostics["normality"]["by_group"][group_key]["message"] = msg
                                    
                                    # Only warn for non-normality in groups if sample size is small
                                    if len(group_data) < 30:
                                        diagnostics["warnings"].append(
                                            f"{msg}. With small sample size (n={len(group_data)}), "
                                            f"this may affect test validity."
                                        )
                            except Exception as e:
                                diagnostics["normality"]["by_group"][str(group_val)] = {
                                    "message": f"Could not test normality: {str(e)}"
                                }
                
                # 6. Homogeneity of variance (for continuous outcome with groups)
                if (outcome and group and outcome in df_filtered.columns and group in df_filtered.columns 
                        and diagnostics["data_type"]["is_continuous"] and len(group_counts) >= 2):
                    try:
                        # Get the data for each group
                        group_data = []
                        for group_val in group_counts.keys():
                            data = df_filtered[df_filtered[group] == group_val][outcome].dropna()
                            if len(data) > 0:  # Only include non-empty groups
                                group_data.append(data)
                        
                        if len(group_data) >= 2:  # Need at least 2 groups
                            # Levene's test for homogeneity of variance
                            stat, p_val = stats.levene(*group_data)
                            equal_variance = p_val > 0.05
                            
                            diagnostics["variance_homogeneity"]["test_statistic"] = stat
                            diagnostics["variance_homogeneity"]["p_value"] = p_val
                            diagnostics["variance_homogeneity"]["equal_variance"] = equal_variance
                            
                            if not equal_variance:
                                msg = f"Unequal variances detected between groups (Levene's test p={p_val:.4f}). Consider using Welch's correction or non-parametric alternatives."
                                diagnostics["variance_homogeneity"]["message"] = msg
                                diagnostics["warnings"].append(msg)
                    except Exception as e:
                        diagnostics["variance_homogeneity"]["message"] = f"Could not test homogeneity of variance: {str(e)}"
        
        # 7. Multicollinearity check (for multiple covariates)
        if covariates and len(covariates) > 1:
            try:
                # Filter to only numeric covariates
                numeric_covs = [cov for cov in covariates if cov in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[cov])]
                
                if len(numeric_covs) > 1:
                    # Check for high correlations between covariates
                    cov_df = df_filtered[numeric_covs].copy()
                    corr_matrix = cov_df.corr()
                    
                    # Check for correlations > 0.7 (common threshold for multicollinearity concern)
                    high_correlations = False
                    for i in range(len(numeric_covs)):
                        for j in range(i+1, len(numeric_covs)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                high_correlations = True
                                msg = f"High correlation ({abs(corr_matrix.iloc[i, j]):.2f}) detected between covariates '{numeric_covs[i]}' and '{numeric_covs[j]}', which may affect regression results."
                                if not diagnostics["multicollinearity"]["message"]:
                                    diagnostics["multicollinearity"]["message"] = msg
                                else:
                                    diagnostics["multicollinearity"]["message"] += " " + msg
                                diagnostics["warnings"].append(msg)
                    
                    diagnostics["multicollinearity"]["detected"] = high_correlations
            except Exception as e:
                diagnostics["multicollinearity"]["message"] = f"Could not check for multicollinearity: {str(e)}"
        
        # 8. Central Limit Theorem warning (general guidance)
        if clean_sample_size < 30:
            clt_warning = (
                "Sample size is below 30, where the Central Limit Theorem may not fully apply. "
                "Test results should be interpreted with caution. Consider non-parametric alternatives."
            )
            if clt_warning not in diagnostics["warnings"]:
                diagnostics["warnings"].append(clt_warning)
        
        # 9. Extremely small sample warning
        if clean_sample_size < 10:
            extreme_warning = (
                "CRITICAL: Sample size is extremely small (n < 10). "
                "Statistical tests may have very low power and results should be considered preliminary. "
                "Consider descriptive statistics or non-parametric approaches."
            )
            diagnostics["warnings"].append(extreme_warning)
        
        return diagnostics

    @asyncSlot()
    async def auto_select_test(self, for_outcome=None):
        """Automatically select the appropriate statistical test using the robust TestSelectionEngine."""
        # Set button to working state
        self.set_button_working_state(self.auto_select_button, True)
        
        try:
            # Get recommendations using the engine
            recommendations = self.get_test_recommendations()
            if not recommendations:
                # Error already shown by get_test_recommendations
                self.set_button_working_state(self.auto_select_button, False)
                return
            
            # Get the selected variables for diagnostics
            outcome = next((col for col, role in self.column_roles.items() 
                        if role == VariableRole.OUTCOME), None)
            group = next((col for col, role in self.column_roles.items() 
                        if role == VariableRole.GROUP), None)
            covariates = [col for col, role in self.column_roles.items() 
                        if role == VariableRole.COVARIATE]
            subject_id = next((col for col, role in self.column_roles.items() 
                        if role == VariableRole.SUBJECT_ID), None)
            time = next((col for col, role in self.column_roles.items() 
                        if role == VariableRole.TIME), None)
            
            # Perform statistical assumption checks (for updating UI indicators)
            diagnostics = self.check_statistical_assumptions(
                self.current_dataframe, outcome, group, subject_id, time, covariates
            )
            
            # Update the visual indicators to reflect the diagnostics
            self.update_assumption_indicators(diagnostics)
            
            # Get info from top recommendation
            if len(recommendations["top_recommendations"]) > 0:
                top_test = recommendations["top_recommendations"][0]
                test_key = top_test["test"]
                test_name = top_test["name"]
                warnings = top_test["warnings"]
                
                # Update the status bar with test selection
                self.status_bar.showMessage(f"Selected test: {test_name}")
                
                # Look for the selected test in the dropdown
                for i in range(self.test_combo.count()):
                    if self.test_combo.itemData(i) == test_key:
                        self.test_combo.setCurrentIndex(i)
                        break
                    
                self.status_bar.showMessage(f"Recommended test: {test_name}")
        except Exception as e:
            self.status_bar.showMessage(f"Auto-selection failed: {str(e)}")
            print(f"Error in auto_select_test: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset button state when done
            self.set_button_working_state(self.auto_select_button, False)

    @asyncSlot()
    async def generate_hypothesis_for_test(self, outcome, group=None, subject_id=None, time=None, test_name=None, study_type=None, hypothesis_id=None):
        """Generate a hypothesis using LLM based on the current test setup and test results.
        
        Args:
            outcome: The outcome variable name
            group: The group variable name (optional)
            subject_id: The subject ID variable name (optional)
            time: The time variable name (optional)
            test_name: The name of the test (optional, uses current test if not provided)
            study_type: The study design type (optional, uses current design if not provided)
            hypothesis_id: Direct ID of hypothesis to update (optional, skips matching if provided)
            
        Returns:
            String: The hypothesis ID if successful, None otherwise
        """
        try:
            if self.current_dataframe is None or self.current_dataframe.empty:
                return None
                
            df = self.current_dataframe
            
            # Get test and design details
            test_name = test_name or self.test_combo.currentText()
            study_type_val = study_type.value if study_type else self.design_type_combo.currentData().value
            
            # Check for existing matching hypothesis
            matching_hypothesis = None
            
            # 1. If hypothesis_id is provided, use it directly
            if hypothesis_id:
                print(f"Using directly provided hypothesis ID: {hypothesis_id}")
                # Get the main window to access studies manager
                main_window = self.window()
                if hasattr(main_window, 'studies_manager'):
                    matching_hypothesis = main_window.studies_manager.get_hypothesis(hypothesis_id)
                    if matching_hypothesis:
                        self.status_bar.showMessage(f"Updating specified hypothesis: {matching_hypothesis.get('title')}")
                    else:
                        print(f"Warning: Could not find hypothesis with ID {hypothesis_id}")
            
            # 2. If no direct ID or matching hypothesis not found, check dropdown selection
            if not matching_hypothesis and self.hypothesis_combo.currentData():
                hyp_id = self.hypothesis_combo.currentData()
                print(f"Checking hypothesis selected in dropdown: {hyp_id}")
                potential_match = main_window.studies_manager.get_hypothesis(hyp_id)
                
                # Only use the selected hypothesis if its outcome variable matches the current one
                if potential_match:
                    hyp_outcome = potential_match.get('outcome_variables', '')
                    hyp_related_outcome = potential_match.get('related_outcome', '')
                    
                    print(f"Dropdown hypothesis outcome: '{hyp_outcome}', related outcome: '{hyp_related_outcome}'")
                    print(f"Current outcome: '{outcome}'")
                    
                    # Only consider it a match if the outcome variables match
                    if (hyp_outcome and hyp_outcome.lower() == outcome.lower()) or \
                       (hyp_related_outcome and hyp_related_outcome.lower() == outcome.lower()):
                        matching_hypothesis = potential_match
                        self.status_bar.showMessage(f"Using selected hypothesis: {matching_hypothesis.get('title')}")
                        print(f"Found matching hypothesis from dropdown: {matching_hypothesis.get('title')}")
                    else:
                        print(f"Ignoring selected hypothesis because outcome doesn't match: {potential_match.get('title')}")
                        print(f"Selected hypothesis outcome: '{hyp_outcome}', Current outcome: '{outcome}'")
            
            # 3. If still no match, try to find by outcome and dataset
            # ONLY if no hypothesis_id was explicitly provided
            if not matching_hypothesis and not hypothesis_id:
                # Try to find a matching hypothesis based on outcome and dataset
                matching_hypothesis = self.find_matching_hypothesis(outcome, self.current_name)
                if matching_hypothesis:
                    # First confirm with LLM if this hypothesis is a good match
                    confirm_prompt = f"""
                    I found an existing hypothesis in the study related to the outcome variable '{outcome}'.
                    The hypothesis is: {matching_hypothesis.get('title')}
                    Alternative hypothesis: {matching_hypothesis.get('alternative_hypothesis')}
                    
                    I'm now running a {test_name} on the same outcome variable.
                    
                    Please determine if I should:
                    1. Update the existing hypothesis with new test results
                    2. Create a new hypothesis
                    
                    Consider whether the existing hypothesis aligns with the current statistical test
                    and study design ({study_type_val.replace('_', ' ').title()}).
                    
                    Return your decision as one of:
                    "UPDATE" or "CREATE_NEW"
                    
                    IMPORTANT: Only return one of these exact words, no other text.
                    """
                    
                    # Call LLM to confirm match
                    decision = await call_llm_async(confirm_prompt)
                    decision = decision.strip().upper()
                    
                    if "UPDATE" not in decision:
                        matching_hypothesis = None
                        self.status_bar.showMessage(f"LLM recommended creating new hypothesis instead of updating existing one")
                    else:
                        self.status_bar.showMessage(f"LLM recommended updating existing hypothesis: {matching_hypothesis.get('title')}")
            
            # Get data characteristics to inform the LLM
            outcome_type = infer_data_type(df, outcome)
            
            # Get test results if available and prepare them for hypothesis generation
            test_results = None
            if self.last_test_result and self.last_test_result.get('outcome') == outcome:
                raw_results = self.last_test_result.get('result', {})
                # Use the targeted preparation method with hypothesis purpose
                test_results = self._prepare_for_llm(raw_results, purpose="hypothesis")
            
            # Prepare group information with size limits
            group_info = ""
            if group and group in df.columns:
                # Limit to 10 values to prevent token explosion
                unique_group_values = df[group].unique()
                display_values = unique_group_values[:10]
                # Convert numpy types to native Python types
                display_values = [value.item() if hasattr(value, 'item') else value for value in display_values]
                group_info = f"\nGroup variable '{group}' with values: {display_values}"
                if len(unique_group_values) > 10:
                    group_info += f" (and {len(unique_group_values) - 10} more values...)"
            
            # Prepare time information with size limits
            time_info = ""
            if time and time in df.columns:
                # Limit to 10 values to prevent token explosion
                unique_time_values = df[time].unique()
                display_values = sorted(unique_time_values)[:10] 
                # Convert numpy types to native Python types
                display_values = [value.item() if hasattr(value, 'item') else value for value in display_values]
                time_info = f"\nTime variable '{time}' with values: {display_values}"
                if len(unique_time_values) > 10:
                    time_info += f" (and {len(unique_time_values) - 10} more values...)"
                
            # Add existing hypothesis info if updating
            existing_info = ""
            if matching_hypothesis:
                existing_info = f"""
                You are updating an existing hypothesis:
                Title: {matching_hypothesis.get('title')}
                Null hypothesis: {matching_hypothesis.get('null_hypothesis')}
                Alternative hypothesis: {matching_hypothesis.get('alternative_hypothesis')}
                
                Please ensure your updated hypothesis is consistent with the previous one, but can
                be refined based on new test results or information.
                """
                
            # Format prompt for generating hypothesis
            prompt = f"""
            {existing_info}
            Generate a clear, scientific hypothesis for a statistical test with the following details:

            Test: {test_name}
            Study Design: {study_type_val.replace('_', ' ').title()}
            Outcome Variable: {outcome} (type: {outcome_type.value if outcome_type else 'Unknown'})
            {group_info}{time_info}
            
            Test Results: {json.dumps(test_results) if test_results else 'Not yet tested'}

            Based on this information, please create:
            1. A title for the hypothesis
            2. A formal null hypothesis (H₀)
            3. A formal alternative hypothesis (H₁)
            4. Specify directionality (non-directional, greater than, or less than)
            5. Determine the hypothesis status based on test results (if available)

            Return your response in this JSON format:
            {{
                "title": "Brief descriptive title for the hypothesis",
                "null_hypothesis": "Formal statement of H₀",
                "alternative_hypothesis": "Formal statement of H₁",
                "directionality": "non-directional|greater|less",
                "notes": "Any additional notes on this hypothesis",
                "status": "untested|confirmed|rejected|inconclusive",
                "status_reason": "Explanation of the status determination"
            }}

            If test results are available, determine the status as:
            - confirmed: if p < 0.05 and effect is in expected direction
            - rejected: if p ≥ 0.05 or effect is in opposite direction
            - inconclusive: if results are ambiguous or assumptions were violated
            - untested: if no test results are available
            """

            # Call LLM
            response = await call_llm_async(prompt)
            
            # Parse JSON response
            match = re.search(r'({.*})', response, re.DOTALL)
            if not match:
                print("Failed to parse hypothesis response from LLM")
                return None
                
            hypothesis_data = json.loads(match.group(1))
            
            # Add additional test-related fields
            hypothesis_data['outcome_variables'] = outcome
            # Always include related_outcome field to support proper matching in the future
            hypothesis_data['related_outcome'] = outcome
            if group:
                hypothesis_data['predictor_variables'] = group
            hypothesis_data['expected_test'] = test_name
            hypothesis_data['alpha_level'] = 0.05  # Default
            hypothesis_data['dataset_name'] = self.current_name  # Store dataset name
            
            # Add minimal test results if available
            if test_results:
                # Include only the essential information about test results
                hypothesis_data['test_results'] = {
                    'significant': test_results.get('significant', False),
                    'p_value': test_results.get('p_value', None),
                    'test_name': test_results.get('test', test_name)
                }
                hypothesis_data['test_date'] = datetime.now().isoformat()
            
            # Get the main window to access studies manager
            main_window = self.window()
            if hasattr(main_window, 'studies_manager'):
                if matching_hypothesis:
                    # Update existing hypothesis
                    # Add ID from existing hypothesis to ensure update targets the right one
                    hypothesis_id = matching_hypothesis.get('id')
                    
                    # Add a note about the test change if different from previous test
                    prev_test = matching_hypothesis.get('expected_test', '')
                    if prev_test and prev_test != test_name:
                        notes = hypothesis_data.get('notes', '')
                        update_note = f"Updated from {prev_test} to {test_name} on {datetime.now().strftime('%Y-%m-%d')}."
                        if notes:
                            hypothesis_data['notes'] = f"{notes}\n\n{update_note}"
                        else:
                            hypothesis_data['notes'] = update_note
                    
                    # Update the hypothesis
                    success = main_window.studies_manager.update_hypothesis(
                        hypothesis_id=hypothesis_id,
                        update_data=hypothesis_data
                    )
                    
                    if success:
                        test_update_msg = ""
                        if prev_test and prev_test != test_name:
                            test_update_msg = f" (changed test from {prev_test} to {test_name})"
                        self.status_bar.showMessage(f"Updated hypothesis: {hypothesis_data['title']} ({hypothesis_data['status']}){test_update_msg}")
                        # Refresh the hypotheses dropdown
                        self.refresh_hypotheses()
                        # Set dropdown to the updated hypothesis
                        for i in range(self.hypothesis_combo.count()):
                            if self.hypothesis_combo.itemData(i) == hypothesis_id:
                                self.hypothesis_combo.setCurrentIndex(i)
                                break
                        # Return the full hypothesis data instead of just the ID
                        return hypothesis_data
                else:
                    # Create new hypothesis
                    hypothesis_data['id'] = str(uuid.uuid4())
                    
                    # Add the hypothesis through studies manager
                    main_window.studies_manager.add_hypothesis_to_study(
                        hypothesis_text=hypothesis_data['title'],
                        related_outcome=outcome,
                        hypothesis_data=hypothesis_data
                    )
                    self.status_bar.showMessage(f"Generated new hypothesis: {hypothesis_data['title']} ({hypothesis_data['status']})")
                    
                    # Refresh the hypotheses dropdown
                    self.refresh_hypotheses()
                    # Set dropdown to the new hypothesis
                    for i in range(self.hypothesis_combo.count()):
                        if self.hypothesis_combo.itemData(i) == hypothesis_data['id']:
                            self.hypothesis_combo.setCurrentIndex(i)
                            break
                    
                    # Return the full hypothesis data instead of just the ID
                    return hypothesis_data
            
            return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating hypothesis: {str(e)}")
            return None

    @asyncSlot()
    async def run_statistical_test(self):
        """Run the selected statistical test with the selected variables."""
        # Set button to working state
        self.set_button_working_state(self.run_test_button, True)
        
        if self.current_dataframe is None or self.current_dataframe.empty:
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage("Test failed: No dataset selected")
            return
        
        # Get the selected study type
        study_type = self.design_type_combo.currentData()
        if not study_type:
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage("Test failed: No study design selected")
            return
        
        # Get the specification for this study type
        spec = STUDY_DESIGN_REGISTRY.get(study_type)
        if not spec:
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage(f"Test failed: No specification for study type: {study_type}")
            return
        
        # Get the selected variables
        outcome = next((col for col, role in self.column_roles.items() 
                       if role == VariableRole.OUTCOME), None)
        group = next((col for col, role in self.column_roles.items() 
                     if role == VariableRole.GROUP), None)
        covariates = [col for col, role in self.column_roles.items() 
                     if role == VariableRole.COVARIATE]
        subject_id = next((col for col, role in self.column_roles.items() 
                          if role == VariableRole.SUBJECT_ID), None)
        time = next((col for col, role in self.column_roles.items() 
                    if role == VariableRole.TIME), None)
        pair_id = next((col for col, role in self.column_roles.items() 
                  if role == VariableRole.PAIR_ID), None)
        event = next((col for col, role in self.column_roles.items() 
                  if role == VariableRole.EVENT), None)
        
        # Validate required fields based on the study design specification
        missing_vars = []
        for var_name, requirement in spec.variable_requirements.items():
            if requirement == VariableRequirement.REQUIRED:
                var_value = None
                if var_name == "outcome":
                    var_value = outcome
                elif var_name == "group":
                    var_value = group
                elif var_name == "subject_id":
                    var_value = subject_id
                elif var_name == "time":
                    var_value = time
                elif var_name == "covariate" and not covariates:
                    var_value = None
                elif var_name == "pair_id":
                    var_value = pair_id
                elif var_name == "event":
                    var_value = event
                
                if not var_value:
                    missing_vars.append(var_name.replace('_', ' ').title())
        
        if missing_vars:
            self.status_bar.showMessage(f"Test failed: Missing required variables for {study_type.display_name} design")
            self.set_button_working_state(self.run_test_button, False)
            return
        
        # Get the selected test
        test_index = self.test_combo.currentIndex()
        if test_index < 0:
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage("Please select a statistical test")
            return
        
        test_key = self.test_combo.itemData(test_index)
        test_name = self.test_combo.currentText()

        # Collect test parameters
        kwargs = {}
        
        # Add mu parameter for one-sample tests if available
        if hasattr(self, 'mu_input'):
            current_test = self.test_combo.currentText()
            is_one_sample_test = any(test in current_test.lower() for test in ["one_sample", "one sample"])
            if is_one_sample_test:
                # Always use the current UI value directly from mu_input
                mu_value = float(self.mu_input.value())
                print(f"Using mu value from UI: {mu_value}")
                # Store it for future reference
                self._current_mu_value = mu_value
                kwargs["mu"] = mu_value
        
        # Check if test is compatible with current design
        compatible_tests = [test.value for test in spec.compatible_tests]
        if test_key not in compatible_tests:
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage(f"Test failed: Selected test not compatible with {study_type.display_name} design")
            return
        
        # Run statistical assumption checks
        diagnostics = self.check_statistical_assumptions(
            self.current_dataframe, outcome, group, subject_id, time, covariates
        )
        
        # Always update the indicators
        self.update_assumption_indicators(diagnostics)
        
        # Add diagnostics to kwargs to include in test results
        kwargs['diagnostics'] = diagnostics
        
        test_success = False
        try:
            # Run the appropriate test based on the test type
            self.status_bar.showMessage(f"Running {self.test_combo.currentText()}...")
            
            # Get the test function from the registry
            test_data = TEST_REGISTRY.get(test_key)
            if not test_data:
                self.set_button_working_state(self.run_test_button, False)
                self.status_bar.showMessage(f"Test failed: Test '{test_key}' not found in registry")
                return
            
            test_function = test_data.test_function
            
            # Get the appropriate test executor
            test_executor = TestExecutorFactory.get_executor(test_key)
            if test_executor:
                result = test_executor.execute(
                    df=self.current_dataframe,
                    outcome=outcome,
                    group=group,
                    covariates=covariates,
                    subject_id=subject_id,
                    time=time,
                    test_function=test_function,
                    pair_id=pair_id,
                    event=event,
                    **kwargs
                )
            else:
                self.set_button_working_state(self.run_test_button, False)
                self.status_bar.showMessage(f"Test failed: No executor found for test: {test_key}")
                return
            
            if result:
                # Add diagnostics warnings to result for display
                if "warnings" not in result:
                    result["warnings"] = []
                result["warnings"].extend(diagnostics["warnings"])
                
                # Also add assumption checks to the result
                if "assumptions" not in result:
                    result["assumptions"] = {}
                
                # Add sample size assumption
                result["assumptions"]["sample_size"] = {
                    "description": "Sufficient sample size",
                    "satisfied": diagnostics["sample_size"]["sufficient"],
                    "details": {
                        "n": diagnostics["sample_size"]["after_na_drop"],
                        "min_recommended": diagnostics["sample_size"]["min_recommended"]
                    },
                    "message": diagnostics["sample_size"]["message"] if not diagnostics["sample_size"]["sufficient"] else ""
                }
                
                # Add normality assumption if relevant
                is_parametric = any(t in test_name.lower() for t in ['t-test', 't test', 'anova', 'regression', 'pearson'])
                if is_parametric and "outcome" in diagnostics["normality"] and diagnostics["normality"]["outcome"]["normal"] is not None:
                    result["assumptions"]["normality"] = {
                        "description": "Normal distribution of data",
                        "satisfied": diagnostics["normality"]["outcome"]["normal"],
                        "details": {
                            "test": "Shapiro-Wilk",
                            "statistic": diagnostics["normality"]["outcome"]["test_statistic"],
                            "p_value": diagnostics["normality"]["outcome"]["p_value"]
                        },
                        "message": diagnostics["normality"]["outcome"]["message"] if not diagnostics["normality"]["outcome"]["normal"] else ""
                    }
                
                # Add group balance assumption if relevant
                if group and "balanced" in diagnostics["group_balance"]:
                    result["assumptions"]["group_balance"] = {
                        "description": "Balanced group sizes",
                        "satisfied": diagnostics["group_balance"]["balanced"],
                        "details": {
                            "min_group_size": diagnostics["group_balance"]["min_group_size"],
                            "group_counts": diagnostics["group_balance"]["group_counts"]
                        },
                        "message": diagnostics["group_balance"]["message"] if not diagnostics["group_balance"]["balanced"] else ""
                    }
                        
                # Generate hypothesis if enabled in settings
                hypothesis_id = None
                if self.test_settings['update_hypothesis']:
                    hypothesis_id = await self.generate_hypothesis_for_test(
                        outcome=outcome, 
                        group=group, 
                        subject_id=subject_id, 
                        time=time, 
                        test_name=self.test_combo.currentText(),
                        study_type=study_type
                    )
                        
                # Format and display the test results based on interpretation setting
                if self.test_settings['generate_interpretation']:
                    self.display_test_results(test_key, result)
                else:
                    # Use basic display without AI interpretation
                    self.display_basic_results(test_key, result)
                
                # Store the results for later use
                self.last_test_result = {
                    'test_key': test_key,
                    'test_name': self.test_combo.currentText(),
                    'result': result,
                    'dataset': self.current_name,
                    'outcome': outcome,
                    'group': group,
                    'covariates': covariates,
                    'subject_id': subject_id,
                    'time': time,
                    'design': study_type.value.replace('_', '-'),
                    'timestamp': datetime.now().isoformat(),
                }
                
                # Store mu value if it was used
                if "mu" in kwargs:
                    self.last_test_result['mu'] = kwargs["mu"]
                
                # Store in the outcomes dictionary too
                self.test_results[outcome] = self.last_test_result
                
                # Update study documentation if enabled
                if self.test_settings['update_documentation']:
                    await self.update_study_design()

                test_success = True
            else:
                self.status_bar.showMessage("No results returned from the test.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.set_button_working_state(self.run_test_button, False)
            self.status_bar.showMessage(f"Test failed with error: {str(e)[:50]}...")
            return
        
        # Always update status bar at end of method
        if test_success:
            self.status_bar.showMessage(f"Test complete: {self.test_combo.currentText()} on {outcome}")
        else:
            self.status_bar.showMessage("Test failed: No results returned")
        
        # Reset button state
        self.set_button_working_state(self.run_test_button, False)
        
        # After identifying the test but before running it, save the test data to StudiesManager
        test_data = {
            'test_key': test_key,
            'test_name': self.test_combo.currentText(),
            'outcome': outcome,
            'group': group,
            'covariates': covariates,
            'subject_id': subject_id,
            'time': time,
            'pair_id': pair_id,
            'event': event,
            'dataset_name': self.current_name,
            'df': self.current_dataframe.copy() if self.current_dataframe is not None else None,  # Include dataframe copy for convenience
        }
        
        # Store the test data in the StudiesManager
        if hasattr(self, 'studies_manager') and self.studies_manager:
            self.studies_manager.store_current_test_data(test_data)
    
    def display_test_results(self, test_key, result):
        """Display statistical test results in the results text area with focus on interpretation."""

        QApplication.processEvents()  # Update UI
        
        # Call LLM asynchronously to interpret results
        self.interpret_results_with_llm(test_key, result)
    
    @asyncSlot()
    async def interpret_results_with_llm(self, test_key, result):
        """Use LLM to interpret statistical test results and display them."""
        try:
            # Format the results for the LLM
            test_name = result.get('test', result.get('test_name', 'Statistical Test'))
            test_success = not result.get('error') and result.get('success', True)
            p_value = result.get('p_value', result.get('p', result.get('p-val', None)))
            significant = result.get('significant', False)
            
            # Get variable information
            variables = result.get('variables', {})
            outcome = variables.get('outcome', 'unknown')
            group = variables.get('group', 'N/A')
            
            # Use the targeted preparation method for LLM interpretation
            minimal_result = self._prepare_for_llm(result, purpose="interpretation")
            
            # Create the prompt for the LLM
            prompt = f"""
            Analyze these statistical test results and provide a clear interpretation.
            
            Test Information:
            - Test: {test_name}
            - Success: {test_success}
            - P-value: {p_value}
            - Significant: {significant}
            - Outcome variable: {outcome}
            - Group variable: {group}
            
            Test Results:
            {json.dumps(minimal_result, indent=2, default=str)}
            
            Please provide a concise, clear interpretation of these results that would help a researcher understand the findings.
            Your response should be in JSON format with the following structure:
            {{
                "title": "A clear, one-line summary of the result",
                "significant": true or false based on p-value and test result,
                "conclusion": "A one-sentence definitive conclusion about the outcome",
                "key_points": [
                    "Point 1 about the result",
                    "Point 2 about the result",
                    "Point 3 about the result"
                ],
                "effect_size_interpretation": "Brief comment on effect size if available",
                "recommendations": "A brief recommendation for the researcher",
                "cautions": "Any cautions about interpretation"
            }}
            
            Be precise, factual, and direct in your interpretation. Use statistical language appropriately.
            """
            
            # Call the LLM
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                interpretation = json.loads(json_str.group(1))
                self.display_formatted_results(test_key, result, interpretation)
            else:
                # Fallback to basic display if JSON parsing fails
                self.display_basic_results(test_key, result)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Fallback to basic display if LLM call fails
            self.display_basic_results(test_key, result)
            
    def _sanitize_for_json(self, obj):
        """Helper method to sanitize dictionary values for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(item) for item in obj)
        elif isinstance(obj, pd.DataFrame):
            return f"DataFrame with shape {obj.shape}"
        elif hasattr(obj, '__class__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return f"Object of type {obj.__class__.__name__}"
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
    
    def display_formatted_results(self, test_key, result, interpretation):
        """Display nicely formatted results using the LLM interpretation."""
        # Test name and general information
        test_name = result.get('test', result.get('test_name', 'Statistical Test'))
        
        # Determine if test was successful
        test_success = True
        failure_reason = ""
        
        # Check for error or failed flag in result
        if 'error' in result:
            test_success = False
            failure_reason = result['error']
        elif 'success' in result:
            test_success = result['success']
            if not test_success and 'error_message' in result:
                failure_reason = result['error_message']
        
        # Display assumptions in the assumptions tab
        self.display_assumptions(result)
        
        # Display JSON interpretation as a tree view
        self.json_tree.clear()
        self._add_json_to_tree(interpretation, self.json_tree.invisibleRootItem())
        self.json_tree.expandAll()
        
        # Store the result for potential saving
        self.current_test_result = result
        self.current_test_key = test_key
        
    def _add_json_to_tree(self, data, parent_item):
        """Recursively add JSON data to tree widget."""
        if isinstance(data, dict):
            for key, value in data.items():
                item = QTreeWidgetItem(parent_item)
                
                # Convert key to title case for better readability
                display_key = str(key).replace('_', ' ').title()
                item.setText(0, display_key)
                
                # Special handling for 'significant' key to add icon
                if key.lower() == 'significant' and isinstance(value, bool):
                    if value:  # If significant is True
                        check_icon = load_bootstrap_icon("check-circle-fill", color="#43A047", size=16)
                        item.setIcon(0, check_icon)
                    else:  # If significant is False
                        warning_icon = load_bootstrap_icon("x-circle", color="#E53935", size=16)
                        item.setIcon(0, warning_icon)
                
                if isinstance(value, (dict, list)):
                    # For complex types, add as children
                    self._add_json_to_tree(value, item)
                else:
                    # For simple values, display directly
                    item.setText(1, str(value))
        elif isinstance(data, list):
            for i, value in enumerate(data):
                item = QTreeWidgetItem(parent_item)
                item.setText(0, f"Item {i+1}")  # More readable than just showing [i]
                
                if isinstance(value, (dict, list)):
                    # For complex types, add as children
                    self._add_json_to_tree(value, item)
                else:
                    # For simple values, display directly
                    item.setText(1, str(value))
                    
    def display_assumptions(self, result):
        """Display test assumptions using a tree widget with bootstrap icons."""
        # Extract assumptions from result
        assumptions = result.get('assumptions', {})
        
        # Clear the tree
        self.assumptions_tree.clear()
        
        # If there are no assumptions, display a message
        if not assumptions:
            no_assumptions = QTreeWidgetItem(self.assumptions_tree)
            no_assumptions.setText(0, "No assumptions were checked for this test")
            no_assumptions.setText(1, "")
        else:
            # Add assumptions to tree recursively
            self._add_assumptions_to_tree(assumptions, self.assumptions_tree)
        
        # Resize columns to content
        for i in range(2):
            self.assumptions_tree.resizeColumnToContents(i)
    
    def _add_assumptions_to_tree(self, assumptions, parent_item):
        """Recursively add assumptions to the tree widget with proper status checks."""
        if isinstance(assumptions, dict):
            # Handle dictionary of assumptions
            for name, assumption_data in assumptions.items():
                # Skip if name is empty
                if not name:
                    continue
                    
                assumption_item = QTreeWidgetItem(parent_item)
                
                # Get description
                if isinstance(assumption_data, dict):
                    description = assumption_data.get('description', name.replace('_', ' ').title())
                else:
                    description = name.replace('_', ' ').title()
                
                # Set the assumption name/description
                assumption_item.setText(0, description)
                
                # Determine status using recursive check
                status = self._extract_assumption_status(assumption_data)
                
                # Set status icon based on result
                if status == 'PASSED':
                    check_icon = load_bootstrap_icon("check-circle-fill", color="#43A047", size=16)
                    assumption_item.setIcon(1, check_icon)
                    assumption_item.setText(1, "Passed")
                elif status == 'FAILED':
                    warning_icon = load_bootstrap_icon("x-circle-fill", color="#E53935", size=16)
                    assumption_item.setIcon(1, warning_icon)
                    assumption_item.setText(1, "Failed")
                elif status == 'WARNING':
                    warning_icon = load_bootstrap_icon("exclamation-triangle-fill", color="#FB8C00", size=16)
                    assumption_item.setIcon(1, warning_icon)
                    assumption_item.setText(1, "Warning")
                elif status == 'NOT_APPLICABLE':
                    info_icon = load_bootstrap_icon("info-circle-fill", color="#1976D2", size=16)
                    assumption_item.setIcon(1, info_icon)
                    assumption_item.setText(1, "N/A")
                else:
                    # Unknown status
                    question_icon = load_bootstrap_icon("question-circle-fill", color="#757575", size=16)
                    assumption_item.setIcon(1, question_icon)
                    assumption_item.setText(1, "Unknown")
                
                # Recursively add nested assumptions if present
                if isinstance(assumption_data, dict) and any(isinstance(v, (dict, list)) for v in assumption_data.values()):
                    self._add_assumptions_to_tree(assumption_data, assumption_item)
        
        elif isinstance(assumptions, list):
            # Handle list of assumptions
            for assumption in assumptions:
                # Skip if empty
                if not assumption:
                    continue
                    
                assumption_item = QTreeWidgetItem(parent_item)
                
                # Handle both dictionary and string formats
                if isinstance(assumption, dict):
                    description = assumption.get('description', '')
                else:
                    description = str(assumption)
                
                # Set the assumption name
                assumption_item.setText(0, description)
                
                # Determine status using recursive check
                status = self._extract_assumption_status(assumption)
                
                # Set status icon based on result
                if status == 'PASSED':
                    check_icon = load_bootstrap_icon("check-circle-fill", color="#43A047", size=16)
                    assumption_item.setIcon(1, check_icon)
                    assumption_item.setText(1, "Passed")
                elif status == 'FAILED':
                    warning_icon = load_bootstrap_icon("x-circle-fill", color="#E53935", size=16)
                    assumption_item.setIcon(1, warning_icon)
                    assumption_item.setText(1, "Failed")
                elif status == 'WARNING':
                    warning_icon = load_bootstrap_icon("exclamation-triangle-fill", color="#FB8C00", size=16)
                    assumption_item.setIcon(1, warning_icon)
                    assumption_item.setText(1, "Warning")
                elif status == 'NOT_APPLICABLE':
                    info_icon = load_bootstrap_icon("info-circle-fill", color="#1976D2", size=16)
                    assumption_item.setIcon(1, info_icon)
                    assumption_item.setText(1, "N/A")
                else:
                    # Unknown status
                    question_icon = load_bootstrap_icon("question-circle-fill", color="#757575", size=16)
                    assumption_item.setIcon(1, question_icon)
                    assumption_item.setText(1, "Unknown")
                
                # Recursively add nested assumptions if present
                if isinstance(assumption, dict) and any(isinstance(v, (dict, list)) for v in assumption.values()):
                    self._add_assumptions_to_tree(assumption, assumption_item)
    
    def _extract_assumption_status(self, check_info):
        """
        Extract the status of an assumption, handling nested structures.
        Possible status values: PASSED, FAILED, WARNING, NOT_APPLICABLE, UNKNOWN
        """
        if not isinstance(check_info, dict):
            return 'UNKNOWN'
        
        if not check_info:
            return 'UNKNOWN'
        
        # Check for 'satisfied' key first
        if isinstance(check_info, dict) and 'satisfied' in check_info:
            return 'PASSED' if check_info['satisfied'] else 'FAILED'
        
        # Check for 'result' key
        if isinstance(check_info, dict) and 'result' in check_info:
            if isinstance(check_info['result'], str):
                result_value = check_info['result'].upper()
                if result_value in ['PASSED', 'FAILED', 'WARNING', 'NOT_APPLICABLE']:
                    return result_value
            elif hasattr(check_info['result'], 'name'):
                result_value = check_info['result'].name
                return result_value
        
        # Recursively check nested dictionaries and lists
        statuses = []
        
        if isinstance(check_info, dict):
            for key, value in check_info.items():
                if isinstance(value, (dict, list)):
                    nested_status = self._extract_assumption_status(value)
                    if nested_status != 'UNKNOWN':
                        statuses.append(nested_status)
        
        elif isinstance(check_info, list):
            for item in check_info:
                if isinstance(item, (dict, list)):
                    nested_status = self._extract_assumption_status(item)
                    if nested_status != 'UNKNOWN':
                        statuses.append(nested_status)
        
        # Determine overall status based on collected statuses
        if statuses:
            if all(status == 'PASSED' for status in statuses):
                return 'PASSED'
            elif all(status == 'FAILED' for status in statuses):
                return 'FAILED'
            elif any(status == 'FAILED' for status in statuses):
                return 'WARNING'
            elif any(status == 'WARNING' for status in statuses):
                return 'WARNING'
            else:
                return statuses[0]  # Return first status if all are the same type
        
        return 'UNKNOWN'
    
    def display_basic_results(self, test_key, result):
        """Display basic results without LLM interpretation as a fallback."""
        # Display assumptions in the assumptions tab
        self.display_assumptions(result)
        
        # Create a simple interpretation for the JSON view
        basic_interpretation = {
            "test_name": result.get('test', result.get('test_name', 'Statistical Test')),
            "significant": result.get('significant', False),
            "p_value": result.get('p_value', result.get('p', result.get('p-val', "Unknown"))),
            "summary": "Basic test results without AI interpretation"
        }
        
        # Add any available statistics
        if 'statistic' in result:
            basic_interpretation['statistic'] = result['statistic']
        
        if 'df' in result:
            basic_interpretation['degrees_of_freedom'] = result['df']
        
        if 'effect_size' in result:
            basic_interpretation['effect_size'] = result['effect_size']
        
        # Display the basic interpretation
        self.json_tree.clear()
        self._add_json_to_tree(basic_interpretation, self.json_tree.invisibleRootItem())
        self.json_tree.expandAll()
        
        # Store the result for potential saving
        self.current_test_result = result
        self.current_test_key = test_key
        
    def save_results_to_study(self):
        """Save the current test results to the active study."""
        if not self.studies_manager:
            self.status_bar.showMessage("Studies manager not available")
            return False
            
        if not self.current_test_key or not self.current_test_result:
            self.status_bar.showMessage("No test results to save")
            return False
        
        # Extract variable information
        outcome_var = self.outcome_combo.currentText()
        group_vars = []
        if self.group_combo.isEnabled() and self.group_combo.currentText():
            group_vars = [self.group_combo.currentText()]
        
        subject_id_var = self.subject_id_combo.currentText() if self.subject_id_combo.isEnabled() else None
        time_var = self.time_combo.currentText() if self.time_combo.isEnabled() else None
        
        # Get selected covariates from the QListWidget instead of QPlainTextEdit
        covariate_vars = [self.covariates_list.item(i).text() 
                         for i in range(self.covariates_list.count())]
        
        # Get dataset name
        dataset_name = self.dataset_selector.currentText()
        
        # Create variable metadata structure
        variable_metadata = {
            "outcome": outcome_var,
            "group": group_vars,
            "subject_id": subject_id_var,
            "time": time_var,
            "event": self.event_combo.currentText() if self.event_combo.isEnabled() else None,
            "covariates": covariate_vars,
            "role_definitions": {
                outcome_var: "outcome"
            }
        }
        
        # Add group variables to roles
        for var in group_vars:
            variable_metadata["role_definitions"][var] = "group"
        
        # Add subject_id to roles if present
        if subject_id_var:
            variable_metadata["role_definitions"][subject_id_var] = "subject_id"
        
        # Add time to roles if present
        if time_var:
            variable_metadata["role_definitions"][time_var] = "time"
        
        # Add event to roles if present
        event_var = self.event_combo.currentText() if self.event_combo.isEnabled() and self.event_combo.currentText() != "Select..." else None
        if event_var:
            variable_metadata["role_definitions"][event_var] = "event"
        
        # Add covariates to roles
        for var in covariate_vars:
            variable_metadata["role_definitions"][var] = "covariate"
        
        # Create test details structure
        test_details = {
            "test_key": self.current_test_key,
            "test_name": self.test_combo.currentText(),
            "results": self.current_test_result,
            "timestamp": datetime.now().isoformat(),
            "design": self.design_type_combo.currentText(),
            "dataset": dataset_name,
            "variables": variable_metadata,
        }
        
        # Store in studies manager
        success = self.studies_manager.add_statistical_results_to_study(
            outcome_name=outcome_var,
            test_details=test_details
        )
        
        if success:
            return True
        else:
            return False
    
    def on_outcome_changed(self, index):
        """Handle selection of a different outcome variable."""
        if index < 0 or not self.outcome_selector.count():
            return
            
        # Get the selected outcome
        outcome = self.outcome_selector.currentText()
        
        # Check if we have results for this outcome
        if outcome in self.test_results:
            result = self.test_results[outcome]
            
            # Update the UI to show these results
            self.display_test_results(result['test_key'], result['result'])
            
            # Update the test combo box
            for i in range(self.test_combo.count()):
                if self.test_combo.itemData(i) == result['test_key']:
                    self.test_combo.setCurrentIndex(i)
                    break
                    
            self.status_bar.showMessage(f"Showing results for outcome: {outcome}")
        else:
            # Clear previous results
            self.status_bar.showMessage(f"Selected outcome: {outcome} (no test results yet)")
            
        # Highlight the current outcome in the table
        self.highlight_outcome(outcome)
        
        # Check if there's a matching hypothesis for this outcome
        matching_hypothesis = self.find_matching_hypothesis(outcome, self.current_name)
        
        # Update hypothesis dropdown if a matching hypothesis is found
        if matching_hypothesis:
            # Find the hypothesis in the dropdown
            for i in range(self.hypothesis_combo.count()):
                if self.hypothesis_combo.itemData(i) == matching_hypothesis.get('id'):
                    # Set the dropdown to this hypothesis
                    self.hypothesis_combo.setCurrentIndex(i)
                    self.status_bar.showMessage(f"Found matching hypothesis for {outcome}")
                    break
    
    def highlight_outcome(self, outcome):
        """Highlight the current outcome in the data table."""
        if self.current_dataframe is None or self.current_dataframe.empty or not outcome:
            return
            
        # Reset all column roles related to outcome
        for col, role in self.column_roles.items():
            if role == VariableRole.OUTCOME:
                self.column_roles[col] = VariableRole.NONE
                
        # Set the new outcome
        if outcome in self.column_roles:
            self.column_roles[outcome] = VariableRole.OUTCOME
            
        # Update the UI
        self.populate_variable_dropdowns()  # Update dropdowns to match new roles
    
    @asyncSlot()
    async def analyze_all_outcomes(self):
        """Run auto-select test for all potential outcome variables."""
        # Set button to working state
        self.set_button_working_state(self.analyze_all_outcomes_button, True)
        
        if self.current_dataframe is None or self.current_dataframe.empty:
            self.set_button_working_state(self.analyze_all_outcomes_button, False)
            self.status_bar.showMessage("No dataset selected")
            return
            
        # Determine potential outcome variables (usually numeric)
        potential_outcomes = []
        for col in self.current_dataframe.columns:
            if pd.api.types.is_numeric_dtype(self.current_dataframe[col]):
                if col not in self.variable_roles or self.variable_roles[col] != VariableRole.GROUP:
                    potential_outcomes.append(col)
        
        if not potential_outcomes:
            self.set_button_working_state(self.analyze_all_outcomes_button, False)
            self.status_bar.showMessage("No potential outcome variables found")
            return
            
        # Store current outcome value
        current_outcome = self.outcome_combo.currentText()
        
        # Analyze all outcomes
        try:
            self.test_results = {}  # Clear previous results
            
            for outcome in potential_outcomes:
                # Check if outcome isn't already the selected outcome
                if outcome != current_outcome:
                    # Set as current outcome
                    outcome_index = self.outcome_combo.findText(outcome)
                    if outcome_index >= 0:
                        self.outcome_combo.setCurrentIndex(outcome_index)
                    else:
                        continue  # Skip if not found
                
                # Run test selection and execution
                await self.auto_select_test(for_outcome=outcome)
                await self.run_statistical_test()
                
                self.status_bar.showMessage(f"Analyzed {outcome}")
                
            # Restore original outcome
            original_outcome_index = self.outcome_combo.findText(current_outcome)
            if original_outcome_index >= 0:
                self.outcome_combo.setCurrentIndex(original_outcome_index)
            
            self.status_bar.showMessage(f"Completed analysis of {len(potential_outcomes)} outcome variables")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"Error analyzing outcomes: {str(e)}")
        finally:
            # Reset button state
            self.set_button_working_state(self.analyze_all_outcomes_button, False)
            
            # Update outcome selector with all outcomes we analyzed
            self.outcome_selector.blockSignals(True)
            self.outcome_selector.clear()
            for outcome in self.test_results.keys():
                self.outcome_selector.addItem(outcome)
            self.outcome_selector.blockSignals(False)
            
            if self.outcome_selector.count() > 0:
                self.outcome_selector.setCurrentIndex(0)
                
        self.status_bar.showMessage("Analysis of all outcomes complete")

    def manual_role_changed(self, role):
        """Handle selection changes in the manual variable role dropdowns."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return
        
        logging.info(f"Manual role change: {role}")
        
        if role == VariableRole.OUTCOME:
            new_var = self.outcome_combo.currentText()
            if new_var and new_var != "Select...":
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
                # Update outcome-specific UI elements
                self.highlight_outcome(new_var)
        
        elif role == VariableRole.GROUP:
            new_var = self.group_combo.currentText()
            if new_var and new_var != "Select...":
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
        
        elif role == VariableRole.SUBJECT_ID:
            new_var = self.subject_id_combo.currentText()
            if new_var and new_var != "Select...":
                # Check if this variable is already assigned as an OUTCOME
                if self.column_roles.get(new_var) == VariableRole.OUTCOME:
                    logging.warning(f"Cannot set {new_var} as SUBJECT_ID because it's already assigned as OUTCOME")
                    QMessageBox.warning(self, "Role Conflict", 
                                      f"Cannot set {new_var} as Subject ID because it's already assigned as Outcome.")
                    
                    # Reset the combo box to the current subject_id or to "Select..."
                    current_subject_id = next((col for col, r in self.column_roles.items() 
                                            if r == VariableRole.SUBJECT_ID), None)
                    
                    self.subject_id_combo.blockSignals(True)
                    if current_subject_id:
                        self.subject_id_combo.setCurrentText(current_subject_id)
                    else:
                        self.subject_id_combo.setCurrentText("Select...")
                    self.subject_id_combo.blockSignals(False)
                    
                    return
                
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
        
        elif role == VariableRole.TIME:
            new_var = self.time_combo.currentText()
            if new_var and new_var != "Select...":
                # Check if this variable is already assigned as an OUTCOME
                if self.column_roles.get(new_var) == VariableRole.OUTCOME:
                    logging.warning(f"Cannot set {new_var} as TIME because it's already assigned as OUTCOME")
                    QMessageBox.warning(self, "Role Conflict", 
                                      f"Cannot set {new_var} as Time because it's already assigned as Outcome.")
                    
                    # Reset the combo box to the current time or to "Select..."
                    current_time = next((col for col, r in self.column_roles.items() 
                                      if r == VariableRole.TIME), None)
                    
                    self.time_combo.blockSignals(True)
                    if current_time:
                        self.time_combo.setCurrentText(current_time)
                    else:
                        self.time_combo.setCurrentText("Select...")
                    self.time_combo.blockSignals(False)
                    
                    return
                
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
        
        elif role == VariableRole.EVENT:
            new_var = self.event_combo.currentText()
            if new_var and new_var != "Select...":
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
        
        elif role == VariableRole.PAIR_ID:
            new_var = self.pair_id_combo.currentText()
            if new_var and new_var != "Select...":
                # Clear any existing variable with this role
                for col, existing_role in self.column_roles.items():
                    if existing_role == role:
                        self.column_roles[col] = VariableRole.NONE
                # Set the new variable with this role
                self.column_roles[new_var] = role
        
        
        # Update required fields and test dropdown based on the new role assignments
        self.update_required_fields()
        self.update_test_dropdown()
        self.update_design_type_combo()
        
        # Update the correlation panel if visible
        if hasattr(self, 'correlation_panel') and self.correlation_panel.isVisible():
            self.update_correlation_panel()

    def add_covariate(self):
        """Add a covariate to the list of covariates."""
        new_cov = self.covariates_combo.currentText()
        if new_cov and new_cov != "Select...":
            # Set the new variable as a covariate
            self.column_roles[new_cov] = VariableRole.COVARIATE
            
            # Add to ordered list if not already present
            if new_cov not in self.covariate_order:
                self.covariate_order.append(new_cov)
                
            # Update the UI
            self.update_covariates_display()
            
            # Update required fields and test dropdown based on the new covariate
            self.update_required_fields()
            self.update_test_dropdown()
            self.update_design_type_combo()

    def update_covariates_display(self):
        """Update the display of selected covariates."""
        # Get covariates from the ordered list that are still marked as covariates
        covariates = [col for col in self.covariate_order 
                     if col in self.column_roles and self.column_roles[col] == VariableRole.COVARIATE]
        
        # Also include any covariates in column_roles that aren't in the ordered list
        for col, role in self.column_roles.items():
            if role == VariableRole.COVARIATE and col not in covariates:
                covariates.append(col)
                self.covariate_order.append(col)
        
        # Update the ordered list to match current covariates (remove any that no longer exist)
        self.covariate_order = [col for col in self.covariate_order 
                               if col in self.column_roles and self.column_roles[col] == VariableRole.COVARIATE]
        
        # Save the current selection if any
        current_row = self.covariates_list.currentRow()
        
        # Clear the list
        self.covariates_list.clear()
        
        # Add each covariate
        if covariates:
            self.covariates_list.addItems(covariates)
            
            # Restore selection if possible
            if current_row >= 0 and current_row < self.covariates_list.count():
                self.covariates_list.setCurrentRow(current_row)
        
        # Update the button states
        self._update_covariate_button_states()

    def _update_covariate_button_states(self):
        """Update the enabled state of covariate movement buttons."""
        current_row = self.covariates_list.currentRow()
        count = self.covariates_list.count()
        
        # Enable/disable the up button
        self.move_up_button.setEnabled(current_row > 0)
        
        # Enable/disable the down button
        self.move_down_button.setEnabled(current_row >= 0 and current_row < count - 1)
        
        # Enable/disable the remove button
        self.remove_covariate_button.setEnabled(current_row >= 0)

    def move_covariate_up(self):
        """Move the selected covariate up in the list."""
        current_row = self.covariates_list.currentRow()
        if current_row > 0:
            # Get the covariate name
            covariate = self.covariates_list.item(current_row).text()
            
            # Update the order list
            idx = self.covariate_order.index(covariate)
            self.covariate_order.pop(idx)
            self.covariate_order.insert(idx - 1, covariate)
            
            # Update the UI
            self.update_covariates_display()
            self.covariates_list.setCurrentRow(current_row - 1)
            
            # Update the correlation panel if visible
            if hasattr(self, 'correlation_panel') and self.correlation_panel.isVisible():
                self.update_correlation_panel()

    def move_covariate_down(self):
        """Move the selected covariate down in the list."""
        current_row = self.covariates_list.currentRow()
        if current_row >= 0 and current_row < self.covariates_list.count() - 1:
            # Get the covariate name
            covariate = self.covariates_list.item(current_row).text()
            
            # Update the order list
            idx = self.covariate_order.index(covariate)
            self.covariate_order.pop(idx)
            self.covariate_order.insert(idx + 1, covariate)
            
            # Update the UI
            self.update_covariates_display()
            self.covariates_list.setCurrentRow(current_row + 1)
            
            # Update the correlation panel if visible
            if hasattr(self, 'correlation_panel') and self.correlation_panel.isVisible():
                self.update_correlation_panel()

    def remove_covariate(self):
        """Remove the selected covariate from the list."""
        current_row = self.covariates_list.currentRow()
        if current_row >= 0:
            # Get the covariate name
            covariate = self.covariates_list.item(current_row).text()
            
            # Remove the covariate role
            self.column_roles[covariate] = VariableRole.NONE
            
            # Remove from ordered list
            if covariate in self.covariate_order:
                self.covariate_order.remove(covariate)
            
            # Update the UI
            self.update_covariates_display()
            
            # Update required fields and test dropdown based on the removed covariate
            self.update_required_fields()
            self.update_test_dropdown()
            self.update_design_type_combo()
            
            # Update the correlation panel if visible
            if hasattr(self, 'correlation_panel') and self.correlation_panel.isVisible():
                self.update_correlation_panel()

    def clear_all_assignments(self):
        """Clear all variable role assignments."""
        # Reset all roles
        for col in self.column_roles:
            self.column_roles[col] = VariableRole.NONE
        
        # Clear covariate order
        self.covariate_order = []
        
        # Reset all dropdowns
        self.populate_variable_dropdowns()
        
        # Update UI
        self.update_covariates_display()
        
        # Update required fields and test dropdown
        self.update_required_fields()
        self.update_test_dropdown()
        self.update_design_type_combo()
        
        self.status_bar.showMessage("All variable assignments cleared")

    def populate_variable_dropdowns(self):
        """Populate the variable selection dropdowns based on current column roles."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            return
        
        logging.info("Populating variable dropdowns with current roles:")
        
        # Temporarily block signals to prevent recursive calls
        self.outcome_combo.blockSignals(True)
        self.group_combo.blockSignals(True)
        self.subject_id_combo.blockSignals(True)
        self.time_combo.blockSignals(True)
        self.event_combo.blockSignals(True)
        self.pair_id_combo.blockSignals(True)
        self.covariates_combo.blockSignals(True)
        
        # Clear and repopulate all dropdowns
        self.outcome_combo.clear()
        self.group_combo.clear()
        self.subject_id_combo.clear()
        self.time_combo.clear()
        self.event_combo.clear()
        self.pair_id_combo.clear()
        self.covariates_combo.clear()
        
        # Add placeholder items
        self.outcome_combo.addItem("Select...")
        self.group_combo.addItem("Select...")
        self.subject_id_combo.addItem("Select...")
        self.time_combo.addItem("Select...")
        self.event_combo.addItem("Select...")
        self.pair_id_combo.addItem("Select...")
        self.covariates_combo.addItem("Select...")
        
        # Add all columns
        for col in self.current_dataframe.columns:
            self.outcome_combo.addItem(col)
            self.group_combo.addItem(col)
            self.subject_id_combo.addItem(col)
            self.time_combo.addItem(col)
            self.event_combo.addItem(col)
            self.pair_id_combo.addItem(col)
            self.covariates_combo.addItem(col)
        
        # Create a role-to-variable mapping for easier access
        role_mappings = {}
        for col, role in self.column_roles.items():
            logging.info(f"Role mapping: {col} -> {role}")
            if role not in role_mappings:
                role_mappings[role] = []
            role_mappings[role].append(col)
        
        # Set dropdowns based on current role assignments
        outcome = next((col for col, role in self.column_roles.items() if role == VariableRole.OUTCOME), None)
        group = next((col for col, role in self.column_roles.items() if role == VariableRole.GROUP), None)
        subject_id = next((col for col, role in self.column_roles.items() if role == VariableRole.SUBJECT_ID), None)
        time = next((col for col, role in self.column_roles.items() if role == VariableRole.TIME), None)
        event = next((col for col, role in self.column_roles.items() if role == VariableRole.EVENT), None)
        pair_id = next((col for col, role in self.column_roles.items() if role == VariableRole.PAIR_ID), None)
        
        logging.info(f"Setting dropdowns: outcome={outcome}, group={group}, subject_id={subject_id}, time={time}")
        
        if outcome:
            self.outcome_combo.setCurrentText(outcome)
        if group:
            self.group_combo.setCurrentText(group)
        if subject_id:
            self.subject_id_combo.setCurrentText(subject_id)
        if time:
            self.time_combo.setCurrentText(time)
        if event:
            self.event_combo.setCurrentText(event)
        if pair_id:
            self.pair_id_combo.setCurrentText(pair_id)
        
        # Re-enable signals
        self.outcome_combo.blockSignals(False)
        self.group_combo.blockSignals(False)
        self.subject_id_combo.blockSignals(False)
        self.time_combo.blockSignals(False)
        self.event_combo.blockSignals(False)
        self.pair_id_combo.blockSignals(False)
        self.covariates_combo.blockSignals(False)
        
        # Update covariates display
        self.update_covariates_display()
        self.update_design_type_combo()
    
    def _extract_timepoints_from_outcomes(self, outcome_vars):
        """
        Extract potential timepoints from outcome variable names.
        
        Args:
            outcome_vars: List of outcome variable names
            
        Returns:
            dict: Dictionary of timepoints extracted from outcome names
        """
        timepoints = {}
        time_terms = ["baseline", "followup", "follow-up", "week", "month", "day", "visit", "time", "pre", "post"]
        
        # First pass: Try to extract timepoints from variable names
        for var in outcome_vars:
            # Skip if variable doesn't have underscore
            if "_" not in var:
                continue
                
            parts = var.split("_")
            # Check if the last part contains time information
            last_part = parts[-1].lower()
            
            # Check for numeric patterns like Week6, Day30, etc.
            time_match = re.match(r"([a-z]+)(\d+)", last_part)
            if time_match:
                time_unit = time_match.group(1)
                time_value = time_match.group(2)
                if time_unit in time_terms:
                    timepoint_name = f"{time_unit.capitalize()} {time_value}"
                    if timepoint_name not in timepoints:
                        timepoints[timepoint_name] = []
                    timepoints[timepoint_name].append(var)
                    continue
            
            # Check for common timepoint terms
            for term in time_terms:
                if term in last_part:
                    # Convert to a proper timepoint name
                    timepoint_name = last_part.capitalize()
                    if timepoint_name not in timepoints:
                        timepoints[timepoint_name] = []
                    timepoints[timepoint_name].append(var)
                    break
        
        # Second pass: Group by common prefixes if no timepoints found
        if not timepoints:
            prefixes = {}
            for var in outcome_vars:
                if "_" in var:
                    prefix = var.split("_")[0]
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(var)
            
            # If we have multiple variables with the same prefix, they might be related to different timepoints
            for prefix, vars_list in prefixes.items():
                if len(vars_list) > 1:
                    for var in vars_list:
                        suffix = var.split("_", 1)[1]
                        timepoint_name = suffix.capitalize()
                        if timepoint_name not in timepoints:
                            timepoints[timepoint_name] = []
                        timepoints[timepoint_name].append(var)
        
        return timepoints

    @asyncSlot()
    async def update_study_design(self):
        """Use AI to update the study design based on the current statistical model."""
        # Set button to working state
        self.set_button_working_state(self.update_design_button, True, "Updating...")
        
        if self.current_dataframe is None or self.current_dataframe.empty:
            self.set_button_working_state(self.update_design_button, False)
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        # Check if we have a statistical model
        if not any(role != VariableRole.NONE for role in self.column_roles.values()):
            self.set_button_working_state(self.update_design_button, False)
            QMessageBox.warning(self, "Error", "Please build a statistical model first")
            return
        
        # Get the main window to access study design section
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager') or not hasattr(main_window, 'study_design_section'):
            self.set_button_working_state(self.update_design_button, False)
            QMessageBox.warning(self, "Error", "Could not access study design section")
            return
        
        # Get the current statistical model information
        df = self.current_dataframe
        
        # Get variable roles
        outcome_vars = [col for col, role in self.column_roles.items() if role == VariableRole.OUTCOME]
        group_vars = [col for col, role in self.column_roles.items() if role == VariableRole.GROUP]
        covariate_vars = [col for col, role in self.column_roles.items() if role == VariableRole.COVARIATE]
        subject_id_var = next((col for col, role in self.column_roles.items() if role == VariableRole.SUBJECT_ID), None)
        time_var = next((col for col, role in self.column_roles.items() if role == VariableRole.TIME), None)
        
        # Calculate sample size
        sample_size = len(df)
        # For longitudinal data, count unique subjects if subject_id is available
        subject_count = None
        if subject_id_var and subject_id_var in df.columns:
            subject_count = df[subject_id_var].nunique()
        
        # Extract timepoints from outcome names if no dedicated time variable
        timepoints_info = {}
        if not time_var:
            timepoints_info = self._extract_timepoints_from_outcomes(outcome_vars)
        
        # Get explicit time values if a time variable is present
        time_values = None
        if time_var and time_var in df.columns:
            time_values = sorted(df[time_var].unique().tolist())
        
        # Get group sizes
        group_sizes = {}
        if group_vars and len(group_vars) > 0 and group_vars[0] in df.columns:
            main_group_var = group_vars[0]
            group_counts = df[main_group_var].value_counts().to_dict()
            group_sizes = {str(k): int(v) for k, v in group_counts.items()}
        
        # Get study design type
        study_type = self.design_type_combo.currentData()
        if not study_type:
            self.set_button_working_state(self.update_design_button, False)
            QMessageBox.warning(self, "Error", "Please select a study design type")
            return
        
        design_type = study_type.value.replace('_', '-')
        
        # Get selected test if any
        test_name = self.test_combo.currentText() if self.test_combo.currentIndex() >= 0 else ""
        
        # Extract existing study design data if available
        existing_design = {}
        study_design_section = main_window.study_design_section
        
        # Extract arms data
        existing_arms = []
        for row in range(study_design_section.arms_table.rowCount()):
            arm = {
                "name": study_design_section.arms_table.item(row, 0).text() if study_design_section.arms_table.item(row, 0) else "",
                "interventions": study_design_section.arms_table.item(row, 1).text().split(';') if study_design_section.arms_table.item(row, 1) else [],
                "description": study_design_section.arms_table.item(row, 2).text() if study_design_section.arms_table.item(row, 2) else ""
            }
            # Check if sample size column exists (column 3) and add it if available
            if study_design_section.arms_table.columnCount() > 5 and study_design_section.arms_table.item(row, 5):
                sample_size_text = study_design_section.arms_table.item(row, 5).text()
                if sample_size_text and sample_size_text.strip().isdigit():
                    arm["sample_size"] = int(sample_size_text)
                
            existing_arms.append(arm)
        existing_design["arms"] = existing_arms
        
        # Extract outcomes data
        existing_outcomes = []
        for row in range(study_design_section.outcomes_table.rowCount()):
            outcome = {
                "name": study_design_section.outcomes_table.item(row, 0).text() if study_design_section.outcomes_table.item(row, 0) else "",
                "description": study_design_section.outcomes_table.item(row, 1).text() if study_design_section.outcomes_table.item(row, 1) else "",
                "timepoints": study_design_section.outcomes_table.item(row, 2).text() if study_design_section.outcomes_table.item(row, 2) else "",
                "data_type": study_design_section.outcomes_table.cellWidget(row, 3).currentText() if study_design_section.outcomes_table.cellWidget(row, 3) else "",
                "category": study_design_section.outcomes_table.cellWidget(row, 4).currentText() if study_design_section.outcomes_table.cellWidget(row, 4) else ""
            }
            existing_outcomes.append(outcome)
        existing_design["outcomes"] = existing_outcomes
        
        # Extract covariates data
        existing_covariates = []
        for row in range(study_design_section.covariates_table.rowCount()):
            covariate = {
                "name": study_design_section.covariates_table.item(row, 0).text() if study_design_section.covariates_table.item(row, 0) else "",
                "description": study_design_section.covariates_table.item(row, 1).text() if study_design_section.covariates_table.item(row, 1) else "",
                "data_type": study_design_section.covariates_table.cellWidget(row, 2).currentText() if study_design_section.covariates_table.cellWidget(row, 2) else ""
            }
            existing_covariates.append(covariate)
        existing_design["covariates"] = existing_covariates
        
        # Extract existing timepoints data (if present in the study design section)
        existing_timepoints = []
        if hasattr(study_design_section, 'timepoints_table'):
            for row in range(study_design_section.timepoints_table.rowCount()):
                timepoint = {
                    "name": study_design_section.timepoints_table.item(row, 0).text() if study_design_section.timepoints_table.item(row, 0) else "",
                    "description": study_design_section.timepoints_table.item(row, 1).text() if study_design_section.timepoints_table.item(row, 1) else "",
                    "order": row  # Use the current order in the table
                }
                existing_timepoints.append(timepoint)
        existing_design["timepoints"] = existing_timepoints
        
        # Get the column mapping
        column_mapping = get_column_mapping(df, include_actual=False)

        # Create a prompt for the LLM
        prompt = f"""
        I need to update the study design based on our statistical model and dataset. Please infer the appropriate study design elements but be minimalist and factual - do not make up information that isn't clearly implied by the data.

        Dataset: {json.dumps(column_mapping, indent=2)}
        Total Sample Size: {sample_size} observations
        {f"Unique Subject Count: {subject_count} subjects" if subject_count else ""}
        
        Statistical Model:
        - Study Design Type: {design_type}
        - Outcome Variables: {', '.join(outcome_vars) if outcome_vars else 'None'}
        - Group Variables: {', '.join(group_vars) if group_vars else 'None'}
        - Covariate Variables: {', '.join(covariate_vars) if covariate_vars else 'None'}
        - Subject ID Variable: {subject_id_var if subject_id_var else 'None'}
        - Time Variable: {time_var if time_var else 'None'}
        - Statistical Test: {test_name}
        
        Group Variable Values and Counts (if available):
        {self._get_group_values(df, group_vars)}
        {json.dumps(group_sizes, indent=2) if group_sizes else ""}
        
        Time Information:
        {self._get_time_values(df, time_var)}
        
        Extracted Timepoints from Outcome Names:
        {json.dumps(timepoints_info, indent=2) if timepoints_info else 'None identified'}
        
        Existing Study Design (if any):
        {json.dumps(existing_design, indent=2)}
        
        Based on this information, please provide updated study design elements in JSON format:
        
        1. Study Arms and Interventions (include sample size per arm if available)
        2. Timepoints (measurement occasions)
        3. Outcome Measures with their applicable timepoints
        4. Covariates
        
        Return your response in this JSON format:
        {{
            "arms": [
                {{
                    "name": "arm name",
                    "description": "brief description",
                    "interventions": ["intervention1 (type)", "intervention2 (type)"],
                    "sample_size": 42  # ALWAYS include sample size for each arm based on group counts or calculations
                }}
            ],
            "timepoints": [
                {{
                    "name": "timepoint name (e.g., Baseline, Week 6)",
                    "description": "what happens at this timepoint",
                    "order": 1  # use integers for logical ordering
                }}
            ],
            "outcomes": [
                {{
                    "name": "outcome name",
                    "description": "what this measures",
                    "timepoints": ["Baseline", "Week 6"],  # list of timepoint names that apply to this outcome
                    "data_type": "Continuous/Categorical/etc",
                    "category": "Primary/Secondary",
                    "applicable_arms": ["all arms or specific ones"]
                }}
            ],
            "covariates": [
                {{
                    "name": "covariate name",
                    "description": "what this represents",
                    "data_type": "Continuous/Categorical/etc"
                }}
            ],
            "explanation": "brief explanation of your design choices"
        }}
        
        Please:
        1. When updating existing elements, maintain their information unless the statistical model clearly conflicts with it
        2. Only add new elements if they're clearly indicated by the statistical model
        3. Provide simple, factual descriptions based only on available information
        4. For intervention types, use one of: Drug, Device, Procedure, Behavioral, Other
        5. For outcomes, infer data types from the variable itself
        6. Do not invent specific details not supported by the data
        7. Include all detected timepoints, whether they come from a time variable or are inferred from outcome names
        8. IMPORTANT: Include sample size information for each arm. If specific arm sizes are not known, divide the total sample size appropriately
        9. If the study has only one arm, assign the total sample size to that arm
        """
        
        # Update status bar instead of showing a dialog
        self.status_bar.showMessage("Updating study design with AI. Please wait...")
        
        try:
            # Call LLM API asynchronously
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group(1))
                
                # Get the design elements
                arms = result.get("arms", [])
                timepoints = result.get("timepoints", [])
                outcomes = result.get("outcomes", [])
                covariates = result.get("covariates", [])
                explanation = result.get("explanation", "")
                
                # Ensure arms have sample size information
                if arms:
                    # If no arm has a sample size, add it based on group sizes or total sample
                    if not any('sample_size' in arm for arm in arms):
                        if group_sizes and len(arms) == len(group_sizes):
                            # Match arms to group sizes if possible
                            for i, (group_name, group_size) in enumerate(group_sizes.items()):
                                if i < len(arms):
                                    arms[i]['sample_size'] = group_size
                        elif len(arms) == 1:
                            # If there's only one arm, assign the total sample size
                            arms[0]['sample_size'] = sample_size
                        else:
                            # Distribute sample size evenly
                            equal_size = sample_size // len(arms)
                            for arm in arms:
                                arm['sample_size'] = equal_size
                
                # Confirm with user
                message = f"<b>AI recommends the following updates to your study design:</b><br><br>"
                message += f"<b>Sample Size:</b> {sample_size} observations"
                if subject_count:
                    message += f" ({subject_count} unique subjects)<br><br>"
                else:
                    message += "<br><br>"
                
                if arms:
                    message += f"<b>Arms and Interventions ({len(arms)}):</b><br>"
                    for arm in arms[:3]:  # Show just first three for brevity
                        sample_size_info = f" (n={arm.get('sample_size', 'unknown')})"
                        message += f"- {arm['name']}{sample_size_info}: {len(arm.get('interventions', []))} intervention(s)<br>"
                    if len(arms) > 3:
                        message += f"- (and {len(arms) - 3} more)<br>"
                    message += "<br>"
                
                if timepoints:
                    message += f"<b>Timepoints ({len(timepoints)}):</b><br>"
                    for timepoint in timepoints[:3]:  # Show just first three
                        message += f"- {timepoint['name']} (Order: {timepoint.get('order', 'unknown')})<br>"
                    if len(timepoints) > 3:
                        message += f"- (and {len(timepoints) - 3} more)<br>"
                    message += "<br>"
                
                if outcomes:
                    message += f"<b>Outcome Measures ({len(outcomes)}):</b><br>"
                    for outcome in outcomes[:3]:  # Show just first three
                        timepoints_info = ""
                        if outcome.get('timepoints'):
                            timepoints_info = f" at {', '.join(outcome['timepoints'])}"
                        message += f"- {outcome['name']} ({outcome.get('category', 'Unknown')}){timepoints_info}<br>"
                    if len(outcomes) > 3:
                        message += f"- (and {len(outcomes) - 3} more)<br>"
                    message += "<br>"
                
                if covariates:
                    message += f"<b>Covariates ({len(covariates)}):</b><br>"
                    for covariate in covariates[:3]:  # Show just first three
                        message += f"- {covariate['name']}<br>"
                    if len(covariates) > 3:
                        message += f"- (and {len(covariates) - 3} more)<br>"
                    message += "<br>"
                
                if explanation:
                    message += f"<b>Explanation:</b><br>{explanation}<br><br>"
                
                message += "Apply these updates to your study design?"
                
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Update Study Design?")
                msg_box.setText(message)
                msg_box.setTextFormat(Qt.TextFormat.RichText)
                msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
                
                if msg_box.exec() == QMessageBox.StandardButton.Yes:
                    # Update the study design section
                    self._update_arms_in_study_design(arms, study_design_section)
                    # Update timepoints if the study design section supports it
                    if hasattr(study_design_section, 'timepoints_table') and timepoints:
                        self._update_timepoints_in_study_design(timepoints, study_design_section)
                    self._update_outcomes_in_study_design(outcomes, study_design_section)
                    self._update_covariates_in_study_design(covariates, study_design_section)
                    
                    # Also update the study design in the studies manager
                    self._update_study_design_in_manager(arms, timepoints, outcomes, covariates)
                    
                    # Success message
                    QMessageBox.information(
                        self, "Success", 
                        "Study design has been updated based on the statistical model."
                    )
            else:
                QMessageBox.warning(self, "Error", "Could not parse AI response")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI study design update failed: {str(e)}")
        
        # Reset button state
        self.set_button_working_state(self.update_design_button, False)

    def _update_study_design_in_manager(self, arms, timepoints, outcomes, covariates):
        """
        Update the study design in the StudiesManager based on the updated components
        
        Args:
            arms: List of arm dictionaries
            timepoints: List of timepoint dictionaries
            outcomes: List of outcome dictionaries
            covariates: List of covariate dictionaries
        """
        active_study = self.studies_manager.get_active_study()
        if not active_study or not hasattr(active_study, 'study_design'):
            return
            
        # Get current study design
        study_design = active_study.study_design
        
        # Convert our data to the format expected by the study design
        try:
            # Update arms
            if hasattr(study_design, 'arms'):
                # Clear existing arms
                study_design.arms = []
                # Add updated arms
                from study_model.study_model import Arm, Intervention, InterventionType
                for arm_data in arms:
                    interventions = []
                    for intervention_str in arm_data.get('interventions', []):
                        # Parse intervention strings like "Drug A (Drug)"
                        parts = intervention_str.split('(')
                        name = parts[0].strip()
                        type_str = parts[1].replace(')', '').strip() if len(parts) > 1 else "Other"
                        # Try to convert to enum
                        try:
                            int_type = next((t for t in InterventionType if t.value == type_str), InterventionType.OTHER)
                        except:
                            int_type = InterventionType.OTHER
                            
                        interventions.append(Intervention(name=name, type=int_type))
                        
                    # Create the arm object
                    arm = Arm(
                        name=arm_data.get('name', ''),
                        description=arm_data.get('description', ''),
                        interventions=interventions,
                        cohort_size=arm_data.get('sample_size', None)
                    )
                    study_design.arms.append(arm)
                    
            # Update timepoints if the attribute exists
            if hasattr(study_design, 'timepoints') and timepoints:
                # Clear existing timepoints
                study_design.timepoints = []
                # Add updated timepoints
                from study_model.study_model import StudyTimepoint, TimePoint
                for tp_data in sorted(timepoints, key=lambda x: x.get('order', 999)):
                    # Try to map to TimePoint enum
                    try:
                        point_type = TimePoint.BASELINE  # Default
                        for tp in TimePoint:
                            if tp.value.lower() in tp_data.get('name', '').lower():
                                point_type = tp
                                break
                    except:
                        point_type = TimePoint.OTHER
                        
                    timepoint = StudyTimepoint(
                        name=tp_data.get('name', ''),
                        point_type=point_type,
                        order=tp_data.get('order', 0),
                        description=tp_data.get('description', '')
                    )
                    study_design.timepoints.append(timepoint)
                    
            # Update outcomes
            if hasattr(study_design, 'outcome_measures'):
                # Clear existing outcomes
                study_design.outcome_measures = []
                # Add updated outcomes
                from study_model.study_model import OutcomeMeasure, CFDataType, OutcomeCategory
                for outcome_data in outcomes:
                    # Try to map to enums
                    try:
                        data_type = next((dt for dt in CFDataType if dt.value == outcome_data.get('data_type', '')), CFDataType.CONTINUOUS)
                    except:
                        data_type = CFDataType.CONTINUOUS
                        
                    try:
                        category = next((oc for oc in OutcomeCategory if oc.value == outcome_data.get('category', '')), OutcomeCategory.PRIMARY)
                    except:
                        category = OutcomeCategory.PRIMARY
                        
                    outcome = OutcomeMeasure(
                        name=outcome_data.get('name', ''),
                        description=outcome_data.get('description', ''),
                        timepoints=outcome_data.get('timepoints', []),
                        data_type=data_type,
                        category=category,
                        applicable_arms=outcome_data.get('applicable_arms', [])
                    )
                    study_design.outcome_measures.append(outcome)
                    
            # Update covariates
            if hasattr(study_design, 'covariates'):
                # Clear existing covariates
                study_design.covariates = []
                # Add updated covariates
                from study_model.study_model import Covariate, CFDataType
                for covariate_data in covariates:
                    # Try to map to enum
                    try:
                        data_type = next((dt for dt in CFDataType if dt.value == covariate_data.get('data_type', '')), CFDataType.CONTINUOUS)
                    except:
                        data_type = CFDataType.CONTINUOUS
                        
                    covariate = Covariate(
                        name=covariate_data.get('name', ''),
                        description=covariate_data.get('description', ''),
                        data_type=data_type
                    )
                    study_design.covariates.append(covariate)
            
            # Update the timestamp
            from datetime import datetime
            if hasattr(active_study, 'updated_at'):
                active_study.updated_at = datetime.now().isoformat()
                
        except Exception as e:
            print(f"Error updating study design in StudiesManager: {e}")
            # We don't show an error to the user here since this is just syncing to the manager
    
    def _get_group_values(self, df, group_vars):
        """Helper to get unique values for group variables."""
        result = ""
        for var in group_vars:
            if var in df.columns:
                unique_vals = df[var].unique().tolist()
                result += f"- {var}: {', '.join(map(str, unique_vals))}\n"
        return result
    
    def _get_time_values(self, df, time_var):
        """Helper to get unique values for time variable."""
        if not time_var or time_var not in df.columns:
            return "None"
        unique_vals = df[time_var].unique().tolist()
        return f"Values: {', '.join(map(str, unique_vals))}"
    
    def _update_arms_in_study_design(self, arms, study_design_section):
        """Update the arms table in study design section."""
        # Clear existing arms if we have new ones
        if arms:
            study_design_section.arms_table.setRowCount(0)
            
            # Check if we need to add a sample size column
            if any('sample_size' in arm for arm in arms):
                column_headers = [study_design_section.arms_table.horizontalHeaderItem(i).text()
                                 for i in range(study_design_section.arms_table.columnCount())]
                
                # Look for "Cohort Size" column (the standard column name in study design)
                cohort_size_col = -1
                if "Cohort Size" in column_headers:
                    cohort_size_col = column_headers.index("Cohort Size")
                elif study_design_section.arms_table.columnCount() > 5:
                    # Assuming column 5 is for cohort size based on study_design.py implementation
                    cohort_size_col = 5
            
            # Add each arm to the table
            for arm in arms:
                row_position = study_design_section.arms_table.rowCount()
                study_design_section.arms_table.insertRow(row_position)
                
                # Set arm name and description
                study_design_section.arms_table.setItem(row_position, 0, QTableWidgetItem(arm.get('name', '')))
                interventions_text = '; '.join(arm.get('interventions', []))
                study_design_section.arms_table.setItem(row_position, 1, QTableWidgetItem(interventions_text))
                study_design_section.arms_table.setItem(row_position, 2, QTableWidgetItem(arm.get('description', '')))
                
                # Add empty cells for start date and end date (columns 3 and 4)
                study_design_section.arms_table.setItem(row_position, 3, QTableWidgetItem(''))
                study_design_section.arms_table.setItem(row_position, 4, QTableWidgetItem(''))
                
                # Add sample size to "Cohort Size" column (column 5)
                if 'sample_size' in arm and study_design_section.arms_table.columnCount() > 5:
                    study_design_section.arms_table.setItem(row_position, 5, 
                                                           QTableWidgetItem(str(arm.get('sample_size', ''))))
                
                # Set empty cells for any other columns
                for col in range(6, study_design_section.arms_table.columnCount()):
                    study_design_section.arms_table.setItem(row_position, col, QTableWidgetItem(''))
                
            # Resize columns to content
            study_design_section.arms_table.resizeColumnsToContents()
    
    def _update_outcomes_in_study_design(self, outcomes, study_design_section):
        """Update the outcomes table in study design section."""
        # Clear existing outcomes if we have new ones
        if outcomes:
            study_design_section.outcomes_table.setRowCount(0)
            
            # Add each outcome to the table
            for outcome in outcomes:
                row_position = study_design_section.outcomes_table.rowCount()
                study_design_section.outcomes_table.insertRow(row_position)
                
                # Set outcome properties
                study_design_section.outcomes_table.setItem(row_position, 0, QTableWidgetItem(outcome.get('name', '')))
                study_design_section.outcomes_table.setItem(row_position, 1, QTableWidgetItem(outcome.get('description', '')))
                
                # Set timepoints if available
                timepoints_text = ', '.join(outcome.get('timepoints', []))
                study_design_section.outcomes_table.setItem(row_position, 2, QTableWidgetItem(timepoints_text))
                
                # Data Type ComboBox
                data_type_combo = QComboBox()
                data_type_combo.addItems([dt.value for dt in CFDataType])
                data_type = outcome.get('data_type', '')
                if data_type:
                    index = data_type_combo.findText(data_type)
                    if index >= 0:
                        data_type_combo.setCurrentIndex(index)
                study_design_section.outcomes_table.setCellWidget(row_position, 3, data_type_combo)
                
                # Category ComboBox
                category_combo = QComboBox()
                category_combo.addItems([oc.value for oc in OutcomeCategory])
                category = outcome.get('category', '')
                if category:
                    index = category_combo.findText(category)
                    if index >= 0:
                        category_combo.setCurrentIndex(index)
                study_design_section.outcomes_table.setCellWidget(row_position, 4, category_combo)
                
                # Collection Method ComboBox
                collection_method_combo = QComboBox()
                collection_method_combo.addItems([cm.value for cm in DataCollectionMethod])
                study_design_section.outcomes_table.setCellWidget(row_position, 5, collection_method_combo)
                
                # Set applicable arms
                applicable_arms = ', '.join(outcome.get('applicable_arms', []))
                study_design_section.outcomes_table.setItem(row_position, 6, QTableWidgetItem(applicable_arms))
                
                # Set units
                study_design_section.outcomes_table.setItem(row_position, 7, QTableWidgetItem(outcome.get('units', '')))
                
            # Resize columns to content
            study_design_section.outcomes_table.resizeColumnsToContents()
    
    def _update_covariates_in_study_design(self, covariates, study_design_section):
        """Update the covariates table in study design section."""
        # Clear existing covariates if we have new ones
        if covariates:
            study_design_section.covariates_table.setRowCount(0)
            
            # Add each covariate to the table
            for covariate in covariates:
                row_position = study_design_section.covariates_table.rowCount()
                study_design_section.covariates_table.insertRow(row_position)
                
                # Set covariate properties
                study_design_section.covariates_table.setItem(row_position, 0, QTableWidgetItem(covariate.get('name', '')))
                study_design_section.covariates_table.setItem(row_position, 1, QTableWidgetItem(covariate.get('description', '')))
                
                # Data Type ComboBox
                data_type_combo = QComboBox()
                data_type_combo.addItems([dt.value for dt in CFDataType])
                data_type = covariate.get('data_type', '')
                if data_type:
                    index = data_type_combo.findText(data_type)
                    if index >= 0:
                        data_type_combo.setCurrentIndex(index)
                study_design_section.covariates_table.setCellWidget(row_position, 2, data_type_combo)
                
            # Resize columns to content
            study_design_section.covariates_table.resizeColumnsToContents()
    
    def _update_timepoints_in_study_design(self, timepoints, study_design_section):
        """Update the timepoints table in study design section."""
        # Only proceed if the timepoints_table attribute exists
        if not hasattr(study_design_section, 'timepoints_table'):
            return
            
        # Clear existing timepoints if we have new ones
        if timepoints:
            study_design_section.timepoints_table.setRowCount(0)
            
            # Add each timepoint to the table
            for timepoint in sorted(timepoints, key=lambda x: x.get('order', 999)):
                row_position = study_design_section.timepoints_table.rowCount()
                study_design_section.timepoints_table.insertRow(row_position)
                
                # Set timepoint properties
                study_design_section.timepoints_table.setItem(row_position, 0, QTableWidgetItem(timepoint.get('name', '')))
                study_design_section.timepoints_table.setItem(row_position, 1, QTableWidgetItem(timepoint.get('description', '')))
                
                # Set additional properties if the table has more columns
                for col in range(2, study_design_section.timepoints_table.columnCount()):
                    study_design_section.timepoints_table.setItem(row_position, col, QTableWidgetItem(''))
                
            # Resize columns to content
            study_design_section.timepoints_table.resizeColumnsToContents()
    
    def load_test_from_studies_manager(self):
        """
        Load a previously identified test from the StudiesManager.
        """
        if not hasattr(self, 'studies_manager') or not self.studies_manager:
            self.status_bar.showMessage("No studies manager available")
            return False
        
        test_data = self.studies_manager.get_current_test_data()
        if not test_data:
            self.status_bar.showMessage("No previous test selection found")
            return False
        
        # Load the dataset if needed
        if self.current_dataset_name != test_data.get('dataset_name'):
            datasets = self.studies_manager.get_datasets_from_active_study()
            for name, df in datasets:
                if name == test_data.get('dataset_name'):
                    self.display_dataset(df)
                    self.current_dataset_name = name
                    break
        
        # Set the variable roles
        self.clear_all_assignments()
        
        # Set outcome
        if test_data.get('outcome'):
            outcome_index = self.outcome_combo.findText(test_data['outcome'])
            if outcome_index >= 0:
                self.outcome_combo.setCurrentIndex(outcome_index)
        
        # Set group
        if test_data.get('group'):
            group_index = self.group_combo.findText(test_data['group'])
            if group_index >= 0:
                self.group_combo.setCurrentIndex(group_index)
        
        # Set covariates - now properly handle the order
        if test_data.get('covariates'):
            # Clear the covariate order list
            self.covariate_order = []
            
            # Add covariates in the order they appear in the test data
            for covariate in test_data['covariates']:
                if covariate in self.current_dataframe.columns:
                    # Set the role
                    self.column_roles[covariate] = VariableRole.COVARIATE
                    # Add to the ordered list
                    self.covariate_order.append(covariate)
            
            # Update the UI
            self.update_covariates_display()
        
        # Set subject_id
        if test_data.get('subject_id'):
            subject_id_index = self.subject_id_combo.findText(test_data['subject_id'])
            if subject_id_index >= 0:
                self.subject_id_combo.setCurrentIndex(subject_id_index)
        
        # Set time
        if test_data.get('time'):
            time_index = self.time_combo.findText(test_data['time'])
            if time_index >= 0:
                self.time_combo.setCurrentIndex(time_index)
        
        # Set pair_id
        if test_data.get('pair_id'):
            pair_id_index = self.pair_id_combo.findText(test_data['pair_id'])
            if pair_id_index >= 0:
                self.pair_id_combo.setCurrentIndex(pair_id_index)
        
        # Set test type if available
        if test_data.get('test_key'):
            for i in range(self.test_combo.count()):
                test_key = self.test_combo.itemData(i)
                if test_key == test_data['test_key']:
                    self.test_combo.setCurrentIndex(i)
                    break
        
        self.update_design_description()
        self.status_bar.showMessage(f"Test '{test_data.get('test_name')}' has been loaded from the previous selection")
        
        return True

    def on_mu_value_changed(self, value):
        """Track when the mu value is changed by the user."""
        # Store the value as an attribute to ensure it's saved
        self._current_mu_value = value
        
        # If a test is already selected and it's a one-sample test, consider updating
        if hasattr(self, 'test_combo') and self.test_combo.currentIndex() >= 0:
            current_test = self.test_combo.currentText().lower()
            if any(test in current_test for test in ["one_sample", "one sample"]):
                print(f"Mu value updated to: {value}")
                # Optionally, we could re-run the test automatically here:
                # self.run_statistical_test(quiet=True)

    def on_test_changed(self, index):
        """Handle test selection changes and update UI accordingly."""
        if index < 0:
            return
            
        # Get the current test key and display name
        current_test_key = self.test_combo.currentData()
        current_test_text = self.test_combo.currentText().lower()
        
        # Check for different test types
        is_one_sample_test = "one_sample" in current_test_text or "one sample" in current_test_text
        is_correlation_test = current_test_key in [
            StatisticalTest.PEARSON_CORRELATION.value,
            StatisticalTest.SPEARMAN_CORRELATION.value, 
            StatisticalTest.KENDALL_TAU_CORRELATION.value,
            StatisticalTest.POINT_BISERIAL_CORRELATION.value
        ]
        
        # Handle one-sample test UI
        if hasattr(self, 'mu_input') and hasattr(self, 'mu_label') and hasattr(self, 'test_params_group'):
            self.mu_input.setVisible(is_one_sample_test)
            self.mu_label.setVisible(is_one_sample_test)
            self.test_params_group.setVisible(is_one_sample_test)
            
            # If showing mu input and we have a previously stored value, restore it
            if is_one_sample_test and hasattr(self, '_current_mu_value'):
                self.mu_input.setValue(self._current_mu_value)
        
        # Handle correlation test UI
        if hasattr(self, 'correlation_panel'):
            self.correlation_panel.setVisible(is_correlation_test)
            
            # If showing correlation panel, update it with the current variable roles
            if is_correlation_test:
                self.update_correlation_panel()

    def update_assumption_indicators(self, diagnostics=None):
        """Update the assumption status indicators based on diagnostics."""
        if not diagnostics:
            # Reset to gray/unknown state if no diagnostics provided
            self.clt_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
            self.clt_status_icon.setToolTip("Sample size status unknown")
            
            self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
            self.normality_status_icon.setToolTip("Normality status unknown")
            
            self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
            self.balance_status_icon.setToolTip("Group balance status unknown")
            return
        
        # Update sample size/CLT indicator
        sample_size = diagnostics["sample_size"]["after_na_drop"]
        if sample_size >= 30:
            # Green: CLT applies
            self.clt_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#43A047", size=16))
            self.clt_status_icon.setToolTip(f"Sample size {sample_size} ≥ 30: Central Limit Theorem applies")
        elif sample_size >= 25:
            # Yellow: borderline
            self.clt_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#FB8C00", size=16))
            self.clt_status_icon.setToolTip(f"Sample size {sample_size} is borderline (25-29)")
        else:
            # Red: too small
            self.clt_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#E53935", size=16))
            self.clt_status_icon.setToolTip(f"Sample size {sample_size} < 25: May be too small for reliable results")
        
        # Update normality indicator
        if "outcome" in diagnostics["normality"] and diagnostics["normality"]["outcome"]["normal"] is not None:
            is_normal = diagnostics["normality"]["outcome"]["normal"]
            p_val = diagnostics["normality"]["outcome"]["p_value"]
            
            if is_normal:
                # Green: data is normal
                self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#43A047", size=16))
                self.normality_status_icon.setToolTip(f"Data appears normally distributed (p={p_val:.4f})")
            else:
                # Red or yellow based on sample size
                if sample_size >= 30:
                    # Yellow: not normal but CLT applies
                    self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#FB8C00", size=16))
                    self.normality_status_icon.setToolTip(
                        f"Data not normally distributed (p={p_val:.4f}), but sample size ≥ 30 so CLT applies"
                    )
                else:
                    # Red: not normal and sample size too small
                    self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#E53935", size=16))
                    self.normality_status_icon.setToolTip(
                        f"Data not normally distributed (p={p_val:.4f}) and sample size < 30. Consider non-parametric tests."
                    )
        else:
            # Gray: unknown normality
            self.normality_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
            self.normality_status_icon.setToolTip("Normality status unknown or not applicable")
        
        # Update group balance indicator
        if diagnostics["group_balance"]["group_counts"]:
            is_balanced = diagnostics["group_balance"]["balanced"]
            ratio = diagnostics["group_balance"].get("ratio")
            
            if is_balanced and (ratio is None or ratio < 3):
                # Green: well balanced
                self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#43A047", size=16))
                ratio_text = f" (ratio {ratio:.1f}:1)" if ratio else ""
                self.balance_status_icon.setToolTip(f"Groups are well balanced{ratio_text}")
            elif not is_balanced:
                # Red: group too small
                self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#E53935", size=16))
                self.balance_status_icon.setToolTip(diagnostics["group_balance"]["message"])
            elif ratio and ratio >= 3:
                # Yellow: imbalanced
                self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#FB8C00", size=16))
                self.balance_status_icon.setToolTip(f"Groups are imbalanced (ratio {ratio:.1f}:1)")
        else:
            # Gray: no groups
            self.balance_status_icon.setPixmap(get_indicator_pixmap("circle-fill", color="#9E9E9E", size=16))
            self.balance_status_icon.setToolTip("Group balance status unknown or not applicable")

    def show_assumption_indicators_help(self):
        """Show explanation of the assumption status indicators."""
        help_message = """
        <h3>Statistical Assumption Indicators</h3>
        
        <p>These indicators help you quickly see if your data meets key assumptions for statistical tests:</p>
        
        <p><b>Sample Size:</b></p>
        <ul>
        <li><span style="color:#43A047">●</span> <b>Green:</b> Sample size ≥ 30 (Central Limit Theorem applies)</li>
        <li><span style="color:#FB8C00">●</span> <b>Yellow:</b> Sample size between 25-29 (borderline)</li>
        <li><span style="color:#E53935">●</span> <b>Red:</b> Sample size < 25 (may be too small)</li>
        </ul>
        
        <p><b>Normality:</b></p>
        <ul>
        <li><span style="color:#43A047">●</span> <b>Green:</b> Data is normally distributed</li>
        <li><span style="color:#FB8C00">●</span> <b>Yellow:</b> Data not normal but N ≥ 30 (CLT applies)</li>
        <li><span style="color:#E53935">●</span> <b>Red:</b> Data not normal and N < 30</li>
        </ul>
        
        <p><b>Group Balance:</b></p>
        <ul>
        <li><span style="color:#43A047">●</span> <b>Green:</b> All groups have ≥ 5 observations and ratio < 3:1</li>
        <li><span style="color:#FB8C00">●</span> <b>Yellow:</b> Groups imbalanced (ratio > 3:1)</li>
        <li><span style="color:#E53935">●</span> <b>Red:</b> One or more groups < 5 observations</li>
        <li><span style="color:#9E9E9E">●</span> <b>Gray:</b> No groups in analysis</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Assumption Indicators Help")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(help_message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def show_test_settings(self):
        """Update test execution settings directly without dialog."""
        # Toggle settings directly
        self.test_settings['update_documentation'] = not self.test_settings['update_documentation']
        self.test_settings['update_hypothesis'] = not self.test_settings['update_hypothesis']
        self.test_settings['generate_interpretation'] = not self.test_settings['generate_interpretation']
        
        # Show current settings in status bar
        status = []
        if self.test_settings['update_documentation']:
            status.append("Doc updates ON")
        if self.test_settings['update_hypothesis']:
            status.append("Hypothesis ON") 
        if self.test_settings['generate_interpretation']:
            status.append("AI Interpretation ON")
            
        if not status:
            self.status_bar.showMessage("All test settings are OFF")
        else:
            self.status_bar.showMessage("Settings: " + ", ".join(status))

    @asyncSlot()
    async def show_manual_hypothesis_dialog(self):
        """Generate hypothesis without dialog confirmation."""
        if not self.last_test_result:
            self.status_bar.showMessage("No test results available")
            return

        # Get current test information
        outcome = self.last_test_result.get('outcome')
        group = self.last_test_result.get('group')
        subject_id = self.last_test_result.get('subject_id')
        time = self.last_test_result.get('time')
        test_name = self.last_test_result.get('test_name')
        study_type = self.design_type_combo.currentData()

        # Generate hypothesis
        hypothesis_id = await self.generate_hypothesis_for_test(
            outcome=outcome,
            group=group,
            subject_id=subject_id,
            time=time,
            test_name=test_name,
            study_type=study_type
        )

        if hypothesis_id:
            self.status_bar.showMessage("Hypothesis generated and updated with test results")
        else:
            self.status_bar.showMessage("Failed to generate hypothesis")

    @asyncSlot()
    async def show_manual_interpretation_dialog(self):
        """Interpret test results without dialog confirmation."""
        if not self.last_test_result:
            self.status_bar.showMessage("No test results available")
            return

        test_key = self.last_test_result.get('test_key')
        result = self.last_test_result.get('result')

        if not test_key or not result:
            self.status_bar.showMessage("Test results incomplete")
            return

        # Call LLM for interpretation
        await self.interpret_results_with_llm(test_key, result)
        self.status_bar.showMessage("Test results interpretation updated")

    def clear_all(self):
        """Reset all selections and results."""
        # Clear variable assignments
        self.clear_all_assignments()
        
        # Clear test results
        self.test_results = {}
        self.last_test_result = None
        
        # Reset test selection
        if self.test_combo.count() > 0:
            self.test_combo.setCurrentIndex(0)
            
        # Clear results display
        if hasattr(self, 'results_tree'):
            self.results_tree.clear()
            
        # Reset assumptions indicators
        self.update_assumption_indicators()
        
        # Clear outcome selector
        if self.outcome_selector.count() > 0:
            self.outcome_selector.setCurrentIndex(0)
            
        # Update status
        self.status_bar.showMessage("All selections and results cleared")

    def get_test_recommendations(self, study_type=None):
        """Get comprehensive test recommendations using the TestSelectionEngine."""
        if self.current_dataframe is None or self.current_dataframe.empty:
            self.status_bar.showMessage("No dataset loaded")
            return None
            
        # Use current study type if not provided
        if study_type is None:
            study_type = self.design_type_combo.currentData()
            if not study_type:
                self.status_bar.showMessage("No study design selected")
                return None
        
        # Create the engine
        from data.selection.test_selection_engine import TestSelectionEngine
        engine = TestSelectionEngine(self.current_dataframe, self.column_roles)
        
        # Run all checks
        engine.run_all_checks(study_type)
        
        # Get recommendations
        recommendations = engine.get_test_recommendations(top_n=3)
        
        # Update UI based on recommendations
        if len(recommendations["top_recommendations"]) > 0:
            recommended_test = recommendations["top_recommendations"][0]["test"]
            
            # Find the test in the dropdown and select it
            for i in range(self.test_combo.count()):
                if self.test_combo.itemData(i) == recommended_test:
                    self.test_combo.setCurrentIndex(i)
                    break
                
            # Show status message
            test_name = recommendations["top_recommendations"][0]["name"]
            self.status_bar.showMessage(f"Recommended test: {test_name}")
        else:
            self.status_bar.showMessage("No suitable statistical tests found for the current configuration")
        
        return recommendations

    def set_outcome_variable(self, outcome_var):
        """Set the outcome variable, update UI and column roles."""
        if not outcome_var or outcome_var == "Select...":
            return
        
        logging.info(f"Setting outcome variable: {outcome_var}")
        
        # Clear any existing outcome variable assignments
        for col in self.column_roles:
            if self.column_roles[col] == VariableRole.OUTCOME:
                self.column_roles[col] = VariableRole.NONE
        
        # Set the new outcome variable
        self.column_roles[outcome_var] = VariableRole.OUTCOME
        
        # Update dropdown
        if self.outcome_combo.currentText() != outcome_var:
            self.outcome_combo.setCurrentText(outcome_var)
        
        # Update test dropdown
        self.update_test_dropdown()
        
        # Update the design type combo
        self.update_design_type_combo()
        
        # Update outcome selector
        self.update_outcome_selector()
    
    def set_group_variable(self, group_var):
        """Set the group variable, update UI and column roles."""
        if not group_var or group_var == "Select...":
            return
        
        logging.info(f"Setting group variable: {group_var}")
        
        # Clear any existing group variable assignments
        for col in self.column_roles:
            if self.column_roles[col] == VariableRole.GROUP:
                self.column_roles[col] = VariableRole.NONE
        
        # Set the new group variable
        self.column_roles[group_var] = VariableRole.GROUP
        
        # Update dropdown if needed
        if self.group_combo.currentText() != group_var:
            self.group_combo.setCurrentText(group_var)
        
        # Update test dropdown
        self.update_test_dropdown()
        
        # Update the design type combo
        self.update_design_type_combo()
    
    def set_subject_id_variable(self, subject_id_var):
        """Set the subject ID variable, update UI and column roles."""
        if not subject_id_var or subject_id_var == "Select...":
            return
        
        logging.info(f"Setting subject ID variable: {subject_id_var}")
        
        # Clear any existing subject ID variable assignments
        for col in self.column_roles:
            if self.column_roles[col] == VariableRole.SUBJECT_ID:
                self.column_roles[col] = VariableRole.NONE
        
        # Important check: Don't let a variable be both an outcome and subject_id
        if self.column_roles.get(subject_id_var) == VariableRole.OUTCOME:
            logging.warning(f"Cannot set {subject_id_var} as SUBJECT_ID because it's already set as OUTCOME")
            QMessageBox.warning(self, "Role Conflict", 
                               f"Cannot set {subject_id_var} as Subject ID because it's already set as Outcome.")
            
            # Reset the dropdown to the previously selected value or to "Select..."
            self.populate_variable_dropdowns()
            return
        
        # Set the new subject ID variable
        self.column_roles[subject_id_var] = VariableRole.SUBJECT_ID
        
        # Update dropdown if needed
        if self.subject_id_combo.currentText() != subject_id_var:
            self.subject_id_combo.setCurrentText(subject_id_var)
        
        # Update test dropdown
        self.update_test_dropdown()
        
        # Update the design type combo
        self.update_design_type_combo()
    
    def set_time_variable(self, time_var):
        """Set the time variable, update UI and column roles."""
        if not time_var or time_var == "Select...":
            return
        
        logging.info(f"Setting time variable: {time_var}")
        
        # Clear any existing time variable assignments
        for col in self.column_roles:
            if self.column_roles[col] == VariableRole.TIME:
                self.column_roles[col] = VariableRole.NONE
        
        # Important check: Don't let a variable be both an outcome and time
        if self.column_roles.get(time_var) == VariableRole.OUTCOME:
            logging.warning(f"Cannot set {time_var} as TIME because it's already set as OUTCOME")
            QMessageBox.warning(self, "Role Conflict", 
                               f"Cannot set {time_var} as Time because it's already set as Outcome.")
            
            # Reset the dropdown to the previously selected value or to "Select..."
            self.populate_variable_dropdowns()
            return
        
        # Set the new time variable
        self.column_roles[time_var] = VariableRole.TIME
        
        # Update dropdown if needed
        if self.time_combo.currentText() != time_var:
            self.time_combo.setCurrentText(time_var)
        
        # Update test dropdown
        self.update_test_dropdown()
        
        # Update the design type combo
        self.update_design_type_combo()
        
    def set_event_variable(self, event_var):
        """Set the event variable, update UI and column roles."""
        if not event_var or event_var == "Select...":
            return
        
        logging.info(f"Setting event variable: {event_var}")
        
        # Clear any existing event variable assignments
        for col in self.column_roles:
            if self.column_roles[col] == VariableRole.EVENT:
                self.column_roles[col] = VariableRole.NONE
        
        # Important check: Don't let a variable be both an outcome and event
        if self.column_roles.get(event_var) == VariableRole.OUTCOME:
            logging.warning(f"Cannot set {event_var} as EVENT because it's already set as OUTCOME")
            QMessageBox.warning(self, "Role Conflict", 
                               f"Cannot set {event_var} as Event because it's already set as Outcome.")
            
            # Reset the dropdown to the previously selected value or to "Select..."
            self.populate_variable_dropdowns()
            return
        
        # Set the new event variable
        self.column_roles[event_var] = VariableRole.EVENT
        
        # Update dropdown if needed
        if self.event_combo.currentText() != event_var:
            self.event_combo.setCurrentText(event_var)
        
        # Update test dropdown
        self.update_test_dropdown()
        
        # Update the design type combo
        self.update_design_type_combo()

    def determine_possible_study_designs(self, df, column_roles):
        """
        Determine possible study designs based on the assigned variable roles.
        Returns a dict with the recommended design, explanation, and confidence.
        """
        # Initialize empty lists for each variable role
        outcomes = []
        groups = []
        covariates = []
        subject_ids = []
        times = []
        pair_ids = []
        events = []
        
        # Populate the lists based on assigned roles
        for col, role in column_roles.items():
            if role == "OUTCOME":
                outcomes.append(col)
            elif role == "GROUP":
                groups.append(col)
            elif role == "COVARIATE":
                covariates.append(col)
            elif role == "SUBJECT_ID":
                subject_ids.append(col)
            elif role == "TIME":
                times.append(col)
            elif role == "PAIR_ID":
                pair_ids.append(col)
            elif role == "EVENT":
                events.append(col)
        
        # Log variable assignments for debugging
        logging.info(f"Outcomes: {outcomes}")
        logging.info(f"Groups: {groups}")
        logging.info(f"Subject IDs: {subject_ids}")
        logging.info(f"Times: {times}")
        logging.info(f"Events: {events}")
        
        # Check for survival analysis design
        if outcomes and groups and events:
            logging.info("Detected survival analysis design (outcome, group, and event variables)")
            return {
                "design": "survival_analysis",
                "recommendation": "Survival analysis design",
                "explanation": "This appears to be a survival analysis design with time-to-event outcome, group comparison, and event indicator.",
                "confidence": "high"
            }
        
        # Check for one-sample design first (prioritize)
        if outcomes and not groups and not subject_ids and not times:
            logging.info("Detected one-sample design (just outcome variables, no groups or time)")
            return {
                "design": "one_sample",
                "recommendation": "One-sample design",
                "explanation": "This appears to be a one-sample design since you have only outcome variables without grouping or time variables.",
                "confidence": "high"
            }
        
        # Between-subjects design (outcome + group)
        if outcomes and groups and not times:
            logging.info("Detected between-subjects design (outcome + group)")
            return {
                "design": "between_subjects",
                "recommendation": "Between-subjects design",
                "explanation": "This appears to be a between-subjects design with different participants in each group.",
                "confidence": "high"
            }
        
        # Within-subjects design (outcome + subject_id + time)
        if outcomes and subject_ids and times:
            # Check if multiple time points per subject exist
            if df is not None and len(subject_ids) > 0 and len(times) > 0:
                subject_id = subject_ids[0]
                time_var = times[0]
                # Check if subjects have multiple time points
                has_multiple_timepoints = False
                try:
                    multiple_times = df.groupby(subject_id)[time_var].nunique()
                    has_multiple_timepoints = (multiple_times > 1).any()
                except:
                    has_multiple_timepoints = False
                
                if has_multiple_timepoints:
                    if not groups:
                        logging.info("Detected within-subjects design (outcome + subject_id + time with multiple timepoints)")
                        return {
                            "design": "within_subjects",
                            "recommendation": "Within-subjects design",
                            "explanation": "This appears to be a within-subjects design where the same participants are measured at different time points.",
                            "confidence": "high"
                        }
                    else:
                        logging.info("Detected mixed design (outcome + group + subject_id + time with multiple timepoints)")
                        return {
                            "design": "mixed",
                            "recommendation": "Mixed design",
                            "explanation": "This appears to be a mixed design with both between-subjects and within-subjects factors.",
                            "confidence": "high"
                        }
            
        # Cross-over design (outcome + subject_id + time + group)
        if outcomes and subject_ids and times and groups:
            logging.info("Detected possible cross-over design (outcome + subject_id + time + group)")
            # Additional check for cross-over: subjects should receive all treatments
            if df is not None and len(subject_ids) > 0 and len(groups) > 0:
                try:
                    subject_id = subject_ids[0]
                    group_var = groups[0]
                    # Check if subjects have multiple group assignments
                    multiple_groups = df.groupby(subject_id)[group_var].nunique()
                    has_multiple_groups = (multiple_groups > 1).any()
                    
                    if has_multiple_groups:
                        return {
                            "design": "cross_over",
                            "recommendation": "Cross-over design",
                            "explanation": "This appears to be a cross-over design where participants receive multiple treatments in sequence.",
                            "confidence": "high"
                        }
                except:
                    pass
        
        # If we reach here and have an outcome, default to one-sample design
        if outcomes:
            logging.info("Defaulting to one-sample design as fallback")
            return {
                "design": "one_sample",
                "recommendation": "One-sample design (default)",
                "explanation": "Based on the available variables, a one-sample design is the most appropriate default.",
                "confidence": "medium"
            }
        
        # No clear design can be determined
        return {
            "design": "one_sample",  # Default to simplest design
            "recommendation": "Unable to determine design",
            "explanation": "Could not determine a suitable study design based on the available variables.",
            "confidence": "low"
        }

    def update_correlation_panel(self):
        """Update the correlation panel with the current variable assignments."""
        if not hasattr(self, 'correlation_panel'):
            return
            
        # Get the current variable assignments
        outcome = next((col for col, role in self.column_roles.items() 
                      if role == VariableRole.OUTCOME), "Not set")
        
        # Get the second variable (first covariate from ordered list, or group)
        second_var = "Not set"
        ordered_covariates = [col for col in self.covariate_order 
                             if col in self.column_roles and self.column_roles[col] == VariableRole.COVARIATE]
        
        if ordered_covariates:
            second_var = ordered_covariates[0]
        else:
            # Fallback to group if no covariates
            second_var = next((col for col, role in self.column_roles.items() 
                            if role == VariableRole.GROUP), "Not set")
        
        # Update the labels
        self.correlation_x_label.setText(f"X Variable: {outcome}")
        self.correlation_y_label.setText(f"Y Variable: {second_var}")
        
        # Add validation tooltips
        if outcome == "Not set":
            self.correlation_x_label.setStyleSheet("color: red;")
            self.correlation_x_label.setToolTip("Select an outcome variable")
        else:
            self.correlation_x_label.setStyleSheet("")
            self.correlation_x_label.setToolTip("")
            
        if second_var == "Not set":
            self.correlation_y_label.setStyleSheet("color: red;")
            self.correlation_y_label.setToolTip("Select a covariate or group variable")
        else:
            self.correlation_y_label.setStyleSheet("")
            self.correlation_y_label.setToolTip("")

    def swap_correlation_variables(self):
        """Swap the X and Y variables for correlation analysis."""
        # Get the current variable assignments
        outcome = next((col for col, role in self.column_roles.items() 
                  if role == VariableRole.OUTCOME), None)
        
        # Get the second variable from the ordered covariate list or group
        ordered_covariates = [col for col in self.covariate_order 
                         if col in self.column_roles and self.column_roles[col] == VariableRole.COVARIATE]
        
        if ordered_covariates:
            second_var = ordered_covariates[0]
            using_covariate = True
        else:
            # Fallback to group if no covariates
            second_var = next((col for col, role in self.column_roles.items() 
                            if role == VariableRole.GROUP), None)
            using_covariate = False
        
        # If we don't have both variables, can't swap
        if not outcome or not second_var:
            QMessageBox.warning(self, "Cannot Swap", 
                              "Both X and Y variables must be selected to swap them.")
            return
        
        # Clear the current roles
        self.column_roles[outcome] = VariableRole.NONE
        
        if using_covariate:
            # Remove from covariate order
            self.covariate_order.remove(second_var)
            
            self.column_roles[second_var] = VariableRole.NONE
            # Set the second variable as outcome
            self.column_roles[second_var] = VariableRole.OUTCOME
            # Set the original outcome as covariate
            self.column_roles[outcome] = VariableRole.COVARIATE
            
            # Add the original outcome to the start of the covariate order
            self.covariate_order.insert(0, outcome)
        else:  # Using group
            self.column_roles[second_var] = VariableRole.NONE
            # Set the group as outcome
            self.column_roles[second_var] = VariableRole.OUTCOME
            # Set the original outcome as group
            self.column_roles[outcome] = VariableRole.GROUP
        
        # Update the UI
        self.populate_variable_dropdowns()
        self.update_correlation_panel()

    def refresh_hypotheses(self):
        """Refresh the list of available hypotheses from the studies manager."""
        # Clear existing items
        self.hypothesis_combo.clear()
        
        # Get main window to access studies manager
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager') or not main_window.studies_manager:
            self.hypothesis_combo.addItem("No studies manager", None)
            return
            
        # Get active study
        active_study = main_window.studies_manager.get_active_study()
        if not active_study:
            self.hypothesis_combo.addItem("No active study", None)
            return
            
        # Get hypotheses
        hypotheses = main_window.studies_manager.get_study_hypotheses()
        if not hypotheses:
            self.hypothesis_combo.addItem("No hypotheses available", None)
            return
            
        # Add a placeholder item
        self.hypothesis_combo.addItem("-- Create New Hypothesis --", None)
        
        # Add hypotheses to dropdown
        for hypothesis in hypotheses:
            title = hypothesis.get('title', 'Untitled Hypothesis')
            # Include outcome in display text if available
            outcome = hypothesis.get('outcome_variables', '')
            if outcome:
                title = f"{title} ({outcome})"
            self.hypothesis_combo.addItem(title, hypothesis.get('id'))
            
        self.status_bar.showMessage(f"Loaded {len(hypotheses)} hypotheses")

    def find_matching_hypothesis(self, outcome, dataset_name=None):
        """
        Find a hypothesis that matches the current outcome and dataset.
        
        Args:
            outcome: The outcome variable name
            dataset_name: The name of the dataset (optional)
            
        Returns:
            Dict: The matching hypothesis or None if no match found
        """
        # Get main window to access studies manager
        main_window = self.window()
        if not hasattr(main_window, 'studies_manager') or not main_window.studies_manager:
            print("No studies manager available when finding matching hypothesis")
            return None
    
        # Get active study
        active_study = main_window.studies_manager.get_active_study()
        if not active_study:
            print("No active study available when finding matching hypothesis")
            return None
    
        # Get hypotheses
        hypotheses = main_window.studies_manager.get_study_hypotheses()
        if not hypotheses:
            print(f"No hypotheses found in active study when searching for match to '{outcome}'")
            return None
    
        print(f"Searching for hypothesis matching outcome '{outcome}' among {len(hypotheses)} hypotheses")
    
        # Look for matching hypothesis based on outcome
        matching_hypotheses = []
        for hypothesis in hypotheses:
            hyp_outcome = hypothesis.get('outcome_variables', '')
            hyp_related_outcome = hypothesis.get('related_outcome', '')
            hyp_title = hypothesis.get('title', 'Untitled')
    
            # Debug each hypothesis being examined
            print(f"Checking hypothesis '{hyp_title}' with outcome '{hyp_outcome}', related outcome '{hyp_related_outcome}'")
    
            # Check several ways the outcome might match
            outcome_match = False
    
            # 1. Case-insensitive matching on outcome_variables
            if isinstance(hyp_outcome, str) and hyp_outcome.lower() == outcome.lower():
                print(f"Found case-insensitive match on outcome_variables: {hyp_title}")
                outcome_match = True
    
            # 2. Check for exact ID match (for compatibility with planning module)
            elif hypothesis.get('id') == outcome:  # If outcome is actually an ID
                print(f"Found ID match: {hyp_title}")
                return hypothesis
    
            # 3. Check related_outcome field which could be used instead
            elif isinstance(hyp_related_outcome, str) and hyp_related_outcome.lower() == outcome.lower():
                print(f"Found match on related_outcome: {hyp_title}")
                outcome_match = True
    
            if outcome_match:
                matching_hypotheses.append(hypothesis)
    
        if not matching_hypotheses:
            print(f"No matching hypotheses found for outcome '{outcome}'")
            return None
    
        print(f"Found {len(matching_hypotheses)} matching hypotheses for outcome '{outcome}'")
    
        # If we have dataset name, filter by dataset name as well
        if dataset_name and len(matching_hypotheses) > 1:
            dataset_matches = []
            for hypothesis in matching_hypotheses:
                hyp_dataset = hypothesis.get('dataset_name', '')
                hyp_title = hypothesis.get('title', 'Untitled')
    
                if hyp_dataset == dataset_name:
                    print(f"Found dataset match: {hyp_title} with dataset {dataset_name}")
                    dataset_matches.append(hypothesis)
    
            if dataset_matches:
                print(f"Returning hypothesis with matching dataset: {dataset_matches[0].get('title')}")
                return dataset_matches[0]
    
        # Return the first match if multiple exist
        if matching_hypotheses:
            print(f"Returning first matching hypothesis: {matching_hypotheses[0].get('title')}")
        return matching_hypotheses[0] if matching_hypotheses else None

    def set_studies_manager(self, studies_manager):
        """Set the studies manager reference."""
        self.studies_manager = studies_manager
        # Refresh the hypotheses dropdown when studies manager is set
        self.refresh_hypotheses()

    def _prepare_for_llm(self, result, purpose="interpretation"):
        """
        Intelligently prepare test results for LLM processing based on a specific purpose.
        Instead of blindly recursing through the entire structure, this targets known
        problematic elements to create a concise representation suitable for LLM processing.
        
        Args:
            result: Test result dictionary
            purpose: "interpretation" or "hypothesis" to customize detail level
            
        Returns:
            Dict with essential information and problematic elements replaced with summaries
        """
        if not result:
            return {}
            
        # Create a clean copy to avoid modifying the original
        minimal = {}
        
        # 1. Essential test information - always include
        minimal["test"] = result.get('test', result.get('test_name', 'Unknown Test'))
        minimal["significant"] = result.get('significant', False)
        
        # Get p-value from any of the common keys
        for p_key in ['p_value', 'p', 'p-val']:
            if p_key in result:
                minimal["p_value"] = result[p_key]
                break
        
        # 2. Variables information - essential context
        if 'variables' in result:
            minimal["variables"] = {
                "outcome": result["variables"].get("outcome", ""),
                "group": result["variables"].get("group", "")
            }
            
            # Add additional variables only if they exist and are simple types
            for var_key in ["subject_id", "time", "pair_id"]:
                if var_key in result["variables"] and isinstance(result["variables"][var_key], (str, int, float, bool)):
                    minimal["variables"][var_key] = result["variables"][var_key]
        
        # 3. Add key statistics - essential for interpretation
        stat_keys = ['statistic', 'df', 'cohens_d', 'hedges_g', 'eta_squared', 
                    'mean_difference', 'effect_size', 'effect_size_r', 
                    'ci_lower', 'ci_upper', 'effect_magnitude']
                    
        for key in stat_keys:
            if key in result and isinstance(result[key], (int, float, str, bool, list)):
                # Additional check for lists to ensure they're not too large
                if isinstance(result[key], list) and len(result[key]) > 20:
                    minimal[key] = f"List with {len(result[key])} items (truncated)"
                else:
                    minimal[key] = result[key]
        
        # 4. Group means (common in many tests)
        if 'group_means' in result:
            if isinstance(result['group_means'], dict) and len(result['group_means']) < 20:
                minimal['group_means'] = {}
                for k, v in result['group_means'].items():
                    # Convert numpy values to Python native types
                    if hasattr(v, 'item'):
                        minimal['group_means'][k] = float(v)
                    else:
                        minimal['group_means'][k] = v
            else:
                minimal['group_means'] = "Group means available but too large for display"
        
        # 5. Handle assumptions - especially important for interpretation
        if purpose == "interpretation" and 'assumptions' in result:
            minimal['assumptions_summary'] = {}
            
            # Extract just the essential information from assumptions
            if isinstance(result['assumptions'], dict):
                for name, data in result['assumptions'].items():
                    if isinstance(data, dict):
                        minimal['assumptions_summary'][name] = {
                            "satisfied": data.get("satisfied", None),
                            "description": data.get("description", ""),
                            "message": data.get("message", "")
                        }
        
        # 6. Add warnings (but limit quantity)
        if 'warnings' in result and isinstance(result['warnings'], list):
            max_warnings = 3 if purpose == "interpretation" else 2
            minimal['warnings'] = result['warnings'][:max_warnings] if result['warnings'] else []
        
        # 7. For hypothesis generation, be even more minimal
        if purpose == "hypothesis":
            # Remove any keys that aren't absolutely essential
            minimal = {k: v for k, v in minimal.items() if k in 
                     ["test", "significant", "p_value", "variables", "effect_size", "mean_difference"]}
            
            # Include extremely basic group_means if available
            if 'group_means' in result and isinstance(result['group_means'], dict):
                minimal['group_means'] = {k: v for k, v in result['group_means'].items()}
        
        # 8. Handle specific known problematic keys
        known_large_keys = ['model', 'residuals', 'fit', 'data', 'diagnostics']
        for key in known_large_keys:
            if key in result:
                if key == 'model':
                    model_type = result[key].__class__.__name__ if hasattr(result[key], '__class__') else 'Unknown'
                    minimal[key] = f"Statistical model of type {model_type}"
                elif key == 'diagnostics' and purpose == "interpretation":
                    # For interpretation, include a summary of diagnostics
                    if isinstance(result[key], dict):
                        minimal['diagnostics_summary'] = {
                            "sample_size": result[key].get("sample_size", {}).get("after_na_drop", 0),
                            "normality": result[key].get("normality", {}).get("outcome", {}).get("normal", None),
                            "warnings_count": len(result[key].get("warnings", []))
                        }
                else:
                    # For other large structures, just note their presence
                    minimal[key] = f"{key.capitalize()} data available but excluded for conciseness"
        
        # IMPORTANT: Sanitize all values in the minimal dictionary to ensure they're JSON serializable
        # This will convert NumPy types to standard Python types
        minimal = self._sanitize_for_json(minimal)
        
        return minimal
        
    @asyncSlot()
    async def interpret_results_with_llm(self, test_key, result):
        """Use LLM to interpret statistical test results and display them."""
        try:
            # Format the results for the LLM
            test_name = result.get('test', result.get('test_name', 'Statistical Test'))
            test_success = not result.get('error') and result.get('success', True)
            p_value = result.get('p_value', result.get('p', result.get('p-val', None)))
            significant = result.get('significant', False)
            
            # Get variable information
            variables = result.get('variables', {})
            outcome = variables.get('outcome', 'unknown')
            group = variables.get('group', 'N/A')
            
            # Use the targeted preparation method for LLM interpretation
            minimal_result = self._prepare_for_llm(result, purpose="interpretation")
            
            # Create the prompt for the LLM
            prompt = f"""
            Analyze these statistical test results and provide a clear interpretation.
            
            Test Information:
            - Test: {test_name}
            - Success: {test_success}
            - P-value: {p_value}
            - Significant: {significant}
            - Outcome variable: {outcome}
            - Group variable: {group}
            
            Test Results:
            {json.dumps(minimal_result, indent=2, default=str)}
            
            Please provide a concise, clear interpretation of these results that would help a researcher understand the findings.
            Your response should be in JSON format with the following structure:
            {{
                "title": "A clear, one-line summary of the result",
                "significant": true or false based on p-value and test result,
                "conclusion": "A one-sentence definitive conclusion about the outcome",
                "key_points": [
                    "Point 1 about the result",
                    "Point 2 about the result",
                    "Point 3 about the result"
                ],
                "effect_size_interpretation": "Brief comment on effect size if available",
                "recommendations": "A brief recommendation for the researcher",
                "cautions": "Any cautions about interpretation"
            }}
            
            Be precise, factual, and direct in your interpretation. Use statistical language appropriately.
            """
            
            # Call the LLM
            response = await call_llm_async(prompt)
            
            # Parse the JSON response
            json_str = re.search(r'({.*})', response, re.DOTALL)
            if json_str:
                interpretation = json.loads(json_str.group(1))
                self.display_formatted_results(test_key, result, interpretation)
            else:
                # Fallback to basic display if JSON parsing fails
                self.display_basic_results(test_key, result)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Fallback to basic display if LLM call fails
            self.display_basic_results(test_key, result)
            
    @asyncSlot()
    async def generate_hypothesis_for_test(self, outcome, group=None, subject_id=None, time=None, test_name=None, study_type=None, hypothesis_id=None):
        """Generate a hypothesis using LLM based on the current test setup and test results.
        
        Args:
            outcome: The outcome variable name
            group: The group variable name (optional)
            subject_id: The subject ID variable name (optional)
            time: The time variable name (optional)
            test_name: The name of the test (optional, uses current test if not provided)
            study_type: The study design type (optional, uses current design if not provided)
            hypothesis_id: Direct ID of hypothesis to update (optional, skips matching if provided)
            
        Returns:
            Dict: The hypothesis data if successful, None otherwise
        """
        try:
            if self.current_dataframe is None or self.current_dataframe.empty:
                return None
                
            df = self.current_dataframe
            
            # Get test and design details
            test_name = test_name or self.test_combo.currentText()
            study_type_val = study_type.value if study_type else self.design_type_combo.currentData().value
            
            # Print debugging information about the current state
            print(f"\n=== Generating hypothesis for test ===")
            print(f"Outcome: {outcome}")
            print(f"Dataset: {self.current_name}")
            print(f"Test: {test_name}")
            print(f"Study design: {study_type_val}")
            
            # Get the main window to access studies manager
            main_window = self.window()
            if not hasattr(main_window, 'studies_manager'):
                print("Error: No studies manager available")
                return None
                
            # Check for existing matching hypothesis
            matching_hypothesis = None
            
            # 1. If hypothesis_id is provided, use it directly
            if hypothesis_id:
                print(f"Using directly provided hypothesis ID: {hypothesis_id}")
                matching_hypothesis = main_window.studies_manager.get_hypothesis(hypothesis_id)
                if matching_hypothesis:
                    self.status_bar.showMessage(f"Updating specified hypothesis: {matching_hypothesis.get('title')}")
                    print(f"Found hypothesis with ID {hypothesis_id}: {matching_hypothesis.get('title')}")
                else:
                    print(f"Warning: Could not find hypothesis with ID {hypothesis_id}")
            
            # 2. If no direct ID or matching hypothesis not found, check dropdown selection
            if not matching_hypothesis and self.hypothesis_combo.currentData():
                hyp_id = self.hypothesis_combo.currentData()
                print(f"Checking hypothesis selected in dropdown: {hyp_id}")
                potential_match = main_window.studies_manager.get_hypothesis(hyp_id)
                
                # Only use the selected hypothesis if its outcome variable matches the current one
                if potential_match:
                    hyp_outcome = potential_match.get('outcome_variables', '')
                    hyp_related_outcome = potential_match.get('related_outcome', '')
                    
                    print(f"Dropdown hypothesis outcome: '{hyp_outcome}', related outcome: '{hyp_related_outcome}'")
                    print(f"Current outcome: '{outcome}'")
                    
                    # Only consider it a match if the outcome variables match
                    if (hyp_outcome and hyp_outcome.lower() == outcome.lower()) or \
                       (hyp_related_outcome and hyp_related_outcome.lower() == outcome.lower()):
                        matching_hypothesis = potential_match
                        self.status_bar.showMessage(f"Using selected hypothesis: {matching_hypothesis.get('title')}")
                        print(f"Found matching hypothesis from dropdown: {matching_hypothesis.get('title')}")
                    else:
                        print(f"Ignoring selected hypothesis because outcome doesn't match: {potential_match.get('title')}")
                        print(f"Selected hypothesis outcome: '{hyp_outcome}', Current outcome: '{outcome}'")
            
            # 3. If still no match, try to find by outcome and dataset
            # ONLY if no hypothesis_id was explicitly provided
            if not matching_hypothesis and not hypothesis_id:
                print(f"Searching for matching hypothesis by outcome and dataset...")
                # Try to find a matching hypothesis based on outcome and dataset
                matching_hypothesis = self.find_matching_hypothesis(outcome, self.current_name)
                if matching_hypothesis:
                    print(f"Found matching hypothesis by outcome and dataset: {matching_hypothesis.get('title')}")
                    
                    # First confirm with LLM if this hypothesis is a good match
                    confirm_prompt = f"""
                    I found an existing hypothesis in the study related to the outcome variable '{outcome}'.
                    The hypothesis is: {matching_hypothesis.get('title')}
                    Alternative hypothesis: {matching_hypothesis.get('alternative_hypothesis')}
                    
                    I'm now running a {test_name} on the same outcome variable.
                    
                    Please determine if I should:
                    1. Update the existing hypothesis with new test results
                    2. Create a new hypothesis
                    
                    Consider whether the existing hypothesis aligns with the current statistical test
                    and study design ({study_type_val.replace('_', ' ').title()}).
                    
                    Return your decision as one of:
                    "UPDATE" or "CREATE_NEW"
                    
                    IMPORTANT: Only return one of these exact words, no other text.
                    """
                    
                    # Call LLM to confirm match
                    decision = await call_llm_async(confirm_prompt)
                    decision = decision.strip().upper()
                    
                    if "UPDATE" not in decision:
                        matching_hypothesis = None
                        print(f"LLM recommended creating new hypothesis (decision: {decision})")
                        self.status_bar.showMessage(f"LLM recommended creating new hypothesis instead of updating existing one")
                    else:
                        print(f"LLM recommended updating existing hypothesis: {matching_hypothesis.get('title')}")
                        self.status_bar.showMessage(f"LLM recommended updating existing hypothesis: {matching_hypothesis.get('title')}")
            
            # Get data characteristics to inform the LLM
            outcome_type = infer_data_type(df, outcome)
            
            # Get test results if available and prepare them for hypothesis generation
            test_results = None
            if self.last_test_result and self.last_test_result.get('outcome') == outcome:
                raw_results = self.last_test_result.get('result', {})
                # Use the targeted preparation method with hypothesis purpose
                test_results = self._prepare_for_llm(raw_results, purpose="hypothesis")
                print(f"Including test results in hypothesis generation")
            
            # Prepare group information with size limits
            group_info = ""
            if group and group in df.columns:
                # Limit to 10 values to prevent token explosion
                unique_group_values = df[group].unique()
                display_values = unique_group_values[:10]
                # Convert numpy types to native Python types
                display_values = [value.item() if hasattr(value, 'item') else value for value in display_values]
                group_info = f"\nGroup variable '{group}' with values: {display_values}"
                if len(unique_group_values) > 10:
                    group_info += f" (and {len(unique_group_values) - 10} more values...)"
            
            # Prepare time information with size limits
            time_info = ""
            if time and time in df.columns:
                # Limit to 10 values to prevent token explosion
                unique_time_values = df[time].unique()
                display_values = sorted(unique_time_values)[:10] 
                # Convert numpy types to native Python types
                display_values = [value.item() if hasattr(value, 'item') else value for value in display_values]
                time_info = f"\nTime variable '{time}' with values: {display_values}"
                if len(unique_time_values) > 10:
                    time_info += f" (and {len(unique_time_values) - 10} more values...)"
                
            # Add existing hypothesis info if updating
            existing_info = ""
            if matching_hypothesis:
                existing_info = f"""
                You are updating an existing hypothesis:
                Title: {matching_hypothesis.get('title')}
                Null hypothesis: {matching_hypothesis.get('null_hypothesis')}
                Alternative hypothesis: {matching_hypothesis.get('alternative_hypothesis')}
                
                Please ensure your updated hypothesis is consistent with the previous one, but can
                be refined based on new test results or information.
                """
                
            # Format prompt for generating hypothesis
            prompt = f"""
            {existing_info}
            Generate a clear, scientific hypothesis for a statistical test with the following details:

            Test: {test_name}
            Study Design: {study_type_val.replace('_', ' ').title()}
            Outcome Variable: {outcome} (type: {outcome_type.value if outcome_type else 'Unknown'})
            {group_info}{time_info}
            
            Test Results: {json.dumps(test_results) if test_results else 'Not yet tested'}

            Based on this information, please create:
            1. A title for the hypothesis
            2. A formal null hypothesis (H₀)
            3. A formal alternative hypothesis (H₁)
            4. Specify directionality (non-directional, greater than, or less than)
            5. Determine the hypothesis status based on test results (if available)

            Return your response in this JSON format:
            {{
                "title": "Brief descriptive title for the hypothesis",
                "null_hypothesis": "Formal statement of H₀",
                "alternative_hypothesis": "Formal statement of H₁",
                "directionality": "non-directional|greater|less",
                "notes": "Any additional notes on this hypothesis",
                "status": "untested|confirmed|rejected|inconclusive",
                "status_reason": "Explanation of the status determination"
            }}

            If test results are available, determine the status as:
            - confirmed: if p < 0.05 and effect is in expected direction
            - rejected: if p ≥ 0.05 or effect is in opposite direction
            - inconclusive: if results are ambiguous or assumptions were violated
            - untested: if no test results are available
            """

            # Call LLM
            response = await call_llm_async(prompt)
            
            # Parse JSON response
            match = re.search(r'({.*})', response, re.DOTALL)
            if not match:
                print("Failed to parse hypothesis response from LLM")
                return None
                
            hypothesis_data = json.loads(match.group(1))
            
            # Add additional test-related fields
            hypothesis_data['outcome_variables'] = outcome
            # Always include related_outcome field to support proper matching in the future
            hypothesis_data['related_outcome'] = outcome
            if group:
                hypothesis_data['predictor_variables'] = group
            hypothesis_data['expected_test'] = test_name
            hypothesis_data['alpha_level'] = 0.05  # Default
            hypothesis_data['dataset_name'] = self.current_name  # Store dataset name
            
            # Add minimal test results if available
            if test_results:
                # Include only the essential information about test results
                hypothesis_data['test_results'] = {
                    'significant': test_results.get('significant', False),
                    'p_value': test_results.get('p_value', None),
                    'test_name': test_results.get('test', test_name)
                }
                hypothesis_data['test_date'] = datetime.now().isoformat()
            
            # Process the hypothesis (update existing or create new)
            if matching_hypothesis:
                print(f"Updating existing hypothesis: {matching_hypothesis.get('title')}")
                # Add ID from existing hypothesis to ensure update targets the right one
                hypothesis_id = matching_hypothesis.get('id')
                
                # Add a note about the test change if different from previous test
                prev_test = matching_hypothesis.get('expected_test', '')
                if prev_test and prev_test != test_name:
                    notes = hypothesis_data.get('notes', '')
                    update_note = f"Updated from {prev_test} to {test_name} on {datetime.now().strftime('%Y-%m-%d')}."
                    if notes:
                        hypothesis_data['notes'] = f"{notes}\n\n{update_note}"
                    else:
                        hypothesis_data['notes'] = update_note
                
                # Update the hypothesis
                success = main_window.studies_manager.update_hypothesis(
                    hypothesis_id=hypothesis_id,
                    update_data=hypothesis_data
                )
                
                if success:
                    print(f"Successfully updated hypothesis with ID {hypothesis_id}")
                    test_update_msg = ""
                    if prev_test and prev_test != test_name:
                        test_update_msg = f" (changed test from {prev_test} to {test_name})"
                    self.status_bar.showMessage(f"Updated hypothesis: {hypothesis_data['title']} ({hypothesis_data['status']}){test_update_msg}")
                    # Refresh the hypotheses dropdown
                    self.refresh_hypotheses()
                    # Set dropdown to the updated hypothesis
                    for i in range(self.hypothesis_combo.count()):
                        if self.hypothesis_combo.itemData(i) == hypothesis_id:
                            self.hypothesis_combo.setCurrentIndex(i)
                            break
                    # Return the full hypothesis data
                    return hypothesis_data
                else:
                    print(f"Failed to update hypothesis with ID {hypothesis_id}")
            else:
                print(f"Creating new hypothesis for outcome '{outcome}'")
                # Create new hypothesis
                hypothesis_data['id'] = str(uuid.uuid4())
                
                # Make sure related_outcome is explicitly set
                hypothesis_data['related_outcome'] = outcome
                
                # Debug info about the hypothesis being created
                print(f"Creating new hypothesis with ID {hypothesis_data['id']}")
                print(f"Title: {hypothesis_data.get('title')}")
                print(f"Outcome: {outcome}")
                print(f"Dataset: {self.current_name}")
                
                # Double-check that this hypothesis doesn't already exist
                existing = self.find_matching_hypothesis(outcome, self.current_name)
                if existing:
                    print(f"WARNING: find_matching_hypothesis found match after initial check failed")
                    print(f"Existing hypothesis: {existing.get('title')}")
                    print(f"Will update this hypothesis instead of creating a new one")
                    
                    # Update existing hypothesis instead
                    hypothesis_id = existing.get('id')
                    success = main_window.studies_manager.update_hypothesis(
                        hypothesis_id=hypothesis_id,
                        update_data=hypothesis_data
                    )
                    
                    if success:
                        print(f"Successfully updated existing hypothesis with ID {hypothesis_id}")
                        self.status_bar.showMessage(f"Updated hypothesis: {hypothesis_data['title']} ({hypothesis_data['status']})")
                        # Refresh the hypotheses dropdown
                        self.refresh_hypotheses()
                        # Set dropdown to the updated hypothesis
                        for i in range(self.hypothesis_combo.count()):
                            if self.hypothesis_combo.itemData(i) == hypothesis_id:
                                self.hypothesis_combo.setCurrentIndex(i)
                                break
                        # Return the full hypothesis data
                        return hypothesis_data
                else:
                    # Add the hypothesis through studies manager
                    print(f"Adding new hypothesis with ID {hypothesis_data['id']} to active study")
                    result = main_window.studies_manager.add_hypothesis_to_study(
                        hypothesis_text=hypothesis_data['title'],
                        related_outcome=outcome,
                        hypothesis_data=hypothesis_data
                    )
                    
                    if result:
                        print(f"Successfully added hypothesis with ID {result}")
                        self.status_bar.showMessage(f"Generated new hypothesis: {hypothesis_data['title']} ({hypothesis_data['status']})")
                        
                        # Refresh the hypotheses dropdown
                        self.refresh_hypotheses()
                        # Set dropdown to the new hypothesis
                        for i in range(self.hypothesis_combo.count()):
                            if self.hypothesis_combo.itemData(i) == hypothesis_data['id']:
                                self.hypothesis_combo.setCurrentIndex(i)
                                break
                        
                        # Return the full hypothesis data
                        return hypothesis_data
                    else:
                        print(f"Failed to add new hypothesis through studies_manager")
                        self.status_bar.showMessage(f"Failed to add new hypothesis")
                        return None
            
            return None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating hypothesis: {str(e)}")
            return None